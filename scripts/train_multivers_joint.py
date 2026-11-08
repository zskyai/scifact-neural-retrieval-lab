"""Train the Longformer joint SciFact model on official train claims.

The script keeps complete rationale sets, masks unavailable sentence labels,
and records the negative ratio and window policy for each run.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re, sys, time
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_lab.bm25 import BM25Config, BM25Index
from retrieval_lab.data import load_corpus
from retrieval_lab.multivers_joint import MultiVerSJointModel, encode_windows, joint_loss
from retrieval_lab.evidence_training import isolated_document_pools
from train_domain_verifier import group_split

LABELS = {"NEI": 0, "SUPPORT": 1, "CONTRADICT": 2}

def gold_label(row):
    labels = {a["label"] for groups in row.get("evidence", {}).values() for a in groups}
    if len(labels) > 1:
        raise ValueError("Conflicting labels")
    return LABELS.get(next(iter(labels), "NEI"), 0)

def negative_exclusions(row, rows, mode):
    positive = set(map(str, row.get('evidence', {})))
    if mode == 'none':
        return positive
    excluded = positive | set(map(str, row.get('cited_doc_ids', [])))
    tokens = set(re.findall(r'[a-z0-9]+', row['claim'].lower()))
    for other in rows:
        other_tokens = set(re.findall(r'[a-z0-9]+', other['claim'].lower()))
        similarity = len(tokens & other_tokens) / max(len(tokens | other_tokens), 1)
        if similarity >= .8:
            excluded.update(map(str, other.get('evidence', {})))
    return excluded


def build_examples(rows, index, by_id, tokenizer, *, max_length, negatives, allowed_negative_ids, false_negative_filter='known_related', rationale_mode='complete', audit=None):
    examples = []
    stats = Counter()
    for row in rows:
        positive = {str(x) for x in row.get("evidence", {})}
        excluded = negative_exclusions(row, rows, false_negative_filter)
        wanted = max(negatives * max(1, len(positive)), negatives)
        candidate_ids = list(sorted(positive))
        for hit in index.search(row["claim"], top_k=len(by_id)):
            doc_id = str(hit.doc_id)
            if doc_id in excluded or doc_id not in allowed_negative_ids:
                stats['filtered_candidates'] += 1
                continue
            if doc_id not in positive and doc_id not in candidate_ids:
                candidate_ids.append(doc_id)
            if len(candidate_ids) >= len(positive) + wanted:
                break
        for doc_id in candidate_ids:
            stats['positive_documents' if doc_id in positive else 'hard_negative_documents'] += 1
            doc = by_id[doc_id]
            windows = encode_windows(tokenizer, row["claim"], list(doc.abstract), max_length=max_length)
            rationale_sets = row.get("evidence", {}).get(doc_id, [])
            document_label = 0
            if rationale_sets:
                labels = {a["label"] for a in rationale_sets}
                if len(labels) != 1:
                    raise ValueError("Conflicting document evidence labels")
                document_label = LABELS[next(iter(labels))]
            complete_sets = [set(a["sentences"]) for a in rationale_sets]
            stats['gold_rationale_sets'] += len(complete_sets)
            stats['covered_complete_sets'] += sum(any(s.issubset(set(w.sentence_ids)) for w in windows) for s in complete_sets)
            for window in windows:
                retained = set(window.sentence_ids)
                contained = [s for s in complete_sets if s.issubset(retained)]
                complete = set.union(*contained) if contained else None
                targets = [-1] * len(window.sentence_ids)
                label_mask = not bool(rationale_sets)
                if rationale_sets and complete is not None:
                    targets = [int(s in complete) for s in window.sentence_ids]
                    label_mask = True
                elif not rationale_sets:
                    targets = [-1] * len(window.sentence_ids)
                if rationale_mode == 'membership' and rationale_sets:
                    all_members = set.union(*complete_sets)
                    targets = [int(s in all_members) for s in window.sentence_ids]
                    label_mask = complete is not None
                if not label_mask and all(t < 0 for t in targets):
                    stats['unsupervised_windows_skipped'] += 1
                    continue
                examples.append({"claim_id": str(row["id"]), "doc_id": doc_id,
                                 "input_ids": window.input_ids,
                                 "global_attention_mask": window.global_attention_mask,
                                 "sentence_positions": window.sentence_positions,
                                 "sentence_ids": window.sentence_ids,
                                 "label": document_label, "label_mask": label_mask,
                                 "rationale_targets": targets, "weight": 1.0,
                                 "complete_rationale_in_window": complete is not None})
        stats['claims'] += 1
    stats['windows'] = len(examples)
    if audit is not None:
        audit.update(stats)
    return examples

def collate(batch, pad_id):
    width = max(len(x["input_ids"]) for x in batch)
    sentence_width = max(len(x["sentence_positions"]) for x in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attn = torch.zeros_like(ids); global_attn = torch.zeros_like(ids)
    positions = torch.zeros((len(batch), sentence_width), dtype=torch.long)
    sentence_mask = torch.zeros_like(positions, dtype=torch.bool)
    targets = torch.full_like(positions, -1)
    labels = []; label_mask = []; weights = []
    for i, row in enumerate(batch):
        n = len(row["input_ids"]); ids[i, :n] = torch.tensor(row["input_ids"]); attn[i, :n] = 1; global_attn[i, :n] = torch.tensor(row["global_attention_mask"])
        m = len(row["sentence_positions"]); positions[i, :m] = torch.tensor(row["sentence_positions"]); sentence_mask[i, :m] = 1; targets[i, :m] = torch.tensor(row["rationale_targets"])
        labels.append(row["label"]); label_mask.append(row["label_mask"]); weights.append(row["weight"])
    return {"input_ids": ids, "attention_mask": attn, "global_attention_mask": global_attn,
            "sentence_positions": positions, "sentence_mask": sentence_mask,
            "rationale_targets": targets, "labels": torch.tensor(labels),
            "label_mask": torch.tensor(label_mask), "weights": torch.tensor(weights)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/scifact"))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--negatives", type=int, choices=(10, 20), default=10)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accumulation", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument('--false-negative-filter', choices=('none', 'known_related'), default='known_related')
    p.add_argument('--rationale-mode', choices=('complete', 'membership'), default='complete')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--checkpoint-every', type=int, default=100)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(args.output)
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accumulation < 1:
        raise ValueError("Training budgets must be positive")
    torch.set_num_threads(args.threads); torch.manual_seed(42); random.seed(42)
    source = args.data_root / "claims_train.jsonl"
    rows = [json.loads(s) for s in source.read_text(encoding="utf8").splitlines() if s.strip()]
    train_rows, calibration_rows, groups = group_split(rows)
    documents = load_corpus(args.data_root / "corpus.jsonl"); by_id = {d.doc_id: d for d in documents}
    train_pool, calibration_pool = isolated_document_pools(documents, train_rows, calibration_rows)
    index = BM25Index(documents, BM25Config())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    train_audit, calibration_audit = {}, {}
    train_examples = build_examples(train_rows, index, by_id, tokenizer, max_length=args.max_length, negatives=args.negatives, allowed_negative_ids=train_pool, false_negative_filter=args.false_negative_filter, rationale_mode=args.rationale_mode, audit=train_audit)
    calibration_examples = build_examples(calibration_rows, index, by_id, tokenizer, max_length=args.max_length, negatives=args.negatives, allowed_negative_ids=calibration_pool, false_negative_filter=args.false_negative_filter, rationale_mode=args.rationale_mode, audit=calibration_audit)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'model': str(args.model.resolve()), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'corpus_sha256': hashlib.sha256((args.data_root/'corpus.jsonl').read_bytes()).hexdigest(), 'official_train_claims': len(rows), 'train_claims': len(train_rows), 'calibration_claims': len(calibration_rows), 'train_pool_documents': len(train_pool), 'calibration_pool_documents': len(calibration_pool), 'pool_overlap': len(train_pool & calibration_pool), 'negatives': args.negatives, 'max_length': args.max_length, 'false_negative_filter': args.false_negative_filter, 'rationale_mode': args.rationale_mode, 'epochs': args.epochs, 'batch_size': args.batch_size, 'grad_accumulation': args.grad_accumulation, 'seed': 42, 'train_audit': train_audit, 'calibration_audit': calibration_audit, 'test_used': False, 'dev_used_for_training': False}
    preflight = args.output/'preflight.json'
    if args.resume and json.loads(preflight.read_text(encoding='utf8')) != protocol:
        raise ValueError('Resume configuration or data differs from recorded preflight')
    preflight.write_text(json.dumps(protocol, indent=2), encoding='utf8')
    (args.output/'split_groups.json').write_text(json.dumps(groups, indent=2), encoding='utf8')
    print(json.dumps({'preflight': protocol}), flush=True)
    if args.preflight_only:
        return
    model = MultiVerSJointModel(checkpoint=args.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    resume_state = None
    if args.resume and (args.output/'checkpoint.pt').exists():
        resume_state = torch.load(args.output/'checkpoint.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(resume_state['model']); optimizer.load_state_dict(resume_state['optimizer'])
    history = []; started_total = time.perf_counter()
    for epoch in range(args.epochs):
        if resume_state and epoch < resume_state['epoch']:
            continue
        model.train(); optimizer.zero_grad(set_to_none=True); loss_sum = 0.0; batches = 0; updates = 0; started = time.perf_counter()
        loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42+epoch), collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id or 0))
        if resume_state and epoch == resume_state['epoch']:
            torch.set_rng_state(resume_state['torch_rng'])
        else:
            torch.manual_seed(42+epoch)
        for batch_index, batch in enumerate(loader, start=1):
            if resume_state and epoch == resume_state['epoch'] and batch_index <= resume_state['batch']:
                continue
            output = model(**{k: v for k, v in batch.items() if k not in {"labels", "label_mask", "rationale_targets", "weights"}})
            losses = joint_loss(output["label_logits"], output["rationale_logits"], batch["labels"], batch["rationale_targets"], batch["sentence_mask"], label_mask=batch["label_mask"], weights=batch["weights"])
            (losses["loss"] / args.grad_accumulation).backward(); loss_sum += float(losses["loss"].detach()); batches += 1
            if batch_index % args.grad_accumulation == 0 or batch_index == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True); updates += 1
                if updates % args.checkpoint_every == 0 or batch_index == len(loader):
                    pending = args.output/'checkpoint.pending.pt'
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'batch': batch_index, 'torch_rng': torch.get_rng_state()}, pending)
                    pending.replace(args.output/'checkpoint.pt')
            if batch_index % 25 == 0:
                progress = {'epoch': epoch + 1, 'batch': batch_index, 'total_batches': len(loader), 'mean_loss': loss_sum / max(batches, 1), 'updates': updates, 'seconds': time.perf_counter()-started}
                print(json.dumps(progress), flush=True)
                with (args.output/'progress.jsonl').open('a', encoding='utf8') as stream: stream.write(json.dumps(progress)+'\n')
        record = {"epoch": epoch + 1, "mean_loss": loss_sum / max(batches, 1), "updates": updates, "seconds": time.perf_counter() - started, "train_examples": len(train_examples), "calibration_examples": len(calibration_examples)}
        history.append(record); print(json.dumps(record), flush=True)
    torch.save(model.state_dict(), args.output / "model.pt")
    tokenizer.save_pretrained(args.output / "tokenizer")
    manifest = {"complete": True, "model": str(args.model), "official_train_claims": len(rows), "train_claims": len(train_rows), "calibration_claims": len(calibration_rows), "negatives_per_positive": args.negatives, "max_length": args.max_length, "batch_size": args.batch_size, "grad_accumulation": args.grad_accumulation, "seed": 42, "train_examples": len(train_examples), "calibration_examples": len(calibration_examples), "train_pool_documents": len(train_pool), "calibration_pool_documents": len(calibration_pool), "pool_overlap": len(train_pool & calibration_pool), "test_used": False, "dev_used_for_training": False, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "history": history, "total_seconds": time.perf_counter() - started_total, "quality_evaluation_pending": True}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps(manifest), flush=True)

if __name__ == "__main__":
    main()
