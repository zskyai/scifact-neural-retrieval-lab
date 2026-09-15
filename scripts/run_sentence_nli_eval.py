"""Evaluate real sentence-level NLI evidence selection on SciFact Dev.

Retrieval candidates are generated without labels.  Gold evidence is read only
for scoring, and the fixed entailment threshold is never tuned on test data.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import pipeline

from retrieval_lab.bm25 import BM25Config, BM25Index
from retrieval_lab.data import load_claims, load_corpus


def _norm_label(label: str) -> str:
    text = label.lower()
    if "entail" in text:
        return "ENTAILMENT"
    if "contrad" in text:
        return "CONTRADICTION"
    return "NEUTRAL"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--nli-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--retrieval-depth", type=int, default=10)
    p.add_argument("--entailment-threshold", type=float, default=0.5)
    p.add_argument("--max-evidence", type=int, default=3)
    p.add_argument("--query-limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    docs = load_corpus(args.data_root / "corpus.jsonl")
    by_id = {d.doc_id: d for d in docs}
    claims_path = args.data_root / "claims_dev.jsonl"
    claims = load_claims(claims_path, require_labels=False)
    if args.query_limit:
        claims = claims[: args.query_limit]
    raw = {str(row["id"]): row for row in _read_jsonl(claims_path)}
    bm25 = BM25Index(docs, BM25Config())
    nli = pipeline(
        "text-classification",
        model=str(args.nli_model),
        tokenizer=str(args.nli_model),
        local_files_only=True,
        device=-1,
    )

    pairs: list[tuple[str, str, str, int]] = []
    query_candidates: dict[str, list[str]] = {}
    gold_by_query: dict[str, set[tuple[str, int]]] = {}
    for query in claims:
        candidates = bm25.search(query.text, top_k=args.retrieval_depth)
        query_candidates[query.query_id] = [x.doc_id for x in candidates]
        row = raw[query.query_id]
        gold: set[tuple[str, int]] = set()
        for doc_id, annotations in (row.get("evidence") or {}).items():
            for ann in annotations:
                for sentence_id in ann.get("sentences", []):
                    gold.add((str(doc_id), int(sentence_id)))
        gold_by_query[query.query_id] = gold
        for doc_id in query_candidates[query.query_id]:
            for sentence_id, sentence in enumerate(by_id[doc_id].abstract):
                pairs.append((query.query_id, doc_id, query.text, sentence_id))

    started = time.perf_counter()
    predictions = nli(
        [{"text": by_id[doc_id].abstract[sentence_id], "text_pair": claim}
         for query_id, doc_id, claim, sentence_id in pairs],
        batch_size=args.batch_size,
        truncation=True,
    )
    by_query: dict[str, list[dict]] = {}
    label_counts = {"ENTAILMENT": 0, "CONTRADICTION": 0, "NEUTRAL": 0}
    for (query_id, doc_id, claim, sentence_id), pred in zip(pairs, predictions):
        label = _norm_label(str(pred["label"]))
        score = float(pred["score"])
        label_counts[label] += 1
        by_query.setdefault(query_id, []).append(
            {"doc_id": doc_id, "sentence_id": sentence_id, "label": label, "score": score}
        )

    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "valid": 0, "cited": 0, "support": 0}
    for query in claims:
        scored = by_query.get(query.query_id, [])
        entailed = [x for x in scored if x["label"] == "ENTAILMENT" and x["score"] >= args.entailment_threshold]
        entailed.sort(key=lambda x: (-x["score"], x["doc_id"], x["sentence_id"]))
        selected = entailed[: args.max_evidence]
        predicted = {(x["doc_id"], x["sentence_id"]) for x in selected}
        gold = gold_by_query[query.query_id]
        tp = len(predicted & gold)
        fp = len(predicted - gold)
        fn = len(gold - predicted)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        cited_docs = {x["doc_id"] for x in selected}
        retrieved_docs = set(query_candidates[query.query_id])
        valid = len(cited_docs & retrieved_docs)
        totals["valid"] += valid
        totals["cited"] += len(cited_docs)
        if selected:
            totals["support"] += sum(1 for x in selected if x["label"] == "ENTAILMENT")
        rows.append({
            "query_id": query.query_id,
            "gold_evidence": sorted([{"doc_id": d, "sentence_id": s} for d, s in gold], key=lambda x: (x["doc_id"], x["sentence_id"])),
            "retrieved_doc_ids": query_candidates[query.query_id],
            "predicted_evidence": selected,
            "predicted_contradictions": [x for x in scored if x["label"] == "CONTRADICTION" and x["score"] >= args.entailment_threshold],
        })

    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    metrics = {
        "queries": len(claims),
        "queries_with_gold_evidence": sum(bool(x) for x in gold_by_query.values()),
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "citation_validity": totals["valid"] / max(totals["cited"], 1),
        "citation_support_rate": totals["support"] / max(sum(len(x["predicted_evidence"]) for x in rows), 1),
        "nli_label_counts": label_counts,
        "entailment_threshold": args.entailment_threshold,
        "retrieval_depth": args.retrieval_depth,
        "latency_seconds": time.perf_counter() - started,
        "nli_model_path": str(args.nli_model),
        "gold_evidence_used_for_input": False,
        "test_used": False,
    }
    payload = {"metrics": metrics, "queries": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
