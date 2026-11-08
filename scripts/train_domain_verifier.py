"""Train a scientific stance verifier on all official training groups.

Document-connected groups are isolated before building positive and hard-negative
examples. Official dev/test annotations never enter training or model selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_lab.bm25 import BM25Index, BM25Config
from retrieval_lab.data import load_corpus
from retrieval_lab.evidence import select_evidence_sentences
from retrieval_lab.domain_verifier import LABELS, classification_metrics


def group_split(rows):
    parent = {str(r['id']):str(r['id']) for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    owners = {}
    for row in rows:
        rid = str(row['id'])
        tokens = set(re.findall(r'[a-z0-9]+', row['claim'].lower())) - {'not','no','never','does','do','is','are'}
        keys = ['doc:' + str(d) for d in row.get('cited_doc_ids', [])] + ['claim:' + ' '.join(sorted(tokens))]
        for key in keys:
            if key in owners: parent[find(rid)] = find(owners[key])
            else: owners[key] = rid
    components = {}
    for row in rows: components.setdefault(find(str(row['id'])), []).append(row)
    train, calibration = [], []
    groups = []
    for component in components.values():
        ids = sorted(str(r['id']) for r in component)
        hashed = int(hashlib.sha256(('|'.join(ids)).encode()).hexdigest()[:8],16) / 2**32
        split = 'calibration' if hashed < .2 else 'train'
        (calibration if split == 'calibration' else train).extend(component)
        groups.append({'ids':ids,'split':split})
    return train, calibration, groups


def examples(rows, index, by_id, *, contradict_hard_negative_top_k=20):
    pairs = []
    for row in rows:
        claim = row['claim']; known = set(map(str, row.get('cited_doc_ids', []))) | set(row['evidence'])
        seen = set()
        def add(text, label, kind, doc_id):
            key = (text, label)
            if key not in seen and text.strip():
                pairs.append({'claim_id':row['id'],'claim':claim,'text':text,'label':label,'kind':kind,'doc_id':str(doc_id)})
                seen.add(key)
        for doc_id, annotations in row['evidence'].items():
            doc = by_id[str(doc_id)]; all_gold = {s for a in annotations for s in a['sentences']}
            for annotation in annotations:
                label = 1 if annotation['label'] == 'SUPPORT' else 2
                add(doc.text, label, 'abstract', doc_id)
                add(' '.join(doc.abstract[s] for s in annotation['sentences']), label, 'rationale_set', doc_id)
                for s in annotation['sentences']:
                    add(doc.abstract[s], label, 'gold_sentence', doc_id)
            negatives = [s for s in select_evidence_sentences(claim, doc, top_k=len(doc.abstract)) if s.sentence_id not in all_gold][:2]
            for sentence in negatives:
                kind = 'contradict_same_document_unannotated' if any(a['label'] == 'CONTRADICT' for a in annotations) else 'same_document_unannotated'
                add(sentence.text, 0, kind, doc_id)
        if not row['evidence']:
            for doc_id in known:
                if doc_id in by_id: add(by_id[doc_id].text, 0, 'nei_abstract', doc_id)
        pool_k = contradict_hard_negative_top_k if any(
            a['label'] == 'CONTRADICT' for anns in row.get('evidence', {}).values() for a in anns
        ) else 20
        for hit in [h for h in index.search(claim, top_k=pool_k) if h.doc_id not in known][:3]:
            doc = by_id[hit.doc_id]
            add(doc.text, 0, 'bm25_hard_document', hit.doc_id)
            for sentence in select_evidence_sentences(claim, doc, top_k=1):
                add(sentence.text, 0, 'bm25_hard_sentence', hit.doc_id)
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('data/scifact'))
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--contradict-hard-negative-top-k', type=int, default=50)
    p.add_argument('--contradict-weight', type=float, default=2.0)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()): raise FileExistsError(a.output)
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads); torch.manual_seed(42); random.seed(42)
    source = a.data_root/'claims_train.jsonl'
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    train, calibration, groups = group_split(rows)
    documents = load_corpus(a.data_root/'corpus.jsonl'); by_id = {d.doc_id:d for d in documents}
    index = BM25Index(documents, BM25Config())
    train_pairs = examples(train, index, by_id, contradict_hard_negative_top_k=a.contradict_hard_negative_top_k)
    heldout_pairs = examples(calibration, index, by_id, contradict_hard_negative_top_k=a.contradict_hard_negative_top_k)
    train_docs = set(str(d) for r in train for d in r.get('cited_doc_ids', []))
    heldout_docs = set(str(d) for r in calibration for d in r.get('cited_doc_ids', []))
    if train_docs & heldout_docs: raise ValueError('Document split overlap')
    manifest = {'official_train_claims':len(rows), 'train_claims':len(train), 'calibration_claims':len(calibration),
                'groups':groups, 'train_ids':[r['id'] for r in train], 'calibration_ids':[r['id'] for r in calibration],
                'train_examples':len(train_pairs),'calibration_examples':len(heldout_pairs),
                'train_labels':dict(Counter(LABELS[x['label']] for x in train_pairs)),
                'training_kinds':dict(Counter(x['kind'] for x in train_pairs)),
                'contradict_hard_negative_top_k': a.contradict_hard_negative_top_k,
                'contradict_weight': a.contradict_weight,
                'document_overlap':0, 'dev_used':False,'test_used':False,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (a.output/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('groups','train_ids','calibration_ids')}),flush=True)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, local_files_only=True, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={l:i for i,l in enumerate(LABELS)}, ignore_mismatched_sizes=True)
    def collate(batch):
        encoded = tokenizer([r['claim'] for r in batch], [r['text'] for r in batch], padding=True, truncation=True, max_length=a.max_length, return_tensors='pt')
        return encoded, torch.tensor([r['label'] for r in batch])
    counts = Counter(r['label'] for r in train_pairs)
    weights = torch.tensor([(len(train_pairs) / (3 * counts[i])) ** .5 for i in range(3)])
    weights[2] *= a.contradict_weight
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    history = []; best = -1.0
    for epoch in range(a.epochs):
        tick = time.perf_counter(); model.train(); total = 0.0; batches = 0
        for encoded, labels in DataLoader(train_pairs, batch_size=a.batch_size, shuffle=True, collate_fn=collate):
            optimizer.zero_grad(set_to_none=True)
            logits = model(**encoded).logits
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights, label_smoothing=.05)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            total += loss.item(); batches += 1
            if batches % 100 == 0: print(json.dumps({'epoch':epoch+1,'batches':batches,'loss':total/batches}),flush=True)
        model.eval(); gold = []; predicted = []
        with torch.inference_mode():
            for encoded, labels in DataLoader(heldout_pairs,batch_size=a.batch_size,collate_fn=collate):
                predicted.extend(LABELS[i] for i in model(**encoded).logits.argmax(-1).tolist())
                gold.extend(LABELS[i] for i in labels.tolist())
        metrics = classification_metrics(gold,predicted)
        record = {'epoch':epoch+1,'loss':total/max(batches,1),'validation_pair_metrics':metrics,'seconds':time.perf_counter()-tick,
                  'hard_negative_examples':sum('hard' in r['kind'] for r in train_pairs)}
        history.append(record); print(json.dumps(record),flush=True)
        if metrics['macro_f1'] > best:
            best = metrics['macro_f1']; model.save_pretrained(a.output); tokenizer.save_pretrained(a.output)
            (a.output/'verifier_config.json').write_text(json.dumps({'max_length':a.max_length,'decision':{'threshold':.5},'selected_epoch':epoch+1,'selection':'heldout_training_groups_pair_macro_f1','labels':LABELS}),encoding='utf-8')
        (a.output/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    print(json.dumps({'complete':True,'selected_pair_macro_f1':best,'output':str(a.output)}),flush=True)


if __name__ == '__main__': main()
