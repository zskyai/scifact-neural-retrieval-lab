"""Reconstruct checkpoint training sources without modifying the checkpoint."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_domain_verifier import examples, group_split
from retrieval_lab.bm25 import BM25Index, BM25Config
from retrieval_lab.data import load_corpus


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('data/scifact'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.checkpoint/'split_manifest.json').read_text(encoding='utf-8'))
    source = a.data_root/'claims_train.jsonl'
    rows = [json.loads(s) for s in source.read_text(encoding='utf-8').splitlines() if s.strip()]
    train, calibration, _ = group_split(rows)
    assert set(manifest['train_ids']) == {r['id'] for r in train}
    assert set(manifest['calibration_ids']) == {r['id'] for r in calibration}
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['source_sha256']
    docs = load_corpus(a.data_root/'corpus.jsonl'); by_id = {d.doc_id:d for d in docs}
    index = BM25Index(docs, BM25Config())
    train_pairs = examples(train, index, by_id); heldout_pairs = examples(calibration, index, by_id)
    assert len(train_pairs) == manifest['train_examples']
    assert len(heldout_pairs) == manifest['calibration_examples']
    train_cited = {str(d) for r in train for d in r.get('cited_doc_ids', [])}
    heldout_cited = {str(d) for r in calibration for d in r.get('cited_doc_ids', [])}
    negative_exposure = [r for r in train_pairs if 'hard' in r['kind'] and r['doc_id'] in heldout_cited]
    dev = [json.loads(s) for s in (a.data_root/'claims_dev.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    normalize = lambda text: ' '.join(text.lower().split())
    dev_duplicates = [{'train_claim_id': t['id'], 'dev_claim_id': d['id']} for t in train for d in dev if normalize(t['claim']) == normalize(d['claim'])]
    report = {'checkpoint': str(a.checkpoint), 'train_claims': len(train), 'calibration_claims': len(calibration),
              'train_examples': len(train_pairs), 'calibration_examples': len(heldout_pairs),
              'claim_id_overlap': len(set(manifest['train_ids']) & set(manifest['calibration_ids'])),
              'cited_document_overlap': len(train_cited & heldout_cited),
              'all_pair_document_overlap': len({r['doc_id'] for r in train_pairs} & {r['doc_id'] for r in heldout_pairs}),
              'calibration_cited_docs_seen_as_train_hard_negatives': sorted({r['doc_id'] for r in negative_exposure}),
              'train_negative_examples_on_calibration_cited_docs': len(negative_exposure),
              'strict_paper_text_isolation': not bool(negative_exposure),
              'train_dev_exact_claim_duplicates': dev_duplicates,
              'dev_text_used_only_for_post_evaluation_audit': True,
              'selection_note': 'Calibration groups also select checkpoint epoch; calibration scores are model-selection scores, not independent evaluation.',
              'negative_label_note': 'Unannotated sentences and BM25 documents are weak negatives and may contain unannotated evidence.',
              'dev_used_for_training_or_selection': False, 'test_used': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as out: json.dump(report, out, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
