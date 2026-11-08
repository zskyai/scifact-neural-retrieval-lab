"""Audit all official-train rationale sets without truncating their supervision."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_domain_verifier import group_split
from retrieval_lab.data import load_corpus
from retrieval_lab.evidence_training import isolated_document_pools, build_complete_rationale_examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('data/scifact'))
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    source = a.data_root / 'claims_train.jsonl'
    corpus = a.data_root / 'corpus.jsonl'
    rows = [json.loads(s) for s in source.read_text(encoding='utf8').splitlines() if s.strip()]
    train, calibration, groups = group_split(rows)
    documents = load_corpus(corpus); by_id = {d.doc_id:d for d in documents}
    pools = isolated_document_pools(documents, train, calibration)
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    artifacts = {}
    for name, subset, pool in zip(('train', 'calibration'), (train, calibration), pools):
        kept, excluded, audit = build_complete_rationale_examples(subset, by_id, tokenizer, max_length=a.max_length)
        if any(r['doc_id'] not in pool for r in kept + excluded):
            raise ValueError('Rationale paper is outside its assigned partition')
        artifacts[name] = (kept, excluded, audit)
    a.output.mkdir(parents=True)
    for name, (kept, excluded, audit) in artifacts.items():
        for suffix, records in (('positive_sets', kept), ('excluded_sets', excluded)):
            with (a.output / (name + '_' + suffix + '.jsonl')).open('x', encoding='utf8') as stream:
                for r in records:
                    stream.write(json.dumps(r, ensure_ascii=False) + '\n')
    report = {'official_train_claims':len(rows), 'train_claims':len(train), 'calibration_claims':len(calibration),
              'train_ids':[r['id'] for r in train], 'calibration_ids':[r['id'] for r in calibration],
              'groups':groups, 'audits':{name:value[2] for name,value in artifacts.items()},
              'document_pools':{'train':sorted(pools[0]), 'calibration':sorted(pools[1])},
              'all_candidate_paper_overlap':len(pools[0] & pools[1]),
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'corpus_sha256':hashlib.sha256(corpus.read_bytes()).hexdigest(),
              'tokenizer':str(a.tokenizer), 'max_length':a.max_length,
              'dev_annotations_read':False, 'test_annotations_read':False,
              'model_trained':False, 'negative_examples_constructed':False}
    (a.output / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups','train_ids','calibration_ids','document_pools')}))


if __name__ == '__main__': main()
