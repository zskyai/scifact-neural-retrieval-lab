"""Apply frozen TRAIN-selected citation policy to complete saved dev inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_lab.domain_verifier import select_citations


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    summary = json.loads((a.run/'summary.json').read_text(encoding='utf8'))
    config = json.loads(a.calibration.read_text(encoding='utf8'))
    if not summary['complete'] or config['dev_used'] or config['test_used']:
        raise ValueError('Requires a complete run and training-only policy selection')
    if config['parent_calibration_sha256'] != summary['health']['calibration_sha256']:
        raise ValueError('Sentence policy was calibrated against a different claim decision')
    rows = [json.loads(s) for s in (a.run/'predictions.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    if len(rows) != summary['expected_claims'] or len({r['claim_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate or missing saved predictions')
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output/'predictions.jsonl').open('x',encoding='utf8') as stream:
        for row in rows:
            prediction = row['prediction']
            units = [{'doc_id':d['doc_id'],'kind':'sentence','sentence_id':s['sentence_id'],
                      'probabilities':s['nli']['probabilities']} for d in prediction['evidence'] for s in d['sentences']]
            prediction['citations'] = select_citations(units,prediction,config['evidence_policy'])
            prediction['provenance']['decision_config'] = config
            prediction['provenance']['evidence_policy_replay'] = True
            prediction['trace'].append({'tool':'frozen_evidence_policy','policy':config['evidence_policy']})
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    # The common scorer reads health metadata next to cached predictions.
    health = dict(summary['health'], calibration_sha256=hashlib.sha256(a.calibration.read_bytes()).hexdigest(),
                  mode='saved_full_inference_evidence_policy_replay')
    (a.output/'summary.json').write_text(json.dumps({'health':health,'unscored':True,'source_run':str(a.run)},indent=2),encoding='utf8')
    print(json.dumps({'claims':len(rows),'labels_unchanged':True,'output':str(a.output)}))


if __name__ == '__main__':main()
