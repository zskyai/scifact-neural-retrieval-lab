"""Select sentence-set policy on cached held-out TRAIN predictions only."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_lab.domain_verifier import decide, select_citations


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--base-calibration', type=Path, required=True)
    p.add_argument('--train-claims', type=Path, default=Path('data/scifact/claims_train.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    base = json.loads(a.base_calibration.read_text(encoding='utf8'))
    if base.get('dev_used') is not False or base.get('test_used') is not False:
        raise ValueError('Only training-group calibration is supported')
    rows = [json.loads(s) for s in a.scores.read_text(encoding='utf8').splitlines() if s.strip()]
    if len({r['claim_id'] for r in rows}) != len(rows) or {r['claim_id'] for r in rows} != set(base['calibration_ids']):
        raise ValueError('Cached scores do not match calibration membership')
    annotations = {r['id']: r for r in (json.loads(s) for s in a.train_claims.read_text(encoding='utf8').splitlines() if s.strip())}
    gold = {cid: {(str(doc), int(s)) for doc, groups in annotations[cid]['evidence'].items() for g in groups for s in g['sentences']} for cid in base['calibration_ids']}
    decisions = {r['claim_id']: decide(r['ranked'], base) for r in rows}
    grid = []
    for maximum in [1, 2, 3, 4, 5]:
        for threshold in [0., .1, .2, .3, .4, .5]:
            for relative in [0., .5, .75, .9]:
                policy = {'max_sentences': maximum, 'min_sentence_probability': threshold, 'min_relative_probability': relative}
                predicted = matched = expected = 0
                for row in rows:
                    cid = row['claim_id']
                    citations = {(c['doc_id'], c['sentence_id']) for c in select_citations(row['units'], decisions[cid], policy)}
                    predicted += len(citations); expected += len(gold[cid]); matched += len(citations & gold[cid])
                precision = matched / max(predicted, 1); recall = matched / max(expected, 1)
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
                grid.append({'policy': policy, 'evidence_precision': precision, 'evidence_recall': recall, 'evidence_f1': f1, 'citations': predicted})
    best = max(grid, key=lambda r: (r['evidence_f1'], r['evidence_precision'], -r['policy']['max_sentences']))
    config = {**base, 'evidence_policy': best['policy'], 'evidence_selection_objective': 'union_evidence_micro_f1',
              'evidence_calibration_metrics': best, 'evidence_scores_sha256': hashlib.sha256(a.scores.read_bytes()).hexdigest(),
              'parent_calibration_sha256': hashlib.sha256(a.base_calibration.read_bytes()).hexdigest(),
              'claim_decision_unchanged': True}
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'calibration.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    (a.output/'evidence_policy_grid.json').write_text(json.dumps(grid, indent=2), encoding='utf8')
    print(json.dumps(best), flush=True)


if __name__ == '__main__': main()
