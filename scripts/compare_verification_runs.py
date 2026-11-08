"""Paired claim comparison of complete official-dev verification runs."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np


def load_run(path):
    summary = json.loads((path/'summary.json').read_text(encoding='utf-8'))
    if not summary.get('complete') or summary.get('split') != 'official_dev':
        raise ValueError('Only complete official-dev runs can be compared')
    rows = [json.loads(s) for s in (path/'predictions.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    ids = [r['claim_id'] for r in rows]
    if len(ids) != len(set(ids)) or len(rows) != summary['expected_claims']:
        raise ValueError('Incomplete or duplicate claim membership')
    return summary, {r['claim_id']: r for r in rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    old, old_rows = load_run(a.baseline); new, new_rows = load_run(a.candidate)
    replay = new.get('health', {}).get('mode') == 'saved_full_inference_evidence_policy_replay'
    if set(old_rows) != set(new_rows) or old['source_sha256'] != new['source_sha256']:
        raise ValueError('Runs must cover the identical official claim set')
    ids = sorted(old_rows); paired = []
    for cid in ids:
        b, c = old_rows[cid], new_rows[cid]
        if b['claim'] != c['claim'] or b['gold_label'] != c['gold_label']:
            raise ValueError('Claim or gold mismatch')
        gold = b['gold_label']; bp = b['prediction']['label']; cp = c['prediction']['label']
        paired.append({'claim_id': cid, 'gold_label': gold, 'baseline': bp, 'candidate': cp,
                       'baseline_correct': bp == gold, 'candidate_correct': cp == gold,
                       'baseline_errors': b['errors'], 'candidate_errors': c['errors']})
    delta = np.array([int(r['candidate_correct'])-int(r['baseline_correct']) for r in paired])
    rng = np.random.default_rng(42)
    samples = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    keys = ['label_accuracy', 'claim_macro_f1', 'evidence_precision_micro', 'evidence_recall_micro',
            'evidence_f1_micro', 'citation_id_validity', 'nei_refusal_accuracy', 'over_refusal_rate']
    metrics = {k: {'baseline': old[k], 'candidate': new[k], 'delta': new[k]-old[k] if old[k] is not None and new[k] is not None else None} for k in keys}
    for name, rows in [('baseline', old_rows), ('candidate', new_rows)]:
        nei = [r for r in rows.values() if r['gold_label'] == 'NEI']
        metrics.setdefault('unsafe_answer_rate', {})[name] = sum(r['prediction']['label'] != 'NEI' for r in nei)/max(len(nei), 1)
    metrics['unsafe_answer_rate']['delta'] = metrics['unsafe_answer_rate']['candidate']-metrics['unsafe_answer_rate']['baseline']
    report = {'claims': len(ids), 'baseline': str(a.baseline), 'candidate': str(a.candidate), 'metrics': metrics,
              'corrected_claims': int((delta == 1).sum()), 'regressed_claims': int((delta == -1).sum()),
              'accuracy_delta_bootstrap_95_ci': np.quantile(samples, [.025, .975]).tolist(),
              'bootstrap_seed': 42, 'bootstrap_replicates': 10000,
              'candidate_confusion': dict(Counter(r['gold_label']+'->'+r['candidate'] for r in paired)),
              'candidate_error_categories': new['error_categories'],
              'latency': {'baseline': old['latency_ms'], 'candidate': new['latency_ms'],
                          'note': ('Candidate reuses saved inference timings; evidence-policy execution latency was not measured.' if replay else 'Baseline HTTP service and candidate direct engine; different retrieval and verifier workloads, not an isolated component benchmark.')},
              'comparison_scope': ('Frozen evidence-policy replay; source run and unchanged predictions must be checked for the intended ablation.' if replay else 'End-to-end systems differ in retrieval, verifier, evidence selection and routing; this is not a verifier-only ablation.'),
              'evidence_scoring': 'Union of annotated sentences, not official joint evidence-set scorer; semantic citation support remains unmeasured.',
              'human_reviews': 0, 'test_used': False}
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (a.output/'paired_claims.jsonl').open('x', encoding='utf-8') as out:
        for row in paired: out.write(json.dumps(row)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
