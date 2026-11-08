"""Evaluate every official SciFact dev claim through the deployed application."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import httpx


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * q) - 1)]


def canonical_gold_label(value):
    labels = {'SUPPORT': 'SUPPORTS', 'CONTRADICT': 'CONTRADICTS', 'SUPPORTS': 'SUPPORTS', 'CONTRADICTS': 'CONTRADICTS', 'NEI': 'NEI'}
    if value not in labels:
        raise ValueError('unknown SciFact label: ' + str(value))
    return labels[value]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--data-root', type=Path, default=Path('data/scifact'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--max-retries', type=int, default=1)
    parser.add_argument('--predictions', type=Path, help='Rescore a complete saved run without repeating inference')
    parser.add_argument('--domain-checkpoint', type=Path, help='Run the frozen BM25 domain verifier directly')
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.data_root / 'claims_dev.jsonl'
    claims = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    corpus = {str(row['doc_id']): row for row in (json.loads(line) for line in (args.data_root / 'corpus.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())}
    labels = ('SUPPORTS', 'CONTRADICTS', 'NEI')
    rows, failures = [], []
    cached = None
    engine = None
    if args.domain_checkpoint:
        if args.predictions or not args.calibration or args.max_retries != 0:
            raise ValueError('Domain evaluation requires calibration, zero retries, and no cached predictions')
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import torch
        from retrieval_lab.domain_verifier import DomainEvidenceVerifier, DomainReviewEngine
        from retrieval_lab.data import load_corpus
        torch.set_num_threads(args.threads)
        verifier = DomainEvidenceVerifier(args.domain_checkpoint, calibration=args.calibration)
        engine = DomainReviewEngine(load_corpus(args.data_root / 'corpus.jsonl'), verifier)
        domain_health = {'status': 'ok', 'mode': 'offline_domain_verifier', 'corpus_documents': len(engine.documents),
                         'checkpoint_sha256': hashlib.sha256((args.domain_checkpoint / 'model.safetensors').read_bytes()).hexdigest(),
                         'calibration_sha256': hashlib.sha256(args.calibration.read_bytes()).hexdigest()}
    if args.predictions:
        cached_rows = [json.loads(line) for line in args.predictions.read_text(encoding='utf-8').splitlines() if line.strip()]
        if any('failure' in row for row in cached_rows):
            raise ValueError('cached run contains failures')
        cached = {str(row['claim_id']): row for row in cached_rows}
        if len(cached) != len(cached_rows) or set(cached) != {str(c['id']) for c in claims}:
            raise ValueError('cached run must contain every official dev claim exactly once')
    with httpx.Client(base_url=args.url, timeout=300) as client, (args.output / 'predictions.jsonl').open('w', encoding='utf-8') as out:
        health = domain_health if engine is not None else (json.loads(args.predictions.with_name('summary.json').read_text(encoding='utf-8'))['health'] if cached is not None else client.get('/healthz').raise_for_status().json())
        for i, item in enumerate(claims):
            # Only the claim and fixed configuration cross the inference boundary.
            try:
                if engine is not None:
                    prediction = engine.verify_claim(item['claim'], top_k=args.top_k, max_retries=args.max_retries)
                elif cached is not None:
                    saved = cached[str(item['id'])]
                    if saved['claim'] != item['claim']:
                        raise ValueError('cached claim text mismatch')
                    prediction = saved['prediction']
                else:
                    prediction = client.post('/v1/agent/verify', json={'claim': item['claim'], 'top_k': args.top_k, 'max_retries': args.max_retries}).raise_for_status().json()
                annotations = item.get('evidence') or {}
                gold_labels = {canonical_gold_label(group['label']) for groups in annotations.values() for group in groups}
                if len(gold_labels) > 1:
                    raise ValueError('conflicting gold labels require explicit scoring policy')
                gold_label = next(iter(gold_labels), 'NEI')
                gold = {(str(doc), int(s)) for doc, groups in annotations.items() for group in groups for s in group['sentences']}
                retrieved = {str(d['doc_id']) for d in prediction['evidence']}
                citations = {(str(c['doc_id']), int(c['sentence_id'])) for c in prediction.get('citations', [])}
                selected = {(str(d['doc_id']), int(s['sentence_id'])) for d in prediction['evidence'] for s in d['sentences']}
                valid = {c for c in citations if c[0] in retrieved and c[0] in corpus and 0 <= c[1] < len(corpus[c[0]]['abstract'])}
                errors = []
                if gold and not retrieved.intersection(doc for doc, _ in gold): errors.append('document_miss')
                elif gold and not selected.intersection(gold): errors.append('sentence_miss')
                if citations - valid: errors.append('invalid_citation')
                if prediction['label'] == 'NEI' and gold_label != 'NEI': errors.append('over_refusal')
                if prediction['label'] != 'NEI' and gold_label == 'NEI': errors.append('should_abstain')
                if prediction['label'] != gold_label and not errors: errors.append('verification_or_routing_error')
                row = {'claim_id': item['id'], 'claim': item['claim'], 'gold_label': gold_label, 'prediction': prediction,
                       'gold_evidence_count': len(gold), 'cited_count': len(citations), 'correct_cited_count': len(citations & gold),
                       'valid_cited_count': len(valid), 'selected_gold_count': len(selected & gold),
                       'gold_document_count': len({doc for doc, _ in gold}), 'retrieved_gold_document_count': len(retrieved & {doc for doc, _ in gold}),
                       'errors': errors}
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + '\n'); out.flush()
            except Exception as exc:
                failure = {'claim_id': item['id'], 'error': str(exc), 'type': type(exc).__name__}
                failures.append(failure)
                out.write(json.dumps({'failure': failure}) + '\n'); out.flush()
            if (i + 1) % 25 == 0:
                print(json.dumps({'completed': i + 1, 'expected': len(claims), 'failed': len(failures)}), flush=True)
    count = len(rows)
    per_label = {}
    for label in labels:
        tp = sum(r['gold_label'] == label and r['prediction']['label'] == label for r in rows)
        fp = sum(r['gold_label'] != label and r['prediction']['label'] == label for r in rows)
        fn = sum(r['gold_label'] == label and r['prediction']['label'] != label for r in rows)
        per_label[label] = {'gold': tp + fn, 'predicted': tp + fp, 'f1': 2 * tp / max(2 * tp + fp + fn, 1)}
    cited = sum(r['cited_count'] for r in rows); gold = sum(r['gold_evidence_count'] for r in rows)
    matched = sum(r['correct_cited_count'] for r in rows)
    precision = matched / cited if cited else 0
    recall = matched / gold if gold else 0
    latencies = [r['prediction']['timings_ms']['total_ms'] for r in rows]
    nei = [r for r in rows if r['gold_label'] == 'NEI']; answerable = [r for r in rows if r['gold_label'] != 'NEI']
    report = {'expected_claims': len(claims), 'completed_claims': count, 'failures': failures, 'complete': count == len(claims) and not failures,
              'split': 'official_dev', 'test_used': False, 'threshold_selection': False,
              'scoring_revision': 'canonical_official_support_contradict_v2',
              'rescored_from': str(args.predictions) if args.predictions else None,
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'health': health,
              'config': {'top_k': args.top_k, 'max_retries': args.max_retries},
              'label_accuracy': sum(r['gold_label'] == r['prediction']['label'] for r in rows) / max(count, 1),
              'claim_macro_f1': sum(v['f1'] for v in per_label.values()) / len(labels), 'per_label': per_label,
              'citation_id_validity': sum(r['valid_cited_count'] for r in rows) / cited if cited else None,
              'citation_gold_evidence_match': precision, 'citation_support_rate': None,
              'evidence_precision_micro': precision, 'evidence_recall_micro': recall,
              'evidence_f1_micro': 2 * precision * recall / (precision + recall) if precision + recall else 0,
              'retrieved_document_recall_micro': sum(r['retrieved_gold_document_count'] for r in rows) / max(sum(r['gold_document_count'] for r in rows), 1),
              'nei_refusal_accuracy': sum(r['prediction']['label'] == 'NEI' for r in nei) / len(nei) if nei else None,
              'over_refusal_rate': sum(r['prediction']['label'] == 'NEI' for r in answerable) / max(len(answerable), 1),
              'latency_ms': {'p50': percentile(latencies, .5), 'p95': percentile(latencies, .95)},
              'error_categories': dict(Counter(error for r in rows for error in r['errors'])),
              'annotation_policy': 'union of annotated evidence sentences; not the official joint evidence-set scorer',
              'human_reviews': 0, 'generation_tokens': 0, 'generation_cost': 0,
              'provenance': rows[0]['prediction'].get('provenance') if rows else None}
    (args.output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (args.output / 'errors_for_review.jsonl').open('w', encoding='utf-8') as out:
        for row in rows:
            if row['errors']: out.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report['complete']: raise SystemExit(1)


if __name__ == '__main__': main()
