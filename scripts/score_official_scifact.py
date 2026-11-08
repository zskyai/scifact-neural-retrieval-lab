"""Score complete saved runs with the inspected upstream SciFact evaluator."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_lab.domain_verifier import classification_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('data/scifact'))
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--official-evaluator', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    source = a.data_root / 'claims_dev.jsonl'
    gold = [json.loads(s) for s in source.read_text(encoding='utf8').splitlines() if s.strip()]
    records = [json.loads(s) for s in a.predictions.read_text(encoding='utf8').splitlines() if s.strip()]
    by_id = {str(r['claim_id']):r for r in records}
    if len(by_id) != len(records) or set(by_id) != {str(r['id']) for r in gold}:
        raise ValueError('Predictions must contain every official dev claim exactly once')
    corpus = {str(r['doc_id']):r for r in (json.loads(s) for s in (a.data_root/'corpus.jsonl').read_text(encoding='utf8').splitlines() if s.strip())}
    spec = importlib.util.spec_from_file_location('upstream_scifact_evaluator', a.official_evaluator)
    upstream = importlib.util.module_from_spec(spec); spec.loader.exec_module(upstream)
    predicted = []; gold_labels = []; predicted_labels = []; valid = 0; citations = 0
    selected_count = gold_count = overlap_count = label_supported = 0
    canonical = {'SUPPORTS':'SUPPORT', 'CONTRADICTS':'CONTRADICT'}
    for row in gold:
        saved = by_id[str(row['id'])]
        if saved.get('claim') != row['claim']:
            raise ValueError('Claim text mismatch')
        result = saved['prediction']; label = result['label']; evidence = {}
        if label not in ('NEI', 'SUPPORTS', 'CONTRADICTS'):
            raise ValueError('Invalid claim prediction label')
        retrieved = {str(d['doc_id']) for d in result.get('evidence', [])}
        if label == 'NEI' and result.get('citations'):
            raise ValueError('NEI response contains supporting citations')
        positions = set()
        for citation in result.get('citations', []):
            doc_id = str(citation['doc_id']); sid = citation['sentence_id']; citations += 1
            if doc_id not in corpus or doc_id not in retrieved or type(sid) is not int or not 0 <= sid < len(corpus[doc_id]['abstract']):
                raise ValueError('Citation is outside retrieved real source positions')
            if (doc_id, sid) in positions:
                raise ValueError('Duplicate citation position')
            positions.add((doc_id, sid)); valid += 1
            evidence.setdefault(doc_id, {'label':canonical[label], 'sentences':[]})['sentences'].append(sid)
        predicted.append({'id':row['id'], 'evidence':evidence})
        labels = {a['label'] for groups in row['evidence'].values() for a in groups}
        if len(labels) > 1:
            raise ValueError('Conflicting gold claim labels')
        gold_labels.append({'SUPPORT':'SUPPORTS','CONTRADICT':'CONTRADICTS'}.get(next(iter(labels), None), 'NEI'))
        predicted_labels.append(label)
        gold_positions = {(str(doc), sid) for doc, annotations in row['evidence'].items() for annotation in annotations for sid in annotation['sentences']}
        selected_count += len(positions); gold_count += len(gold_positions); overlap_count += len(positions & gold_positions)
        label_supported += sum(any(sid in annotation['sentences'] and annotation['label'] == canonical.get(label) for annotation in row['evidence'].get(doc, [])) for doc, sid in positions)
    scorer = upstream.Evaluator(verbose=True)
    metrics = scorer.evaluate([upstream.unify_label(r) for r in gold], predicted)
    counts = dict(scorer.counts_sentence)
    evidence_precision = overlap_count / selected_count if selected_count else 0.0
    evidence_recall = overlap_count / gold_count if gold_count else 0.0
    nei_tp = sum(g == p == 'NEI' for g, p in zip(gold_labels, predicted_labels))
    predicted_nei = predicted_labels.count('NEI'); gold_nei = gold_labels.count('NEI')
    report = {'complete':True, 'split':'official_dev', 'claims':len(gold),
              'official_evaluator_sha256':hashlib.sha256(a.official_evaluator.read_bytes()).hexdigest(),
              'gold_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'predictions_sha256':hashlib.sha256(a.predictions.read_bytes()).hexdigest(),
              'official_metrics':metrics, 'claim_metrics':classification_metrics(gold_labels, predicted_labels),
              'citation_validity':valid/citations if citations else None,
              'sentence_membership_micro': {'precision': evidence_precision, 'recall': evidence_recall, 'f1': 2*evidence_precision*evidence_recall/(evidence_precision+evidence_recall) if evidence_precision+evidence_recall else 0.0, 'gold_count': gold_count, 'predicted_count': selected_count, 'overlap': overlap_count, 'definition': 'Union of all annotated rationale sets per claim/document; use official_metrics for complete-rationale scorer parity'},
              'citation_gold_label_membership_rate': label_supported/citations if citations else None,
              'hallucinated_citation_id_rate': (citations-valid)/citations if citations else None,
              'refusal': {'precision': nei_tp/predicted_nei if predicted_nei else None, 'recall': nei_tp/gold_nei if gold_nei else None, 'accuracy_on_gold_nei': nei_tp/gold_nei if gold_nei else None, 'refusal_rate': predicted_nei/len(gold), 'coverage': 1-predicted_nei/len(gold), 'selective_label_accuracy': sum(g == p and p != 'NEI' for g,p in zip(gold_labels,predicted_labels))/(len(gold)-predicted_nei) if predicted_nei<len(gold) else None},
              'citation_support_rate_gold_complete_rationale':counts.get('correct_label',0)/citations if citations else None,
              'citation_support_definition':'Gold label plus complete annotated rationale membership, as counted by the official evaluator; not a semantic NLI or human judgment',
              'semantic_citation_support_rate':None, 'human_reviews':0, 'test_used':False, 'threshold_selection':False}
    a.output.mkdir(parents=True)
    (a.output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    with (a.output/'official_predictions.jsonl').open('x', encoding='utf8') as stream:
        for row in predicted: stream.write(json.dumps(row)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
