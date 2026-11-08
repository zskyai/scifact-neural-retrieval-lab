"""Domain stance prediction with source-grounded evidence and held-out calibration."""
from __future__ import annotations
import json
import hashlib
import math
import time
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ('NEI', 'SUPPORTS', 'CONTRADICTS')


def classification_metrics(gold, predicted):
    if len(gold) != len(predicted) or any(x not in LABELS for x in (*gold, *predicted)):
        raise ValueError('Gold and predictions must have matching lengths and valid labels')
    values = []
    confusion = {}
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        values.append(2 * tp / max(2 * tp + fp + fn, 1))
        confusion[label] = {'gold': tp + fn, 'predicted': tp + fp, 'f1': values[-1]}
    answerable = sum(g != 'NEI' for g in gold)
    nei = sum(g == 'NEI' for g in gold)
    return {'accuracy': sum(g == p for g, p in zip(gold, predicted)) / max(len(gold), 1),
            'macro_f1': sum(values) / 3, 'per_label': confusion,
            'over_refusal_rate': sum(g != 'NEI' and p == 'NEI' for g, p in zip(gold, predicted)) / max(answerable, 1),
            'unsafe_answer_rate': sum(g == 'NEI' and p != 'NEI' for g, p in zip(gold, predicted)) / max(nei, 1)}


def aggregate_units(units):
    """Each source document gets one stance score; unrelated documents do not veto it."""
    by_doc = {}
    for unit in units:
        probs = unit['probabilities']
        if len(probs) != 3 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-5):
            raise ValueError('Expected three finite normalized probabilities')
        if unit['kind'] not in {'abstract', 'sentence'}:
            raise ValueError('Unknown source unit kind')
        if unit['kind'] == 'sentence' and (type(unit['sentence_id']) is not int or unit['sentence_id'] < 0):
            raise ValueError('Sentence evidence requires a non-negative source position')
        by_doc.setdefault(str(unit['doc_id']), []).append(unit)
    ranked = []
    for doc_id, rows in by_doc.items():
        abstract = next((r for r in rows if r['kind'] == 'abstract'), None)
        if abstract is None:
            continue
        sentences = [r for r in rows if r['kind'] == 'sentence']
        stance = []
        for label in (1, 2):
            evidence = max(sentences, key=lambda r: r['probabilities'][label], default=None)
            sentence_score = evidence['probabilities'][label] if evidence else 0.0
            # The abstract branch supplies context, subject to tokenizer length
            # limits; the sentence branch supplies a source-grounded anchor.
            score = math.sqrt(max(0.0, abstract['probabilities'][label] * sentence_score))
            stance.append({'label': LABELS[label], 'score': score, 'doc_id': doc_id,
                           'abstract_probability': abstract['probabilities'][label],
                           'sentence_probability': sentence_score,
                           'sentence_id': evidence['sentence_id'] if evidence else None})
        ranked.extend(stance)
    return sorted(ranked, key=lambda r: (-r['score'], r['doc_id'], r['label']))


def decide(ranked, config):
    for key, default, low, high in [('threshold', .5, 0, 1), ('min_stance_margin', 0., 0, 1), ('support_bias', 0., -1, 1)]:
        value = config.get(key, default)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid decision parameter: ' + key)
    if not ranked:
        return {'label': 'NEI', 'confidence': 0.0, 'citations': [], 'reason': 'no_evidence'}
    adjusted = [(r['score'] + (config.get('support_bias', 0.0) if r['label'] == 'SUPPORTS' else 0.0), r) for r in ranked]
    adjusted.sort(key=lambda item: -item[0])
    best_score, best = adjusted[0]
    opposite = max((score for score, row in adjusted if row['label'] != best['label']), default=0.0)
    threshold = config.get('threshold', .5)
    if best['sentence_id'] is None or best_score < threshold or best_score - opposite < config.get('min_stance_margin', 0.0):
        return {'label': 'NEI', 'confidence': max(0.0, min(1.0, 1.0 - best['score'])), 'citations': [], 'reason': 'calibrated_evidence_risk'}
    citations = [{'doc_id':best['doc_id'], 'sentence_id':best['sentence_id']}] if best['sentence_id'] is not None else []
    return {'label':best['label'], 'confidence':best['score'], 'citations':citations, 'reason':'domain_evidence_supported'}


def select_citations(units, decision, policy):
    """Select a bounded evidence set within the chosen source document."""
    maximum = policy.get('max_sentences', 1)
    threshold = policy.get('min_sentence_probability', 0.0)
    relative = policy.get('min_relative_probability', 0.0)
    if type(maximum) is not int or not 1 <= maximum <= 10:
        raise ValueError('max_sentences must be in 1..10')
    if any(not isinstance(x, (float, int)) or not math.isfinite(x) or not 0 <= x <= 1 for x in (threshold, relative)):
        raise ValueError('Evidence probability thresholds must be finite and in [0, 1]')
    if decision['label'] == 'NEI' or not decision['citations']:
        return []
    doc_id = str(decision['citations'][0]['doc_id'])
    label = LABELS.index(decision['label'])
    candidates = [u for u in units if u['kind'] == 'sentence' and str(u['doc_id']) == doc_id]
    if not candidates:
        raise ValueError('Decision citation has no retrieved sentence source')
    candidates.sort(key=lambda u: (-u['probabilities'][label], u['sentence_id']))
    best = candidates[0]['probabilities'][label]
    selected = [u for u in candidates if u['probabilities'][label] >= threshold and u['probabilities'][label] >= best * relative][:maximum]
    # The claim decision already required a source-grounded anchor sentence.
    # Thresholds restrict added evidence rather than silently dropping that anchor.
    if not selected:
        selected = candidates[:1]
    return [{'doc_id': doc_id, 'sentence_id': u['sentence_id']} for u in selected]


class DomainEvidenceVerifier:
    def __init__(self, checkpoint, *, device='cpu', calibration=None):
        self.checkpoint = Path(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
        self.device = device
        self.metadata = json.loads((self.checkpoint / 'verifier_config.json').read_text(encoding='utf-8'))
        self.calibration = json.loads(Path(calibration).read_text(encoding='utf-8')) if calibration else self.metadata.get('decision', {'threshold':.5})
        decide([], self.calibration)
        if calibration:
            manifest = self.checkpoint / 'split_manifest.json'
            if self.calibration.get('manifest_sha256') != hashlib.sha256(manifest.read_bytes()).hexdigest():
                raise ValueError('Calibration does not match the checkpoint split manifest')
        if tuple(self.model.config.id2label[i] for i in range(3)) != LABELS:
            raise ValueError('Unexpected domain verifier label mapping')

    @torch.inference_mode()
    def score_pairs(self, claim, units, batch_size=16):
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        scored = []
        for start in range(0, len(units), batch_size):
            batch = units[start:start + batch_size]
            inputs = self.tokenizer([claim] * len(batch), [u['text'] for u in batch], padding=True, truncation=True, max_length=self.metadata.get('max_length', 256), return_tensors='pt')
            inputs = {k:v.to(self.device) for k,v in inputs.items()}
            probabilities = self.model(**inputs).logits.softmax(-1).cpu().tolist()
            scored.extend(dict(row, probabilities=values) for row, values in zip(batch, probabilities))
        return scored

    def verify(self, claim, units):
        scored = self.score_pairs(claim, units)
        ranked = aggregate_units(scored)
        decision = decide(ranked, self.calibration)
        if 'evidence_policy' in self.calibration:
            decision['citations'] = select_citations(scored, decision, self.calibration['evidence_policy'])
        return {**decision, 'units': scored, 'ranked_stances':ranked}


def make_units(documents):
    return [unit for doc in documents for unit in (
        [{'doc_id': doc.doc_id, 'kind': 'abstract', 'text': doc.text, 'sentence_id': None}] +
        [{'doc_id': doc.doc_id, 'kind': 'sentence', 'text': text, 'sentence_id': i}
         for i, text in enumerate(doc.abstract) if text.strip()])]


class DomainReviewEngine:
    """Frozen BM25 retrieval and learned scientific stance verification."""

    def __init__(self, documents, verifier):
        from .bm25 import BM25Index, BM25Config
        self.documents = documents
        self.by_id = {d.doc_id: d for d in documents}
        self.verifier = verifier
        self.bm25 = BM25Index(documents, BM25Config())
        config = verifier.calibration
        if type(config.get('top_k')) is not int or not 1 <= config['top_k'] <= 50:
            raise ValueError('Calibrated top_k must be an integer in 1..50')
        if config.get('retrieval') != 'bm25' or config.get('max_retries') != 0:
            raise ValueError('Domain review requires BM25 calibration with zero retries')
        if config.get('dev_used') is not False or config.get('test_used') is not False:
            raise ValueError('Dev/test calibrated configurations cannot be deployed as held-out evaluation')

    def verify_claim(self, claim, *, top_k=10, max_retries=0):
        if not claim.strip():
            raise ValueError('Claim must not be blank')
        if top_k != self.verifier.calibration['top_k'] or max_retries != 0:
            raise ValueError('Retrieval settings must match the frozen calibration')
        start = time.perf_counter()
        hits = self.bm25.search(claim, top_k=top_k)
        retrieval_ms = (time.perf_counter() - start) * 1000
        verification_start = time.perf_counter()
        result = self.verifier.verify(claim, make_units([self.by_id[h.doc_id] for h in hits]))
        verification_ms = (time.perf_counter() - verification_start) * 1000
        positions = {(str(u['doc_id']), u['sentence_id']) for u in result['units'] if u['kind'] == 'sentence'}
        if any((str(c['doc_id']), c['sentence_id']) not in positions for c in result['citations']):
            raise ValueError('Verifier returned a citation outside retrieved source units')
        evidence = []
        for hit in hits:
            doc = self.by_id[hit.doc_id]
            sentences = []
            for u in result['units']:
                if str(u['doc_id']) == hit.doc_id and u['kind'] == 'sentence':
                    probs = u['probabilities']; label = max(range(3), key=lambda i: probs[i])
                    sentences.append({'sentence_id': u['sentence_id'], 'text': u['text'], 'score': max(probs[1:]),
                                      'nli': {'label': LABELS[label], 'score': probs[label], 'probabilities': probs}})
            evidence.append({'doc_id': doc.doc_id, 'title': doc.title, 'retrieval_score': hit.score, 'sentences': sentences})
        return {'action': 'abstain' if result['label'] == 'NEI' else 'answer',
                'label': result['label'], 'confidence': result['confidence'], 'confidence_kind': 'model_score_not_calibrated_probability',
                'rationale': result['reason'], 'citations': result['citations'], 'evidence': evidence,
                'trace': [{'tool': 'bm25', 'documents': len(hits)},
                          {'tool': 'domain_verifier', 'units': len(result['units']), 'decision': result['reason'],
                           'ranked_stances': result['ranked_stances']}],
                'timings_ms': {'retrieval_ms': retrieval_ms, 'verification_ms': verification_ms,
                               'rerank_ms': 0.0, 'generation_ms': 0.0, 'total_ms': (time.perf_counter()-start)*1000},
                'provenance': {'retrieval': 'bm25', 'verifier': str(self.verifier.checkpoint),
                               'threshold_status': self.verifier.calibration.get('status'),
                               'decision_config': self.verifier.calibration, 'top_k': top_k, 'max_retries': 0}}
