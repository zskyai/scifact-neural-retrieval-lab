"""Complete rationale supervision with explicit token and paper boundaries.

An individual member of a multi-sentence rationale is not automatically a
stance-positive training example. Unannotated sentences are not assumed NEI.
"""
from __future__ import annotations
import hashlib
from collections import Counter


def referenced_documents(rows):
    return {str(doc_id) for row in rows
            for doc_id in set(map(str, row.get('cited_doc_ids', []))) | set(map(str, row.get('evidence', {})))}


def isolated_document_pools(documents, train_rows, calibration_rows):
    """Assign background papers once, including papers used as hard negatives."""
    train = referenced_documents(train_rows)
    calibration = referenced_documents(calibration_rows)
    if train & calibration:
        raise ValueError('Referenced papers cross the training/calibration boundary')
    available = {str(doc.doc_id) for doc in documents}
    if (train | calibration) - available:
        raise ValueError('Referenced paper missing from corpus')
    for doc_id in sorted(available - train - calibration):
        bucket = int(hashlib.sha256(('background:' + doc_id).encode()).hexdigest()[:8], 16) / 2**32
        (calibration if bucket < .2 else train).add(doc_id)
    if train & calibration or train | calibration != available:
        raise AssertionError('Paper partition is not exhaustive and disjoint')
    return train, calibration


def build_complete_rationale_examples(rows, by_id, tokenizer, *, max_length=256):
    """Keep complete gold sets; explicitly exclude rather than truncate long sets.

    This constructs positive stance supervision only. Negative mining and its
    label assumptions must be recorded separately by the training caller.
    """
    if type(max_length) is not int or max_length < 4:
        raise ValueError('max_length must be an integer >= 4')
    examples, excluded = [], []
    seen = set()
    for row in rows:
        for doc_id, annotations in row.get('evidence', {}).items():
            doc = by_id[str(doc_id)]
            for annotation in annotations:
                label = {'SUPPORT': 1, 'CONTRADICT': 2}.get(annotation['label'])
                if label is None:
                    raise ValueError('Unknown gold stance label')
                positions = sorted(set(annotation['sentences']))
                if not positions or any(type(s) is not int or not 0 <= s < len(doc.abstract) for s in positions):
                    raise ValueError('Invalid gold evidence positions')
                if any(not doc.abstract[s].strip() for s in positions):
                    raise ValueError('Gold evidence contains an empty source sentence')
                key = (str(row['id']), str(doc_id), tuple(positions), label)
                if key in seen:
                    continue
                seen.add(key)
                text = ' '.join(doc.abstract[s] for s in positions)
                encoded = tokenizer(row['claim'], text, truncation=False, add_special_tokens=True)
                length = len(encoded['input_ids'])
                example = {'claim_id': row['id'], 'claim': row['claim'], 'doc_id': str(doc_id),
                           'sentence_ids': positions, 'text': text, 'label': label,
                           'kind': 'complete_rationale_set', 'token_length': length}
                if length > max_length:
                    excluded.append({**example, 'reason': 'complete_set_exceeds_token_budget'})
                else:
                    examples.append(example)
    audit = {'kept_rationale_sets': len(examples), 'excluded_rationale_sets': len(excluded),
             'kept_labels': dict(Counter(str(r['label']) for r in examples)),
             'kept_set_sizes': dict(Counter(str(len(r['sentence_ids'])) for r in examples)),
             'max_length': max_length, 'truncation_used': False,
             'multi_sentence_members_labeled_independently': False,
             'unannotated_sentences_assumed_nei': False}
    return examples, excluded, audit
