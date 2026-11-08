"""Rebuild official SciFact dev features from complete abstracts."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_lab.bm25 import BM25Config, BM25Index
from retrieval_lab.data import load_corpus
from retrieval_lab.multivers_joint import encode_windows

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('data/scifact'))
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--overlap', type=int, default=1)
    a = p.parse_args()
    if a.top_k < 1 or a.max_length < 16 or a.overlap < 0:
        raise ValueError('top-k, max-length and overlap must be positive')
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(a.output)
    from transformers import AutoTokenizer
    claims_path, corpus_path = a.data_root/'claims_dev.jsonl', a.data_root/'corpus.jsonl'
    rows = [json.loads(x) for x in claims_path.read_text(encoding='utf8').splitlines() if x.strip()]
    documents = load_corpus(corpus_path); by_id = {str(d.doc_id): d for d in documents}
    index = BM25Index(documents, BM25Config())
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    a.output.mkdir(parents=True)
    sentence_windows = candidate_docs = gold_docs_added = max_tokens = 0
    with (a.output/'features.jsonl').open('w', encoding='utf8') as stream:
        for number, row in enumerate(rows, 1):
            gold = {str(x) for x in row.get('evidence', {})}
            hits = index.search(row['claim'], top_k=min(a.top_k, len(documents)))
            ids = [str(x.doc_id) for x in hits]; scores = {str(x.doc_id): float(x.score) for x in hits}
            for doc_id in sorted(gold):
                if doc_id in by_id and doc_id not in ids:
                    ids.append(doc_id); gold_docs_added += 1
            for doc_id in ids:
                doc = by_id[doc_id]
                windows = encode_windows(tokenizer, row['claim'], list(doc.abstract), max_length=a.max_length, overlap=a.overlap)
                rationale = [{'label': ann['label'], 'sentence_ids': sorted(set(ann['sentences']))}
                             for ann in row.get('evidence', {}).get(doc_id, [])]
                for wi, window in enumerate(windows):
                    max_tokens = max(max_tokens, len(window.input_ids)); sentence_windows += 1
                    stream.write(json.dumps({'claim_id': str(row['id']), 'claim': row['claim'], 'split': 'official_dev',
                        'doc_id': doc_id, 'title': doc.title, 'sentence_ids': window.sentence_ids,
                        'sentence_texts': [doc.abstract[i] for i in window.sentence_ids], 'input_ids': window.input_ids,
                        'global_attention_mask': window.global_attention_mask, 'sentence_positions': window.sentence_positions,
                        'window_number': wi, 'window_count': len(windows), 'retrieval_rank': ids.index(doc_id)+1,
                        'retrieval_score': scores.get(doc_id), 'gold_rationale': rationale,
                        'complete_abstract_sentence_count': len(doc.abstract)}, ensure_ascii=False)+'\n')
            candidate_docs += len(ids)
            if number % 25 == 0 or number == len(rows):
                print(json.dumps({'completed_claims': number, 'total_claims': len(rows), 'candidate_docs': candidate_docs,
                                  'sentence_windows': sentence_windows}), flush=True)
    manifest = {'complete': True, 'split': 'official_dev', 'claims': len(rows), 'corpus_documents': len(documents),
        'candidate_top_k': a.top_k, 'max_length': a.max_length, 'sentence_overlap': a.overlap,
        'candidate_docs': candidate_docs, 'sentence_windows': sentence_windows,
        'gold_docs_added_outside_bm25_top_k': gold_docs_added, 'max_window_tokens': max_tokens,
        'claims_sha256': sha256(claims_path), 'corpus_sha256': sha256(corpus_path), 'test_used': False,
        'training_used': False, 'threshold_selection': 'none',
        'feature_definition': 'complete abstract sentence windows with gold rationale alignment'}
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__':
    main()
