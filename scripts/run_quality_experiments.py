"""Run reproducible retrieval quality ablations on official SciFact dev.

The script consumes saved runs so model training and evaluation remain
separate.  Fusion uses only ranks; evidence-aware reranking uses the
cross-encoder score plus query-to-sentence max similarity and coverage.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from retrieval_lab.data import load_claims, load_corpus
from retrieval_lab.metrics import SearchResult, evaluate_run, load_run, save_run
from retrieval_lab.rerank import CrossEncoderReranker


def rrf_runs(runs, *, k=60, top_k=100):
    ids = set().union(*(run.keys() for run in runs))
    output = {}
    for qid in ids:
        scores = {}
        for run in runs:
            for rank, item in enumerate(run.get(qid, ()), 1):
                scores[item.doc_id] = scores.get(item.doc_id, 0.0) + 1.0 / (k + rank)
        ranked = sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)[:top_k]
        output[qid] = [SearchResult(doc_id=doc_id, score=score) for doc_id, score in ranked]
    return output


def adaptive_rrf(bm25, dense, queries, *, k=60, top_k=100):
    """Use lexical weight for identifier-heavy/short claims, dense otherwise."""
    output = {}
    for query in queries:
        text = query.text
        tokens = re.findall(r"[A-Za-z0-9-]+", text)
        identifier_ratio = sum(bool(re.search(r"\d|[-]", t)) for t in tokens) / max(len(tokens), 1)
        dense_weight = 0.75 if len(tokens) >= 18 and identifier_ratio < 0.12 else 0.35
        scores = {}
        for rank, item in enumerate(bm25.get(query.query_id, ()), 1):
            scores[item.doc_id] = scores.get(item.doc_id, 0.0) + (1.0 - dense_weight) / (k + rank)
        for rank, item in enumerate(dense.get(query.query_id, ()), 1):
            scores[item.doc_id] = scores.get(item.doc_id, 0.0) + dense_weight / (k + rank)
        ranked = sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)[:top_k]
        output[query.query_id] = [SearchResult(doc_id=doc_id, score=score) for doc_id, score in ranked]
    return output


def split_sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def lexical_sentence_score(query: str, sentence: str) -> float:
    q = set(re.findall(r"[A-Za-z0-9]+", query.lower()))
    s = set(re.findall(r"[A-Za-z0-9]+", sentence.lower()))
    return len(q & s) / max(len(q), 1)


def evidence_rerank(run, queries, documents, scorer, *, depth=10, alpha=0.7, beta=0.2, gamma=0.1):
    by_id = {doc.doc_id: doc for doc in documents}
    output = {}
    timings = []
    for query in queries:
        head = [x for x in run[query.query_id][:depth] if x.doc_id in by_id]
        texts = [by_id[x.doc_id].text for x in head]
        started = time.perf_counter()
        ce_scores = scorer.score(query.text, texts)
        rescored = []
        for item, ce in zip(head, ce_scores):
            sentences = split_sentences(by_id[item.doc_id].text)
            # Sentence-level evidence uses a deterministic lexical coverage
            # signal; the cross-encoder remains the document-level teacher.
            ss = [lexical_sentence_score(query.text, sentence) for sentence in sentences]
            max_sim = max(ss) if ss else 0.0
            threshold = float(np.percentile(ss, 75)) if ss else 0.0
            coverage = sum(s >= threshold for s in ss) / len(ss) if ss else 0.0
            rescored.append(SearchResult(item.doc_id, alpha * float(ce) + beta * float(max_sim) + gamma * coverage))
        rescored.sort(key=lambda x: (x.score, x.doc_id), reverse=True)
        output[query.query_id] = rescored + list(run[query.query_id][depth:])
        timings.append((time.perf_counter() - started) * 1000)
    return output, {"queries": len(timings), "mean_query_ms": float(np.mean(timings)), "p95_query_ms": float(np.percentile(timings, 95))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bm25-run", type=Path, required=True)
    parser.add_argument("--zero-run", type=Path, required=True)
    parser.add_argument("--hn-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank-depth", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    queries = load_claims(args.data_root / "claims_dev.jsonl", require_labels=True)
    documents = load_corpus(args.data_root / "corpus.jsonl")
    bm25, zero, hn = (load_run(p) for p in (args.bm25_run, args.zero_run, args.hn_run))
    runs = {
        "bm25": bm25,
        "dense_zero": zero,
        "dense_hard_negative": hn,
        "hybrid_rrf_hn": rrf_runs((bm25, hn)),
        "adaptive_rrf_hn": adaptive_rrf(bm25, hn, queries),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {name: evaluate_run(queries, run, ks=[1, 5, 10, 20, 100]) for name, run in runs.items()}
    for name, run in runs.items():
        save_run(args.output_dir / f"{name}.run.jsonl", run)
    scorer = CrossEncoderReranker(args.rerank_model, batch_size=16, max_length=256, device=args.device)
    reranked, timing = evidence_rerank(runs["hybrid_rrf_hn"], queries, documents, scorer, depth=args.rerank_depth)
    report["hybrid_ce_evidence"] = evaluate_run(queries, reranked, ks=[1, 5, 10, 20, 100])
    report["hybrid_ce_evidence_timing"] = timing
    save_run(args.output_dir / "hybrid_ce_evidence.run.jsonl", reranked)
    (args.output_dir / "quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
