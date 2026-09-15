"""Measure warm BM25 latency on a fixed SciFact corpus and claim split."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from retrieval_lab.bm25 import BM25Config, BM25Index
from retrieval_lab.data import load_claims, load_corpus


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    documents = load_corpus(args.data_root / "corpus.jsonl")
    queries = load_claims(args.data_root / "claims_dev.jsonl", require_labels=True)
    index = BM25Index(documents, BM25Config(k1=1.5, b=0.75, title_boost=2))

    for query in queries[: args.warmup]:
        index.search(query.text, top_k=args.top_k)

    timings_ms: list[float] = []
    started = time.perf_counter()
    for query in queries:
        query_started = time.perf_counter()
        index.search(query.text, top_k=args.top_k)
        timings_ms.append((time.perf_counter() - query_started) * 1000.0)
    elapsed = time.perf_counter() - started

    result = {
        "status": "measured",
        "scope": "warm single-process BM25; not a production service SLO",
        "corpus_documents": len(documents),
        "queries": len(queries),
        "top_k": args.top_k,
        "latency_ms": {
            "mean": statistics.mean(timings_ms),
            "p50": percentile(timings_ms, 0.50),
            "p95": percentile(timings_ms, 0.95),
            "p99": percentile(timings_ms, 0.99),
        },
        "sequential_qps": len(queries) / elapsed,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
