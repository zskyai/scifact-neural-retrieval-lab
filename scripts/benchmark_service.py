"""Run a fixed-concurrency HTTP benchmark against the local RAG service."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

from retrieval_lab.data import load_claims


async def main_async(args) -> None:
    queries = load_claims(Path(args.data_root) / "claims_dev.jsonl", require_labels=True)
    selected = [query.text for query in queries[: args.requests]]
    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:
        for query in selected[: args.warmup]:
            await client.post(args.endpoint, json={"query": query, "top_k": 5})

        semaphore = asyncio.Semaphore(args.concurrency)
        latencies: list[float] = []
        statuses: list[int] = []

        async def request(query: str) -> None:
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.post(args.endpoint, json={"query": query, "top_k": 5})
                    statuses.append(response.status_code)
                except Exception:
                    statuses.append(599)
                latencies.append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        await asyncio.gather(*(request(query) for query in selected))
        elapsed = time.perf_counter() - started

    ordered = sorted(latencies)
    percentile = lambda fraction: ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]
    result = {
        "scope": f"local FastAPI + Qdrant local mode; endpoint={args.endpoint}",
        "requests": len(selected),
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "success_rate": sum(status == 200 for status in statuses) / len(statuses),
        "qps": len(selected) / elapsed,
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        },
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", choices=["/v1/retrieve", "/v1/rag"], default="/v1/rag")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
