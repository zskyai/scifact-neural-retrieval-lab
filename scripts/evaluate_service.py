"""Evaluate service-side hybrid retrieval on the official SciFact dev split."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from retrieval_lab.data import load_claims
from retrieval_lab.metrics import SearchResult, evaluate_run


async def main_async(args) -> None:
    queries = load_claims(Path(args.data_root) / "claims_dev.jsonl", require_labels=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    run: dict[str, list[SearchResult]] = {}

    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:
        async def evaluate(query) -> None:
            async with semaphore:
                response = await client.post("/v1/retrieve", json={"query": query.text, "top_k": args.top_k})
                response.raise_for_status()
                payload = response.json()
                run[query.query_id] = [
                    SearchResult(doc_id=str(item["doc_id"]), score=float(item["score"]))
                    for item in payload["results"]
                ]

        await asyncio.gather(*(evaluate(query) for query in queries))

    result = {
        "scope": "FastAPI service-side hybrid retrieval on official SciFact dev",
        "queries": len(queries),
        "top_k": args.top_k,
        "metrics": evaluate_run(queries, run, ks=[1, 5, 10]),
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
