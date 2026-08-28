from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .data import QueryExample


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    score: float


Run = Mapping[str, Sequence[SearchResult]]


def _ranked_unique(results: Sequence[SearchResult]) -> list[str]:
    seen: set[str] = set()
    ranked: list[str] = []
    for result in results:
        if result.doc_id not in seen:
            ranked.append(result.doc_id)
            seen.add(result.doc_id)
    return ranked


def _dcg(relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def evaluate_run(
    queries: Sequence[QueryExample],
    run: Run,
    ks: Iterable[int] = (1, 5, 10, 20, 100),
) -> dict[str, float | int | list[int]]:
    normalized_ks = sorted(set(int(k) for k in ks))
    if not normalized_ks or normalized_ks[0] <= 0:
        raise ValueError("ks must contain positive integers")

    labeled_queries = [query for query in queries if query.relevant_doc_ids]
    if not labeled_queries:
        raise ValueError("No labeled queries available for evaluation")

    totals: dict[str, float] = {}
    for k in normalized_ks:
        totals[f"recall@{k}"] = 0.0
        totals[f"mrr@{k}"] = 0.0
        totals[f"ndcg@{k}"] = 0.0

    for query in labeled_queries:
        ranked = _ranked_unique(run.get(query.query_id, ()))
        relevant = query.relevant_doc_ids
        for k in normalized_ks:
            top_k = ranked[:k]
            hits = [1 if doc_id in relevant else 0 for doc_id in top_k]
            totals[f"recall@{k}"] += sum(hits) / len(relevant)

            reciprocal_rank = 0.0
            for rank, doc_id in enumerate(top_k, start=1):
                if doc_id in relevant:
                    reciprocal_rank = 1.0 / rank
                    break
            totals[f"mrr@{k}"] += reciprocal_rank

            ideal_hits = [1] * min(k, len(relevant))
            ideal_dcg = _dcg(ideal_hits)
            totals[f"ndcg@{k}"] += _dcg(hits) / ideal_dcg if ideal_dcg else 0.0

    count = len(labeled_queries)
    metrics: dict[str, float | int | list[int]] = {
        "evaluated_queries": count,
        "total_queries": len(queries),
        "ks": normalized_ks,
    }
    metrics.update({name: value / count for name, value in totals.items()})
    return metrics


def save_run(path: str | Path, run: Run) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for query_id, results in run.items():
            for rank, result in enumerate(results, start=1):
                row = {
                    "query_id": query_id,
                    "doc_id": result.doc_id,
                    "rank": rank,
                    "score": result.score,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_run(path: str | Path) -> dict[str, list[SearchResult]]:
    run: dict[str, list[tuple[int, SearchResult]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                query_id = str(row["query_id"])
                rank = int(row["rank"])
                result = SearchResult(doc_id=str(row["doc_id"]), score=float(row["score"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid run row at {path}:{line_number}") from exc
            run.setdefault(query_id, []).append((rank, result))
    return {
        query_id: [result for _, result in sorted(entries, key=lambda item: item[0])]
        for query_id, entries in run.items()
    }
