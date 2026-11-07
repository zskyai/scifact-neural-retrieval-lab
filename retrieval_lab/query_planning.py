"""Deterministic query planning for scientific claim retrieval.

The planner is deliberately model-agnostic: an LLM or a domain rewrite model
can be plugged in later, while the default path remains reproducible and
auditable. Plans keep the original claim so retrieval never loses provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence


_CLAUSE_BOUNDARY = re.compile(r"\s+(?:and|or|but|while|whereas)\s+|[;:!?]\s*")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "with", "and", "or", "but", "does",
    "do", "did", "this", "that", "these", "those", "whether", "how",
}


@dataclass(frozen=True)
class QueryPlan:
    original: str
    rewrites: tuple[str, ...]
    subqueries: tuple[str, ...]
    strategy: str = "deterministic"

    @property
    def all_queries(self) -> tuple[str, ...]:
        seen: set[str] = set()
        values: list[str] = []
        for query in (self.original, *self.rewrites, *self.subqueries):
            normalized = " ".join(query.split()).strip()
            if normalized and normalized.lower() not in seen:
                values.append(normalized)
                seen.add(normalized.lower())
        return tuple(values)


class ScientificQueryPlanner:
    """Build lexical variants and conservative subqueries from a claim."""

    def __init__(self, *, max_subqueries: int = 3, min_terms: int = 3) -> None:
        if max_subqueries <= 0 or min_terms <= 0:
            raise ValueError("max_subqueries and min_terms must be positive")
        self.max_subqueries = max_subqueries
        self.min_terms = min_terms

    def plan(self, claim: str) -> QueryPlan:
        original = " ".join(str(claim).split()).strip()
        if not original:
            raise ValueError("claim must not be empty")
        terms = self._content_terms(original)
        rewrites = []
        if terms:
            rewrites.append(" ".join(terms))
            if len(terms) >= 5:
                rewrites.append(" ".join(terms[: min(8, len(terms))]))
        clauses = self._clauses(original)
        subqueries = [clause for clause in clauses if len(self._content_terms(clause)) >= self.min_terms]
        return QueryPlan(
            original=original,
            rewrites=tuple(dict.fromkeys(rewrites)),
            subqueries=tuple(dict.fromkeys(subqueries[: self.max_subqueries])),
        )

    def _clauses(self, text: str) -> list[str]:
        return [part.strip(" ,.") for part in _CLAUSE_BOUNDARY.split(text) if part.strip()]

    def _content_terms(self, text: str) -> list[str]:
        terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}|\d+(?:\.\d+)?", text.lower())
        return [term for term in terms if term not in _STOPWORDS]


def merge_ranked_runs(
    runs: Sequence[Sequence[tuple[str, float]]],
    *,
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse query-plan runs with weighted reciprocal rank fusion."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not runs:
        return []
    weights = tuple(weights or (1.0,) * len(runs))
    if len(weights) != len(runs) or any(weight < 0 for weight in weights):
        raise ValueError("weights must match runs and be non-negative")
    fused: dict[str, float] = {}
    for weight, run in zip(weights, runs):
        for rank, (doc_id, _score) in enumerate(run, start=1):
            fused[str(doc_id)] = fused.get(str(doc_id), 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
