"""Bounded evidence retrieval agent for claim verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .adaptive_routing import AdaptiveEvidenceRouter, RetrievalSignals
from .query_planning import ScientificQueryPlanner, merge_ranked_runs


@dataclass(frozen=True)
class EvidenceAgentResult:
    action: str
    claim: str
    label: str
    confidence: float
    evidence: tuple[dict[str, Any], ...]
    trace: tuple[dict[str, Any], ...]


class EvidenceRetrievalAgent:
    """Run retrieve, verify, targeted retry, and final answer/NEI routing.

    Tools are injected so the controller can run against local indexes, Milvus,
    Qdrant, or a service API without embedding provider-specific logic here.
    """

    def __init__(
        self,
        retrieve: Callable[[str, int], Sequence[dict[str, Any]]],
        verify: Callable[[str, Sequence[dict[str, Any]]], dict[str, float]],
        *,
        planner: ScientificQueryPlanner | None = None,
        router: AdaptiveEvidenceRouter | None = None,
        max_retries: int = 1,
        top_k: int = 10,
    ) -> None:
        if max_retries < 0 or top_k <= 0:
            raise ValueError("max_retries must be non-negative and top_k positive")
        self.retrieve = retrieve
        self.verify = verify
        self.planner = planner or ScientificQueryPlanner()
        self.router = router or AdaptiveEvidenceRouter()
        self.max_retries = max_retries
        self.top_k = top_k

    def run(self, claim: str) -> EvidenceAgentResult:
        plan = self.planner.plan(claim)
        queries = [claim]
        trace: list[dict[str, Any]] = []
        candidates: dict[str, dict[str, Any]] = {}
        verification: dict[str, float] = {}
        decision = None
        selected: list[dict[str, Any]] = []

        for attempt in range(self.max_retries + 1):
            if attempt:
                queries = [q for q in plan.all_queries if q != claim][(attempt - 1) * self.top_k : attempt * self.top_k]
                if not queries:
                    queries = list(plan.all_queries[1:])[:1]
            runs = []
            for query in queries:
                hits = list(self.retrieve(query, self.top_k * (attempt + 1)))
                runs.append([(str(hit["doc_id"]), float(hit.get("score", 0.0))) for hit in hits])
                for hit in hits:
                    candidates.setdefault(str(hit["doc_id"]), dict(hit))
                trace.append({"step": "retrieve", "attempt": attempt, "query": query, "hits": len(hits)})

            ranked_ids = [doc_id for doc_id, _ in merge_ranked_runs(runs)[: self.top_k]]
            selected = [candidates[doc_id] for doc_id in ranked_ids if doc_id in candidates]
            verification = self.verify(claim, selected)
            top_score = float(verification.get("coverage", 0.0)) if selected else 0.0
            first_score = float(selected[0].get("score", 0.0)) if selected else 0.0
            second_score = float(selected[1].get("score", 0.0)) if len(selected) > 1 else 0.0
            margin = max(0.0, first_score - second_score) / max(abs(first_score), 1e-8)
            signals = RetrievalSignals(
                top1_score=max(0.0, min(1.0, top_score)),
                top1_margin=max(0.0, min(1.0, margin)),
                evidence_coverage=float(verification.get("coverage", 0.0)),
                support_score=float(verification.get("support", 0.0)),
                contradiction_score=float(verification.get("contradiction", 0.0)),
            )
            decision = self.router.decide(signals, retry_budget=self.max_retries - attempt)
            trace.append({"step": "verify_and_route", "attempt": attempt, "action": decision.action, "reasons": list(decision.reasons), **verification})
            if decision.action != "retry":
                break

        assert decision is not None
        label = "SUPPORTS" if verification.get("support", 0.0) >= verification.get("contradiction", 0.0) else "CONTRADICTS"
        if decision.action == "abstain":
            label = "NEI"
        return EvidenceAgentResult(
            action=decision.action,
            claim=claim,
            label=label,
            confidence=decision.confidence,
            evidence=tuple(selected),
            trace=tuple(trace),
        )
