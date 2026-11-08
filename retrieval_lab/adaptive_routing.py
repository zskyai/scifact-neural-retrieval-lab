"""Evidence-aware adaptive retrieval and abstention decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RetrievalSignals:
    top1_score: float
    top1_margin: float
    evidence_coverage: float
    support_score: float
    contradiction_score: float

    def __post_init__(self) -> None:
        for name in ("evidence_coverage", "support_score", "contradiction_score"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class RoutingDecision:
    action: str
    confidence: float
    reasons: tuple[str, ...]
    retry_budget: int


class AdaptiveEvidenceRouter:
    """Select answer, retry, or abstain from validation-calibrated thresholds."""

    def __init__(
        self,
        *,
        min_top1: float = 0.15,
        min_margin: float = 0.02,
        min_coverage: float = 0.25,
        min_support: float = 0.50,
        max_contradiction: float = 0.45,
        abstain_confidence: float = 0.35,
    ) -> None:
        if min_support <= 0 or max_contradiction < 0:
            raise ValueError("invalid evidence thresholds")
        self.min_top1 = min_top1
        self.min_margin = min_margin
        self.min_coverage = min_coverage
        self.min_support = min_support
        self.max_contradiction = max_contradiction
        self.abstain_confidence = abstain_confidence

    def decide(self, signals: RetrievalSignals, *, retry_budget: int = 1) -> RoutingDecision:
        if retry_budget < 0:
            raise ValueError("retry_budget must be non-negative")
        reasons: list[str] = []
        if signals.contradiction_score > self.max_contradiction and signals.support_score >= self.min_support:
            reasons.append("conflicting_evidence")
        if signals.top1_score < self.min_top1:
            reasons.append("weak_top1")
        if signals.top1_margin < self.min_margin:
            reasons.append("ambiguous_rank_margin")
        if signals.evidence_coverage < self.min_coverage:
            reasons.append("low_evidence_coverage")
        if max(signals.support_score, signals.contradiction_score) < self.min_support:
            reasons.append("weak_entailment")
        confidence = self._confidence(signals)
        if not reasons and confidence >= self.abstain_confidence:
            return RoutingDecision("answer", confidence, (), 0)
        retry_reasons = {
            "conflicting_evidence",
            "weak_top1",
            "ambiguous_rank_margin",
            "low_evidence_coverage",
            "weak_entailment",
        }
        if retry_budget and any(reason in retry_reasons for reason in reasons):
            return RoutingDecision("retry", confidence, tuple(reasons), retry_budget)
        return RoutingDecision("abstain", confidence, tuple(reasons), 0)

    def _confidence(self, signals: RetrievalSignals) -> float:
        rank = max(0.0, min(1.0, signals.top1_score))
        margin = max(0.0, min(1.0, signals.top1_margin / max(self.min_margin * 4, 1e-6)))
        evidence = max(0.0, min(1.0, signals.evidence_coverage))
        support = max(0.0, min(1.0, max(signals.support_score, signals.contradiction_score)))
        contradiction_penalty = max(0.0, min(1.0, min(signals.support_score, signals.contradiction_score)))
        return max(0.0, min(1.0, 0.20 * rank + 0.20 * margin + 0.25 * evidence + 0.40 * support - 0.25 * contradiction_penalty))


def select_best_evidence(
    candidates: Sequence[tuple[str, float, float, float]],
    *,
    max_items: int = 3,
) -> list[tuple[str, float]]:
    """Rank sentence candidates by support, coverage, and retrieval score."""

    if max_items <= 0:
        raise ValueError("max_items must be positive")
    scored = [
        (sentence_id, 0.55 * support + 0.30 * coverage + 0.15 * retrieval)
        for sentence_id, retrieval, support, coverage in candidates
    ]
    return sorted(scored, key=lambda item: (-item[1], item[0]))[:max_items]
