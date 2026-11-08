"""Feature-based evidence fusion and selective refusal for SciFact.

The original verifier multiplied an abstract score by the best sentence score.
That is easy to audit, but it cannot distinguish a supported sentence from a
conflicting document, a one-off lexical match, or a claim with no stable
evidence.  This module exposes those signals as a small CPU-trainable model.
All fitted parameters must come from a calibration split; no gold evidence is
read at inference time.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

LABELS = ("NEI", "SUPPORTS", "CONTRADICTS")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {x.lower() for x in _TOKEN_RE.findall(str(text)) if len(x) > 2}


def _overlap(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / max(len(a), 1)


def _finite_probability(row: Mapping[str, object], index: int) -> float:
    values = row.get("probabilities")
    if not isinstance(values, Sequence) or len(values) != 3:
        raise ValueError("units require three probabilities")
    value = float(values[index])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("unit probability must be finite and in [0, 1]")
    return value


def _unit_groups(units: Iterable[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    groups: dict[str, list[Mapping[str, object]]] = {}
    for unit in units:
        if unit.get("kind") not in {"abstract", "sentence"}:
            raise ValueError("unknown evidence unit kind")
        groups.setdefault(str(unit.get("doc_id")), []).append(unit)
    return groups


def _opposing_indices(label_index: int) -> tuple[int, ...]:
    # NEI must be compared with both answerable stances; answerable labels
    # should only compete with their semantic opposite, not high NEI mass.
    return (1, 2) if label_index == 0 else ((2,) if label_index == 1 else (1,))


def _doc_features(claim: str, rows: Sequence[Mapping[str, object]], label_index: int, rank: int) -> list[float]:
    abstract = next((r for r in rows if r.get("kind") == "abstract"), None)
    sentences = [r for r in rows if r.get("kind") == "sentence"]
    if abstract is None:
        return []
    abstract_prob = _finite_probability(abstract, label_index)
    abstract_other = max(_finite_probability(abstract, i) for i in _opposing_indices(label_index))
    values = [(_finite_probability(r, label_index), r) for r in sentences]
    values.sort(key=lambda item: -item[0])
    top = values[0][0] if values else 0.0
    second = values[1][0] if len(values) > 1 else 0.0
    sentence_other = max((_finite_probability(r, i) for r in sentences for i in _opposing_indices(label_index)), default=0.0)
    best_sentence = values[0][1] if values else None
    coverage = sum(score >= 0.20 for score, _ in values) / max(len(values), 1)
    agreement = float(abstract_prob >= abstract_other and top >= sentence_other)
    lexical = _overlap(claim, str(best_sentence.get("text", ""))) if best_sentence else 0.0
    retrieval_rank = float((best_sentence or abstract).get("retrieval_rank", rank))
    rank_signal = 1.0 / max(1.0, 1.0 + retrieval_rank)
    baseline = math.sqrt(max(0.0, abstract_prob * top))
    top3_mean = sum(score for score, _ in values[:3]) / max(min(len(values), 3), 1)
    return [
        abstract_prob,
        top,
        abstract_prob - abstract_other,
        top - sentence_other,
        agreement,
        coverage,
        sentence_other,
        lexical,
        rank_signal,
        baseline,
        second,
        top3_mean,
        min(1.0, len(sentences) / 20.0),
        float(bool(sentences)),
    ]


def extract_features(claim: str, units: Iterable[Mapping[str, object]]) -> dict[str, list[float]]:
    """Return one fused feature vector per label.

    The vector summarizes the strongest retrieved document, while retaining
    conflict, coverage, lexical overlap and retrieval-rank signals.  It is
    deliberately numeric and serializable so each decision can be replayed.
    """
    groups = _unit_groups(units)
    result: dict[str, list[float]] = {}
    for label_index, label in enumerate(LABELS):
        candidates = []
        for rank, rows in enumerate(groups.values()):
            values = _doc_features(claim, rows, label_index, rank)
            if values:
                candidates.append(values)
        if not candidates:
            result[label] = [0.0] * 14
            continue
        candidates.sort(key=lambda vector: (vector[9], vector[1], vector[0]), reverse=True)
        best = candidates[0][:]
        best[10] = candidates[1][1] if len(candidates) > 1 else 0.0
        best[11] = sum(vector[1] for vector in candidates[:3]) / min(len(candidates), 3)
        result[label] = best
    return result


def evidence_quality(features: Sequence[float]) -> float:
    """Estimate citation quality before deciding whether to answer."""
    if len(features) < 14:
        raise ValueError("invalid evidence feature vector")
    return max(0.0, min(1.0, 0.55 * features[1] + 0.25 * features[5] + 0.20 * (1.0 - features[6])))


@dataclass(frozen=True)
class RefusalPolicy:
    label_thresholds: dict[str, float]
    min_margin: float
    min_quality: float
    objective: str


class FeatureFusionCalibrator:
    """Small multinomial logistic model with validation-only refusal policy."""

    def __init__(self, *, epochs: int = 500, learning_rate: float = 0.08, l2: float = 0.01, class_weights: Sequence[float] = (1.0, 1.0, 1.0)) -> None:
        if epochs <= 0 or learning_rate <= 0 or l2 < 0:
            raise ValueError("invalid optimizer arguments")
        if len(class_weights) != 3 or any(not math.isfinite(float(value)) or float(value) <= 0 for value in class_weights):
            raise ValueError("class_weights must contain three positive finite values")
        self.epochs, self.learning_rate, self.l2 = epochs, learning_rate, l2
        self.class_weights = [float(value) for value in class_weights]
        self.mean: list[float] | None = None
        self.scale: list[float] | None = None
        self.weights: list[list[float]] | None = None
        self.bias: list[float] | None = None
        self.policy: RefusalPolicy | None = None

    @staticmethod
    def _softmax(logits: Sequence[float]) -> list[float]:
        pivot = max(logits)
        exp = [math.exp(min(40.0, x - pivot)) for x in logits]
        total = sum(exp)
        return [x / total for x in exp]

    def _normalise(self, values: Sequence[float]) -> list[float]:
        if self.mean is None or self.scale is None:
            raise RuntimeError("calibrator is not fitted")
        return [(float(x) - mean) / scale for x, mean, scale in zip(values, self.mean, self.scale)]

    def _logits(self, row: Mapping[str, object]) -> list[float]:
        if self.weights is None or self.bias is None:
            raise RuntimeError("calibrator is not fitted")
        features = extract_features(str(row["claim"]), row["units"])
        return [bias + sum(w * x for w, x in zip(weights, self._normalise(features[label]))) for bias, weights, label in zip(self.bias, self.weights, LABELS)]

    def fit(self, rows: Sequence[Mapping[str, object]]) -> "FeatureFusionCalibrator":
        if not rows:
            raise ValueError("calibration rows must not be empty")
        feature_rows = [extract_features(str(row["claim"]), row["units"]) for row in rows]
        flat = [values for row in feature_rows for values in row.values()]
        dimension = len(flat[0])
        self.mean = [sum(row[i] for row in flat) / len(flat) for i in range(dimension)]
        self.scale = [math.sqrt(sum((row[i] - self.mean[i]) ** 2 for row in flat) / len(flat)) or 1.0 for i in range(dimension)]
        self.weights = [[0.0] * dimension for _ in LABELS]
        self.bias = [0.0] * len(LABELS)
        gold = [LABELS.index(str(row["gold_label"])) for row in rows]
        for _ in range(self.epochs):
            gradients = [[0.0] * dimension for _ in LABELS]
            bias_gradients = [0.0] * len(LABELS)
            for row, target in zip(feature_rows, gold):
                vectors = [self._normalise(row[label]) for label in LABELS]
                probs = self._softmax([bias + sum(w * x for w, x in zip(weights, vector)) for bias, weights, vector in zip(self.bias, self.weights, vectors)])
                sample_weight = self.class_weights[target]
                for class_index in range(3):
                    error = sample_weight * (probs[class_index] - float(class_index == target))
                    bias_gradients[class_index] += error
                    for i, value in enumerate(vectors[class_index]):
                        gradients[class_index][i] += error * value
            rate = self.learning_rate / max(len(rows), 1)
            for class_index in range(3):
                self.bias[class_index] -= rate * bias_gradients[class_index]
                for i in range(dimension):
                    gradient = gradients[class_index][i] + self.l2 * self.weights[class_index][i]
                    self.weights[class_index][i] -= rate * gradient
        self.policy = self._fit_policy(rows)
        return self

    def predict_proba(self, row: Mapping[str, object]) -> dict[str, float]:
        values = self._softmax(self._logits(row))
        return dict(zip(LABELS, values))

    @staticmethod
    def _label_metrics(gold: Sequence[str], predicted: Sequence[str]) -> dict[str, float]:
        if not gold or len(gold) != len(predicted):
            raise ValueError("aligned non-empty labels are required")
        over = sum(g != "NEI" and p == "NEI" for g, p in zip(gold, predicted)) / max(sum(g != "NEI" for g in gold), 1)
        unsafe = sum(g == "NEI" and p != "NEI" for g, p in zip(gold, predicted)) / max(sum(g == "NEI" for g in gold), 1)
        f1s = []
        for label in LABELS:
            tp = sum(g == p == label for g, p in zip(gold, predicted))
            fp = sum(g != label and p == label for g, p in zip(gold, predicted))
            fn = sum(g == label and p != label for g, p in zip(gold, predicted))
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        return {"macro_f1": sum(f1s) / 3, "over_refusal_rate": over, "unsafe_answer_rate": unsafe}

    def _fit_policy(self, rows: Sequence[Mapping[str, object]]) -> RefusalPolicy:
        probabilities = [self.predict_proba(row) for row in rows]
        features = [extract_features(str(row["claim"]), row["units"]) for row in rows]
        gold = [str(row["gold_label"]) for row in rows]
        best = None
        for support_threshold in (0.15, 0.20, 0.25, 0.30, 0.35):
            for contradiction_threshold in (0.15, 0.20, 0.25, 0.30, 0.35):
                for margin in (0.0, 0.02, 0.05, 0.08, 0.12):
                    for quality in (0.10, 0.18, 0.25, 0.32):
                        policy = RefusalPolicy({"SUPPORTS": support_threshold, "CONTRADICTS": contradiction_threshold}, margin, quality, "macro_f1_minus_refusal_penalty")
                        predictions = [self._decide_with_policy(prob, feat, policy)["label"] for prob, feat in zip(probabilities, features)]
                        metrics = self._label_metrics(gold, predictions)
                        objective = metrics["macro_f1"] - 0.30 * metrics["over_refusal_rate"] - 0.10 * metrics["unsafe_answer_rate"]
                        if best is None or objective > best[0]:
                            best = (objective, policy)
        assert best is not None
        return best[1]

    @staticmethod
    def _decide_with_policy(probabilities: Mapping[str, float], features: Mapping[str, Sequence[float]], policy: RefusalPolicy) -> dict[str, object]:
        ordered = sorted(((float(score), label) for label, score in probabilities.items()), reverse=True)
        best_score, best_label = ordered[0]
        second_score = ordered[1][0]
        quality = evidence_quality(features[best_label])
        threshold = policy.label_thresholds.get(best_label, 1.0)
        if best_label == "NEI":
            return {"label": "NEI", "confidence": best_score, "reason": "no_answerable_evidence", "quality": quality}
        if best_score < threshold:
            return {"label": "NEI", "confidence": best_score, "reason": "weak_evidence", "quality": quality}
        if best_score - second_score < policy.min_margin:
            reason = "conflicting_evidence" if features[best_label][6] >= 0.25 else "low_label_margin"
            return {"label": "NEI", "confidence": best_score, "reason": reason, "quality": quality}
        if quality < policy.min_quality:
            return {"label": "NEI", "confidence": best_score, "reason": "low_evidence_quality", "quality": quality}
        return {"label": best_label, "confidence": best_score, "reason": "fused_evidence_supported", "quality": quality}

    def decide(self, row: Mapping[str, object]) -> dict[str, object]:
        if self.policy is None:
            raise RuntimeError("fit must be called before decide")
        probabilities = self.predict_proba(row)
        features = extract_features(str(row["claim"]), row["units"])
        return {**self._decide_with_policy(probabilities, features, self.policy), "probabilities": probabilities, "features": features}

    def state_dict(self) -> dict[str, object]:
        if self.mean is None or self.scale is None or self.weights is None or self.bias is None or self.policy is None:
            raise RuntimeError("fit must be called before serialization")
        return {"labels": list(LABELS), "mean": self.mean, "scale": self.scale, "weights": self.weights, "bias": self.bias, "class_weights": self.class_weights, "policy": {"label_thresholds": self.policy.label_thresholds, "min_margin": self.policy.min_margin, "min_quality": self.policy.min_quality, "objective": self.policy.objective}}


def select_evidence_set(claim: str, units: Iterable[Mapping[str, object]], decision: Mapping[str, object], *, max_sentences: int = 3, redundancy_penalty: float = 0.25, conflict_penalty: float = 0.35) -> list[dict[str, object]]:
    """Select a compact evidence set with coverage gain and conflict control."""
    if max_sentences < 1:
        raise ValueError("max_sentences must be positive")
    label = str(decision.get("label", "NEI"))
    if label == "NEI":
        return []
    citations = decision.get("citations") or []
    doc_id = str(citations[0]["doc_id"]) if citations else None
    candidates = [u for u in units if u.get("kind") == "sentence" and (doc_id is None or str(u.get("doc_id")) == doc_id)]
    label_index = LABELS.index(label)
    target_tokens = _tokens(claim)
    selected: list[Mapping[str, object]] = []
    covered: set[str] = set()
    for _ in range(min(max_sentences, len(candidates))):
        best = None
        for candidate in candidates:
            if candidate in selected:
                continue
            text = str(candidate.get("text", ""))
            tokens = _tokens(text)
            gain = len((tokens & target_tokens) - covered) / max(len(target_tokens), 1)
            probability = _finite_probability(candidate, label_index)
            opposite = max(_finite_probability(candidate, i) for i in _opposing_indices(label_index))
            redundancy = max((_overlap(text, str(previous.get("text", ""))) for previous in selected), default=0.0)
            score = probability + 0.35 * gain - redundancy_penalty * redundancy - conflict_penalty * opposite
            key = (score, probability, -int(candidate.get("sentence_id", 0)))
            if best is None or key > best[0]:
                best = (key, candidate)
        if best is None or best[0][0] <= 0.0:
            break
        selected.append(best[1]); covered.update(_tokens(str(best[1].get("text", ""))) & target_tokens)
    return [{"doc_id": str(row["doc_id"]), "sentence_id": int(row["sentence_id"])} for row in selected]
