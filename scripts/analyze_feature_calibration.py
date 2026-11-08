"""Summarize feature distributions on the training-only calibration groups."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_lab.evidence_fusion import LABELS, extract_features


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--claims-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    claim_by_id = {str(row["id"]): str(row["claim"]) for row in read_jsonl(args.claims_train)}
    buckets: dict[str, list[dict[str, list[float]]]] = defaultdict(list)
    for row in read_jsonl(args.scores):
        claim = claim_by_id.get(str(row["claim_id"]))
        if claim is None:
            raise ValueError(f"missing claim text for {row['claim_id']}")
        buckets[str(row["gold_label"])].append(extract_features(claim, row["units"]))
    names = ["abstract_prob", "sentence_prob", "abstract_margin", "sentence_margin", "abstract_sentence_agreement", "rationale_coverage", "opposite_conflict", "claim_sentence_overlap", "retrieval_rank_signal", "baseline_geometric_score", "second_sentence_prob", "top3_sentence_mean", "sentence_count_cap", "has_sentence"]
    report = {"split": "official_train_calibration_groups", "test_used": False, "claims": sum(len(rows) for rows in buckets.values()), "by_gold_label": {}}
    for label in LABELS:
        rows = buckets.get(label, [])
        report["by_gold_label"][label] = {"count": len(rows), "mean_features": {candidate: {name: sum(features[candidate][i] for features in rows) / max(len(rows), 1) for i, name in enumerate(names)} for candidate in LABELS}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
