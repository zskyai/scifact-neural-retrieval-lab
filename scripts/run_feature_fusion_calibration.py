"""Fit feature fusion on the held-out training-group calibration split.

The script consumes saved verifier probabilities, so it does not silently
rerun a different model.  It applies the fitted policy to the official dev
predictions and writes a complete replayable prediction file.  The official
dev remains evaluation-only; it is never used to fit thresholds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_lab.evidence_fusion import FeatureFusionCalibrator, LABELS, select_evidence_set


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dev_units(prediction: dict) -> list[dict]:
    """Recover source-grounded units from the saved full inference trace."""
    trace = next((row for row in prediction.get("trace", []) if row.get("tool") == "domain_verifier"), {})
    stance = {str(row["doc_id"]): row for row in trace.get("ranked_stances", [])}
    rows: list[dict] = []
    for rank, document in enumerate(prediction.get("evidence", [])):
        doc_id = str(document["doc_id"])
        by_label = {}
        for row in trace.get("ranked_stances", []):
            if str(row["doc_id"]) == doc_id:
                by_label[row["label"]] = row
        support = float(by_label.get("SUPPORTS", {}).get("abstract_probability", 0.05))
        contradict = float(by_label.get("CONTRADICTS", {}).get("abstract_probability", 0.05))
        nei = max(0.0, 1.0 - support - contradict)
        rows.append({"doc_id": doc_id, "kind": "abstract", "text": document.get("title", ""), "probabilities": [nei, support, contradict], "retrieval_rank": rank})
        for sentence in document.get("sentences", []):
            values = sentence.get("nli", {}).get("probabilities")
            if not values or len(values) != 3:
                continue
            rows.append({"doc_id": doc_id, "kind": "sentence", "sentence_id": int(sentence["sentence_id"]), "text": sentence.get("text", ""), "probabilities": values, "retrieval_rank": rank})
    if not rows and stance:
        raise ValueError("saved prediction has stance trace but no source evidence")
    return rows


def best_anchor(units: list[dict], label: str) -> dict | None:
    index = LABELS.index(label)
    candidates = [row for row in units if row.get("kind") == "sentence"]
    return max(candidates, key=lambda row: float(row["probabilities"][index]), default=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-scores", type=Path, required=True)
    parser.add_argument("--claims-train", type=Path, required=True)
    parser.add_argument("--dev-predictions", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--evidence-max", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    calibration_rows = read_jsonl(args.calibration_scores)
    claim_by_id = {str(row["id"]): str(row["claim"]) for row in read_jsonl(args.claims_train)}
    training = [{"claim": claim_by_id.get(str(row["claim_id"]), ""), "units": row["units"], "gold_label": row["gold_label"]} for row in calibration_rows]
    if any(not row["claim"] for row in training):
        raise ValueError("calibration scores must preserve claim text")
    calibrator = FeatureFusionCalibrator().fit(training)
    dev_rows = read_jsonl(args.dev_predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    output_rows = []
    gold, predicted = [], []
    reasons = {}
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for saved in dev_rows:
            prediction = saved["prediction"]
            units = dev_units(prediction)
            row = {"claim": saved["claim"], "units": units, "gold_label": saved.get("gold_label", "NEI")}
            decision = calibrator.decide(row)
            citations = []
            if decision["label"] != "NEI":
                anchor = best_anchor(units, decision["label"])
                if anchor is not None:
                    decision = {**decision, "citations": [{"doc_id": anchor["doc_id"], "sentence_id": anchor["sentence_id"]}]}
                    citations = select_evidence_set(saved["claim"], units, decision, max_sentences=args.evidence_max, redundancy_penalty=0.10, conflict_penalty=0.20)
                else:
                    decision = {**decision, "label": "NEI", "reason": "no_sentence_anchor"}
            evidence = [{"doc_id": str(row["doc_id"]), "sentences": []} for row in units if row["kind"] == "abstract"]
            result = {"action": "abstain" if decision["label"] == "NEI" else "answer", "label": decision["label"], "confidence": decision["confidence"], "confidence_kind": "feature_fusion_calibrated", "rationale": decision["reason"], "citations": citations, "evidence": evidence, "trace": [{"tool": "feature_fusion", "probabilities": decision["probabilities"], "features": decision["features"], "quality": decision["quality"]}]}
            output = {"claim_id": saved["claim_id"], "claim": saved["claim"], "gold_label": saved.get("gold_label", "NEI"), "prediction": result}
            stream.write(json.dumps(output, ensure_ascii=False) + "\n")
            output_rows.append(output)
            gold.append(output["gold_label"]); predicted.append(result["label"]); reasons[result["rationale"]] = reasons.get(result["rationale"], 0) + 1
    calibrator_metrics = calibrator._label_metrics(gold, predicted)
    calibrator_metrics["accuracy"] = sum(g == p for g, p in zip(gold, predicted)) / max(len(gold), 1)
    calibrator_metrics["per_label"] = {}
    for label in LABELS:
        tp = sum(g == p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        calibrator_metrics["per_label"][label] = {"gold": gold.count(label), "predicted": predicted.count(label), "f1": 2 * tp / max(2 * tp + fp + fn, 1)}
    (args.output / "calibrator.json").write_text(json.dumps(calibrator.state_dict(), indent=2), encoding="utf-8")
    summary = {"complete": True, "fit_split": "official_train_calibration_groups", "evaluation_split": "official_dev", "calibration_rows": len(training), "dev_claims": len(output_rows), "test_used": False, "threshold_selection_on_test": False, "metrics": calibrator_metrics, "decision_reason_counts": reasons, "feature_names": ["abstract_prob", "sentence_prob", "abstract_margin", "sentence_margin", "abstract_sentence_agreement", "rationale_coverage", "opposite_conflict", "claim_sentence_overlap", "retrieval_rank_signal", "baseline_geometric_score", "second_sentence_prob", "top3_sentence_mean", "sentence_count_cap", "has_sentence"]}
    if args.baseline_summary:
        baseline = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
        baseline_metrics = baseline.get("claim_metrics", baseline.get("metrics", {}))
        if not baseline_metrics:
            baseline_metrics = {"macro_f1": baseline.get("claim_macro_f1"), "over_refusal_rate": baseline.get("over_refusal_rate"), "unsafe_answer_rate": 1.0 - float(baseline.get("nei_refusal_accuracy", 0.0))}
        summary["baseline_comparison"] = {"baseline_source": str(args.baseline_summary), "baseline_metrics": baseline_metrics, "delta_macro_f1": calibrator_metrics["macro_f1"] - float(baseline_metrics.get("macro_f1", 0.0)), "delta_over_refusal_rate": calibrator_metrics["over_refusal_rate"] - float(baseline_metrics.get("over_refusal_rate", 0.0)), "delta_unsafe_answer_rate": calibrator_metrics["unsafe_answer_rate"] - float(baseline_metrics.get("unsafe_answer_rate", 0.0))}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
