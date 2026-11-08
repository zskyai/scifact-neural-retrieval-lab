import unittest

from retrieval_lab.evidence_fusion import FeatureFusionCalibrator, extract_features, select_evidence_set


def units(support=True):
    return [
        {"doc_id": "d1", "kind": "abstract", "text": "claim abstract", "probabilities": [0.05, 0.80 if support else 0.05, 0.10 if support else 0.80]},
        {"doc_id": "d1", "kind": "sentence", "sentence_id": 0, "text": "the claim is supported by this result", "probabilities": [0.05, 0.86 if support else 0.04, 0.09 if support else 0.86]},
        {"doc_id": "d1", "kind": "sentence", "sentence_id": 1, "text": "a second result provides context", "probabilities": [0.20, 0.55 if support else 0.10, 0.25 if support else 0.60]},
    ]


class EvidenceFusionTests(unittest.TestCase):
    def test_features_expose_coverage_and_conflict(self):
        features = extract_features("claim supported result", units())
        self.assertEqual(len(features["SUPPORTS"]), 14)
        self.assertGreater(features["SUPPORTS"][5], 0.0)
        self.assertGreater(features["SUPPORTS"][7], 0.0)
        self.assertAlmostEqual(features["SUPPORTS"][6], 0.25)

    def test_calibrator_decides_and_serializes(self):
        rows = [{"claim": "claim supported result", "units": units(True), "gold_label": "SUPPORTS"},
                {"claim": "claim contradicted result", "units": units(False), "gold_label": "CONTRADICTS"}]
        model = FeatureFusionCalibrator(epochs=30).fit(rows)
        result = model.decide(rows[0])
        self.assertIn(result["label"], {"NEI", "SUPPORTS", "CONTRADICTS"})
        self.assertEqual(len(model.state_dict()["weights"]), 3)

    def test_set_selector_stays_inside_decision_document(self):
        decision = {"label": "SUPPORTS", "citations": [{"doc_id": "d1", "sentence_id": 0}]}
        selected = select_evidence_set("claim supported result", units(), decision, max_sentences=2)
        self.assertTrue(selected)
        self.assertTrue(all(row["doc_id"] == "d1" for row in selected))
        self.assertLessEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
