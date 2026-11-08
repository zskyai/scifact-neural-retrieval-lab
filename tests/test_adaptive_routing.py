import unittest

from retrieval_lab.adaptive_routing import AdaptiveEvidenceRouter, RetrievalSignals, select_best_evidence


class AdaptiveRoutingTests(unittest.TestCase):
    def test_strong_refutation_is_decisive_evidence_not_conflict(self):
        decision = AdaptiveEvidenceRouter().decide(
            RetrievalSignals(0.8, 0.2, 0.9, 0.01, 0.95), retry_budget=0
        )
        self.assertEqual(decision.action, "answer")
        self.assertNotIn("conflicting_evidence", decision.reasons)

    def test_good_evidence_answers_without_retry(self):
        decision = AdaptiveEvidenceRouter().decide(
            RetrievalSignals(0.8, 0.2, 0.9, 0.9, 0.0), retry_budget=1
        )
        self.assertEqual(decision.action, "answer")
        self.assertEqual(decision.retry_budget, 0)

    def test_weak_evidence_retries_before_abstaining(self):
        decision = AdaptiveEvidenceRouter().decide(
            RetrievalSignals(0.05, 0.001, 0.1, 0.2, 0.0), retry_budget=1
        )
        self.assertEqual(decision.action, "retry")
        self.assertIn("weak_top1", decision.reasons)

    def test_conflict_without_budget_abstains(self):
        decision = AdaptiveEvidenceRouter().decide(
            RetrievalSignals(0.8, 0.2, 0.9, 0.9, 0.9), retry_budget=0
        )
        self.assertEqual(decision.action, "abstain")
        self.assertIn("conflicting_evidence", decision.reasons)

    def test_evidence_selection_is_deterministic(self):
        selected = select_best_evidence([("s2", .9, .7, .4), ("s1", .8, .8, .4)], max_items=1)
        self.assertEqual(selected[0][0], "s1")
