import unittest

from retrieval_lab.query_planning import ScientificQueryPlanner, merge_ranked_runs


class QueryPlanningTests(unittest.TestCase):
    def test_plan_keeps_original_and_removes_stopwords_from_rewrite(self):
        plan = ScientificQueryPlanner().plan("Does exercise improve nitric oxide production?")
        self.assertEqual(plan.original, "Does exercise improve nitric oxide production?")
        self.assertIn("exercise improve nitric oxide production", plan.rewrites)
        self.assertIn(plan.original, plan.all_queries)

    def test_empty_claim_is_rejected(self):
        with self.assertRaises(ValueError):
            ScientificQueryPlanner().plan("  ")

    def test_weighted_rrf_deduplicates_documents(self):
        fused = merge_ranked_runs(
            [[("a", 1.0), ("b", 0.5)], [("b", 0.9), ("c", 0.2)]],
            weights=[2.0, 1.0],
        )
        self.assertEqual([doc_id for doc_id, _ in fused], ["b", "a", "c"])


if __name__ == "__main__":
    unittest.main()
