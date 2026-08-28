import unittest

from retrieval_lab.data import QueryExample
from retrieval_lab.metrics import SearchResult, evaluate_run


class MetricsTest(unittest.TestCase):
    def test_binary_retrieval_metrics(self):
        queries = [
            QueryExample("q1", "one", frozenset({"d1"})),
            QueryExample("q2", "two", frozenset({"d2"})),
        ]
        run = {
            "q1": [SearchResult("d1", 2.0), SearchResult("d9", 1.0)],
            "q2": [SearchResult("d9", 2.0), SearchResult("d2", 1.0)],
        }
        metrics = evaluate_run(queries, run, ks=[1, 2])
        self.assertAlmostEqual(metrics["recall@1"], 0.5)
        self.assertAlmostEqual(metrics["recall@2"], 1.0)
        self.assertAlmostEqual(metrics["mrr@2"], 0.75)
        expected_ndcg = (1.0 + 1.0 / 1.584962500721156) / 2.0
        self.assertAlmostEqual(metrics["ndcg@2"], expected_ndcg)

    def test_unlabeled_queries_are_reported_but_not_scored(self):
        queries = [
            QueryExample("q1", "one", frozenset({"d1"})),
            QueryExample("q2", "unlabeled", frozenset()),
        ]
        metrics = evaluate_run(queries, {"q1": [SearchResult("d1", 1.0)]}, ks=[1])
        self.assertEqual(metrics["evaluated_queries"], 1)
        self.assertEqual(metrics["total_queries"], 2)


if __name__ == "__main__":
    unittest.main()
