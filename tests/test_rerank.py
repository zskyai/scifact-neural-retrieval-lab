import unittest

from retrieval_lab.data import Document, QueryExample
from retrieval_lab.metrics import SearchResult
from retrieval_lab.rerank import rerank_run


class KeywordScorer:
    def score(self, query, documents):
        return [float(document.lower().count("relevant")) for document in documents]


class RerankTest(unittest.TestCase):
    def test_reranker_only_changes_head(self):
        documents = [
            Document("d1", "irrelevant", ()),
            Document("d2", "relevant relevant", ()),
            Document("d3", "tail", ()),
        ]
        queries = [QueryExample("q1", "query", frozenset({"d2"}))]
        run = {
            "q1": [
                SearchResult("d1", 3.0),
                SearchResult("d2", 2.0),
                SearchResult("d3", 1.0),
            ]
        }
        reranked = rerank_run(run, queries, documents, KeywordScorer(), rerank_depth=2)
        self.assertEqual([item.doc_id for item in reranked["q1"]], ["d2", "d1", "d3"])


if __name__ == "__main__":
    unittest.main()

