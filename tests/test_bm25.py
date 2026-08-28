import unittest

from retrieval_lab.bm25 import BM25Config, BM25Index, tokenize
from retrieval_lab.data import Document, QueryExample


class BM25Test(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document("d1", "Cancer immunotherapy", ("PD-1 blockade activates T cells.",)),
            Document("d2", "Astronomy", ("A telescope observes distant galaxies.",)),
            Document("d3", "Cancer imaging", ("MRI measures tumor volume.",)),
        ]

    def test_tokenizer_is_case_insensitive(self):
        self.assertEqual(tokenize("PD-1 Blockade"), ["pd", "1", "blockade"])

    def test_relevant_document_ranks_first(self):
        index = BM25Index(self.documents, BM25Config(title_boost=2))
        results = index.search("PD-1 cancer T cells", top_k=3)
        self.assertEqual(results[0].doc_id, "d1")

    def test_batch_search_uses_query_ids(self):
        index = BM25Index(self.documents)
        run = index.batch_search([QueryExample("q1", "galaxies", frozenset({"d2"}))], top_k=2)
        self.assertEqual(run["q1"][0].doc_id, "d2")


if __name__ == "__main__":
    unittest.main()
