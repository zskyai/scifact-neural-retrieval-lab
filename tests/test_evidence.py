import unittest

from retrieval_lab.data import Document
from retrieval_lab.evidence import select_evidence_sentences


class EvidenceSelectionTest(unittest.TestCase):
    def test_selector_returns_abstract_indices_in_reading_order(self):
        document = Document(
            "d1",
            "",
            (
                "This sentence discusses astronomy.",
                "PD-1 blockade activates T cells in cancer.",
                "This sentence discusses microscopy.",
            ),
        )
        selected = select_evidence_sentences("PD-1 cancer T cells", document, top_k=2)
        self.assertEqual([snippet.sentence_id for snippet in selected], [0, 1])
        self.assertEqual(selected[-1].text, "PD-1 blockade activates T cells in cancer.")

    def test_selector_rejects_non_positive_k(self):
        with self.assertRaises(ValueError):
            select_evidence_sentences("query", Document("d1", "", ("text",)), top_k=0)


if __name__ == "__main__":
    unittest.main()
