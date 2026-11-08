import unittest

from retrieval_lab.evidence_agent import EvidenceRetrievalAgent


class EvidenceRetrievalAgentTests(unittest.TestCase):
    def test_refuting_evidence_returns_contradicts(self):
        agent = EvidenceRetrievalAgent(
            lambda _query, _k: [{"doc_id": "d1", "score": 0.08}, {"doc_id": "d2", "score": 0.01}],
            lambda _claim, _docs: {"coverage": 0.9, "support": 0.01, "contradiction": 0.95},
            max_retries=0,
        )
        result = agent.run("The treatment improves survival")
        self.assertEqual(result.action, "answer")
        self.assertEqual(result.label, "CONTRADICTS")

    def test_weak_first_pass_retries_then_answers(self):
        calls = []

        def retrieve(query, top_k):
            calls.append((query, top_k))
            return [{"doc_id": "d1", "score": 0.04, "text": "evidence"}]

        def verify(_claim, _documents):
            if len(calls) == 1:
                return {"coverage": 0.1, "support": 0.2, "contradiction": 0.0}
            return {"coverage": 0.9, "support": 0.9, "contradiction": 0.0}

        result = EvidenceRetrievalAgent(retrieve, verify, max_retries=1, top_k=1).run(
            "A treatment reduces the outcome and improves recovery"
        )
        self.assertEqual(result.action, "answer")
        self.assertEqual(len(result.trace), 4)
        self.assertEqual(calls[0][1], 1)
        self.assertEqual(calls[1][1], 2)

    def test_conflicting_evidence_exhausts_to_nei(self):
        agent = EvidenceRetrievalAgent(
            lambda _query, _k: [{"doc_id": "d1", "score": 0.04}],
            lambda _claim, _docs: {"coverage": 0.9, "support": 0.6, "contradiction": 0.9},
            max_retries=0,
        )
        result = agent.run("The study supports the treatment")
        self.assertEqual(result.action, "abstain")
        self.assertEqual(result.label, "NEI")


if __name__ == "__main__":
    unittest.main()
