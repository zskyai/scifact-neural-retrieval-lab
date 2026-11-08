import unittest
from retrieval_lab.data import Document
from retrieval_lab.evidence_training import isolated_document_pools, build_complete_rationale_examples


class EvidenceTrainingTests(unittest.TestCase):
    def row(self):
        return {'id': 1, 'claim': 'A scientific claim', 'cited_doc_ids': [10],
                'evidence': {'10': [{'label': 'SUPPORT', 'sentences': [0, 2]}]}}

    def tokenizer(self, claim, text, **kwargs):
        self.assertIs(kwargs['truncation'], False)
        return {'input_ids': list(range(len(claim.split()) + len(text.split()) + 3))}

    def test_joint_rationale_is_complete_and_not_split_into_positive_members(self):
        doc = Document('10', 'Title', ('First premise.', 'Irrelevant sentence.', 'Second premise.'))
        rows, excluded, audit = build_complete_rationale_examples([self.row()], {'10': doc}, self.tokenizer)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['sentence_ids'], [0, 2])
        self.assertEqual(rows[0]['text'], 'First premise. Second premise.')
        self.assertEqual(excluded, [])
        self.assertFalse(audit['multi_sentence_members_labeled_independently'])

    def test_over_budget_gold_is_recorded_instead_of_partially_supervised(self):
        doc = Document('10', 'Title', ('One premise.', 'Ignored.', 'Another premise.'))
        rows, excluded, audit = build_complete_rationale_examples([self.row()], {'10': doc}, self.tokenizer, max_length=6)
        self.assertEqual(rows, [])
        self.assertEqual(excluded[0]['sentence_ids'], [0, 2])
        self.assertEqual(audit['excluded_rationale_sets'], 1)
        self.assertFalse(audit['truncation_used'])

    def test_paper_isolation_includes_background_negative_candidates(self):
        docs = [Document(str(i), 'Title', ('Text.',)) for i in range(30)]
        train = [self.row()]
        heldout = [{'id': 2, 'cited_doc_ids': [11], 'evidence': {}}]
        pools = isolated_document_pools(docs, train, heldout)
        self.assertEqual(pools[0] & pools[1], set())
        self.assertEqual(pools[0] | pools[1], {str(i) for i in range(30)})
        self.assertIn('10', pools[0])
        self.assertIn('11', pools[1])
        self.assertEqual(pools, isolated_document_pools(list(reversed(docs)), train, heldout))
        with self.assertRaises(ValueError):
            isolated_document_pools(docs, train, [{'cited_doc_ids': [10]}])


if __name__ == '__main__': unittest.main()
