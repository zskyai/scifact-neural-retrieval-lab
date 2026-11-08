import unittest
from types import SimpleNamespace
from retrieval_lab.data import Document
from retrieval_lab.domain_verifier import aggregate_units, decide, make_units, classification_metrics, DomainReviewEngine, select_citations


class DomainVerifierTests(unittest.TestCase):
    def test_evidence_sets_stay_inside_chosen_document_and_respect_budget(self):
        units = self.units()
        units.append({'doc_id': 'a', 'kind': 'sentence', 'probabilities': [.2, .7, .1], 'sentence_id': 3})
        decision = decide(aggregate_units(units), {'threshold': .5})
        self.assertEqual(select_citations(units, decision, {'max_sentences': 2}),
                         [{'doc_id': 'a', 'sentence_id': 2}, {'doc_id': 'a', 'sentence_id': 3}])
        self.assertEqual(len(select_citations(units, decision, {'max_sentences': 2, 'min_relative_probability': .9})), 1)
        self.assertEqual(select_citations(units, {'label': 'NEI', 'citations': []}, {}), [])
        with self.assertRaises(ValueError): select_citations(units, decision, {'max_sentences': 0})

    def units(self):
        return [
            {'doc_id': 'a', 'kind': 'abstract', 'probabilities': [.1, .85, .05], 'sentence_id': None},
            {'doc_id': 'a', 'kind': 'sentence', 'probabilities': [.1, .85, .05], 'sentence_id': 2},
            {'doc_id': 'b', 'kind': 'abstract', 'probabilities': [.9, .05, .05], 'sentence_id': None},
            {'doc_id': 'b', 'kind': 'sentence', 'probabilities': [.05, .05, .9], 'sentence_id': 0},
        ]

    def test_unrelated_sentence_contradiction_is_checked_against_its_document(self):
        result = decide(aggregate_units(self.units()), {'threshold': .5, 'min_stance_margin': .1})
        self.assertEqual(result['label'], 'SUPPORTS')
        self.assertEqual(result['citations'], [{'doc_id': 'a', 'sentence_id': 2}])

    def test_insufficient_or_conflicting_evidence_abstains(self):
        ranked = aggregate_units(self.units())
        self.assertEqual(decide(ranked, {'threshold': .9})['label'], 'NEI')
        self.assertEqual(decide(ranked, {'min_stance_margin': .8})['citations'], [])
        self.assertEqual(decide([], {})['label'], 'NEI')

    def test_abstract_without_sentence_cannot_generate_citation(self):
        ranked = aggregate_units(self.units()[:1])
        self.assertEqual(decide(ranked, {'threshold': 0, 'support_bias': .1})['label'], 'NEI')

    def test_invalid_probabilities_positions_and_thresholds_fail(self):
        for bad in ([float('nan'), 0, 1], [.1, .1, .1], [-1, 1, 1]):
            units = self.units(); units[0]['probabilities'] = bad
            with self.assertRaises(ValueError): aggregate_units(units)
        units = self.units(); units[1]['sentence_id'] = -1
        with self.assertRaises(ValueError): aggregate_units(units)
        with self.assertRaises(ValueError): decide([], {'threshold': float('nan')})

    def test_metrics_keep_false_refusal_and_unsafe_answer_separate(self):
        m = classification_metrics(['SUPPORTS', 'NEI', 'CONTRADICTS'], ['NEI', 'SUPPORTS', 'CONTRADICTS'])
        self.assertEqual(m['over_refusal_rate'], .5)
        self.assertEqual(m['unsafe_answer_rate'], 1)
        with self.assertRaises(ValueError): classification_metrics(['NEI'], [])

    def test_source_positions_are_preserved_when_blank_sentences_are_skipped(self):
        doc = Document('a', 'Title', ('', 'Real evidence'))
        units = make_units([doc])
        self.assertEqual([u['sentence_id'] for u in units], [None, 1])

    def test_deployment_rejects_changed_retrieval_or_dev_calibration(self):
        config = {'retrieval': 'bm25', 'max_retries': 0, 'top_k': 10, 'dev_used': False, 'test_used': False}
        verifier = SimpleNamespace(calibration=config)
        engine = DomainReviewEngine([Document('a', 'Title', ('Evidence',))], verifier)
        with self.assertRaises(ValueError): engine.verify_claim('Claim', top_k=5)
        with self.assertRaises(ValueError): engine.verify_claim('Claim', max_retries=1)
        verifier.calibration = dict(config, dev_used=True)
        with self.assertRaises(ValueError): DomainReviewEngine(engine.documents, verifier)


if __name__ == '__main__': unittest.main()
