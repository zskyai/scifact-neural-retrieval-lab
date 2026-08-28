import unittest

from retrieval_lab.negatives import filter_false_negatives


class FalseNegativeFilterTest(unittest.TestCase):
    def test_gold_and_teacher_ambiguous_candidates_are_removed(self):
        filtered = filter_false_negatives(
            ["positive", "near_duplicate", "safe_negative"],
            ["positive"],
            num_negatives=2,
            candidate_teacher_scores=[1.0, 0.95, 0.1],
            positive_teacher_scores=[1.0],
            teacher_margin=0.1,
        )
        self.assertEqual(filtered.selected_doc_ids, ("safe_negative",))
        self.assertEqual(filtered.removed_gold, 1)
        self.assertEqual(filtered.removed_by_teacher, 1)

    def test_duplicate_candidates_are_not_repeated(self):
        filtered = filter_false_negatives(
            ["n1", "n1", "n2"],
            [],
            num_negatives=3,
        )
        self.assertEqual(filtered.selected_doc_ids, ("n1", "n2"))


if __name__ == "__main__":
    unittest.main()

