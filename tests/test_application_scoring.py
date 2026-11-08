import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('application_eval', Path(__file__).resolve().parents[1] / 'scripts/evaluate_review_application.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OfficialLabelTests(unittest.TestCase):
    def test_official_labels_map_to_application_vocabulary(self):
        self.assertEqual(module.canonical_gold_label('SUPPORT'), 'SUPPORTS')
        self.assertEqual(module.canonical_gold_label('CONTRADICT'), 'CONTRADICTS')
        self.assertEqual(module.canonical_gold_label('NEI'), 'NEI')
        with self.assertRaises(ValueError): module.canonical_gold_label('UNKNOWN')


if __name__ == '__main__': unittest.main()
