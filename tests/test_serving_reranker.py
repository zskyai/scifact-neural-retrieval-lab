import unittest
from types import SimpleNamespace
import torch
from retrieval_lab.serving import CrossEncoderScorer


class RerankerOutputTests(unittest.TestCase):
    def scorer(self, logits):
        obj = CrossEncoderScorer.__new__(CrossEncoderScorer)
        obj.device = torch.device('cpu')
        obj.max_length = 128
        obj.tokenizer = lambda *args, **kwargs: {'input_ids': torch.ones((len(args[0]), 2), dtype=torch.long)}
        obj.model = lambda **kwargs: SimpleNamespace(logits=torch.tensor(logits, dtype=torch.float32))
        return obj

    def test_binary_head_returns_one_margin_per_candidate(self):
        scorer = self.scorer([[3, 1], [-2, 4]])
        self.assertEqual(scorer.score('claim', ['irrelevant', 'relevant']), [-2.0, 6.0])

    def test_single_head_preserves_relevance_scores(self):
        self.assertEqual(self.scorer([[.5], [2]]).score('claim', ['a', 'b']), [.5, 2])


if __name__ == '__main__': unittest.main()
