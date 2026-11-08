import unittest
import torch
from retrieval_lab.multivers_joint import encode_windows, joint_loss


class WindowTests(unittest.TestCase):
    class Tokenizer:
        bos_token_id = 0
        sep_token_id = 2
        def __call__(self, text, add_special_tokens=False):
            return {'input_ids': [3] * len(text.split())}

    def test_windows_preserve_every_sentence_and_explicit_positions(self):
        sentences = ['a b c d'] * 8
        windows = encode_windows(self.Tokenizer(), 'scientific claim', sentences, max_length=16)
        self.assertEqual(set(s for w in windows for s in w.sentence_ids), set(range(8)))
        for window in windows:
            self.assertLessEqual(len(window.input_ids), 16)
            self.assertEqual(len(window.sentence_ids), len(window.sentence_positions))
            self.assertTrue(all(window.input_ids[p] == 2 for p in window.sentence_positions))
            self.assertEqual(window.global_attention_mask[:5], [1] * 5)
            self.assertTrue(all(value == 0 for value in window.global_attention_mask[5:]))

    def test_oversized_sentence_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            encode_windows(self.Tokenizer(), 'claim', ['word ' * 30], max_length=16)


class JointLossTests(unittest.TestCase):
    def test_partial_window_and_unknown_rationale_targets_have_zero_gradient(self):
        labels = torch.tensor([[1., 2., 3.], [3., 2., 1.]], requires_grad=True)
        rationales = torch.zeros((2, 3), requires_grad=True)
        result = joint_loss(labels, rationales, torch.tensor([1, 0]),
                            torch.tensor([[1, 0, -1], [-1, -1, -1]]),
                            torch.ones((2, 3), dtype=torch.bool),
                            label_mask=torch.tensor([True, False]))
        result['loss'].backward()
        self.assertTrue(torch.equal(labels.grad[1], torch.zeros(3)))
        self.assertEqual(rationales.grad[0, 2].item(), 0.)
        self.assertTrue(torch.equal(rationales.grad[1], torch.zeros(3)))
        self.assertLess(rationales.grad[0, 0].item(), 0.)
        self.assertGreater(rationales.grad[0, 1].item(), 0.)

    def test_no_rationale_annotations_produce_finite_zero_rationale_loss(self):
        result = joint_loss(torch.zeros((1, 3)), torch.zeros((1, 2)), torch.tensor([0]),
                            torch.full((1, 2), -1), torch.ones((1, 2), dtype=torch.bool))
        self.assertEqual(result['rationale_loss'].item(), 0.)
        self.assertTrue(torch.isfinite(result['loss']).item())


if __name__ == '__main__': unittest.main()
