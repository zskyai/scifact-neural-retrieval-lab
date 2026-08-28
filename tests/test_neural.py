import unittest

import torch

from retrieval_lab.neural import info_nce_loss, mean_pool


class NeuralComponentsTest(unittest.TestCase):
    def test_mean_pool_ignores_padding(self):
        hidden = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [100.0, 100.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = mean_pool(hidden, mask)
        torch.testing.assert_close(pooled, torch.tensor([[2.0, 1.0]]))

    def test_infonce_prefers_aligned_positive(self):
        query = torch.tensor([[1.0, 0.0]], requires_grad=True)
        positive_good = torch.tensor([[1.0, 0.0]])
        positive_bad = torch.tensor([[-1.0, 0.0]])
        negatives = torch.tensor([[[0.0, 1.0]]])
        good_loss = info_nce_loss(query, positive_good, negatives, temperature=0.2)
        bad_loss = info_nce_loss(query, positive_bad, negatives, temperature=0.2)
        self.assertLess(float(good_loss), float(bad_loss))
        good_loss.backward()
        self.assertIsNotNone(query.grad)

    def test_false_negative_mask_changes_loss(self):
        query = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[1.0, 0.0]])
        negatives = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        unmasked = info_nce_loss(query, positive, negatives)
        masked = info_nce_loss(
            query,
            positive,
            negatives,
            negative_mask=torch.tensor([[False, True]]),
        )
        self.assertLess(float(masked), float(unmasked))


if __name__ == "__main__":
    unittest.main()
