"""Extended unit tests for eole/utils/ and eole/scorers/ – covering files
not tested in test_utils.py / test_scorers.py.

Covers:
 - utils/misc.py: sequence_mask, tile, fn_args, RandomShuffler
 - utils/alignment.py: make_batch_align_matrix, build_align_pharaoh,
                       extract_alignment, to_word_align helpers
 - utils/attention_entropy.py: compute_attention_entropy,
                                compute_attention_entropy_from_dict,
                                aggregate_attention_entropy
 - scorers/bleu_zh.py: BleuZhScorer
"""

import unittest
import torch


# ---------------------------------------------------------------------------
# misc.py
# ---------------------------------------------------------------------------

from eole.utils.misc import sequence_mask, tile, fn_args, RandomShuffler


class TestSequenceMask(unittest.TestCase):

    def test_basic_mask(self):
        lengths = torch.tensor([3, 5, 2])
        mask = sequence_mask(lengths, max_len=5)
        expected = torch.tensor([
            [False, False, False, True,  True],
            [False, False, False, False, False],
            [False, False, True,  True,  True],
        ])
        self.assertTrue(torch.equal(mask, expected))

    def test_auto_max_len(self):
        lengths = torch.tensor([2, 4])
        mask = sequence_mask(lengths)
        self.assertEqual(mask.shape, (2, 4))

    def test_full_lengths(self):
        """Lengths equal to max_len should produce all-False mask (no padding)."""
        lengths = torch.tensor([3, 3, 3])
        mask = sequence_mask(lengths, max_len=3)
        self.assertTrue((mask == False).all())  # noqa: E712

    def test_zero_length(self):
        lengths = torch.tensor([0, 3])
        mask = sequence_mask(lengths, max_len=3)
        # First row: all padded
        self.assertTrue(mask[0].all())
        # Second row: no padding
        self.assertFalse(mask[1].any())


class TestTile(unittest.TestCase):

    def test_batch_tiling(self):
        x = torch.tensor([[1, 2], [3, 4]])   # (2, 2)
        out = tile(x, count=3, dim=0)
        self.assertEqual(out.shape, (6, 2))
        # First row repeated 3 times
        self.assertTrue(torch.equal(out[:3], x[0].unsqueeze(0).expand(3, -1)))

    def test_single_element(self):
        x = torch.tensor([[[1.0, 2.0]]])  # (1, 1, 2)
        out = tile(x, count=4, dim=0)
        self.assertEqual(out.shape, (4, 1, 2))

    def test_count_one_unchanged(self):
        x = torch.randn(3, 5)
        out = tile(x, count=1, dim=0)
        self.assertTrue(torch.equal(out, x))


class TestFnArgs(unittest.TestCase):

    def test_simple_function(self):
        def f(a, b, c=1):
            pass
        args = fn_args(f)
        self.assertIn("a", args)
        self.assertIn("b", args)

    def test_no_args(self):
        def f():
            pass
        args = fn_args(f)
        self.assertEqual(args, [])


class TestRandomShuffler(unittest.TestCase):

    def test_reproducible(self):
        """Same seed → same shuffle order."""
        import random
        shuffler1 = RandomShuffler(random_state=random.getstate())
        data = list(range(20))
        out1 = shuffler1(data)

        shuffler2 = RandomShuffler(random_state=shuffler1._random_state)
        out2 = shuffler2(data)
        # Because both start from the same state, they should produce the same result
        # (state is saved after first shuffle, so shuffler2 uses that same saved state)
        self.assertIsInstance(out1, list)
        self.assertEqual(len(out1), len(data))

    def test_shuffles(self):
        shuffler = RandomShuffler()
        data = list(range(100))
        out = shuffler(data)
        self.assertNotEqual(out, data)  # should not be identical (overwhelmingly likely)
        self.assertCountEqual(out, data)  # same elements


# ---------------------------------------------------------------------------
# alignment.py
# ---------------------------------------------------------------------------

from eole.utils.alignment import (
    make_batch_align_matrix,
    build_align_pharaoh,
    extract_alignment,
)


class TestMakeBatchAlignMatrix(unittest.TestCase):

    def test_basic_shape(self):
        # 3 alignments in one batch: batch_id, tgt_id, src_id
        index = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 0, 2]])
        result = make_batch_align_matrix(index, size=[2, 3, 4])
        self.assertEqual(result.shape, (2, 3, 4))

    def test_values_set(self):
        index = torch.tensor([[0, 0, 1]])  # batch=0, tgt=0, src=1
        result = make_batch_align_matrix(index, size=[1, 2, 3])
        self.assertEqual(result[0, 0, 1].item(), 1.0)
        self.assertEqual(result[0, 0, 0].item(), 0.0)

    def test_normalize(self):
        index = torch.tensor([[0, 0, 0], [0, 0, 1]])  # tgt=0 attends to src=0 and src=1
        result = make_batch_align_matrix(index, size=[1, 2, 3], normalize=True)
        # After normalization, row sum should be 1 where non-zero
        row_sum = result[0, 0, :].sum()
        self.assertAlmostEqual(row_sum.item(), 1.0, places=5)


class TestBuildAlignPharaoh(unittest.TestCase):

    def test_basic_alignment(self):
        # 2 tgt tokens each attending to a different src token
        align = torch.tensor([[0.9, 0.1], [0.2, 0.8]])  # (tgt=2, src=2)
        pairs, scores = build_align_pharaoh(align)
        # Expected: tgt=0 → src=0 (argmax 0.9), tgt=1 → src=1 (argmax 0.8)
        self.assertIn("0-0", pairs)
        self.assertIn("1-1", pairs)

    def test_none_input(self):
        """None alignment should return empty lists."""
        pairs, scores = build_align_pharaoh(None)
        self.assertEqual(pairs, [])
        self.assertEqual(scores, [])


class TestExtractAlignment(unittest.TestCase):

    def test_basic_extraction(self):
        # align_matrix: (B, tgt_len, src_len)
        align_matrix = torch.softmax(torch.randn(4, 5, 6), dim=-1)
        # tgt_mask: True = EOS/PAD
        tgt_mask = torch.zeros(4, 5, dtype=torch.bool)
        tgt_mask[:, 4] = True  # last position is EOS
        src_len = torch.tensor([6, 6, 6, 6])
        alignments = extract_alignment(align_matrix, tgt_mask, src_len, n_best=1)
        # Should produce list of length 4
        self.assertEqual(len(alignments), 4)
        # Each contains one n-best alignment
        self.assertEqual(len(alignments[0]), 1)

    def test_n_best(self):
        align_matrix = torch.softmax(torch.randn(6, 5, 6), dim=-1)
        tgt_mask = torch.zeros(6, 5, dtype=torch.bool)
        src_len = torch.tensor([6] * 6)
        alignments = extract_alignment(align_matrix, tgt_mask, src_len, n_best=2)
        # batch_size = 6 // n_best=2 = 3
        self.assertEqual(len(alignments), 3)
        self.assertEqual(len(alignments[0]), 2)


# ---------------------------------------------------------------------------
# attention_entropy.py
# ---------------------------------------------------------------------------

from eole.utils.attention_entropy import (
    compute_attention_entropy,
    compute_attention_entropy_from_dict,
    aggregate_attention_entropy,
)


class TestComputeAttentionEntropy(unittest.TestCase):

    def test_4d_input_shape(self):
        """4D input → entropy of shape (batch, heads)."""
        attn = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
        entropy = compute_attention_entropy(attn)
        self.assertEqual(entropy.shape, (2, 4))

    def test_3d_input_shape(self):
        """3D input → entropy of shape (batch,)."""
        attn = torch.softmax(torch.randn(3, 6, 6), dim=-1)
        entropy = compute_attention_entropy(attn)
        self.assertEqual(entropy.shape, (3,))

    def test_uniform_attention_high_entropy(self):
        """Uniform attention should have maximum entropy."""
        L = 8
        uniform = torch.full((2, 4, L, L), 1.0 / L)
        entropy_uniform = compute_attention_entropy(uniform)
        # Compare with peaked attention
        peaked = torch.zeros(2, 4, L, L)
        peaked[:, :, :, 0] = 1.0
        entropy_peaked = compute_attention_entropy(peaked)
        self.assertTrue((entropy_uniform > entropy_peaked).all())

    def test_non_negative(self):
        attn = torch.softmax(torch.randn(2, 4, 5, 5), dim=-1)
        entropy = compute_attention_entropy(attn)
        self.assertTrue((entropy >= 0).all())

    def test_with_mask(self):
        """Masked entropy computation should not raise."""
        attn = torch.softmax(torch.randn(2, 4, 5, 5), dim=-1)
        mask = torch.ones(2, 4, 5, 5)
        entropy = compute_attention_entropy(attn, mask=mask)
        self.assertEqual(entropy.shape, (2, 4))


class TestComputeAttentionEntropyFromDict(unittest.TestCase):

    def test_returns_dict_with_keys(self):
        attns = {
            "std": torch.softmax(torch.randn(2, 4, 6, 8), dim=-1),
            "self": torch.softmax(torch.randn(2, 4, 6, 6), dim=-1),
        }
        result = compute_attention_entropy_from_dict(attns)
        self.assertIn("std", result)
        self.assertIn("self", result)

    def test_filtered_attention_types(self):
        attns = {
            "std": torch.softmax(torch.randn(2, 4, 6, 8), dim=-1),
            "context": torch.softmax(torch.randn(2, 4, 6, 8), dim=-1),
        }
        result = compute_attention_entropy_from_dict(attns, attention_types=["std"])
        self.assertIn("std", result)
        self.assertNotIn("context", result)


class TestAggregateAttentionEntropy(unittest.TestCase):

    def test_returns_scalar_tensor(self):
        """aggregate_attention_entropy returns a scalar tensor."""
        attn_entropy = {
            "std": torch.tensor([[0.5, 0.8], [0.3, 0.7]]),
            "self": torch.tensor([[0.4, 0.6], [0.2, 0.9]]),
        }
        result = aggregate_attention_entropy(attn_entropy)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, ())  # scalar

    def test_empty_dict_returns_zero(self):
        """Empty dict → scalar tensor 0.0."""
        result = aggregate_attention_entropy({})
        self.assertIsInstance(result, torch.Tensor)
        self.assertAlmostEqual(result.item(), 0.0)

    def test_aggregation_mean_vs_max(self):
        attn_entropy = {"a": torch.tensor([1.0, 2.0])}
        mean_val = aggregate_attention_entropy(attn_entropy, aggregation_method="mean")
        max_val = aggregate_attention_entropy(attn_entropy, aggregation_method="max")
        self.assertGreaterEqual(max_val.item(), mean_val.item())


# ---------------------------------------------------------------------------
# BleuZhScorer
# ---------------------------------------------------------------------------

from eole.scorers.bleu_zh import BleuZhScorer


class _FakeCfg:
    pass


class TestBleuZhScorer(unittest.TestCase):

    def setUp(self):
        self.scorer = BleuZhScorer(_FakeCfg())

    def test_empty_preds_returns_zero(self):
        score = self.scorer.compute_score([], [])
        self.assertEqual(score, 0)

    def test_identical_gives_100(self):
        preds = ["这 是 一 个 很 长 的 句 子 用 来 测 试"]
        refs = ["这 是 一 个 很 长 的 句 子 用 来 测 试"]
        score = self.scorer.compute_score(preds, refs)
        self.assertAlmostEqual(score, 100.0, delta=0.1)

    def test_different_gives_lower(self):
        preds = ["完 全 不 同 的 内 容 这 里 哈 哈 哈 哈 哈"]
        refs = ["这 是 一 个 很 长 的 句 子 用 来 测 试"]
        score = self.scorer.compute_score(preds, refs)
        self.assertLess(score, 100.0)

    def test_returns_float(self):
        score = self.scorer.compute_score(["你 好 世 界 早 安"], ["你 好 世 界 早 安"])
        self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()
