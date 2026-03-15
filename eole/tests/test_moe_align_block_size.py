"""Unit tests for moe_align_block_size in fused_moe_int4.py.

These tests run on CPU and do NOT require CUDA or Triton.
They verify that the Python helper correctly:
  - Produces sorted_token_ids with every valid token index present.
  - Pads each expert's token count to a multiple of block_size.
  - Sets expert_ids correctly for each GEMM block.
  - Uses M as the sentinel value for padding rows.
"""

import collections
import importlib.util
import os
import unittest

import torch


def _import_fused_moe_int4():
    """Import fused_moe_int4 directly, bypassing the triton import."""
    import sys
    import types

    if "triton" not in sys.modules:
        triton_stub = types.ModuleType("triton")
        triton_stub.jit = lambda fn: fn  # no-op decorator
        triton_stub.cdiv = lambda a, b: (a + b - 1) // b
        tl_stub = types.ModuleType("triton.language")
        tl_stub.constexpr = None
        tl_stub.float32 = None
        triton_stub.language = tl_stub
        sys.modules["triton"] = triton_stub
        sys.modules["triton.language"] = tl_stub

    mod_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "triton",
        "fused_moe_int4.py",
    )
    spec = importlib.util.spec_from_file_location("eole.triton.fused_moe_int4", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _import_fused_moe_int4()
moe_align_block_size = _mod.moe_align_block_size
_select_block_m = _mod._select_block_m


def _rand_topk_ids(M: int, topk: int, num_experts: int, seed: int = 0) -> torch.Tensor:
    """Generate (M, topk) expert IDs where each token has **unique** expert assignments.

    Real MoE routers always select K distinct experts per token; using randint can
    accidentally produce duplicates which would make pair-counting tests flaky.
    """
    torch.manual_seed(seed)
    rows = [torch.randperm(num_experts)[:topk] for _ in range(M)]
    return torch.stack(rows)  # (M, topk)


class TestMoeAlignBlockSize(unittest.TestCase):
    """Tests for moe_align_block_size."""

    def _run(self, M, topk, num_experts, block_size, seed=0):
        """Helper: generate random topk_ids, run moe_align_block_size, check invariants."""
        topk_ids = _rand_topk_ids(M, topk, num_experts, seed=seed)

        sorted_token_ids, expert_ids, num_tokens_pt, _ = moe_align_block_size(
            topk_ids, block_size, num_experts
        )
        total_padded = int(num_tokens_pt.item())

        # 1. total_padded is a multiple of block_size
        self.assertEqual(total_padded % block_size, 0)

        # 2. sorted_token_ids has exactly total_padded elements
        self.assertEqual(sorted_token_ids.numel(), total_padded)

        # 3. Valid token indices are in [0, M); padding rows == M
        valid_mask = sorted_token_ids < M
        self.assertTrue((sorted_token_ids[valid_mask] >= 0).all())
        self.assertTrue((sorted_token_ids[~valid_mask] == M).all())

        # 4. Number of valid rows == M * topk
        self.assertEqual(int(valid_mask.sum().item()), M * topk)

        # 5. expert_ids has total_padded // block_size elements
        num_blocks = total_padded // block_size
        self.assertEqual(expert_ids.numel(), num_blocks)
        self.assertTrue((expert_ids >= 0).all())
        self.assertTrue((expert_ids < num_experts).all())

        # 6. expert_ids is correct for each block
        expert_counts = torch.bincount(topk_ids.flatten().int(), minlength=num_experts)
        padded_counts = ((expert_counts + block_size - 1) // block_size) * block_size
        offset = 0
        for e in range(num_experts):
            padded = int(padded_counts[e].item())
            n_blocks = padded // block_size
            for b in range(n_blocks):
                blk_idx = offset // block_size + b
                self.assertEqual(int(expert_ids[blk_idx].item()), e,
                                 f"Block {blk_idx} should have expert {e}")
            offset += padded

        # 7. Every (token, expert) pair appears with correct multiplicity
        seen: collections.Counter = collections.Counter()
        for i in range(total_padded):
            tok = int(sorted_token_ids[i].item())
            if tok == M:
                continue
            blk = i // block_size
            e = int(expert_ids[blk].item())
            seen[(tok, e)] += 1

        expected: collections.Counter = collections.Counter()
        for i in range(M):
            for k in range(topk):
                expected[(i, int(topk_ids[i, k].item()))] += 1

        self.assertEqual(seen, expected,
                         "Mismatch between seen and expected (token, expert) pairs")

        return total_padded, expert_ids

    def test_single_token_single_expert(self):
        """M=1 token, topk=1, 4 experts."""
        self._run(M=1, topk=1, num_experts=4, block_size=16)

    def test_single_token_topk8(self):
        """M=1, topk=8 (typical single-token MoE decode)."""
        self._run(M=1, topk=8, num_experts=8, block_size=16)

    def test_small_batch(self):
        """M=4, topk=2, 8 experts."""
        self._run(M=4, topk=2, num_experts=8, block_size=16)

    def test_medium_batch(self):
        """M=32, topk=8, 8 experts (typical batch decode)."""
        self._run(M=32, topk=8, num_experts=8, block_size=16)

    def test_large_batch(self):
        """M=128, topk=2, 16 experts."""
        self._run(M=128, topk=2, num_experts=16, block_size=32)

    def test_block_size_32(self):
        """block_size=32 – larger tile."""
        self._run(M=16, topk=4, num_experts=8, block_size=32)

    def test_block_size_64(self):
        """block_size=64 – large tile."""
        self._run(M=64, topk=2, num_experts=4, block_size=64)

    def test_padding_exact_multiple(self):
        """Expert counts already multiples of block_size – no extra padding rows."""
        topk_ids = torch.cat([
            torch.zeros(8, 1, dtype=torch.long),
            torch.ones(8, 1, dtype=torch.long),
        ])
        sorted_token_ids, expert_ids, num_tokens_pt, _ = moe_align_block_size(
            topk_ids, block_size=8, num_experts=2
        )
        self.assertEqual(int(num_tokens_pt.item()), 16)
        self.assertEqual((sorted_token_ids == 16).sum().item(), 0)

    def test_empty_expert(self):
        """Some experts receive no tokens."""
        topk_ids = torch.zeros(4, 1, dtype=torch.long)  # all to expert 0
        sorted_token_ids, expert_ids, num_tokens_pt, _ = moe_align_block_size(
            topk_ids, block_size=16, num_experts=4
        )
        self.assertEqual(int(num_tokens_pt.item()), 16)
        self.assertEqual(expert_ids.numel(), 1)
        self.assertEqual(int(expert_ids[0].item()), 0)
        valid = sorted_token_ids[sorted_token_ids < 4]
        self.assertEqual(valid.numel(), 4)

    def test_sentinel_value(self):
        """Padding rows must be set to M."""
        M, topk, E = 3, 1, 4
        topk_ids = torch.zeros(M, topk, dtype=torch.long)
        sorted_token_ids, _, _, _ = moe_align_block_size(
            topk_ids, block_size=16, num_experts=E
        )
        pad_rows = sorted_token_ids[sorted_token_ids >= M]
        self.assertTrue((pad_rows == M).all())

    def test_ordering_by_expert(self):
        """expert_ids must be non-decreasing (all tokens for expert e come before e+1)."""
        topk_ids = _rand_topk_ids(M=8, topk=2, num_experts=4, seed=7)
        sorted_token_ids, expert_ids, _, _ = moe_align_block_size(topk_ids, 16, 4)
        for i in range(len(expert_ids) - 1):
            self.assertLessEqual(int(expert_ids[i].item()), int(expert_ids[i + 1].item()),
                                 "expert_ids must be non-decreasing")


class TestSelectBlockM(unittest.TestCase):
    """Tests for the BLOCK_M selection heuristic."""

    def test_small_batch_selects_small_block(self):
        bm = _select_block_m(M=1, topk=8, num_experts=8)
        self.assertEqual(bm, 16)

    def test_large_batch_selects_large_block(self):
        bm = _select_block_m(M=512, topk=2, num_experts=8)
        self.assertEqual(bm, 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
