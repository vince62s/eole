"""Unit tests for eole/modules/ – Step 4: RoPE, MLP, MHA helpers, MoE utility functions."""

import math
import unittest

import torch
import torch.nn as nn

from eole.modules.rope import (
    apply_rotary_emb,
    NoOpPosition,
    RotaryPosition,
    build_rope,
)
from eole.modules.multi_headed_attn import bld_to_blhd, blhd_to_bld
from eole.modules.transformer_mlp import MLP
from eole.modules.moe import naive_moe, vectorized_moe
from eole.constants import PositionEncodingType


# ---------------------------------------------------------------------------
# Helpers to build lightweight model configs
# ---------------------------------------------------------------------------


def _make_transformer_config(
    hidden_size=64,
    heads=4,
    transformer_ff=128,
    position_encoding_type=PositionEncodingType.Rotary,
    rotary_interleave=False,
    rotary_theta=10000,
    mlp_activation_fn="relu",
    add_ffnbias=False,
    num_experts=0,
    num_experts_per_tok=2,
    moe_transformer_ff=None,
    head_dim=None,
    num_shared_experts=0,
):
    from eole.config.models import CustomModelConfig
    from eole.constants import PositionEncodingType as PET

    cfg = CustomModelConfig(
        hidden_size=hidden_size,
        heads=heads,
        transformer_ff=transformer_ff,
        position_encoding_type=position_encoding_type,
        mlp_activation_fn=mlp_activation_fn,
        add_ffnbias=add_ffnbias,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_shared_experts=num_shared_experts,
    )
    if position_encoding_type == PET.Rotary and cfg.rope_config is not None:
        cfg.rope_config.rotary_interleave = rotary_interleave
        cfg.rope_config.rotary_theta = rotary_theta
    return cfg


class _FakeRunningConfig:
    """Minimal running config that MLP / modules expect."""
    parallel_gpu = 1
    dropout = [0.0]
    attention_dropout = [0.0]
    use_ckpting = []
    self_attn_backend = "pytorch"


# ===========================================================================
# NoOpPosition
# ===========================================================================


class TestNoOpPosition(unittest.TestCase):

    def test_cos_sin_is_none(self):
        nop = NoOpPosition()
        self.assertIsNone(nop.cos_sin)

    def test_update_returns_none(self):
        nop = NoOpPosition()
        result = nop.update(32)
        self.assertIsNone(result)


# ===========================================================================
# build_rope
# ===========================================================================


class TestBuildRope(unittest.TestCase):

    def test_returns_rotary_for_rotary_encoding(self):
        cfg = _make_transformer_config(position_encoding_type=PositionEncodingType.Rotary)
        rope = build_rope(cfg)
        self.assertIsInstance(rope, RotaryPosition)

    def test_returns_noop_for_sinusoidal(self):
        cfg = _make_transformer_config(position_encoding_type=PositionEncodingType.SinusoidalInterleaved)
        rope = build_rope(cfg)
        self.assertIsInstance(rope, NoOpPosition)


# ===========================================================================
# RotaryPosition
# ===========================================================================


class TestRotaryPosition(unittest.TestCase):

    def _make_rope(self, hidden_size=64, heads=4, interleave=False):
        cfg = _make_transformer_config(hidden_size=hidden_size, heads=heads, rotary_interleave=interleave)
        return RotaryPosition(cfg)

    def test_inv_freq_shape(self):
        rope = self._make_rope(hidden_size=64, heads=4)  # dim_per_head = 16
        # inv_freq has shape (dim_per_head/2,)
        self.assertEqual(rope.inv_freq.shape, (8,))

    def test_update_returns_cos_sin_tensor(self):
        rope = self._make_rope()
        result = rope.update(seq_len=16)
        self.assertIsInstance(result, torch.Tensor)
        # shape: (seq_len, dim_per_head)
        self.assertEqual(result.shape[0], 32768)  # pre-allocated to 32768

    def test_cos_sin_cached_after_update(self):
        rope = self._make_rope()
        rope.update(seq_len=16)
        self.assertIsNotNone(rope.cos_sin)

    def test_dim_per_head(self):
        rope = self._make_rope(hidden_size=128, heads=8)
        self.assertEqual(rope.dim_per_head, 16)


# ===========================================================================
# apply_rotary_emb
# ===========================================================================


class TestApplyRotaryEmb(unittest.TestCase):

    def _make_cos_sin(self, seq_len, dim):
        """Build a (seq_len, dim) cos/sin tensor similar to what RotaryPosition produces."""
        t = torch.arange(seq_len, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 2).float() / (dim // 2)))
        rope = torch.outer(t, inv_freq)
        cos = rope.cos()
        sin = rope.sin()
        return torch.cat([cos, sin], dim=-1)

    def test_output_shape_matches_input(self):
        B, S, H, D = 2, 8, 4, 16
        query = torch.randn(B, S, H, D)
        key = torch.randn(B, S, H, D)
        cos_sin = self._make_cos_sin(S, D)
        q_out, k_out = apply_rotary_emb(query, key, cos_sin, interleave=False)
        self.assertEqual(q_out.shape, query.shape)
        self.assertEqual(k_out.shape, key.shape)

    def test_interleaved_output_shape(self):
        B, S, H, D = 2, 8, 4, 16
        query = torch.randn(B, S, H, D)
        key = torch.randn(B, S, H, D)
        cos_sin = self._make_cos_sin(S, D)
        q_out, k_out = apply_rotary_emb(query, key, cos_sin, interleave=True)
        self.assertEqual(q_out.shape, query.shape)
        self.assertEqual(k_out.shape, key.shape)

    def test_rotation_changes_values(self):
        """Applying RoPE should change the query/key tensors."""
        B, S, H, D = 1, 4, 2, 8
        query = torch.ones(B, S, H, D)
        key = torch.ones(B, S, H, D)
        cos_sin = self._make_cos_sin(S, D)
        q_out, k_out = apply_rotary_emb(query, key, cos_sin, interleave=False)
        self.assertFalse(torch.allclose(q_out, query))

    def test_gqa_key_fewer_heads(self):
        """GQA: key has fewer heads than query."""
        B, S = 2, 6
        H_q, H_k, D = 8, 2, 16
        query = torch.randn(B, S, H_q, D)
        key = torch.randn(B, S, H_k, D)
        cos_sin = self._make_cos_sin(S, D)
        q_out, k_out = apply_rotary_emb(query, key, cos_sin, interleave=False)
        self.assertEqual(q_out.shape, (B, S, H_q, D))
        self.assertEqual(k_out.shape, (B, S, H_k, D))


# ===========================================================================
# bld_to_blhd / blhd_to_bld
# ===========================================================================


class TestBLDHelpers(unittest.TestCase):

    def test_bld_to_blhd_shape(self):
        B, L, D = 3, 10, 64
        dim_per_head = 16
        x = torch.randn(B, L, D)
        result = bld_to_blhd(x, dim_per_head)
        expected_heads = D // dim_per_head
        self.assertEqual(result.shape, (B, L, expected_heads, dim_per_head))

    def test_blhd_to_bld_shape(self):
        B, L, H, D = 3, 10, 4, 16
        x = torch.randn(B, L, H, D)
        result = blhd_to_bld(x)
        self.assertEqual(result.shape, (B, L, H * D))

    def test_roundtrip(self):
        B, L, D = 2, 5, 32
        dim_per_head = 8
        x = torch.randn(B, L, D)
        result = blhd_to_bld(bld_to_blhd(x, dim_per_head))
        self.assertTrue(torch.allclose(x, result))


# ===========================================================================
# MLP (simple and gated)
# ===========================================================================


class TestMLP(unittest.TestCase):

    def _make_simple_mlp(self, hidden_size=32, transformer_ff=64):
        cfg = _make_transformer_config(hidden_size=hidden_size, transformer_ff=transformer_ff,
                                       mlp_activation_fn="relu")
        return MLP(cfg, running_config=_FakeRunningConfig())

    def _make_gated_mlp(self, hidden_size=32, transformer_ff=64):
        cfg = _make_transformer_config(hidden_size=hidden_size, transformer_ff=transformer_ff,
                                       mlp_activation_fn="gated-silu")
        return MLP(cfg, running_config=_FakeRunningConfig())

    def test_simple_forward_shape(self):
        mlp = self._make_simple_mlp()
        mlp.eval()
        with torch.no_grad():
            x = torch.randn(2, 5, 32)
            y = mlp(x)
        self.assertEqual(y.shape, x.shape)

    def test_gated_forward_shape(self):
        mlp = self._make_gated_mlp()
        mlp.eval()
        with torch.no_grad():
            x = torch.randn(2, 5, 32)
            y = mlp(x)
        self.assertEqual(y.shape, x.shape)

    def test_simple_forward_values_are_finite(self):
        mlp = self._make_simple_mlp()
        # skip_init leaves weights uninitialized; explicitly initialize them
        nn.init.normal_(mlp.gate_up_proj.weight)
        nn.init.normal_(mlp.down_proj.weight)
        mlp.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 32)
            y = mlp(x)
        self.assertTrue(torch.isfinite(y).all())

    def test_update_dropout(self):
        mlp = self._make_simple_mlp()
        mlp.update_dropout(0.5)
        self.assertAlmostEqual(mlp.dropout_1.p, 0.5)
        self.assertAlmostEqual(mlp.dropout_2.p, 0.5)

    def test_gated_mlp_has_up_proj(self):
        mlp = self._make_gated_mlp()
        self.assertIsNotNone(mlp.up_proj)

    def test_simple_mlp_no_up_proj(self):
        mlp = self._make_simple_mlp()
        self.assertIsNone(mlp.up_proj)


# ===========================================================================
# naive_moe / vectorized_moe utilities
# ===========================================================================


class _SimpleExpert(nn.Module):
    """Minimal expert module for MoE tests."""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.linear(x)


class TestNaiveMoE(unittest.TestCase):

    def _run(self, BT, hidden, num_experts, K):
        experts = nn.ModuleList([_SimpleExpert(hidden) for _ in range(num_experts)])
        x = torch.randn(BT * K, hidden)
        # topk_weights and topk_ids: (BT, K)
        topk_weights = torch.softmax(torch.randn(BT, K), dim=-1)
        topk_ids = torch.zeros(BT, K, dtype=torch.long)
        for i in range(BT):
            topk_ids[i] = torch.randperm(num_experts)[:K]
        return naive_moe(x, topk_weights, topk_ids, K, experts)

    def test_output_shape(self):
        out = self._run(BT=4, hidden=16, num_experts=4, K=2)
        self.assertEqual(out.shape, (4, 16))

    def test_output_is_finite(self):
        out = self._run(BT=4, hidden=16, num_experts=4, K=2)
        self.assertTrue(torch.isfinite(out).all())


class TestVectorizedMoE(unittest.TestCase):

    def _run(self, BT, hidden, num_experts, K):
        experts = nn.ModuleList([_SimpleExpert(hidden) for _ in range(num_experts)])
        x = torch.randn(BT * K, hidden)
        topk_weights = torch.softmax(torch.randn(BT, K), dim=-1)
        topk_ids = torch.zeros(BT, K, dtype=torch.long)
        for i in range(BT):
            topk_ids[i] = torch.randperm(num_experts)[:K]
        return vectorized_moe(x, topk_weights, topk_ids, K, experts)

    def test_output_shape(self):
        out = self._run(BT=4, hidden=16, num_experts=4, K=2)
        self.assertEqual(out.shape, (4, 16))

    def test_output_is_finite(self):
        out = self._run(BT=4, hidden=16, num_experts=4, K=2)
        self.assertTrue(torch.isfinite(out).all())

    def test_empty_input(self):
        """Vectorized MoE handles BT=0 gracefully."""
        experts = nn.ModuleList([_SimpleExpert(16) for _ in range(4)])
        x = torch.randn(0, 16)
        topk_weights = torch.softmax(torch.randn(0, 2), dim=-1)
        topk_ids = torch.zeros(0, 2, dtype=torch.long)
        out = vectorized_moe(x, topk_weights, topk_ids, 2, experts)
        self.assertEqual(out.shape[1], 16)

    def test_naive_and_vectorized_equivalent(self):
        """naive_moe and vectorized_moe should produce the same output."""
        BT, hidden, num_experts, K = 6, 16, 4, 2
        experts = nn.ModuleList([_SimpleExpert(hidden) for _ in range(num_experts)])
        # Use the same experts for both
        x = torch.randn(BT * K, hidden)
        topk_weights = torch.softmax(torch.randn(BT, K), dim=-1)
        # Use a fixed assignment for reproducibility
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]])

        with torch.no_grad():
            out_naive = naive_moe(x.clone(), topk_weights.clone(), topk_ids.clone(), K, experts)
            out_vec = vectorized_moe(x.clone(), topk_weights.clone(), topk_ids.clone(), K, experts)

        self.assertTrue(torch.allclose(out_naive, out_vec, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
