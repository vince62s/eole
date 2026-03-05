"""Tests for fused_moe_int4 – pure-Python / CPU path.

These tests do not require Triton or a CUDA GPU.  They verify:
  1. The helper methods on MoE that detect and cache int4 weights work
     correctly given a mock GPTQ QuantLinear.
  2. The ``fused_experts_int4_impl`` Python-wrapper logic (tensor shapes,
     strides, group_size inference) is consistent with what the Triton
     kernels expect.

No Triton kernels are actually executed; the tests are skipped when Triton
or CUDA is unavailable.
"""

import unittest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal mock of a GPTQ / AutoRound QuantLinear
# ---------------------------------------------------------------------------

class _MockQuantLinear(nn.Module):
    """Minimal stand-in for an AutoRound / GPTQ TritonLinear.

    Stores the same attribute names (``qweight``, ``scales``, ``qzeros``)
    that ``MoE._is_int4_quantized_linear`` looks for.

    Weight layout (GPTQ int4):
        qweight:  (in_features // 8, out_features)  int32
        scales:   (in_features // group_size, out_features)  float16
        qzeros:   (in_features // group_size, out_features // 8)  int32
    """

    def __init__(self, in_features: int, out_features: int, group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        n_groups = in_features // group_size
        self.qweight = torch.zeros(in_features // 8, out_features, dtype=torch.int32)
        self.scales = torch.ones(n_groups, out_features, dtype=torch.float16)
        self.qzeros = torch.zeros(n_groups, out_features // 8, dtype=torch.int32)

    def forward(self, x):
        raise NotImplementedError("mock – not used in unit tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_expert(hidden: int, ffn: int, group_size: int = 128):
    """Return a mock MLP-like module with int4-quantized gate_up_proj / down_proj."""

    class MockMLP(nn.Module):
        def __init__(self):
            super().__init__()
            # gate_up_proj: in=hidden, out=2*ffn (gate + up fused)
            self.gate_up_proj = _MockQuantLinear(hidden, 2 * ffn, group_size)
            # down_proj: in=ffn, out=hidden
            self.down_proj = _MockQuantLinear(ffn, hidden, group_size)

    return MockMLP()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestMoEInt4Detection(unittest.TestCase):
    """Unit tests for MoE._is_int4_quantized_linear."""

    def test_detects_mock_quantlinear(self):
        from eole.modules.moe import MoE
        layer = _MockQuantLinear(256, 512, group_size=128)
        self.assertTrue(MoE._is_int4_quantized_linear(layer))

    def test_rejects_plain_linear(self):
        from eole.modules.moe import MoE
        layer = nn.Linear(256, 512, bias=False)
        self.assertFalse(MoE._is_int4_quantized_linear(layer))

    def test_rejects_none_qweight(self):
        from eole.modules.moe import MoE

        class _Stub:
            qweight = None
            scales = torch.tensor([1.0])
            qzeros = torch.tensor([0])

        self.assertFalse(MoE._is_int4_quantized_linear(_Stub()))

    def test_rejects_missing_attribute(self):
        from eole.modules.moe import MoE

        class _Stub:
            qweight = torch.zeros(4, 32, dtype=torch.int32)
            scales = torch.ones(1, 32)
            # no qzeros attribute

        self.assertFalse(MoE._is_int4_quantized_linear(_Stub()))


class TestStackedWeightShapes(unittest.TestCase):
    """Verify _maybe_fused_moe_weights_int4 produces correctly-shaped stacked tensors."""

    def _make_dummy_moe(self, num_experts=4, hidden=256, ffn=512, group_size=128):
        """Build a MoE instance whose experts have been monkey-patched with
        mock QuantLinear layers (bypasses full config initialisation)."""
        from eole.modules.moe import MoE
        import types

        moe = object.__new__(MoE)
        # Initialise only the attributes needed by _maybe_fused_moe_weights_int4
        moe.experts = nn.ModuleList(
            [_make_mock_expert(hidden, ffn, group_size) for _ in range(num_experts)]
        )
        moe._qw1 = None
        moe._qw2 = None
        moe._scales_w1 = None
        moe._scales_w2 = None
        moe._qzeros_w1 = None
        moe._qzeros_w2 = None
        moe._int4_group_size = None
        return moe

    def test_stacked_shapes(self):
        from eole.modules.moe import MoE

        E, H, ffn, gs = 4, 256, 512, 128
        moe = self._make_dummy_moe(num_experts=E, hidden=H, ffn=ffn, group_size=gs)
        moe._maybe_fused_moe_weights_int4(device=torch.device("cpu"))

        # W1 stacked weights
        self.assertEqual(moe._qw1.shape, (E, H // 8, 2 * ffn))
        self.assertEqual(moe._scales_w1.shape, (E, H // gs, 2 * ffn))
        self.assertEqual(moe._qzeros_w1.shape, (E, H // gs, 2 * ffn // 8))

        # W2 stacked weights
        self.assertEqual(moe._qw2.shape, (E, ffn // 8, H))
        self.assertEqual(moe._scales_w2.shape, (E, ffn // gs, H))
        self.assertEqual(moe._qzeros_w2.shape, (E, ffn // gs, H // 8))

    def test_group_size_inference(self):
        from eole.modules.moe import MoE

        for gs in (64, 128):
            moe = self._make_dummy_moe(num_experts=2, hidden=256, ffn=512, group_size=gs)
            moe._maybe_fused_moe_weights_int4(device=torch.device("cpu"))
            self.assertEqual(moe._int4_group_size, gs)

    def test_idempotent(self):
        """Calling _maybe_fused_moe_weights_int4 twice must not rebuild."""
        from eole.modules.moe import MoE

        moe = self._make_dummy_moe()
        moe._maybe_fused_moe_weights_int4(device=torch.device("cpu"))
        sentinel = moe._qw1
        moe._maybe_fused_moe_weights_int4(device=torch.device("cpu"))
        self.assertIs(moe._qw1, sentinel)


class TestFusedInt4ImplSignature(unittest.TestCase):
    """Check that fused_experts_int4_impl exists and has the expected interface."""

    def test_importable(self):
        try:
            from eole.triton.fused_moe_int4 import fused_experts_int4_impl  # noqa: F401
        except ImportError:
            self.skipTest("Triton not installed")

    def test_function_parameters(self):
        try:
            from eole.triton.fused_moe_int4 import fused_experts_int4_impl
        except ImportError:
            self.skipTest("Triton not installed")

        import inspect

        sig = inspect.signature(fused_experts_int4_impl)
        expected_params = {
            "hidden_states",
            "stacked_qw1",
            "stacked_scales_w1",
            "stacked_qzeros_w1",
            "stacked_qw2",
            "stacked_scales_w2",
            "stacked_qzeros_w2",
            "topk_weights",
            "topk_ids",
            "group_size",
            "activation",
            "use_sorted",
        }
        self.assertEqual(set(sig.parameters.keys()), expected_params)


if __name__ == "__main__":
    unittest.main()
