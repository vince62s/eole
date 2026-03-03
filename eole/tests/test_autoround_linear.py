"""Tests for eole.modules.autoround_linear.

These tests use lightweight mock objects so they run without gptqmodel or
auto_round_extension installed.  They verify:

1. Marlin alignment checks in replace_autoround_linear
   (out_features % 64, in_features % 128).
2. _check_marlin_shape_consistency detects the specific condition that
   triggers gptq_marlin_repack's "Shape mismatch" CUDA assertion:
   qweight.size(0) != in_features // pack_factor.
3. post_init_autoround_linear calls post_init() on each auto_round module
   and propagates exceptions (including the clear ValueError from step 2).
"""
import math
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal stubs for auto_round_extension modules
# ---------------------------------------------------------------------------

class _FakeMarlinModule(nn.Module):
    """Mimics MarlinQuantLinear: buffers in GPTQ format, post_init repacks them."""

    QUANT_TYPE = "marlin"
    __module__ = "auto_round_extension.cuda.fake_marlin"

    def __init__(self, bits, group_size, in_features, out_features, bias):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        self.pack_factor = 32 // bits  # e.g. 8 for 4-bit
        num_groups = math.ceil(in_features / group_size)
        self.register_buffer("qweight", torch.zeros(in_features // self.pack_factor, out_features, dtype=torch.int32))
        self.register_buffer("scales", torch.ones(num_groups, out_features, dtype=torch.float16))
        self.register_buffer("qzeros", torch.zeros(num_groups, out_features // self.pack_factor, dtype=torch.int32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def post_init(self):
        # Simulate in-place modification of qweight (like gptq_marlin_repack)
        self.qweight.fill_(42)


# ---------------------------------------------------------------------------
# Helper: build a model that contains FakeMarlin leaves
# ---------------------------------------------------------------------------

class _Container(nn.Module):
    def __init__(self):
        super().__init__()


def _make_model_with_marlin(bits=4, group_size=128,
                             in_features=2048, out_features=8192):
    """Return a two-level container with a Marlin leaf."""
    root = _Container()
    inner = _Container()
    layer = _FakeMarlinModule(
        bits=bits, group_size=group_size,
        in_features=in_features, out_features=out_features,
        bias=False,
    )
    inner.add_module("proj", layer)
    root.add_module("block", inner)
    return root


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestMarlinAlignmentCheck(unittest.TestCase):
    """replace_autoround_linear must use fallback for non-aligned layers."""

    def _run_replace(self, in_features, out_features):
        """Call replace_autoround_linear with mocked QuantLinear classes."""
        from eole.modules.autoround_linear import replace_autoround_linear

        linear = nn.Linear(in_features, out_features, bias=False)
        container = nn.Module()
        container.add_module("proj", linear)

        marlin_created = []
        fallback_created = []

        def marlin_factory(**kwargs):
            m = MagicMock()
            marlin_created.append(kwargs)
            return m

        def fallback_factory(**kwargs):
            f = MagicMock()
            fallback_created.append(kwargs)
            return f

        marlin_cls = MagicMock(side_effect=marlin_factory)
        fb_cls = MagicMock(side_effect=fallback_factory)

        with (
            patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls") as mock_get_cls,
        ):
            # First call returns Marlin, second returns fallback
            mock_get_cls.side_effect = [
                (marlin_cls, True),  # Marlin selection
                (fb_cls, False),     # fallback selection
            ]
            replace_autoround_linear(
                container,
                module_to_convert=["proj"],
                w_bit=4,
                group_size=128,
                packing_format="auto_round:auto_gptq",
                sym=True,
            )

        return marlin_created, fallback_created

    def test_aligned_layer_uses_marlin(self):
        """A layer with out_features % 64 == 0 and in_features % 128 == 0 → Marlin."""
        marlin_created, fallback_created = self._run_replace(2048, 8192)
        self.assertEqual(len(marlin_created), 1)
        self.assertEqual(len(fallback_created), 0)

    def test_out_features_not_divisible_by_64_uses_fallback(self):
        """out_features not divisible by 64 → fallback, not Marlin."""
        marlin_created, fallback_created = self._run_replace(2048, 100)
        self.assertEqual(len(marlin_created), 0)
        self.assertEqual(len(fallback_created), 1)

    def test_in_features_not_divisible_by_128_uses_fallback(self):
        """in_features not divisible by 128 → fallback."""
        marlin_created, fallback_created = self._run_replace(100, 8192)
        self.assertEqual(len(marlin_created), 0)
        self.assertEqual(len(fallback_created), 1)


class TestCheckMarlinShapeConsistency(unittest.TestCase):
    """_check_marlin_shape_consistency detects the root cause of Marlin 'Shape mismatch'.

    The gptq_marlin_repack CUDA kernel raises:
        "Shape mismatch: b_q_weight.size(0) = X, size_k = Y, pack_factor = Z"
    when qweight.size(0) != in_features // pack_factor.  This test verifies that
    _check_marlin_shape_consistency detects that condition before the kernel fires.
    """

    def _make_module(self, in_features, out_features, bits=4, qweight_rows=None):
        """Create a fake Marlin-like module with configurable qweight shape."""
        mod = nn.Module()
        pack_factor = 32 // bits
        rows = qweight_rows if qweight_rows is not None else in_features // pack_factor
        mod.register_buffer("qweight", torch.zeros(rows, out_features, dtype=torch.int32))
        mod.in_features = in_features
        mod.out_features = out_features
        mod.pack_factor = pack_factor
        return mod

    def test_consistent_shape_no_error(self):
        """No exception when qweight.size(0) == in_features // pack_factor."""
        from eole.modules.autoround_linear import _check_marlin_shape_consistency
        mod = self._make_module(2048, 8192)
        # Should not raise
        _check_marlin_shape_consistency("proj", mod)

    def test_mismatch_raises_value_error(self):
        """ValueError raised when qweight.size(0) != in_features // pack_factor."""
        from eole.modules.autoround_linear import _check_marlin_shape_consistency
        # Model config says in_features=4096 (pack_factor=8 → expected rows=512)
        # but the checkpoint has qweight with only 256 rows (implying in_features=2048)
        mod = self._make_module(in_features=4096, out_features=8192, qweight_rows=256)
        with self.assertRaises(ValueError) as ctx:
            _check_marlin_shape_consistency("in_proj_qkv", mod)
        msg = str(ctx.exception)
        self.assertIn("Marlin shape mismatch", msg)
        self.assertIn("in_proj_qkv", msg)
        self.assertIn("in_features=4096", msg)
        self.assertIn("in_features=2048", msg)  # checkpoint-implied value

    def test_mismatch_error_mentions_config_fields(self):
        """Error message hints at which config fields to check."""
        from eole.modules.autoround_linear import _check_marlin_shape_consistency
        mod = self._make_module(in_features=4096, out_features=8192, qweight_rows=256)
        with self.assertRaises(ValueError) as ctx:
            _check_marlin_shape_consistency("in_proj_qkv", mod)
        msg = str(ctx.exception)
        self.assertIn("hidden_size", msg)
        self.assertIn("linear_num_key_heads", msg)

    def test_missing_attributes_no_error(self):
        """Modules without qweight/pack_factor/in_features are silently skipped."""
        from eole.modules.autoround_linear import _check_marlin_shape_consistency
        mod = nn.Module()  # no qweight, no pack_factor, no in_features
        _check_marlin_shape_consistency("layer", mod)  # must not raise


class TestPostInitAutoround(unittest.TestCase):
    """post_init_autoround_linear must call post_init() and propagate exceptions."""

    def test_successful_post_init_repacks_in_place(self):
        """When post_init succeeds, qweight is modified in-place (Marlin format)."""
        from eole.modules.autoround_linear import post_init_autoround_linear

        model = _make_model_with_marlin()
        proj = model.block.proj
        self.assertTrue((proj.qweight == 0).all())

        post_init_autoround_linear(model)

        # After successful post_init, FakeMarlinModule.post_init sets qweight to 42
        self.assertTrue((model.block.proj.qweight == 42).all())
        # Module is still the Marlin type (not replaced)
        self.assertIsInstance(model.block.proj, _FakeMarlinModule)

    def test_shape_mismatch_raises_clear_error(self):
        """A shape mismatch between in_features and qweight raises a clear ValueError."""
        from eole.modules.autoround_linear import post_init_autoround_linear

        model = _make_model_with_marlin(in_features=4096, out_features=8192)
        proj = model.block.proj
        # Simulate a checkpoint mismatch: replace qweight with fewer rows
        # (as if the checkpoint was quantized from a model with in_features=2048)
        proj.register_buffer("qweight", torch.zeros(256, 8192, dtype=torch.int32))

        with self.assertRaises(ValueError) as ctx:
            post_init_autoround_linear(model)
        self.assertIn("Marlin shape mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
