"""Tests for eole.modules.autoround_linear.

These tests use lightweight mock objects so they run without gptqmodel or
auto_round_extension installed.  They verify:

1. Marlin alignment checks in replace_autoround_linear
   (out_features % 64, in_features % 128).
2. post_init_autoround_linear falls back to Triton/PyTorch when post_init()
   raises an exception, and correctly preserves the GPTQ-format weights.
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

    def __init__(self, bits, group_size, in_features, out_features, bias, should_fail=False):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        self._should_fail = should_fail
        pack_factor = 32 // bits  # e.g. 8 for 4-bit
        num_groups = math.ceil(in_features / group_size)
        self.register_buffer("qweight", torch.zeros(in_features // pack_factor, out_features, dtype=torch.int32))
        self.register_buffer("scales", torch.ones(num_groups, out_features, dtype=torch.float16))
        self.register_buffer("qzeros", torch.zeros(num_groups, out_features // pack_factor, dtype=torch.int32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def post_init(self):
        if self._should_fail:
            raise RuntimeError("fake Marlin post_init failure")
        # Simulate in-place modification of qweight (like gptq_marlin_repack)
        self.qweight.fill_(42)


class _FakeTritonModule(nn.Module):
    """Mimics Triton QuantLinear: no post_init logic, same GPTQ-format buffers."""

    QUANT_TYPE = "tritonv2"
    __module__ = "auto_round_extension.triton.fake_triton"

    def __init__(self, bits, group_size, infeatures, outfeatures, bias, **kwargs):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        pack_factor = 32 // bits
        num_groups = math.ceil(infeatures / group_size)
        self.register_buffer("qweight", torch.zeros(infeatures // pack_factor, outfeatures, dtype=torch.int32))
        self.register_buffer("scales", torch.ones(num_groups, outfeatures, dtype=torch.float16))
        self.register_buffer("qzeros", torch.zeros(num_groups, outfeatures // pack_factor, dtype=torch.int32))
        if bias:
            self.register_buffer("bias", torch.zeros(outfeatures, dtype=torch.float16))
        else:
            self.bias = None

    def post_init(self):
        pass  # no-op, same as real Triton QuantLinear


# ---------------------------------------------------------------------------
# Helper: build a model that contains FakeMarlin leaves
# ---------------------------------------------------------------------------

class _Container(nn.Module):
    def __init__(self):
        super().__init__()


def _make_model_with_marlin(bits=4, group_size=128,
                             in_features=2048, out_features=8192,
                             should_fail=False):
    """Return a two-level container with a Marlin leaf."""
    root = _Container()
    inner = _Container()
    layer = _FakeMarlinModule(
        bits=bits, group_size=group_size,
        in_features=in_features, out_features=out_features,
        bias=False, should_fail=should_fail,
    )
    layer._eole_use_gptq_zp = False
    inner.add_module("proj", layer)
    root.add_module("block", inner)
    return root


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestMarlinAlignmentCheck(unittest.TestCase):
    """replace_autoround_linear must use fallback for non-aligned layers."""

    def _run_replace(self, in_features, out_features, use_marlin_cls, fallback_cls):
        """Call replace_autoround_linear with mocked QuantLinear classes."""
        from eole.modules.autoround_linear import replace_autoround_linear

        linear = nn.Linear(in_features, out_features, bias=False)
        container = nn.Module()
        container.add_module("proj", linear)

        marlin_created = []
        fallback_created = []

        def marlin_factory(**kwargs):
            m = MagicMock()
            m._eole_use_gptq_zp = False
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
        marlin_created, fallback_created = self._run_replace(2048, 8192, True, False)
        self.assertEqual(len(marlin_created), 1)
        self.assertEqual(len(fallback_created), 0)

    def test_out_features_not_divisible_by_64_uses_fallback(self):
        """out_features not divisible by 64 → fallback, not Marlin."""
        marlin_created, fallback_created = self._run_replace(2048, 100, True, False)
        self.assertEqual(len(marlin_created), 0)
        self.assertEqual(len(fallback_created), 1)

    def test_in_features_not_divisible_by_128_uses_fallback(self):
        """in_features not divisible by 128 → fallback."""
        marlin_created, fallback_created = self._run_replace(100, 8192, True, False)
        self.assertEqual(len(marlin_created), 0)
        self.assertEqual(len(fallback_created), 1)


class TestPostInitFallback(unittest.TestCase):
    """post_init_autoround_linear must replace failing Marlin layers gracefully."""

    def test_successful_post_init_repacks_in_place(self):
        """When post_init succeeds, qweight is modified in-place (Marlin format)."""
        from eole.modules.autoround_linear import post_init_autoround_linear

        model = _make_model_with_marlin(should_fail=False)
        # Confirm initial qweight is all zeros
        proj = model.block.proj
        self.assertTrue((proj.qweight == 0).all())

        with patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls") as mock_cls:
            mock_cls.return_value = (_FakeTritonModule, False)
            post_init_autoround_linear(model)

        # After successful post_init, FakeMarlinModule.post_init sets qweight to 42
        self.assertTrue((model.block.proj.qweight == 42).all())
        # Module is still the Marlin type (not replaced)
        self.assertIsInstance(model.block.proj, _FakeMarlinModule)

    def test_failing_post_init_replaced_with_fallback(self):
        """When post_init fails, module is replaced with Triton fallback."""
        from eole.modules.autoround_linear import post_init_autoround_linear

        model = _make_model_with_marlin(should_fail=True)
        proj = model.block.proj
        # Pre-load the scales with distinctive values so we can verify they're copied
        proj.scales.fill_(3.14)
        proj.qweight.fill_(7)

        with patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls") as mock_cls:
            mock_cls.return_value = (_FakeTritonModule, False)
            post_init_autoround_linear(model)

        # Module must have been replaced with the fallback type
        replacement = model.block.proj
        self.assertIsInstance(replacement, _FakeTritonModule)
        # The GPTQ-format qweight (value 7) must be preserved
        self.assertTrue((replacement.qweight == 7).all())
        # The scales (value 3.14, rounded to float16) must be preserved
        expected = torch.tensor(3.14, dtype=torch.float16)
        self.assertTrue(torch.allclose(replacement.scales, expected.expand_as(replacement.scales)))

    def test_failing_post_init_preserves_original_weights_not_repacked(self):
        """The saved weights are the GPTQ-format originals, not any partial Marlin repack."""
        from eole.modules.autoround_linear import post_init_autoround_linear

        class _PartialFailMarlin(_FakeMarlinModule):
            """Modifies qweight then fails (simulates partial Marlin repack)."""
            __module__ = "auto_round_extension.cuda.fake_marlin"

            def post_init(self):
                self.qweight.fill_(99)  # simulate partial modification
                raise RuntimeError("fail after partial modification")

        layer = _PartialFailMarlin(
            bits=4, group_size=128, in_features=128, out_features=128,
            bias=False, should_fail=False,  # _should_fail unused, we override post_init
        )
        layer.qweight.fill_(5)   # original GPTQ value
        layer._eole_use_gptq_zp = False

        container = nn.Module()
        container.add_module("proj", layer)

        with patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls") as mock_cls:
            mock_cls.return_value = (_FakeTritonModule, False)
            post_init_autoround_linear(container)

        # The replacement must have the PRE-post_init value (5), not 99
        self.assertIsInstance(container.proj, _FakeTritonModule)
        self.assertTrue((container.proj.qweight == 5).all())


if __name__ == "__main__":
    unittest.main()
