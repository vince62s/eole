"""Tests for eole.modules.autoround_linear.

These tests use lightweight mock objects so they run without gptqmodel or
auto_round_extension installed.  They verify:

1. replace_autoround_linear replaces nn.Linear with the chosen QuantLinear.
2. post_init_autoround_linear calls post_init() on each auto_round module.
3. _get_autoround_quant_linear_cls selects the right backend.
"""
import math
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal stubs for auto_round_extension modules
# ---------------------------------------------------------------------------

class _FakeQuantModule(nn.Module):
    """Mimics a QuantLinear: buffers in packed format, post_init repacks them."""

    __module__ = "auto_round_extension.fake_quant"

    def __init__(self, bits, group_size, infeatures, outfeatures, bias):
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
        # Simulate in-place modification of qweight (like a repack)
        self.qweight.fill_(42)


# ---------------------------------------------------------------------------
# Helper: build a model that contains _FakeQuantModule leaves
# ---------------------------------------------------------------------------

class _Container(nn.Module):
    def __init__(self):
        super().__init__()


def _make_model_with_quant(bits=4, group_size=128, infeatures=2048, outfeatures=8192):
    """Return a two-level container with a quantized leaf."""
    root = _Container()
    inner = _Container()
    layer = _FakeQuantModule(
        bits=bits, group_size=group_size,
        infeatures=infeatures, outfeatures=outfeatures,
        bias=False,
    )
    inner.add_module("proj", layer)
    root.add_module("block", inner)
    return root


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestReplaceAutoroundLinear(unittest.TestCase):
    """replace_autoround_linear replaces nn.Linear with the chosen backend."""

    def test_backend_replaces_linear(self):
        """An nn.Linear in module_to_convert is replaced with a QuantLinear."""
        from eole.modules.autoround_linear import replace_autoround_linear

        linear = nn.Linear(2048, 8192, bias=False)
        container = nn.Module()
        container.add_module("proj", linear)

        created = []

        def factory(**kwargs):
            m = MagicMock()
            created.append(kwargs)
            return m

        quant_cls = MagicMock(side_effect=factory)
        with patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls") as mock_get_cls:
            mock_get_cls.return_value = quant_cls
            replace_autoround_linear(
                container,
                module_to_convert=["proj"],
                w_bit=4,
                group_size=128,
            )

        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["infeatures"], 2048)
        self.assertEqual(created[0]["outfeatures"], 8192)
        self.assertEqual(created[0]["bits"], 4)
        self.assertEqual(created[0]["group_size"], 128)

    def test_module_to_not_convert_is_skipped(self):
        """Modules listed in module_to_not_convert are not replaced."""
        from eole.modules.autoround_linear import replace_autoround_linear

        linear = nn.Linear(128, 128, bias=False)
        container = nn.Module()
        container.add_module("proj", linear)

        with patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls") as mock_get_cls:
            quant_cls = MagicMock(return_value=MagicMock())
            mock_get_cls.return_value = quant_cls
            replace_autoround_linear(
                container,
                module_to_convert=["proj"],
                module_to_not_convert=["proj"],
            )

        # proj was in both lists — to_not_convert takes precedence at the parent level
        self.assertIsInstance(container.proj, nn.Linear)


class TestPostInitAutoround(unittest.TestCase):
    """post_init_autoround_linear must call post_init() on auto_round modules."""

    def test_post_init_called_on_auto_round_module(self):
        """post_init() is called and qweight is modified in-place."""
        from eole.modules.autoround_linear import post_init_autoround_linear

        model = _make_model_with_quant()
        proj = model.block.proj
        self.assertTrue((proj.qweight == 0).all())

        post_init_autoround_linear(model)

        # _FakeQuantModule.post_init sets qweight to 42
        self.assertTrue((model.block.proj.qweight == 42).all())
        self.assertIsInstance(model.block.proj, _FakeQuantModule)

    def test_non_auto_round_modules_are_skipped(self):
        """Regular nn.Module containers are recursed into but not called."""
        from eole.modules.autoround_linear import post_init_autoround_linear

        # A plain container with no auto_round modules — should not raise
        model = _Container()
        model.add_module("linear", nn.Linear(16, 16))
        post_init_autoround_linear(model)  # must not raise


class TestGetAutoRoundQuantLinearCls(unittest.TestCase):
    """_get_autoround_quant_linear_cls selects Triton on CUDA, PyTorch otherwise."""

    def test_triton_selected_when_cuda_available(self):
        """Triton QuantLinear is returned when CUDA is available and triton imports."""
        from eole.modules.autoround_linear import _get_autoround_quant_linear_cls

        mock_quant = MagicMock()
        fake_mod = MagicMock()
        fake_mod.QuantLinear = mock_quant
        with patch("eole.modules.autoround_linear.cuda_is_available", return_value=True), \
             patch.dict(sys.modules, {"auto_round_extension.triton.qlinear_tritonv2": fake_mod}):
            result = _get_autoround_quant_linear_cls(use_gptq_zp=False)

        self.assertIs(result, mock_quant)

    def test_triton_zp_variant_selected_for_gptq_packing(self):
        """The _zp Triton variant is used when use_gptq_zp=True."""
        from eole.modules.autoround_linear import _get_autoround_quant_linear_cls

        mock_quant = MagicMock()
        fake_mod = MagicMock()
        fake_mod.QuantLinear = mock_quant
        with patch("eole.modules.autoround_linear.cuda_is_available", return_value=True), \
             patch.dict(sys.modules, {"auto_round_extension.triton.qlinear_tritonv2_zp": fake_mod}):
            result = _get_autoround_quant_linear_cls(use_gptq_zp=True)

        self.assertIs(result, mock_quant)

    def test_pytorch_fallback_when_no_cuda(self):
        """PyTorch backend is returned when CUDA is not available."""
        from eole.modules.autoround_linear import _get_autoround_quant_linear_cls

        mock_quant = MagicMock()
        fake_mod = MagicMock()
        fake_mod.QuantLinear = mock_quant
        with patch("eole.modules.autoround_linear.cuda_is_available", return_value=False), \
             patch.dict(sys.modules, {"auto_round_extension.torch.qlinear_torch": fake_mod}):
            result = _get_autoround_quant_linear_cls(use_gptq_zp=False)

        self.assertIs(result, mock_quant)

    def test_pytorch_fallback_when_triton_import_fails(self):
        """PyTorch backend is used when Triton is unavailable even with CUDA."""
        from eole.modules.autoround_linear import _get_autoround_quant_linear_cls

        mock_quant = MagicMock()
        fake_mod = MagicMock()
        fake_mod.QuantLinear = mock_quant

        # Remove triton variant from sys.modules so the import falls through
        triton_key = "auto_round_extension.triton.qlinear_tritonv2"
        saved = sys.modules.pop(triton_key, None)
        try:
            with patch("eole.modules.autoround_linear.cuda_is_available", return_value=True), \
                 patch.dict(sys.modules, {"auto_round_extension.torch.qlinear_torch": fake_mod}):
                result = _get_autoround_quant_linear_cls(use_gptq_zp=False)
        finally:
            if saved is not None:
                sys.modules[triton_key] = saved

        self.assertIs(result, mock_quant)

    def test_raises_when_nothing_available(self):
        """ImportError is raised when neither Triton nor PyTorch backend can be imported."""
        from eole.modules.autoround_linear import _get_autoround_quant_linear_cls

        triton_key = "auto_round_extension.triton.qlinear_tritonv2"
        torch_key = "auto_round_extension.torch.qlinear_torch"
        saved_triton = sys.modules.pop(triton_key, None)
        saved_torch = sys.modules.pop(torch_key, None)
        try:
            with patch("eole.modules.autoround_linear.cuda_is_available", return_value=False):
                with self.assertRaises(ImportError):
                    _get_autoround_quant_linear_cls(use_gptq_zp=False)
        finally:
            if saved_triton is not None:
                sys.modules[triton_key] = saved_triton
            if saved_torch is not None:
                sys.modules[torch_key] = saved_torch


if __name__ == "__main__":
    unittest.main()
