"""Tests for eole.modules.autoround_linear.

These tests use lightweight mock objects so they run without gptqmodel or
auto_round_extension installed.  They verify:

1. replace_autoround_linear replaces nn.Linear with the chosen QuantLinear.
2. When Marlin raises NotImplementedError for a layer, the PyTorch fallback is used.
3. post_init_autoround_linear calls post_init() on each auto_round module.
4. _get_autoround_quant_linear_cls selects the right backend.
5. _preflight_marlin_import is a no-op when gptqmodel/CUDA are absent.
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

class _FakeMarlinModule(nn.Module):
    """Mimics MarlinQuantLinear: buffers in GPTQ format, post_init repacks them."""

    __module__ = "auto_round_extension.cuda.fake_marlin"

    def __init__(self, bits, group_size, in_features, out_features, bias, **kwargs):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        pack_factor = 32 // bits
        num_groups = math.ceil(in_features / group_size)
        self.register_buffer("qweight", torch.zeros(in_features // pack_factor, out_features, dtype=torch.int32))
        self.register_buffer("scales", torch.ones(num_groups, out_features, dtype=torch.float16))
        self.register_buffer("qzeros", torch.zeros(num_groups, out_features // pack_factor, dtype=torch.int32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def post_init(self):
        # Simulate in-place modification of qweight (like gptq_marlin_repack)
        self.qweight.fill_(42)


class _FakeQuantModule(nn.Module):
    """Mimics a non-Marlin QuantLinear (Triton or PyTorch)."""

    __module__ = "auto_round_extension.fake_quant"

    def __init__(self, bits, group_size, infeatures, outfeatures, bias):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        pack_factor = 32 // bits
        num_groups = math.ceil(infeatures / group_size)
        self.register_buffer("qweight", torch.zeros(infeatures // pack_factor, outfeatures, dtype=torch.int32))
        self.register_buffer("scales", torch.ones(num_groups, outfeatures, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.zeros(outfeatures, dtype=torch.float16))
        else:
            self.bias = None

    def post_init(self):
        self.qweight.fill_(7)


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

    def test_marlin_backend_replaces_linear(self):
        """When Marlin is selected and layer dimensions are supported, it is used."""
        from eole.modules.autoround_linear import replace_autoround_linear

        linear = nn.Linear(2048, 8192, bias=False)
        container = nn.Module()
        container.add_module("proj", linear)

        created = []

        def marlin_factory(**kwargs):
            m = MagicMock()
            created.append(kwargs)
            return m

        marlin_cls = MagicMock(side_effect=marlin_factory)
        with patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls") as mock_get_cls:
            mock_get_cls.return_value = (marlin_cls, True)
            replace_autoround_linear(
                container,
                module_to_convert=["proj"],
                w_bit=4,
                group_size=128,
            )

        self.assertEqual(len(created), 1)
        # Marlin uses in_features/out_features (not infeatures/outfeatures)
        self.assertEqual(created[0]["in_features"], 2048)
        self.assertEqual(created[0]["out_features"], 8192)

    def test_marlin_raises_uses_pytorch_fallback(self):
        """When Marlin raises NotImplementedError (unsupported shape), PyTorch fallback is used."""
        from eole.modules.autoround_linear import replace_autoround_linear

        linear = nn.Linear(2048, 100, bias=False)  # out_features=100, not Marlin-aligned
        container = nn.Module()
        container.add_module("proj", linear)

        fallback_created = []

        def marlin_factory(**kwargs):
            raise NotImplementedError("out_features not divisible by 64")

        def fallback_factory(**kwargs):
            m = MagicMock()
            fallback_created.append(kwargs)
            return m

        marlin_cls = MagicMock(side_effect=marlin_factory)
        fb_cls = MagicMock(side_effect=fallback_factory)

        with patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls") as mock_get_cls:
            mock_get_cls.side_effect = [
                (marlin_cls, True),   # primary call -> Marlin
                (fb_cls, False),      # fallback call -> PyTorch (force_pytorch=True)
            ]
            replace_autoround_linear(
                container,
                module_to_convert=["proj"],
                w_bit=4,
                group_size=128,
            )

        self.assertEqual(len(fallback_created), 1)
        self.assertEqual(fallback_created[0]["infeatures"], 2048)

    def test_marlin_fallback_uses_force_pytorch(self):
        """The fallback for Marlin-rejected layers calls _get_autoround_quant_linear_cls
        with force_pytorch=True."""
        from eole.modules.autoround_linear import replace_autoround_linear

        linear = nn.Linear(2048, 32, bias=False)
        container = nn.Module()
        container.add_module("proj", linear)

        def marlin_factory(**kwargs):
            raise NotImplementedError("unsupported shape")

        marlin_cls = MagicMock(side_effect=marlin_factory)
        fb_cls = MagicMock(return_value=MagicMock())

        with patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls") as mock_get_cls:
            mock_get_cls.side_effect = [
                (marlin_cls, True),
                (fb_cls, False),
            ]
            replace_autoround_linear(
                container,
                module_to_convert=["proj"],
                w_bit=4,
                group_size=128,
            )

        self.assertEqual(mock_get_cls.call_count, 2)
        _primary_call, fallback_call = mock_get_cls.call_args_list
        self.assertTrue(
            fallback_call.kwargs.get("force_pytorch") is True,
            "fallback must pass force_pytorch=True",
        )

    def test_non_marlin_backend_replaces_linear(self):
        """When a non-Marlin backend (Triton/PyTorch) is selected, it is used directly."""
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
            mock_get_cls.return_value = (quant_cls, False)
            replace_autoround_linear(
                container,
                module_to_convert=["proj"],
                w_bit=4,
                group_size=128,
            )

        self.assertEqual(len(created), 1)
        # Non-Marlin backends use infeatures/outfeatures
        self.assertEqual(created[0]["infeatures"], 2048)
        self.assertEqual(created[0]["outfeatures"], 8192)

    def test_module_to_not_convert_is_skipped(self):
        """Modules listed in module_to_not_convert are not replaced."""
        from eole.modules.autoround_linear import replace_autoround_linear

        linear = nn.Linear(128, 128, bias=False)
        container = nn.Module()
        container.add_module("proj", linear)

        with patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls") as mock_get_cls:
            quant_cls = MagicMock(return_value=MagicMock())
            mock_get_cls.return_value = (quant_cls, False)
            replace_autoround_linear(
                container,
                module_to_convert=["proj"],
                module_to_not_convert=["proj"],
            )

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

        # _FakeQuantModule.post_init sets qweight to 7
        self.assertTrue((model.block.proj.qweight == 7).all())

    def test_non_auto_round_modules_are_skipped(self):
        """Regular nn.Module containers are recursed into but not called."""
        from eole.modules.autoround_linear import post_init_autoround_linear

        model = _Container()
        model.add_module("linear", nn.Linear(16, 16))
        post_init_autoround_linear(model)  # must not raise


class TestGetAutoRoundQuantLinearCls(unittest.TestCase):
    """_get_autoround_quant_linear_cls selects Marlin > Triton > PyTorch."""

    def test_marlin_selected_when_cuda_and_sym(self):
        """Marlin is selected when CUDA is available and sym=True."""
        from eole.modules.autoround_linear import _get_autoround_quant_linear_cls

        mock_cls = MagicMock()
        fake_mod = MagicMock()
        fake_mod.get_marlin_layer = MagicMock(return_value=mock_cls)
        with patch("eole.modules.autoround_linear.cuda_is_available", return_value=True), \
             patch.dict(sys.modules, {
                 "auto_round_extension.cuda.gptqmodel_marlin": fake_mod,
             }):
            cls, use_marlin = _get_autoround_quant_linear_cls(use_gptq_zp=False, sym=True)

        self.assertIs(cls, mock_cls)
        self.assertTrue(use_marlin)

    def test_triton_selected_when_cuda_but_no_marlin(self):
        """Triton is used when Marlin/gptqmodel is not available but CUDA is."""
        from eole.modules.autoround_linear import _get_autoround_quant_linear_cls

        mock_quant = MagicMock()
        fake_mod = MagicMock()
        fake_mod.QuantLinear = mock_quant

        marlin_key = "auto_round_extension.cuda.gptqmodel_marlin"
        saved = sys.modules.pop(marlin_key, None)
        try:
            with patch("eole.modules.autoround_linear.cuda_is_available", return_value=True), \
                 patch.dict(sys.modules, {
                     "auto_round_extension.triton.qlinear_tritonv2": fake_mod,
                 }):
                cls, use_marlin = _get_autoround_quant_linear_cls(use_gptq_zp=False, sym=True)
        finally:
            if saved is not None:
                sys.modules[marlin_key] = saved

        self.assertIs(cls, mock_quant)
        self.assertFalse(use_marlin)

    def test_pytorch_fallback_when_no_cuda(self):
        """PyTorch backend is returned when CUDA is not available."""
        from eole.modules.autoround_linear import _get_autoround_quant_linear_cls

        mock_quant = MagicMock()
        fake_mod = MagicMock()
        fake_mod.QuantLinear = mock_quant
        with patch("eole.modules.autoround_linear.cuda_is_available", return_value=False), \
             patch.dict(sys.modules, {"auto_round_extension.torch.qlinear_torch": fake_mod}):
            cls, use_marlin = _get_autoround_quant_linear_cls(use_gptq_zp=False)

        self.assertIs(cls, mock_quant)
        self.assertFalse(use_marlin)

    def test_raises_when_nothing_available(self):
        """ImportError is raised when no backend can be imported."""
        from eole.modules.autoround_linear import _get_autoround_quant_linear_cls

        keys = [
            "auto_round_extension.cuda.gptqmodel_marlin",
            "auto_round_extension.triton.qlinear_tritonv2",
            "auto_round_extension.torch.qlinear_torch",
        ]
        saved = {k: sys.modules.pop(k, None) for k in keys}
        try:
            with patch("eole.modules.autoround_linear.cuda_is_available", return_value=False):
                with self.assertRaises(ImportError):
                    _get_autoround_quant_linear_cls(use_gptq_zp=False)
        finally:
            for k, v in saved.items():
                if v is not None:
                    sys.modules[k] = v


class TestPreflightMarlinImport(unittest.TestCase):
    """_preflight_marlin_import triggers gptqmodel import before FLA can create
    Autotuner instances, mirroring vLLM's worker-startup import pattern."""

    def test_noop_when_cuda_unavailable(self):
        """When CUDA is not available, _preflight_marlin_import does nothing."""
        from eole.modules.autoround_linear import _preflight_marlin_import

        with patch("eole.modules.autoround_linear.cuda_is_available", return_value=False):
            _preflight_marlin_import()  # must not raise

    def test_noop_when_import_fails(self):
        """When auto_round_extension.cuda.gptqmodel_marlin is not installed,
        _preflight_marlin_import silently passes."""
        from eole.modules.autoround_linear import _preflight_marlin_import

        marlin_key = "auto_round_extension.cuda.gptqmodel_marlin"
        saved = sys.modules.pop(marlin_key, None)
        try:
            with patch("eole.modules.autoround_linear.cuda_is_available", return_value=True):
                _preflight_marlin_import()  # must not raise
        finally:
            if saved is not None:
                sys.modules[marlin_key] = saved

    def test_triggers_import_when_cuda_available(self):
        """When CUDA is available, _preflight_marlin_import attempts to import
        auto_round_extension.cuda.gptqmodel_marlin (so gptqmodel __init__ runs)."""
        from eole.modules.autoround_linear import _preflight_marlin_import

        fake_mod = MagicMock()
        fake_mod.get_marlin_layer = MagicMock(return_value=MagicMock())
        with patch("eole.modules.autoround_linear.cuda_is_available", return_value=True), \
             patch.dict(sys.modules, {"auto_round_extension.cuda.gptqmodel_marlin": fake_mod}):
            _preflight_marlin_import()
            self.assertIn("auto_round_extension.cuda.gptqmodel_marlin", sys.modules)


if __name__ == "__main__":
    unittest.main()
