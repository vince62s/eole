"""Tests for eole.modules.autoround_linear.

These tests use lightweight mock objects so they run without gptqmodel or
auto_round_extension installed.  They verify:

1. replace_autoround_linear replaces nn.Linear with the chosen QuantLinear.
2. post_init_autoround_linear calls post_init() on each auto_round module.
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

class TestReplaceAutoroundLinear(unittest.TestCase):
    """replace_autoround_linear replaces nn.Linear with the chosen backend."""

    def _run_replace(self, in_features, out_features, use_marlin=True):
        """Call replace_autoround_linear with a mocked QuantLinear class."""
        from eole.modules.autoround_linear import replace_autoround_linear

        linear = nn.Linear(in_features, out_features, bias=False)
        container = nn.Module()
        container.add_module("proj", linear)

        created = []

        def factory(**kwargs):
            m = MagicMock()
            created.append(kwargs)
            return m

        quant_cls = MagicMock(side_effect=factory)

        with patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls") as mock_get_cls:
            mock_get_cls.return_value = (quant_cls, use_marlin)
            replace_autoround_linear(
                container,
                module_to_convert=["proj"],
                w_bit=4,
                group_size=128,
                packing_format="auto_round:auto_gptq",
                sym=True,
            )

        return created

    def test_marlin_backend_replaces_linear(self):
        """When Marlin is selected, the layer is replaced with Marlin QuantLinear."""
        created = self._run_replace(2048, 8192, use_marlin=True)
        self.assertEqual(len(created), 1)
        self.assertIn("in_features", created[0])
        self.assertEqual(created[0]["in_features"], 2048)

    def test_non_marlin_backend_replaces_linear(self):
        """When a non-Marlin backend is selected, the layer is also replaced."""
        created = self._run_replace(2048, 8192, use_marlin=False)
        self.assertEqual(len(created), 1)
        self.assertIn("infeatures", created[0])
        self.assertEqual(created[0]["infeatures"], 2048)

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

        # proj was in both lists — to_not_convert takes precedence at the parent level,
        # but since the container itself is the parent we pass module_to_not_convert at
        # that parent level; verify the original nn.Linear is still in place.
        self.assertIsInstance(container.proj, nn.Linear)


class TestPostInitAutoround(unittest.TestCase):
    """post_init_autoround_linear must call post_init() on auto_round modules."""

    def test_post_init_called_on_auto_round_module(self):
        """post_init() is called and qweight is modified in-place."""
        from eole.modules.autoround_linear import post_init_autoround_linear

        model = _make_model_with_marlin()
        proj = model.block.proj
        self.assertTrue((proj.qweight == 0).all())

        post_init_autoround_linear(model)

        # FakeMarlinModule.post_init sets qweight to 42
        self.assertTrue((model.block.proj.qweight == 42).all())
        self.assertIsInstance(model.block.proj, _FakeMarlinModule)

    def test_non_auto_round_modules_are_skipped(self):
        """Regular nn.Module containers are recursed into but not called."""
        from eole.modules.autoround_linear import post_init_autoround_linear

        # A plain container with no auto_round modules — should not raise
        model = _Container()
        model.add_module("linear", nn.Linear(16, 16))
        post_init_autoround_linear(model)  # must not raise


if __name__ == "__main__":
    unittest.main()

