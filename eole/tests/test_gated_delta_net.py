"""Tests for eole.modules.gated_delta_net fallback behaviour.

When the dedicated ``causal_conv1d`` package is not installed, the module
must fall back to pure-PyTorch paths instead of trying to wrap FLA's
causal_conv1d kernel (which has changed its API across versions and can
crash at runtime with certain installed FLA versions).
"""
import sys
import types
import unittest


def _reload_gdn():
    """Import gated_delta_net with the causal_conv1d package absent so the
    module-level variables are set to None (PyTorch fallback)."""
    sys.modules.pop("causal_conv1d", None)

    # Stub out fla sub-packages the module tries to import
    for name in ["fla", "fla.modules", "fla.modules.convolution",
                 "fla.ops", "fla.ops.gated_delta_rule"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    sys.modules.pop("eole.modules.gated_delta_net", None)

    import importlib
    return importlib.import_module("eole.modules.gated_delta_net")


class TestCausalConv1dFallback(unittest.TestCase):
    """When causal_conv1d package is absent, module-level fns must be None."""

    def test_fn_is_none_without_causal_conv1d_package(self):
        """causal_conv1d_fn is None when the package is not installed."""
        gdn = _reload_gdn()
        self.assertIsNone(
            gdn.causal_conv1d_fn,
            "causal_conv1d_fn should be None when the causal_conv1d package "
            "is absent; forward() will use the PyTorch F.silu(conv1d(...)) path.",
        )

    def test_update_is_none_without_causal_conv1d_package(self):
        """causal_conv1d_update is None; GatedDeltaNet falls back to _torch_causal_conv1d_update."""
        gdn = _reload_gdn()
        self.assertIsNone(
            gdn.causal_conv1d_update,
            "causal_conv1d_update should be None; the layer uses "
            "_torch_causal_conv1d_update instead.",
        )

    def test_torch_fallback_update_is_callable(self):
        """_torch_causal_conv1d_update is always present and callable."""
        gdn = _reload_gdn()
        self.assertTrue(
            callable(gdn._torch_causal_conv1d_update),
            "_torch_causal_conv1d_update must always be importable and callable.",
        )


if __name__ == "__main__":
    unittest.main()
