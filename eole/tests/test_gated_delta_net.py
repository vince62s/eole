"""Tests for the FLA causal_conv1d wrapper in eole.modules.gated_delta_net.

These tests verify that the FLA compatibility layer correctly handles both:
  - Old FLA (pre-Nov 2024): head-first x[B, D, L], no `residual` parameter
  - New FLA (post-Nov 2024): seq-first x[B, T, D], with `residual` parameter

They use lightweight mock objects so they run without causal_conv1d or
fla-core installed.
"""
import sys
import types
import unittest

import torch


# ---------------------------------------------------------------------------
# Helpers: build minimal FLA mock modules
# ---------------------------------------------------------------------------

def _make_fla_mock(seq_first: bool):
    """Return a minimal mock of fla.modules.convolution for testing.

    seq_first=True  → new FLA API (post-Nov 2024): `residual` parameter present,
                       x expected in [B, T, D] seq-first format.
    seq_first=False → old FLA API (pre-Nov 2024): no `residual` parameter,
                       x expected in [B, D, L] head-first format.
    """
    if seq_first:
        def causal_conv1d(x, weight=None, bias=None, residual=None,
                          initial_state=None, output_final_state=False,
                          activation=None, **kwargs):
            # new FLA: receives [B, T, D] seq-first; return same shape
            return x.clone(), None

        def causal_conv1d_update(x, cache, residual=None, weight=None,
                                  bias=None, activation=None):
            return x.clone(), cache
    else:
        def causal_conv1d(x, weight=None, bias=None, activation=None,
                          initial_states=None, output_final_state=False):
            # old FLA: receives [B, D, L] head-first; return same shape
            return x.clone(), None

        def causal_conv1d_update(x, cache, weight=None, bias=None,
                                  activation=None):
            return x.clone(), cache

    mod = types.ModuleType("fla.modules.convolution")
    mod.causal_conv1d = causal_conv1d
    mod.causal_conv1d_update = causal_conv1d_update
    return mod


def _reload_gdn(fla_mock):
    """(Re-)import gated_delta_net with a given fla.modules.convolution mock.

    Injects the mock so the module-level import picks it up.  Returns the
    freshly loaded module object so tests can inspect module-level state such
    as ``_fla_seq_first`` and patch ``_fla_causal_conv1d`` directly.
    """
    # Ensure causal_conv1d package is absent so the FLA path is taken
    sys.modules.pop("causal_conv1d", None)

    # Inject the mock at the right path
    sys.modules["fla"] = types.ModuleType("fla")
    sys.modules["fla.modules"] = types.ModuleType("fla.modules")
    sys.modules["fla.modules.convolution"] = fla_mock

    # Stub out other fla sub-packages the module tries to import
    for name in ["fla.ops", "fla.ops.gated_delta_rule"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # Force re-execution of the module
    sys.modules.pop("eole.modules.gated_delta_net", None)

    import importlib
    return importlib.import_module("eole.modules.gated_delta_net")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestFLACausalConv1dWrapper(unittest.TestCase):
    """The causal_conv1d_fn wrapper must adapt [B,D,L] ↔ FLA's native format."""

    def _test_fn_format(self, seq_first: bool):
        """Verify the wrapper calls FLA with the correct tensor layout."""
        fla_mock = _make_fla_mock(seq_first)
        gdn = _reload_gdn(fla_mock)

        # Verify detection worked
        self.assertEqual(gdn._fla_seq_first, seq_first,
                         "Module-level _fla_seq_first not detected correctly")

        # Spy on _fla_causal_conv1d AFTER the module is loaded
        received_shapes = []
        original_fn = gdn._fla_causal_conv1d

        def spy(x, **kwargs):
            received_shapes.append(x.shape)
            return original_fn(x, **kwargs)

        gdn._fla_causal_conv1d = spy
        try:
            B, D, L = 2, 16, 8
            x = torch.randn(B, D, L)
            weight = torch.randn(D, 4)

            out = gdn.causal_conv1d_fn(x, weight=weight, bias=None, activation=None)

            self.assertEqual(len(received_shapes), 1, "FLA fn should be called once")
            if seq_first:
                # New FLA: wrapper must transpose [B, D, L] → [B, L, D] before calling
                self.assertEqual(received_shapes[0], torch.Size([B, L, D]),
                                 "New FLA: expected [B, L, D] (seq-first)")
            else:
                # Old FLA: wrapper must pass [B, D, L] directly (no transpose)
                self.assertEqual(received_shapes[0], torch.Size([B, D, L]),
                                 "Old FLA: expected [B, D, L] (head-first)")

            # In both cases the output must be [B, D, L]
            self.assertEqual(out.shape, torch.Size([B, D, L]))
        finally:
            gdn._fla_causal_conv1d = original_fn

    def test_new_fla_seq_first(self):
        """New FLA (seq-first): wrapper transposes x to [B, T, D] before calling FLA."""
        self._test_fn_format(seq_first=True)

    def test_old_fla_head_first(self):
        """Old FLA (head-first): wrapper passes x directly in [B, D, L]."""
        self._test_fn_format(seq_first=False)

    def _test_update_signature(self, seq_first: bool):
        """Verify causal_conv1d_update calls FLA with the correct kwargs."""
        fla_mock = _make_fla_mock(seq_first)
        gdn = _reload_gdn(fla_mock)

        self.assertEqual(gdn._fla_update_has_residual, seq_first,
                         "Module-level _fla_update_has_residual not detected correctly")

        received_kwargs = []
        original_fn = gdn._fla_causal_conv1d_update

        def spy(x, cache, **kwargs):
            received_kwargs.append(dict(kwargs))
            return original_fn(x, cache, **kwargs)

        gdn._fla_causal_conv1d_update = spy
        try:
            B, D = 2, 16
            x = torch.randn(B, D, 1)
            state = torch.randn(B, D, 4)
            weight = torch.randn(D, 4)

            out = gdn.causal_conv1d_update(x, state, weight=weight, bias=None)

            self.assertEqual(len(received_kwargs), 1)
            kw = received_kwargs[0]

            if seq_first:
                # New API: 'residual' keyword must be passed (as None)
                self.assertIn('residual', kw,
                              "New FLA update must receive residual=None")
                self.assertIsNone(kw['residual'])
            else:
                # Old API: 'residual' must NOT be passed
                self.assertNotIn('residual', kw,
                                 "Old FLA update must NOT receive residual kwarg")

            # Output shape must be restored to [B, D, 1]
            self.assertEqual(out.shape, torch.Size([B, D, 1]))
        finally:
            gdn._fla_causal_conv1d_update = original_fn

    def test_update_new_fla(self):
        """New FLA update: passes residual=None keyword."""
        self._test_update_signature(seq_first=True)

    def test_update_old_fla(self):
        """Old FLA update: does NOT pass residual keyword."""
        self._test_update_signature(seq_first=False)


if __name__ == "__main__":
    unittest.main()
