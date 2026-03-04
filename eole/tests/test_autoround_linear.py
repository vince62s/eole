"""Tests for eole.modules.autoround_linear.

These tests use lightweight mock objects so they run without gptqmodel or
auto_round_extension installed.  They verify:

1. replace_autoround_linear replaces nn.Linear with the chosen QuantLinear.
2. When Marlin raises NotImplementedError for a layer, the non-Marlin fallback is used.
3. post_init_autoround_linear calls post_init() on each auto_round module.
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

    def _run_replace(self, in_features, out_features, use_marlin=True, marlin_raises=False):
        """Call replace_autoround_linear with mocked QuantLinear classes."""
        from eole.modules.autoround_linear import replace_autoround_linear

        linear = nn.Linear(in_features, out_features, bias=False)
        container = nn.Module()
        container.add_module("proj", linear)

        marlin_created = []
        fallback_created = []

        def marlin_factory(**kwargs):
            if marlin_raises:
                raise NotImplementedError("unsupported dimensions")
            m = MagicMock()
            marlin_created.append(kwargs)
            return m

        def fallback_factory(**kwargs):
            f = MagicMock()
            fallback_created.append(kwargs)
            return f

        marlin_cls = MagicMock(side_effect=marlin_factory)
        fb_cls = MagicMock(side_effect=fallback_factory)

        with patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls") as mock_get_cls:
            mock_get_cls.side_effect = [
                (marlin_cls, use_marlin),   # primary call
                (fb_cls, False),             # fallback call (only reached when Marlin raises)
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

    def test_marlin_backend_replaces_linear(self):
        """When Marlin is selected and layer is supported, it is replaced with Marlin QuantLinear."""
        marlin_created, fallback_created = self._run_replace(2048, 8192, use_marlin=True, marlin_raises=False)
        self.assertEqual(len(marlin_created), 1)
        self.assertEqual(len(fallback_created), 0)
        self.assertIn("in_features", marlin_created[0])
        self.assertEqual(marlin_created[0]["in_features"], 2048)

    def test_marlin_raises_uses_fallback(self):
        """When Marlin raises NotImplementedError (e.g. out_features % 64 != 0),
        the pure-PyTorch fallback is used (not Triton, which has the same shape
        constraints and would also crash at runtime)."""
        marlin_created, fallback_created = self._run_replace(2048, 100, use_marlin=True, marlin_raises=True)
        self.assertEqual(len(marlin_created), 0)
        self.assertEqual(len(fallback_created), 1)
        self.assertIn("infeatures", fallback_created[0])
        self.assertEqual(fallback_created[0]["infeatures"], 2048)

    def test_marlin_fallback_uses_force_pytorch(self):
        """Fallback for Marlin-rejected layers must call _get_autoround_quant_linear_cls
        with force_pytorch=True to bypass Triton (which has the same shape constraints).
        """
        from eole.modules.autoround_linear import replace_autoround_linear

        linear = nn.Linear(2048, 32, bias=False)  # out_features=32, not divisible by 64
        container = nn.Module()
        container.add_module("proj", linear)

        def marlin_factory(**kwargs):
            raise NotImplementedError("out_features not divisible by 64")

        marlin_cls = MagicMock(side_effect=marlin_factory)
        fb_cls = MagicMock(return_value=MagicMock())

        with patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls") as mock_get_cls:
            mock_get_cls.side_effect = [
                (marlin_cls, True),   # primary call
                (fb_cls, False),      # fallback call — must use force_pytorch=True
            ]
            replace_autoround_linear(
                container,
                module_to_convert=["proj"],
                w_bit=4,
                group_size=128,
                packing_format="auto_round:auto_gptq",
                sym=True,
            )

        # Verify the fallback call used force_pytorch=True to skip Triton
        self.assertEqual(mock_get_cls.call_count, 2)
        _primary_call, fallback_call = mock_get_cls.call_args_list
        self.assertTrue(
            fallback_call.kwargs.get("force_pytorch") is True,
            "fallback must pass force_pytorch=True to _get_autoround_quant_linear_cls",
        )

    def test_non_marlin_backend_replaces_linear(self):
        """When a non-Marlin backend is selected, the layer is replaced without fallback."""
        marlin_created, fallback_created = self._run_replace(2048, 8192, use_marlin=False)
        # The primary (non-Marlin) backend replaces the layer; no fallback is needed.
        self.assertEqual(len(fallback_created), 0)

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

        # proj was in both lists — to_not_convert takes precedence at the parent level
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


class TestProtectTritonJit(unittest.TestCase):
    """_protect_triton_jit prevents gptqmodel's side effects from breaking Triton."""

    # ------------------------------------------------------------------
    # Helper: inject / remove fake triton modules around a test
    # ------------------------------------------------------------------
    @staticmethod
    def _install_fake_triton(jit_mod):
        import sys
        import types
        saved = {k: sys.modules.get(k) for k in
                 ("triton", "triton.runtime", "triton.runtime.jit")}
        fake_triton = types.ModuleType("triton")
        fake_runtime = types.ModuleType("triton.runtime")
        fake_triton.runtime = fake_runtime
        sys.modules["triton"] = fake_triton
        sys.modules["triton.runtime"] = fake_runtime
        sys.modules["triton.runtime.jit"] = jit_mod
        return saved

    @staticmethod
    def _restore_triton(saved):
        import sys
        for key, val in saved.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val

    @staticmethod
    def _restore_or_remove_module(key, saved):
        """Restore ``key`` in sys.modules to ``saved``, or pop it if ``saved`` is None."""
        import sys
        if saved is not None:
            sys.modules[key] = saved
        else:
            sys.modules.pop(key, None)

    # ------------------------------------------------------------------
    # Defence 1: nogil_patcher stub
    # ------------------------------------------------------------------
    def test_nogil_patcher_stub_installed_before_yield(self):
        """A no-op stub for gptqmodel.utils.nogil_patcher is in sys.modules
        inside the block so that any gptqmodel import finds it there."""
        import sys
        from eole.modules.autoround_linear import _protect_triton_jit

        _KEY = "gptqmodel.utils.nogil_patcher"
        saved_stub = sys.modules.pop(_KEY, None)  # start with key absent
        try:
            with _protect_triton_jit():
                stub = sys.modules.get(_KEY)
                self.assertIsNotNone(stub,
                                     "stub must be installed before yield")
                # Any attribute access on the stub must return a callable no-op
                noop = stub.patch_triton
                self.assertTrue(callable(noop),
                                "stub attributes must be callable")
                self.assertIsNone(noop(),
                                  "stub callables must return None (no-op)")
        finally:
            self._restore_or_remove_module(_KEY, saved_stub)

    def test_nogil_patcher_stub_persists_after_context_exit(self):
        """The stub remains in sys.modules after the context exits so that
        later imports (e.g. auto_round_extension.triton) are also protected."""
        import sys
        from eole.modules.autoround_linear import _protect_triton_jit

        _KEY = "gptqmodel.utils.nogil_patcher"
        saved_stub = sys.modules.pop(_KEY, None)
        try:
            with _protect_triton_jit():
                pass
            self.assertIn(_KEY, sys.modules,
                          "stub must persist after context exit")
        finally:
            self._restore_or_remove_module(_KEY, saved_stub)

    # ------------------------------------------------------------------
    # Defence 2: threadx DeviceThreadPool stub
    # ------------------------------------------------------------------
    def test_threadx_stub_installed_before_yield(self):
        """A no-op stub for gptqmodel.utils.threadx is in sys.modules
        inside the block so that gptqmodel finds it before spawning threads."""
        import sys
        from eole.modules.autoround_linear import _protect_triton_jit

        _KEY = "gptqmodel.utils.threadx"
        saved_stub = sys.modules.pop(_KEY, None)
        try:
            with _protect_triton_jit():
                stub = sys.modules.get(_KEY)
                self.assertIsNotNone(stub,
                                     "threadx stub must be installed before yield")
                # DeviceThreadPool must be the no-op class
                pool_cls = stub.DeviceThreadPool
                self.assertTrue(callable(pool_cls),
                                "DeviceThreadPool must be a callable class")
                # Instantiate: must not spawn any threads
                import threading
                before = threading.active_count()
                pool = pool_cls(workers={"cuda:per": 4}, inference_mode=True)
                after = threading.active_count()
                self.assertEqual(
                    before, after,
                    "no new threads must be spawned by the stub DeviceThreadPool",
                )
        finally:
            self._restore_or_remove_module(_KEY, saved_stub)

    def test_threadx_stub_persists_after_context_exit(self):
        """The threadx stub remains in sys.modules after the context exits."""
        import sys
        from eole.modules.autoround_linear import _protect_triton_jit

        _KEY = "gptqmodel.utils.threadx"
        saved_stub = sys.modules.pop(_KEY, None)
        try:
            with _protect_triton_jit():
                pass
            self.assertIn(_KEY, sys.modules,
                          "threadx stub must persist after context exit")
        finally:
            self._restore_or_remove_module(_KEY, saved_stub)

    # ------------------------------------------------------------------
    # Defence 3: save/restore JITFunction class dict (belt-and-suspenders)
    # ------------------------------------------------------------------
    def test_changed_attributes_are_restored(self):
        """Attributes changed on JITFunction inside the block are restored."""
        import sys
        import types
        from eole.modules.autoround_linear import _protect_triton_jit

        fake_jit_mod = types.ModuleType("triton.runtime.jit")

        class FakeJITFunction:
            def run(self, *args, **kwargs):
                pass

        fake_jit_mod.JITFunction = FakeJITFunction
        original_run = FakeJITFunction.run
        saved_triton = self._install_fake_triton(fake_jit_mod)

        # Ensure stub keys are absent so the defences install fresh stubs
        _KEY = "gptqmodel.utils.nogil_patcher"
        _TX_KEY = "gptqmodel.utils.threadx"
        saved_patcher = sys.modules.pop(_KEY, None)
        saved_threadx = sys.modules.pop(_TX_KEY, None)
        try:
            with _protect_triton_jit():
                def patched_run(self, *args, **kwargs):
                    pass
                FakeJITFunction.run = patched_run

            self.assertIs(FakeJITFunction.run, original_run,
                          "run must be restored to its original value")
        finally:
            self._restore_triton(saved_triton)
            self._restore_or_remove_module(_KEY, saved_patcher)
            self._restore_or_remove_module(_TX_KEY, saved_threadx)

    def test_added_attributes_are_deleted(self):
        """Attributes *added* to JITFunction inside the block are deleted."""
        import sys
        import types
        from eole.modules.autoround_linear import _protect_triton_jit

        fake_jit_mod = types.ModuleType("triton.runtime.jit")

        class FakeJITFunction:
            pass

        fake_jit_mod.JITFunction = FakeJITFunction
        saved_triton = self._install_fake_triton(fake_jit_mod)

        _KEY = "gptqmodel.utils.nogil_patcher"
        _TX_KEY = "gptqmodel.utils.threadx"
        saved_patcher = sys.modules.pop(_KEY, None)
        saved_threadx = sys.modules.pop(_TX_KEY, None)
        try:
            with _protect_triton_jit():
                # Simulate patcher adding a brand-new attribute
                FakeJITFunction.new_attr_from_patcher = "injected"

            self.assertFalse(
                hasattr(FakeJITFunction, "new_attr_from_patcher"),
                "attributes added by the patcher must be removed on exit",
            )
        finally:
            self._restore_triton(saved_triton)
            self._restore_or_remove_module(_KEY, saved_patcher)
            self._restore_or_remove_module(_TX_KEY, saved_threadx)

    def test_no_triton_is_noop(self):
        """_protect_triton_jit is a no-op (JIT snapshot) when triton is not installed."""
        import sys
        from eole.modules.autoround_linear import _protect_triton_jit

        saved = sys.modules.pop("triton.runtime.jit", None)
        _PATCHER_KEY = "gptqmodel.utils.nogil_patcher"
        _THREADX_KEY = "gptqmodel.utils.threadx"
        saved_patcher = sys.modules.pop(_PATCHER_KEY, None)
        saved_threadx = sys.modules.pop(_THREADX_KEY, None)
        try:
            with _protect_triton_jit():
                pass  # must not raise
        finally:
            if saved is not None:
                sys.modules["triton.runtime.jit"] = saved
            self._restore_or_remove_module(_PATCHER_KEY, saved_patcher)
            self._restore_or_remove_module(_THREADX_KEY, saved_threadx)


if __name__ == "__main__":
    unittest.main()


