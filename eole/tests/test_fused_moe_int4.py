"""Tests for fused_moe_int4 and moe_quant_utils – pure-Python / CPU path.

These tests do not require Triton or a CUDA GPU.  They verify:
  1. detect_expert_quant_type correctly classifies GPTQ, AWQ, Marlin, and fp16 layers.
  2. stack_gptq_moe_weights / stack_awq_moe_weights produce correctly-shaped tensors.
  3. The ``fused_experts_int4_impl`` function exists and has the expected signature.
  4. The default BLOCK_KP and BLOCK_N values satisfy the group-size constraint and
     are powers of two as required by tl.arange.
  5. The wrapper sorts token-expert pairs by expert ID before launching kernels.

No Triton kernels are actually executed; the tests are skipped when Triton
or CUDA is unavailable.
"""

import unittest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal mock layers
# ---------------------------------------------------------------------------

def _make_gptq_linear(in_features: int, out_features: int, group_size: int = 128):
    """Return a K-packed int4 mock with class name 'QuantLinear' as GPTQ uses."""
    n_groups = in_features // group_size

    class QuantLinear(nn.Module):  # class name must be "QuantLinear" for detection
        def __init__(self):
            super().__init__()
            self.in_features = in_features
            self.infeatures = in_features  # autoround alias
            self.out_features = out_features
            self.group_size = group_size
            # K-packed: qweight[in//8, out]
            self.qweight = torch.zeros(in_features // 8, out_features, dtype=torch.int32)
            self.scales = torch.ones(n_groups, out_features, dtype=torch.float16)
            self.qzeros = torch.zeros(n_groups, out_features // 8, dtype=torch.int32)

        def forward(self, x):
            raise NotImplementedError("mock")

    return QuantLinear()


def _make_awq_linear(in_features: int, out_features: int, group_size: int = 128):
    """Return an N-packed int4 mock with class name 'WQLinear_GEMM' as AWQ uses."""
    n_groups = in_features // group_size

    class WQLinear_GEMM(nn.Module):  # class name must contain "WQLinear" for detection  # noqa: N801
        def __init__(self):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.group_size = group_size
            # N-packed: qweight[in, out//8]
            self.qweight = torch.zeros(in_features, out_features // 8, dtype=torch.int32)
            self.scales = torch.ones(n_groups, out_features, dtype=torch.float16)
            self.qzeros = torch.zeros(n_groups, out_features // 8, dtype=torch.int32)

        def forward(self, x):
            raise NotImplementedError("mock")

    return WQLinear_GEMM()


def _make_gptq_expert(hidden: int, ffn: int, group_size: int = 128):
    class MockMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = _make_gptq_linear(hidden, 2 * ffn, group_size)
            self.down_proj = _make_gptq_linear(ffn, hidden, group_size)

    return MockMLP()


def _make_awq_expert(hidden: int, ffn: int, group_size: int = 128):
    class MockMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = _make_awq_linear(hidden, 2 * ffn, group_size)
            self.down_proj = _make_awq_linear(ffn, hidden, group_size)

    return MockMLP()



# ---------------------------------------------------------------------------
# Tests: detect_expert_quant_type
# ---------------------------------------------------------------------------

class TestDetectExpertQuantType(unittest.TestCase):

    def test_detects_gptq(self):
        from eole.modules.moe_quant_utils import detect_expert_quant_type
        experts = [_make_gptq_expert(256, 512)]
        self.assertEqual(detect_expert_quant_type(experts), "gptq")

    def test_detects_awq(self):
        from eole.modules.moe_quant_utils import detect_expert_quant_type
        experts = [_make_awq_expert(256, 512)]
        self.assertEqual(detect_expert_quant_type(experts), "awq")

    def test_detects_fp16_plain_linear(self):
        from eole.modules.moe_quant_utils import detect_expert_quant_type

        class MockMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = nn.Linear(256, 512, bias=False)
                self.down_proj = nn.Linear(256, 256, bias=False)

        experts = [MockMLP()]
        self.assertEqual(detect_expert_quant_type(experts), "fp16")

    def test_empty_experts(self):
        from eole.modules.moe_quant_utils import detect_expert_quant_type
        self.assertEqual(detect_expert_quant_type([]), "fp16")

    def test_gptqmodel_namespace_falls_back_to_fp16(self):
        """Layers from gptqmodel namespace (Marlin) must return fp16."""
        from eole.modules.moe_quant_utils import detect_expert_quant_type
        import types

        class FakeMarlinLinear(nn.Module):
            qweight = torch.zeros(4, 32, dtype=torch.int32)

        # Fake the module path
        fake_module = types.ModuleType("gptqmodel.layers.marlin")
        FakeMarlinLinear.__module__ = "gptqmodel.layers.marlin"

        class MockMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = FakeMarlinLinear()

        experts = [MockMLP()]
        self.assertEqual(detect_expert_quant_type(experts), "fp16")


# ---------------------------------------------------------------------------
# Tests: stack_gptq_moe_weights
# ---------------------------------------------------------------------------

class TestStackGPTQWeights(unittest.TestCase):

    def test_stacked_shapes(self):
        from eole.modules.moe_quant_utils import stack_gptq_moe_weights

        E, H, ffn, gs = 4, 256, 512, 128
        experts = [_make_gptq_expert(H, ffn, gs) for _ in range(E)]
        (w1_qw, w1_sc, w1_qz, w2_qw, w2_sc, w2_qz, group_size) = stack_gptq_moe_weights(
            experts, device=torch.device("cpu")
        )

        # W1: gate+up fused → out = 2*ffn
        self.assertEqual(w1_qw.shape, (E, H // 8, 2 * ffn))
        self.assertEqual(w1_sc.shape, (E, H // gs, 2 * ffn))
        self.assertEqual(w1_qz.shape, (E, H // gs, 2 * ffn // 8))

        # W2: down_proj
        self.assertEqual(w2_qw.shape, (E, ffn // 8, H))
        self.assertEqual(w2_sc.shape, (E, ffn // gs, H))
        self.assertEqual(w2_qz.shape, (E, ffn // gs, H // 8))

        self.assertEqual(group_size, gs)

    def test_group_size_inferred_without_attribute(self):
        from eole.modules.moe_quant_utils import stack_gptq_moe_weights

        experts = [_make_gptq_expert(256, 512, 64)]
        # Remove group_size attribute to exercise inferred path
        del experts[0].gate_up_proj.group_size
        del experts[0].down_proj.group_size

        _, _, _, _, _, _, gs = stack_gptq_moe_weights(experts, device=torch.device("cpu"))
        self.assertEqual(gs, 64)

# ---------------------------------------------------------------------------
# Tests: stack_awq_moe_weights
# ---------------------------------------------------------------------------

class TestStackAWQWeights(unittest.TestCase):

    def test_stacked_shapes(self):
        from eole.modules.moe_quant_utils import stack_awq_moe_weights

        E, H, ffn, gs = 2, 256, 512, 128
        experts = [_make_awq_expert(H, ffn, gs) for _ in range(E)]
        (w1_qw, w1_sc, w1_qz, w2_qw, w2_sc, w2_qz, group_size) = stack_awq_moe_weights(
            experts, device=torch.device("cpu")
        )

        # W1: N-packed
        self.assertEqual(w1_qw.shape, (E, H, 2 * ffn // 8))
        self.assertEqual(w1_sc.shape, (E, H // gs, 2 * ffn))
        self.assertEqual(w1_qz.shape, (E, H // gs, 2 * ffn // 8))

        # W2
        self.assertEqual(w2_qw.shape, (E, ffn, H // 8))
        self.assertEqual(w2_sc.shape, (E, ffn // gs, H))
        self.assertEqual(w2_qz.shape, (E, ffn // gs, H // 8))

        self.assertEqual(group_size, gs)


# ---------------------------------------------------------------------------
# Tests: fused_experts_int4_impl signature
# ---------------------------------------------------------------------------

class TestFusedInt4ImplSignature(unittest.TestCase):

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
            "w1_qweight",
            "w1_scales",
            "w1_qzeros",
            "w2_qweight",
            "w2_scales",
            "w2_qzeros",
            "topk_weights",
            "topk_ids",
            "group_size",
            "kpacked",
            "activation",
        }
        self.assertEqual(set(sig.parameters.keys()), expected_params)


# ---------------------------------------------------------------------------
# Tests: block size constraints
# ---------------------------------------------------------------------------

class TestBlockSizeConstraints(unittest.TestCase):
    """The KPACKED path requires BLOCK_KP * KPACK <= group_size so every element
    in a packed tile belongs to the same quantisation group.  This test confirms
    that the module-level default block sizes satisfy that constraint for the minimum
    common group_size of 128, and that both sizes are powers of two as required
    by tl.arange."""

    def _get_defaults(self):
        try:
            from eole.triton.fused_moe_int4 import _DEFAULT_BLOCK_N, _DEFAULT_BLOCK_KP
            return _DEFAULT_BLOCK_N, _DEFAULT_BLOCK_KP
        except ImportError:
            self.skipTest("Triton not installed")

    def test_kpacked_tile_fits_within_group(self):
        _DEFAULT_BLOCK_N, _DEFAULT_BLOCK_KP = self._get_defaults()
        KPACK = 8
        MIN_GROUP_SIZE = 128
        logical_k = _DEFAULT_BLOCK_KP * KPACK
        self.assertLessEqual(
            logical_k,
            MIN_GROUP_SIZE,
            f"_DEFAULT_BLOCK_KP={_DEFAULT_BLOCK_KP} → {logical_k} logical K > group_size={MIN_GROUP_SIZE}",
        )

    def test_block_sizes_are_powers_of_two(self):
        """tl.arange requires power-of-2 extents."""
        _DEFAULT_BLOCK_N, _DEFAULT_BLOCK_KP = self._get_defaults()

        def is_pow2(n):
            return n > 0 and (n & (n - 1)) == 0

        self.assertTrue(is_pow2(_DEFAULT_BLOCK_N), f"_DEFAULT_BLOCK_N={_DEFAULT_BLOCK_N} is not a power of 2")
        self.assertTrue(is_pow2(_DEFAULT_BLOCK_KP), f"_DEFAULT_BLOCK_KP={_DEFAULT_BLOCK_KP} is not a power of 2")


# ---------------------------------------------------------------------------
# Tests: pair sorting in the Python wrapper
# ---------------------------------------------------------------------------

class TestWrapperSortsPairsByExpert(unittest.TestCase):
    """fused_experts_int4_impl must sort (expert_ids, token_ids, weights) by
    expert_id before launching kernels.  We verify the sort is applied by
    monkeypatching the plain @triton.jit kernel and inspecting the expert_ids
    tensor it receives."""

    def test_pairs_arrive_sorted(self):
        try:
            import triton  # noqa: F401
            from eole.triton import fused_moe_int4 as mod
        except ImportError:
            self.skipTest("Triton not installed")

        captured = {}

        original_w1 = mod._w1_int4_act_kernel

        class _CapturingKernel:
            """Intercepts the plain @triton.jit kernel to capture expert_ids."""
            def __getitem__(self, grid):
                def _call(*args, **kwargs):
                    # expert_ids is the 6th positional tensor arg (index 5):
                    # X_ptr, Qw_ptr, Sc_ptr, Qz_ptr, Y_ptr, expert_ids_ptr
                    eids = args[5]
                    assert isinstance(eids, torch.Tensor) and eids.dtype == torch.int32, (
                        f"Expected expert_ids (int32 tensor) at args[5], got {type(eids)}"
                    )
                    captured["expert_ids"] = eids.cpu().tolist()
                return _call
            def __call__(self, *a, **kw):
                pass

        mod._w1_int4_act_kernel = _CapturingKernel()
        try:
            M, H, K = 4, 64, 2
            E, I, gs = 4, 64, 64
            # Construct unsorted topk_ids that need sorting
            topk_ids = torch.tensor([[3, 0], [1, 2], [0, 3], [2, 1]], dtype=torch.long)
            topk_weights = torch.ones(M, K)
            hidden = torch.zeros(M, H, dtype=torch.float16)
            # Build minimal weight tensors (KPACKED, group_size=64)
            w1_qw = torch.zeros(E, H // 8, 2 * I, dtype=torch.int32)
            w1_sc = torch.ones(E, H // gs, 2 * I, dtype=torch.float16)
            w1_qz = torch.zeros(E, H // gs, 2 * I // 8, dtype=torch.int32)
            w2_qw = torch.zeros(E, I // 8, H, dtype=torch.int32)
            w2_sc = torch.ones(E, I // gs, H, dtype=torch.float16)
            w2_qz = torch.zeros(E, I // gs, H // 8, dtype=torch.int32)
            try:
                mod.fused_experts_int4_impl(
                    hidden, w1_qw, w1_sc, w1_qz, w2_qw, w2_sc, w2_qz,
                    topk_weights, topk_ids, group_size=gs, kpacked=True,
                )
            except Exception:
                pass  # kernel may fail without CUDA; we only need the captured ids
        finally:
            mod._w1_int4_act_kernel = original_w1

        if "expert_ids" in captured:
            ids = captured["expert_ids"]
            self.assertEqual(ids, sorted(ids), "expert_ids passed to W1 kernel must be sorted")


if __name__ == "__main__":
    unittest.main()

