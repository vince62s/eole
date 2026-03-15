"""Unit tests for moe_quant_utils.py.

These tests run on CPU and do NOT require CUDA, Triton, or any quantisation
backend (gptqmodel, autoGPTQ, etc.).  They verify the detection and stacking
logic using lightweight fakes that mirror the real layer interfaces.
"""

import importlib.util
import os
import types
import unittest

import torch


# ---------------------------------------------------------------------------
# Import moe_quant_utils without triggering optional heavy dependencies
# ---------------------------------------------------------------------------


def _import_moe_quant_utils():
    mod_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "modules",
        "moe_quant_utils.py",
    )
    spec = importlib.util.spec_from_file_location("eole.modules.moe_quant_utils", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _import_moe_quant_utils()
detect_expert_quant_type = _mod.detect_expert_quant_type
stack_marlin_moe_weights = _mod.stack_marlin_moe_weights


# ---------------------------------------------------------------------------
# Fake layer helpers
# ---------------------------------------------------------------------------


def _fake_linear(in_f, out_f):
    """Plain nn.Linear."""
    return torch.nn.Linear(in_f, out_f, bias=False)


def _fake_gptq_layer(in_f, out_f, group_size=128):
    """Fake GPTQ QuantLinear (K-packed)."""
    # Dynamically create a class with the right module path
    QuantLinear = type(
        "QuantLinear",
        (),
        {"__module__": "auto_gptq.nn_modules.qlinear.qlinear_triton"},
    )
    layer = QuantLinear()
    layer.qweight = torch.zeros(in_f // 8, out_f, dtype=torch.int32)
    layer.scales = torch.ones(in_f // group_size, out_f, dtype=torch.float16)
    layer.qzeros = torch.zeros(in_f // group_size, out_f // 8, dtype=torch.int32)
    layer.group_size = group_size
    # detect_expert_quant_type checks both 'infeatures' and 'in_features' attribute names
    layer.infeatures = in_f
    layer.in_features = in_f
    return layer


def _fake_awq_layer(in_f, out_f, group_size=128):
    """Fake AWQ WQLinear_GEMM (N-packed)."""
    WQLinear_GEMM = type(
        "WQLinear_GEMM",
        (),
        {"__module__": "awq.modules.linear.gemm"},
    )
    layer = WQLinear_GEMM()
    layer.qweight = torch.zeros(in_f, out_f // 8, dtype=torch.int32)
    layer.scales = torch.ones(in_f // group_size, out_f, dtype=torch.float16)
    layer.qzeros = torch.zeros(in_f // group_size, out_f // 8, dtype=torch.int32)
    layer.group_size = group_size
    return layer


def _fake_marlin_layer(in_f, out_f, group_size=128):
    """Fake gptqmodel MarlinQuantLinear (post-post_init state)."""
    MarlinQuantLinear = type(
        "MarlinQuantLinear",
        (),
        {"__module__": "gptqmodel.nn_modules.qlinear.marlin"},
    )
    layer = MarlinQuantLinear()
    # After post_init the qweight is in Marlin tile format — shape differs
    # from GPTQ format (in_f//8, out_f).  We use a plausible shape here.
    layer.qweight = torch.zeros(in_f // 16, out_f * 2, dtype=torch.int32)
    layer.scales = torch.ones(in_f // group_size, out_f, dtype=torch.float16)
    layer.qzeros = torch.zeros(0, dtype=torch.int32)       # empty for Marlin
    layer.g_idx = torch.zeros(0, dtype=torch.int32)         # empty (no act_order)
    layer.g_idx_sort_indices = torch.zeros(0, dtype=torch.int32)
    layer.workspace = torch.zeros(64, dtype=torch.int32)    # mock workspace
    layer.in_features = in_f
    layer.out_features = out_f
    layer.group_size = group_size
    # ScalarType mock — just needs a .id attribute
    scalar_type = types.SimpleNamespace(id=1)
    layer.weight_type = scalar_type
    layer.is_k_full = True
    return layer


def _fake_expert(gate_up, down):
    e = types.SimpleNamespace()
    e.gate_up_proj = gate_up
    e.up_proj = None   # gate_up already concatenates gate+up
    e.down_proj = down
    return e


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDetectExpertQuantType(unittest.TestCase):
    """Tests for detect_expert_quant_type()."""

    def _make_experts(self, layer_factory_gate_up, layer_factory_down, n=2):
        experts = []
        for _ in range(n):
            e = _fake_expert(layer_factory_gate_up(), layer_factory_down())
            experts.append(e)
        return experts

    def test_empty(self):
        self.assertEqual(detect_expert_quant_type([]), "fp16")

    def test_fp16_linear(self):
        experts = self._make_experts(
            lambda: _fake_linear(256, 512),
            lambda: _fake_linear(256, 256),
        )
        self.assertEqual(detect_expert_quant_type(experts), "fp16")

    def test_gptq(self):
        experts = self._make_experts(
            lambda: _fake_gptq_layer(256, 512),
            lambda: _fake_gptq_layer(256, 256),
        )
        self.assertEqual(detect_expert_quant_type(experts), "gptq")

    def test_awq(self):
        experts = self._make_experts(
            lambda: _fake_awq_layer(256, 512),
            lambda: _fake_awq_layer(256, 256),
        )
        self.assertEqual(detect_expert_quant_type(experts), "awq")

    def test_marlin(self):
        """gptqmodel Marlin layers must be detected as 'marlin'."""
        experts = self._make_experts(
            lambda: _fake_marlin_layer(256, 512),
            lambda: _fake_marlin_layer(256, 256),
        )
        self.assertEqual(detect_expert_quant_type(experts), "marlin")


class TestStackMarlinMoeWeights(unittest.TestCase):
    """Tests for stack_marlin_moe_weights()."""

    def _build_experts(self, num_experts=4, H=256, I=128, group_size=64):
        """Build fake MarlinQuantLinear experts."""
        experts = []
        for _ in range(num_experts):
            gate_up = _fake_marlin_layer(H, 2 * I, group_size=group_size)
            down    = _fake_marlin_layer(I, H,     group_size=group_size)
            experts.append(_fake_expert(gate_up, down))
        return experts

    def test_returns_correct_lengths(self):
        E = 4
        experts = self._build_experts(E)
        result = stack_marlin_moe_weights(experts, device=torch.device("cpu"))
        (
            w1_qw, w1_sc, w1_qz, w1_gi, w1_gs,
            w2_qw, w2_sc, w2_qz, w2_gi, w2_gs,
            workspaces, quant_type,
            in_features, intermediate_features, out_features,
            is_k_full,
        ) = result
        for lst in (w1_qw, w1_sc, w1_qz, w1_gi, w1_gs,
                    w2_qw, w2_sc, w2_qz, w2_gi, w2_gs, workspaces):
            self.assertEqual(len(lst), E)

    def test_metadata(self):
        H, I, gs = 512, 256, 128
        experts = self._build_experts(num_experts=3, H=H, I=I, group_size=gs)
        result = stack_marlin_moe_weights(experts, device=torch.device("cpu"))
        (
            *_,
            quant_type,
            in_features, intermediate_features, out_features,
            is_k_full,
        ) = result
        self.assertEqual(in_features, H)
        self.assertEqual(intermediate_features, I)   # W1 out = 2*I, so I = out//2
        self.assertEqual(out_features, H)
        self.assertTrue(is_k_full)
        self.assertIsNotNone(quant_type)

    def test_w1_w2_shapes(self):
        H, I, gs = 256, 128, 64
        experts = self._build_experts(num_experts=2, H=H, I=I, group_size=gs)
        (
            w1_qw, w1_sc, w1_qz, w1_gi, w1_gs,
            w2_qw, w2_sc, w2_qz, w2_gi, w2_gs,
            *_
        ) = stack_marlin_moe_weights(experts, device=torch.device("cpu"))

        for e in range(2):
            # W1 qweight: (H//16, 2I*2)  [Marlin tile for int4]
            self.assertEqual(w1_qw[e].shape, (H // 16, 2 * I * 2))
            # W1 scales: (H//gs, 2I)
            self.assertEqual(w1_sc[e].shape, (H // gs, 2 * I))
            # W2 qweight: (I//16, H*2)
            self.assertEqual(w2_qw[e].shape, (I // 16, H * 2))
            # W2 scales: (I//gs, H)
            self.assertEqual(w2_sc[e].shape, (I // gs, H))


if __name__ == "__main__":
    unittest.main(verbosity=2)
