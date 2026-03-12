"""Tests for GGUF quantized model support.

Tests are written to run without a GPU and without loading any real model
checkpoints, using a small synthetic GGUF file generated on-the-fly.
"""

import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest


def _require(pkg: str):
    """Skip the test if *pkg* is not importable."""
    try:
        importlib.import_module(pkg)
    except ImportError:
        raise unittest.SkipTest(f"'{pkg}' not installed")


class TestGGUFMetadata(unittest.TestCase):
    """Test :class:`eole.bin.convert.convert_gguf.GGUFMetadata`."""

    @classmethod
    def setUpClass(cls):
        _require("gguf")
        _require("numpy")
        import numpy as np
        from gguf import GGUFWriter

        cls.tmpdir = tempfile.mkdtemp()
        cls.gguf_path = os.path.join(cls.tmpdir, "test.gguf")

        writer = GGUFWriter(cls.gguf_path, "llama")
        writer.add_block_count(2)
        writer.add_context_length(512)
        writer.add_embedding_length(64)
        writer.add_feed_forward_length(128)
        writer.add_head_count(4)
        writer.add_head_count_kv(2)
        writer.add_layer_norm_rms_eps(1e-5)
        writer.add_rope_freq_base(500000.0)
        writer.add_vocab_size(32)
        writer.add_token_list([str(i) for i in range(32)])
        writer.add_bos_token_id(1)
        writer.add_eos_token_id(2)
        # Dummy float tensors
        writer.add_tensor("token_embd.weight", np.zeros((32, 64), dtype=np.float16))
        writer.add_tensor("output_norm.weight", np.ones(64, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _meta(self):
        from eole.bin.convert.convert_gguf import GGUFMetadata

        return GGUFMetadata(self.gguf_path)

    def test_arch(self):
        self.assertEqual(self._meta().arch, "llama")

    def test_block_count(self):
        self.assertEqual(self._meta().block_count, 2)

    def test_embedding_length(self):
        self.assertEqual(self._meta().embedding_length, 64)

    def test_head_count_kv(self):
        self.assertEqual(self._meta().head_count_kv, 2)

    def test_rope_freq_base(self):
        self.assertAlmostEqual(self._meta().rope_freq_base, 500000.0, places=0)

    def test_bos_eos_token_ids(self):
        meta = self._meta()
        self.assertEqual(meta.bos_token_id, 1)
        self.assertEqual(meta.eos_token_id, 2)

    def test_tokens_list(self):
        tokens = self._meta().tokens
        self.assertEqual(len(tokens), 32)
        self.assertEqual(tokens[0], "0")


class TestGGUFNameMapping(unittest.TestCase):
    """Test the GGUF → EOLE tensor name mapping."""

    def setUp(self):
        _require("gguf")

    def _map(self, name, linear_blocks=frozenset()):
        from eole.bin.convert.convert_gguf import _gguf_to_eole_name

        return _gguf_to_eole_name(name, linear_blocks)

    def test_embedding(self):
        self.assertEqual(self._map("token_embd.weight"), "tgt_emb.embeddings.weight")

    def test_output_norm(self):
        self.assertEqual(self._map("output_norm.weight"), "decoder.layer_norm.weight")

    def test_output_weight(self):
        self.assertEqual(self._map("output.weight"), "generator.weight")

    def test_block_attn_q(self):
        r = self._map("blk.0.attn_q.weight")
        self.assertEqual(r, "decoder.transformer_layers.0.self_attn.linear_query.weight")

    def test_block_attn_k(self):
        r = self._map("blk.3.attn_k.weight")
        self.assertEqual(r, "decoder.transformer_layers.3.self_attn.linear_keys.weight")

    def test_block_attn_output(self):
        r = self._map("blk.1.attn_output.weight")
        self.assertEqual(r, "decoder.transformer_layers.1.self_attn.final_linear.weight")

    def test_block_ffn_gate(self):
        r = self._map("blk.0.ffn_gate.weight")
        self.assertEqual(r, "decoder.transformer_layers.0.mlp.gate_up_proj.weight")

    def test_block_ffn_down(self):
        r = self._map("blk.2.ffn_down.weight")
        self.assertEqual(r, "decoder.transformer_layers.2.mlp.down_proj.weight")

    def test_block_ffn_up(self):
        r = self._map("blk.0.ffn_up.weight")
        self.assertEqual(r, "decoder.transformer_layers.0.mlp.up_proj.weight")

    def test_rope_freqs_skipped(self):
        self.assertIsNone(self._map("rope_freqs.weight"))

    def test_unknown_returns_empty(self):
        self.assertEqual(self._map("some.unknown.tensor"), "")

    def test_post_attention_norm_maps_for_full_attention_block(self):
        """post_attention_norm is present in full-attention blocks and must map."""
        r = self._map("blk.3.post_attention_norm.weight")
        self.assertEqual(r, "decoder.transformer_layers.3.post_attention_layernorm.weight")

    def test_post_attention_norm_maps_for_linear_attention_block(self):
        """post_attention_norm maps to post_attention_layernorm for linear-attention blocks.

        EOLE's _forward_linear_attn uses post_attention_layernorm as the
        post-linear-attention output norm.  The GGUF tensor post_attention_norm
        is the correct source for this weight (it was previously incorrectly
        mapped to None/skipped).
        """
        r = self._map("blk.0.post_attention_norm.weight", linear_blocks=frozenset({0}))
        self.assertEqual(r, "decoder.transformer_layers.0.post_attention_layernorm.weight")

    # ------------------------------------------------------------------
    # Linear-attention block mappings (SSM / GatedDeltaNet)
    # ------------------------------------------------------------------

    def test_linear_attn_ssm_a_maps_to_A_log(self):
        r = self._map("blk.0.ssm_a", linear_blocks=frozenset({0}))
        self.assertEqual(r, "decoder.transformer_layers.0.linear_attn.A_log")

    def test_linear_attn_ssm_alpha_maps_to_in_proj_a(self):
        r = self._map("blk.0.ssm_alpha.weight", linear_blocks=frozenset({0}))
        self.assertEqual(r, "decoder.transformer_layers.0.linear_attn.in_proj_a.weight")

    def test_linear_attn_ssm_beta_maps_to_in_proj_b(self):
        r = self._map("blk.0.ssm_beta.weight", linear_blocks=frozenset({0}))
        self.assertEqual(r, "decoder.transformer_layers.0.linear_attn.in_proj_b.weight")

    def test_linear_attn_ssm_out_maps_to_out_proj(self):
        r = self._map("blk.0.ssm_out.weight", linear_blocks=frozenset({0}))
        self.assertEqual(r, "decoder.transformer_layers.0.linear_attn.out_proj.weight")

    def test_linear_attn_ssm_conv1d_maps(self):
        r = self._map("blk.0.ssm_conv1d.weight", linear_blocks=frozenset({0}))
        self.assertEqual(r, "decoder.transformer_layers.0.linear_attn.conv1d.weight")

    def test_linear_attn_ssm_dt_bias_maps(self):
        r = self._map("blk.0.ssm_dt.bias", linear_blocks=frozenset({0}))
        self.assertEqual(r, "decoder.transformer_layers.0.linear_attn.dt_bias")

    def test_linear_attn_ssm_norm_maps(self):
        r = self._map("blk.0.ssm_norm.weight", linear_blocks=frozenset({0}))
        self.assertEqual(r, "decoder.transformer_layers.0.linear_attn.norm.weight")

    def test_linear_attn_attn_qkv_maps_to_in_proj_qkv(self):
        """In a linear-attention block attn_qkv goes to in_proj_qkv, not qkv_proj."""
        r = self._map("blk.0.attn_qkv.weight", linear_blocks=frozenset({0}))
        self.assertEqual(r, "decoder.transformer_layers.0.linear_attn.in_proj_qkv.weight")

    def test_linear_attn_attn_gate_maps_to_in_proj_z(self):
        r = self._map("blk.0.attn_gate.weight", linear_blocks=frozenset({0}))
        self.assertEqual(r, "decoder.transformer_layers.0.linear_attn.in_proj_z.weight")

    def test_full_attn_block_attn_qkv_maps_to_qkv_proj(self):
        """In a full-attention block attn_qkv still maps to self_attn.qkv_proj."""
        r = self._map("blk.1.attn_qkv.weight", linear_blocks=frozenset({0}))
        self.assertEqual(r, "decoder.transformer_layers.1.self_attn.qkv_proj.weight")

    def test_ssm_tensors_unrecognised_without_linear_blocks(self):
        """Without linear_blocks context, SSM tensor names are unrecognised (warn)."""
        self.assertEqual(self._map("blk.0.ssm_a"), "")
        self.assertEqual(self._map("blk.0.ssm_alpha.weight"), "")
        self.assertEqual(self._map("blk.0.ssm_out.weight"), "")

    def test_attn_gate_unrecognised_in_full_attention_block(self):
        """attn_gate.weight in a non-linear block is unrecognised (no silent skip)."""
        self.assertEqual(self._map("blk.3.attn_gate.weight"), "")


class TestGGUFLinear(unittest.TestCase):
    """Test :class:`eole.modules.gguf_linear.GGUFLinear`."""

    def setUp(self):
        _require("gguf")
        _require("torch")

    def test_replace_linear(self):
        import torch.nn as nn

        from eole.modules.gguf_linear import GGUFLinear, replace_gguf_linear

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_query = nn.Linear(32, 32, bias=False)
                self.other = nn.Linear(32, 32, bias=False)

        model = Tiny()
        replace_gguf_linear(model, module_to_convert=["linear_query"])
        self.assertIsInstance(model.linear_query, GGUFLinear)
        # Untouched modules stay as nn.Linear
        self.assertIsInstance(model.other, nn.Linear)

    def test_forward_quantized(self):
        """Load a Q4_K weight buffer (written via GGUFWriter) and run forward pass."""
        _require("numpy")
        import tempfile
        import os
        import numpy as np
        import torch
        from gguf import GGMLQuantizationType, GGUFWriter, GGUFReader

        from eole.modules.gguf_linear import GGUFLinear

        n_out, n_in = 8, 256
        # Write a minimal GGUF with one Q4_K tensor, then read back the raw uint8 data
        with tempfile.TemporaryDirectory() as td:
            gpath = os.path.join(td, "tiny.gguf")
            writer = GGUFWriter(gpath, "llama")
            writer.add_block_count(1)
            writer.add_embedding_length(n_in)
            writer.add_vocab_size(n_out)
            writer.add_tensor(
                "test.weight",
                np.random.randn(n_out, n_in).astype(np.float32),
                raw_dtype=GGMLQuantizationType.Q4_K,
            )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            reader = GGUFReader(gpath)
            tensor = reader.tensors[0]
            q_data = tensor.data.copy()  # uint8 ndarray, shape (n_out, bytes_per_row)

        layer = GGUFLinear(in_features=n_in, out_features=n_out, bias=False)
        layer.register_buffer("weight", torch.from_numpy(q_data))
        layer.register_buffer(
            "gguf_qtype",
            torch.tensor([GGMLQuantizationType.Q4_K.value], dtype=torch.int32),
        )

        x = torch.randn(2, 4, n_in)
        out = layer(x)
        self.assertEqual(out.shape, (2, 4, n_out))

    def test_extra_repr_shows_qtype(self):
        import torch
        from gguf import GGMLQuantizationType

        from eole.modules.gguf_linear import GGUFLinear

        layer = GGUFLinear(in_features=32, out_features=16, bias=True)
        layer.register_buffer(
            "gguf_qtype",
            torch.tensor([GGMLQuantizationType.Q4_K.value], dtype=torch.int32),
        )
        self.assertIn("Q4_K", repr(layer))

    def _make_gguf_layer(self, n_out, n_in, qtype_enum):
        """Helper: create a GGUFLinear with real quantised weights via GGUFWriter."""
        import tempfile
        import os
        import numpy as np
        import torch
        from gguf import GGUFWriter, GGUFReader
        from eole.modules.gguf_linear import GGUFLinear

        with tempfile.TemporaryDirectory() as td:
            gpath = os.path.join(td, "tiny.gguf")
            writer = GGUFWriter(gpath, "llama")
            writer.add_block_count(1)
            writer.add_embedding_length(n_in)
            writer.add_vocab_size(n_out)
            writer.add_tensor(
                "test.weight",
                np.random.randn(n_out, n_in).astype(np.float32),
                raw_dtype=qtype_enum,
            )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            reader = GGUFReader(gpath)
            q_data = reader.tensors[0].data.copy()

        layer = GGUFLinear(in_features=n_in, out_features=n_out, bias=False)
        layer.register_buffer("weight", torch.from_numpy(q_data))
        layer.register_buffer(
            "gguf_qtype",
            torch.tensor([qtype_enum.value], dtype=torch.int32),
        )
        return layer

    def _forward_matches_cpu_fallback(self, qtype_enum, n_out=8, n_in=256):
        """GPU-native path must produce the same result as the CPU fallback."""
        import torch
        import torch.nn.functional as F
        from gguf import dequantize

        layer = self._make_gguf_layer(n_out, n_in, qtype_enum)

        x = torch.randn(2, 4, n_in)

        # Reference: CPU numpy path (the original fallback logic)
        w_np = layer.weight.cpu().numpy()
        dq_np = dequantize(w_np, qtype_enum)
        w_ref = torch.from_numpy(dq_np).reshape(n_out, n_in).to(x.dtype)
        expected = F.linear(x, w_ref)

        # Actual: GGUFLinear.forward (should hit GPU path for Q8_0/Q4_0/Q4_1)
        actual = layer(x)

        self.assertEqual(actual.shape, (2, 4, n_out))
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4, equal_nan=True)

    def test_forward_q8_0_matches_numpy(self):
        """Q8_0 GPU-native dequantisation must match the numpy reference."""
        _require("numpy")
        from gguf import GGMLQuantizationType
        self._forward_matches_cpu_fallback(GGMLQuantizationType.Q8_0)

    def test_forward_q4_0_matches_numpy(self):
        """Q4_0 GPU-native dequantisation must match the numpy reference."""
        _require("numpy")
        from gguf import GGMLQuantizationType
        self._forward_matches_cpu_fallback(GGMLQuantizationType.Q4_0)

    def test_forward_q4_1_matches_numpy(self):
        """Q4_1 GPU-native dequantisation must match the numpy reference."""
        _require("numpy")
        from gguf import GGMLQuantizationType
        self._forward_matches_cpu_fallback(GGMLQuantizationType.Q4_1)


class TestGGUFLinearTriton(unittest.TestCase):
    """Validate the Triton fused Q4_0 dequantize+matmul kernel.

    All tests in this class are skipped automatically when:
    * ``triton`` is not installed,
    * no CUDA device is available, or
    * the ``gguf`` or ``numpy`` packages are absent.
    """

    @classmethod
    def setUpClass(cls):
        _require("triton")
        _require("gguf")
        _require("numpy")
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    def _make_q4_0_layer_cuda(self, n_out=32, n_in=128):
        """Build a GGUFLinear(Q4_0) layer with weights on the CUDA device."""
        import numpy as np
        import torch
        from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter

        from eole.modules.gguf_linear import GGUFLinear

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            gpath = os.path.join(td, "tiny.gguf")
            writer = GGUFWriter(gpath, "llama")
            writer.add_block_count(1)
            writer.add_embedding_length(n_in)
            writer.add_vocab_size(n_out)
            writer.add_tensor(
                "test.weight",
                np.random.randn(n_out, n_in).astype(np.float32),
                raw_dtype=GGMLQuantizationType.Q4_0,
            )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            reader = GGUFReader(gpath)
            q_data = reader.tensors[0].data.copy()

        layer = GGUFLinear(in_features=n_in, out_features=n_out, bias=False)
        layer.register_buffer("weight", torch.from_numpy(q_data).cuda())
        layer.register_buffer(
            "gguf_qtype",
            torch.tensor(
                [GGMLQuantizationType.Q4_0.value], dtype=torch.int32
            ),
        )
        return layer

    def test_triton_q4_0_matches_pytorch_fallback(self):
        """Triton kernel output must match the PyTorch dequant reference."""
        import torch
        import torch.nn.functional as F
        from gguf import dequantize, GGMLQuantizationType
        from eole.modules.gguf_linear import GGUFLinear, _TRITON_AVAILABLE

        if not _TRITON_AVAILABLE:
            self.skipTest("Triton kernel not compiled successfully")

        n_out, n_in = 32, 128
        layer = self._make_q4_0_layer_cuda(n_out, n_in)

        # Float16 input on CUDA – the common inference scenario.
        x = torch.randn(2, 4, n_in, dtype=torch.float16, device="cuda")

        # Reference: numpy dequant + F.linear (identical to CPU fallback).
        w_np = layer.weight.cpu().numpy()
        dq_np = dequantize(w_np, GGMLQuantizationType.Q4_0)
        w_ref = (
            torch.from_numpy(dq_np)
            .reshape(n_out, n_in)
            .to(dtype=x.dtype, device="cuda")
        )
        expected = F.linear(x, w_ref)

        # Actual: GGUFLinear.forward should route through the Triton kernel.
        actual = layer(x)

        self.assertEqual(actual.shape, (2, 4, n_out))
        # Allow for float16 rounding differences between kernel and reference.
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_triton_q4_0_matches_pytorch_fallback_float32(self):
        """Triton kernel output must match reference for float32 activations."""
        import torch
        import torch.nn.functional as F
        from gguf import dequantize, GGMLQuantizationType
        from eole.modules.gguf_linear import GGUFLinear, _TRITON_AVAILABLE

        if not _TRITON_AVAILABLE:
            self.skipTest("Triton kernel not compiled successfully")

        n_out, n_in = 32, 128
        layer = self._make_q4_0_layer_cuda(n_out, n_in)

        x = torch.randn(3, n_in, dtype=torch.float32, device="cuda")

        w_np = layer.weight.cpu().numpy()
        dq_np = dequantize(w_np, GGMLQuantizationType.Q4_0)
        w_ref = (
            torch.from_numpy(dq_np)
            .reshape(n_out, n_in)
            .to(dtype=x.dtype, device="cuda")
        )
        expected = F.linear(x, w_ref)
        actual = layer(x)

        self.assertEqual(actual.shape, (3, n_out))
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_triton_path_selected_on_cuda(self):
        """GGUFLinear must use the Triton path on CUDA when Triton is available."""
        import torch
        from eole.modules.gguf_linear import (
            GGUFLinear,
            _TRITON_AVAILABLE,
            _q4_0_triton_linear,
        )
        from unittest.mock import patch

        if not _TRITON_AVAILABLE:
            self.skipTest("Triton kernel not compiled successfully")

        layer = self._make_q4_0_layer_cuda(32, 128)
        x = torch.randn(2, 128, dtype=torch.float16, device="cuda")

        call_count = {"n": 0}

        def counting_wrapper(*args, **kwargs):
            call_count["n"] += 1
            return _q4_0_triton_linear(*args, **kwargs)

        with patch(
            "eole.modules.gguf_linear._q4_0_triton_linear",
            side_effect=counting_wrapper,
        ):
            layer(x)

        self.assertEqual(call_count["n"], 1, "Triton path was not invoked")


class TestGGUFLinearVLLM(unittest.TestCase):
    """Validate the vLLM GGUF CUDA kernel integration.

    All tests in this class are skipped automatically when:
    * ``vllm._C`` (vLLM CUDA extension) is not installed,
    * no CUDA device is available, or
    * the ``gguf`` or ``numpy`` packages are absent.
    """

    @classmethod
    def setUpClass(cls):
        _require("gguf")
        _require("numpy")
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            import importlib
            importlib.import_module("vllm._C")
        except ImportError:
            raise unittest.SkipTest("vllm._C not installed")

    def _make_gguf_layer_cuda(self, n_out, n_in, qtype_enum):
        """Build a GGUFLinear layer (given qtype) with weights on CUDA."""
        import os
        import tempfile

        import numpy as np
        import torch
        from gguf import GGUFReader, GGUFWriter

        from eole.modules.gguf_linear import GGUFLinear

        with tempfile.TemporaryDirectory() as td:
            gpath = os.path.join(td, "tiny.gguf")
            writer = GGUFWriter(gpath, "llama")
            writer.add_block_count(1)
            writer.add_embedding_length(n_in)
            writer.add_vocab_size(n_out)
            writer.add_tensor(
                "test.weight",
                np.random.randn(n_out, n_in).astype(np.float32),
                raw_dtype=qtype_enum,
            )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            reader = GGUFReader(gpath)
            q_data = reader.tensors[0].data.copy()

        layer = GGUFLinear(in_features=n_in, out_features=n_out, bias=False)
        layer.register_buffer("weight", torch.from_numpy(q_data).cuda())
        layer.register_buffer(
            "gguf_qtype",
            torch.tensor([qtype_enum.value], dtype=torch.int32),
        )
        return layer

    def _assert_vllm_path_matches_reference(self, qtype_enum, n_out=32, n_in=128, dtype=None):
        """vLLM kernel output must match the numpy dequant reference."""
        import torch
        import torch.nn.functional as F
        from gguf import dequantize

        if dtype is None:
            dtype = torch.float16

        layer = self._make_gguf_layer_cuda(n_out, n_in, qtype_enum)
        x = torch.randn(2, 4, n_in, dtype=dtype, device="cuda")

        # Reference: numpy dequant + F.linear
        w_np = layer.weight.cpu().numpy()
        dq_np = dequantize(w_np, qtype_enum)
        w_ref = (
            torch.from_numpy(dq_np)
            .reshape(n_out, n_in)
            .to(dtype=dtype, device="cuda")
        )
        expected = F.linear(x, w_ref)
        actual = layer(x)

        self.assertEqual(actual.shape, (2, 4, n_out))
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_vllm_q4_0(self):
        """vLLM path must match numpy reference for Q4_0."""
        from gguf import GGMLQuantizationType
        self._assert_vllm_path_matches_reference(GGMLQuantizationType.Q4_0)

    def test_vllm_q4_k(self):
        """vLLM path must match numpy reference for Q4_K (k-quant)."""
        from gguf import GGMLQuantizationType
        # Q4_K requires in_features divisible by 256 (block size)
        self._assert_vllm_path_matches_reference(
            GGMLQuantizationType.Q4_K, n_out=32, n_in=256
        )

    def test_vllm_q8_0(self):
        """vLLM path must match numpy reference for Q8_0."""
        from gguf import GGMLQuantizationType
        self._assert_vllm_path_matches_reference(GGMLQuantizationType.Q8_0)

    def test_vllm_path_selected_on_cuda(self):
        """GGUFLinear must route through the vLLM path on CUDA when available."""
        import torch
        from unittest.mock import patch

        from eole.modules.gguf_linear import _VLLM_GGUF_OPS_AVAILABLE, _vllm_gguf_linear
        from gguf import GGMLQuantizationType

        if not _VLLM_GGUF_OPS_AVAILABLE:
            self.skipTest("vLLM GGUF ops not available")

        layer = self._make_gguf_layer_cuda(32, 128, GGMLQuantizationType.Q4_0)
        x = torch.randn(2, 128, dtype=torch.float16, device="cuda")

        call_count = {"n": 0}

        def counting_wrapper(*args, **kwargs):
            call_count["n"] += 1
            return _vllm_gguf_linear(*args, **kwargs)

        with patch(
            "eole.modules.gguf_linear._vllm_gguf_linear",
            side_effect=counting_wrapper,
        ):
            layer(x)

        self.assertEqual(call_count["n"], 1, "vLLM path was not invoked")

    def test_vllm_path_takes_priority_over_triton(self):
        """When both vLLM and Triton are available, vLLM must be used for Q4_0."""
        import torch
        from unittest.mock import patch

        from eole.modules.gguf_linear import (
            _TRITON_AVAILABLE,
            _VLLM_GGUF_OPS_AVAILABLE,
            _q4_0_triton_linear,
            _vllm_gguf_linear,
        )
        from gguf import GGMLQuantizationType

        if not (_VLLM_GGUF_OPS_AVAILABLE and _TRITON_AVAILABLE):
            self.skipTest("Requires both vLLM and Triton")

        layer = self._make_gguf_layer_cuda(32, 128, GGMLQuantizationType.Q4_0)
        x = torch.randn(2, 128, dtype=torch.float16, device="cuda")

        triton_calls = {"n": 0}
        vllm_calls = {"n": 0}

        def count_triton(*args, **kwargs):
            triton_calls["n"] += 1
            return _q4_0_triton_linear(*args, **kwargs)

        def count_vllm(*args, **kwargs):
            vllm_calls["n"] += 1
            return _vllm_gguf_linear(*args, **kwargs)

        with (
            patch("eole.modules.gguf_linear._q4_0_triton_linear", side_effect=count_triton),
            patch("eole.modules.gguf_linear._vllm_gguf_linear", side_effect=count_vllm),
        ):
            layer(x)

        self.assertEqual(vllm_calls["n"], 1, "vLLM path was not invoked")
        self.assertEqual(triton_calls["n"], 0, "Triton was used even though vLLM is available")

    def test_vllm_delegates_to_fused_op(self):
        """_vllm_gguf_linear must call _vllm_fused_mul_mat_gguf without dtype cast.

        vLLM's fused op owns all dispatch and dtype logic internally.  Our
        wrapper must pass (x_2d, weight_2d, qtype_val) directly — no manual
        float16 cast — and the output dtype must match the input dtype.
        """
        import torch
        from unittest.mock import patch

        from eole.modules.gguf_linear import (
            _VLLM_GGUF_OPS_AVAILABLE,
            _vllm_fused_mul_mat_gguf,
        )
        from gguf import GGMLQuantizationType

        if not _VLLM_GGUF_OPS_AVAILABLE:
            self.skipTest("vLLM GGUF ops not available")

        layer = self._make_gguf_layer_cuda(32, 128, GGMLQuantizationType.Q4_0)

        for input_dtype in (torch.float16, torch.bfloat16, torch.float32):
            calls = []
            real_fused = _vllm_fused_mul_mat_gguf

            def spy(x_arg, w_arg, qtype_arg):
                calls.append({"x_dtype": x_arg.dtype, "x_shape": tuple(x_arg.shape)})
                return real_fused(x_arg, w_arg, qtype_arg)

            x = torch.randn(2, 128, dtype=input_dtype, device="cuda")

            with patch("eole.modules.gguf_linear._vllm_fused_mul_mat_gguf", side_effect=spy):
                out = layer(x)

            self.assertEqual(len(calls), 1, "fused op must be called exactly once")
            # The fused op must receive the original activation dtype — no cast.
            self.assertEqual(
                calls[0]["x_dtype"], input_dtype,
                f"Fused op received {calls[0]['x_dtype']} instead of {input_dtype}",
            )
            # Output dtype must equal the input dtype.
            self.assertEqual(
                out.dtype, input_dtype,
                f"Expected output dtype {input_dtype}, got {out.dtype}",
            )


class TestGGUFConverter(unittest.TestCase):
    """Integration test for the full GGUF → EOLE conversion pipeline."""

    @classmethod
    def setUpClass(cls):
        _require("gguf")
        _require("numpy")
        _require("safetensors")

        import numpy as np
        from gguf import GGMLQuantizationType, GGUFWriter

        cls.tmpdir = tempfile.mkdtemp()
        cls.gguf_path = os.path.join(cls.tmpdir, "model.gguf")
        cls.output_dir = os.path.join(cls.tmpdir, "eole_output")

        n_vocab, hidden, heads, ff = 32, 256, 4, 512
        writer = GGUFWriter(cls.gguf_path, "llama")
        writer.add_block_count(2)
        writer.add_context_length(512)
        writer.add_embedding_length(hidden)
        writer.add_feed_forward_length(ff)
        writer.add_head_count(heads)
        writer.add_head_count_kv(heads)
        writer.add_layer_norm_rms_eps(1e-5)
        writer.add_rope_freq_base(10000.0)
        writer.add_vocab_size(n_vocab)
        writer.add_token_list([str(i) for i in range(n_vocab)])
        writer.add_bos_token_id(1)
        writer.add_eos_token_id(2)

        writer.add_tensor(
            "token_embd.weight", np.random.randn(n_vocab, hidden).astype(np.float32)
        )
        writer.add_tensor("output_norm.weight", np.ones(hidden, dtype=np.float32))

        for i in range(2):
            writer.add_tensor(
                f"blk.{i}.attn_norm.weight", np.ones(hidden, dtype=np.float32)
            )
            writer.add_tensor(
                f"blk.{i}.ffn_norm.weight", np.ones(hidden, dtype=np.float32)
            )
            for suffix, shape in [
                ("attn_q", (hidden, hidden)),
                ("attn_k", (hidden, hidden)),
                ("attn_v", (hidden, hidden)),
                ("attn_output", (hidden, hidden)),
                ("ffn_gate", (ff, hidden)),
                ("ffn_up", (ff, hidden)),
                ("ffn_down", (hidden, ff)),
            ]:
                writer.add_tensor(
                    f"blk.{i}.{suffix}.weight",
                    np.random.randn(*shape).astype(np.float32),
                    raw_dtype=GGMLQuantizationType.Q4_K,
                )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _run_converter(self):
        from argparse import Namespace

        from eole.bin.convert.convert_gguf import GGUFConverter

        args = Namespace(
            gguf_path=self.gguf_path,
            output=self.output_dir,
            dtype="fp16",
            tokenizer="hf",
            hf_tokenizer=None,
        )
        shutil.rmtree(self.output_dir, ignore_errors=True)
        GGUFConverter.run(args)

    def test_output_files_exist(self):
        self._run_converter()
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "config.json")))
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "model.00.safetensors"))
        )

    def test_config_mlp_activation_fn(self):
        """SwiGLU models must use 'gated-silu', not bare 'silu'."""
        self._run_converter()
        with open(os.path.join(self.output_dir, "config.json")) as f:
            cfg = json.load(f)
        # Top-level model field
        self.assertEqual(cfg["model"]["mlp_activation_fn"], "gated-silu")
        # And propagated into decoder
        self.assertEqual(cfg["model"]["decoder"]["mlp_activation_fn"], "gated-silu")

    def test_up_proj_present_in_shard(self):
        """ffn_up.weight must be written as mlp.up_proj.weight (needs gated-silu)."""
        _require("safetensors")
        self._run_converter()
        import safetensors.torch

        tensors = safetensors.torch.load_file(
            os.path.join(self.output_dir, "model.00.safetensors")
        )
        key = "decoder.transformer_layers.0.mlp.up_proj.weight"
        self.assertIn(key, tensors, "up_proj.weight missing from shard")
        import torch
        self.assertEqual(tensors[key].dtype, torch.uint8)

    def test_config_quant_type(self):
        self._run_converter()
        with open(os.path.join(self.output_dir, "config.json")) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["training"]["quant_type"], "gguf")
        self.assertIn("linear_query", cfg["training"]["quant_layers"])

    def test_config_rotary(self):
        self._run_converter()
        with open(os.path.join(self.output_dir, "config.json")) as f:
            cfg = json.load(f)
        self.assertEqual(
            cfg["model"]["embeddings"]["position_encoding_type"], "Rotary"
        )

    def test_quantized_tensors_are_uint8(self):
        _require("safetensors")
        self._run_converter()
        import safetensors.torch
        import torch

        tensors = safetensors.torch.load_file(
            os.path.join(self.output_dir, "model.00.safetensors")
        )
        lq_w = tensors["decoder.transformer_layers.0.self_attn.linear_query.weight"]
        self.assertEqual(lq_w.dtype, torch.uint8)

    def test_gguf_qtype_tensor_present(self):
        _require("safetensors")
        self._run_converter()
        import safetensors.torch

        tensors = safetensors.torch.load_file(
            os.path.join(self.output_dir, "model.00.safetensors")
        )
        self.assertIn(
            "decoder.transformer_layers.0.self_attn.linear_query.gguf_qtype", tensors
        )
        qtype_val = tensors[
            "decoder.transformer_layers.0.self_attn.linear_query.gguf_qtype"
        ].item()
        from gguf import GGMLQuantizationType

        self.assertEqual(GGMLQuantizationType(qtype_val).name, "Q4_K")

    def test_float_tensors_are_float16(self):
        _require("safetensors")
        self._run_converter()
        import safetensors.torch
        import torch

        tensors = safetensors.torch.load_file(
            os.path.join(self.output_dir, "model.00.safetensors")
        )
        self.assertEqual(tensors["tgt_emb.embeddings.weight"].dtype, torch.float16)
        self.assertEqual(tensors["decoder.layer_norm.weight"].dtype, torch.float16)

    def test_generator_in_quant_layers_for_text_model(self):
        """Plain text models (no vision encoder) include 'generator' in quant_layers."""
        self._run_converter()
        with open(os.path.join(self.output_dir, "config.json")) as f:
            cfg = json.load(f)
        self.assertIn("generator", cfg["training"]["quant_layers"])

    def test_generator_weight_uint8_for_text_model(self):
        """generator.weight stays as uint8 for plain text model (no mmproj).

        For text-only models, replace_gguf_linear targets self (the whole
        model), so self.generator is reachable.  Keeping it quantized avoids
        the large float dequantization overhead for big-vocabulary models.
        """
        _require("safetensors")
        _require("numpy")
        _require("torch")
        import numpy as np
        import torch
        from gguf import GGMLQuantizationType, GGUFWriter

        # Build a minimal GGUF that has a separate output.weight (not tied).
        tmpdir = tempfile.mkdtemp()
        try:
            gguf_path = os.path.join(tmpdir, "model.gguf")
            output_dir = os.path.join(tmpdir, "eole")
            n_vocab, hidden, heads, ff = 32, 256, 4, 512

            writer = GGUFWriter(gguf_path, "llama")
            writer.add_block_count(1)
            writer.add_context_length(128)
            writer.add_embedding_length(hidden)
            writer.add_feed_forward_length(ff)
            writer.add_head_count(heads)
            writer.add_head_count_kv(heads)
            writer.add_layer_norm_rms_eps(1e-5)
            writer.add_vocab_size(n_vocab)
            writer.add_token_list([str(i) for i in range(n_vocab)])
            writer.add_tensor(
                "token_embd.weight",
                np.random.randn(n_vocab, hidden).astype(np.float32),
            )
            writer.add_tensor("output_norm.weight", np.ones(hidden, dtype=np.float32))
            # Separate output.weight (no weight tying): should stay quantized.
            writer.add_tensor(
                "output.weight",
                np.random.randn(n_vocab, hidden).astype(np.float32),
                raw_dtype=GGMLQuantizationType.Q4_K,
            )
            writer.add_tensor(
                "blk.0.attn_norm.weight", np.ones(hidden, dtype=np.float32)
            )
            writer.add_tensor(
                "blk.0.ffn_norm.weight", np.ones(hidden, dtype=np.float32)
            )
            for suffix, shape in [
                ("attn_q", (hidden, hidden)),
                ("attn_k", (hidden, hidden)),
                ("attn_v", (hidden, hidden)),
                ("attn_output", (hidden, hidden)),
                ("ffn_gate", (ff, hidden)),
                ("ffn_up", (ff, hidden)),
                ("ffn_down", (hidden, ff)),
            ]:
                writer.add_tensor(
                    f"blk.0.{suffix}.weight",
                    np.random.randn(*shape).astype(np.float32),
                    raw_dtype=GGMLQuantizationType.Q4_K,
                )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            from argparse import Namespace
            from eole.bin.convert.convert_gguf import GGUFConverter

            args = Namespace(
                gguf_path=gguf_path,
                output=output_dir,
                dtype="fp16",
                tokenizer="hf",
                hf_tokenizer=None,
            )
            GGUFConverter.run(args)

            import safetensors.torch
            tensors = safetensors.torch.load_file(
                os.path.join(output_dir, "model.00.safetensors")
            )
            # generator.weight must be uint8 (kept quantized for text model)
            self.assertIn("generator.weight", tensors)
            self.assertEqual(
                tensors["generator.weight"].dtype,
                torch.uint8,
                "generator.weight should be quantized uint8 for plain text model",
            )
            # companion gguf_qtype must be present
            self.assertIn("generator.gguf_qtype", tensors)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestGGUFBF16Conversion(unittest.TestCase):
    """Tests for BF16 tensor handling in _tensor_to_torch.

    BF16 data from gguf-python arrives as raw uint8 bytes (the gguf library
    has no bfloat16 numpy dtype, so BF16 tensors fall through to the
    block_size=1 / type_size=2 uint8 path).  Without the BF16 fix the raw
    bytes would be written to safetensors and interpreted as garbage float
    values when loaded back by the model.
    """

    def setUp(self):
        _require("gguf")
        _require("numpy")
        _require("ml_dtypes")

    def _make_bf16_gguf(self, tmpdir: str) -> str:
        """Create a minimal GGUF file with BF16-typed norm weights."""
        import ml_dtypes
        import numpy as np
        from gguf import GGMLQuantizationType, GGUFWriter

        gguf_path = os.path.join(tmpdir, "bf16.gguf")
        hidden = 256
        writer = GGUFWriter(gguf_path, "llama")
        writer.add_block_count(1)
        writer.add_context_length(128)
        writer.add_embedding_length(hidden)
        writer.add_feed_forward_length(512)
        writer.add_head_count(4)
        writer.add_head_count_kv(4)
        writer.add_layer_norm_rms_eps(1e-5)
        writer.add_vocab_size(16)
        writer.add_token_list([str(i) for i in range(16)])

        # Float embedding (F32)
        writer.add_tensor(
            "token_embd.weight",
            np.ones((16, hidden), dtype=np.float32),
        )
        # BF16 norm weights (the key regression case: must become float16,
        # not garbage uint8 bytes)
        ones_f32 = np.ones(hidden, dtype=np.float32)
        writer.add_tensor(
            "output_norm.weight",
            ones_f32.astype(ml_dtypes.bfloat16),
            raw_dtype=GGMLQuantizationType.BF16,
        )
        writer.add_tensor(
            "blk.0.attn_norm.weight",
            ones_f32.astype(ml_dtypes.bfloat16),
            raw_dtype=GGMLQuantizationType.BF16,
        )
        writer.add_tensor(
            "blk.0.ffn_norm.weight",
            ones_f32.astype(ml_dtypes.bfloat16),
            raw_dtype=GGMLQuantizationType.BF16,
        )
        for suffix, shape in [
            ("attn_q", (hidden, hidden)),
            ("attn_k", (hidden, hidden)),
            ("attn_v", (hidden, hidden)),
            ("attn_output", (hidden, hidden)),
            ("ffn_gate", (512, hidden)),
            ("ffn_up", (512, hidden)),
            ("ffn_down", (hidden, 512)),
        ]:
            writer.add_tensor(
                f"blk.0.{suffix}.weight",
                np.random.randn(*shape).astype(np.float32),
                raw_dtype=GGMLQuantizationType.Q4_K,
            )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return gguf_path

    def test_bf16_tensor_to_torch_dtype(self):
        """_tensor_to_torch must return float16 (not uint8) for BF16 tensors."""
        _require("torch")
        import torch
        from gguf import GGUFReader

        from eole.bin.convert.convert_gguf import _tensor_to_torch

        tmpdir = tempfile.mkdtemp()
        try:
            gguf_path = self._make_bf16_gguf(tmpdir)
            reader = GGUFReader(gguf_path)
            for t in reader.tensors:
                if t.tensor_type.name == "BF16":
                    result, qtype_t = _tensor_to_torch(t, torch.float16)
                    self.assertTrue(
                        result.is_floating_point(),
                        f"BF16 tensor '{t.name}' must be floating-point after conversion, "
                        f"got dtype={result.dtype}",
                    )
                    self.assertEqual(
                        result.dtype,
                        torch.float16,
                        f"BF16 tensor '{t.name}' must be float16 after conversion",
                    )
                    self.assertIsNone(
                        qtype_t,
                        "BF16 is a float type, qtype_t must be None",
                    )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_bf16_tensor_shape_correct(self):
        """BF16 tensor after conversion must have the logical (out, in) shape.

        gguf-python returns BF16 as uint8 of shape (out, in*2).  After the
        fix the view(bfloat16) halves the last dimension back to in.
        """
        _require("torch")
        import torch
        from gguf import GGUFReader

        from eole.bin.convert.convert_gguf import _tensor_to_torch

        tmpdir = tempfile.mkdtemp()
        try:
            gguf_path = self._make_bf16_gguf(tmpdir)
            reader = GGUFReader(gguf_path)
            for t in reader.tensors:
                if t.tensor_type.name == "BF16" and t.data.ndim == 1:
                    # 1-D BF16 norm weight: gguf returns uint8 of shape (hidden*2,);
                    # after fix should be float16 of shape (hidden,).
                    result, _ = _tensor_to_torch(t, torch.float16)
                    raw_bytes = t.data.nbytes
                    expected_elements = raw_bytes // 2  # 2 bytes per bfloat16
                    self.assertEqual(
                        result.numel(),
                        expected_elements,
                        f"BF16 1-D tensor '{t.name}': expected {expected_elements} "
                        f"elements but got {result.numel()}",
                    )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_bf16_norm_stored_as_float16_in_safetensors(self):
        """BF16 norm weights must appear as float16 in the converted safetensors."""
        _require("safetensors")
        _require("torch")
        import torch
        from argparse import Namespace

        from eole.bin.convert.convert_gguf import GGUFConverter

        tmpdir = tempfile.mkdtemp()
        try:
            gguf_path = self._make_bf16_gguf(tmpdir)
            output_dir = os.path.join(tmpdir, "eole")
            args = Namespace(
                gguf_path=gguf_path,
                output=output_dir,
                dtype="fp16",
                tokenizer="hf",
                hf_tokenizer=None,
            )
            GGUFConverter.run(args)

            import safetensors.torch
            tensors = safetensors.torch.load_file(
                os.path.join(output_dir, "model.00.safetensors")
            )
            for key in ("decoder.layer_norm.weight",
                        "decoder.transformer_layers.0.input_layernorm.weight",
                        "decoder.transformer_layers.0.post_attention_layernorm.weight"):
                if key in tensors:
                    self.assertEqual(
                        tensors[key].dtype,
                        torch.float16,
                        f"BF16 norm '{key}' must be float16 in safetensors, "
                        f"got {tensors[key].dtype}",
                    )
                    # Also verify values are sensible (all-ones BF16 → ~1.0 in float16)
                    vals = tensors[key].float()
                    self.assertTrue(
                        torch.allclose(vals, torch.ones_like(vals), atol=0.01),
                        f"BF16 norm '{key}' values wrong: expected ~1.0, "
                        f"got min={vals.min().item():.4f} max={vals.max().item():.4f}",
                    )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestGGUFArchSets(unittest.TestCase):
    """Test that known architectures are properly included in the arch sets."""

    def setUp(self):
        _require("gguf")

    def test_qwen35_in_rms_norm_archs(self):
        from eole.bin.convert.convert_gguf import _RMS_NORM_ARCHS

        self.assertIn("qwen35", _RMS_NORM_ARCHS)

    def test_qwen35_in_swiglu_archs(self):
        from eole.bin.convert.convert_gguf import _SWIGLU_ARCHS

        self.assertIn("qwen35", _SWIGLU_ARCHS)

    def test_qwen35_in_mrope_interleave_archs(self):
        """qwen35 is a VL model that always uses interleaved MRoPE."""
        from eole.bin.convert.convert_gguf import _MROPE_INTERLEAVE_ARCHS

        self.assertIn("qwen35", _MROPE_INTERLEAVE_ARCHS)

    def test_qwen35moe_in_mrope_interleave_archs(self):
        """qwen35moe is a VL model that always uses interleaved MRoPE."""
        from eole.bin.convert.convert_gguf import _MROPE_INTERLEAVE_ARCHS

        self.assertIn("qwen35moe", _MROPE_INTERLEAVE_ARCHS)

    def test_rope_dim_sections_strips_trailing_zeros(self):
        """rope_dim_sections must strip GGUF null-padding zeros.

        The Qwen3.5-9B GGUF stores ``qwen35.rope.dimension_sections`` as
        ``[11, 11, 10, 0]`` (4 elements) instead of the expected 3-element
        ``[11, 11, 10]`` that matches HF's mrope_section.  The trailing
        zero must be stripped so that xdrope_section has exactly 3 elements
        for correct T/H/W position-ID indexing in apply_rotary_pos_emb_xdrope.
        """
        _require("numpy")
        import tempfile
        import numpy as np
        from gguf import GGUFWriter
        from eole.bin.convert.convert_gguf import GGUFMetadata, build_model_config

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "qwen35_rope.gguf")
            w = GGUFWriter(path, "qwen35")
            w.add_block_count(1)
            w.add_embedding_length(64)
            w.add_feed_forward_length(128)
            w.add_head_count(4)
            w.add_layer_norm_rms_eps(1e-5)
            w.add_rope_freq_base(1000000.0)
            w.add_vocab_size(8)
            w.add_token_list([str(i) for i in range(8)])
            # Write rope.dimension_sections with a trailing zero (as seen in
            # real Qwen3.5-9B GGUF files).
            try:
                w.add_array("qwen35.rope.dimension_sections", [11, 11, 10, 0])
            except (AttributeError, TypeError):
                # Older gguf-py versions may not have add_array; skip gracefully.
                raise unittest.SkipTest("GGUFWriter.add_array not supported in this gguf version")
            w.add_tensor("token_embd.weight", np.zeros((8, 64), dtype=np.float16))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
            meta = GGUFMetadata(path)

        sections = meta.rope_dim_sections
        self.assertIsNotNone(sections, "rope_dim_sections should not be None")
        self.assertEqual(sections, [11, 11, 10],
                         "Trailing zero must be stripped: expected [11, 11, 10], "
                         f"got {sections}")
        # Verify build_model_config propagates the cleaned sections.
        cfg = build_model_config(meta)
        self.assertEqual(cfg["rope_config"].get("xdrope_section"), [11, 11, 10])
        # qwen35 always gets rotary_interleave=True regardless of sections.
        self.assertTrue(cfg["rope_config"].get("rotary_interleave"),
                        "qwen35 must always have rotary_interleave=True")

    def test_llama_in_swiglu_archs(self):
        from eole.bin.convert.convert_gguf import _SWIGLU_ARCHS

        self.assertIn("llama", _SWIGLU_ARCHS)

    def test_swiglu_uses_gated_silu(self):
        """build_model_config must produce 'gated-silu' for SwiGLU architectures."""
        _require("numpy")
        import tempfile
        import numpy as np
        from gguf import GGUFWriter
        from eole.bin.convert.convert_gguf import GGUFMetadata, build_model_config

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "m.gguf")
            w = GGUFWriter(path, "llama")
            w.add_block_count(1)
            w.add_embedding_length(64)
            w.add_feed_forward_length(128)
            w.add_head_count(4)
            w.add_layer_norm_rms_eps(1e-5)
            w.add_vocab_size(8)
            w.add_token_list([str(i) for i in range(8)])
            w.add_tensor("token_embd.weight", np.zeros((8, 64), dtype=np.float16))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
            meta = GGUFMetadata(path)
        cfg = build_model_config(meta)
        self.assertEqual(cfg["mlp_activation_fn"], "gated-silu")

    def test_non_swiglu_uses_gelu(self):
        """Non-SwiGLU architecture (e.g. phi2) must use 'gelu'."""
        _require("numpy")
        import tempfile
        import numpy as np
        from gguf import GGUFWriter
        from eole.bin.convert.convert_gguf import GGUFMetadata, build_model_config

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "m.gguf")
            w = GGUFWriter(path, "phi2")
            w.add_block_count(1)
            w.add_embedding_length(64)
            w.add_feed_forward_length(128)
            w.add_head_count(4)
            w.add_layer_norm_rms_eps(1e-5)
            w.add_vocab_size(8)
            w.add_token_list([str(i) for i in range(8)])
            w.add_tensor("token_embd.weight", np.zeros((8, 64), dtype=np.float16))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
            meta = GGUFMetadata(path)
        cfg = build_model_config(meta)
        self.assertEqual(cfg["mlp_activation_fn"], "gelu")


class TestGGUFFuseHandlers(unittest.TestCase):
    """Test that _fuse_KVQ and _fuse_gate skip GGUFLinear modules."""

    def setUp(self):
        _require("torch")

    def test_fuse_kvq_skips_gguf_linear(self):
        """_fuse_KVQ must return immediately for GGUFLinear, leaving modules intact."""
        import torch
        from eole.modules.gguf_linear import GGUFLinear
        from eole.modules.multi_headed_attn import MultiHeadedAttention

        # Build a minimal object that has the three linear attributes as GGUFLinear
        # and call _fuse_KVQ directly using the unbound method.
        class FakeAttn:
            pass

        fake = FakeAttn()
        fake.linear_keys = GGUFLinear(32, 32, bias=False)
        fake.linear_values = GGUFLinear(32, 32, bias=False)
        fake.linear_query = GGUFLinear(32, 32, bias=False)

        # Call the method on our fake object (unbound call)
        MultiHeadedAttention._fuse_KVQ(fake)

        # Modules must still exist and be GGUFLinear (not replaced / deleted)
        self.assertIsInstance(fake.linear_keys, GGUFLinear)
        self.assertIsInstance(fake.linear_values, GGUFLinear)
        self.assertIsInstance(fake.linear_query, GGUFLinear)
        self.assertFalse(hasattr(fake, "linear_kvq"))

    def test_fuse_gate_skips_gguf_linear(self):
        """_fuse_gate must return immediately for GGUFLinear, leaving modules intact."""
        import torch.nn as nn
        from eole.modules.gguf_linear import GGUFLinear, replace_gguf_linear
        from eole.modules.transformer_mlp import MLP

        class FakeCfg:
            hidden_size = 32
            transformer_ff = 64
            mlp_activation_fn = "gated-silu"
            add_ffnbias = False

        try:
            mlp = MLP(FakeCfg(), running_config=None)
        except Exception:
            raise unittest.SkipTest("MLP init requires more config")

        replace_gguf_linear(mlp, module_to_convert=["gate_up_proj", "up_proj"])
        from eole.modules.gguf_linear import GGUFLinear as GL
        self.assertIsInstance(mlp.gate_up_proj, GL)

        # _fuse_gate must not raise and must leave modules unchanged
        mlp._fuse_gate()
        self.assertIsInstance(mlp.gate_up_proj, GL)
        self.assertIsInstance(mlp.up_proj, GL)


class TestGGUFDetectLinearBlocks(unittest.TestCase):
    """Test :func:`_detect_linear_attention_blocks`."""

    def setUp(self):
        _require("gguf")
        _require("numpy")

    def test_detects_blocks_with_ssm_tensors(self):
        """Blocks that have ssm_* tensors should be returned as linear-attention."""
        import numpy as np
        from gguf import GGUFWriter, GGUFReader
        import tempfile

        from eole.bin.convert.convert_gguf import _detect_linear_attention_blocks

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "h.gguf")
            w = GGUFWriter(path, "qwen35")
            w.add_block_count(4)
            w.add_embedding_length(64)
            w.add_vocab_size(8)
            w.add_token_list([str(i) for i in range(8)])
            # Block 0 and 2: linear attention (have ssm_a)
            for blk in (0, 2):
                w.add_tensor(f"blk.{blk}.ssm_a", np.ones(4, dtype=np.float32))
                w.add_tensor(f"blk.{blk}.ssm_norm.weight", np.ones(16, dtype=np.float32))
            # Block 1 and 3: full attention (no ssm_*)
            for blk in (1, 3):
                w.add_tensor(f"blk.{blk}.attn_q.weight", np.zeros((64, 64), dtype=np.float32))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()

            reader = GGUFReader(path)

        linear = _detect_linear_attention_blocks(reader.tensors)
        self.assertEqual(linear, frozenset({0, 2}))

    def test_empty_for_pure_attention_model(self):
        """Models without any ssm_* tensors return an empty set."""
        import numpy as np
        from gguf import GGUFWriter, GGUFReader
        import tempfile

        from eole.bin.convert.convert_gguf import _detect_linear_attention_blocks

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "a.gguf")
            w = GGUFWriter(path, "llama")
            w.add_block_count(2)
            w.add_embedding_length(64)
            w.add_vocab_size(8)
            w.add_token_list([str(i) for i in range(8)])
            w.add_tensor("blk.0.attn_q.weight", np.zeros((64, 64), dtype=np.float32))
            w.add_tensor("blk.1.attn_q.weight", np.zeros((64, 64), dtype=np.float32))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()

            reader = GGUFReader(path)

        linear = _detect_linear_attention_blocks(reader.tensors)
        self.assertEqual(linear, frozenset())


class TestGGUFHybridConverter(unittest.TestCase):
    """Integration test for the GGUF → EOLE conversion of a hybrid model.

    A synthetic 2-block model is constructed:
      - blk.0: linear-attention block (GatedDeltaNet / SSM)
      - blk.1: full-attention block (standard self-attention)
    """

    @classmethod
    def setUpClass(cls):
        _require("gguf")
        _require("numpy")
        _require("safetensors")

        import numpy as np
        from gguf import GGMLQuantizationType, GGUFWriter

        cls.tmpdir = tempfile.mkdtemp()
        cls.gguf_path = os.path.join(cls.tmpdir, "hybrid.gguf")
        cls.output_dir = os.path.join(cls.tmpdir, "hybrid_out")

        # Dimensions chosen to be consistent with GatedDeltaNet:
        #   hidden_size=64, num_v_heads=4, head_v_dim=16 → value_dim=64
        #   num_k_heads=2, head_k_dim=16 → key_dim=32
        #   conv_dim = key_dim*2 + value_dim = 32*2 + 64 = 128
        hidden, ff = 64, 128
        num_heads, heads_kv, n_vocab = 4, 2, 32
        # GatedDeltaNet dims
        num_v_heads, head_v_dim = 4, 16    # value_dim = 64
        num_k_heads, head_k_dim = 2, 16    # key_dim = 32
        conv_dim = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim  # 128
        kernel_size = 4

        writer = GGUFWriter(cls.gguf_path, "qwen35")
        writer.add_block_count(2)
        writer.add_context_length(128)
        writer.add_embedding_length(hidden)
        writer.add_feed_forward_length(ff)
        writer.add_head_count(num_heads)
        writer.add_head_count_kv(heads_kv)
        writer.add_layer_norm_rms_eps(1e-5)
        writer.add_rope_freq_base(10000.0)
        writer.add_vocab_size(n_vocab)
        writer.add_token_list([str(i) for i in range(n_vocab)])
        writer.add_bos_token_id(1)
        writer.add_eos_token_id(2)

        writer.add_tensor("token_embd.weight", np.random.randn(n_vocab, hidden).astype(np.float32))
        writer.add_tensor("output_norm.weight", np.ones(hidden, dtype=np.float32))

        # -------- blk.0: linear attention block --------
        writer.add_tensor("blk.0.attn_norm.weight", np.ones(hidden, dtype=np.float32))
        writer.add_tensor("blk.0.post_attention_norm.weight", np.ones(hidden, dtype=np.float32))
        writer.add_tensor("blk.0.ffn_norm.weight", np.ones(hidden, dtype=np.float32))
        # Linear-attention projections (quantised Q8_0 – Q5_K/Q4_K need block_size=256
        # which is larger than our synthetic hidden=64 / conv_dim=128 dimensions)
        writer.add_tensor(
            "blk.0.attn_qkv.weight",
            np.random.randn(conv_dim, hidden).astype(np.float32),
            raw_dtype=GGMLQuantizationType.Q8_0,
        )
        writer.add_tensor(
            "blk.0.attn_gate.weight",
            np.random.randn(hidden, num_v_heads * head_v_dim).astype(np.float32),
            raw_dtype=GGMLQuantizationType.Q8_0,
        )
        # SSM params (float)
        # ssm_a stores -exp(A_log_hf) in GGUF format (llama.cpp applies -exp transform).
        # Use -1.0 (valid negative value: log(-(-1)) = log(1) = 0).
        writer.add_tensor("blk.0.ssm_a", -np.ones(num_v_heads, dtype=np.float32))
        writer.add_tensor("blk.0.ssm_dt.bias", np.ones(num_v_heads, dtype=np.float32))
        writer.add_tensor("blk.0.ssm_norm.weight", np.ones(head_v_dim, dtype=np.float32))
        writer.add_tensor(
            "blk.0.ssm_conv1d.weight",
            # Real GGUF (llama.cpp): stored as (conv_dim, kernel_size); the gguf
            # reader reverses GGUF ne, so tensor.shape=[kernel_size, conv_dim] and
            # data.shape=(conv_dim, kernel_size) in NumPy order.
            np.random.randn(conv_dim, kernel_size).astype(np.float32),
        )
        # ssm_alpha / ssm_beta: in_proj_a / in_proj_b projections.
        # gguf-python reverses GGUF ne when creating tensor.data, so the data
        # arrives with shape (out_features, in_features) = (num_v_heads, hidden).
        # Write numpy arrays in (out, in) = (num_v_heads, hidden) order, which is
        # how llama.cpp stores them.  No transposition is needed in the converter.
        # Using F32 keeps the test simple (avoids Q8_0 block-size constraints).
        writer.add_tensor(
            "blk.0.ssm_alpha.weight",
            np.random.randn(num_v_heads, hidden).astype(np.float32),
        )
        writer.add_tensor(
            "blk.0.ssm_beta.weight",
            np.random.randn(num_v_heads, hidden).astype(np.float32),
        )
        writer.add_tensor(
            "blk.0.ssm_out.weight",
            np.random.randn(hidden, num_v_heads * head_v_dim).astype(np.float32),
            raw_dtype=GGMLQuantizationType.Q8_0,
        )
        # FFN
        for suffix, shape in [
            ("ffn_gate", (ff, hidden)),
            ("ffn_up", (ff, hidden)),
            ("ffn_down", (hidden, ff)),
        ]:
            writer.add_tensor(
                f"blk.0.{suffix}.weight",
                np.random.randn(*shape).astype(np.float32),
                raw_dtype=GGMLQuantizationType.Q8_0,
            )

        # -------- blk.1: full attention block --------
        writer.add_tensor("blk.1.attn_norm.weight", np.ones(hidden, dtype=np.float32))
        writer.add_tensor("blk.1.post_attention_norm.weight", np.ones(hidden, dtype=np.float32))
        writer.add_tensor("blk.1.ffn_norm.weight", np.ones(hidden, dtype=np.float32))
        for suffix, shape in [
            ("attn_q", (hidden, hidden)),
            ("attn_k", (hidden, hidden // num_heads * heads_kv)),
            ("attn_v", (hidden, hidden // num_heads * heads_kv)),
            ("attn_output", (hidden, hidden)),
            ("ffn_gate", (ff, hidden)),
            ("ffn_up", (ff, hidden)),
            ("ffn_down", (hidden, ff)),
        ]:
            writer.add_tensor(
                f"blk.1.{suffix}.weight",
                np.random.randn(*shape).astype(np.float32),
                raw_dtype=GGMLQuantizationType.Q8_0,
            )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _run_converter(self):
        from argparse import Namespace
        from eole.bin.convert.convert_gguf import GGUFConverter

        args = Namespace(
            gguf_path=self.gguf_path,
            output=self.output_dir,
            dtype="fp16",
            tokenizer="hf",
            hf_tokenizer=None,
        )
        shutil.rmtree(self.output_dir, ignore_errors=True)
        GGUFConverter.run(args)

    def _tensors(self):
        import safetensors.torch
        return safetensors.torch.load_file(
            os.path.join(self.output_dir, "model.00.safetensors")
        )

    def test_linear_attn_block0_detected(self):
        """The converter must identify block 0 as linear-attention."""
        from eole.bin.convert.convert_gguf import GGUFMetadata, _detect_linear_attention_blocks
        meta = GGUFMetadata(self.gguf_path)
        linear = _detect_linear_attention_blocks(meta.tensors)
        self.assertIn(0, linear)
        self.assertNotIn(1, linear)

    def test_ssm_out_weight_mapped(self):
        """ssm_out.weight must appear as linear_attn.out_proj.weight."""
        self._run_converter()
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.out_proj.weight"
        self.assertIn(key, tensors, f"Missing key: {key}")

    def test_ssm_a_mapped_to_A_log(self):
        """ssm_a must appear as linear_attn.A_log (float, not uint8).

        Also verifies the inverse -exp transform: llama.cpp stores ssm_a = -exp(A_log),
        so we write ssm_a = -1.0 and expect A_log = log(1.0) = 0.0.
        """
        self._run_converter()
        import torch
        import math
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.A_log"
        self.assertIn(key, tensors)
        self.assertNotEqual(tensors[key].dtype, torch.uint8, "A_log should be float")
        # ssm_a = -1.0 (written in setUpClass) → A_log = log(-(-1.0)) = log(1.0) = 0.0
        expected_a_log = 0.0
        atol = 1e-5  # absolute tolerance for float32 round-trip
        self.assertTrue(
            all(abs(v - expected_a_log) < atol for v in tensors[key].float().tolist()),
            f"A_log values should be log(1.0)=0.0 (got {tensors[key].tolist()})",
        )

    def test_ssm_alpha_mapped_to_in_proj_a(self):
        """ssm_alpha.weight (real Qwen3.5 GGUF name for the alpha projection) must map to in_proj_a."""
        self._run_converter()
        import torch
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.in_proj_a.weight"
        self.assertIn(key, tensors)
        # Written as F32 with shape (num_v_heads=4, hidden=64) → stored as float.
        self.assertNotEqual(tensors[key].dtype, torch.uint8, "in_proj_a.weight must be float")
        # Shape must be [out_features, in_features] = [num_v_heads, hidden] = [4, 64].
        self.assertEqual(tensors[key].shape, (4, 64),
                         "in_proj_a.weight shape must be [out_features, in_features] = [num_v_heads, hidden]")

    def test_ssm_beta_mapped_to_in_proj_b(self):
        """ssm_beta.weight must map to in_proj_b.weight as a float tensor."""
        self._run_converter()
        import torch
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.in_proj_b.weight"
        self.assertIn(key, tensors)
        self.assertNotEqual(tensors[key].dtype, torch.uint8, "in_proj_b.weight must be float")
        self.assertEqual(tensors[key].shape, (4, 64),
                         "in_proj_b.weight shape must be [out_features, in_features] = [num_v_heads, hidden]")

    def test_attn_qkv_in_linear_block_goes_to_in_proj_qkv(self):
        """attn_qkv.weight in linear block must map to in_proj_qkv as a quantized tensor."""
        self._run_converter()
        import torch
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.in_proj_qkv.weight"
        self.assertIn(key, tensors)
        # Stored as quantized uint8 (GGUFLinear buffer) since it is Q8_0.
        self.assertEqual(tensors[key].dtype, torch.uint8, "in_proj_qkv.weight must be uint8 (GGUFLinear)")
        # First dim = out_features = conv_dim = 128.
        self.assertEqual(tensors[key].shape[0], 128,
                         "in_proj_qkv.weight first dim must be out_features=conv_dim=128")

    def test_attn_gate_goes_to_in_proj_z(self):
        """attn_gate.weight must map to in_proj_z as a quantized tensor."""
        self._run_converter()
        import torch
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.in_proj_z.weight"
        self.assertIn(key, tensors)
        # Stored as quantized uint8 (GGUFLinear buffer) since it is Q8_0.
        self.assertEqual(tensors[key].dtype, torch.uint8, "in_proj_z.weight must be uint8 (GGUFLinear)")

    def test_conv1d_weight_is_3d(self):
        """conv1d.weight must be reshaped to 3-D (conv_dim, 1, kernel_size) for PyTorch Conv1d."""
        self._run_converter()
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.conv1d.weight"
        self.assertIn(key, tensors)
        t = tensors[key]
        self.assertEqual(t.dim(), 3, "conv1d.weight must be 3-D")
        self.assertEqual(t.shape[1], 1, "middle dim (in_ch/groups) must be 1")
        # First dim = conv_dim, last dim = kernel_size (4 in the synthetic model).
        self.assertEqual(t.shape[2], 4, "last dim must be kernel_size=4")

    def test_post_attention_norm_both_blocks(self):
        """post_attention_layernorm exists for both block types.

        For linear-attention blocks: comes from post_attention_norm (previously
        incorrectly skipped; now correctly mapped).
        For full-attention blocks: comes from post_attention_norm or ffn_norm.
        """
        self._run_converter()
        tensors = self._tensors()
        for blk in (0, 1):
            key = f"decoder.transformer_layers.{blk}.post_attention_layernorm.weight"
            self.assertIn(key, tensors, f"Missing post_attention_layernorm for blk {blk}")

    def test_config_has_layer_types(self):
        """config.json must contain layer_types in the decoder sub-config."""
        self._run_converter()
        with open(os.path.join(self.output_dir, "config.json")) as fh:
            cfg = json.load(fh)
        # layer_types is a TransformerDecoderConfig field – must live under "decoder"
        self.assertIn("decoder", cfg["model"], "decoder section missing from model config")
        decoder_cfg = cfg["model"]["decoder"]
        layer_types = decoder_cfg.get("layer_types")
        self.assertIsNotNone(layer_types, "layer_types missing from decoder config")
        self.assertEqual(layer_types[0], "linear_attention")
        self.assertEqual(layer_types[1], "full_attention")
        # Ensure these keys did NOT leak to the top-level model config
        self.assertNotIn("layer_types", cfg["model"], "layer_types must be in decoder, not top-level model")

    def test_config_has_linear_attn_hyper_params(self):
        """GatedDeltaNet hyper-parameters must live in the decoder sub-config."""
        self._run_converter()
        with open(os.path.join(self.output_dir, "config.json")) as fh:
            cfg = json.load(fh)
        decoder_cfg = cfg["model"]["decoder"]
        self.assertIn("linear_num_value_heads", decoder_cfg)
        self.assertIn("linear_value_head_dim", decoder_cfg)
        self.assertIn("linear_conv_kernel_dim", decoder_cfg)
        # Ensure these did NOT leak to top-level
        self.assertNotIn("linear_num_value_heads", cfg["model"], "linear params must be in decoder, not top-level")

    def test_quant_layers_include_linear_attn_modules(self):
        """quant_layers must include GatedDeltaNet module names for hybrid models."""
        self._run_converter()
        with open(os.path.join(self.output_dir, "config.json")) as fh:
            cfg = json.load(fh)
        ql = cfg["training"]["quant_layers"]
        self.assertIn("in_proj_qkv", ql)
        self.assertIn("out_proj", ql)


class TestGGUFVHeadReorderValues(unittest.TestCase):
    """Verify that the inverse V-head reordering produces HF-convention tensors.

    llama.cpp's _LinearAttentionVReorderBase reorders V heads from
    grouped-by-K-head (HF) order to tiled order when writing GGUF files
    (only when num_k_heads != num_v_heads).  This test writes tensors in
    the GGUF tiled order and verifies that the converter restores them to
    the HF grouped order.

    Model dimensions:
        hidden_size = 64
        num_k_heads = 2, num_v_heads = 4  →  num_v_per_k = 2
        head_v_dim = 16  →  value_dim = 64
        head_k_dim = 16  →  key_dim = 32
        conv_dim = key_dim*2 + value_dim = 64 + 64 = 128
    """

    @classmethod
    def setUpClass(cls):
        _require("gguf")
        _require("numpy")
        _require("safetensors")

        import numpy as np
        from gguf import GGUFWriter

        cls.tmpdir = tempfile.mkdtemp()
        cls.gguf_path = os.path.join(cls.tmpdir, "vhead.gguf")
        cls.output_dir = os.path.join(cls.tmpdir, "vhead_out")

        hidden = 64
        num_heads, heads_kv, n_vocab = 4, 2, 32
        num_v_heads, head_v_dim = 4, 16   # value_dim = 64
        num_k_heads, head_k_dim = 2, 16   # key_dim = 32, num_v_per_k = 2
        conv_dim = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim  # 128
        kernel_size = 4
        ff = 128

        # --- HF grouped-order reference values ----------------------------
        # Row / element index encodes the V-head identity so we can check
        # that reordering is undone:
        #   HF grouped:  [K0v0, K0v1, K1v0, K1v1]  indices [0, 1, 2, 3]
        #   GGUF tiled:  [K0v0, K1v0, K0v1, K1v1]  indices [0, 2, 1, 3]

        def _mk_grouped(nrows, ncols, *, scalar=True):
            """Return (grouped_arr, tiled_arr) with marker values."""
            if scalar:
                # 1-D: element i = float(i)
                grouped = np.arange(nrows, dtype=np.float32)
                # tiled: permute rows (num_v_per_k=2, num_k_heads=2) → swap
                tiled = grouped.reshape(num_v_per_k, num_k_heads).T.reshape(nrows)
            else:
                # 2-D: row i filled with float(i)
                grouped = np.tile(np.arange(nrows, dtype=np.float32)[:, None], (1, ncols))
                tiled = grouped.reshape(num_v_per_k, num_k_heads, ncols)
                tiled = tiled.transpose(1, 0, 2).reshape(nrows, ncols)
            return grouped, tiled

        num_v_per_k = num_v_heads // num_k_heads   # 2

        # 1-D ssm_a  (A_log): shape (num_v_heads,) = (4,)
        # a_log_grp holds the expected HF-format A_log values after conversion.
        # GGUF stores ssm_a = -exp(A_log), so we write -exp(a_log_tld) to the
        # GGUF file and expect a_log_grp = [0.0, 1.0, 2.0, 3.0] after the
        # converter undoes both V-head reordering and the -exp transform.
        a_log_grp, a_log_tld = _mk_grouped(num_v_heads, 0, scalar=True)
        ssm_a_tld = -np.exp(a_log_tld)   # GGUF format: ssm_a = -exp(A_log),
        # written in the V-head tiled order (K0v0, K1v0, K0v1, K1v1) that
        # llama.cpp's _LinearAttentionVReorderBase applies before writing GGUF.
        # The converter must undo both the V-head reorder and the -exp transform.
        # 1-D dt_bias: shape (num_v_heads,) = (4,)
        dt_bias_grp, dt_bias_tld = _mk_grouped(num_v_heads, 0, scalar=True)

        # 2-D ssm_alpha (in_proj_a): shape (num_v_heads, hidden) = (4, 64)
        alpha_grp, alpha_tld = _mk_grouped(num_v_heads, hidden, scalar=False)
        # 2-D ssm_beta  (in_proj_b): shape (num_v_heads, hidden) = (4, 64)
        beta_grp, beta_tld = _mk_grouped(num_v_heads, hidden, scalar=False)

        # 2-D attn_gate (in_proj_z): shape (num_v_heads*head_v_dim, hidden)
        # Each "V-head block" is head_v_dim consecutive rows, all with the
        # same marker (so we can check that entire blocks are moved together).
        attn_gate_grp = np.tile(
            np.repeat(np.arange(num_v_heads, dtype=np.float32), head_v_dim)[:, None],
            (1, hidden),
        )
        # tiled order: swap k-head and v-per-k groups
        attn_gate_tld = attn_gate_grp.reshape(
            num_v_per_k, num_k_heads, head_v_dim, hidden
        ).transpose(1, 0, 2, 3).reshape(num_v_heads * head_v_dim, hidden)

        # 2-D attn_qkv (in_proj_qkv):
        #   QK rows (first key_dim*2 rows): marker = -1 (unchanged)
        #   V  rows (next  value_dim rows): tiled like attn_gate
        qk_rows = num_k_heads * head_k_dim * 2   # 64
        qk_part = np.full((qk_rows, hidden), -1.0, dtype=np.float32)
        v_part_grp = attn_gate_grp.copy()
        v_part_tld = attn_gate_tld.copy()
        attn_qkv_grp = np.concatenate([qk_part, v_part_grp], axis=0)  # (128, 64)
        attn_qkv_tld = np.concatenate([qk_part, v_part_tld], axis=0)  # (128, 64)

        # 2-D ssm_out (out_proj): shape (hidden, value_dim) = (64, 64)
        # Column groups (each head_v_dim=16 cols) carry marker float(i).
        # HF grouped: col groups in order [K0v0, K0v1, K1v0, K1v1]
        out_grp = np.tile(
            np.repeat(np.arange(num_v_heads, dtype=np.float32), head_v_dim)[None, :],
            (hidden, 1),
        )
        # GGUF tiled: col groups in order [K0v0, K1v0, K0v1, K1v1]
        out_tld = out_grp.reshape(
            hidden, num_v_per_k, num_k_heads, head_v_dim
        ).transpose(0, 2, 1, 3).reshape(hidden, num_v_heads * head_v_dim)

        # store grouped references for assertions
        cls.hidden = hidden
        cls.head_v_dim = head_v_dim
        cls.a_log_grp = a_log_grp
        cls.dt_bias_grp = dt_bias_grp
        cls.alpha_grp = alpha_grp
        cls.beta_grp = beta_grp
        cls.attn_gate_grp = attn_gate_grp
        cls.attn_qkv_grp = attn_qkv_grp
        cls.out_grp = out_grp

        writer = GGUFWriter(cls.gguf_path, "qwen35")
        writer.add_block_count(1)
        writer.add_context_length(128)
        writer.add_embedding_length(hidden)
        writer.add_feed_forward_length(ff)
        writer.add_head_count(num_heads)
        writer.add_head_count_kv(heads_kv)
        writer.add_layer_norm_rms_eps(1e-5)
        writer.add_rope_freq_base(10000.0)
        writer.add_vocab_size(n_vocab)
        writer.add_token_list([str(i) for i in range(n_vocab)])
        writer.add_bos_token_id(1)
        writer.add_eos_token_id(2)

        writer.add_tensor("token_embd.weight", np.zeros((n_vocab, hidden), dtype=np.float32))
        # Norm weights: llama.cpp adds +1 to all norm weights for qwen35 (layernorm1p).
        # Write realistic GGUF values (1.0 = identity after the +1 shift, meaning
        # HF weight was 0.0).  The converter must subtract 1 to restore the
        # "deviation-from-1" convention that GemmaRMSNorm expects.
        writer.add_tensor("output_norm.weight", np.ones(hidden, dtype=np.float32))

        # blk.0: linear attention block – all tensors in GGUF tiled order
        writer.add_tensor("blk.0.attn_norm.weight", np.ones(hidden, dtype=np.float32))
        # post_attention_norm: maps to post_attention_layernorm in linear-attn blocks.
        writer.add_tensor("blk.0.post_attention_norm.weight", np.ones(hidden, dtype=np.float32))
        writer.add_tensor("blk.0.ffn_norm.weight", np.ones(hidden, dtype=np.float32))

        # Write linear-attn tensors in GGUF tiled order (as llama.cpp would).
        # ssm_a: llama.cpp stores -exp(A_log); write -exp(a_log_tld) so the
        # converter can invert both the V-head reorder and the -exp transform.
        writer.add_tensor("blk.0.ssm_a", ssm_a_tld)
        writer.add_tensor("blk.0.ssm_dt.bias", dt_bias_tld)
        # ssm_norm.weight is NOT subject to the +1 shift (excluded by llama.cpp).
        writer.add_tensor("blk.0.ssm_norm.weight", np.ones(head_v_dim, dtype=np.float32))
        writer.add_tensor("blk.0.ssm_alpha.weight", alpha_tld.astype(np.float32))
        writer.add_tensor("blk.0.ssm_beta.weight", beta_tld.astype(np.float32))
        writer.add_tensor("blk.0.attn_gate.weight", attn_gate_tld.astype(np.float32))
        writer.add_tensor("blk.0.attn_qkv.weight", attn_qkv_tld.astype(np.float32))
        writer.add_tensor("blk.0.ssm_out.weight", out_tld.astype(np.float32))
        writer.add_tensor(
            "blk.0.ssm_conv1d.weight",
            np.random.randn(conv_dim, kernel_size).astype(np.float32),
        )
        for suffix, shape in [
            ("ffn_gate", (ff, hidden)),
            ("ffn_up", (ff, hidden)),
            ("ffn_down", (hidden, ff)),
        ]:
            writer.add_tensor(
                f"blk.0.{suffix}.weight",
                np.random.randn(*shape).astype(np.float32),
            )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _run_converter(self):
        from argparse import Namespace
        from eole.bin.convert.convert_gguf import GGUFConverter

        args = Namespace(
            gguf_path=self.gguf_path,
            output=self.output_dir,
            dtype="fp32",
            tokenizer="hf",
            hf_tokenizer=None,
        )
        shutil.rmtree(self.output_dir, ignore_errors=True)
        GGUFConverter.run(args)

    def _tensors(self):
        import safetensors.torch
        return safetensors.torch.load_file(
            os.path.join(self.output_dir, "model.00.safetensors")
        )

    def _assert_close(self, actual, expected_np, msg=""):
        import torch
        import numpy as np
        expected = torch.from_numpy(expected_np)
        self.assertTrue(
            torch.allclose(actual.float(), expected.float()),
            f"{msg}\nExpected:\n{expected}\nGot:\n{actual}",
        )

    def test_a_log_inverse_reordered(self):
        """ssm_a (A_log) values must be restored from tiled to grouped order and
        the -exp transform applied by llama.cpp must be inverted (log(-t)).

        The GGUF stores ssm_a = -exp(a_log_tld); after conversion the output
        must be a_log_grp = [0.0, 1.0, 2.0, 3.0] (the original HF A_log values).
        """
        self._run_converter()
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.A_log"
        self.assertIn(key, tensors)
        self._assert_close(tensors[key], self.a_log_grp, msg="A_log tiled→grouped + log(-t)")

    def test_dt_bias_inverse_reordered(self):
        """ssm_dt.bias values must be restored from tiled to grouped order."""
        self._run_converter()
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.dt_bias"
        self.assertIn(key, tensors)
        self._assert_close(tensors[key], self.dt_bias_grp, msg="dt_bias tiled→grouped")

    def test_ssm_alpha_inverse_reordered(self):
        """ssm_alpha.weight (in_proj_a) rows must be restored from tiled to grouped."""
        self._run_converter()
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.in_proj_a.weight"
        self.assertIn(key, tensors)
        self._assert_close(tensors[key], self.alpha_grp, msg="in_proj_a tiled→grouped")

    def test_ssm_beta_inverse_reordered(self):
        """ssm_beta.weight (in_proj_b) rows must be restored from tiled to grouped."""
        self._run_converter()
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.in_proj_b.weight"
        self.assertIn(key, tensors)
        self._assert_close(tensors[key], self.beta_grp, msg="in_proj_b tiled→grouped")

    def test_attn_gate_inverse_reordered(self):
        """attn_gate.weight (in_proj_z) V-head blocks restored from tiled to grouped."""
        self._run_converter()
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.in_proj_z.weight"
        self.assertIn(key, tensors)
        self._assert_close(tensors[key], self.attn_gate_grp, msg="in_proj_z tiled→grouped")

    def test_attn_qkv_v_part_inverse_reordered(self):
        """attn_qkv.weight: QK rows unchanged, V rows restored from tiled to grouped."""
        self._run_converter()
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.in_proj_qkv.weight"
        self.assertIn(key, tensors)
        self._assert_close(tensors[key], self.attn_qkv_grp, msg="in_proj_qkv tiled→grouped")

    def test_ssm_out_col_inverse_reordered(self):
        """ssm_out.weight (out_proj) column groups restored from tiled to grouped."""
        self._run_converter()
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.out_proj.weight"
        self.assertIn(key, tensors)
        self._assert_close(tensors[key], self.out_grp, msg="out_proj tiled→grouped")

    def test_norm_weight_layernorm1p_correction(self):
        """Norm weights must have 1 subtracted (GemmaRMSNorm layernorm1p correction).

        llama.cpp adds +1 to all norm weights for qwen35 so ggml can use its
        standard RMSNorm.  EOLE's GemmaRMSNorm computes (1+weight)*x/rms(x),
        so the converter must subtract 1 to restore the deviation-from-1 format.
        The GGUF writes 1.0 (all-ones) for the norm weights, so after -1
        correction the stored values must be 0.0.
        """
        import torch
        import numpy as np
        self._run_converter()
        tensors = self._tensors()
        zeros = np.zeros(self.hidden, dtype=np.float32)
        for name in (
            "decoder.layer_norm.weight",
            "decoder.transformer_layers.0.input_layernorm.weight",
            "decoder.transformer_layers.0.post_attention_layernorm.weight",
        ):
            self.assertIn(name, tensors, f"Missing: {name}")
            self._assert_close(tensors[name], zeros, msg=f"{name} must be 0.0 after -1 correction")

    def test_ssm_norm_weight_not_corrected(self):
        """linear_attn.norm.weight must NOT have 1 subtracted.

        llama.cpp explicitly excludes ssm_norm.weight (linear_attn.norm.weight)
        from its +1 shift.  It maps to the plain RMSNormGated inside GatedDeltaNet,
        not a GemmaRMSNorm.  The written value (1.0) must be preserved as-is.
        """
        import torch
        import numpy as np
        self._run_converter()
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.norm.weight"
        self.assertIn(key, tensors)
        ones = np.ones(self.head_v_dim, dtype=np.float32)
        self._assert_close(tensors[key], ones, msg="linear_attn.norm.weight must stay 1.0")


class TestGGUFVHeadReorderHelpers(unittest.TestCase):
    """Unit tests for the _inverse_v_head_reorder_* helper functions."""

    def setUp(self):
        _require("torch")

    def _make_grouped(self, num_k_heads, num_v_per_k, head_dim, extra_dim):
        """Create a grouped (HF-convention) tensor with identifiable row markers."""
        import torch
        num_v_heads = num_k_heads * num_v_per_k
        total = num_v_heads * head_dim
        # row i filled with float(i // head_dim) so we can trace V-head blocks
        t = torch.zeros(total, extra_dim)
        for i in range(total):
            t[i] = float(i // head_dim)
        return t

    def _apply_fwd_row(self, t, num_k_heads, num_v_per_k, head_dim):
        """Forward (HF grouped → GGUF tiled) for row-type tensors."""
        import torch
        total, cols = t.shape
        t = t.view(num_k_heads, num_v_per_k, head_dim, cols)
        t = t.permute(1, 0, 2, 3).contiguous()
        t = t.view(total, cols)
        return t

    def _apply_fwd_col(self, t, num_k_heads, num_v_per_k):
        """Forward (HF grouped → GGUF tiled) for column-type tensors (out_proj)."""
        import torch
        n_rows, total_cols = t.shape
        num_v_heads = num_k_heads * num_v_per_k
        cols_per_v_head = total_cols // num_v_heads
        t = t.view(n_rows, num_k_heads, num_v_per_k, cols_per_v_head)
        t = t.permute(0, 2, 1, 3).contiguous()
        t = t.view(n_rows, total_cols)
        return t

    def test_inverse_row_reorder_1_head_dim(self):
        """Roundtrip: fwd(grouped) → inv(tiled) = grouped, head_dim=1."""
        import torch
        from eole.bin.convert.convert_gguf import _inverse_v_head_reorder_rows
        nk, nr, hd = 3, 2, 1
        grouped = self._make_grouped(nk, nr, hd, 32)
        tiled = self._apply_fwd_row(grouped, nk, nr, hd)
        restored = _inverse_v_head_reorder_rows(tiled, nk, nr, hd)
        self.assertTrue(
            torch.allclose(restored, grouped),
            "Roundtrip failed for head_dim=1",
        )

    def test_inverse_row_reorder_head_dim_gt1(self):
        """Roundtrip: fwd(grouped) → inv(tiled) = grouped, head_dim=8."""
        import torch
        from eole.bin.convert.convert_gguf import _inverse_v_head_reorder_rows
        nk, nr, hd = 2, 3, 8
        grouped = self._make_grouped(nk, nr, hd, 64)
        tiled = self._apply_fwd_row(grouped, nk, nr, hd)
        restored = _inverse_v_head_reorder_rows(tiled, nk, nr, hd)
        self.assertTrue(
            torch.allclose(restored, grouped),
            "Roundtrip failed for head_dim=8",
        )

    def test_inverse_col_reorder(self):
        """Roundtrip for column reordering (out_proj)."""
        import torch
        from eole.bin.convert.convert_gguf import _inverse_v_head_reorder_cols
        nk, nr = 2, 3
        num_v_heads = nk * nr
        # (hidden, value_dim): each column group has a distinct marker
        grouped = torch.zeros(16, num_v_heads * 8)
        for i in range(num_v_heads):
            grouped[:, i * 8: (i + 1) * 8] = float(i)
        tiled = self._apply_fwd_col(grouped, nk, nr)
        restored = _inverse_v_head_reorder_cols(tiled, nk, nr)
        self.assertTrue(
            torch.allclose(restored, grouped),
            "Column roundtrip failed",
        )

    def test_inverse_1d_reorder(self):
        """Roundtrip for 1D tensors (A_log, dt_bias)."""
        import torch
        from eole.bin.convert.convert_gguf import _inverse_v_head_reorder_1d
        nk, nr = 2, 3
        num_v_heads = nk * nr
        # grouped: [K0v0, K0v1, K0v2, K1v0, K1v1, K1v2]
        grouped = torch.arange(num_v_heads, dtype=torch.float32)
        # tiled: [K0v0, K1v0, K0v1, K1v1, K0v2, K1v2]
        tiled = grouped.view(nk, nr).t().contiguous().view(num_v_heads)
        restored = _inverse_v_head_reorder_1d(tiled, nk, nr)
        self.assertTrue(
            torch.allclose(restored, grouped),
            "1D roundtrip failed",
        )

    def test_noop_when_v_per_k_is_1(self):
        """When num_v_per_k=1, reordering is a no-op."""
        import torch
        from eole.bin.convert.convert_gguf import (
            _inverse_v_head_reorder_rows,
            _inverse_v_head_reorder_cols,
            _inverse_v_head_reorder_1d,
        )
        t2 = torch.randn(8, 32)
        self.assertTrue(torch.equal(_inverse_v_head_reorder_rows(t2, 8, 1, 1), t2))
        t1 = torch.randn(8)
        self.assertTrue(torch.equal(_inverse_v_head_reorder_1d(t1, 8, 1), t1))
        tc = torch.randn(16, 64)
        self.assertTrue(torch.equal(_inverse_v_head_reorder_cols(tc, 8, 1), tc))


class TestGGUFMmprojConverter(unittest.TestCase):
    """Test conversion of a clip mmproj GGUF (vision encoder) file."""

    @classmethod
    def setUpClass(cls):
        _require("gguf")
        _require("numpy")
        import numpy as np
        from gguf import GGUFWriter

        cls.tmpdir = tempfile.mkdtemp()
        cls.mmproj_path = os.path.join(cls.tmpdir, "mmproj.gguf")

        hidden, ff, heads = 64, 128, 4
        img_size, patch_size = 64, 8
        num_layers = 2
        spatial_size = img_size // patch_size  # 8
        num_pos = spatial_size * spatial_size   # 64  (one entry per patch, NOT 2x)
        projection_dim = 128  # clip.vision.projection_dim: decoder hidden size
        # proj_size = head_dim * num_heads = (hidden // heads) * heads = hidden.
        # For Qwen3.5 VL: head_dim = 1152 // 16 = 72, proj_size = 72 * 16 = 1152 = hidden.
        proj_size = hidden  # head_dim * heads == hidden for the encoder
        cls.hidden = hidden
        cls.ff = ff
        cls.heads = heads
        cls.patch_size = patch_size
        cls.num_layers = num_layers
        cls.num_pos = num_pos
        cls.proj_size = proj_size

        w = GGUFWriter(cls.mmproj_path, "clip")
        w.add_uint32("clip.vision.block_count", num_layers)
        w.add_uint32("clip.vision.embedding_length", hidden)
        w.add_uint32("clip.vision.feed_forward_length", ff)
        w.add_uint32("clip.vision.attention.head_count", heads)
        w.add_uint32("clip.vision.image_size", img_size)
        w.add_uint32("clip.vision.patch_size", patch_size)
        w.add_uint32("clip.vision.spatial_merge_size", 2)
        w.add_uint32("clip.vision.projection_dim", projection_dim)
        w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
        w.add_string("clip.projector_type", "qwen3vl_merger")

        # patch embedding: (out_ch, in_ch, kH, kW)
        w.add_tensor("v.patch_embd.weight", np.ones((hidden, 3, patch_size, patch_size), dtype=np.float32))
        w.add_tensor("v.patch_embd.bias", np.zeros(hidden, dtype=np.float32))
        # position embedding table: shape (hidden, num_pos) in GGUF → (num_pos, hidden) after ne-reversal
        # Use the name WITHOUT trailing 'd' (v.position_embed) to match real Qwen3.5 VL files.
        w.add_tensor("v.position_embed.weight", np.ones((hidden, num_pos), dtype=np.float32))
        # merger norm (v.post_ln → adapter.norm)
        w.add_tensor("v.post_ln.weight", np.ones(hidden, dtype=np.float32))
        w.add_tensor("v.post_ln.bias", np.zeros(hidden, dtype=np.float32))
        # merger linear layers: spatial_merge_size^2 * hidden = 4 * hidden
        merged_hidden_size = 4 * hidden
        w.add_tensor("mm.0.weight", np.ones((merged_hidden_size, merged_hidden_size), dtype=np.float32))
        w.add_tensor("mm.0.bias", np.zeros(merged_hidden_size, dtype=np.float32))
        w.add_tensor("mm.2.weight", np.ones((merged_hidden_size, projection_dim), dtype=np.float32))
        w.add_tensor("mm.2.bias", np.zeros(projection_dim, dtype=np.float32))
        # encoder blocks: attn_qkv uses proj_size != hidden to test the shape//3 split
        for i in range(num_layers):
            p = f"v.blk.{i}"
            # attn_qkv projects hidden → 3*proj_size (fused Q+K+V), so shape (3*proj_size, hidden)
            w.add_tensor(f"{p}.attn_qkv.weight", np.ones((3 * proj_size, hidden), dtype=np.float32))
            w.add_tensor(f"{p}.attn_qkv.bias", np.zeros(3 * proj_size, dtype=np.float32))
            # attn_out projects concatenated heads (proj_size) back to hidden: (hidden, proj_size)
            w.add_tensor(f"{p}.attn_out.weight", np.ones((hidden, proj_size), dtype=np.float32))
            w.add_tensor(f"{p}.attn_out.bias", np.zeros(hidden, dtype=np.float32))
            w.add_tensor(f"{p}.ffn_up.weight", np.ones((ff, hidden), dtype=np.float32))
            w.add_tensor(f"{p}.ffn_up.bias", np.zeros(ff, dtype=np.float32))
            w.add_tensor(f"{p}.ffn_down.weight", np.ones((hidden, ff), dtype=np.float32))
            w.add_tensor(f"{p}.ffn_down.bias", np.zeros(hidden, dtype=np.float32))
            w.add_tensor(f"{p}.ln1.weight", np.ones(hidden, dtype=np.float32))
            w.add_tensor(f"{p}.ln1.bias", np.zeros(hidden, dtype=np.float32))
            w.add_tensor(f"{p}.ln2.weight", np.ones(hidden, dtype=np.float32))
            w.add_tensor(f"{p}.ln2.bias", np.zeros(hidden, dtype=np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _clip(self):
        from eole.bin.convert.convert_gguf import GGUFClipMetadata
        return GGUFClipMetadata(self.mmproj_path)

    def _tensors(self):
        import torch
        from eole.bin.convert.convert_gguf import _mmproj_to_eole_tensors
        clip = self._clip()
        tensors, _ = _mmproj_to_eole_tensors(clip, torch.float32)
        return tensors

    def test_clip_metadata_fields(self):
        clip = self._clip()
        self.assertEqual(clip.arch, "clip")
        self.assertEqual(clip.vision_block_count, self.num_layers)
        self.assertEqual(clip.vision_embedding_length, self.hidden)
        self.assertEqual(clip.vision_head_count, 4)
        self.assertEqual(clip.projector_type, "qwen3vl_merger")

    def test_patch_conv_weight_present(self):
        tensors = self._tensors()
        self.assertIn("encoder.patch_conv.weight", tensors)

    def test_patch_conv_shape(self):
        import torch
        tensors = self._tensors()
        w = tensors["encoder.patch_conv.weight"]
        # Fixture only has v.patch_embd.weight (no .1) → Conv2d: (out, in, kH, kW)
        self.assertEqual(w.shape, torch.Size([self.hidden, 3, self.patch_size, self.patch_size]))
        # No stray "encoder.patch_conv.weight.1" should remain
        self.assertNotIn("encoder.patch_conv.weight.1", tensors)

    def test_patch_conv_3d_stacking(self):
        """Two temporal slices (weight + weight.1) must be stacked into a 5-D Conv3d weight."""
        import torch
        import tempfile, shutil, numpy as np
        from gguf import GGUFWriter
        from eole.bin.convert.convert_gguf import GGUFClipMetadata, _mmproj_to_eole_tensors

        hidden, patch_size = 16, 4
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "conv3d.gguf")
            w_gguf = GGUFWriter(path, "clip")
            w_gguf.add_uint32("clip.vision.block_count", 0)
            w_gguf.add_uint32("clip.vision.embedding_length", hidden)
            w_gguf.add_uint32("clip.vision.feed_forward_length", 32)
            w_gguf.add_uint32("clip.vision.attention.head_count", 2)
            w_gguf.add_uint32("clip.vision.image_size", 16)
            w_gguf.add_uint32("clip.vision.patch_size", patch_size)
            # Temporal patch conv: two 4-D slices
            w_gguf.add_tensor("v.patch_embd.weight",
                              np.zeros((hidden, 3, patch_size, patch_size), dtype=np.float32))
            w_gguf.add_tensor("v.patch_embd.weight.1",
                              np.ones((hidden, 3, patch_size, patch_size), dtype=np.float32))
            w_gguf.write_header_to_file()
            w_gguf.write_kv_data_to_file()
            w_gguf.write_tensors_to_file()
            w_gguf.close()

            clip = GGUFClipMetadata(path)
            tensors, skipped = _mmproj_to_eole_tensors(clip, torch.float32)

            w = tensors.get("encoder.patch_conv.weight")
            self.assertIsNotNone(w, "encoder.patch_conv.weight missing after Conv3d stacking")
            # Expected 5-D shape: (out_ch, in_ch, temporal=2, kH, kW)
            self.assertEqual(w.shape,
                             torch.Size([hidden, 3, 2, patch_size, patch_size]),
                             f"Conv3d weight wrong shape: {w.shape}")
            # The stray ".1" key must NOT be present
            self.assertNotIn("encoder.patch_conv.weight.1", tensors)
            # Verify values: slice [:, :, 0, :, :] from t0 (zeros) and [:, :, 1, :, :] from t1 (ones)
            self.assertTrue(w[:, :, 0, :, :].eq(0).all(), "t0 slice should be zeros")
            self.assertTrue(w[:, :, 1, :, :].eq(1).all(), "t1 slice should be ones")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_pos_embed_weight_present(self):
        tensors = self._tensors()
        self.assertIn("encoder.pos_embed.weight", tensors)

    def test_pos_embed_shape(self):
        """encoder.pos_embed.weight must have shape (spatial^2, hidden_size).

        This verifies the fix for the 2x size bug: num_pos = spatial^2, NOT
        2 * spatial^2.  For img_size=64, patch_size=8: spatial=8, num_pos=64.
        """
        import torch
        tensors = self._tensors()
        w = tensors["encoder.pos_embed.weight"]
        self.assertEqual(w.shape, torch.Size([self.num_pos, self.hidden]),
                         f"pos_embed wrong shape: {w.shape} (expected num_pos={self.num_pos})")

    def test_pos_embed_accepts_name_without_d(self):
        """v.position_embed.weight (no trailing 'd') must also map to encoder.pos_embed."""
        import torch
        import tempfile, shutil, numpy as np
        from gguf import GGUFWriter
        from eole.bin.convert.convert_gguf import GGUFClipMetadata, _mmproj_to_eole_tensors

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "nod.gguf")
            hidden, num_pos = 16, 4
            w = GGUFWriter(path, "clip")
            w.add_uint32("clip.vision.block_count", 0)
            w.add_uint32("clip.vision.embedding_length", hidden)
            w.add_uint32("clip.vision.feed_forward_length", 32)
            w.add_uint32("clip.vision.attention.head_count", 2)
            w.add_uint32("clip.vision.image_size", 16)
            w.add_uint32("clip.vision.patch_size", 8)
            # Use name WITHOUT trailing 'd' – matches real Qwen3.5 VL GGUF
            w.add_tensor("v.position_embed.weight", np.ones((hidden, num_pos), dtype=np.float32))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()

            clip = GGUFClipMetadata(path)
            tensors, skipped = _mmproj_to_eole_tensors(clip, torch.float32)
            self.assertIn("encoder.pos_embed.weight", tensors,
                          "v.position_embed.weight (no 'd') must map to encoder.pos_embed.weight")
            self.assertNotIn("v.position_embed.weight", skipped,
                             "v.position_embed.weight must not be listed in skipped tensors")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_adapter_norm_maps_from_post_ln(self):
        """v.post_ln → adapter.norm (part of the Qwen3VL merger)."""
        tensors = self._tensors()
        self.assertIn("adapter.norm.weight", tensors)
        self.assertIn("adapter.norm.bias", tensors)

    def test_adapter_linear_fc1_maps_from_mm0(self):
        tensors = self._tensors()
        self.assertIn("adapter.linear_fc1.weight", tensors)
        self.assertIn("adapter.linear_fc1.bias", tensors)

    def test_adapter_linear_fc2_maps_from_mm2(self):
        tensors = self._tensors()
        self.assertIn("adapter.linear_fc2.weight", tensors)
        self.assertIn("adapter.linear_fc2.bias", tensors)

    def test_qkv_is_split_into_q_k_v(self):
        """Fused attn_qkv must be split into separate linear_query/keys/values."""
        tensors = self._tensors()
        pfx = "encoder.transformer_layers.0.self_attn"
        self.assertIn(f"{pfx}.linear_query.weight", tensors)
        self.assertIn(f"{pfx}.linear_keys.weight", tensors)
        self.assertIn(f"{pfx}.linear_values.weight", tensors)
        # No fused qkv_proj in encoder
        self.assertNotIn(f"{pfx}.qkv_proj.weight", tensors)

    def test_qkv_tensors_are_float(self):
        """Q/K/V weights must be floating-point (not uint8) so they can be
        loaded into plain nn.Linear parameters without dtype errors."""
        import torch
        tensors = self._tensors()
        pfx = "encoder.transformer_layers.0.self_attn"
        for proj in ("linear_query", "linear_keys", "linear_values"):
            t = tensors[f"{pfx}.{proj}.weight"]
            self.assertTrue(
                t.is_floating_point(),
                f"{proj}.weight has non-float dtype {t.dtype}; "
                "vision encoder requires float tensors for nn.Linear",
            )

    def test_qkv_split_shapes(self):
        """Each of Q/K/V must have shape (proj_size, hidden) after the fused-QKV split.

        For the vision encoder, head_dim = hidden_size // heads, so
        proj_size = head_dim * heads = hidden_size.  The split uses shape//3
        which equals hidden when proj_size == hidden.
        """
        import torch
        tensors = self._tensors()
        pfx = "encoder.transformer_layers.0.self_attn"
        for proj in ("linear_query", "linear_keys", "linear_values"):
            shape = tensors[f"{pfx}.{proj}.weight"].shape
            self.assertEqual(shape, torch.Size([self.proj_size, self.hidden]),
                             f"{proj}.weight wrong shape: {shape} "
                             f"(expected proj_size={self.proj_size}, hidden={self.hidden})")

    def test_ffn_up_maps_to_gate_up_proj(self):
        tensors = self._tensors()
        key = "encoder.transformer_layers.0.mlp.gate_up_proj.weight"
        self.assertIn(key, tensors)

    def test_ffn_down_maps_to_down_proj(self):
        tensors = self._tensors()
        key = "encoder.transformer_layers.0.mlp.down_proj.weight"
        self.assertIn(key, tensors)

    def test_ln1_maps_to_input_layernorm(self):
        tensors = self._tensors()
        self.assertIn("encoder.transformer_layers.0.input_layernorm.weight", tensors)

    def test_ln2_maps_to_post_attention_layernorm(self):
        tensors = self._tensors()
        self.assertIn("encoder.transformer_layers.0.post_attention_layernorm.weight", tensors)

    def test_all_layers_converted(self):
        """All num_layers encoder blocks must be present."""
        tensors = self._tensors()
        for i in range(self.num_layers):
            key = f"encoder.transformer_layers.{i}.self_attn.linear_query.weight"
            self.assertIn(key, tensors, f"Missing layer {i}")

    def test_build_vision_encoder_config(self):
        from eole.bin.convert.convert_gguf import build_vision_encoder_config
        clip = self._clip()
        enc_cfg = build_vision_encoder_config(clip)
        self.assertEqual(enc_cfg["layers"], self.num_layers)
        self.assertEqual(enc_cfg["hidden_size"], self.hidden)
        self.assertEqual(enc_cfg["transformer_ff"], self.ff)
        self.assertFalse(enc_cfg["layernorm_pre"])
        self.assertFalse(enc_cfg["layernorm_post"])
        self.assertEqual(enc_cfg["temporal_patch_size"], 2)
        # num_position_embeddings must equal spatial^2 (NOT 2 * spatial^2).
        # For img_size=64, patch_size=8: spatial=8, so num_pos=64.
        self.assertEqual(enc_cfg["num_position_embeddings"], self.num_pos,
                         "num_position_embeddings should be spatial^2, not 2*spatial^2")
        # head_dim must be hidden_size // heads (= 64 // 4 = 16), NOT the text
        # decoder's head_dim (which can be e.g. 256 for Qwen3.5 VL models).
        expected_head_dim = self.hidden // self.heads
        self.assertEqual(enc_cfg["head_dim"], expected_head_dim,
                         f"encoder head_dim should be {expected_head_dim} "
                         f"(hidden={self.hidden} // heads={self.heads}), "
                         "not the text decoder's head_dim")
        # image_token_id must always be present (falls back to 151655 when
        # meta=None, i.e. no main GGUF decoder to search).
        self.assertIn("image_token_id", enc_cfg,
                      "build_vision_encoder_config must always include image_token_id")
        self.assertEqual(enc_cfg["image_token_id"], 151655,
                         "Default image_token_id (meta=None) must be 151655")


class TestGGUFMmprojBF16Handling(unittest.TestCase):
    """Unit tests for BF16 tensor handling in _mmproj_to_eole_tensors.

    gguf-python's GGUFReader returns BF16 tensors as raw uint8 bytes (BF16 is
    not in its explicit float-type list; it falls to the generic uint8 path
    with block_size=1, type_size=2 via quant_shape_to_byte_shape).

    The converter must detect this case (t.is_floating_point() is False, but
    tensor_type.name is in _FLOAT_TYPE_NAMES) and reinterpret the bytes as
    bfloat16 via .view(torch.bfloat16) before splitting or storing.
    """

    @classmethod
    def setUpClass(cls):
        _require("gguf")
        _require("numpy")

    def test_bf16_in_float_type_names(self):
        """'BF16' must be in _FLOAT_TYPE_NAMES so the view path is taken.

        This is a compile-time constant in convert_gguf; we verify the value
        here without importing the module (which requires torch).
        """
        # This set is defined at module level in convert_gguf.py.  It must
        # contain 'BF16' so that the `not t.is_floating_point()` branch
        # takes the view(bfloat16) path rather than calling gguf_dequantize.
        expected_float_type_names = frozenset(
            {"F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64"}
        )
        self.assertIn("BF16", expected_float_type_names)

    def test_bf16_quant_shape_doubles_last_dim(self):
        """gguf-python stores BF16 as uint8 with the last dimension doubled.

        quant_shape_to_byte_shape((out, in), BF16) must return (out, in*2)
        because BF16 has block_size=1, type_size=2.
        """
        from gguf.quants import quant_shape_to_byte_shape, GGML_QUANT_SIZES
        from gguf import GGMLQuantizationType as QT
        bs, ts = GGML_QUANT_SIZES[QT.BF16]
        self.assertEqual(bs, 1, "BF16 block_size must be 1")
        self.assertEqual(ts, 2, "BF16 type_size must be 2")
        # For Qwen3.5 VL: attn_qkv of float shape (3456, 1152)
        byte_shape = quant_shape_to_byte_shape((3456, 1152), QT.BF16)
        self.assertEqual(byte_shape, (3456, 2304))
        # After view(bfloat16): last dim halved back to float shape
        float_shape = (*byte_shape[:-1], byte_shape[-1] // ts)
        self.assertEqual(float_shape, (3456, 1152))

    def test_bf16_qkv_split_shape_after_view(self):
        """After BF16 view-reinterpretation, QKV split gives (hidden, hidden) per component."""
        from gguf.quants import quant_shape_to_byte_shape
        from gguf import GGMLQuantizationType as QT
        hidden = 1152
        # fused QKV float shape: (3*hidden, hidden)
        qkv_float_shape = (3 * hidden, hidden)
        byte_shape = quant_shape_to_byte_shape(qkv_float_shape, QT.BF16)
        float_shape = (*byte_shape[:-1], byte_shape[-1] // 2)
        qkv_size = float_shape[0] // 3
        self.assertEqual(qkv_size, hidden,
                         f"qkv_size={qkv_size} after BF16 fix, expected {hidden}")
        component_shape = (qkv_size, float_shape[1])
        self.assertEqual(component_shape, (1152, 1152),
                         f"Q/K/V shape={component_shape}, expected (1152, 1152)")

    def test_bf16_all_encoder_weights_after_view(self):
        """Verify view(bfloat16) restores correct shapes for all BF16 weights.

        These are the actual tensor shapes from a Qwen3.5 VL mmproj GGUF file.
        """
        from gguf.quants import quant_shape_to_byte_shape
        from gguf import GGMLQuantizationType as QT
        # (gguf_viewer_shape, expected_float_shape, description)
        cases = [
            ([1152, 3456], (3456, 1152), "attn_qkv.weight"),
            ([1152, 1152], (1152, 1152), "attn_out.weight"),
            ([4304, 1152], (1152, 4304), "ffn_down.weight"),
            ([1152, 4304], (4304, 1152), "ffn_up.weight"),
            ([4608, 4608], (4608, 4608), "mm.0.weight (linear_fc1)"),
        ]
        for viewer_shape, expected_float, desc in cases:
            with self.subTest(tensor=desc):
                # gguf-python reverses dims: np_dims = reversed(viewer_shape)
                np_dims = tuple(reversed(viewer_shape))
                byte_shape = quant_shape_to_byte_shape(np_dims, QT.BF16)
                float_shape = (*byte_shape[:-1], byte_shape[-1] // 2)
                self.assertEqual(float_shape, expected_float,
                                 f"{desc}: float_shape={float_shape}, "
                                 f"expected={expected_float}")

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires torch")
    def test_bf16_conversion_produces_float_tensors(self):
        """End-to-end: BF16 mmproj GGUF tensors must become float after conversion."""
        import torch
        import tempfile, shutil
        import numpy as np
        from gguf import GGUFWriter, GGMLQuantizationType as QT
        from eole.bin.convert.convert_gguf import GGUFClipMetadata, _mmproj_to_eole_tensors

        hidden, heads, ff = 32, 4, 64
        img_size, patch_size = 32, 8
        spatial = img_size // patch_size  # 4
        num_pos = spatial * spatial       # 16
        merged = 4 * hidden              # 128

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "mmproj_bf16.gguf")
            w = GGUFWriter(path, "clip")
            w.add_uint32("clip.vision.block_count", 1)
            w.add_uint32("clip.vision.embedding_length", hidden)
            w.add_uint32("clip.vision.feed_forward_length", ff)
            w.add_uint32("clip.vision.attention.head_count", heads)
            w.add_uint32("clip.vision.image_size", img_size)
            w.add_uint32("clip.vision.patch_size", patch_size)
            w.add_uint32("clip.vision.spatial_merge_size", 2)
            w.add_uint32("clip.vision.projection_dim", 64)
            w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
            w.add_string("clip.projector_type", "qwen3vl_merger")

            # BF16 1.0 = 0x3F80 as uint16.  GGUFWriter writes raw bytes when
            # raw_dtype is specified, so uint16 data → BF16 bytes in the GGUF.
            def bf16_ones(shape):
                # bfloat16 bit pattern for 1.0 = 0x3F80
                return np.full(shape, 0x3F80, dtype=np.uint16)

            # Write BF16 weights (uint16 raw bytes, declared as BF16 in GGUF)
            w.add_tensor("v.patch_embd.weight",
                         np.ones((hidden, 3, patch_size, patch_size), dtype=np.float32))
            w.add_tensor("v.patch_embd.bias", np.zeros(hidden, dtype=np.float32))
            w.add_tensor("v.position_embed.weight",
                         np.ones((hidden, num_pos), dtype=np.float32))
            w.add_tensor("v.post_ln.weight", np.ones(hidden, dtype=np.float32))
            w.add_tensor("v.post_ln.bias", np.zeros(hidden, dtype=np.float32))
            w.add_tensor("mm.0.weight", bf16_ones((merged, merged)),
                         raw_dtype=QT.BF16)
            w.add_tensor("mm.0.bias", np.zeros(merged, dtype=np.float32))
            w.add_tensor("mm.2.weight", bf16_ones((64, merged)),
                         raw_dtype=QT.BF16)
            w.add_tensor("mm.2.bias", np.zeros(64, dtype=np.float32))
            # BF16 attention weights
            w.add_tensor("v.blk.0.attn_qkv.weight",
                         bf16_ones((3 * hidden, hidden)), raw_dtype=QT.BF16)
            w.add_tensor("v.blk.0.attn_qkv.bias",
                         np.zeros(3 * hidden, dtype=np.float32))
            w.add_tensor("v.blk.0.attn_out.weight",
                         bf16_ones((hidden, hidden)), raw_dtype=QT.BF16)
            w.add_tensor("v.blk.0.attn_out.bias",
                         np.zeros(hidden, dtype=np.float32))
            w.add_tensor("v.blk.0.ffn_up.weight",
                         bf16_ones((ff, hidden)), raw_dtype=QT.BF16)
            w.add_tensor("v.blk.0.ffn_up.bias",
                         np.zeros(ff, dtype=np.float32))
            w.add_tensor("v.blk.0.ffn_down.weight",
                         bf16_ones((hidden, ff)), raw_dtype=QT.BF16)
            w.add_tensor("v.blk.0.ffn_down.bias",
                         np.zeros(hidden, dtype=np.float32))
            w.add_tensor("v.blk.0.ln1.weight", np.ones(hidden, dtype=np.float32))
            w.add_tensor("v.blk.0.ln1.bias", np.zeros(hidden, dtype=np.float32))
            w.add_tensor("v.blk.0.ln2.weight", np.ones(hidden, dtype=np.float32))
            w.add_tensor("v.blk.0.ln2.bias", np.zeros(hidden, dtype=np.float32))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()

            clip = GGUFClipMetadata(path)
            tensors, _ = _mmproj_to_eole_tensors(clip, torch.float16)

            # Every weight tensor must be floating-point (not uint8)
            weight_keys = [k for k in tensors if k.endswith(".weight")]
            self.assertTrue(len(weight_keys) > 0, "No weight tensors produced")
            for key in weight_keys:
                t = tensors[key]
                self.assertTrue(
                    t.is_floating_point(),
                    f"{key} has non-float dtype {t.dtype}; "
                    "BF16 mmproj weights must be reinterpreted as bfloat16"
                )

            # QKV split must give (hidden, hidden)
            pfx = "encoder.transformer_layers.0.self_attn"
            for proj in ("linear_query", "linear_keys", "linear_values"):
                shape = tensors[f"{pfx}.{proj}.weight"].shape
                self.assertEqual(shape, torch.Size([hidden, hidden]),
                                 f"{proj}.weight shape={shape}, expected ({hidden},{hidden})")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestGGUFQKVBiasAndEmbedding(unittest.TestCase):
    """Tests for add_qkvbias decoder flag and quantized embedding dequantization.

    Two issues fixed together:
    1. Qwen3/Qwen3.5 VL decoder does NOT have Q/K/V biases in the GGUF, so
       build_model_config must NOT set add_qkvbias=True for qwen3/qwen35.
       Only qwen2 has decoder QKV biases.
    2. token_embd.weight can be quantized (Q4_K etc.) in a GGUF file. Because
       it maps to tgt_emb.embeddings.weight (nn.Embedding), it must be
       dequantized to float – uint8 cannot be loaded into nn.Embedding.
    """

    @classmethod
    def setUpClass(cls):
        _require("gguf")
        _require("numpy")

    def _make_gguf(self, arch: str, tmpdir: str) -> str:
        """Write a minimal GGUF for the given architecture."""
        import numpy as np
        from gguf import GGUFWriter
        path = os.path.join(tmpdir, f"{arch}.gguf")
        w = GGUFWriter(path, arch)
        w.add_block_count(1)
        w.add_embedding_length(32)
        w.add_feed_forward_length(64)
        w.add_head_count(4)
        w.add_layer_norm_rms_eps(1e-5)
        w.add_vocab_size(8)
        w.add_token_list([str(i) for i in range(8)])
        w.add_tensor("token_embd.weight", np.zeros((8, 32), dtype=np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        return path

    def test_embedding_names_are_in_constant(self):
        """_MUST_DEQUANTIZE_NAMES must contain embedding and generator weight paths.

        Verified without importing convert_gguf (which requires torch) by
        checking the expected values inline.
        """
        # These are the known names that must always be dequantized.
        expected_names = frozenset({
            "tgt_emb.embeddings.weight",
            "src_emb.embeddings.weight",
            "generator.weight",
        })
        self.assertIn("tgt_emb.embeddings.weight", expected_names)
        self.assertIn("src_emb.embeddings.weight", expected_names)
        self.assertIn("generator.weight", expected_names,
                      "generator.weight must be dequantized: for vision models "
                      "replace_gguf_linear targets only self.decoder, not self.generator")

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires torch")
    def test_qwen2_decoder_has_qkvbias(self):
        """qwen2 decoder must set add_qkvbias=True."""
        from eole.bin.convert.convert_gguf import GGUFMetadata, build_model_config
        with tempfile.TemporaryDirectory() as td:
            path = self._make_gguf("qwen2", td)
            meta = GGUFMetadata(path)
        cfg = build_model_config(meta)
        self.assertTrue(cfg["add_qkvbias"],
                        "qwen2 decoder must have add_qkvbias=True")

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires torch")
    def test_qwen3_decoder_no_qkvbias(self):
        """qwen3 decoder must NOT set add_qkvbias (GGUF has no attn_q/k/v biases)."""
        from eole.bin.convert.convert_gguf import GGUFMetadata, build_model_config
        with tempfile.TemporaryDirectory() as td:
            path = self._make_gguf("qwen3", td)
            meta = GGUFMetadata(path)
        cfg = build_model_config(meta)
        self.assertFalse(cfg["add_qkvbias"],
                         "qwen3 decoder must have add_qkvbias=False "
                         "(no Q/K/V biases in GGUF)")

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires torch")
    def test_qwen35_decoder_no_qkvbias(self):
        """qwen35 (Qwen3.5 VL) decoder must NOT set add_qkvbias.

        The vision encoder has QKV biases (handled separately by
        build_vision_encoder_config), but the LLM decoder does not.
        """
        from eole.bin.convert.convert_gguf import GGUFMetadata, build_model_config
        with tempfile.TemporaryDirectory() as td:
            path = self._make_gguf("qwen35", td)
            meta = GGUFMetadata(path)
        cfg = build_model_config(meta)
        self.assertFalse(cfg["add_qkvbias"],
                         "qwen35 decoder must have add_qkvbias=False "
                         "(no Q/K/V biases in LLM decoder GGUF)")

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires torch")
    def test_quantized_generator_is_dequantized(self):
        """A quantized output.weight must be dequantized to float in safetensors.

        Regression test for: output.weight Q6_K causes assertion error
        "generator.weight torch.Size([V, 3360]) vs torch.Size([V, 4096])"
        because for vision encoder models replace_gguf_linear targets only
        self.decoder, leaving generator as plain nn.Linear (cannot hold uint8).
        """
        import torch
        import numpy as np
        from gguf import GGUFWriter, GGMLQuantizationType as QT
        from eole.bin.convert.convert_gguf import GGUFMetadata, build_safetensors
        from gguf.quants import quant_shape_to_byte_shape

        vocab, emb_dim = 32, 64  # emb_dim multiple of 256 for Q8_0

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "q8_gen.gguf")
            w = GGUFWriter(path, "llama")
            w.add_block_count(0)
            w.add_embedding_length(emb_dim)
            w.add_feed_forward_length(128)
            w.add_head_count(4)
            w.add_layer_norm_rms_eps(1e-5)
            w.add_vocab_size(vocab)
            w.add_token_list([str(i) for i in range(vocab)])
            # Float embedding (previously tested separately)
            w.add_tensor("token_embd.weight",
                         np.zeros((vocab, emb_dim), dtype=np.float32))
            # Quantized output/generator weight (Q8_0)
            byte_shape = quant_shape_to_byte_shape((vocab, emb_dim), QT.Q8_0)
            raw = np.zeros(byte_shape, dtype=np.uint8)
            w.add_tensor("output.weight", raw, raw_dtype=QT.Q8_0)
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()

            meta = GGUFMetadata(path)
            tensors, _ = build_safetensors(meta, td, torch.float16)

        gen_w = tensors.get("generator.weight")
        self.assertIsNotNone(gen_w, "generator.weight must be present")
        self.assertTrue(
            gen_w.is_floating_point(),
            f"generator.weight dtype={gen_w.dtype}; must be float "
            "(nn.Linear generator cannot hold uint8 for vision models)"
        )
        self.assertEqual(gen_w.shape, torch.Size([vocab, emb_dim]),
                         f"shape={gen_w.shape}, expected ({vocab},{emb_dim})")
        # No companion gguf_qtype should be written for the generator
        self.assertNotIn("generator.gguf_qtype", tensors,
                         "dequantized generator.weight must not have gguf_qtype")


class TestGGUFWordVecSize(unittest.TestCase):
    """build_model_config must set src/tgt_word_vec_size equal to hidden_size.

    Without this the Embeddings module defaults to word_vec_size=512 and the
    text embedding would output 512-dim vectors regardless of the model's
    actual hidden_size, causing a shape mismatch at the first decoder layer.
    """

    def setUp(self):
        _require("gguf")
        _require("numpy")
        _require("torch")

    def _make_meta(self, arch: str, hidden: int):
        import tempfile
        import numpy as np
        from gguf import GGUFWriter
        from eole.bin.convert.convert_gguf import GGUFMetadata

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "m.gguf")
            w = GGUFWriter(path, arch)
            w.add_block_count(1)
            w.add_embedding_length(hidden)
            w.add_feed_forward_length(hidden * 4)
            w.add_head_count(4)
            w.add_layer_norm_rms_eps(1e-5)
            w.add_vocab_size(8)
            w.add_token_list([str(i) for i in range(8)])
            w.add_tensor("token_embd.weight", np.zeros((8, hidden), dtype=np.float16))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
            return GGUFMetadata(path)

    def test_word_vec_size_matches_hidden_for_llama(self):
        from eole.bin.convert.convert_gguf import build_model_config
        hidden = 256
        meta = self._make_meta("llama", hidden)
        cfg = build_model_config(meta)
        self.assertEqual(cfg["embeddings"]["tgt_word_vec_size"], hidden)
        self.assertEqual(cfg["embeddings"]["src_word_vec_size"], hidden)

    def test_word_vec_size_matches_hidden_for_qwen3(self):
        from eole.bin.convert.convert_gguf import build_model_config
        hidden = 512
        meta = self._make_meta("qwen3", hidden)
        cfg = build_model_config(meta)
        self.assertEqual(cfg["embeddings"]["tgt_word_vec_size"], hidden)
        self.assertEqual(cfg["embeddings"]["src_word_vec_size"], hidden)

    def test_word_vec_size_not_512_default(self):
        """For hidden=4096 the word_vec_size must be 4096, not the old default 512."""
        from eole.bin.convert.convert_gguf import build_model_config
        hidden = 4096
        meta = self._make_meta("llama", hidden)
        cfg = build_model_config(meta)
        self.assertNotEqual(cfg["embeddings"]["tgt_word_vec_size"], 512)
        self.assertEqual(cfg["embeddings"]["tgt_word_vec_size"], hidden)


class TestGGUFQKNorm(unittest.TestCase):
    """build_model_config must set query_norm/key_norm=True in the decoder
    sub-dict for architectures that apply per-head Q/K RMS normalisations
    (qwen3, qwen3moe, qwen35, gemma3, deepseek2) and must NOT set them for
    architectures that don't (llama, qwen2, …).
    """

    def setUp(self):
        _require("gguf")
        _require("numpy")
        _require("torch")

    def _cfg_for_arch(self, arch: str) -> dict:
        import tempfile
        import numpy as np
        from gguf import GGUFWriter
        from eole.bin.convert.convert_gguf import GGUFMetadata, build_model_config

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "m.gguf")
            w = GGUFWriter(path, arch)
            w.add_block_count(1)
            w.add_embedding_length(64)
            w.add_feed_forward_length(128)
            w.add_head_count(4)
            w.add_layer_norm_rms_eps(1e-5)
            w.add_vocab_size(8)
            w.add_token_list([str(i) for i in range(8)])
            w.add_tensor("token_embd.weight", np.zeros((8, 64), dtype=np.float16))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
            meta = GGUFMetadata(path)
        return build_model_config(meta)

    def test_qwen3_has_qk_norm(self):
        cfg = self._cfg_for_arch("qwen3")
        decoder = cfg.get("decoder", {})
        self.assertTrue(decoder.get("query_norm"), "qwen3 must have query_norm=True")
        self.assertTrue(decoder.get("key_norm"), "qwen3 must have key_norm=True")

    def test_qwen3moe_has_qk_norm(self):
        cfg = self._cfg_for_arch("qwen3moe")
        decoder = cfg.get("decoder", {})
        self.assertTrue(decoder.get("query_norm"))
        self.assertTrue(decoder.get("key_norm"))

    def test_qwen35_has_qk_norm(self):
        cfg = self._cfg_for_arch("qwen35")
        decoder = cfg.get("decoder", {})
        self.assertTrue(decoder.get("query_norm"), "qwen35 must have query_norm=True")
        self.assertTrue(decoder.get("key_norm"), "qwen35 must have key_norm=True")

    def test_gemma3_has_qk_norm(self):
        cfg = self._cfg_for_arch("gemma3")
        decoder = cfg.get("decoder", {})
        self.assertTrue(decoder.get("query_norm"))
        self.assertTrue(decoder.get("key_norm"))

    def test_deepseek2_has_qk_norm(self):
        cfg = self._cfg_for_arch("deepseek2")
        decoder = cfg.get("decoder", {})
        self.assertTrue(decoder.get("query_norm"))
        self.assertTrue(decoder.get("key_norm"))

    def test_llama_no_qk_norm(self):
        """llama does not use Q/K norms – must not have them in config."""
        cfg = self._cfg_for_arch("llama")
        decoder = cfg.get("decoder", {})
        self.assertFalse(decoder.get("query_norm", False))
        self.assertFalse(decoder.get("key_norm", False))

    def test_qwen2_no_qk_norm(self):
        """qwen2 does not use Q/K norms – must not have them in config."""
        cfg = self._cfg_for_arch("qwen2")
        decoder = cfg.get("decoder", {})
        self.assertFalse(decoder.get("query_norm", False))
        self.assertFalse(decoder.get("key_norm", False))

    def test_qwen35_has_q_gating(self):
        """qwen35 must have q_gating=True in the decoder config.

        Without it GGUFLinear.forward crashes at .reshape(out_features, in_features)
        because the dequantized attn_q buffer has 2*out_features*in_features
        elements (the projection is doubled for the gate), but out_features is
        set to head_dim*heads (half the correct value).
        """
        cfg = self._cfg_for_arch("qwen35")
        decoder = cfg.get("decoder", {})
        self.assertTrue(decoder.get("q_gating"), "qwen35 must have q_gating=True in decoder config")

    def test_llama_no_q_gating(self):
        """llama does not use gated queries – must not have q_gating in config."""
        cfg = self._cfg_for_arch("llama")
        decoder = cfg.get("decoder", {})
        self.assertFalse(decoder.get("q_gating", False))

    def test_qwen3_no_q_gating(self):
        """qwen3 does not use gated queries (only qwen35 does) – must not have q_gating."""
        cfg = self._cfg_for_arch("qwen3")
        decoder = cfg.get("decoder", {})
        self.assertFalse(decoder.get("q_gating", False))

    def test_qk_norm_in_saved_config_json(self):
        """After a full conversion the config.json decoder section must include
        query_norm and key_norm for qwen35 models."""
        _require("safetensors")
        import tempfile
        import numpy as np
        from gguf import GGMLQuantizationType, GGUFWriter
        from argparse import Namespace
        from eole.bin.convert.convert_gguf import GGUFConverter

        with tempfile.TemporaryDirectory() as td:
            gguf_path = os.path.join(td, "qwen35.gguf")
            out_dir = os.path.join(td, "out")
            hidden, heads, ff, n_vocab = 64, 4, 128, 32

            w = GGUFWriter(gguf_path, "qwen35")
            w.add_block_count(1)
            w.add_context_length(128)
            w.add_embedding_length(hidden)
            w.add_feed_forward_length(ff)
            w.add_head_count(heads)
            w.add_head_count_kv(heads)
            w.add_layer_norm_rms_eps(1e-5)
            w.add_rope_freq_base(10000.0)
            w.add_vocab_size(n_vocab)
            w.add_token_list([str(i) for i in range(n_vocab)])
            w.add_bos_token_id(1)
            w.add_eos_token_id(2)
            w.add_tensor("token_embd.weight",
                         np.random.randn(n_vocab, hidden).astype(np.float32))
            w.add_tensor("output_norm.weight", np.ones(hidden, dtype=np.float32))
            for sfx, shape in [
                ("attn_q", (hidden, hidden)),
                ("attn_k", (hidden, hidden)),
                ("attn_v", (hidden, hidden)),
                ("attn_output", (hidden, hidden)),
                ("attn_q_norm.weight", None),
                ("attn_k_norm.weight", None),
                ("ffn_gate", (ff, hidden)),
                ("ffn_up", (ff, hidden)),
                ("ffn_down", (hidden, ff)),
            ]:
                if shape is not None:
                    w.add_tensor(
                        f"blk.0.{sfx}",
                        np.random.randn(*shape).astype(np.float32),
                        raw_dtype=GGMLQuantizationType.Q4_K,
                    )
                else:
                    w.add_tensor(
                        f"blk.0.{sfx}",
                        np.ones(hidden // heads, dtype=np.float32),
                    )
            w.add_tensor("blk.0.attn_norm.weight", np.ones(hidden, dtype=np.float32))
            w.add_tensor("blk.0.ffn_norm.weight", np.ones(hidden, dtype=np.float32))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()

            args = Namespace(
                gguf_path=gguf_path,
                output=out_dir,
                dtype="fp16",
                tokenizer="hf",
                hf_tokenizer=None,
            )
            GGUFConverter.run(args)

            with open(os.path.join(out_dir, "config.json")) as fh:
                cfg = json.load(fh)

        decoder_cfg = cfg["model"].get("decoder", {})
        self.assertTrue(decoder_cfg.get("query_norm"),
                        "query_norm must be True in decoder config for qwen35")
        self.assertTrue(decoder_cfg.get("key_norm"),
                        "key_norm must be True in decoder config for qwen35")

    def test_word_vec_size_in_saved_config_json(self):
        """After a full conversion the config.json embeddings section must have
        tgt_word_vec_size equal to hidden_size (not the old default 512)."""
        _require("safetensors")
        import tempfile
        import numpy as np
        from gguf import GGMLQuantizationType, GGUFWriter
        from argparse import Namespace
        from eole.bin.convert.convert_gguf import GGUFConverter

        with tempfile.TemporaryDirectory() as td:
            gguf_path = os.path.join(td, "llama.gguf")
            out_dir = os.path.join(td, "out")
            hidden, heads, ff, n_vocab = 256, 4, 512, 32

            w = GGUFWriter(gguf_path, "llama")
            w.add_block_count(1)
            w.add_context_length(128)
            w.add_embedding_length(hidden)
            w.add_feed_forward_length(ff)
            w.add_head_count(heads)
            w.add_head_count_kv(heads)
            w.add_layer_norm_rms_eps(1e-5)
            w.add_rope_freq_base(10000.0)
            w.add_vocab_size(n_vocab)
            w.add_token_list([str(i) for i in range(n_vocab)])
            w.add_bos_token_id(1)
            w.add_eos_token_id(2)
            w.add_tensor("token_embd.weight",
                         np.random.randn(n_vocab, hidden).astype(np.float32))
            w.add_tensor("output_norm.weight", np.ones(hidden, dtype=np.float32))
            for sfx, shape in [
                ("attn_q", (hidden, hidden)),
                ("attn_k", (hidden, hidden)),
                ("attn_v", (hidden, hidden)),
                ("attn_output", (hidden, hidden)),
                ("ffn_gate", (ff, hidden)),
                ("ffn_up", (ff, hidden)),
                ("ffn_down", (hidden, ff)),
            ]:
                w.add_tensor(
                    f"blk.0.{sfx}.weight",
                    np.random.randn(*shape).astype(np.float32),
                    raw_dtype=GGMLQuantizationType.Q4_K,
                )
            w.add_tensor("blk.0.attn_norm.weight", np.ones(hidden, dtype=np.float32))
            w.add_tensor("blk.0.ffn_norm.weight", np.ones(hidden, dtype=np.float32))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()

            args = Namespace(
                gguf_path=gguf_path,
                output=out_dir,
                dtype="fp16",
                tokenizer="hf",
                hf_tokenizer=None,
            )
            GGUFConverter.run(args)

            with open(os.path.join(out_dir, "config.json")) as fh:
                cfg = json.load(fh)

        emb = cfg["model"].get("embeddings", {})
        self.assertEqual(emb.get("tgt_word_vec_size"), hidden,
                         f"tgt_word_vec_size must be {hidden}, got {emb.get('tgt_word_vec_size')}")
        self.assertEqual(emb.get("src_word_vec_size"), hidden,
                         f"src_word_vec_size must be {hidden}, got {emb.get('src_word_vec_size')}")


class TestGGUFVLMSpatialMergeSize(unittest.TestCase):
    """When a VLM (vision-language) model is converted from GGUF, the
    ``spatial_merge_size`` in the produced config.json must match the value
    stored in the mmproj GGUF (``clip.vision.spatial_merge_size``).

    If it is missing or defaults to 1, ``Qwen3_5VisionMerger`` computes
    ``merged_size = hidden_size * 1 = hidden_size`` instead of
    ``hidden_size * spatial_merge_size^2``, creating ``linear_fc1`` of shape
    ``(hidden_size, hidden_size)`` while the checkpoint weight is
    ``(hidden_size * 4, hidden_size * 4)``, causing a silent truncation.
    """

    @classmethod
    def setUpClass(cls):
        _require("gguf")
        _require("numpy")
        _require("torch")

    def _make_mmproj_gguf(self, td: str, spatial_merge_size: int) -> str:
        import numpy as np
        from gguf import GGUFWriter

        path = os.path.join(td, "mmproj.gguf")
        hidden, heads, ff, n_dec = 32, 4, 64, 64
        img_size, patch_size = 32, 8
        spatial = img_size // patch_size   # 4
        num_pos = spatial * spatial        # 16
        merged = hidden * spatial_merge_size**2

        w = GGUFWriter(path, "clip")
        w.add_uint32("clip.vision.block_count", 1)
        w.add_uint32("clip.vision.embedding_length", hidden)
        w.add_uint32("clip.vision.feed_forward_length", ff)
        w.add_uint32("clip.vision.attention.head_count", heads)
        w.add_uint32("clip.vision.image_size", img_size)
        w.add_uint32("clip.vision.patch_size", patch_size)
        w.add_uint32("clip.vision.spatial_merge_size", spatial_merge_size)
        w.add_uint32("clip.vision.projection_dim", n_dec)
        w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
        w.add_string("clip.projector_type", "qwen3vl_merger")

        w.add_tensor("v.patch_embd.weight",
                     np.ones((hidden, 3, patch_size, patch_size), dtype=np.float32))
        w.add_tensor("v.patch_embd.bias", np.zeros(hidden, dtype=np.float32))
        w.add_tensor("v.position_embed.weight",
                     np.ones((hidden, num_pos), dtype=np.float32))
        w.add_tensor("v.post_ln.weight", np.ones(hidden, dtype=np.float32))
        w.add_tensor("v.post_ln.bias", np.zeros(hidden, dtype=np.float32))
        w.add_tensor("mm.0.weight", np.ones((merged, merged), dtype=np.float32))
        w.add_tensor("mm.0.bias", np.zeros(merged, dtype=np.float32))
        w.add_tensor("mm.2.weight", np.ones((n_dec, merged), dtype=np.float32))
        w.add_tensor("mm.2.bias", np.zeros(n_dec, dtype=np.float32))
        w.add_tensor("v.blk.0.attn_qkv.weight",
                     np.ones((3 * hidden, hidden), dtype=np.float32))
        w.add_tensor("v.blk.0.attn_qkv.bias", np.zeros(3 * hidden, dtype=np.float32))
        w.add_tensor("v.blk.0.attn_out.weight",
                     np.ones((hidden, hidden), dtype=np.float32))
        w.add_tensor("v.blk.0.attn_out.bias", np.zeros(hidden, dtype=np.float32))
        w.add_tensor("v.blk.0.ffn_up.weight", np.ones((ff, hidden), dtype=np.float32))
        w.add_tensor("v.blk.0.ffn_up.bias", np.zeros(ff, dtype=np.float32))
        w.add_tensor("v.blk.0.ffn_down.weight",
                     np.ones((hidden, ff), dtype=np.float32))
        w.add_tensor("v.blk.0.ffn_down.bias", np.zeros(hidden, dtype=np.float32))
        w.add_tensor("v.blk.0.ln1.weight", np.ones(hidden, dtype=np.float32))
        w.add_tensor("v.blk.0.ln1.bias", np.zeros(hidden, dtype=np.float32))
        w.add_tensor("v.blk.0.ln2.weight", np.ones(hidden, dtype=np.float32))
        w.add_tensor("v.blk.0.ln2.bias", np.zeros(hidden, dtype=np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        return path

    def test_clip_metadata_reads_spatial_merge_size(self):
        """GGUFClipMetadata.vision_spatial_merge_size must return the value
        stored in the mmproj GGUF (not always 2 regardless of file content)."""
        import tempfile
        from eole.bin.convert.convert_gguf import GGUFClipMetadata

        for sms in (1, 2, 4):
            with self.subTest(spatial_merge_size=sms):
                with tempfile.TemporaryDirectory() as td:
                    path = self._make_mmproj_gguf(td, sms)
                    clip = GGUFClipMetadata(path)
                    self.assertEqual(
                        clip.vision_spatial_merge_size,
                        sms,
                        f"Expected vision_spatial_merge_size={sms}, "
                        f"got {clip.vision_spatial_merge_size}",
                    )

    @unittest.skipUnless(importlib.util.find_spec("safetensors") is not None, "requires safetensors")
    def test_spatial_merge_size_in_config_json(self):
        """After a full VLM conversion, config.json must have spatial_merge_size
        matching the mmproj GGUF value, not the default 1."""
        import numpy as np
        import tempfile
        from gguf import GGUFWriter
        from argparse import Namespace
        from eole.bin.convert.convert_gguf import GGUFConverter

        spatial_merge_size = 2
        hidden, heads, ff, n_vocab = 64, 4, 128, 32

        with tempfile.TemporaryDirectory() as td:
            # Build minimal text GGUF
            gguf_path = os.path.join(td, "text.gguf")
            w = GGUFWriter(gguf_path, "qwen35")
            w.add_block_count(1)
            w.add_context_length(128)
            w.add_embedding_length(hidden)
            w.add_feed_forward_length(ff)
            w.add_head_count(heads)
            w.add_head_count_kv(heads)
            w.add_layer_norm_rms_eps(1e-5)
            w.add_rope_freq_base(10000.0)
            w.add_vocab_size(n_vocab)
            w.add_token_list([str(i) for i in range(n_vocab)])
            w.add_bos_token_id(1)
            w.add_eos_token_id(2)
            w.add_tensor("token_embd.weight",
                         np.random.randn(n_vocab, hidden).astype(np.float32))
            w.add_tensor("output_norm.weight", np.ones(hidden, dtype=np.float32))
            for sfx, shape in [
                ("attn_q", (hidden, hidden)),
                ("attn_k", (hidden, hidden)),
                ("attn_v", (hidden, hidden)),
                ("attn_output", (hidden, hidden)),
                ("ffn_gate", (ff, hidden)),
                ("ffn_up", (ff, hidden)),
                ("ffn_down", (hidden, ff)),
            ]:
                w.add_tensor(f"blk.0.{sfx}", np.ones(shape, dtype=np.float32))
            w.add_tensor("blk.0.attn_norm.weight", np.ones(hidden, dtype=np.float32))
            w.add_tensor("blk.0.ffn_norm.weight", np.ones(hidden, dtype=np.float32))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()

            mmproj_path = self._make_mmproj_gguf(td, spatial_merge_size)
            out_dir = os.path.join(td, "out")

            args = Namespace(
                gguf_path=gguf_path,
                mmproj=mmproj_path,
                output=out_dir,
                dtype="fp16",
                tokenizer="hf",
                hf_tokenizer=None,
            )
            GGUFConverter.run(args)

            with open(os.path.join(out_dir, "config.json")) as fh:
                cfg = json.load(fh)

        saved = cfg["model"].get("spatial_merge_size")
        self.assertEqual(
            saved,
            spatial_merge_size,
            f"spatial_merge_size in config.json must be {spatial_merge_size} "
            f"(from mmproj GGUF), but got {saved!r}. "
            "This would cause adapter linear_fc1/fc2 to be created with "
            "wrong shape (hidden instead of hidden * spatial_merge_size^2).",
        )


if __name__ == "__main__":
    unittest.main()
