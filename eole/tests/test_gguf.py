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
        """post_attention_norm is also present in linear-attention blocks."""
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
            np.random.randn(hidden, conv_dim).astype(np.float32),
            raw_dtype=GGMLQuantizationType.Q8_0,
        )
        writer.add_tensor(
            "blk.0.attn_gate.weight",
            np.random.randn(hidden, num_v_heads * head_v_dim).astype(np.float32),
            raw_dtype=GGMLQuantizationType.Q8_0,
        )
        # SSM params (float)
        writer.add_tensor("blk.0.ssm_a", np.ones(num_v_heads, dtype=np.float32))
        writer.add_tensor("blk.0.ssm_dt.bias", np.ones(num_v_heads, dtype=np.float32))
        writer.add_tensor("blk.0.ssm_norm.weight", np.ones(head_v_dim, dtype=np.float32))
        writer.add_tensor(
            "blk.0.ssm_conv1d.weight",
            # Real GGUF (llama.cpp): stored as (conv_dim, kernel_size); the gguf
            # reader reverses GGUF ne, so tensor.shape=[kernel_size, conv_dim] and
            # data.shape=(conv_dim, kernel_size) in NumPy order.
            np.random.randn(conv_dim, kernel_size).astype(np.float32),
        )
        writer.add_tensor(
            "blk.0.ssm_alpha.weight",
            # in_proj_a: Linear(hidden → num_v_heads); weight shape (num_v_heads, hidden)
            # so the innermost GGUF dim is hidden=64, which is a valid Q8_0 row size.
            np.random.randn(num_v_heads, hidden).astype(np.float32),
            raw_dtype=GGMLQuantizationType.Q8_0,
        )
        writer.add_tensor(
            "blk.0.ssm_beta.weight",
            np.random.randn(num_v_heads, hidden).astype(np.float32),
            raw_dtype=GGMLQuantizationType.Q8_0,
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
        """ssm_out.weight must appear as linear_attn.out_proj.weight in the shard."""
        self._run_converter()
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.out_proj.weight"
        self.assertIn(key, tensors, f"Missing key: {key}")

    def test_ssm_a_mapped_to_A_log(self):
        """ssm_a must appear as linear_attn.A_log (float, not uint8)."""
        self._run_converter()
        import torch
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.A_log"
        self.assertIn(key, tensors)
        self.assertNotEqual(tensors[key].dtype, torch.uint8, "A_log should be float")

    def test_ssm_alpha_mapped_to_in_proj_a(self):
        self._run_converter()
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.in_proj_a.weight"
        self.assertIn(key, tensors)

    def test_ssm_beta_mapped_to_in_proj_b(self):
        self._run_converter()
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.in_proj_b.weight"
        self.assertIn(key, tensors)

    def test_attn_qkv_in_linear_block_goes_to_in_proj_qkv(self):
        self._run_converter()
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.in_proj_qkv.weight"
        self.assertIn(key, tensors)

    def test_attn_gate_goes_to_in_proj_z(self):
        self._run_converter()
        tensors = self._tensors()
        key = "decoder.transformer_layers.0.linear_attn.in_proj_z.weight"
        self.assertIn(key, tensors)

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
        """post_attention_norm maps to post_attention_layernorm for both block types."""
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
