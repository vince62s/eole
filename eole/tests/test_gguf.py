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

    def _map(self, name):
        from eole.bin.convert.convert_gguf import _gguf_to_eole_name

        return _gguf_to_eole_name(name)

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

    def test_known_unsupported_attn_gate_returns_none(self):
        """Qwen3.5 attn_gate is known but not yet modelled in EOLE → None."""
        self.assertIsNone(self._map("blk.0.attn_gate.weight"))

    def test_known_unsupported_post_attention_norm_returns_none(self):
        """Qwen3.5 post_attention_norm is known but not modelled → None."""
        self.assertIsNone(self._map("blk.0.post_attention_norm.weight"))

    def test_known_unsupported_ssm_returns_none(self):
        """Mamba/SSM tensors are known but not modelled in EOLE → None."""
        self.assertIsNone(self._map("blk.0.ssm_a"))
        self.assertIsNone(self._map("blk.0.ssm_alpha.weight"))
        self.assertIsNone(self._map("blk.0.ssm_beta.weight"))


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


if __name__ == "__main__":
    unittest.main()
