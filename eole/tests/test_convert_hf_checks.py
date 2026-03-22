"""Tests for the post-conversion validation helpers added to convert_HF.py."""

import os
import json
import tempfile
import unittest

import torch
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Helpers to build minimal safetensors files without needing a full HF model
# ---------------------------------------------------------------------------

def _write_safetensors(path, tensors: dict):
    """Write a minimal safetensors file using the official library."""
    save_file(tensors, path)


# ---------------------------------------------------------------------------
# Minimal stub for HuggingfaceFiles used by the functions under test
# ---------------------------------------------------------------------------

class _FakeHF:
    """Minimal stand-in for HuggingfaceFiles used by the validation helpers."""

    def __init__(self, model_path=None, wmap_path=None, wmap=None, base_dir=None):
        self.model_path = model_path
        self.wmap_path = wmap_path
        self._wmap = wmap  # dict with "weight_map" key
        self.base_dir = base_dir

    @property
    def wmap(self):
        return self._wmap

    def get_load_ckpt(self, dir_path, file_path):
        """Return the safetensors path (mirrors real HuggingfaceFiles behaviour)."""
        full = os.path.join(dir_path, file_path)
        # safetensors files are returned as a path; .bin would be torch.load'd
        return full


# ---------------------------------------------------------------------------
# Import the functions under test
# ---------------------------------------------------------------------------

from eole.bin.convert.convert_HF import (
    _collect_all_source_keys,
    check_conversion_completeness,
    check_conversion_equality,
)


class TestCollectAllSourceKeys(unittest.TestCase):
    """Tests for _collect_all_source_keys."""

    def test_single_file(self):
        """Keys should be collected from a single safetensors file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.safetensors")
            tensors = {
                "model.embed_tokens.weight": torch.ones(32, 8),
                "model.norm.weight": torch.ones(8),
            }
            _write_safetensors(path, tensors)

            hf = _FakeHF(model_path=path)
            keys = _collect_all_source_keys(hf)

        self.assertEqual(keys, set(tensors.keys()))

    def test_wmap(self):
        """Keys should be read from the wmap JSON without loading checkpoint files."""
        expected_keys = {
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.norm.weight",
        }
        wmap = {"weight_map": {k: "some_file.safetensors" for k in expected_keys}}

        hf = _FakeHF(wmap_path="/fake/index.json", wmap=wmap)
        keys = _collect_all_source_keys(hf)

        self.assertEqual(keys, expected_keys)


class TestCheckConversionCompleteness(unittest.TestCase):
    """Tests for check_conversion_completeness."""

    def test_all_converted(self):
        """Should return an empty set when every source key was consumed."""
        src = {"a", "b", "c"}
        result = check_conversion_completeness(src, src.copy())
        self.assertEqual(result, set())

    def test_some_unconverted(self):
        """Should return the set of keys that were not consumed."""
        src = {"a", "b", "c", "d"}
        consumed = {"a", "c"}
        result = check_conversion_completeness(src, consumed)
        self.assertEqual(result, {"b", "d"})

    def test_empty_source(self):
        """Should cope gracefully with an empty source set."""
        result = check_conversion_completeness(set(), set())
        self.assertEqual(result, set())


class TestCheckConversionEquality(unittest.TestCase):
    """Tests for check_conversion_equality."""

    def _write_ckpt(self, tmpdir, name, tensors):
        path = os.path.join(tmpdir, name)
        _write_safetensors(path, tensors)
        return path

    def test_no_mismatch_simple(self):
        """Tensors copied without transformation should match exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            src_path = self._write_ckpt(tmpdir, "src.safetensors", {"model.norm.weight": src_t})
            out_path = self._write_ckpt(tmpdir, "model.00.safetensors", {"decoder.layer_norm.weight": src_t})

            hf = _FakeHF(model_path=src_path)
            details = [
                {
                    "shard_path": out_path,
                    "eole_key": "decoder.layer_norm.weight",
                    "srckey": "model.norm.weight",
                    "srcmap": None,
                    "context": {},
                    "special": None,
                }
            ]
            mismatches = check_conversion_equality(hf, details, target_dtype=None)

        self.assertEqual(mismatches, [])

    def test_mismatch_detected(self):
        """A difference in values should be reported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_t = torch.ones(4, dtype=torch.float32)
            out_t = torch.zeros(4, dtype=torch.float32)  # deliberately wrong
            src_path = self._write_ckpt(tmpdir, "src.safetensors", {"model.norm.weight": src_t})
            out_path = self._write_ckpt(tmpdir, "model.00.safetensors", {"decoder.layer_norm.weight": out_t})

            hf = _FakeHF(model_path=src_path)
            details = [
                {
                    "shard_path": out_path,
                    "eole_key": "decoder.layer_norm.weight",
                    "srckey": "model.norm.weight",
                    "srcmap": None,
                    "context": {},
                    "special": None,
                }
            ]
            mismatches = check_conversion_equality(hf, details, target_dtype=None)

        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0][0], "decoder.layer_norm.weight")

    def test_with_srcmap_transformation(self):
        """Sliced tensors should still compare correctly after re-applying srcmap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hidden_size = 4
            # Simulate a QKV-packed weight: shape (3*hidden_size, hidden_size)
            qkv = torch.arange(float(3 * hidden_size * hidden_size)).reshape(3 * hidden_size, hidden_size)
            q_expected = qkv[:hidden_size, :]

            src_path = self._write_ckpt(tmpdir, "src.safetensors", {"model.layers.0.self_attn.qkv_proj.weight": qkv})
            out_path = self._write_ckpt(
                tmpdir, "model.00.safetensors", {"decoder.transformer_layers.0.self_attn.linear_query.weight": q_expected}
            )

            hf = _FakeHF(model_path=src_path)
            details = [
                {
                    "shard_path": out_path,
                    "eole_key": "decoder.transformer_layers.0.self_attn.linear_query.weight",
                    "srckey": "model.layers.0.self_attn.qkv_proj.weight",
                    "srcmap": "[:hidden_size, :]",
                    "context": {"hidden_size": hidden_size, "transformer_ff": 16},
                    "special": None,
                }
            ]
            mismatches = check_conversion_equality(hf, details, target_dtype=None)

        self.assertEqual(mismatches, [])

    def test_with_dtype_conversion(self):
        """Tensors cast to a different dtype should match after the same cast."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_t = torch.arange(8, dtype=torch.float32)
            out_t = src_t.to(torch.float16)

            src_path = self._write_ckpt(tmpdir, "src.safetensors", {"a": src_t})
            out_path = self._write_ckpt(tmpdir, "model.00.safetensors", {"b": out_t})

            hf = _FakeHF(model_path=src_path)
            details = [
                {
                    "shard_path": out_path,
                    "eole_key": "b",
                    "srckey": "a",
                    "srcmap": None,
                    "context": {},
                    "special": None,
                }
            ]
            mismatches = check_conversion_equality(hf, details, target_dtype=torch.float16)

        self.assertEqual(mismatches, [])

    def test_unsqueeze_special(self):
        """Tensors stored with an extra leading dimension should verify correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_t = torch.ones(4, 4, dtype=torch.float32)
            out_t = src_t.unsqueeze(0)

            src_path = self._write_ckpt(tmpdir, "src.safetensors", {"encoder.class_embedding": src_t})
            out_path = self._write_ckpt(
                tmpdir, "model.00.safetensors", {"encoder.class_embedding.weight": out_t}
            )

            hf = _FakeHF(model_path=src_path)
            details = [
                {
                    "shard_path": out_path,
                    "eole_key": "encoder.class_embedding.weight",
                    "srckey": "encoder.class_embedding",
                    "srcmap": None,
                    "context": {},
                    "special": "unsqueeze(0)",
                }
            ]
            mismatches = check_conversion_equality(hf, details, target_dtype=None)

        self.assertEqual(mismatches, [])

    def test_wmap_lookup(self):
        """Equality check should resolve the checkpoint via the wmap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_t = torch.ones(4, dtype=torch.float32)
            ckpt_name = "model-00001-of-00002.safetensors"
            src_path = self._write_ckpt(tmpdir, ckpt_name, {"model.norm.weight": src_t})
            out_path = self._write_ckpt(tmpdir, "model.00.safetensors", {"decoder.layer_norm.weight": src_t})

            wmap = {"weight_map": {"model.norm.weight": ckpt_name}}
            hf = _FakeHF(wmap_path=os.path.join(tmpdir, "index.json"), wmap=wmap, base_dir=tmpdir)

            details = [
                {
                    "shard_path": out_path,
                    "eole_key": "decoder.layer_norm.weight",
                    "srckey": "model.norm.weight",
                    "srcmap": None,
                    "context": {},
                    "special": None,
                }
            ]
            mismatches = check_conversion_equality(hf, details, target_dtype=None)

        self.assertEqual(mismatches, [])

    def test_empty_details(self):
        """No conversion details should produce no mismatches."""
        hf = _FakeHF(model_path="/does/not/matter")
        mismatches = check_conversion_equality(hf, [], target_dtype=None)
        self.assertEqual(mismatches, [])


if __name__ == "__main__":
    unittest.main()
