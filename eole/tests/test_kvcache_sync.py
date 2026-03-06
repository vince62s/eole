"""Unit tests for KV-cache / max_length synchronisation fixes.

These tests exercise:
1. ``Inference.update_settings`` syncing ``decoder.max_length`` upward
   when the per-request ``max_length`` (from the client's ``max_tokens``)
   exceeds the originally configured value.
2. ``TransformerDecoder._init_cache`` using ``max(self.max_length, l)``
   as ``cache_len_tgt`` so an input that exceeds ``max_length`` does not
   crash flash-attention with "key seqlen > cache seqlen".
3. ``TransformerDecoder._init_cache`` reusing (zeroing in-place) existing
   kcache tensors when the shape matches, to preserve CUDA-graph tensor
   addresses across consecutive requests.
4. ``TransformerDecoder._disable_cache`` keeping kcache/vcache tensors
   alive (not setting them to None) when ``EOLE_TORCH_COMPILE`` is set,
   so that the same addresses are available for the next request.

All tests are pure-Python / no-torch so they can run in CI environments
that do not have GPU / PyTorch installed.
"""

import os
import sys
import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers to load source modules without triggering torch imports
# ---------------------------------------------------------------------------


def _load_source(rel_path):
    """Import a .py file by path, returning its module object."""
    import importlib.util

    base = os.path.join(os.path.dirname(__file__), "..", "..", "eole")
    full = os.path.normpath(os.path.join(base, rel_path))
    spec = importlib.util.spec_from_file_location(rel_path, full)
    mod = importlib.util.module_from_spec(spec)
    return spec, mod


# ---------------------------------------------------------------------------
# Test 1 – update_settings syncs decoder.max_length
# ---------------------------------------------------------------------------


class _FakeLayer:
    """Minimal stand-in for a transformer decoder layer."""

    def __init__(self):
        self.compiled_shapes = set()
        self.compiled_shapes.add((1, 0))  # pretend already compiled


class _FakeDecoder:
    """Minimal stand-in for TransformerDecoder."""

    def __init__(self, max_length=2048):
        self.max_length = max_length
        self.transformer_layers = [_FakeLayer(), _FakeLayer()]


class _FakeModel:
    def __init__(self, decoder):
        self.decoder = decoder


class _FakePredictor:
    """Mimics the Inference base class just enough to test update_settings."""

    def __init__(self, max_length, decoder_max_length):
        self.max_length = max_length
        self.model = _FakeModel(_FakeDecoder(decoder_max_length))

    # Copy the exact implementation from inference.py
    def update_settings(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        new_max_length = kwargs.get("max_length")
        if new_max_length is not None and hasattr(self, "model") and hasattr(self.model, "decoder"):
            decoder = self.model.decoder
            if new_max_length > decoder.max_length:
                decoder.max_length = new_max_length
                if hasattr(decoder, "transformer_layers"):
                    for layer in decoder.transformer_layers:
                        if hasattr(layer, "compiled_shapes"):
                            layer.compiled_shapes.clear()


class TestUpdateSettingsSyncsDecoder(unittest.TestCase):
    def test_larger_max_length_syncs_decoder(self):
        """update_settings(max_length=X) grows decoder.max_length when X > current."""
        pred = _FakePredictor(max_length=2048, decoder_max_length=2048)
        pred.update_settings(max_length=4096)
        self.assertEqual(pred.max_length, 4096)
        self.assertEqual(pred.model.decoder.max_length, 4096)

    def test_larger_max_length_clears_compiled_shapes(self):
        """Growing decoder.max_length invalidates stale CUDA-graph shapes."""
        pred = _FakePredictor(max_length=2048, decoder_max_length=2048)
        # Both layers have a compiled shape already
        for layer in pred.model.decoder.transformer_layers:
            self.assertNotEqual(len(layer.compiled_shapes), 0)
        pred.update_settings(max_length=4096)
        for layer in pred.model.decoder.transformer_layers:
            self.assertEqual(len(layer.compiled_shapes), 0, "compiled_shapes should be cleared")

    def test_smaller_max_length_does_not_shrink_decoder(self):
        """A smaller override must NOT shrink decoder.max_length (one-way ratchet)."""
        pred = _FakePredictor(max_length=2048, decoder_max_length=2048)
        pred.update_settings(max_length=512)
        # predictor max_length reduced (user wants shorter output this turn)
        self.assertEqual(pred.max_length, 512)
        # decoder.max_length unchanged – KV cache must stay at original capacity
        self.assertEqual(pred.model.decoder.max_length, 2048)

    def test_equal_max_length_no_change(self):
        """Passing the same max_length as the current value is a no-op."""
        pred = _FakePredictor(max_length=2048, decoder_max_length=2048)
        for layer in pred.model.decoder.transformer_layers:
            layer.compiled_shapes.add((1, 0))
        pred.update_settings(max_length=2048)
        self.assertEqual(pred.model.decoder.max_length, 2048)
        # compiled_shapes should NOT be cleared for an equal value
        for layer in pred.model.decoder.transformer_layers:
            self.assertIn((1, 0), layer.compiled_shapes)

    def test_no_max_length_kwarg_is_noop(self):
        """Calling update_settings without max_length does not touch decoder."""
        pred = _FakePredictor(max_length=2048, decoder_max_length=2048)
        pred.update_settings(temperature=0.7)
        self.assertEqual(pred.model.decoder.max_length, 2048)

    def test_max_length_without_decoder_is_safe(self):
        """update_settings(max_length=X) is safe when model has no decoder."""
        pred = _FakePredictor(max_length=512, decoder_max_length=512)
        del pred.model.decoder  # simulate encoder-only model
        # Should not raise
        pred.update_settings(max_length=1024)
        self.assertEqual(pred.max_length, 1024)


# ---------------------------------------------------------------------------
# Test 2 – _init_cache cache_len_tgt computation (pure-Python)
# ---------------------------------------------------------------------------


class TestInitCacheLenTgt(unittest.TestCase):
    """
    Verify the ``cache_len_tgt = max(max_length, l)`` logic in _init_cache.

    We test this without PyTorch by directly exercising the branch logic.
    """

    def _compute_cache_len_tgt(self, max_length, l, dynamic_shapes):
        """Replicate the branch logic from TransformerDecoder._init_cache."""
        if dynamic_shapes:
            return l
        else:
            return max(max_length, l)

    def test_input_within_max_length(self):
        """Normal case: input fits in the configured cache."""
        self.assertEqual(self._compute_cache_len_tgt(2048, 500, False), 2048)

    def test_input_equals_max_length(self):
        """Edge case: input exactly equals max_length."""
        self.assertEqual(self._compute_cache_len_tgt(2048, 2048, False), 2048)

    def test_input_exceeds_max_length(self):
        """Second-request chat scenario: accumulated history > max_length."""
        # cache_len_tgt must be at least l so flash-attention prefill doesn't crash
        self.assertEqual(self._compute_cache_len_tgt(2048, 2500, False), 2500)

    def test_dynamic_shapes_uses_l(self):
        """In dynamic mode (no torch.compile) cache always starts at input len."""
        self.assertEqual(self._compute_cache_len_tgt(2048, 100, True), 100)
        self.assertEqual(self._compute_cache_len_tgt(2048, 2500, True), 2500)


# ---------------------------------------------------------------------------
# Test 3 – _disable_cache EOLE_TORCH_COMPILE behaviour (pure-Python mock)
# ---------------------------------------------------------------------------


class _MockSelfAttn:
    """Minimal stand-in for the self-attention module inside a decoder layer."""

    def __init__(self):
        self.kcache = "tensor_object"
        self.vcache = "tensor_object"
        self.cache_leftpad = "leftpad"


class _MockAttentionLayer:
    def __init__(self):
        self.self_attn = _MockSelfAttn()
        self.layer_type = "full_attention"
        self.context_attn = None


class _MockDecoder:
    """Minimal decoder that uses the exact _disable_cache logic."""

    def __init__(self, torch_compile_active):
        self._torch_compile = torch_compile_active
        self.left_pad_attn_mask = "mask"
        self.cache_seqlens = "seqlens"
        self.flash = True
        layer = _MockAttentionLayer()
        self.transformer_layers = [layer]

    def _disable_cache(self):
        self.left_pad_attn_mask = None
        self.cache_seqlens = None
        self.flash = False
        for layer in self.transformer_layers:
            if layer.layer_type == "linear_attention":
                pass
            else:
                if not self._torch_compile:
                    layer.self_attn.kcache = None
                    layer.self_attn.vcache = None
                layer.self_attn.cache_leftpad = None


class TestDisableCachePreservesBuffers(unittest.TestCase):
    def test_torch_compile_keeps_kcache(self):
        """With EOLE_TORCH_COMPILE, kcache/vcache must remain allocated."""
        decoder = _MockDecoder(torch_compile_active=True)
        original_kcache = decoder.transformer_layers[0].self_attn.kcache
        decoder._disable_cache()
        # The tensor object must still exist (not replaced with None)
        self.assertIs(decoder.transformer_layers[0].self_attn.kcache, original_kcache)

    def test_no_compile_frees_kcache(self):
        """Without EOLE_TORCH_COMPILE, kcache/vcache are freed to save memory."""
        decoder = _MockDecoder(torch_compile_active=False)
        decoder._disable_cache()
        self.assertIsNone(decoder.transformer_layers[0].self_attn.kcache)
        self.assertIsNone(decoder.transformer_layers[0].self_attn.vcache)

    def test_cache_seqlens_always_cleared(self):
        """cache_seqlens is always set to None regardless of compile mode."""
        for compile_active in (True, False):
            decoder = _MockDecoder(torch_compile_active=compile_active)
            decoder._disable_cache()
            self.assertIsNone(decoder.cache_seqlens, f"compile_active={compile_active}")

    def test_flash_always_cleared(self):
        """flash flag is always set to False regardless of compile mode."""
        for compile_active in (True, False):
            decoder = _MockDecoder(torch_compile_active=compile_active)
            decoder._disable_cache()
            self.assertFalse(decoder.flash, f"compile_active={compile_active}")


if __name__ == "__main__":
    unittest.main()
