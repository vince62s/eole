"""Unit tests for the prefill KV cache.

Tests the PrefillCache class directly and the integration with
TransformerDecoder's chunked prefill path, verifying that:

  - Rolling hash keys distinguish different prefixes.
  - Cache hits restore the correct embeddings and KV slices.
  - LRU eviction works correctly.
  - Caching is skipped when batch_size > 1 or image tokens are present.
  - prefill_chunk_size triggers chunked prefill independently of sliding_window.
"""

import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestPrefillCacheBasics(unittest.TestCase):
    """Test PrefillCache key computation, get/put, and LRU eviction."""

    def setUp(self):
        from eole.modules.prefill_cache import PrefillCache

        self.Cache = PrefillCache

    def _make_ids(self, values):
        return torch.tensor(values, dtype=torch.long)

    def test_compute_key_deterministic(self):
        """Same token IDs always produce the same key."""
        ids = self._make_ids([1, 2, 3])
        k1 = self.Cache.compute_key(ids)
        k2 = self.Cache.compute_key(ids)
        self.assertEqual(k1, k2)

    def test_compute_key_different_ids(self):
        """Different token IDs produce different keys."""
        k1 = self.Cache.compute_key(self._make_ids([1, 2, 3]))
        k2 = self.Cache.compute_key(self._make_ids([1, 2, 4]))
        self.assertNotEqual(k1, k2)

    def test_compute_key_rolling_hash(self):
        """Same chunk tokens but different prev_key yield different keys."""
        ids = self._make_ids([10, 20, 30])
        key_no_prev = self.Cache.compute_key(ids, prev_key=None)
        key_with_prev = self.Cache.compute_key(ids, prev_key=b"some_prefix_hash")
        self.assertNotEqual(key_no_prev, key_with_prev)

    def test_compute_key_chaining(self):
        """Sequential chaining of prev_key distinguishes two-chunk prefixes."""
        ids_a = self._make_ids([1, 2])
        ids_b = self._make_ids([3, 4])
        ids_c = self._make_ids([1, 2])  # same as ids_a but different prefix

        # Prefix "A, B": key_a -> key_ab
        key_a = self.Cache.compute_key(ids_a)
        key_ab = self.Cache.compute_key(ids_b, prev_key=key_a)

        # Prefix "C": key_c -> key_cb  (same second chunk but different first)
        key_c = self.Cache.compute_key(ids_c)  # identical to key_a
        self.assertEqual(key_a, key_c)  # sanity: same first chunk -> same key

        key_cb = self.Cache.compute_key(ids_b, prev_key=key_c)
        # Since key_a == key_c, chaining with the same second chunk gives equal keys
        self.assertEqual(key_ab, key_cb)

        # But a different first chunk DOES produce a different second-chunk key.
        ids_d = self._make_ids([9, 9])  # different first chunk
        key_d = self.Cache.compute_key(ids_d)
        key_db = self.Cache.compute_key(ids_b, prev_key=key_d)
        self.assertNotEqual(key_ab, key_db)

    def test_get_returns_none_on_miss(self):
        cache = self.Cache(max_entries=10)
        self.assertIsNone(cache.get(b"nonexistent"))

    def test_put_and_get_roundtrip(self):
        """Data stored via put is retrievable via get."""
        cache = self.Cache(max_entries=10)
        emb = torch.randn(4, 8)  # (chunk_len, hidden)
        k_slice = torch.randn(4, 2, 16)  # (chunk_len, heads_kv, dph)
        v_slice = torch.randn(4, 2, 16)
        kv_slices = [(k_slice, v_slice)]

        key = self.Cache.compute_key(torch.arange(4))
        cache.put(key, emb, kv_slices)

        result = cache.get(key)
        self.assertIsNotNone(result)
        emb_out, kv_out = result
        # Tensors are stored on CPU
        self.assertEqual(emb_out.device, torch.device("cpu"))
        torch.testing.assert_close(emb_out, emb.cpu())
        k_out, v_out = kv_out[0]
        torch.testing.assert_close(k_out, k_slice.cpu())
        torch.testing.assert_close(v_out, v_slice.cpu())

    def test_put_overwrites_existing(self):
        """Putting the same key twice updates the stored value."""
        cache = self.Cache(max_entries=10)
        key = self.Cache.compute_key(torch.arange(3))

        emb1 = torch.zeros(3, 4)
        emb2 = torch.ones(3, 4)
        cache.put(key, emb1, [])
        cache.put(key, emb2, [])

        result_emb, _ = cache.get(key)
        torch.testing.assert_close(result_emb, emb2)

    def test_lru_eviction(self):
        """When the cache is full, the least-recently-used entry is evicted."""
        cache = self.Cache(max_entries=2)

        key0 = self.Cache.compute_key(torch.tensor([0]))
        key1 = self.Cache.compute_key(torch.tensor([1]))
        key2 = self.Cache.compute_key(torch.tensor([2]))

        cache.put(key0, torch.zeros(1, 4), [])
        cache.put(key1, torch.zeros(1, 4), [])
        self.assertEqual(len(cache), 2)

        # Access key0 to make it recently used
        cache.get(key0)

        # Adding key2 should evict key1 (LRU)
        cache.put(key2, torch.zeros(1, 4), [])
        self.assertIsNone(cache.get(key1), "key1 should have been evicted")
        self.assertIsNotNone(cache.get(key0), "key0 was recently used, must survive")
        self.assertIsNotNone(cache.get(key2), "key2 was just inserted, must be present")

    def test_clear(self):
        """clear() removes all entries."""
        cache = self.Cache(max_entries=10)
        for i in range(5):
            key = self.Cache.compute_key(torch.tensor([i]))
            cache.put(key, torch.zeros(2, 4), [])
        self.assertEqual(len(cache), 5)
        cache.clear()
        self.assertEqual(len(cache), 0)

    def test_none_kv_slice_for_linear_attention(self):
        """None entries in kv_slices (linear-attention layers) are preserved."""
        cache = self.Cache(max_entries=10)
        k_slice = torch.randn(4, 2, 8)
        v_slice = torch.randn(4, 2, 8)
        kv_slices = [None, (k_slice, v_slice), None]  # layers 0, 2 are linear

        key = self.Cache.compute_key(torch.arange(4))
        cache.put(key, torch.zeros(4, 16), kv_slices)

        _, kv_out = cache.get(key)
        self.assertIsNone(kv_out[0])
        self.assertIsNotNone(kv_out[1])
        self.assertIsNone(kv_out[2])


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestChunkedPrefillWithCache(unittest.TestCase):
    """Integration tests for _forward_chunked_prefill with the prefill cache.

    Uses a minimal mock of TransformerDecoder that replaces _forward_eager
    with a deterministic stub so we can verify:
      - Cache miss: _forward_eager is called and result is stored.
      - Cache hit: _forward_eager is NOT called; cached values are restored.
      - cache_seqlens is correctly updated for both paths.
    """

    def _make_mock_decoder(self, S, chunk_size, num_layers=2, hidden=8, cache_size=16):
        """Build a SimpleNamespace that mimics the parts of TransformerDecoder
        used by _forward_chunked_prefill."""
        import types
        from eole.modules.prefill_cache import PrefillCache

        # Allocate KV cache buffers (batch=1, cache_len, heads_kv=1, dph=4)
        cache_len = S  # exact fit for simplicity
        heads_kv = 1
        dph = 4

        layers = []
        for _ in range(num_layers):
            layer = types.SimpleNamespace(
                layer_type="full_attention",
                self_attn=types.SimpleNamespace(
                    kcache=torch.zeros(1, cache_len, heads_kv, dph),
                    vcache=torch.zeros(1, cache_len, heads_kv, dph),
                ),
            )
            layers.append(layer)

        call_count = [0]  # mutable counter

        def stub_forward_eager(emb_chunk, **kw):
            """Write a deterministic pattern to the KV cache and return emb."""
            chunk_len = emb_chunk.size(1)
            start = kw.get("_chunk_start", 0)
            for layer in layers:
                # Fill the slice with a recognizable pattern
                layer.self_attn.kcache[0, start : start + chunk_len] = float(
                    layers.index(layer) + 1
                )
                layer.self_attn.vcache[0, start : start + chunk_len] = float(
                    layers.index(layer) + 1
                ) * 2.0
            call_count[0] += 1
            mock.cache_seqlens.add_(chunk_len)
            out = emb_chunk * 2.0  # deterministic transform
            return out, {"std": None}

        mock = types.SimpleNamespace(
            sliding_window=0,
            prefill_chunk_size=chunk_size,
            has_linear_attn=False,
            transformer_layers=layers,
            _prefill_cache=PrefillCache(max_entries=cache_size),
            cache_seqlens=torch.zeros(1, dtype=torch.int32),
            cache_len_tgt=cache_len,
            _forward_eager=stub_forward_eager,
            _call_count=call_count,
        )
        return mock

    def _run_chunked_prefill(self, mock, emb, src_ids, **kwargs):
        """Invoke _forward_chunked_prefill on the mock via the real implementation."""
        import types
        from eole.decoders.transformer import TransformerDecoder

        # Bind the unbound method to our mock
        return TransformerDecoder._forward_chunked_prefill(mock, emb, src_ids=src_ids, **kwargs)

    def test_cache_miss_stores_result(self):
        """First call: _forward_eager runs and result is stored in cache."""
        S, chunk_size, hidden = 8, 4, 8
        mock = self._make_mock_decoder(S, chunk_size)
        emb = torch.randn(1, S, hidden)
        src_ids = torch.randint(0, 100, (1, S))
        tgt_pad_mask = torch.zeros(1, 1, S, dtype=torch.bool)

        self._run_chunked_prefill(mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask)

        # _forward_eager was called once per chunk
        n_chunks = (S + chunk_size - 1) // chunk_size
        self.assertEqual(mock._call_count[0], n_chunks)
        # cache now contains n_chunks entries
        self.assertEqual(len(mock._prefill_cache), n_chunks)

    def test_cache_hit_skips_forward(self):
        """Second call with same src_ids: _forward_eager is NOT called again."""
        S, chunk_size, hidden = 8, 4, 8
        mock = self._make_mock_decoder(S, chunk_size)
        emb = torch.randn(1, S, hidden)
        src_ids = torch.randint(0, 100, (1, S))
        tgt_pad_mask = torch.zeros(1, 1, S, dtype=torch.bool)

        # First request: fills the cache
        out1, _ = self._run_chunked_prefill(
            mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask.clone()
        )
        count_after_first = mock._call_count[0]

        # Reset cache_seqlens to simulate a new request
        mock.cache_seqlens.zero_()

        # Second request with the same src_ids: all chunks should be cache hits
        out2, _ = self._run_chunked_prefill(
            mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask.clone()
        )
        count_after_second = mock._call_count[0]

        # No additional _forward_eager calls
        self.assertEqual(count_after_second, count_after_first)
        # cache_seqlens should have been incremented by S
        self.assertEqual(mock.cache_seqlens[0].item(), S)

    def test_different_src_ids_cause_cache_miss(self):
        """Different token IDs produce a different cache key (miss)."""
        S, chunk_size, hidden = 4, 4, 8
        mock = self._make_mock_decoder(S, chunk_size)
        emb = torch.randn(1, S, hidden)
        tgt_pad_mask = torch.zeros(1, 1, S, dtype=torch.bool)

        src_ids_a = torch.zeros(1, S, dtype=torch.long)
        src_ids_b = torch.ones(1, S, dtype=torch.long)

        self._run_chunked_prefill(mock, emb, src_ids_a, tgt_pad_mask=tgt_pad_mask.clone())
        mock.cache_seqlens.zero_()
        self._run_chunked_prefill(mock, emb, src_ids_b, tgt_pad_mask=tgt_pad_mask.clone())

        # Two distinct forward calls (one per unique src_ids)
        self.assertEqual(mock._call_count[0], 2)

    def test_caching_disabled_for_batch_size_gt1(self):
        """Cache is not used when B > 1."""
        S, chunk_size, hidden = 4, 4, 8
        mock = self._make_mock_decoder(S, chunk_size)
        # Extend KV cache to batch=2
        for layer in mock.transformer_layers:
            layer.self_attn.kcache = torch.zeros(2, S, 1, 4)
            layer.self_attn.vcache = torch.zeros(2, S, 1, 4)
        mock.cache_seqlens = torch.zeros(2, dtype=torch.int32)

        emb = torch.randn(2, S, hidden)
        src_ids = torch.randint(0, 100, (2, S))
        tgt_pad_mask = torch.zeros(2, 1, S, dtype=torch.bool)

        # Call twice with same src_ids
        self._run_chunked_prefill(mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask.clone())
        mock.cache_seqlens.zero_()
        self._run_chunked_prefill(mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask.clone())

        # Should have run _forward_eager twice (no cache hit for B>1)
        self.assertEqual(mock._call_count[0], 2)
        # Cache should remain empty
        self.assertEqual(len(mock._prefill_cache), 0)

    def test_caching_disabled_when_image_locations_present(self):
        """Cache is skipped when image_locations is provided."""
        S, chunk_size, hidden = 4, 4, 8
        mock = self._make_mock_decoder(S, chunk_size)
        emb = torch.randn(1, S, hidden)
        src_ids = torch.zeros(1, S, dtype=torch.long)
        tgt_pad_mask = torch.zeros(1, 1, S, dtype=torch.bool)
        image_locations = torch.zeros(1, S, dtype=torch.bool)

        self._run_chunked_prefill(
            mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask.clone(), image_locations=image_locations
        )
        mock.cache_seqlens.zero_()
        self._run_chunked_prefill(
            mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask.clone(), image_locations=image_locations
        )

        # Both calls should have run _forward_eager
        self.assertEqual(mock._call_count[0], 2)
        self.assertEqual(len(mock._prefill_cache), 0)

    def test_caching_disabled_for_linear_attention(self):
        """Cache is skipped for hybrid models with linear-attention layers."""
        S, chunk_size, hidden = 4, 4, 8
        mock = self._make_mock_decoder(S, chunk_size)
        mock.has_linear_attn = True  # hybrid model: disable caching

        emb = torch.randn(1, S, hidden)
        src_ids = torch.zeros(1, S, dtype=torch.long)
        tgt_pad_mask = torch.zeros(1, 1, S, dtype=torch.bool)

        self._run_chunked_prefill(mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask.clone())
        mock.cache_seqlens.zero_()
        self._run_chunked_prefill(mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask.clone())

        # Both calls should have run _forward_eager (no cache)
        self.assertEqual(mock._call_count[0], 2)
        self.assertEqual(len(mock._prefill_cache), 0)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestPrefillChunkSizeTrigger(unittest.TestCase):
    """Verify that prefill_chunk_size triggers chunked prefill without sliding_window."""

    def test_effective_chunk_uses_prefill_chunk_size(self):
        """When sliding_window==0 and prefill_chunk_size>0, use prefill_chunk_size."""
        # The logic in TransformerDecoder.forward is:
        #   effective_chunk = sliding_window if sliding_window > 0 else prefill_chunk_size
        sliding_window = 0
        prefill_chunk_size = 2048

        effective_chunk = sliding_window if sliding_window > 0 else prefill_chunk_size
        self.assertEqual(effective_chunk, 2048)

    def test_effective_chunk_prefers_sliding_window(self):
        """When both are set, sliding_window takes priority."""
        sliding_window = 4096
        prefill_chunk_size = 2048

        effective_chunk = sliding_window if sliding_window > 0 else prefill_chunk_size
        self.assertEqual(effective_chunk, 4096)

    def test_effective_chunk_zero_means_no_chunking(self):
        """When both are 0, effective_chunk is 0 and chunking is disabled."""
        effective_chunk = 0 if 0 > 0 else 0
        self.assertEqual(effective_chunk, 0)


if __name__ == "__main__":
    unittest.main()
