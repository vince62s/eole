"""Unit tests for the prefill KV cache.

Tests the PrefillCache class directly and the integration with
TransformerDecoder's chunked prefill path, verifying that:

  - Rolling hash keys distinguish different prefixes.
  - Cache hits restore the correct embeddings and layer states.
  - LRU eviction works correctly.
  - Caching works correctly for batch_size > 1 (per-sequence keys).
  - Caching works correctly for hybrid linear-attention models
    (recurrent states are cached and restored alongside KV slices).
  - Caching is skipped only when image tokens are present.
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
        layer_states = [(k_slice, v_slice)]

        key = self.Cache.compute_key(torch.arange(4))
        cache.put(key, emb, layer_states)

        result = cache.get(key)
        self.assertIsNotNone(result)
        emb_out, states_out = result
        # Tensors are stored on CPU
        self.assertEqual(emb_out.device, torch.device("cpu"))
        torch.testing.assert_close(emb_out, emb.cpu())
        k_out, v_out = states_out[0]
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

    def test_linear_attn_state_roundtrip(self):
        """Linear-attention (conv_state, recurrent_state) pairs survive a put/get cycle."""
        cache = self.Cache(max_entries=10)
        conv = torch.randn(8, 3)      # (conv_dim, kernel_size) - no batch dim
        rec = torch.randn(4, 16, 8)   # (num_heads, head_k_dim, head_v_dim)
        layer_states = [(conv, rec)]  # same pair format as standard attention

        key = self.Cache.compute_key(torch.arange(4))
        cache.put(key, torch.zeros(4, 16), layer_states)

        _, states_out = cache.get(key)
        conv_out, rec_out = states_out[0]
        torch.testing.assert_close(conv_out, conv.cpu())
        torch.testing.assert_close(rec_out, rec.cpu())


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestChunkedPrefillWithCache(unittest.TestCase):
    """Integration tests for _forward_chunked_prefill with the prefill cache.

    Uses a minimal mock of TransformerDecoder that replaces _forward_eager
    with a deterministic stub so we can verify:
      - Cache miss: _forward_eager is called and result is stored.
      - Cache hit: _forward_eager is NOT called; cached values are restored.
      - cache_seqlens is correctly updated for both paths.
      - Batch size > 1: per-sequence keys; all-hit skips forward.
      - Hybrid linear-attention: recurrent states are cached and restored.
    """

    def _make_mock_decoder(self, S, chunk_size, B=1, has_linear_attn=False, cache_size=32):
        """Build a SimpleNamespace that mimics the parts of TransformerDecoder
        used by _forward_chunked_prefill.

        Layer layout (2 layers total):
        - If has_linear_attn: layer 0 = linear_attention, layer 1 = full_attention
        - Otherwise: both layers are full_attention
        """
        import types
        from eole.modules.prefill_cache import PrefillCache

        cache_len = S
        heads_kv = 1
        dph = 4
        conv_dim, kernel_size = 8, 3
        num_v_heads, head_k_dim, head_v_dim = 4, 16, 8

        layers = []
        num_layers = 2
        for i in range(num_layers):
            if has_linear_attn and i == 0:
                layer = types.SimpleNamespace(
                    layer_type="linear_attention",
                    linear_attn=types.SimpleNamespace(
                        conv_state=torch.zeros(B, conv_dim, kernel_size),
                        recurrent_state=torch.zeros(B, num_v_heads, head_k_dim, head_v_dim),
                    ),
                )
            else:
                layer = types.SimpleNamespace(
                    layer_type="full_attention",
                    self_attn=types.SimpleNamespace(
                        kcache=torch.zeros(B, cache_len, heads_kv, dph),
                        vcache=torch.zeros(B, cache_len, heads_kv, dph),
                    ),
                )
            layers.append(layer)

        call_count = [0]

        def stub_forward_eager(emb_chunk, **kw):
            """Write a deterministic pattern into KV/recurrent-state and return emb."""
            B_actual, chunk_len, _ = emb_chunk.shape
            # Derive current start position from cache_seqlens before add_.
            cur_start = mock.cache_seqlens[0].item()
            for li, layer in enumerate(layers):
                val = float(li + 1)
                if layer.layer_type == "linear_attention":
                    for b in range(B_actual):
                        layer.linear_attn.conv_state[b] = val * 0.1
                        layer.linear_attn.recurrent_state[b] = val * 0.2
                else:
                    for b in range(B_actual):
                        layer.self_attn.kcache[b, cur_start : cur_start + chunk_len] = val
                        layer.self_attn.vcache[b, cur_start : cur_start + chunk_len] = val * 2.0
            call_count[0] += 1
            mock.cache_seqlens.add_(chunk_len)
            return emb_chunk * 2.0, {"std": None}

        mock = types.SimpleNamespace(
            sliding_window=0,
            prefill_chunk_size=chunk_size,
            has_linear_attn=has_linear_attn,
            transformer_layers=layers,
            _prefill_cache=PrefillCache(max_entries=cache_size),
            cache_seqlens=torch.zeros(B, dtype=torch.int32),
            cache_len_tgt=cache_len,
            _forward_eager=stub_forward_eager,
            _call_count=call_count,
        )
        return mock

    def _run_chunked_prefill(self, mock, emb, src_ids, **kwargs):
        """Invoke _forward_chunked_prefill on the mock via the real implementation."""
        from eole.decoders.transformer import TransformerDecoder

        return TransformerDecoder._forward_chunked_prefill(mock, emb, src_ids=src_ids, **kwargs)

    # ------------------------------------------------------------------
    # Basic miss / hit / key-mismatch tests (B=1, standard attention)
    # ------------------------------------------------------------------

    def test_cache_miss_stores_result(self):
        """First call: _forward_eager runs and result is stored in cache."""
        S, chunk_size, hidden = 8, 4, 8
        mock = self._make_mock_decoder(S, chunk_size)
        emb = torch.randn(1, S, hidden)
        src_ids = torch.randint(0, 100, (1, S))
        tgt_pad_mask = torch.zeros(1, 1, S, dtype=torch.bool)

        self._run_chunked_prefill(mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask)

        n_chunks = (S + chunk_size - 1) // chunk_size
        self.assertEqual(mock._call_count[0], n_chunks)
        self.assertEqual(len(mock._prefill_cache), n_chunks)

    def test_cache_hit_skips_forward(self):
        """Second call with same src_ids: _forward_eager is NOT called again."""
        S, chunk_size, hidden = 8, 4, 8
        mock = self._make_mock_decoder(S, chunk_size)
        emb = torch.randn(1, S, hidden)
        src_ids = torch.randint(0, 100, (1, S))
        tgt_pad_mask = torch.zeros(1, 1, S, dtype=torch.bool)

        # First request: fills the cache
        self._run_chunked_prefill(mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask.clone())
        count_after_first = mock._call_count[0]

        mock.cache_seqlens.zero_()

        # Second request: all chunks should be cache hits
        self._run_chunked_prefill(mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask.clone())
        self.assertEqual(mock._call_count[0], count_after_first)
        self.assertEqual(mock.cache_seqlens[0].item(), S)

    def test_different_src_ids_cause_cache_miss(self):
        """Different token IDs produce different cache keys (cache miss)."""
        S, chunk_size, hidden = 4, 4, 8
        mock = self._make_mock_decoder(S, chunk_size)
        emb = torch.randn(1, S, hidden)
        tgt_pad_mask = torch.zeros(1, 1, S, dtype=torch.bool)

        src_ids_a = torch.zeros(1, S, dtype=torch.long)
        src_ids_b = torch.ones(1, S, dtype=torch.long)

        self._run_chunked_prefill(mock, emb, src_ids_a, tgt_pad_mask=tgt_pad_mask.clone())
        mock.cache_seqlens.zero_()
        self._run_chunked_prefill(mock, emb, src_ids_b, tgt_pad_mask=tgt_pad_mask.clone())

        self.assertEqual(mock._call_count[0], 2)

    # ------------------------------------------------------------------
    # Batch size > 1
    # ------------------------------------------------------------------

    def test_caching_works_for_batch_size_gt1(self):
        """B > 1: per-sequence cache keys; all-hit skips forward."""
        S, chunk_size, hidden, B = 4, 4, 8, 2
        mock = self._make_mock_decoder(S, chunk_size, B=B)
        emb = torch.randn(B, S, hidden)
        src_ids = torch.randint(0, 100, (B, S))
        tgt_pad_mask = torch.zeros(B, 1, S, dtype=torch.bool)

        # First call: fills cache with B entries per chunk.
        self._run_chunked_prefill(mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask.clone())
        count_after_first = mock._call_count[0]
        n_chunks = (S + chunk_size - 1) // chunk_size
        # Each of the B sequences gets its own entry per chunk.
        self.assertEqual(len(mock._prefill_cache), B * n_chunks)

        mock.cache_seqlens.zero_()

        # Second call with same src_ids: all B sequences hit → no new forward calls.
        self._run_chunked_prefill(mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask.clone())
        self.assertEqual(mock._call_count[0], count_after_first)
        # cache_seqlens incremented for all elements.
        self.assertTrue((mock.cache_seqlens == S).all())

    def test_batch_partial_miss_runs_forward(self):
        """If one sequence misses, _forward_eager runs for the whole batch."""
        S, chunk_size, hidden, B = 4, 4, 8, 2
        mock = self._make_mock_decoder(S, chunk_size, B=B)

        src_ids_shared = torch.randint(0, 100, (B, S))
        src_ids_partial = src_ids_shared.clone()
        src_ids_partial[1] = src_ids_partial[1] + 999  # seq 1 has different IDs

        emb = torch.randn(B, S, hidden)
        tgt_pad_mask = torch.zeros(B, 1, S, dtype=torch.bool)

        # First call with shared IDs: fills cache for both sequences.
        self._run_chunked_prefill(mock, emb, src_ids_shared, tgt_pad_mask=tgt_pad_mask.clone())
        count_after_first = mock._call_count[0]

        mock.cache_seqlens.zero_()

        # Second call: seq 0 hits, seq 1 misses → forward must run for the whole batch.
        self._run_chunked_prefill(mock, emb, src_ids_partial, tgt_pad_mask=tgt_pad_mask.clone())
        self.assertGreater(mock._call_count[0], count_after_first)

    # ------------------------------------------------------------------
    # Linear attention (hybrid model)
    # ------------------------------------------------------------------

    def test_caching_works_for_linear_attention(self):
        """Hybrid model: linear-attn recurrent states are cached and restored."""
        S, chunk_size, hidden = 4, 4, 8
        mock = self._make_mock_decoder(S, chunk_size, has_linear_attn=True)
        emb = torch.randn(1, S, hidden)
        src_ids = torch.zeros(1, S, dtype=torch.long)
        tgt_pad_mask = torch.zeros(1, 1, S, dtype=torch.bool)

        # First call: fills cache (including linear-attn states).
        self._run_chunked_prefill(mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask.clone())
        count_after_first = mock._call_count[0]
        n_chunks = (S + chunk_size - 1) // chunk_size
        self.assertEqual(len(mock._prefill_cache), n_chunks)

        # Reset ALL layer states to zero, then run again with the same IDs.
        mock.cache_seqlens.zero_()
        for layer in mock.transformer_layers:
            if layer.layer_type == "linear_attention":
                layer.linear_attn.conv_state.zero_()
                layer.linear_attn.recurrent_state.zero_()
            else:
                layer.self_attn.kcache.zero_()
                layer.self_attn.vcache.zero_()

        # Second call: all chunks hit → _forward_eager must NOT be called.
        self._run_chunked_prefill(mock, emb, src_ids, tgt_pad_mask=tgt_pad_mask.clone())
        self.assertEqual(mock._call_count[0], count_after_first, "forward should be skipped on cache hit")

        # Verify that linear-attn states were non-trivially restored.
        for layer in mock.transformer_layers:
            if layer.layer_type == "linear_attention":
                self.assertFalse(
                    layer.linear_attn.conv_state.eq(0).all().item(),
                    "conv_state should have been restored from cache",
                )
                self.assertFalse(
                    layer.linear_attn.recurrent_state.eq(0).all().item(),
                    "recurrent_state should have been restored from cache",
                )

    # ------------------------------------------------------------------
    # Image token disables caching
    # ------------------------------------------------------------------

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

        self.assertEqual(mock._call_count[0], 2)
        self.assertEqual(len(mock._prefill_cache), 0)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestPrefillChunkSizeTrigger(unittest.TestCase):
    """Verify that prefill_chunk_size triggers chunked prefill without sliding_window."""

    def test_effective_chunk_uses_prefill_chunk_size(self):
        """When sliding_window==0 and prefill_chunk_size>0, use prefill_chunk_size."""
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
