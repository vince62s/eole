"""Prefix KV cache for chunked prefill.

Stores per-chunk KV computation results so that identical prompt prefixes
can be served from cache instead of recomputed on subsequent requests.
The cache is keyed by a rolling SHA-256 hash of the token IDs, which
ensures that two different prefixes never share a cache entry.

Usage example::

    cache = PrefillCache(max_entries=512)

    prev_key = None
    for chunk_ids in chunks:
        key = PrefillCache.compute_key(chunk_ids, prev_key)
        cached = cache.get(key)
        if cached is None:
            # compute and store
            emb_out, kv_slices = run_forward(chunk_ids)
            cache.put(key, emb_out, kv_slices)
        else:
            emb_out, kv_slices = cached
            # restore kv_slices to the model cache
        prev_key = key
"""

import hashlib
import threading
from collections import OrderedDict
from typing import Optional

import torch


class PrefillCache:
    """Thread-safe LRU cache for chunked prefill KV data.

    Maps a rolling hash of the token-ID prefix to the hidden-state output
    and per-layer KV tensors produced when that prefix was last processed.
    This avoids recomputing expensive attention for repeated prompt prefixes
    (e.g. the same system prompt across many requests).

    All tensors are stored on CPU to keep GPU memory free for active
    inference; the caller must move them back to the correct device/dtype
    before use.

    Args:
        max_entries (int): Maximum number of chunk-prefix entries to keep.
            Older entries are evicted in LRU order when the cache is full.
    """

    def __init__(self, max_entries: int = 512):
        self._max_entries = max_entries
        self._store: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_key(token_ids: torch.Tensor, prev_key: Optional[bytes] = None) -> bytes:
        """Compute a rolling SHA-256 key for a chunk of token IDs.

        Chaining via ``prev_key`` ensures that two chunks with identical
        token content but different preceding prefixes get different keys.

        Args:
            token_ids: 1-D integer tensor ``(chunk_len,)`` of token IDs.
                The tensor is moved to CPU internally if necessary.
            prev_key: Raw-bytes key returned by the previous chunk call,
                or ``None`` for the very first chunk.

        Returns:
            bytes: 32-byte SHA-256 digest usable as a dict key.
        """
        h = hashlib.sha256()
        if prev_key is not None:
            h.update(prev_key)
        h.update(token_ids.cpu().numpy().tobytes())
        return h.digest()

    # ------------------------------------------------------------------
    # Cache operations
    # ------------------------------------------------------------------

    def get(self, key: bytes):
        """Return ``(emb_out, kv_slices)`` for *key*, or ``None`` if absent.

        ``emb_out`` is a CPU tensor of shape ``(chunk_len, hidden_size)``.
        ``kv_slices`` is a list (one entry per decoder layer) of either
        ``(k_slice, v_slice)`` CPU tensor pairs each of shape
        ``(chunk_len, heads_kv, dim_per_head)``, or ``None`` for
        linear-attention layers that do not use an explicit KV cache.

        The caller is responsible for moving tensors back to the correct
        device and dtype.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                self._store.move_to_end(key)  # mark as recently used
            return entry

    def put(self, key: bytes, emb_out: torch.Tensor, kv_slices: list) -> None:
        """Insert or update a cache entry.

        Args:
            key: Key produced by :meth:`compute_key`.
            emb_out: Hidden-state output for the chunk of shape
                ``(chunk_len, hidden_size)``.  Stored on CPU.
            kv_slices: Per-layer list.  Each element is either a
                ``(k_slice, v_slice)`` pair of tensors with shape
                ``(chunk_len, heads_kv, dim_per_head)``, or ``None``
                for linear-attention layers.  Stored on CPU.
        """
        cpu_emb = emb_out.detach().cpu()
        cpu_kv = []
        for item in kv_slices:
            if item is not None:
                k, v = item
                cpu_kv.append((k.detach().cpu(), v.detach().cpu()))
            else:
                cpu_kv.append(None)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = (cpu_emb, cpu_kv)
                return
            if len(self._store) >= self._max_entries:
                self._store.popitem(last=False)  # evict LRU entry
            self._store[key] = (cpu_emb, cpu_kv)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
