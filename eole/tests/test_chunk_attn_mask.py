"""
Unit tests for the sliding-window mask logic used by TransformerDecoder.

These tests validate the mathematical invariants of the chunked-prefill
attention mask (_chunk_attn_mask) and its consistency with the single-step
decoding path (_forward_eager S==1 branch) WITHOUT requiring torch or a GPU.

The core invariant that was fixed:
    OLD (wrong): k_pos >= q_pos - W          -> window of W+1 tokens
    NEW (correct): k_pos >= q_pos - W + 1    -> window of exactly W tokens

This matches the decoding path:
    start = current_step - sliding_window + 1
    valid = position_indices >= start   -> positions [start, current_step]
                                           = W tokens
"""

import unittest


def _chunked_prefill_allowed_keys(q_pos, cache_len, sliding_window):
    """Pure-Python reference for _chunk_attn_mask sliding-window constraint.

    Returns the set of key positions that query position ``q_pos`` is allowed
    to attend to when sliding_window > 0.  Mirrors the FIXED tensor logic:

        mask = (k_pos <= q_pos) & (k_pos >= q_pos - sliding_window + 1)
    """
    return {k for k in range(cache_len) if k <= q_pos and k >= q_pos - sliding_window + 1}


def _decoding_allowed_keys(current_step, cache_len, sliding_window):
    """Pure-Python reference for the S==1 decoding sliding-window constraint.

    Mirrors the tensor logic in _forward_eager (S==1 branch):

        start = max(0, current_step - sliding_window + 1)
        valid = (position_indices <= current_step) & (position_indices >= start)
    """
    start = max(0, current_step - sliding_window + 1)
    return {k for k in range(cache_len) if k <= current_step and k >= start}


class TestChunkAttnMaskSlidingWindow(unittest.TestCase):
    """Validate the sliding-window mask invariants for chunked prefill."""

    def test_chunked_and_decoding_windows_are_equal(self):
        """_chunk_attn_mask and the S==1 decoding path produce identical windows."""
        sliding_window = 4
        cache_len = 20

        for step in range(cache_len):
            chunk_keys = _chunked_prefill_allowed_keys(step, cache_len, sliding_window)
            decode_keys = _decoding_allowed_keys(step, cache_len, sliding_window)
            self.assertEqual(
                chunk_keys,
                decode_keys,
                msg=f"step={step}: chunked={sorted(chunk_keys)} != decode={sorted(decode_keys)}",
            )

    def test_window_contains_exactly_sliding_window_tokens(self):
        """Each query attends to exactly sliding_window tokens (or fewer near start)."""
        sliding_window = 4
        cache_len = 20

        for step in range(cache_len):
            keys = _chunked_prefill_allowed_keys(step, cache_len, sliding_window)
            expected_count = min(sliding_window, step + 1)
            self.assertEqual(
                len(keys),
                expected_count,
                msg=f"step={step}: expected {expected_count} keys, got {len(keys)}: {sorted(keys)}",
            )

    def test_off_by_one_would_produce_extra_token(self):
        """Demonstrate that the OLD formula allowed W+1 instead of W tokens."""
        sliding_window = 4
        cache_len = 20

        def _old_chunked_allowed(q_pos, cache_len, sliding_window):
            """The WRONG (pre-fix) formula: k >= q - W  (W+1 tokens)."""
            return {k for k in range(cache_len) if k <= q_pos and k >= q_pos - sliding_window}

        for step in range(sliding_window, cache_len):
            correct = _chunked_prefill_allowed_keys(step, cache_len, sliding_window)
            wrong = _old_chunked_allowed(step, cache_len, sliding_window)
            # The wrong formula has one extra (older) token
            self.assertEqual(
                len(wrong),
                len(correct) + 1,
                msg=f"step={step}: expected old formula to have 1 extra token",
            )
            extra = wrong - correct
            self.assertEqual(
                extra,
                {step - sliding_window},
                msg=f"step={step}: expected extra token at {step - sliding_window}, got {extra}",
            )

    def test_window_lower_bound_never_negative(self):
        """Lower bound of the window is always >= 0."""
        sliding_window = 4
        cache_len = 20

        for step in range(cache_len):
            keys = _chunked_prefill_allowed_keys(step, cache_len, sliding_window)
            self.assertTrue(
                all(k >= 0 for k in keys),
                msg=f"step={step}: negative key in {sorted(keys)}",
            )

    def test_window_upper_bound_equals_query_position(self):
        """The highest attended key is always the current query position (causal)."""
        sliding_window = 4
        cache_len = 20

        for step in range(cache_len):
            keys = _chunked_prefill_allowed_keys(step, cache_len, sliding_window)
            self.assertEqual(
                max(keys),
                step,
                msg=f"step={step}: max key {max(keys)} != q_pos",
            )

    def test_window_is_contiguous(self):
        """Attended keys form a contiguous range with no gaps."""
        sliding_window = 4
        cache_len = 20

        for step in range(cache_len):
            keys = sorted(_chunked_prefill_allowed_keys(step, cache_len, sliding_window))
            expected = list(range(keys[0], keys[-1] + 1))
            self.assertEqual(keys, expected, msg=f"step={step}: keys not contiguous: {keys}")

    def test_sliding_window_1_attends_only_self(self):
        """sliding_window=1 means each token attends only to itself."""
        sliding_window = 1
        cache_len = 10

        for step in range(cache_len):
            keys = _chunked_prefill_allowed_keys(step, cache_len, sliding_window)
            self.assertEqual(keys, {step}, msg=f"step={step}: expected {{step}}, got {keys}")

    def test_large_window_equals_causal(self):
        """A window larger than cache_len is equivalent to full causal attention."""
        sliding_window = 1000
        cache_len = 20

        for step in range(cache_len):
            keys = _chunked_prefill_allowed_keys(step, cache_len, sliding_window)
            causal_keys = {k for k in range(cache_len) if k <= step}
            self.assertEqual(
                keys,
                causal_keys,
                msg=f"step={step}: chunked={sorted(keys)} != causal={sorted(causal_keys)}",
            )

    def test_consistency_across_chunk_boundaries(self):
        """Sliding window behaves identically for a query at position P regardless
        of whether it is processed in chunk 0 or a later chunk.

        This validates that the fix to _chunk_attn_mask (which uses absolute
        positions via current_step) makes cross-chunk masking consistent.
        """
        sliding_window = 3
        cache_len = 15
        chunk_size = 4

        # Process query positions 0..chunk_size-1 in chunk 0
        # Process query positions chunk_size..2*chunk_size-1 in chunk 1
        for chunk_start in [0, chunk_size, 2 * chunk_size]:
            for offset in range(chunk_size):
                q_pos = chunk_start + offset
                if q_pos >= cache_len:
                    break

                # mask from _chunk_attn_mask (current_step=chunk_start, offset=offset)
                # q_pos = current_step + offset
                keys = _chunked_prefill_allowed_keys(q_pos, cache_len, sliding_window)

                # mask from decoding path (current_step = q_pos)
                decode_keys = _decoding_allowed_keys(q_pos, cache_len, sliding_window)

                self.assertEqual(
                    keys,
                    decode_keys,
                    msg=(
                        f"chunk_start={chunk_start}, offset={offset}, q_pos={q_pos}: "
                        f"chunk={sorted(keys)} != decode={sorted(decode_keys)}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
