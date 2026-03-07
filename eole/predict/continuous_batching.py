"""Continuous-batching manager for decoder-only LLMs.

This module implements *continuous batching* (also called *in-flight batching*):
new inference requests are inserted into an already-running generation batch
between decode steps, so that the GPU is never idle waiting for a single long
sequence to finish.

Architecture overview
---------------------
::

    ┌──────────────────────────────────────────────────┐
    │               ContinuousBatchingManager           │
    │                                                    │
    │  ┌─────────────┐   ┌─────────────────────────┐   │
    │  │  _pending   │   │    _active slots         │   │
    │  │  Queue      │──▶│  [slot0, slot1, …]       │   │
    │  └─────────────┘   └─────────────────────────┘   │
    │          │                     │                   │
    │          │  insert between     │ one decode step   │
    │          │  decode steps       ▼ at a time         │
    │          └──────▶  transformer decoder loop        │
    └──────────────────────────────────────────────────┘

Each active slot owns a :class:`~eole.predict.streamer.GenerationStreamer`
through which the caller receives decoded tokens as they are generated.

Requirements
------------
- Flash-attention backend (``self_attn_backend='flash'`` in the model config).
- Decoder-only (LM) model.
- ``dynamic_shapes=True`` (or the cache must be large enough for all requests).
"""

import logging
import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Per-request slot
# ---------------------------------------------------------------------------


@dataclass
class _ActiveSlot:
    """Holds all state for one in-flight generation request."""

    request_id: str
    streamer: Any  # GenerationStreamer
    # last token fed to the decoder (shape (1, 1))
    last_token: Any  # torch.Tensor – type as Any to avoid importing torch at module level
    # number of new tokens generated so far (not counting the prompt)
    n_generated: int = 0
    finished: bool = False


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ContinuousBatchingManager:
    """Manages a continuously-batched autoregressive decode loop.

    New requests are placed on :attr:`_pending` and inserted into the active
    batch between decode steps without interrupting ongoing generation.

    Args:
        predictor: A :class:`~eole.predict.GeneratorLM` instance (the
            underlying model).
        transforms: Transform pipeline returned by
            :func:`~eole.transforms.make_transforms`.
        transform_pipe: :class:`~eole.transforms.TransformPipe` used to build
            :class:`~eole.predict.streamer.GenerationStreamer` instances.
        config: The running inference config (``PredictConfig`` or similar).
        device_id (int): GPU device id (``-1`` for CPU).
        logger: Optional logger.
    """

    def __init__(self, predictor, transforms, transform_pipe, config, device_id, logger=None):
        from eole.predict.streamer import GenerationStreamer

        self._GenerationStreamer = GenerationStreamer
        self.predictor = predictor
        self.transforms = transforms
        self.transform_pipe = transform_pipe
        self.config = config
        self.device_id = device_id
        self.logger = logger or logging.getLogger(__name__)

        self._active: List[_ActiveSlot] = []
        self._pending: queue.Queue = queue.Queue()

        # Shared mutual-exclusion lock between the continuous-batching decode
        # loop and the non-streaming inference thread.  The CBM holds this
        # lock for the entire duration that _active is non-empty.  The
        # non-streaming inference thread (in InferenceEnginePY._inference_loop)
        # acquires it before executing each batch task – so non-streaming
        # inference only runs when no streaming requests are active.
        self._model_lock = threading.Lock()

        # Start the decode thread.
        self._running = threading.Event()
        self._running.set()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="eole-cbatch",
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, src_text: str, settings: Optional[Dict[str, Any]] = None):
        """Submit a new request.

        Returns a :class:`~eole.predict.streamer.GenerationStreamer` that
        yields decoded text chunks as tokens are generated.  The first chunk
        may take slightly longer because the prompt must be prefilled before
        any tokens are produced.

        Args:
            src_text (str): Raw source text (will be tokenised internally).
            settings (dict, optional): Per-request inference overrides
                (``temperature``, ``top_k``, ``top_p``, etc.).

        Returns:
            GenerationStreamer: Iterable that yields decoded string chunks.
        """
        streamer = self._GenerationStreamer(
            vocabs=self.predictor.vocabs,
            transform_pipe=self.transform_pipe,
        )
        # Notify caller when the request has been prefilled so that the
        # streamer's per-token timeout starts only after prefill completes
        # (avoids false timeouts while the request is queued).
        started = threading.Event()
        self._pending.put((src_text, settings or {}, streamer, started))
        started.wait()
        return streamer

    def stop(self):
        """Gracefully stop the decode loop and wait for the thread to exit."""
        self._running.clear()
        # Unblock a _pending.get(timeout=…) if the loop is sleeping.
        self._pending.put(None)
        self._thread.join(timeout=10)

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self):
        """Main decode loop – runs in a background daemon thread."""
        _holds_lock = False  # tracks whether this thread currently holds _model_lock

        while self._running.is_set():
            # ── 1. Drain the pending queue ────────────────────────────
            new_items = []
            try:
                # Block briefly if there are no active requests so we don't
                # busy-spin when the server is idle.
                timeout = None if not self._active else 0.0
                item = self._pending.get(timeout=timeout if timeout is None else 0.001)
                if item is not None:
                    new_items.append(item)
                # Drain the rest without blocking.
                while True:
                    try:
                        item = self._pending.get_nowait()
                        if item is not None:
                            new_items.append(item)
                    except queue.Empty:
                        break
            except queue.Empty:
                pass

            # ── 2. Prefill new requests and insert them ───────────────
            if new_items:
                # Acquire the model lock before touching the transformer.
                if not _holds_lock:
                    self._model_lock.acquire()
                    _holds_lock = True
                for item in new_items:
                    self._prefill_and_insert(item)

            # ── 3. Release lock and sleep if no active requests ───────
            if not self._active:
                if _holds_lock:
                    self._model_lock.release()
                    _holds_lock = False
                continue

            # ── 4. Run one decode step ────────────────────────────────
            # (We already hold _model_lock here since _active is non-empty.)
            try:
                self._decode_step()
            except Exception as exc:  # noqa: BLE001
                self.logger.error(f"ContinuousBatchingManager decode step failed: {exc}")
                for slot in self._active:
                    if not slot.finished:
                        slot.streamer.end()
                self._active.clear()
                # Reset decoder cache so the next request starts clean.
                self.predictor.model.decoder._disable_cache()

            # Release the lock once the active list becomes empty so that
            # non-streaming requests can acquire it and run their tasks.
            if not self._active and _holds_lock:
                self._model_lock.release()
                _holds_lock = False

        # Cleanup on shutdown.
        if _holds_lock:
            self._model_lock.release()

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _prefill_and_insert(self, item):
        """Prefill one new request and insert it into the active batch.

        This is the core of continuous batching: while other sequences are
        being decoded, we (temporarily) repurpose the transformer decoder to
        fill the KV cache for the new prompt, then glue its cache entries onto
        the existing live batch.

        The procedure is:

        1. Save references to the existing KV cache tensors.
        2. Call ``_init_cache`` + forward pass for the new prompt – this
           creates new KV tensors in the decoder layers (the old tensors are
           still referenced by the local variables from step 1).
        3. Copy the new KV tensors out.
        4. Restore the old KV tensors in the decoder layers.
        5. Call ``_insert_batch_entries`` to concatenate old and new.
        """
        src_text, settings, streamer, started = item

        try:
            # ------ tokenise -----------------------------------------------
            batch = self._build_batch(src_text, settings)
            if batch is None:
                self.logger.error("ContinuousBatchingManager: failed to build batch for request.")
                streamer.end()
                started.set()
                return

            src = batch["src"]  # (1, src_len)
            src_len = batch["srclen"]  # (1,)
            p_len = int(src_len[0].item())

            device = src.device

            # Apply settings overrides.
            self.predictor.update_settings(**settings)

            # ------ save existing cache (just python object references) ----
            is_first_request = not self._active

            if not is_first_request:
                saved_cache_seqlens = self.predictor.model.decoder.cache_seqlens
                saved_left_pad_attn_mask = self.predictor.model.decoder.left_pad_attn_mask
                saved_position_indices = self.predictor.model.decoder.position_indices
                saved_cache_len_tgt = self.predictor.model.decoder.cache_len_tgt
                saved_flash = self.predictor.model.decoder.flash
                saved_dynamic_shapes = self.predictor.model.decoder.dynamic_shapes
                saved_kcaches = []
                saved_vcaches = []
                saved_cache_leftpads = []
                for layer in self.predictor.model.decoder.transformer_layers:
                    if layer.layer_type == "linear_attention":
                        saved_kcaches.append(None)
                        saved_vcaches.append(None)
                        saved_cache_leftpads.append(None)
                    else:
                        saved_kcaches.append(layer.self_attn.kcache)
                        saved_vcaches.append(layer.self_attn.vcache)
                        saved_cache_leftpads.append(layer.self_attn.cache_leftpad)

            # ------ prefill ------------------------------------------------
            with torch.inference_mode():
                emb = self.predictor.model.tgt_emb(src, step=0)
                pad_mask = src.eq(self.predictor._tgt_pad_idx).unsqueeze(1)

                # _init_cache creates brand-new KV tensors (replaces whatever
                # was in the decoder layers).
                self.predictor.model.decoder._init_cache(emb, pad_mask)

                # Forward pass – fills the newly created KV cache.
                dec_out, _ = self.predictor.model.decoder(
                    emb,
                    enc_out=None,
                    tgt_pad_mask=pad_mask,
                )

                # Sample first token from last-position logits.
                scores = self.predictor.model.generator(dec_out[:, -1, :].unsqueeze(1).squeeze(1))
                from torch.nn.functional import log_softmax

                log_probs = log_softmax(scores, dim=-1)
                first_token = log_probs.argmax(dim=-1, keepdim=True).unsqueeze(1)  # (1, 1, 1) → (1,1)
                first_token = first_token[:, :, 0] if first_token.dim() == 3 else first_token  # (1, 1)

                # Capture new KV tensors before restoring.
                new_cache_seqlens = self.predictor.model.decoder.cache_seqlens.clone()
                new_kcaches = []
                new_vcaches = []
                new_cache_leftpads = []
                for layer in self.predictor.model.decoder.transformer_layers:
                    if layer.layer_type == "linear_attention":
                        new_kcaches.append(None)
                        new_vcaches.append(None)
                        new_cache_leftpads.append(None)
                    else:
                        new_kcaches.append(layer.self_attn.kcache.clone())
                        new_vcaches.append(layer.self_attn.vcache.clone())
                        new_cache_leftpads.append(
                            layer.self_attn.cache_leftpad.clone()
                            if layer.self_attn.cache_leftpad is not None
                            else None
                        )
                new_lp = new_cache_leftpads[0] if new_cache_leftpads else None

            # ------ restore existing cache ---------------------------------
            if not is_first_request:
                self.predictor.model.decoder.cache_seqlens = saved_cache_seqlens
                self.predictor.model.decoder.left_pad_attn_mask = saved_left_pad_attn_mask
                self.predictor.model.decoder.position_indices = saved_position_indices
                self.predictor.model.decoder.cache_len_tgt = saved_cache_len_tgt
                self.predictor.model.decoder.flash = saved_flash
                self.predictor.model.decoder.dynamic_shapes = saved_dynamic_shapes
                for i, layer in enumerate(self.predictor.model.decoder.transformer_layers):
                    if layer.layer_type != "linear_attention":
                        layer.self_attn.kcache = saved_kcaches[i]
                        layer.self_attn.vcache = saved_vcaches[i]
                        layer.self_attn.cache_leftpad = saved_cache_leftpads[i]

                # Insert new KV entries into the live batch.
                self.predictor.model.decoder._insert_batch_entries(
                    new_kcaches=[k for k in new_kcaches if k is not None],
                    new_vcaches=[v for v in new_vcaches if v is not None],
                    new_cache_seqlens=new_cache_seqlens.to(device),
                    new_cache_leftpads=new_lp,
                )
            # For the very first request, the prefill already set up the cache
            # correctly – nothing to restore or merge.

            # ------ stream first token and register slot ------------------
            streamer.put(first_token[0])

            slot = _ActiveSlot(
                request_id=str(uuid.uuid4()),
                streamer=streamer,
                last_token=first_token,
                n_generated=1,
            )
            self._active.append(slot)

        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"Prefill failed for new request: {exc}")
            streamer.end()
        finally:
            # Always signal the caller so it doesn't block forever.
            started.set()

    # ------------------------------------------------------------------
    # Decode step
    # ------------------------------------------------------------------

    def _decode_step(self):
        """Execute one autoregressive decode step for all active sequences.

        Feeds each slot's ``last_token`` through the shared transformer
        decoder (using the live KV cache), samples the next token, pushes it
        to the slot's streamer, and marks finished slots.
        """
        B = len(self._active)
        if B == 0:
            return

        device = self._active[0].last_token.device

        # Build (B, 1) decoder input from each slot's last token.
        decoder_input = torch.cat([slot.last_token for slot in self._active], dim=0)  # (B, 1)

        with torch.inference_mode():
            emb = self.predictor.model.tgt_emb(decoder_input, step=1)
            tgt_pad_mask = decoder_input.eq(self.predictor._tgt_pad_idx).unsqueeze(1)

            dec_out, _ = self.predictor.model.decoder(
                emb,
                enc_out=None,
                tgt_pad_mask=tgt_pad_mask,
            )

            scores = self.predictor.model.generator(dec_out.squeeze(1))
            from torch.nn.functional import log_softmax

            log_probs = log_softmax(scores, dim=-1)
            next_tokens = log_probs.argmax(dim=-1, keepdim=True)  # (B, 1)

        eos_ids = set(self.predictor._tgt_eos_idx)
        max_new = self.config.max_length

        keep = []
        for i, slot in enumerate(self._active):
            tok = next_tokens[i]  # (1,)
            slot.last_token = tok.unsqueeze(0)  # (1, 1)
            slot.n_generated += 1

            slot.streamer.put(tok)

            finished = bool(tok.item() in eos_ids) or slot.n_generated >= max_new
            if finished:
                slot.streamer.end()
                slot.finished = True
            else:
                keep.append(i)

        if len(keep) < B:
            # Remove finished slots and reorder the KV cache accordingly.
            keep_t = torch.tensor(keep, dtype=torch.long, device=device)
            self.predictor.model.decoder.map_state(lambda s: s[keep_t])
            self._active = [self._active[i] for i in keep]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_batch(self, src_text: str, settings: Dict[str, Any]):
        """Tokenise *src_text* and return a single-example batch dict."""
        try:
            from eole.inputters.dynamic_iterator import build_dynamic_dataset_iter
            from eole.constants import CorpusTask

            infer_iter = build_dynamic_dataset_iter(
                self.config,
                self.transforms,
                self.predictor.vocabs,
                task=CorpusTask.INFER,
                src=[src_text],
                device_id=self.device_id,
            )
            for batch, _, __ in infer_iter:
                return batch
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"_build_batch error: {exc}")
        return None
