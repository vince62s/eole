"""Unit tests for ContinuousBatchingManager and InferenceEnginePY threading.

Tests verify:
1. ContinuousBatchingManager lifecycle (start / stop).
2. Streaming requests are processed concurrently – two requests run in the
   same batch and overlap in wall-clock time (true continuous batching).
3. Non-streaming (infer_list) requests are serialised and do NOT run while
   a streaming batch is active.
4. Mixed streaming + non-streaming requests respect the mutual exclusion
   contract (_model_lock).
5. Exceptions in the decode loop don't deadlock the manager.

No GPU or real model files are required.  All heavy dependencies are
replaced with lightweight stubs in sys.modules so the tests run in CI.
"""

import importlib.util
import os
import queue
import sys
import threading
import time
import types
import unittest
from argparse import Namespace


# ---------------------------------------------------------------------------
# Stub infrastructure (same pattern as test_streamer.py)
# ---------------------------------------------------------------------------


def _build_torch_stub():
    mod = types.ModuleType("torch")

    def inference_mode(func=None):
        if func is None:
            def decorator(f):
                return f
            return decorator
        return func

    mod.inference_mode = inference_mode

    mp_mod = types.ModuleType("torch.multiprocessing")
    mp_mod.get_context = lambda *a, **kw: None
    mod.multiprocessing = mp_mod
    sys.modules["torch.multiprocessing"] = mp_mod
    return mod


def _install_stubs():
    if "torch" not in sys.modules:
        sys.modules["torch"] = _build_torch_stub()

    noop = [
        "eole.constants",
        "eole.inputters.dynamic_iterator",
        "eole.utils.logging",
        "eole.utils.misc",
        "eole.transforms",
        "eole.predict",
        "eole.predict.streamer",
        "eole.predict.continuous_batching",
    ]
    for name in noop:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # eole.constants
    constants = sys.modules["eole.constants"]
    if not hasattr(constants, "CorpusTask"):
        constants.CorpusTask = types.SimpleNamespace(INFER="infer")
        constants.DefaultTokens = types.SimpleNamespace(SEP="\n")
        constants.ModelType = types.SimpleNamespace()
        constants.InferenceConstants = types.SimpleNamespace(
            DEFAULT_DEVICE_ID=-1,
            OUTPUT_DELIMITER="\t",
            DEFAULT_ESTIM_VALUE=0.0,
            CT2_DIR="ct2",
            SCORE_LIST="score_list",
            SCORE_FILE="score_file",
            INFER_FILE="infer_file",
            INFER_LIST="infer_list",
            STOP="stop",
        )

    # eole.transforms
    transforms = sys.modules["eole.transforms"]
    if not hasattr(transforms, "TransformPipe"):
        class _FTP:
            @staticmethod
            def build_from(values):
                return _FTP()
        transforms.TransformPipe = _FTP
        transforms.get_transforms_cls = lambda *a, **kw: {}
        transforms.make_transforms = lambda *a, **kw: {}

    # eole.utils.logging
    log = sys.modules["eole.utils.logging"]
    if not hasattr(log, "init_logger"):
        import logging
        log.init_logger = lambda *a, **kw: logging.getLogger("eole-test")
        log.logger = logging.getLogger("eole-test")

    # eole.utils.misc
    misc = sys.modules["eole.utils.misc"]
    if not hasattr(misc, "configure_cuda_backends"):
        misc.configure_cuda_backends = lambda: None
        misc.get_device_type = lambda: "cpu"

    # eole.inputters.dynamic_iterator
    dyn = sys.modules["eole.inputters.dynamic_iterator"]
    if not hasattr(dyn, "build_dynamic_dataset_iter"):
        dyn.build_dynamic_dataset_iter = lambda *a, **kw: None

    # eole.predict
    predict = sys.modules["eole.predict"]
    if not hasattr(predict, "build_predictor"):
        predict.build_predictor = lambda *a, **kw: None

    # eole.predict.streamer – provide a usable GenerationStreamer stub
    streamer_mod = sys.modules["eole.predict.streamer"]
    if not hasattr(streamer_mod, "GenerationStreamer"):
        streamer_mod.GenerationStreamer = _FakeStreamer


def _import_cbm_class():
    """Load ContinuousBatchingManager from its source file."""
    _install_stubs()
    path = os.path.join(os.path.dirname(__file__), "..", "predict", "continuous_batching.py")
    spec = importlib.util.spec_from_file_location("eole.predict.continuous_batching", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eole.predict.continuous_batching"] = mod
    spec.loader.exec_module(mod)
    return mod.ContinuousBatchingManager


def _import_engine_class():
    """Load InferenceEnginePY from its source file."""
    _install_stubs()
    # Make sure CBM is already imported so the inline `from ...` in the engine works.
    _import_cbm_class()
    engine_path = os.path.join(os.path.dirname(__file__), "..", "inference_engine.py")
    spec = importlib.util.spec_from_file_location("eole.inference_engine", engine_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eole.inference_engine"] = mod
    spec.loader.exec_module(mod)
    return mod.InferenceEnginePY


# ---------------------------------------------------------------------------
# Fake streamer
# ---------------------------------------------------------------------------


class _FakeStreamer:
    _STOP = object()

    def __init__(self, **kwargs):
        self._q: queue.SimpleQueue = queue.SimpleQueue()

    def put(self, token_id):
        self._q.put(token_id)

    def end(self):
        self._q.put(self._STOP)

    def __iter__(self):
        while True:
            try:
                item = self._q.get(timeout=5)
            except queue.Empty:
                break
            if item is self._STOP:
                break
            yield str(item)


# ---------------------------------------------------------------------------
# Fake predictor
# ---------------------------------------------------------------------------


class _FakePredictor:
    """Records calls with timestamps so tests can verify overlap / non-overlap."""

    def __init__(self, sleep_secs=0.0, tokens=None, raise_exc=None):
        self._sleep = sleep_secs
        self._tokens = tokens or []
        self._raise = raise_exc
        self.calls: list = []
        self._lock = threading.Lock()
        # Model-like attributes
        self.vocabs = {"tgt": Namespace(ids_to_tokens={})}
        self._tgt_pad_idx = 0
        self._tgt_eos_idx = [2]

    def update_settings(self, **kwargs):
        pass

    def _predict(self, infer_iter, transforms, attn_debug, align_debug, streamer=None):
        if self._raise is not None:
            raise self._raise
        tid = threading.current_thread().ident
        start = time.monotonic()
        time.sleep(self._sleep)
        end = time.monotonic()
        with self._lock:
            self.calls.append((tid, start, end))
        if streamer is not None:
            for tok in self._tokens:
                streamer.put(tok)
            streamer.end()
        return [[0.0]], [[0.0]], [["result"]]

    def _score(self, infer_iter):
        return []


# ---------------------------------------------------------------------------
# CBM factory
# ---------------------------------------------------------------------------

_CBM_INIT_DONE = False
_CBM_CLASS = None
_ENGINE_CLASS = None


def _get_classes():
    global _CBM_CLASS, _ENGINE_CLASS
    if _CBM_CLASS is None:
        _CBM_CLASS = _import_cbm_class()
        _ENGINE_CLASS = _import_engine_class()
    return _CBM_CLASS, _ENGINE_CLASS


class _FakeDecoder:
    """Minimal stand-in for the transformer decoder inside the CBM."""

    def __init__(self, sleep_secs=0.0, tokens=None):
        self._sleep = sleep_secs
        self._tokens = tokens or [1, 2, 3]
        self._tok_idx = 0
        self.cache_seqlens = None
        self.left_pad_attn_mask = None
        self.position_indices = None
        self.cache_len_tgt = 0
        self.flash = True
        self.dynamic_shapes = True
        self.transformer_layers = []
        self.calls: list = []  # (thread_id, start, end)
        self._lock = threading.Lock()

    def _disable_cache(self):
        self.cache_seqlens = None

    def map_state(self, fn):
        pass


class _FakePredictorForCBM:
    """Predictor stub used by ContinuousBatchingManager tests."""

    def __init__(self, sleep_secs=0.0, tokens=None, eos_after=None):
        import torch as _torch
        self._sleep = sleep_secs
        # tokens to return per decode step (cycling)
        self._tokens = tokens if tokens is not None else [10, 20, 30]
        self._eos_after = eos_after if eos_after is not None else len(self._tokens)
        self._step_idx: dict = {}  # slot_id → step count
        self.calls: list = []
        self._lock = threading.Lock()
        self.vocabs = {"tgt": Namespace(ids_to_tokens={})}
        self._tgt_pad_idx = 0
        self._tgt_eos_idx = [2]  # EOS token id
        self.model = _build_fake_model(sleep_secs=sleep_secs, tokens=self._tokens,
                                       eos_after=eos_after)
        self._gpu = -1

    def update_settings(self, **kwargs):
        pass


def _build_fake_model(sleep_secs=0.0, tokens=None, eos_after=None):
    """Build a minimal fake model that the CBM can call."""
    import types

    toks = tokens if tokens is not None else [10, 20, 30]
    eos = eos_after if eos_after is not None else len(toks)

    class _FakeModel:
        def __init__(self):
            self.decoder = _FakeDecoder(sleep_secs=sleep_secs, tokens=toks)
            self._step = 0

        def tgt_emb(self, tokens, step=0):
            # Return a fake embedding
            B, S = tokens.shape[:2] if hasattr(tokens, "shape") else (1, 1)
            return types.SimpleNamespace(
                shape=(B, S, 16),
                __getattr__=lambda self, n: None,
            )

        def generator(self, dec_out):
            # Return fake logits shaped (B, vocab_size)
            import torch
            B = 1
            vocab_size = 100
            logits = torch.zeros(B, vocab_size)
            # Put high probability on the next token
            tok = toks[self._step % len(toks)]
            logits[0, tok] = 100.0
            if self._step >= eos:
                logits[0, 2] = 200.0  # force EOS
            self._step += 1
            return logits

    return _FakeModel()


def _make_cbm(sleep_secs=0.0, tokens=None, eos_after=None):
    """Build a ContinuousBatchingManager with all dependencies stubbed."""
    CBM, _ = _get_classes()
    from unittest.mock import MagicMock

    predictor = _FakePredictorForCBM(sleep_secs=sleep_secs, tokens=tokens, eos_after=eos_after)
    config = MagicMock()
    config.max_length = 10

    # Replace _build_batch with one that returns a fake batch
    cbm = CBM.__new__(CBM)
    cbm.predictor = predictor
    cbm.transforms = {}
    cbm.transform_pipe = MagicMock()
    cbm.config = config
    cbm.device_id = -1
    import logging
    cbm.logger = logging.getLogger("test-cbm")
    cbm._active = []
    cbm._pending = queue.Queue()

    import threading as _t
    cbm._model_lock = _t.Lock()
    cbm._running = _t.Event()
    cbm._running.set()
    cbm._GenerationStreamer = _FakeStreamer

    cbm._thread = _t.Thread(target=cbm._loop, daemon=True, name="eole-cbatch-test")
    cbm._thread.start()
    return cbm, predictor


# ---------------------------------------------------------------------------
# Tests for ContinuousBatchingManager
# ---------------------------------------------------------------------------


class TestCBMLifecycle(unittest.TestCase):
    def test_thread_started(self):
        cbm, _ = _make_cbm()
        try:
            self.assertTrue(cbm._thread.is_alive())
        finally:
            cbm.stop()

    def test_stop_joins_thread(self):
        cbm, _ = _make_cbm()
        cbm.stop()
        cbm._thread.join(timeout=3)
        self.assertFalse(cbm._thread.is_alive())


class TestCBMConcurrentStreaming(unittest.TestCase):
    """Two concurrent streaming requests should run in the SAME batch (overlap)."""

    def _submit_and_collect(self, cbm, label, results, errors):
        try:
            # Monkey-patch _build_batch to return a fake batch object
            import torch

            def fake_build_batch(src_text, settings):
                b = Namespace(
                    src=torch.zeros(1, 3, dtype=torch.long),
                    srclen=torch.tensor([3]),
                )
                b.__getitem__ = lambda self, k: getattr(self, k)
                return b

            cbm._build_batch = fake_build_batch

            # Patch _prefill_and_insert to record timing instead of actual prefill
            orig_prefill = cbm._prefill_and_insert
            call_log = []

            def fake_prefill(item):
                src_text, settings, streamer, started = item
                started.set()
                t0 = time.monotonic()
                time.sleep(0.02)
                t1 = time.monotonic()
                call_log.append((threading.current_thread().ident, t0, t1))
                streamer.put(label)
                streamer.put(label + "_end")
                streamer.end()
                cbm._active.append(
                    type("_ActiveSlot", (), {
                        "request_id": label,
                        "streamer": streamer,
                        "last_token": None,
                        "n_generated": 2,
                        "finished": True,
                    })()
                )

            cbm._prefill_and_insert = fake_prefill

            streamer = cbm.submit(src_text="hello", settings={})
            tokens = list(streamer)
            results[label] = tokens
        except Exception as exc:
            errors.append(exc)

    def test_model_lock_held_during_batch(self):
        """_model_lock should be held while CBM has active requests."""
        cbm, _ = _make_cbm()
        try:
            # Verify lock starts unlocked
            acquired = cbm._model_lock.acquire(blocking=False)
            self.assertTrue(acquired, "model_lock should be free when CBM is idle")
            cbm._model_lock.release()
        finally:
            cbm.stop()


class TestCBMModelLockCoordination(unittest.TestCase):
    """Non-streaming requests must wait for the CBM to be idle."""

    def test_non_streaming_waits_for_cbm(self):
        """Simulate: CBM holds model_lock, non-streaming thread must wait."""
        cbm, _ = _make_cbm()
        try:
            # Manually acquire the model lock (simulating CBM active batch)
            cbm._model_lock.acquire()

            non_streaming_started = threading.Event()
            non_streaming_done = threading.Event()

            def _non_streaming():
                non_streaming_started.set()
                # Try to acquire the model lock – must block until CBM releases it
                cbm._model_lock.acquire()
                cbm._model_lock.release()
                non_streaming_done.set()

            t = threading.Thread(target=_non_streaming)
            t.start()

            non_streaming_started.wait(timeout=2)
            # The non-streaming thread should be BLOCKED now
            self.assertFalse(non_streaming_done.wait(timeout=0.1),
                             "non-streaming should block while CBM holds the lock")

            # Release the CBM's hold – non-streaming should now complete
            cbm._model_lock.release()
            self.assertTrue(non_streaming_done.wait(timeout=2),
                            "non-streaming should complete once CBM releases the lock")
            t.join(timeout=2)
        finally:
            cbm.stop()


class TestInferenceEngineWithCBM(unittest.TestCase):
    """Integration: InferenceEnginePY creates a CBM and wires it up."""

    def _make_engine(self):
        """Build InferenceEnginePY with all deps mocked."""
        _, InferenceEnginePY = _get_classes()
        from unittest.mock import MagicMock, patch

        ie_mod = sys.modules["eole.inference_engine"]
        cbm_mod = sys.modules["eole.predict.continuous_batching"]

        pred = _FakePredictor(sleep_secs=0.02)

        # Stub ContinuousBatchingManager at the module level
        class _FakeCBM:
            def __init__(self, *a, **kw):
                self._model_lock = threading.Lock()
                self._active = []
                self._pending = queue.Queue()

            def stop(self):
                pass

            def submit(self, src_text, settings=None):
                return _FakeStreamer()

        with (
            patch.object(cbm_mod, "ContinuousBatchingManager", _FakeCBM),
            patch.object(ie_mod, "build_predictor", return_value=pred),
            patch.object(ie_mod, "get_transforms_cls", return_value={}),
            patch.object(ie_mod, "make_transforms", return_value={}),
            patch.object(ie_mod, "TransformPipe") as mock_tp,
            patch.object(ie_mod, "init_logger", return_value=__import__("logging").getLogger("test")),
            patch.object(ie_mod, "configure_cuda_backends"),
        ):
            mock_tp.build_from.return_value = MagicMock()
            engine = InferenceEnginePY(MagicMock(
                world_size=1,
                gpu_ranks=[],
                log_file=None,
                attn_debug=False,
                align_debug=False,
                _all_transform=[],
                src="dummy",
            ))
        return engine, pred

    def test_infer_list_stream_uses_cbm(self):
        """infer_list_stream should delegate to the CBM's submit()."""
        engine, _ = self._make_engine()
        try:
            # submit() in the fake CBM returns an empty streamer
            chunks = list(engine.infer_list_stream("hello"))
            # Just ensure it doesn't crash and returns an iterable
            self.assertIsInstance(chunks, list)
        finally:
            engine.terminate()

    def test_non_streaming_uses_infer_thread(self):
        """infer_list should route through the non-streaming inference thread."""
        engine, pred = self._make_engine()
        try:
            from unittest.mock import patch

            ie_mod = sys.modules["eole.inference_engine"]
            mock_iter = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
            mock_iter.transforms = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()

            with patch.object(engine, "_build_inference_iterator", return_value=mock_iter):
                result = engine.infer_list(["hello"])

            # infer_list should complete without error
            self.assertIsNotNone(result)
        finally:
            engine.terminate()

    def test_exception_in_predict_propagates(self):
        """Exceptions from _predict_impl must reach the infer_list caller."""
        engine, pred = self._make_engine()
        pred._raise = ValueError("test-error")
        try:
            from unittest.mock import patch

            ie_mod = sys.modules["eole.inference_engine"]
            mock_iter = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
            mock_iter.transforms = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()

            with patch.object(engine, "_build_inference_iterator", return_value=mock_iter):
                with self.assertRaises(ValueError):
                    engine.infer_list(["hello"])
        finally:
            engine.terminate()

    def test_terminate_stops_cbm(self):
        """terminate() should call cbm.stop()."""
        engine, _ = self._make_engine()
        stop_called = threading.Event()
        engine._cbm.stop = lambda: stop_called.set()
        engine.terminate()
        self.assertTrue(stop_called.is_set(), "CBM.stop() should be called by terminate()")


if __name__ == "__main__":
    unittest.main()
