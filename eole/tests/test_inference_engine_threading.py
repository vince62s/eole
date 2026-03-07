"""Unit tests for InferenceEnginePY single-inference-thread serialization.

Verifies that:
1. A dedicated ``eole-inference`` thread is created at startup.
2. terminate() stops it cleanly.
3. Concurrent infer_list / infer_list_stream calls are serialized and
   never overlap on the predictor (no KV-cache corruption).
4. Exceptions inside the inference thread are propagated to the caller.
5. infer_list_stream's ``started`` event prevents the streamer's per-token
   timeout from firing while the request is queued.

No GPU or real model files are required.  ``torch`` and all heavy eole
dependencies are replaced with lightweight stubs so the tests run in CI.
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
# Minimal torch stub (must be in sys.modules before importing inference_engine)
# ---------------------------------------------------------------------------


def _build_torch_stub():
    """Return a minimal fake ``torch`` module."""
    mod = types.ModuleType("torch")

    # torch.inference_mode() must work as a decorator (pass-through).
    class _InferenceMode:
        def __init__(self, func=None):
            self._func = func

        def __call__(self, *args, **kwargs):
            if self._func is not None:
                return self._func(*args, **kwargs)
            # Called with no args: return a decorator
            def _decorator(fn):
                return fn

            return _decorator

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    # Allow @torch.inference_mode() to work both ways.
    def inference_mode(func=None):
        if func is None:
            # @torch.inference_mode() — returns a decorator
            def decorator(f):
                return f

            return decorator
        # @torch.inference_mode (without parens) — direct decoration
        return func

    mod.inference_mode = inference_mode

    # multiprocessing stub (not exercised in single-process tests)
    mp_mod = types.ModuleType("torch.multiprocessing")
    mp_mod.get_context = lambda *a, **kw: None
    mod.multiprocessing = mp_mod
    sys.modules["torch.multiprocessing"] = mp_mod

    return mod


def _install_stubs():
    """Inject lightweight stubs into sys.modules for all torch-dependent imports."""
    if "torch" not in sys.modules:
        sys.modules["torch"] = _build_torch_stub()

    # Stub out every eole sub-module that inference_engine.py tries to import.
    _noop_mods = [
        "eole.constants",
        "eole.inputters.dynamic_iterator",
        "eole.utils.logging",
        "eole.utils.misc",
        "eole.transforms",
        "eole.predict",
        "eole.predict.streamer",
    ]
    for name in _noop_mods:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # eole.constants needs specific names
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

    # eole.transforms needs TransformPipe
    transforms = sys.modules["eole.transforms"]
    if not hasattr(transforms, "TransformPipe"):

        class _FakeTransformPipe:
            @staticmethod
            def build_from(values):
                return _FakeTransformPipe()

        transforms.TransformPipe = _FakeTransformPipe
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

    # eole.predict.streamer — provide a real-ish GenerationStreamer stub so
    # that infer_list_stream's `from eole.predict.streamer import ...` works.
    streamer_mod = sys.modules["eole.predict.streamer"]
    if not hasattr(streamer_mod, "GenerationStreamer"):
        streamer_mod.GenerationStreamer = _FakeStreamer


def _import_engine_class():
    """Load InferenceEnginePY directly from its source file."""
    _install_stubs()
    engine_path = os.path.join(os.path.dirname(__file__), "..", "inference_engine.py")
    spec = importlib.util.spec_from_file_location("eole.inference_engine", engine_path)
    mod = importlib.util.module_from_spec(spec)
    # Make sure the module can resolve its own imports
    sys.modules["eole.inference_engine"] = mod
    spec.loader.exec_module(mod)
    return mod.InferenceEnginePY


# ---------------------------------------------------------------------------
# Fake streamer (no torch required)
# ---------------------------------------------------------------------------


class _FakeStreamer:
    """Minimal stand-in for GenerationStreamer."""

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
    """Records calls with timestamps so tests can verify non-overlap."""

    def __init__(self, sleep_secs=0.0, tokens=None, raise_exc=None):
        self._sleep = sleep_secs
        self._tokens = tokens or []
        self._raise = raise_exc
        self.calls: list = []  # list of (thread_id, start, end)
        self._call_lock = threading.Lock()

    def update_settings(self, **kwargs):
        pass

    def _predict(self, infer_iter, transforms, attn_debug, align_debug, streamer=None):
        if self._raise is not None:
            raise self._raise
        tid = threading.current_thread().ident
        start = time.monotonic()
        time.sleep(self._sleep)
        end = time.monotonic()
        with self._call_lock:
            self.calls.append((tid, start, end))
        if streamer is not None:
            for tok in self._tokens:
                streamer.put(tok)
            streamer.end()
        return [[0.0]], [[0.0]], [["result"]]

    def _score(self, infer_iter):
        return []

    @property
    def vocabs(self):
        return {"tgt": Namespace(ids_to_tokens={})}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _make_engine(sleep_secs=0.0, tokens=None, raise_exc=None, predictor=None):
    """Construct an InferenceEnginePY with all heavy deps replaced by stubs."""
    InferenceEnginePY = _import_engine_class()

    from unittest.mock import MagicMock

    config = MagicMock()
    config.world_size = 1
    config.gpu_ranks = []
    config.log_file = None
    config.attn_debug = False
    config.align_debug = False
    config._all_transform = []
    config.src = "dummy"

    fake_pred = predictor or _FakePredictor(sleep_secs=sleep_secs, tokens=tokens, raise_exc=raise_exc)

    # Patch the module-level callables that _initialize_single_process uses.
    ie_mod = sys.modules["eole.inference_engine"]
    orig_build = ie_mod.build_predictor
    orig_get_cls = ie_mod.get_transforms_cls
    orig_make = ie_mod.make_transforms
    orig_pipe = ie_mod.TransformPipe
    orig_init_logger = ie_mod.init_logger
    orig_configure = ie_mod.configure_cuda_backends

    from unittest.mock import patch

    with (
        patch.object(ie_mod, "build_predictor", return_value=fake_pred),
        patch.object(ie_mod, "get_transforms_cls", return_value={}),
        patch.object(ie_mod, "make_transforms", return_value={}),
        patch.object(ie_mod, "TransformPipe") as mock_tp,
        patch.object(ie_mod, "init_logger", return_value=__import__("logging").getLogger("test")),
        patch.object(ie_mod, "configure_cuda_backends"),
    ):
        mock_tp.build_from.return_value = MagicMock()
        engine = InferenceEnginePY(config)

    # Replace the iterator builder with a lightweight stub.
    def _fake_iter(src=None, tgt=None):
        it = MagicMock()
        it.transforms = MagicMock()
        return it

    from unittest.mock import MagicMock

    engine._build_inference_iterator = _fake_iter
    return engine, fake_pred


# ===========================================================================
# Tests
# ===========================================================================


class TestInferenceThread(unittest.TestCase):
    """Verify the dedicated inference thread lifecycle."""

    def test_thread_created_on_init(self):
        engine, _ = _make_engine()
        try:
            self.assertTrue(engine._infer_thread.is_alive())
            self.assertEqual(engine._infer_thread.name, "eole-inference")
        finally:
            engine.terminate()

    def test_thread_stopped_by_terminate(self):
        engine, _ = _make_engine()
        self.assertTrue(engine._infer_thread.is_alive())
        engine.terminate()
        engine._infer_thread.join(timeout=2)
        self.assertFalse(engine._infer_thread.is_alive())


class TestNonStreamingSerialization(unittest.TestCase):
    """Concurrent infer_list calls must be serialized on a single thread."""

    def test_concurrent_calls_serialized(self):
        """Two threads calling infer_list concurrently must not overlap."""
        engine, pred = _make_engine(sleep_secs=0.05)
        try:
            results, errors = [], []

            def _call():
                try:
                    results.append(engine.infer_list(["hello"]))
                except Exception as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=_call)
            t2 = threading.Thread(target=_call)
            t1.start()
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)

            self.assertEqual(errors, [], errors)
            self.assertEqual(len(results), 2)
            self.assertEqual(len(pred.calls), 2)

            (tid1, s1, e1), (tid2, s2, e2) = pred.calls
            # Both runs must happen on the SAME dedicated thread.
            self.assertEqual(tid1, tid2, "Both infer_list calls must use the same inference thread")
            # The two calls must not overlap in wall-clock time.
            overlap = min(e1, e2) - max(s1, s2)
            self.assertLessEqual(overlap, 0.0, f"Calls overlapped by {overlap:.4f}s")
        finally:
            engine.terminate()

    def test_exception_propagated(self):
        """Exceptions raised inside _predict_impl must reach the caller."""
        engine, _ = _make_engine(raise_exc=ValueError("boom"))
        try:
            with self.assertRaises(ValueError):
                engine.infer_list(["hello"])
        finally:
            engine.terminate()


class TestStreamingSerialization(unittest.TestCase):
    """Concurrent infer_list_stream calls must be serialized."""

    def _stream(self, engine, idx, collected, errors):
        # GenerationStreamer is already stubbed to _FakeStreamer via _install_stubs()
        try:
            for chunk in engine.infer_list_stream("hello"):
                collected[idx].append(chunk)
        except Exception as exc:
            errors.append(exc)

    def test_concurrent_streams_serialized(self):
        """Two concurrent streaming requests must not overlap."""
        engine, pred = _make_engine(sleep_secs=0.05, tokens=[1, 2, 3])
        try:
            collected = [[], []]
            errors = []

            t1 = threading.Thread(target=self._stream, args=(engine, 0, collected, errors))
            t2 = threading.Thread(target=self._stream, args=(engine, 1, collected, errors))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            self.assertEqual(errors, [], errors)
            self.assertEqual(len(pred.calls), 2)
            (tid1, s1, e1), (tid2, s2, e2) = pred.calls
            self.assertEqual(tid1, tid2, "Both streaming calls must use the same inference thread")
            overlap = min(e1, e2) - max(s1, s2)
            self.assertLessEqual(overlap, 0.0, f"Streaming calls overlapped by {overlap:.4f}s")
        finally:
            engine.terminate()

    def test_stream_exception_propagated(self):
        """Exceptions raised during streaming must reach the generator."""
        engine, _ = _make_engine(raise_exc=RuntimeError("stream-boom"))
        try:
            with self.assertRaises(RuntimeError):
                for _ in engine.infer_list_stream("hello"):
                    pass
        finally:
            engine.terminate()

    def test_started_event_before_first_token(self):
        """Iteration must not hang; tokens must arrive once inference starts."""
        engine, _ = _make_engine(tokens=[42, 43])
        try:
            chunks = list(engine.infer_list_stream("hello"))
            self.assertEqual(chunks, ["42", "43"])
        finally:
            engine.terminate()


class TestMixedConcurrency(unittest.TestCase):
    """A streaming and a batch request submitted simultaneously must not overlap."""

    def test_stream_and_batch_do_not_overlap(self):
        engine, pred = _make_engine(sleep_secs=0.05, tokens=[7, 8])
        try:
            batch_results, errors = [], []

            def _batch():
                try:
                    batch_results.append(engine.infer_list(["batch"]))
                except Exception as exc:
                    errors.append(exc)

            def _stream():
                try:
                    # GenerationStreamer stubbed to _FakeStreamer via _install_stubs()
                    chunks = list(engine.infer_list_stream("stream"))
                    batch_results.append(chunks)
                except Exception as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=_batch)
            t2 = threading.Thread(target=_stream)
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            self.assertEqual(errors, [], errors)
            self.assertEqual(len(pred.calls), 2)
            (tid1, s1, e1), (tid2, s2, e2) = pred.calls
            self.assertEqual(tid1, tid2, "Mixed calls must use the same inference thread")
            overlap = min(e1, e2) - max(s1, s2)
            self.assertLessEqual(overlap, 0.0, f"Mixed calls overlapped by {overlap:.4f}s")
        finally:
            engine.terminate()


if __name__ == "__main__":
    unittest.main()
