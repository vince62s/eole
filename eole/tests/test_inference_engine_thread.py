"""Unit tests for InferenceEnginePY's persistent inference thread.

These tests exercise the thread-management machinery directly without
loading a real model, so they run in environments without GPU / torch model
weights.

The persistent inference thread was introduced to fix a torch.compile /
CUDA-graph AssertionError that occurred when each ``infer_list_stream`` call
spawned a fresh OS thread:

    assert torch._C._is_key_in_tls("tree_manager_containers")  # AssertionError

CUDA-graph trees store their tree-manager in thread-local storage (TLS) that
is initialised only on the thread that first captures the graph.  Using a
single long-lived thread ensures TLS is valid for every subsequent call.
"""

import queue
import threading
import unittest


# ---------------------------------------------------------------------------
# Minimal stub so we can instantiate the thread worker without loading a model
# ---------------------------------------------------------------------------


def _make_worker_triple():
    """Return (infer_queue, thread, stop_event) wired up like InferenceEnginePY."""
    infer_queue: queue.SimpleQueue = queue.SimpleQueue()

    def _worker():
        while True:
            task = infer_queue.get()
            if task is None:
                break
            task()

    thread = threading.Thread(target=_worker, name="eole-inference-test", daemon=True)
    thread.start()
    return infer_queue, thread


def _run_on_thread(infer_queue, fn) -> None:
    """Mirrors InferenceEnginePY._run_on_infer_thread."""
    done = threading.Event()
    exc_holder: list = []

    def _wrapper():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            exc_holder.append(exc)
        finally:
            done.set()

    infer_queue.put(_wrapper)
    done.wait()
    if exc_holder:
        raise exc_holder[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPersistentInferenceThread(unittest.TestCase):
    def setUp(self):
        self.infer_queue, self.thread = _make_worker_triple()

    def tearDown(self):
        # Signal shutdown
        self.infer_queue.put(None)
        self.thread.join(timeout=2)

    # ------------------------------------------------------------------
    # Basic task execution
    # ------------------------------------------------------------------

    def test_task_is_executed(self):
        """_run_on_infer_thread executes the submitted function."""
        results = []
        _run_on_thread(self.infer_queue, lambda: results.append(42))
        self.assertEqual(results, [42])

    def test_multiple_tasks_execute_in_order(self):
        """Tasks submitted sequentially execute in FIFO order."""
        order = []
        for i in range(5):
            _run_on_thread(self.infer_queue, lambda n=i: order.append(n))
        self.assertEqual(order, list(range(5)))

    def test_exception_is_re_raised(self):
        """An exception inside the task is propagated to the caller."""

        def _raise():
            raise ValueError("boom")

        with self.assertRaisesRegex(ValueError, "boom"):
            _run_on_thread(self.infer_queue, _raise)

    # ------------------------------------------------------------------
    # Same thread identity across calls (the key guarantee for CUDA-graph TLS)
    # ------------------------------------------------------------------

    def test_same_thread_used_for_every_call(self):
        """All tasks run on the same persistent OS thread."""
        thread_ids = []

        def _record_tid():
            thread_ids.append(threading.get_ident())

        for _ in range(10):
            _run_on_thread(self.infer_queue, _record_tid)

        # All tasks must have run on the same thread.
        self.assertEqual(len(set(thread_ids)), 1, f"Tasks ran on multiple threads: {thread_ids}")

    def test_caller_thread_is_different_from_worker_thread(self):
        """The caller thread and the inference thread must be distinct."""
        caller_id = threading.get_ident()
        worker_ids = []
        _run_on_thread(self.infer_queue, lambda: worker_ids.append(threading.get_ident()))
        self.assertNotEqual(caller_id, worker_ids[0])

    # ------------------------------------------------------------------
    # Streaming (infer_list_stream) scenario: non-blocking enqueue
    # ------------------------------------------------------------------

    def test_stream_task_does_not_block_caller(self):
        """Putting a task directly on the queue (streaming style) does not block.

        infer_list_stream puts the task on the queue and immediately starts
        iterating the streamer.  The task must NOT be waited on before the
        caller starts consuming.
        """
        consumed: list = []
        in_stream = queue.SimpleQueue()  # simulated streamer internal queue
        _STOP = object()

        def _stream_task():
            for token in ["Hello", " world"]:
                in_stream.put(token)
            in_stream.put(_STOP)

        # Non-blocking enqueue (mirrors infer_list_stream behaviour)
        self.infer_queue.put(_stream_task)

        # Consume streamer output (mirrors "yield from streamer")
        while True:
            item = in_stream.get(timeout=5)
            if item is _STOP:
                break
            consumed.append(item)

        self.assertEqual(consumed, ["Hello", " world"])

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def test_thread_stops_on_none_sentinel(self):
        """Sending None shuts down the worker thread cleanly."""
        self.infer_queue.put(None)
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive(), "Worker thread did not stop after None sentinel")
        # Prevent tearDown from sending a second None (thread already stopped)
        self.infer_queue = queue.SimpleQueue()  # dummy queue for tearDown


if __name__ == "__main__":
    unittest.main()
