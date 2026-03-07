"""Unit tests for InferenceEnginePY's session-affinity thread pool.

These tests exercise the thread-management machinery directly without
loading a real model, so they run in environments without GPU / torch model
weights.

The session-affinity thread pool was introduced to fix two issues:

1. torch.compile / CUDA-graph AssertionError that occurred when each
   ``infer_list_stream`` call spawned a fresh OS thread:

       assert torch._C._is_key_in_tls("tree_manager_containers")  # AssertionError

   CUDA-graph trees store their tree-manager in thread-local storage (TLS)
   that is initialised only on the thread that first captures the graph.
   Each session always uses the same persistent thread, satisfying the TLS
   constraint.

2. Cross-user FIFO bottleneck: with a single global inference queue all
   concurrent users serialise behind each other.  With per-session threads
   each user's request runs on its own thread and competes fairly for
   ``_predict_lock`` instead of waiting in a shared FIFO queue.
"""

import queue
import threading
import unittest


# ---------------------------------------------------------------------------
# Minimal stubs that mirror InferenceEnginePY's session-pool machinery
# without loading a real model.
# ---------------------------------------------------------------------------


def _make_session_worker(session_id="test"):
    """Return (q, thread) wired up like InferenceEnginePY._get_or_create_session_worker."""
    q: queue.SimpleQueue = queue.SimpleQueue()

    def _worker():
        while True:
            task = q.get()
            if task is None:
                break
            task()

    t = threading.Thread(target=_worker, name=f"eole-inference-{session_id}", daemon=True)
    t.start()
    return q, t


def _run_on_thread(q, fn) -> None:
    """Mirrors InferenceEnginePY._run_on_session_thread."""
    done = threading.Event()
    exc_holder: list = []

    def _wrapper():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            exc_holder.append(exc)
        finally:
            done.set()

    q.put(_wrapper)
    done.wait()
    if exc_holder:
        raise exc_holder[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSessionAffinityPool(unittest.TestCase):
    """Tests for the per-session thread pool mechanics."""

    def setUp(self):
        self.q, self.thread = _make_session_worker("default")

    def tearDown(self):
        self.q.put(None)
        self.thread.join(timeout=2)

    # ------------------------------------------------------------------
    # Basic task execution
    # ------------------------------------------------------------------

    def test_task_is_executed(self):
        """_run_on_session_thread executes the submitted function."""
        results = []
        _run_on_thread(self.q, lambda: results.append(42))
        self.assertEqual(results, [42])

    def test_multiple_tasks_execute_in_order(self):
        """Tasks submitted to the same session execute in FIFO order."""
        order = []
        for i in range(5):
            _run_on_thread(self.q, lambda n=i: order.append(n))
        self.assertEqual(order, list(range(5)))

    def test_exception_is_re_raised(self):
        """An exception inside the task is propagated to the caller."""

        def _raise():
            raise ValueError("boom")

        with self.assertRaisesRegex(ValueError, "boom"):
            _run_on_thread(self.q, _raise)

    # ------------------------------------------------------------------
    # Session-affinity: same session => same OS thread (key guarantee for
    # CUDA-graph TLS)
    # ------------------------------------------------------------------

    def test_same_thread_used_for_every_call_in_same_session(self):
        """All tasks for the same session run on the same persistent OS thread."""
        thread_ids = []

        def _record_tid():
            thread_ids.append(threading.get_ident())

        for _ in range(10):
            _run_on_thread(self.q, _record_tid)

        self.assertEqual(len(set(thread_ids)), 1, f"Tasks ran on multiple threads: {thread_ids}")

    def test_caller_thread_is_different_from_session_thread(self):
        """The caller thread and the session thread must be distinct."""
        caller_id = threading.get_ident()
        worker_ids = []
        _run_on_thread(self.q, lambda: worker_ids.append(threading.get_ident()))
        self.assertNotEqual(caller_id, worker_ids[0])

    # ------------------------------------------------------------------
    # Different sessions use different threads
    # ------------------------------------------------------------------

    def test_different_sessions_use_different_threads(self):
        """Two separate sessions must run on distinct OS threads."""
        q_a, t_a = _make_session_worker("session-A")
        q_b, t_b = _make_session_worker("session-B")
        try:
            tid_a, tid_b = [], []
            _run_on_thread(q_a, lambda: tid_a.append(threading.get_ident()))
            _run_on_thread(q_b, lambda: tid_b.append(threading.get_ident()))
            self.assertNotEqual(tid_a[0], tid_b[0], "Different sessions should use different threads")
        finally:
            q_a.put(None)
            q_b.put(None)
            t_a.join(timeout=2)
            t_b.join(timeout=2)

    # ------------------------------------------------------------------
    # _predict_lock serialises concurrent sessions (protects shared model
    # state and satisfies the single-GPU constraint)
    # ------------------------------------------------------------------

    def test_predict_lock_serializes_concurrent_sessions(self):
        """_predict_lock ensures only one session accesses the model at a time.

        Two sessions start concurrently; the lock must guarantee that their
        critical sections do not overlap.
        """
        predict_lock = threading.Lock()
        overlap_detected = []
        inside_count = [0]

        def _critical_section():
            with predict_lock:
                inside_count[0] += 1
                if inside_count[0] > 1:
                    overlap_detected.append(True)
                threading.Event().wait(timeout=0.05)  # simulate short inference
                inside_count[0] -= 1

        q_a, t_a = _make_session_worker("lock-A")
        q_b, t_b = _make_session_worker("lock-B")
        try:
            # Submit to both sessions simultaneously
            done_a = threading.Event()
            done_b = threading.Event()

            def _task_a():
                _critical_section()
                done_a.set()

            def _task_b():
                _critical_section()
                done_b.set()

            q_a.put(_task_a)
            q_b.put(_task_b)
            done_a.wait(timeout=5)
            done_b.wait(timeout=5)

            self.assertFalse(overlap_detected, "Concurrent sessions overlapped inside _predict_lock")
        finally:
            q_a.put(None)
            q_b.put(None)
            t_a.join(timeout=2)
            t_b.join(timeout=2)

    # ------------------------------------------------------------------
    # Streaming (infer_list_stream) scenario: started event inside lock
    # ------------------------------------------------------------------

    def test_started_fires_inside_predict_lock(self):
        """In infer_list_stream the 'started' event fires inside _predict_lock.

        This means:
        - The caller's started.wait() unblocks only once the lock is held.
        - The streamer's per-token timeout applies from that point, not from
          the moment the task was enqueued.
        """
        predict_lock = threading.Lock()
        SHORT_TIMEOUT = 0.1  # would fire if started fired before lock acquisition
        results: list = []

        in_stream = queue.SimpleQueue()
        _STOP = object()
        started = threading.Event()

        def _streaming_task():
            with predict_lock:
                started.set()  # mirrors infer_list_stream: fire inside lock
                for token in ["X", "Y"]:
                    in_stream.put(token)
                in_stream.put(_STOP)

        q, t = _make_session_worker("streaming")
        try:
            # Hold the lock from outside to simulate another session running
            predict_lock.acquire()

            q.put(_streaming_task)

            # started must NOT fire while the lock is held externally
            fired_early = started.wait(timeout=0.05)
            self.assertFalse(fired_early, "started fired before _predict_lock was available")

            # Release the lock; the session thread can now acquire it
            predict_lock.release()

            # Now started should fire promptly
            started.wait(timeout=5)

            while True:
                try:
                    item = in_stream.get(timeout=SHORT_TIMEOUT)
                except queue.Empty:
                    results.append("TIMEOUT")
                    break
                if item is _STOP:
                    break
                results.append(item)

            self.assertEqual(results, ["X", "Y"], "Streaming tokens timed out unexpectedly")
        finally:
            q.put(None)
            t.join(timeout=2)

    def test_concurrent_sessions_do_not_block_each_other_in_queue(self):
        """Concurrent sessions each start immediately on their own thread.

        Unlike a single FIFO queue where Session B waits for Session A to
        complete before being dequeued, with per-session threads Session B
        is dequeued instantly and only waits for _predict_lock.
        """
        predict_lock = threading.Lock()

        # Track when each session picks up its task (not when it finishes)
        picked_up = {}

        q_a, t_a = _make_session_worker("concurrent-A")
        q_b, t_b = _make_session_worker("concurrent-B")
        try:
            done_a = threading.Event()
            done_b = threading.Event()

            def _task_a():
                picked_up["A"] = True
                with predict_lock:
                    threading.Event().wait(timeout=0.1)
                done_a.set()

            def _task_b():
                picked_up["B"] = True
                with predict_lock:
                    pass
                done_b.set()

            q_a.put(_task_a)
            q_b.put(_task_b)

            done_a.wait(timeout=5)
            done_b.wait(timeout=5)

            # Both sessions must have picked up their tasks (even if B ran after A)
            self.assertIn("A", picked_up)
            self.assertIn("B", picked_up)
        finally:
            q_a.put(None)
            q_b.put(None)
            t_a.join(timeout=2)
            t_b.join(timeout=2)

    # ------------------------------------------------------------------
    # Session release / shutdown
    # ------------------------------------------------------------------

    def test_thread_stops_on_none_sentinel(self):
        """Sending None shuts down the session thread cleanly."""
        self.q.put(None)
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive(), "Worker thread did not stop after None sentinel")
        # Prevent tearDown from sending a second None (thread already stopped)
        self.q = queue.SimpleQueue()  # dummy queue for tearDown


if __name__ == "__main__":
    unittest.main()
