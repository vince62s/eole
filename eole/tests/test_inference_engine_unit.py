"""Unit tests for InferenceEngine base class – Step 7.

Tests focus on the parts of InferenceEngine that can be exercised without
loading a real model, following the same stub-injection approach used in
test_streamer.py.
"""

import os
import sys
import tempfile
import unittest


# ---------------------------------------------------------------------------
# Stub InferenceEngine and inject without a real torch model
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal config stub that satisfies InferenceEngine.__init__."""
    world_size = 1
    output = None
    with_score = False
    src = None

    def __init__(self, output=None, with_score=False, src=None):
        self.output = output
        self.with_score = with_score
        self.src = src


# We can import InferenceEngine directly since it only imports torch lazily
# inside method bodies.
from eole.inference_engine import InferenceEngine


class _ConcreteEngine(InferenceEngine):
    """Concrete subclass that returns canned results."""

    def __init__(self, config, fake_results):
        self.config = config
        self.model_type = None
        self._fake_results = fake_results
        self.transforms = None
        self.vocabs = None
        self.device_id = 0

    def _predict(self, infer_iter, settings=None):
        return self._fake_results

    def _score(self, infer_iter, settings=None):
        return self._fake_results

    def _build_inference_iterator(self, src=None, tgt=None):
        return None  # not used


# ===========================================================================
# _flatten_results
# ===========================================================================


class TestFlattenResults(unittest.TestCase):

    def setUp(self):
        self.engine = _ConcreteEngine(_FakeConfig(), ([], [], []))

    def test_flatten_single_beam(self):
        scores = [[0.9], [0.8]]
        estims = [[0.5], [0.6]]
        preds = [["hello"], ["world"]]
        fs, fe, fp = self.engine._flatten_results(scores, estims, preds)
        self.assertEqual(fp, ["hello", "world"])
        self.assertAlmostEqual(fs[0], 0.9)
        self.assertAlmostEqual(fe[1], 0.6)

    def test_flatten_multi_beam(self):
        scores = [[0.9, 0.7], [0.8, 0.6]]
        estims = [[0.5, 0.4], [0.3, 0.2]]
        preds = [["a", "b"], ["c", "d"]]
        fs, fe, fp = self.engine._flatten_results(scores, estims, preds)
        self.assertEqual(len(fp), 4)
        self.assertIn("a", fp)
        self.assertIn("d", fp)

    def test_flatten_none_estims_uses_default(self):
        scores = [[0.9]]
        preds = [["hi"]]
        fs, fe, fp = self.engine._flatten_results(scores, None, preds)
        self.assertEqual(len(fe), 1)
        # Default estim value is 1.0
        self.assertEqual(fe[0], 1.0)

    def test_flatten_empty(self):
        fs, fe, fp = self.engine._flatten_results([], None, [])
        self.assertEqual(fp, [])
        self.assertEqual(fs, [])


# ===========================================================================
# _write_predictions_to_file
# ===========================================================================


class TestWritePredictions(unittest.TestCase):

    def test_write_preds_no_score(self):
        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cfg = _FakeConfig(output=tmp_path, with_score=False)
            engine = _ConcreteEngine(cfg, ([], [], []))
            engine._write_predictions_to_file([0.9], [0.5], ["hello world"], tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                content = f.read().strip()
            self.assertEqual(content, "hello world")
        finally:
            os.unlink(tmp_path)

    def test_write_preds_with_score(self):
        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cfg = _FakeConfig(output=tmp_path, with_score=True)
            engine = _ConcreteEngine(cfg, ([], [], []))
            engine._write_predictions_to_file([0.9], [0.5], ["hello"], tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                content = f.read().strip()
            # Should contain pred, score, and estim separated by |||
            self.assertIn("hello", content)
            self.assertIn("0.9", content)
        finally:
            os.unlink(tmp_path)

    def test_write_empty_scores_with_score_flag(self):
        """When scores list is empty with with_score=True, estims are written."""
        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cfg = _FakeConfig(output=tmp_path, with_score=True)
            engine = _ConcreteEngine(cfg, ([], [], []))
            engine._write_predictions_to_file([], [0.42], ["prediction"], tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                content = f.read().strip()
            self.assertIn("0.42", content)
        finally:
            os.unlink(tmp_path)


# ===========================================================================
# Raise-on-unimplemented paths
# ===========================================================================


class TestRaisesOnUnimplemented(unittest.TestCase):

    def setUp(self):
        cfg = _FakeConfig()
        self.engine = _ConcreteEngine(cfg, ([], [], []))

    def test_infer_list_stream_raises(self):
        """infer_list_stream on base engine raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            list(InferenceEngine.infer_list_stream(self.engine, "hello"))

    def test_infer_file_parallel_raises(self):
        with self.assertRaises(NotImplementedError):
            InferenceEngine.infer_file_parallel(self.engine)

    def test_infer_list_parallel_raises(self):
        with self.assertRaises(NotImplementedError):
            InferenceEngine.infer_list_parallel(self.engine, ["hello"])

    def test_score_file_parallel_raises(self):
        with self.assertRaises(NotImplementedError):
            InferenceEngine.score_file_parallel(self.engine)

    def test_score_list_parallel_raises(self):
        with self.assertRaises(NotImplementedError):
            InferenceEngine.score_list_parallel(self.engine, ["hello"])

    def test_predict_batch_raises(self):
        with self.assertRaises(NotImplementedError):
            InferenceEngine.predict_batch(self.engine, None)


# ===========================================================================
# infer_list / score_list routing via _ConcreteEngine
# ===========================================================================


class TestInferListRouting(unittest.TestCase):

    def test_infer_list_returns_canned_results(self):
        """infer_list delegates to _predict when world_size==1."""
        fake = ([[0.9]], [[0.5]], [["hello"]])
        engine = _ConcreteEngine(_FakeConfig(), fake)
        scores, estims, preds = engine.infer_list(["test"])
        self.assertEqual(scores, [[0.9]])
        self.assertEqual(preds, [["hello"]])

    def test_score_list_returns_canned_results(self):
        """score_list delegates to _score when world_size==1."""
        fake = {"score": 42.0}
        engine = _ConcreteEngine(_FakeConfig(), fake)
        result = engine.score_list(["test"])
        self.assertEqual(result, fake)

    def test_terminate_does_not_raise(self):
        engine = _ConcreteEngine(_FakeConfig(), ([], [], []))
        engine.terminate()  # base terminate() is a no-op


# ===========================================================================
# InferenceConstants
# ===========================================================================


class TestInferenceConstants(unittest.TestCase):

    def test_default_estim_value(self):
        from eole.inference_engine import InferenceConstants
        self.assertEqual(InferenceConstants.DEFAULT_ESTIM_VALUE, 1.0)

    def test_output_delimiter(self):
        from eole.inference_engine import InferenceConstants
        self.assertEqual(InferenceConstants.OUTPUT_DELIMITER, "\t")


if __name__ == "__main__":
    unittest.main()
