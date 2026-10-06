"""Unit tests for eole/scorers/ – Step 8: BLEU, TER, chrF, WER, scorer registry."""

import unittest

from eole.scorers import get_scorers_cls, build_scorers, AVAILABLE_SCORERS
from eole.scorers.scorer import Scorer, build_scorers as build_scorers_direct
from eole.scorers.bleu import BleuScorer
from eole.scorers.ter import TerScorer
from eole.scorers.chrF import ChrFScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeConfig:
    pass


# ===========================================================================
# Scorer registry
# ===========================================================================


class TestScorerRegistry(unittest.TestCase):

    def test_bleu_registered(self):
        self.assertIn("BLEU", AVAILABLE_SCORERS)

    def test_ter_registered(self):
        self.assertIn("TER", AVAILABLE_SCORERS)

    def test_chrf_registered(self):
        self.assertIn("CHRF", AVAILABLE_SCORERS)

    def test_get_scorers_cls_bleu(self):
        cls_dict = get_scorers_cls(["BLEU"])
        self.assertIn("BLEU", cls_dict)
        self.assertIs(cls_dict["BLEU"], BleuScorer)

    def test_get_scorers_cls_multiple(self):
        cls_dict = get_scorers_cls(["BLEU", "TER"])
        self.assertIn("BLEU", cls_dict)
        self.assertIn("TER", cls_dict)

    def test_get_scorers_cls_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_scorers_cls(["UNKNOWN_METRIC"])

    def test_build_scorers_returns_dict(self):
        scorers_cls = get_scorers_cls(["BLEU"])
        scorers = build_scorers(_FakeConfig(), scorers_cls)
        self.assertIn("BLEU", scorers)
        self.assertIn("scorer", scorers["BLEU"])
        self.assertIn("value", scorers["BLEU"])
        self.assertEqual(scorers["BLEU"]["value"], 0)

    def test_build_scorers_direct_function(self):
        """build_scorers from scorer.py module works the same way."""
        scorers_cls = get_scorers_cls(["BLEU"])
        scorers = build_scorers_direct(_FakeConfig(), scorers_cls)
        self.assertIsInstance(scorers["BLEU"]["scorer"], BleuScorer)


# ===========================================================================
# Scorer base class
# ===========================================================================


class TestScorerBase(unittest.TestCase):

    def test_cannot_instantiate_directly(self):
        """Scorer base class has no compute_score implementation."""
        scorer = Scorer(_FakeConfig())
        with self.assertRaises(NotImplementedError):
            scorer.compute_score([], [])

    def test_uses_gpu_default_false(self):
        self.assertFalse(Scorer.uses_gpu)


# ===========================================================================
# BleuScorer
# ===========================================================================


class TestBleuScorer(unittest.TestCase):

    def setUp(self):
        self.scorer = BleuScorer(_FakeConfig())

    def test_empty_preds_returns_zero(self):
        score = self.scorer.compute_score([], [])
        self.assertEqual(score, 0)

    def test_identical_gives_100(self):
        preds = ["the cat is on the mat and it is very comfortable"]
        refs = ["the cat is on the mat and it is very comfortable"]
        score = self.scorer.compute_score(preds, refs)
        self.assertAlmostEqual(score, 100.0, delta=0.01)

    def test_different_gives_lower_than_identical(self):
        preds_good = ["the cat is on the mat and it is very comfortable here"]
        preds_bad = ["a dog ran in the park and played with the owner today"]
        refs = ["the cat is on the mat and it is very comfortable here"]
        score_good = self.scorer.compute_score(preds_good, refs)
        score_bad = self.scorer.compute_score(preds_bad, refs)
        self.assertGreater(score_good, score_bad)

    def test_returns_float(self):
        score = self.scorer.compute_score(["hello"], ["hello"])
        self.assertIsInstance(score, float)

    def test_corpus_level(self):
        """Corpus BLEU over multiple sentences returns a plausible score."""
        preds = ["the cat sat on the mat today", "hello beautiful world around us"]
        refs = ["the cat sat on the mat today", "hello beautiful world around us"]
        score = self.scorer.compute_score(preds, refs)
        self.assertAlmostEqual(score, 100.0, delta=0.01)


# ===========================================================================
# TerScorer
# ===========================================================================


class TestTerScorer(unittest.TestCase):

    def setUp(self):
        self.scorer = TerScorer(_FakeConfig())

    def test_empty_preds_returns_zero(self):
        score = self.scorer.compute_score([], [])
        self.assertEqual(score, 0)

    def test_identical_gives_zero(self):
        preds = ["hello world"]
        refs = ["hello world"]
        score = self.scorer.compute_score(preds, refs)
        self.assertAlmostEqual(score, 0.0)

    def test_different_gives_positive(self):
        preds = ["completely different text here wow"]
        refs = ["hello world"]
        score = self.scorer.compute_score(preds, refs)
        self.assertGreater(score, 0)

    def test_returns_float(self):
        score = self.scorer.compute_score(["hello"], ["hello"])
        self.assertIsInstance(score, float)


# ===========================================================================
# ChrFScorer
# ===========================================================================


class TestChrFScorer(unittest.TestCase):

    def setUp(self):
        self.scorer = ChrFScorer(_FakeConfig())

    def test_empty_preds_returns_zero(self):
        score = self.scorer.compute_score([], [])
        self.assertEqual(score, 0)

    def test_identical_gives_100(self):
        preds = ["hello world"]
        refs = ["hello world"]
        score = self.scorer.compute_score(preds, refs)
        self.assertAlmostEqual(score, 100.0)

    def test_different_gives_lower(self):
        preds = ["hello world"]
        refs = ["goodbye world"]
        score = self.scorer.compute_score(preds, refs)
        self.assertLess(score, 100.0)
        self.assertGreater(score, 0)

    def test_returns_float(self):
        score = self.scorer.compute_score(["hello"], ["hello"])
        self.assertIsInstance(score, float)


# ===========================================================================
# WerScorer (if available)
# ===========================================================================


class TestWerScorer(unittest.TestCase):

    def test_wer_registered_if_available(self):
        """WER scorer should be registered if jiwer is installed."""
        try:
            import jiwer  # noqa: F401
            from eole.scorers.wer import WerScorer
            self.assertIn("WER", AVAILABLE_SCORERS)
            scorer = WerScorer(_FakeConfig())
            score = scorer.compute_score(["hello world"], ["hello world"])
            self.assertAlmostEqual(score, 0.0)
        except ImportError:
            self.skipTest("jiwer not installed, skipping WER tests")


if __name__ == "__main__":
    unittest.main()
