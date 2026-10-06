"""Unit tests for eole/utils/ – Step 3: Statistics, EarlyStopping, LR-decay, optimizers."""

import math
import unittest

import torch
import torch.nn as nn

from eole.utils.statistics import Statistics
from eole.utils.earlystopping import (
    EarlyStopping,
    PPLScorer,
    AccuracyScorer,
    BLEUScorer,
    TERScorer,
    PatienceEnum,
    DEFAULT_SCORERS,
    scorers_from_config,
    SCORER_BUILDER,
)
from eole.utils.optimizers import (
    noam_decay,
    noamwd_decay,
    cosine_decay,
    exponential_decay,
    rsqrt_decay,
    MultipleOptimizer,
    build_torch_optimizer,
)
from eole.utils.report_manager import ReportMgr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stats(loss=10.0, n_tokens=100, n_correct=80, n_sents=10, n_batchs=2):
    s = Statistics(
        loss=loss,
        n_tokens=n_tokens,
        n_correct=n_correct,
        n_sents=n_sents,
        n_batchs=n_batchs,
    )
    return s


def _make_stats_with_bleu(bleu_val):
    s = _make_stats()
    s.computed_metrics = {"BLEU": bleu_val}
    return s


def _make_stats_with_ter(ter_val):
    s = _make_stats()
    s.computed_metrics = {"TER": ter_val}
    return s


# ===========================================================================
# Statistics
# ===========================================================================


class TestStatistics(unittest.TestCase):

    def test_accuracy(self):
        s = _make_stats(n_correct=80, n_tokens=100)
        self.assertAlmostEqual(s.accuracy(), 80.0)

    def test_xent(self):
        s = _make_stats(loss=50.0, n_tokens=100)
        self.assertAlmostEqual(s.xent(), 0.5)

    def test_ppl(self):
        s = _make_stats(loss=0.0, n_tokens=1)
        self.assertAlmostEqual(s.ppl(), 1.0)

    def test_ppl_large_loss_capped(self):
        """ppl is capped at exp(100) to prevent overflow."""
        s = _make_stats(loss=1000.0, n_tokens=1)
        self.assertAlmostEqual(s.ppl(), math.exp(100))

    def test_aux_loss(self):
        s = Statistics(auxloss=5.0, n_sents=5)
        self.assertAlmostEqual(s.aux_loss(), 1.0)

    def test_avg_attention_entropy_zero_samples(self):
        s = Statistics()
        self.assertEqual(s.avg_attention_entropy(), 0.0)

    def test_avg_attention_entropy_nonzero(self):
        s = Statistics(attention_entropy=3.0, n_attention_samples=3)
        self.assertAlmostEqual(s.avg_attention_entropy(), 1.0)

    def test_update_accumulates_loss(self):
        s1 = _make_stats(loss=10.0, n_tokens=100)
        s2 = _make_stats(loss=20.0, n_tokens=50)
        s1.update(s2)
        self.assertAlmostEqual(s1.loss, 30.0)
        self.assertEqual(s1.n_tokens, 150)

    def test_update_accumulates_n_sents(self):
        s1 = _make_stats(n_sents=5)
        s2 = _make_stats(n_sents=7)
        s1.update(s2)
        self.assertEqual(s1.n_sents, 12)

    def test_update_copies_computed_metrics(self):
        s1 = _make_stats()
        s2 = _make_stats()
        s2.computed_metrics = {"BLEU": 42.0}
        s1.update(s2)
        self.assertEqual(s1.computed_metrics["BLEU"], 42.0)

    def test_update_data_stats_accumulates(self):
        s1 = Statistics()
        s1.data_stats = {"corp1": {"count": 10, "index": 5}}
        s2 = Statistics()
        s2.data_stats = {"corp1": {"count": 5, "index": 9}}
        s1.update(s2)
        self.assertEqual(s1.data_stats["corp1"]["count"], 15)
        self.assertEqual(s1.data_stats["corp1"]["index"], 9)

    def test_update_data_stats_new_key(self):
        s1 = Statistics()
        s1.data_stats = {}
        s2 = Statistics()
        s2.data_stats = {"corp2": {"count": 3, "index": 1}}
        s1.update(s2)
        self.assertIn("corp2", s1.data_stats)

    def test_update_n_src_tokens(self):
        s1 = Statistics()
        s1.n_src_tokens = 10
        s2 = Statistics()
        s2.n_src_tokens = 20
        s1.update(s2, update_n_src_tokens=True)
        self.assertEqual(s1.n_src_tokens, 30)

    def test_computed_metric_retrieval(self):
        s = _make_stats_with_bleu(35.0)
        self.assertAlmostEqual(s.computed_metric("BLEU"), 35.0)

    def test_computed_metric_missing_raises(self):
        s = _make_stats()
        with self.assertRaises(AssertionError):
            s.computed_metric("NONEXISTENT")

    def test_elapsed_time_nonnegative(self):
        s = Statistics()
        self.assertGreaterEqual(s.elapsed_time(), 0.0)


# ===========================================================================
# EarlyStopping scorers
# ===========================================================================


class TestPPLScorer(unittest.TestCase):

    def test_init_best_score(self):
        sc = PPLScorer()
        self.assertEqual(sc.best_score, float("inf"))

    def test_is_improving_lower_ppl(self):
        sc = PPLScorer()
        sc.best_score = 10.0
        self.assertTrue(sc.is_improving(_make_stats(loss=5.0, n_tokens=10)))  # ppl < 10

    def test_is_decreasing_higher_ppl(self):
        sc = PPLScorer()
        sc.best_score = 1.5
        # ppl = exp(1.0) ≈ 2.72 > 1.5
        self.assertTrue(sc.is_decreasing(_make_stats(loss=10.0, n_tokens=10)))

    def test_update(self):
        sc = PPLScorer()
        stats = _make_stats(loss=0.0, n_tokens=1)  # ppl = 1.0
        sc.update(stats)
        self.assertAlmostEqual(sc.best_score, 1.0)


class TestAccuracyScorer(unittest.TestCase):

    def test_init_best_score(self):
        sc = AccuracyScorer()
        self.assertEqual(sc.best_score, float("-inf"))

    def test_is_improving_higher_acc(self):
        sc = AccuracyScorer()
        sc.best_score = 75.0
        self.assertTrue(sc.is_improving(_make_stats(n_correct=90, n_tokens=100)))

    def test_is_decreasing_lower_acc(self):
        sc = AccuracyScorer()
        sc.best_score = 90.0
        self.assertTrue(sc.is_decreasing(_make_stats(n_correct=70, n_tokens=100)))

    def test_update(self):
        sc = AccuracyScorer()
        stats = _make_stats(n_correct=80, n_tokens=100)
        sc.update(stats)
        self.assertAlmostEqual(sc.best_score, 80.0)


class TestBLEUScorer(unittest.TestCase):

    def test_is_improving_higher_bleu(self):
        sc = BLEUScorer()
        sc.best_score = 30.0
        self.assertTrue(sc.is_improving(_make_stats_with_bleu(40.0)))

    def test_is_decreasing_lower_bleu(self):
        sc = BLEUScorer()
        sc.best_score = 30.0
        self.assertTrue(sc.is_decreasing(_make_stats_with_bleu(20.0)))

    def test_update(self):
        sc = BLEUScorer()
        sc.update(_make_stats_with_bleu(35.0))
        self.assertAlmostEqual(sc.best_score, 35.0)


class TestTERScorer(unittest.TestCase):

    def test_is_improving_lower_ter(self):
        sc = TERScorer()
        sc.best_score = 50.0
        self.assertTrue(sc.is_improving(_make_stats_with_ter(40.0)))

    def test_is_decreasing_higher_ter(self):
        sc = TERScorer()
        sc.best_score = 40.0
        self.assertTrue(sc.is_decreasing(_make_stats_with_ter(50.0)))

    def test_update(self):
        sc = TERScorer()
        sc.update(_make_stats_with_ter(30.0))
        self.assertAlmostEqual(sc.best_score, 30.0)


# ===========================================================================
# scorers_from_config
# ===========================================================================


class TestScorersFromConfig(unittest.TestCase):

    def _make_config(self, criteria=None):
        class FakeConfig:
            early_stopping_criteria = criteria
        return FakeConfig()

    def test_none_criteria_returns_defaults(self):
        scorers = scorers_from_config(self._make_config(None))
        self.assertEqual(len(scorers), len(DEFAULT_SCORERS))

    def test_specific_criteria_returns_matching(self):
        scorers = scorers_from_config(self._make_config(["BLEU"]))
        self.assertEqual(len(scorers), 1)
        self.assertIsInstance(scorers[0], BLEUScorer)

    def test_unknown_criteria_raises(self):
        with self.assertRaises(AssertionError):
            scorers_from_config(self._make_config(["UNKNOWN_METRIC"]))

    def test_scorer_builder_keys(self):
        for k in ["ppl", "accuracy", "BLEU", "TER"]:
            self.assertIn(k, SCORER_BUILDER)


# ===========================================================================
# EarlyStopping
# ===========================================================================


class TestEarlyStopping(unittest.TestCase):

    def _make_es(self, tolerance=3):
        """Create a fresh EarlyStopping with new scorer instances."""
        return EarlyStopping(tolerance=tolerance, scorers=[PPLScorer(), AccuracyScorer()])

    def test_initial_status_improving(self):
        es = self._make_es(tolerance=2)
        self.assertEqual(es.status, PatienceEnum.IMPROVING)
        self.assertFalse(es.has_stopped())

    def test_improving_resets_tolerance(self):
        es = self._make_es(tolerance=3)
        # Provide improving stats (ppl improves from inf to ~1, acc from -inf to 100)
        stats = _make_stats(loss=0.0, n_tokens=1, n_correct=1)
        es(stats, step=1)
        # After first improving call tolerance stays at full value
        self.assertEqual(es.current_tolerance, 3)
        self.assertEqual(es.status, PatienceEnum.IMPROVING)
        self.assertEqual(es.current_step_best, 1)

    def test_decreasing_reduces_tolerance(self):
        es = self._make_es(tolerance=3)
        # Set a good best score so next stat is worse
        for sc in es.early_stopping_scorers:
            sc.best_score = 0.5 if sc.name == "ppl" else 99.0
        # Make stats that are worse (high ppl, low accuracy)
        bad_stats = _make_stats(loss=1000.0, n_tokens=10, n_correct=1)
        es(bad_stats, step=1)
        self.assertEqual(es.current_tolerance, 2)
        self.assertEqual(es.status, PatienceEnum.DECREASING)

    def test_stopped_after_tolerance_exhausted(self):
        es = self._make_es(tolerance=2)
        # Force the scorer to show a very good best, so anything is worse
        for sc in es.early_stopping_scorers:
            sc.best_score = 0.1 if sc.name == "ppl" else 99.9
        bad_stats = _make_stats(loss=1000.0, n_tokens=10, n_correct=1)
        es(bad_stats, step=1)  # tolerance -> 1
        es(bad_stats, step=2)  # tolerance -> 0, stopped
        self.assertTrue(es.has_stopped())
        self.assertEqual(es.status, PatienceEnum.STOPPED)

    def test_stopped_does_nothing(self):
        es = self._make_es(tolerance=1)
        for sc in es.early_stopping_scorers:
            sc.best_score = 0.1 if sc.name == "ppl" else 99.9
        bad_stats = _make_stats(loss=1000.0, n_tokens=10, n_correct=1)
        es(bad_stats, step=1)  # stopped
        es.status = PatienceEnum.STOPPED
        # Calling again should be a no-op
        es(bad_stats, step=2)
        self.assertTrue(es.has_stopped())

    def test_is_improving_when_improving(self):
        es = self._make_es(tolerance=3)
        self.assertTrue(es.is_improving())

    def test_has_stopped_when_stopped(self):
        es = self._make_es(tolerance=1)
        for sc in es.early_stopping_scorers:
            sc.best_score = 0.1 if sc.name == "ppl" else 99.9
        bad_stats = _make_stats(loss=1000.0, n_tokens=10, n_correct=1)
        es(bad_stats, step=1)
        self.assertTrue(es.has_stopped())


# ===========================================================================
# Learning-rate decay functions
# ===========================================================================


class TestNoamDecay(unittest.TestCase):

    def test_warmup_phase_increases(self):
        """During warmup, rate should increase with step."""
        r1 = noam_decay(step=1, warmup_steps=4000, model_size=512)
        r2 = noam_decay(step=2000, warmup_steps=4000, model_size=512)
        self.assertLess(r1, r2)

    def test_post_warmup_decreases(self):
        """After warmup, rate should decrease with step."""
        r1 = noam_decay(step=5000, warmup_steps=4000, model_size=512)
        r2 = noam_decay(step=100000, warmup_steps=4000, model_size=512)
        self.assertGreater(r1, r2)

    def test_positive_at_warmup_step(self):
        r = noam_decay(step=4000, warmup_steps=4000, model_size=512)
        self.assertGreater(r, 0)


class TestNoamwdDecay(unittest.TestCase):

    def test_returns_float(self):
        r = noamwd_decay(step=1000, warmup_steps=500, model_size=512, rate=0.5, decay_steps=1000)
        self.assertIsInstance(r, float)

    def test_positive(self):
        r = noamwd_decay(step=1000, warmup_steps=500, model_size=512, rate=0.5, decay_steps=1000)
        self.assertGreater(r, 0)


class TestCosineDecay(unittest.TestCase):

    def test_warmup_linear_phase(self):
        """During warmup, returns step/warmup_steps."""
        r = cosine_decay(step=500, warmup_steps=1000, train_steps=10000)
        self.assertAlmostEqual(r, 0.5)

    def test_at_end_near_zero(self):
        """At the last step, cosine decay should be close to 0."""
        r = cosine_decay(step=10000, warmup_steps=1000, train_steps=10000)
        self.assertAlmostEqual(r, 0.0, places=5)

    def test_at_warmup_end_is_one(self):
        r = cosine_decay(step=1000, warmup_steps=1000, train_steps=10000)
        self.assertAlmostEqual(r, 1.0)


class TestExponentialDecay(unittest.TestCase):

    def test_initial_value_is_rate(self):
        """At step=0 with start_step=0 and decay_steps=1 the rate is applied once."""
        r = exponential_decay(step=0, rate=0.5, decay_steps=1, start_step=0)
        self.assertAlmostEqual(r, 0.5)

    def test_value_decreases_over_time(self):
        r1 = exponential_decay(step=0, rate=0.5, decay_steps=1, start_step=0)
        r2 = exponential_decay(step=2, rate=0.5, decay_steps=1, start_step=0)
        self.assertGreater(r1, r2)

    def test_start_step_delays_decay(self):
        """Before start_step the rate should be at its initial value."""
        r_before = exponential_decay(step=0, rate=0.5, decay_steps=100, start_step=1000)
        r_after = exponential_decay(step=2000, rate=0.5, decay_steps=100, start_step=1000)
        self.assertGreater(r_before, r_after)


class TestRsqrtDecay(unittest.TestCase):

    def test_below_warmup_returns_warmup_value(self):
        """For step < warmup_steps the rate equals 1/sqrt(warmup_steps)."""
        r = rsqrt_decay(step=10, warmup_steps=100)
        self.assertAlmostEqual(r, 1.0 / math.sqrt(100))

    def test_above_warmup_decreases(self):
        r1 = rsqrt_decay(step=1000, warmup_steps=100)
        r2 = rsqrt_decay(step=10000, warmup_steps=100)
        self.assertGreater(r1, r2)

    def test_positive(self):
        r = rsqrt_decay(step=500, warmup_steps=100)
        self.assertGreater(r, 0)


# ===========================================================================
# MultipleOptimizer
# ===========================================================================


class TestMultipleOptimizer(unittest.TestCase):

    def _make_optimizers(self):
        p1 = nn.Parameter(torch.randn(4, 4))
        p2 = nn.Parameter(torch.randn(4, 4))
        opt1 = torch.optim.SGD([p1], lr=0.01)
        opt2 = torch.optim.SGD([p2], lr=0.01)
        return [opt1, opt2], [p1, p2]

    def test_param_groups_merged(self):
        opts, _ = self._make_optimizers()
        mo = MultipleOptimizer(opts)
        # Should expose all param groups from both optimizers
        self.assertEqual(len(mo.param_groups), 2)

    def test_zero_grad(self):
        opts, params = self._make_optimizers()
        mo = MultipleOptimizer(opts)
        # Set up fake gradients
        for p in params:
            p.grad = torch.ones_like(p)
        mo.zero_grad()
        for p in params:
            self.assertIsNone(p.grad)

    def test_step_updates_params(self):
        p1 = nn.Parameter(torch.ones(3))
        opt = torch.optim.SGD([p1], lr=1.0)
        mo = MultipleOptimizer([opt])
        loss = p1.sum()
        loss.backward()
        original = p1.data.clone()
        mo.step()
        self.assertFalse(torch.allclose(p1.data, original))

    def test_state_dict_returns_list(self):
        opts, _ = self._make_optimizers()
        mo = MultipleOptimizer(opts)
        sd = mo.state_dict()
        self.assertIsInstance(sd, list)
        self.assertEqual(len(sd), 2)

    def test_load_state_dict(self):
        opts, _ = self._make_optimizers()
        mo = MultipleOptimizer(opts)
        sd = mo.state_dict()
        # Should not raise
        mo.load_state_dict(sd)

    def test_state_property(self):
        opts, _ = self._make_optimizers()
        mo = MultipleOptimizer(opts)
        state = mo.state
        self.assertIsInstance(state, dict)


# ===========================================================================
# build_torch_optimizer
# ===========================================================================


class TestBuildTorchOptimizer(unittest.TestCase):

    def _make_model_and_config(self, optim_name="sgd"):
        model = nn.Linear(8, 4)

        class FakeConfig:
            optim = optim_name
            learning_rate = 0.01
            weight_decay = 0.0
            adam_beta1 = 0.9
            adam_beta2 = 0.999
            adam_eps = 1e-8
            adagrad_accumulator_init = 0.0
            adafactor_beta2 = -0.8
            adafactor_eps = (None, 0.001)
            adafactor_d = 1.0
            use_amp = True  # forces torch.optim path

        return model, FakeConfig()

    def test_sgd(self):
        model, cfg = self._make_model_and_config("sgd")
        opt = build_torch_optimizer(model, cfg)
        self.assertIsInstance(opt, torch.optim.SGD)

    def test_adam(self):
        model, cfg = self._make_model_and_config("adam")
        opt = build_torch_optimizer(model, cfg)
        self.assertIsInstance(opt, torch.optim.Adam)

    def test_adamw(self):
        model, cfg = self._make_model_and_config("adamw")
        opt = build_torch_optimizer(model, cfg)
        self.assertIsInstance(opt, torch.optim.AdamW)

    def test_adagrad(self):
        model, cfg = self._make_model_and_config("adagrad")
        opt = build_torch_optimizer(model, cfg)
        self.assertIsInstance(opt, torch.optim.Adagrad)

    def test_adadelta(self):
        model, cfg = self._make_model_and_config("adadelta")
        opt = build_torch_optimizer(model, cfg)
        self.assertIsInstance(opt, torch.optim.Adadelta)


# ===========================================================================
# ReportMgr
# ===========================================================================


class TestReportMgr(unittest.TestCase):

    def test_start_sets_start_time(self):
        import time
        mgr = ReportMgr(report_every=10)
        before = time.time()
        mgr.start()
        self.assertGreaterEqual(mgr.start_time, before)

    def test_maybe_report_not_triggered_below_interval(self):
        mgr = ReportMgr(report_every=100)
        mgr.start()
        stats = _make_stats()
        # step=5 is below report_every=100, should return same stats unchanged
        result = mgr.report_training(step=5, num_steps=1000, learning_rate=0.01, patience=None,
                                     report_stats=stats)
        # report_training returns the accumulated stats object unchanged when below interval
        self.assertIs(result, stats)

    def test_maybe_report_triggered_at_interval(self):
        mgr = ReportMgr(report_every=10)
        mgr.start()
        stats = _make_stats()
        # step=10 hits the interval, should return a fresh Statistics object
        result = mgr.report_training(step=10, num_steps=1000, learning_rate=0.01, patience=None,
                                     report_stats=stats)
        self.assertIsInstance(result, Statistics)


if __name__ == "__main__":
    unittest.main()
