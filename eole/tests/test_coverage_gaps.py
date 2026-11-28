"""Targeted tests to fill remaining coverage gaps identified by coverage analysis.

Files targeted:
 - modules/rmsnorm.py: extra_repr() (L106, L127)
 - modules/relative_position_bias.py: gen_relative_positions(cache=True) (L36),
       _relative_position_bucket (L74-97), compute_bias (L109-118)
 - modules/transformer_mlp.py: moe_transformer_ff branch (L30), gated_forward with
       dropout (L135/138), _fuse_gate for standard Linear (L68-77)
 - decoders/decoder.py: _init_cache, _extend_cache, _disable_cache, update_dropout (L83-102)
 - decoders/ensemble.py: EnsembleSrcEmb, EnsembleTgtEmb, EnsembleDecoder.forward,
       combine_attns, init_state, map_state, EnsembleGenerator (L36-176)
 - utils/misc.py: get_device_type, get_device, check_path, get_autocast, set_random_seed,
       use_gpu, report_matrix, check_model_config, RandomShuffler.random_state (L33-171)
 - utils/statistics.py: update with data_stats, avg_attention_entropy, computed_metric,
       output method (L75-88, L190-200)
"""

import os
import math
import unittest
import torch
import torch.nn as nn


# ===========================================================================
# rmsnorm.py — extra_repr methods
# ===========================================================================

from eole.modules.rmsnorm import RMSNorm, GemmaRMSNorm


class TestRMSNormRepr(unittest.TestCase):

    def test_rmsnorm_extra_repr(self):
        norm = RMSNorm(hidden_size=16)
        r = repr(norm)
        self.assertIn("16", r)
        self.assertIn("eps", r)

    def test_gemma_rmsnorm_extra_repr(self):
        norm = GemmaRMSNorm(hidden_size=32)
        r = repr(norm)
        self.assertIn("32", r)
        self.assertIn("eps", r)


# ===========================================================================
# relative_position_bias.py — _relative_position_bucket, compute_bias, cache mode
# ===========================================================================

from eole.modules.relative_position_bias import (
    _relative_position_bucket,
    compute_bias,
    gen_relative_positions,
)


class TestRelativePositionBucketAndBias(unittest.TestCase):

    def test_relative_position_bucket_bidirectional(self):
        rp = torch.arange(-5, 6)  # includes positive and negative positions
        buckets = _relative_position_bucket(rp, bidirectional=True, num_buckets=32, max_distance=128)
        self.assertEqual(buckets.shape, rp.shape)
        # All bucket indices must be within [0, num_buckets)
        self.assertTrue((buckets >= 0).all())
        self.assertTrue((buckets < 32).all())

    def test_relative_position_bucket_unidirectional(self):
        """Unidirectional mode: only non-positive relative positions are valid."""
        rp = torch.arange(0, 8)
        buckets = _relative_position_bucket(rp, bidirectional=False, num_buckets=32, max_distance=128)
        self.assertEqual(buckets.shape, rp.shape)
        self.assertTrue((buckets >= 0).all())

    def test_relative_position_bucket_large_distance_clamped(self):
        """Values beyond max_distance should be clamped to the last bucket."""
        # In unidirectional mode (bidirectional=False), positive relative positions
        # become 0 (since we only track backward positions). Use large *negative* input.
        rp = torch.tensor([-1000, -1])
        buckets = _relative_position_bucket(rp, bidirectional=False, num_buckets=32, max_distance=128)
        # Large negative distance → clamped to last bucket (num_buckets - 1 = 31)
        self.assertEqual(buckets[0].item(), 31)

    def test_compute_bias_shape(self):
        bias = compute_bias(
            query_length=8,
            key_length=8,
            is_decoder=False,
            n_positions=128,
            relative_positions_buckets=32,
        )
        self.assertEqual(bias.shape, (8, 8))

    def test_compute_bias_decoder_mode(self):
        """Decoder mode (causal) should also return shape (Q, K)."""
        bias = compute_bias(
            query_length=4,
            key_length=6,
            is_decoder=True,
            n_positions=128,
            relative_positions_buckets=32,
        )
        self.assertEqual(bias.shape, (4, 6))

    def test_gen_relative_positions_cache_mode(self):
        """gen_relative_positions with cache=True returns a 1-row distance matrix."""
        pos = gen_relative_positions(length=6, n_positions=4, cache=True)
        # cache=True: returns (1, length) distance matrix for key-side computation
        self.assertEqual(pos.shape[1], 6)


# ===========================================================================
# transformer_mlp.py — moe_transformer_ff branch, gated_forward with dropout,
#                      _fuse_gate for standard Linear
# ===========================================================================

from eole.modules.transformer_mlp import MLP


def _make_mlp_cfg(activation="relu", hidden=32, ff=128, add_bias=False):
    from eole.config.models import CustomModelConfig
    return CustomModelConfig(
        hidden_size=hidden,
        heads=4,
        transformer_ff=ff,
        mlp_activation_fn=activation,
        add_ffnbias=add_bias,
    )


class _RunCfg:
    parallel_gpu = 1
    dropout = [0.2]
    attention_dropout = [0.0]
    use_ckpting = []


class TestTransformerMLPGaps(unittest.TestCase):

    def test_moe_transformer_ff_branch(self):
        """When moe_transformer_ff is provided, it overrides model_config.transformer_ff."""
        cfg = _make_mlp_cfg()
        mlp = MLP(cfg, moe_transformer_ff=64)
        self.assertEqual(mlp.transformer_ff, 64)

    def test_gated_forward_with_dropout(self):
        """Gated MLP with dropout_p > 0 exercises the dropout branches (L135, L138)."""
        cfg = _make_mlp_cfg(activation="gated-silu")
        mlp = MLP(cfg, running_config=_RunCfg())
        self.assertGreater(mlp.dropout_p, 0.0)
        x = torch.randn(2, 4, 32)
        mlp.eval()  # eval to be deterministic
        y = mlp(x)
        self.assertEqual(y.shape, (2, 4, 32))

    def test_gated_gelu_with_dropout(self):
        """gated-gelu activation with dropout exercises the same branches."""
        cfg = _make_mlp_cfg(activation="gated-gelu")
        mlp = MLP(cfg, running_config=_RunCfg())
        x = torch.randn(2, 4, 32)
        mlp.eval()
        y = mlp(x)
        self.assertEqual(y.shape, (2, 4, 32))

    def test_fuse_gate_standard_linear(self):
        """_fuse_gate converts gated MLP to fused form (L68-77)."""
        cfg = _make_mlp_cfg(activation="gated-silu")
        mlp = MLP(cfg)
        # Before fusing: has separate gate_up_proj and up_proj
        self.assertIsNotNone(mlp.up_proj)
        mlp._fuse_gate()
        # After fusing: up_proj is deleted; gate_up_proj has doubled output dim
        self.assertFalse(hasattr(mlp, "up_proj"))
        x = torch.randn(2, 4, 32)
        y = mlp(x)
        self.assertEqual(y.shape, (2, 4, 32))

    def test_fuse_gate_with_bias(self):
        """_fuse_gate with add_ffnbias=True should also fuse bias tensors."""
        cfg = _make_mlp_cfg(activation="gated-silu", add_bias=True)
        mlp = MLP(cfg)
        mlp._fuse_gate()
        self.assertFalse(hasattr(mlp, "up_proj"))
        x = torch.randn(1, 3, 32)
        y = mlp(x)
        self.assertEqual(y.shape, (1, 3, 32))

    def test_simple_forward_with_dropout(self):
        """simple_forward with dropout_p > 0 exercises the dropout branches (L160, L164)."""
        cfg = _make_mlp_cfg(activation="relu")
        mlp = MLP(cfg, running_config=_RunCfg())
        x = torch.randn(2, 4, 32)
        mlp.eval()
        y = mlp(x)
        self.assertEqual(y.shape, (2, 4, 32))


# ===========================================================================
# decoders/decoder.py — cache API methods and default update_dropout
# ===========================================================================

from eole.decoders.decoder import DecoderBase


class _MinimalDecoder(DecoderBase):
    """Minimal concrete decoder for testing DecoderBase API."""

    def init_state(self, **kwargs):
        pass

    def map_state(self, fn):
        pass

    def forward(self, emb, enc_out=None, step=None, **kwargs):
        return emb, {}


class TestDecoderBaseAPI(unittest.TestCase):

    def setUp(self):
        self.dec = _MinimalDecoder(attentional=True)

    def test_init_cache_noop(self):
        """_init_cache default implementation does nothing and returns None."""
        emb = torch.randn(2, 5, 16)
        result = self.dec._init_cache(emb, pad_mask=None)
        self.assertIsNone(result)

    def test_extend_cache_noop(self):
        result = self.dec._extend_cache()
        self.assertIsNone(result)

    def test_disable_cache_noop(self):
        result = self.dec._disable_cache()
        self.assertIsNone(result)

    def test_update_dropout_noop(self):
        """Default update_dropout does nothing and does not raise."""
        self.dec.update_dropout(0.3, 0.1)

    def test_attentional_flag(self):
        self.assertTrue(self.dec.attentional)

    def test_state_dict_initially_empty(self):
        self.assertIsInstance(self.dec.state, dict)


# ===========================================================================
# decoders/ensemble.py — EnsembleSrcEmb, EnsembleTgtEmb, EnsembleDecoder,
#                        combine_attns, init_state, map_state, EnsembleGenerator
# ===========================================================================

from eole.decoders.ensemble import (
    EnsembleSrcEmb,
    EnsembleTgtEmb,
    EnsembleDecoder,
    EnsembleGenerator,
)


class _FakeSrcEmb(nn.Module):
    """Minimal source embedding stub for ensemble testing."""
    def __init__(self, dim=16):
        super().__init__()
        self.word_padding_idx = 1
        self.emb = nn.Embedding(100, dim)

    def forward(self, src):
        return self.emb(src)


class _FakeTgtEmb(nn.Module):
    """Minimal target embedding stub."""
    def __init__(self, dim=16):
        super().__init__()
        self.word_padding_idx = 1
        self.emb = nn.Embedding(100, dim)

    def forward(self, tgt, step=None):
        return self.emb(tgt)


class _FakeDec(_MinimalDecoder):
    """A decoder with attentional=True that produces deterministic attns."""

    def __init__(self):
        super().__init__(attentional=True)

    def init_state(self, enc_out=None, enc_final_hs=None, **kwargs):
        self.state["enc_out"] = enc_out

    def map_state(self, fn):
        if self.state.get("enc_out") is not None:
            self.state["enc_out"] = fn(self.state["enc_out"])

    def forward(self, emb, enc_out=None, src_len=None, step=None, **kwargs):
        batch = emb.shape[0]
        tgt_len = emb.shape[1]
        src_len_val = enc_out.shape[1] if enc_out is not None else 4
        attns = {"std": torch.softmax(torch.randn(batch, tgt_len, src_len_val), dim=-1)}
        return emb, attns


class TestEnsembleSrcEmb(unittest.TestCase):

    def test_forward_returns_list(self):
        emb1, emb2 = _FakeSrcEmb(), _FakeSrcEmb()
        ens = EnsembleSrcEmb([emb1, emb2])
        src = torch.randint(0, 100, (3, 5))
        result = ens(src)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (3, 5, 16))

    def test_word_padding_idx_inherited(self):
        ens = EnsembleSrcEmb([_FakeSrcEmb()])
        self.assertEqual(ens.word_padding_idx, 1)


class TestEnsembleTgtEmb(unittest.TestCase):

    def test_forward_returns_list(self):
        emb1, emb2 = _FakeTgtEmb(), _FakeTgtEmb()
        ens = EnsembleTgtEmb([emb1, emb2])
        tgt = torch.randint(0, 100, (2, 6))
        result = ens(tgt)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (2, 6, 16))

    def test_word_padding_idx_inherited(self):
        ens = EnsembleTgtEmb([_FakeTgtEmb()])
        self.assertEqual(ens.word_padding_idx, 1)


class TestEnsembleDecoder(unittest.TestCase):

    def _make_ensemble(self, n=2):
        decs = [_FakeDec() for _ in range(n)]
        return EnsembleDecoder(decs)

    def test_forward_returns_output_and_attns(self):
        ens = self._make_ensemble()
        batch, tgt_len, src_len, D = 2, 4, 6, 16
        enc_out = [torch.randn(batch, src_len, D)] * 2
        emb = [torch.randn(batch, tgt_len, D)] * 2
        ens.init_state(enc_out=enc_out)
        out, attns = ens(emb, enc_out=enc_out, src_len=torch.tensor([6, 6]))
        # EnsembleDecoderOutput: indexing individual decoder outputs
        self.assertIsNotNone(out)
        # attns should be combined (mean of model attns)
        self.assertIn("std", attns)

    def test_combine_attns(self):
        ens = self._make_ensemble(n=2)
        attn1 = {"std": torch.softmax(torch.randn(2, 3, 5), dim=-1)}
        attn2 = {"std": torch.softmax(torch.randn(2, 3, 5), dim=-1)}
        combined = ens.combine_attns((attn1, attn2))
        self.assertIn("std", combined)
        self.assertEqual(combined["std"].shape, (2, 3, 5))
        # Mean should be between the two
        expected = (attn1["std"] + attn2["std"]) / 2
        self.assertTrue(torch.allclose(combined["std"], expected, atol=1e-5))

    def test_init_state(self):
        ens = self._make_ensemble()
        enc_out = [torch.randn(2, 5, 16)] * 2
        ens.init_state(enc_out=enc_out, enc_final_hs=None)
        for dec in ens.model_decoders:
            self.assertIsNotNone(dec.state.get("enc_out"))

    def test_map_state(self):
        ens = self._make_ensemble()
        enc_out = [torch.randn(2, 5, 16)] * 2
        ens.init_state(enc_out=enc_out)
        ens.map_state(lambda s: torch.cat([s, s], dim=0))
        for dec in ens.model_decoders:
            self.assertEqual(dec.state["enc_out"].shape[0], 4)


class TestEnsembleGenerator(unittest.TestCase):

    def test_forward_averages_distributions(self):
        class _FakeGen(nn.Module):
            def forward(self, h):
                return torch.log_softmax(h, dim=-1)

        ens = EnsembleGenerator([_FakeGen(), _FakeGen()])
        h1 = torch.randn(2, 4, 10)
        h2 = torch.randn(2, 4, 10)
        out = ens([h1, h2])
        self.assertEqual(out.shape, (2, 4, 10))

    def test_raw_probs_mode(self):
        class _FakeGen(nn.Module):
            def forward(self, h):
                return torch.log_softmax(h, dim=-1)

        ens = EnsembleGenerator([_FakeGen(), _FakeGen()], raw_probs=True)
        out = ens([torch.randn(1, 3, 8), torch.randn(1, 3, 8)])
        self.assertEqual(out.shape, (1, 3, 8))


# ===========================================================================
# utils/misc.py — previously uncovered utilities
# ===========================================================================

from eole.utils.misc import (
    get_device_type,
    get_device,
    get_autocast,
    set_random_seed,
    use_gpu,
    report_matrix,
    check_path,
    RandomShuffler,
)
from contextlib import nullcontext


class TestMiscGetDevice(unittest.TestCase):

    def test_get_device_type_returns_string(self):
        dtype = get_device_type()
        self.assertIn(dtype, ("cpu", "cuda", "mps"))

    def test_get_device_cpu(self):
        d = get_device(device_id=-1)
        self.assertEqual(d, torch.device("cpu"))

    def test_get_device_default(self):
        d = get_device()
        self.assertIsInstance(d, torch.device)

    def test_get_device_id_none(self):
        d = get_device(device_id=None)
        self.assertIsInstance(d, torch.device)


class TestMiscAutocast(unittest.TestCase):

    def test_autocast_disabled_is_nullcontext(self):
        ctx = get_autocast(enabled=False)
        self.assertIsInstance(ctx, nullcontext)

    def test_autocast_cpu_does_not_raise(self):
        ctx = get_autocast(enabled=True, device_type="cpu")
        with ctx:
            x = torch.randn(2, 2)
            _ = x.sum()

    def test_autocast_auto_device_type(self):
        """auto device type should not raise."""
        ctx = get_autocast(enabled=True, device_type="auto")
        with ctx:
            pass


class TestMiscSetRandomSeed(unittest.TestCase):

    def test_seed_zero_skips(self):
        """Seed 0 should be a no-op without raising."""
        set_random_seed(seed=0, is_cuda=False)

    def test_seed_positive_is_reproducible(self):
        set_random_seed(seed=12345, is_cuda=False)
        x1 = torch.randn(5)
        set_random_seed(seed=12345, is_cuda=False)
        x2 = torch.randn(5)
        self.assertTrue(torch.allclose(x1, x2))


class TestMiscUseGpu(unittest.TestCase):

    def test_no_gpu_ranks(self):
        class _Cfg:
            gpu_ranks = []
        self.assertFalse(use_gpu(_Cfg()))

    def test_with_gpu_ranks(self):
        class _Cfg:
            gpu_ranks = [0]
        self.assertTrue(use_gpu(_Cfg()))

    def test_no_attribute(self):
        class _Cfg:
            pass
        self.assertFalse(use_gpu(_Cfg()))


class TestMiscReportMatrix(unittest.TestCase):

    def test_basic_matrix_output(self):
        row_labels = ["a", "b"]
        col_labels = ["x", "y"]
        matrix = [[0.1, 0.2], [0.3, 0.4]]
        output = report_matrix(row_labels, col_labels, matrix)
        self.assertIsInstance(output, str)
        self.assertIn("a", output)
        self.assertIn("x", output)


class TestMiscCheckPath(unittest.TestCase):

    def test_nonexistent_path_creates_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = os.path.join(tmp, "new_subdir", "file.txt")
            check_path(new_dir)  # should create parent dir without raising

    def test_existing_path_exist_ok_warns(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            existing = os.path.join(tmp, "file.txt")
            open(existing, "w").close()
            warnings = []
            check_path(existing, exist_ok=True, log=lambda msg: warnings.append(msg))
            self.assertTrue(any("exists" in w for w in warnings))

    def test_existing_path_raises_ioerror(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            existing = os.path.join(tmp, "file.txt")
            open(existing, "w").close()
            with self.assertRaises(IOError):
                check_path(existing, exist_ok=False)


class TestRandomShufflerRandomState(unittest.TestCase):

    def test_random_state_property_returns_something(self):
        shuffler = RandomShuffler()
        state = shuffler.random_state
        self.assertIsNotNone(state)

    def test_random_state_property_is_tuple(self):
        """random_state should be a tuple (Python's random state format)."""
        shuffler = RandomShuffler()
        state = shuffler.random_state
        self.assertIsInstance(state, tuple)


# ===========================================================================
# utils/statistics.py — update with data_stats, avg_attention_entropy,
#                       computed_metric, output method
# ===========================================================================

from eole.utils.statistics import Statistics


def _make_stats(**kwargs):
    defaults = dict(loss=10.0, n_tokens=100, n_correct=80, n_sents=10, n_batchs=2)
    defaults.update(kwargs)
    return Statistics(**defaults)


class TestStatisticsAdditional(unittest.TestCase):

    def test_update_with_data_stats(self):
        """update() merges data_stats across Statistics objects."""
        s1 = _make_stats()
        s1.data_stats = {"corpus_A": {"count": 5, "index": 0}}
        s2 = _make_stats()
        s2.data_stats = {"corpus_A": {"count": 3, "index": 1}}
        s1.update(s2)
        self.assertEqual(s1.data_stats["corpus_A"]["count"], 8)

    def test_update_adds_new_corpus_key(self):
        s1 = _make_stats()
        s1.data_stats = {"A": {"count": 10, "index": 0}}
        s2 = _make_stats()
        s2.data_stats = {"B": {"count": 7, "index": 1}}
        s1.update(s2)
        self.assertIn("B", s1.data_stats)
        self.assertEqual(s1.data_stats["B"]["count"], 7)

    def test_avg_attention_entropy_with_samples(self):
        s = _make_stats()
        s.attention_entropy = 10.0
        s.n_attention_samples = 5
        self.assertAlmostEqual(s.avg_attention_entropy(), 2.0)

    def test_avg_attention_entropy_no_samples(self):
        s = _make_stats()
        s.attention_entropy = 0.0
        s.n_attention_samples = 0
        self.assertAlmostEqual(s.avg_attention_entropy(), 0.0)

    def test_computed_metric_found(self):
        s = _make_stats()
        s.computed_metrics = {"BLEU": 42.0}
        self.assertAlmostEqual(s.computed_metric("BLEU"), 42.0)

    def test_computed_metric_missing_raises(self):
        s = _make_stats()
        with self.assertRaises(AssertionError):
            s.computed_metric("TER")

    def test_output_does_not_raise(self):
        """output() writes to logger; should not raise with valid inputs."""
        import time
        s = _make_stats()
        s.n_src_tokens = 50
        s.computed_metrics = {}
        # Should complete without error (logger output is captured by test framework)
        s.output(step=100, num_steps=1000, learning_rate=1e-4, start=time.time())


if __name__ == "__main__":
    unittest.main()
