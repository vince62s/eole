"""Extended unit tests for eole/modules/ – covering files not tested in test_modules.py.

Covers:
 - rmsnorm.py: RMSNorm, GemmaRMSNorm
 - weight_norm.py: WeightNormConv1d
 - stacked_rnn.py: StackedLSTM, StackedGRU
 - contextgate.py: context_gate_factory, ContextGate, SourceContextGate,
                   TargetContextGate, BothContextGate
 - global_attention.py: GlobalAttention (dot / general / mlp, single-step)
 - alibi_position_bias.py: AlibiPositionalBias
 - relative_position_bias.py: relative_matmul, gen_relative_positions
 - sparse_activations.py: Sparsemax / LogSparsemax
 - sparse_losses.py: SparsemaxLoss
 - lora.py: LoRALayer Embedding (no bnb)
 - embeddings.py: PositionalEncoding, Embeddings (sinusoidal)
"""

import math
import unittest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# RMSNorm / GemmaRMSNorm
# ---------------------------------------------------------------------------

from eole.modules.rmsnorm import RMSNorm, GemmaRMSNorm


class TestRMSNorm(unittest.TestCase):

    def test_output_shape(self):
        norm = RMSNorm(hidden_size=32)
        nn.init.ones_(norm.weight)
        x = torch.randn(2, 5, 32)
        with torch.no_grad():
            y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_normalizes_rms(self):
        """After RMSNorm with weight=1, each vector should have rms ≈ 1."""
        norm = RMSNorm(hidden_size=16, eps=1e-8)
        nn.init.ones_(norm.weight)
        x = torch.randn(4, 16) * 10  # large-scale input
        with torch.no_grad():
            y = norm(x)
        rms = y.pow(2).mean(-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones(4), atol=1e-4))

    def test_weight_applied(self):
        """Changing weight should scale output proportionally."""
        norm = RMSNorm(hidden_size=8)
        x = torch.randn(1, 8)
        norm.weight.data.fill_(2.0)
        with torch.no_grad():
            y2 = norm(x)
        norm.weight.data.fill_(1.0)
        with torch.no_grad():
            y1 = norm(x)
        self.assertTrue(torch.allclose(y2, y1 * 2.0, atol=1e-5))

    def test_output_dtype_preserved(self):
        norm = RMSNorm(hidden_size=16)
        nn.init.ones_(norm.weight)
        x = torch.randn(2, 16).half()  # fp16 input
        with torch.no_grad():
            y = norm(x.float())  # forward accepts float32
        self.assertEqual(y.dtype, torch.float32)


class TestGemmaRMSNorm(unittest.TestCase):

    def test_output_shape(self):
        norm = GemmaRMSNorm(hidden_size=16)
        nn.init.zeros_(norm.weight)  # effective weight = 1.0 + 0.0 = 1.0
        x = torch.randn(2, 16)
        with torch.no_grad():
            y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_gemma_offset(self):
        """With weight=0, Gemma uses (1+0)=1 so output equals standard RMSNorm with weight=1."""
        hidden = 16
        norm_gemma = GemmaRMSNorm(hidden_size=hidden)
        norm_std = RMSNorm(hidden_size=hidden)
        nn.init.zeros_(norm_gemma.weight)
        nn.init.ones_(norm_std.weight)
        x = torch.randn(3, hidden)
        with torch.no_grad():
            y_gemma = norm_gemma(x)
            y_std = norm_std(x)
        self.assertTrue(torch.allclose(y_gemma, y_std, atol=1e-5))


# ---------------------------------------------------------------------------
# WeightNormConv1d
# ---------------------------------------------------------------------------

from eole.modules.weight_norm import WeightNormConv1d


class TestWeightNormConv1d(unittest.TestCase):

    def test_forward_shape(self):
        conv = WeightNormConv1d(8, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 8, 10)  # (batch, channels, length)
        y = conv(x)
        self.assertEqual(y.shape, (2, 16, 10))

    def test_training_vs_eval(self):
        """Output should be computed in both train and eval mode."""
        conv = WeightNormConv1d(4, 8, kernel_size=1)
        x = torch.randn(1, 4, 5)
        conv.train()
        y_train = conv(x)
        conv.eval()
        y_eval = conv(x)
        # Both should succeed and have the same shape
        self.assertEqual(y_train.shape, y_eval.shape)

    def test_params_registered(self):
        conv = WeightNormConv1d(4, 8, kernel_size=1)
        param_names = [n for n, _ in conv.named_parameters()]
        self.assertIn("V", param_names)
        self.assertIn("g", param_names)
        self.assertIn("b", param_names)


# ---------------------------------------------------------------------------
# StackedRNN
# ---------------------------------------------------------------------------

from eole.modules.stacked_rnn import StackedLSTM, StackedGRU


class TestStackedLSTM(unittest.TestCase):

    def test_forward_shape(self):
        rnn = StackedLSTM(num_layers=2, input_size=16, hidden_size=32, dropout=0.0)
        x = torch.randn(4, 16)  # (batch, input_size)
        h = torch.zeros(2, 4, 32)
        c = torch.zeros(2, 4, 32)
        out, (h_out, c_out) = rnn(x, (h, c))
        self.assertEqual(out.shape, (4, 32))
        self.assertEqual(h_out.shape, (2, 4, 32))

    def test_single_layer(self):
        rnn = StackedLSTM(num_layers=1, input_size=8, hidden_size=16, dropout=0.0)
        x = torch.randn(3, 8)
        h = torch.zeros(1, 3, 16)
        c = torch.zeros(1, 3, 16)
        out, (h_out, c_out) = rnn(x, (h, c))
        self.assertEqual(out.shape, (3, 16))


class TestStackedGRU(unittest.TestCase):

    def test_forward_shape(self):
        rnn = StackedGRU(num_layers=2, input_size=16, hidden_size=32, dropout=0.0)
        x = torch.randn(4, 16)  # (batch, input_size)
        # StackedGRU.forward expects hidden as a TUPLE (h_0,)
        # where h_0 shape is (num_layers, batch, hidden_size)
        h_0 = torch.zeros(2, 4, 32)
        out, (h_1,) = rnn(x, (h_0,))
        self.assertEqual(out.shape, (4, 32))
        self.assertEqual(h_1.shape, (2, 4, 32))

    def test_output_is_finite(self):
        rnn = StackedGRU(num_layers=2, input_size=8, hidden_size=16, dropout=0.0)
        x = torch.randn(2, 8)
        h_0 = torch.zeros(2, 2, 16)
        with torch.no_grad():
            out, _ = rnn(x, (h_0,))
        self.assertTrue(torch.isfinite(out).all())


# ---------------------------------------------------------------------------
# ContextGate
# ---------------------------------------------------------------------------

from eole.modules.contextgate import (
    context_gate_factory,
    ContextGate,
    SourceContextGate,
    TargetContextGate,
    BothContextGate,
)


class TestContextGate(unittest.TestCase):

    def _sizes(self):
        return dict(embeddings_size=8, decoder_size=8, attention_size=8, output_size=8)

    def _inputs(self, batch=3):
        D = 8
        prev_emb = torch.randn(batch, D)
        dec_state = torch.randn(batch, D)
        attn_state = torch.randn(batch, D)
        return prev_emb, dec_state, attn_state

    def test_source_gate_output_shape(self):
        gate = SourceContextGate(**self._sizes())
        y = gate(*self._inputs())
        self.assertEqual(y.shape, (3, 8))

    def test_target_gate_output_shape(self):
        gate = TargetContextGate(**self._sizes())
        y = gate(*self._inputs())
        self.assertEqual(y.shape, (3, 8))

    def test_both_gate_output_shape(self):
        gate = BothContextGate(**self._sizes())
        y = gate(*self._inputs())
        self.assertEqual(y.shape, (3, 8))

    def test_factory_source(self):
        gate = context_gate_factory("source", 8, 8, 8, 8)
        self.assertIsInstance(gate, SourceContextGate)

    def test_factory_target(self):
        gate = context_gate_factory("target", 8, 8, 8, 8)
        self.assertIsInstance(gate, TargetContextGate)

    def test_factory_both(self):
        gate = context_gate_factory("both", 8, 8, 8, 8)
        self.assertIsInstance(gate, BothContextGate)

    def test_factory_invalid_raises(self):
        with self.assertRaises(AssertionError):
            context_gate_factory("unknown", 8, 8, 8, 8)


# ---------------------------------------------------------------------------
# GlobalAttention
# ---------------------------------------------------------------------------

from eole.modules.global_attention import GlobalAttention


class TestGlobalAttention(unittest.TestCase):

    def _make(self, attn_type="general"):
        return GlobalAttention(dim=16, attn_type=attn_type)

    def test_dot_forward_shape(self):
        attn = GlobalAttention(dim=16, attn_type="dot")
        src = torch.randn(2, 5, 16)   # (batch, tgt_len, dim)
        enc = torch.randn(2, 8, 16)   # (batch, src_len, dim)
        out, align = attn(src, enc)
        self.assertEqual(out.shape, (2, 5, 16))
        self.assertEqual(align.shape, (2, 5, 8))

    def test_general_forward_shape(self):
        attn = GlobalAttention(dim=16, attn_type="general")
        src = torch.randn(2, 5, 16)
        enc = torch.randn(2, 8, 16)
        out, align = attn(src, enc)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_mlp_forward_shape(self):
        attn = GlobalAttention(dim=16, attn_type="mlp")
        src = torch.randn(2, 3, 16)
        enc = torch.randn(2, 6, 16)
        out, align = attn(src, enc)
        self.assertEqual(out.shape, (2, 3, 16))

    def test_single_step_input(self):
        """Single query step: src input is (batch, dim) rather than (batch, tgt_len, dim).
        The output is squeezed to (batch, dim) not (batch, 1, dim)."""
        attn = GlobalAttention(dim=16, attn_type="general")
        src = torch.randn(2, 16)   # 2D input
        enc = torch.randn(2, 8, 16)
        out, align = attn(src, enc)
        # Single-step output is squeezed back to (batch, dim)
        self.assertEqual(out.shape, (2, 16))
        self.assertEqual(align.shape, (2, 8))

    def test_attention_weights_sum_to_one(self):
        attn = GlobalAttention(dim=16, attn_type="dot")
        src = torch.randn(2, 3, 16)
        enc = torch.randn(2, 5, 16)
        with torch.no_grad():
            _, align = attn(src, enc)
        sums = align.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_invalid_attn_type_raises(self):
        with self.assertRaises(ValueError):
            GlobalAttention(dim=16, attn_type="invalid_type")


# ---------------------------------------------------------------------------
# AlibiPositionalBias
# ---------------------------------------------------------------------------

from eole.modules.alibi_position_bias import AlibiPositionalBias


class TestAlibiPositionalBias(unittest.TestCase):

    def test_get_bias_shape(self):
        """get_bias returns (1, i, j); slopes broadcast to (heads, i, j) in forward."""
        bias = AlibiPositionalBias(heads=4)
        b = bias.get_bias(i=8, j=8, device=torch.device("cpu"))
        # Returns (1, i, j); broadcast with slopes (heads, 1, 1) gives (heads, i, j)
        self.assertEqual(b.shape, (1, 8, 8))

    def test_forward_shape(self):
        """AlibiPositionalBias.forward adds bias to qk_dots scores."""
        bias = AlibiPositionalBias(heads=4)
        # qk_dots: (batch, heads, tgt_len, src_len)
        qk_dots = torch.randn(2, 4, 6, 6)
        out = bias(qk_dots)
        self.assertEqual(out.shape, qk_dots.shape)

    def test_slopes_positive(self):
        """Alibi slopes should be strictly positive."""
        bias = AlibiPositionalBias(heads=4)
        slopes = bias.slopes
        self.assertTrue((slopes > 0).all())


# ---------------------------------------------------------------------------
# RelativePositionBias helpers
# ---------------------------------------------------------------------------

from eole.modules.relative_position_bias import relative_matmul, gen_relative_positions


class TestRelativePositionBias(unittest.TestCase):

    def test_relative_matmul_shape(self):
        """relative_matmul(x, z, transpose=True): z is (length, length, D); result is (batch, heads, length, length)."""
        batch, heads, L, D = 2, 4, 6, 8
        x = torch.randn(batch, heads, L, D)
        # z must be 3D: (length, length, D) for transpose=True path
        z = torch.randn(L, L, D)
        out = relative_matmul(x, z, transpose=True)
        self.assertEqual(out.shape, (batch, heads, L, L))

    def test_gen_relative_positions_shape(self):
        """gen_relative_positions returns an (L, L) integer index matrix."""
        L = 8
        n_positions = 4
        pos = gen_relative_positions(L, n_positions)
        self.assertEqual(pos.shape, (L, L))

    def test_gen_relative_positions_dtype(self):
        """gen_relative_positions returns integer indices."""
        pos = gen_relative_positions(4, 2)
        self.assertTrue(pos.dtype in (torch.int64, torch.int32, torch.long))


# ---------------------------------------------------------------------------
# Sparsemax / LogSparsemax
# ---------------------------------------------------------------------------

from eole.modules.sparse_activations import Sparsemax, LogSparsemax


class TestSparsemax(unittest.TestCase):

    def _make_input(self):
        return torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])

    def test_output_shape(self):
        sp = Sparsemax(dim=1)
        x = self._make_input()
        y = sp(x)
        self.assertEqual(y.shape, x.shape)

    def test_non_negative(self):
        sp = Sparsemax(dim=1)
        y = sp(self._make_input())
        self.assertTrue((y >= 0).all())

    def test_sums_to_one(self):
        sp = Sparsemax(dim=1)
        y = sp(self._make_input())
        self.assertTrue(torch.allclose(y.sum(dim=1), torch.ones(2), atol=1e-5))

    def test_sparsity(self):
        """Top element should get all weight for very skewed input."""
        sp = Sparsemax(dim=1)
        x = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        y = sp(x)
        self.assertAlmostEqual(y[0, 0].item(), 1.0, places=4)
        self.assertAlmostEqual(y[0, 1].item(), 0.0, places=4)


class TestLogSparsemax(unittest.TestCase):

    def test_output_shape(self):
        log_sp = LogSparsemax(dim=1)
        x = torch.randn(3, 5)
        y = log_sp(x)
        self.assertEqual(y.shape, x.shape)

    def test_non_positive(self):
        """Log-probabilities must be ≤ 0."""
        log_sp = LogSparsemax(dim=1)
        x = torch.randn(3, 5)
        y = log_sp(x)
        self.assertTrue((y <= 0 + 1e-7).all())  # allow tiny fp error


# ---------------------------------------------------------------------------
# SparsemaxLoss
# ---------------------------------------------------------------------------

from eole.modules.sparse_losses import SparsemaxLoss


class TestSparsemaxLoss(unittest.TestCase):

    def test_output_shape_elementwise_mean(self):
        loss_fn = SparsemaxLoss(reduction="elementwise_mean")
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, targets)
        self.assertEqual(loss.shape, ())  # scalar

    def test_output_shape_none(self):
        loss_fn = SparsemaxLoss(reduction="none")
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, targets)
        self.assertEqual(loss.shape, (4,))

    def test_loss_non_negative(self):
        loss_fn = SparsemaxLoss(reduction="none")
        logits = torch.randn(6, 8)
        targets = torch.randint(0, 8, (6,))
        loss = loss_fn(logits, targets)
        self.assertTrue((loss >= 0).all())

    def test_perfect_prediction_near_zero(self):
        """If the model assigns mass 1 to the correct class, loss should be ~0."""
        loss_fn = SparsemaxLoss(reduction="none")
        # Extreme logits so sparsemax concentrates on class 0
        logits = torch.tensor([[100.0, -100.0, -100.0]])
        targets = torch.tensor([0])
        loss = loss_fn(logits, targets)
        self.assertAlmostEqual(loss[0].item(), 0.0, places=3)


# ---------------------------------------------------------------------------
# LoRA Embedding
# ---------------------------------------------------------------------------

from eole.modules.lora import LoRALayer, Embedding as LoRAEmbedding


class TestLoRALayer(unittest.TestCase):
    """Test the LoRALayer mixin class independently."""

    def test_init_sets_rank(self):
        class _DummyLora(LoRALayer):
            def __init__(self):
                super().__init__(r=4, lora_alpha=8, lora_dropout=0.0, merge_weights=False)

        dummy = _DummyLora()
        self.assertEqual(dummy.r, 4)
        self.assertEqual(dummy.lora_alpha, 8)
        # LoRALayer base class does NOT set scaling; only subclasses do
        self.assertFalse(hasattr(dummy, "scaling"))

    def test_zero_rank(self):
        class _DummyLora(LoRALayer):
            def __init__(self):
                super().__init__(r=0, lora_alpha=1, lora_dropout=0.0, merge_weights=False)

        dummy = _DummyLora()
        self.assertEqual(dummy.r, 0)

    def test_merged_initial_state(self):
        class _DummyLora(LoRALayer):
            def __init__(self):
                super().__init__(r=2, lora_alpha=4, lora_dropout=0.0, merge_weights=True)

        dummy = _DummyLora()
        self.assertFalse(dummy.merged)


class TestLoRAEmbedding(unittest.TestCase):
    """Test the LoRA-augmented embedding via the Embedding subclass."""

    def test_forward_shape(self):
        emb = LoRAEmbedding(
            num_embeddings=100,
            embedding_dim=16,
            r=4,
            lora_alpha=8,
        )
        idx = torch.randint(0, 100, (3, 8))
        out = emb(idx)
        self.assertEqual(out.shape, (3, 8, 16))

    def test_scaling(self):
        emb = LoRAEmbedding(num_embeddings=50, embedding_dim=16, r=4, lora_alpha=8)
        self.assertAlmostEqual(emb.scaling, 2.0)  # 8 / 4


# ---------------------------------------------------------------------------
# Embeddings / PositionalEncoding
# ---------------------------------------------------------------------------

from eole.modules.embeddings import PositionalEncoding, Embeddings
from eole.constants import PositionEncodingType


class TestPositionalEncoding(unittest.TestCase):

    def test_sinusoidal_shape(self):
        pe = PositionalEncoding(dim=32, enc_type=PositionEncodingType.SinusoidalInterleaved, max_len=100)
        emb = torch.randn(2, 10, 32)
        out = pe(emb)
        self.assertEqual(out.shape, (2, 10, 32))

    def test_sinusoidal_concat_shape(self):
        pe = PositionalEncoding(dim=32, enc_type=PositionEncodingType.SinusoidalConcat, max_len=100)
        emb = torch.randn(2, 10, 32)
        out = pe(emb)
        self.assertEqual(out.shape, (2, 10, 32))

    def test_invalid_enc_type_raises(self):
        """PositionalEncoding raises ValueError for an unsupported enc_type."""
        with self.assertRaises(ValueError):
            PositionalEncoding(dim=32, enc_type="unsupported_type", max_len=100)


class TestEmbeddings(unittest.TestCase):

    def _make(self, vocab_size=50, vec_size=32, position_encoding_type="SinusoidalInterleaved"):
        return Embeddings(
            word_vec_size=vec_size,
            word_vocab_size=vocab_size,
            word_padding_idx=1,
            position_encoding_type=position_encoding_type,
            dropout=0.0,
        )

    def test_forward_shape(self):
        emb_layer = self._make()
        # Embeddings.forward expects (batch, seq_len) integer input
        src = torch.randint(0, 50, (3, 8))
        out = emb_layer(src)
        self.assertEqual(out.shape, (3, 8, 32))

    def test_step_argument(self):
        """Forward with step= for single-position decoding."""
        emb_layer = self._make()
        src = torch.randint(0, 50, (3, 1))
        out = emb_layer(src, step=0)
        self.assertEqual(out.shape, (3, 1, 32))

    def test_learned_position_encoding_shape(self):
        emb_layer = self._make(position_encoding_type="Learned")
        src = torch.randint(0, 50, (2, 6))
        out = emb_layer(src)
        self.assertEqual(out.shape, (2, 6, 32))


if __name__ == "__main__":
    unittest.main()
