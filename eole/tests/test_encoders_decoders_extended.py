"""Extended unit tests for eole/encoders/ and eole/decoders/ – covering files
not tested in test_encoder_decoder.py.

Covers:
 - encoders/rnn_encoder.py: RNNEncoder (LSTM / GRU / brnn, with/without bridge)
 - encoders/cnn_encoder.py: CNNEncoder
 - decoders/rnn_decoder.py: StdRNNDecoder, InputFeedRNNDecoder
 - decoders/cnn_decoder.py: CNNDecoder
 - decoders/ensemble.py: EnsembleDecoderOutput, EnsembleDecoder helper classes
"""

import unittest
import torch
import torch.nn as nn

from eole.config.models import (
    RnnEncoderConfig,
    CnnEncoderConfig,
    RnnDecoderConfig,
    CnnDecoderConfig,
)
from eole.encoders.rnn_encoder import RNNEncoder
from eole.encoders.cnn_encoder import CNNEncoder
from eole.decoders.rnn_decoder import StdRNNDecoder, InputFeedRNNDecoder
from eole.decoders.cnn_decoder import CNNDecoder
from eole.decoders.ensemble import (
    EnsembleDecoderOutput,
    EnsembleEncoder,
    EnsembleDecoder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRunning:
    """Minimal running config stub that satisfies encoder/decoder init."""
    dropout = [0.0]
    attention_dropout = [0.0]
    use_ckpting = []


def _rnn_enc_config(**kwargs):
    defaults = dict(
        hidden_size=32,
        src_word_vec_size=16,
        layers=2,
        rnn_type="LSTM",
        bridge=False,
        encoder_type="rnn",
    )
    defaults.update(kwargs)
    return RnnEncoderConfig(**defaults)


def _rnn_dec_config(**kwargs):
    defaults = dict(
        hidden_size=32,
        tgt_word_vec_size=16,
        layers=2,
        rnn_type="LSTM",
        global_attention="general",
        bidirectional_encoder=False,
    )
    defaults.update(kwargs)
    return RnnDecoderConfig(**defaults)


# ===========================================================================
# RNNEncoder
# ===========================================================================


class TestRNNEncoder(unittest.TestCase):

    def _make(self, **kwargs):
        cfg = _rnn_enc_config(**kwargs)
        return RNNEncoder(cfg, running_config=_FakeRunning())

    def test_lstm_output_shape(self):
        enc = self._make(rnn_type="LSTM")
        emb = torch.randn(3, 8, 16)  # (batch, seq, embed)
        enc_out, (h, c) = enc(emb)
        self.assertEqual(enc_out.shape, (3, 8, 32))
        # LSTM final states: (num_layers, batch, hidden)
        self.assertEqual(h.shape, (2, 3, 32))

    def test_gru_output_shape(self):
        enc = self._make(rnn_type="GRU")
        emb = torch.randn(3, 8, 16)
        enc_out, h = enc(emb)
        self.assertEqual(enc_out.shape, (3, 8, 32))
        self.assertEqual(h.shape, (2, 3, 32))

    def test_brnn_output_shape(self):
        """Bidirectional RNN: hidden_size must be divisible by 2."""
        enc = self._make(encoder_type="brnn", hidden_size=32, rnn_type="LSTM")
        emb = torch.randn(2, 6, 16)
        enc_out, (h, c) = enc(emb)
        # brnn still produces (batch, seq, hidden_size) output
        self.assertEqual(enc_out.shape, (2, 6, 32))

    def test_bridge_output_shape(self):
        enc = self._make(bridge=True)
        emb = torch.randn(2, 5, 16)
        enc_out, (h, c) = enc(emb)
        self.assertEqual(enc_out.shape, (2, 5, 32))
        self.assertEqual(h.shape, (2, 2, 32))

    def test_output_is_finite(self):
        enc = self._make()
        emb = torch.randn(2, 5, 16)
        with torch.no_grad():
            enc_out, _ = enc(emb)
        self.assertTrue(torch.isfinite(enc_out).all())

    def test_hidden_size_not_divisible_brnn_raises(self):
        """brnn requires hidden_size divisible by 2; RNNEncoder raises ValueError."""
        cfg = RnnEncoderConfig(
            hidden_size=17,
            src_word_vec_size=16,
            layers=1,
            rnn_type="LSTM",
            encoder_type="brnn",
        )
        with self.assertRaises(ValueError):
            RNNEncoder(cfg, running_config=_FakeRunning())


# ===========================================================================
# CNNEncoder
# ===========================================================================


class TestCNNEncoder(unittest.TestCase):

    def _make(self, layers=2, hidden_size=32, kernel_width=3):
        cfg = CnnEncoderConfig(
            hidden_size=hidden_size,
            src_word_vec_size=hidden_size,
            layers=layers,
            cnn_kernel_width=kernel_width,
        )
        return CNNEncoder(cfg, running_config=_FakeRunning())

    def test_output_shape(self):
        enc = self._make()
        emb = torch.randn(2, 8, 32)  # (batch, seq, hidden)
        enc_out, projected = enc(emb)
        self.assertEqual(enc_out.shape, (2, 8, 32))
        self.assertEqual(projected.shape, (2, 8, 32))

    def test_single_layer(self):
        enc = self._make(layers=1)
        emb = torch.randn(2, 6, 32)
        enc_out, _ = enc(emb)
        self.assertEqual(enc_out.shape, (2, 6, 32))

    def test_output_is_finite(self):
        enc = self._make()
        emb = torch.randn(2, 5, 32)
        with torch.no_grad():
            enc_out, _ = enc(emb)
        self.assertTrue(torch.isfinite(enc_out).all())

    def test_update_dropout(self):
        enc = self._make()
        enc.update_dropout(0.3)  # should not raise


# ===========================================================================
# StdRNNDecoder
# ===========================================================================


class TestStdRNNDecoder(unittest.TestCase):

    def _make(self, rnn_type="LSTM", **kwargs):
        cfg = _rnn_dec_config(rnn_type=rnn_type, **kwargs)
        return StdRNNDecoder(cfg, running_config=_FakeRunning())

    def _init_state(self, dec, enc, batch, src_len):
        """Run encoder and set decoder initial state."""
        enc_cfg = _rnn_enc_config(rnn_type=dec.rnn.mode if hasattr(dec.rnn, "mode") else "LSTM")
        # Build an RNN encoder with matching dims
        enc_out = torch.randn(batch, src_len, 32)
        # Build fake LSTM state (num_layers, batch, hidden)
        h = torch.zeros(2, batch, 32)
        c = torch.zeros(2, batch, 32)
        dec.init_state(enc_out=enc_out, enc_final_hs=(h, c))
        return enc_out

    def test_lstm_forward_shape(self):
        dec = self._make()
        batch, tgt_len, src_len = 2, 4, 6
        enc_out = self._init_state(dec, None, batch, src_len)
        emb = torch.randn(batch, tgt_len, 16)
        dec_out, attns = dec(emb, enc_out=enc_out)
        self.assertEqual(dec_out.shape, (batch, tgt_len, 32))
        self.assertIn("std", attns)

    def test_gru_forward_shape(self):
        dec = self._make(rnn_type="GRU")
        batch, tgt_len, src_len = 2, 4, 6
        enc_out = torch.randn(batch, src_len, 32)
        # GRU state: (h,) where h is (num_layers, batch, hidden)
        h = torch.zeros(2, batch, 32)
        dec.init_state(enc_out=enc_out, enc_final_hs=(h,))
        emb = torch.randn(batch, tgt_len, 16)
        dec_out, attns = dec(emb, enc_out=enc_out)
        self.assertEqual(dec_out.shape, (batch, tgt_len, 32))

    def test_map_state(self):
        """map_state should apply fn to all state tensors."""
        dec = self._make()
        enc_out = self._init_state(dec, None, 2, 5)
        # Double the batch dimension
        dec.map_state(lambda s: torch.cat([s, s], dim=0))
        # Hidden should now have batch=4
        self.assertEqual(dec.state["hidden"][0].shape[1], 4)

    def test_update_dropout(self):
        dec = self._make()
        dec.update_dropout(0.2, 0.1)
        self.assertAlmostEqual(dec.dropout.p, 0.2)


# ===========================================================================
# InputFeedRNNDecoder
# ===========================================================================


class TestInputFeedRNNDecoder(unittest.TestCase):

    def _make(self, rnn_type="LSTM", **kwargs):
        cfg = _rnn_dec_config(rnn_type=rnn_type, **kwargs)
        return InputFeedRNNDecoder(cfg, running_config=_FakeRunning())

    def _init_state(self, dec, batch, src_len):
        enc_out = torch.randn(batch, src_len, 32)
        h = torch.zeros(2, batch, 32)
        c = torch.zeros(2, batch, 32)
        dec.init_state(enc_out=enc_out, enc_final_hs=(h, c))
        return enc_out

    def test_lstm_forward_shape(self):
        dec = self._make()
        batch, tgt_len, src_len = 2, 4, 6
        enc_out = self._init_state(dec, batch, src_len)
        emb = torch.randn(batch, tgt_len, 16)
        dec_out, attns = dec(emb, enc_out=enc_out)
        self.assertEqual(dec_out.shape, (batch, tgt_len, 32))
        self.assertIn("std", attns)

    def test_gru_forward_shape(self):
        dec = self._make(rnn_type="GRU")
        batch, tgt_len, src_len = 2, 3, 5
        enc_out = torch.randn(batch, src_len, 32)
        h = torch.zeros(2, batch, 32)
        dec.init_state(enc_out=enc_out, enc_final_hs=(h,))
        emb = torch.randn(batch, tgt_len, 16)
        dec_out, attns = dec(emb, enc_out=enc_out)
        self.assertEqual(dec_out.shape, (batch, tgt_len, 32))

    def test_output_is_finite(self):
        dec = self._make()
        batch, tgt_len, src_len = 2, 3, 4
        enc_out = self._init_state(dec, batch, src_len)
        emb = torch.randn(batch, tgt_len, 16)
        with torch.no_grad():
            dec_out, _ = dec(emb, enc_out=enc_out)
        self.assertTrue(torch.isfinite(dec_out).all())


# ===========================================================================
# CNNDecoder
# ===========================================================================


class TestCNNDecoder(unittest.TestCase):

    def _make(self, layers=2, hidden_size=32):
        cfg = CnnDecoderConfig(
            hidden_size=hidden_size,
            tgt_word_vec_size=hidden_size,
            layers=layers,
            cnn_kernel_width=3,
            global_attention=None,
        )
        return CNNDecoder(cfg, running_config=_FakeRunning())

    def test_forward_shape(self):
        dec = self._make()
        batch, tgt_len, src_len, D = 2, 5, 8, 32
        enc_out = torch.randn(batch, src_len, D)
        enc_final = torch.randn(batch, src_len, D)  # used in init_state
        dec.init_state(enc_out=enc_out, enc_final_hs=enc_final)
        emb = torch.randn(batch, tgt_len, D)
        dec_out, attns = dec(emb, enc_out=enc_out)
        self.assertEqual(dec_out.shape, (batch, tgt_len, D))

    def test_update_dropout(self):
        dec = self._make()
        dec.update_dropout(0.2)  # should not raise

    def test_map_state(self):
        dec = self._make()
        batch, src_len, D = 2, 6, 32
        enc_out = torch.randn(batch, src_len, D)
        dec.init_state(enc_out=enc_out, enc_final_hs=enc_out)
        # Double batch dim
        dec.map_state(lambda s: torch.cat([s, s], dim=0))
        self.assertEqual(dec.state["src"].shape[0], 4)


# ===========================================================================
# Ensemble data classes
# ===========================================================================


class TestEnsembleDecoderOutput(unittest.TestCase):

    def test_indexing(self):
        a = torch.randn(2, 4, 16)
        b = torch.randn(2, 4, 16)
        out = EnsembleDecoderOutput([a, b])
        self.assertTrue(torch.equal(out[0], a))
        self.assertTrue(torch.equal(out[1], b))

    def test_squeeze(self):
        a = torch.randn(2, 1, 16)
        b = torch.randn(2, 1, 16)
        out = EnsembleDecoderOutput([a, b])
        squeezed = out.squeeze(1)
        self.assertEqual(squeezed[0].shape, (2, 16))

    def test_len(self):
        out = EnsembleDecoderOutput([torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4)])
        self.assertEqual(len(out.model_dec_outs), 3)


class TestEnsembleEncoder(unittest.TestCase):
    """EnsembleEncoder delegates to individual real encoders."""

    def _make_simple_encoders(self, n=2):
        """Build n tiny RNNEncoders to combine into an ensemble."""
        encs = []
        for _ in range(n):
            cfg = _rnn_enc_config(hidden_size=16, src_word_vec_size=8, layers=1)
            encs.append(RNNEncoder(cfg, running_config=_FakeRunning()))
        return encs

    def test_forward_calls_all_encoders(self):
        encs = self._make_simple_encoders(2)
        ens = EnsembleEncoder(encs)
        # emb: list of 2 tensors, one per encoder
        emb_list = [torch.randn(2, 5, 8) for _ in range(2)]
        enc_outs, enc_finals = ens(emb_list)
        self.assertEqual(len(enc_outs), 2)
        # Each encoder output: (batch=2, seq=5, hidden=16)
        self.assertEqual(enc_outs[0].shape, (2, 5, 16))


if __name__ == "__main__":
    unittest.main()
