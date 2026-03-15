"""Unit tests for eole/encoders/ and eole/decoders/ – Step 5."""

import unittest
import torch
import torch.nn as nn

from eole.encoders.mean_encoder import MeanEncoder
from eole.encoders.transformer import TransformerEncoder, TransformerEncoderLayer
from eole.decoders.decoder import DecoderBase
from eole.config.models import (
    MeanEncoderConfig,
    TransformerEncoderConfig,
    TransformerDecoderConfig,
)
from eole.constants import PositionEncodingType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transformer_enc_config(
    hidden_size=64,
    heads=4,
    transformer_ff=128,
    layers=2,
    position_encoding_type=PositionEncodingType.SinusoidalInterleaved,
):
    return TransformerEncoderConfig(
        hidden_size=hidden_size,
        heads=heads,
        transformer_ff=transformer_ff,
        layers=layers,
        position_encoding_type=position_encoding_type,
    )


class _FakeRunningConfig:
    parallel_gpu = 1
    dropout = [0.0]
    attention_dropout = [0.0]
    use_ckpting = []
    self_attn_backend = "pytorch"


# ===========================================================================
# MeanEncoder
# ===========================================================================


class TestMeanEncoder(unittest.TestCase):

    def _make(self, num_layers=2):
        cfg = MeanEncoderConfig(layers=num_layers)
        return MeanEncoder(cfg)

    def test_output_shape(self):
        encoder = self._make(num_layers=2)
        B, S, D = 3, 10, 32
        emb = torch.randn(B, S, D)
        enc_out, enc_final_hs = encoder(emb)
        self.assertEqual(enc_out.shape, (B, S, D))

    def test_final_hidden_state_shape(self):
        encoder = self._make(num_layers=2)
        B, S, D = 3, 10, 32
        emb = torch.randn(B, S, D)
        _, enc_final_hs = encoder(emb)
        h, c = enc_final_hs
        self.assertEqual(h.shape, (2, B, D))  # (num_layers, batch, dim)
        self.assertEqual(c.shape, (2, B, D))

    def test_enc_out_unchanged(self):
        """MeanEncoder should return input embeddings unchanged."""
        encoder = self._make()
        emb = torch.randn(2, 5, 16)
        enc_out, _ = encoder(emb)
        self.assertTrue(torch.allclose(enc_out, emb))

    def test_with_pad_mask(self):
        """Mean pooling with a padding mask should not crash."""
        encoder = self._make()
        B, S, D = 2, 8, 16
        emb = torch.randn(B, S, D)
        # Last 2 positions are padding
        pad_mask = torch.zeros(B, S, dtype=torch.bool)
        pad_mask[:, 6:] = True
        enc_out, enc_final_hs = encoder(emb, pad_mask=pad_mask)
        self.assertEqual(enc_out.shape, (B, S, D))

    def test_pad_mask_4d(self):
        """4D pad_mask (batch, 1, 1, seq) should also be handled."""
        encoder = self._make()
        B, S, D = 2, 6, 16
        emb = torch.randn(B, S, D)
        pad_mask = torch.zeros(B, 1, 1, S, dtype=torch.bool)
        enc_out, _ = encoder(emb, pad_mask=pad_mask)
        self.assertEqual(enc_out.shape, (B, S, D))

    def test_no_padding_same_as_global_mean(self):
        """Without padding, MeanEncoder mean should equal global mean."""
        encoder = self._make()
        B, S, D = 2, 5, 8
        emb = torch.randn(B, S, D)
        _, enc_final_hs = encoder(emb)
        h, _ = enc_final_hs
        expected_mean = emb.mean(dim=1)  # (B, D)
        self.assertTrue(torch.allclose(h[0], expected_mean))

    def test_masked_mean_ignores_padding(self):
        """Masked mean should ignore padded positions."""
        encoder = self._make(num_layers=1)
        # Batch of 1 with 2 real tokens and 2 padded
        emb = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [99.0, 0.0], [99.0, 0.0]]])
        pad_mask = torch.tensor([[False, False, True, True]])
        _, enc_final_hs = encoder(emb, pad_mask=pad_mask)
        h, _ = enc_final_hs
        expected_mean = torch.tensor([[2.0, 0.0]])
        self.assertTrue(torch.allclose(h[0], expected_mean))


# ===========================================================================
# TransformerEncoderLayer
# ===========================================================================


class TestTransformerEncoderLayer(unittest.TestCase):

    def _make(self, hidden_size=64, heads=4):
        cfg = _make_transformer_enc_config(hidden_size=hidden_size, heads=heads, layers=1)
        return TransformerEncoderLayer(cfg, running_config=_FakeRunningConfig())

    def _make_attn_mask(self, B, S, padding_positions=None):
        """Build a (B, 1, 1, S) attention mask matching the encoder's internal format."""
        pad_mask = torch.zeros(B, 1, S, dtype=torch.bool)
        if padding_positions is not None:
            for b, pos in padding_positions:
                pad_mask[b, 0, pos] = True
        # attn_mask is ~pad_mask, then unsqueeze once more to (B, 1, 1, S)
        return (~pad_mask).unsqueeze(1)

    def test_output_shape(self):
        layer = self._make()
        B, S, D = 2, 6, 64
        x = torch.randn(B, S, D)
        attn_mask = self._make_attn_mask(B, S)
        y = layer(x, attn_mask)
        self.assertEqual(y.shape, (B, S, D))

    def test_output_is_finite(self):
        layer = self._make()
        # skip_init leaves weights uninitialized; explicitly initialize
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        B, S, D = 2, 5, 64
        x = torch.randn(B, S, D)
        attn_mask = self._make_attn_mask(B, S)
        with torch.no_grad():
            y = layer(x, attn_mask)
        self.assertTrue(torch.isfinite(y).all())

    def test_update_dropout(self):
        layer = self._make()
        layer.update_dropout(0.3, 0.1)
        self.assertAlmostEqual(layer.dropout_p, 0.3)
        self.assertAlmostEqual(layer.dropout.p, 0.3)


# ===========================================================================
# TransformerEncoder
# ===========================================================================


class TestTransformerEncoder(unittest.TestCase):

    def _make(self, hidden_size=64, heads=4, layers=2):
        cfg = _make_transformer_enc_config(hidden_size=hidden_size, heads=heads, layers=layers)
        return TransformerEncoder(cfg, running_config=_FakeRunningConfig())

    def _pad_mask(self, B, S, padding=None):
        """Build a (B, 1, S) pad mask: True = padding position."""
        mask = torch.zeros(B, 1, S, dtype=torch.bool)
        if padding:
            for b, start in padding:
                mask[b, 0, start:] = True
        return mask

    def test_output_shape(self):
        enc = self._make()
        B, S, D = 2, 8, 64
        emb = torch.randn(B, S, D)
        pad_mask = self._pad_mask(B, S)
        enc_out, final_hs = enc(emb, pad_mask=pad_mask)
        self.assertEqual(enc_out.shape, (B, S, D))
        self.assertIsNone(final_hs)

    def test_requires_pad_mask(self):
        enc = self._make()
        emb = torch.randn(2, 5, 64)
        with self.assertRaises(AssertionError):
            enc(emb)  # no pad_mask

    def test_output_finite(self):
        enc = self._make()
        # skip_init leaves weights uninitialized; explicitly initialize
        for m in enc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        B, S, D = 2, 6, 64
        emb = torch.randn(B, S, D)
        pad_mask = self._pad_mask(B, S)
        with torch.no_grad():
            enc_out, _ = enc(emb, pad_mask=pad_mask)
        self.assertTrue(torch.isfinite(enc_out).all())

    def test_update_dropout(self):
        """update_dropout should not raise."""
        enc = self._make()
        enc.update_dropout(0.2, 0.1)  # should not raise

    def test_with_padding(self):
        """Encoder should work with some padded positions."""
        enc = self._make()
        B, S, D = 2, 8, 64
        emb = torch.randn(B, S, D)
        pad_mask = self._pad_mask(B, S, padding=[(0, 6), (1, 5)])
        enc_out, _ = enc(emb, pad_mask=pad_mask)
        self.assertEqual(enc_out.shape, (B, S, D))

    def test_single_layer(self):
        """Works with a single transformer layer."""
        cfg = _make_transformer_enc_config(hidden_size=64, heads=4, layers=1)
        enc = TransformerEncoder(cfg, running_config=_FakeRunningConfig())
        emb = torch.randn(1, 4, 64)
        pad_mask = self._pad_mask(1, 4)
        enc_out, _ = enc(emb, pad_mask=pad_mask)
        self.assertEqual(enc_out.shape, (1, 4, 64))


# ===========================================================================
# DecoderBase: abstract interface contract
# ===========================================================================


class TestDecoderBase(unittest.TestCase):
    """Ensure DecoderBase enforces its abstract interface."""

    def test_cannot_instantiate_directly(self):
        with self.assertRaises(TypeError):
            DecoderBase()

    def test_concrete_subclass_can_instantiate(self):
        class _ConcreteDecoder(DecoderBase):
            def init_state(self, **kwargs):
                pass
            def map_state(self, fn):
                pass
            def forward(self, emb, enc_out=None, step=None, **kwargs):
                return emb, {}

        dec = _ConcreteDecoder(attentional=False)
        self.assertFalse(dec.attentional)
        self.assertEqual(dec.state, {})

    def test_attentional_flag_stored(self):
        class _Dec(DecoderBase):
            def init_state(self, **kwargs): pass
            def map_state(self, fn): pass
            def forward(self, emb, enc_out=None, step=None, **kwargs): return emb, {}

        dec = _Dec(attentional=True)
        self.assertTrue(dec.attentional)


if __name__ == "__main__":
    unittest.main()
