"""Comprehensive unit tests for eole/config/ configuration validation."""

import unittest
import torch
from pydantic import ValidationError

from eole.config.config import Config, get_config_dict
from eole.config.common import (
    DistributedConfig,
    LoRaConfig,
    QuantizeConfig,
    MiscConfig,
    LoggingConfig,
    RunningConfig,
)
from eole.config.data import (
    BaseVocabConfig,
    VocabConfig,
    Dataset,
    DataConfig,
)
from eole.config.inference import DecodingConfig, InferenceConfig
from eole.config.training import OptimizerConfig, TrainingConfig
from eole.config.models import (
    EmbeddingsConfig,
    TransformerConfig,
    TransformerDecoderConfig,
    TransformerEncoderConfig,
    RnnEncoderConfig,
    RnnDecoderConfig,
    CustomModelConfig,
    RnnModelConfig,
    TransformerModelConfig,
    TransformerLMModelConfig,
    TransformerEncoderModelConfig,
    RotaryPositionConfig,
    build_model_config,
)
from eole.config.run import TrainConfig, PredictConfig, BuildVocabConfig
from eole.config import recursive_model_fields_set, reorder_fields
from eole.constants import PositionEncodingType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_train_config(**extra):
    """Return the minimum required kwargs for a valid TrainConfig."""
    base = dict(
        src_vocab="dummy",
        share_vocab=True,
        data={},
        model={"architecture": "custom"},
    )
    base.update(extra)
    return base


def _minimal_predict_config(**extra):
    """Return minimum required kwargs for a valid PredictConfig (no real files)."""
    base = dict(model_path=["dummy"], src="dummy")
    base.update(extra)
    return base


# ===========================================================================
# Config base class
# ===========================================================================


class TestConfigBase(unittest.TestCase):
    """Tests for the Config base class and get_config_dict()."""

    def test_extra_fields_rejected(self):
        """Config base uses extra='forbid' so unknown fields raise ValidationError."""
        with self.assertRaises(ValidationError):
            Config(unknown_field=42)

    def test_update_method_changes_field(self):
        """update() changes field values in-place."""

        class _Simple(Config):
            value: int = 1

        obj = _Simple()
        obj.update(value=99)
        self.assertEqual(obj.value, 99)

    def test_get_config_dict_keys(self):
        """get_config_dict() returns a ConfigDict with the expected settings."""
        cfg = get_config_dict()
        self.assertTrue(cfg.get("validate_assignment"))
        self.assertTrue(cfg.get("validate_default"))
        self.assertEqual(cfg.get("extra"), "forbid")


# ===========================================================================
# DistributedConfig
# ===========================================================================


class TestDistributedConfig(unittest.TestCase):
    def test_defaults(self):
        dc = DistributedConfig()
        self.assertEqual(dc.world_size, 1)
        self.assertEqual(dc.gpu_ranks, [])
        self.assertEqual(dc.parallel_mode, "data_parallel")

    def test_parallel_gpu_data_parallel(self):
        dc = DistributedConfig(world_size=4, parallel_mode="data_parallel")
        self.assertEqual(dc.parallel_gpu, 1)

    def test_parallel_gpu_tensor_parallel(self):
        dc = DistributedConfig(world_size=4, parallel_mode="tensor_parallel")
        self.assertEqual(dc.parallel_gpu, 4)

    def test_invalid_parallel_mode(self):
        with self.assertRaises(ValidationError):
            DistributedConfig(parallel_mode="unsupported_mode")


# ===========================================================================
# LoRaConfig
# ===========================================================================


class TestLoRaConfig(unittest.TestCase):
    def test_defaults(self):
        lc = LoRaConfig()
        self.assertEqual(lc.lora_layers, [])
        self.assertEqual(lc.lora_rank, 2)

    def test_custom_rank(self):
        lc = LoRaConfig(lora_rank=8, lora_alpha=4)
        self.assertEqual(lc.lora_rank, 8)
        self.assertEqual(lc.lora_alpha, 4)


# ===========================================================================
# QuantizeConfig
# ===========================================================================


class TestQuantizeConfig(unittest.TestCase):
    def test_defaults(self):
        qc = QuantizeConfig()
        self.assertEqual(qc.quant_type, "")
        self.assertEqual(qc.quant_layers, [])

    def test_valid_quant_types(self):
        for qtype in ["bnb_8bit", "bnb_FP4", "bnb_NF4", "awq_gemm", "awq_gemv", "autoround", "gguf", ""]:
            qc = QuantizeConfig(quant_type=qtype)
            self.assertEqual(qc.quant_type, qtype)

    def test_invalid_quant_type(self):
        with self.assertRaises(ValidationError):
            QuantizeConfig(quant_type="bad_type")


# ===========================================================================
# MiscConfig / LoggingConfig
# ===========================================================================


class TestMiscConfig(unittest.TestCase):
    def test_default_seed(self):
        mc = MiscConfig()
        self.assertEqual(mc.seed, -1)

    def test_custom_seed(self):
        mc = MiscConfig(seed=42)
        self.assertEqual(mc.seed, 42)


class TestLoggingConfig(unittest.TestCase):
    def test_defaults(self):
        lc = LoggingConfig()
        self.assertEqual(lc.log_file, "")
        self.assertEqual(lc.report_every, 50)
        self.assertFalse(lc.tensorboard)

    def test_tensorboard_enabled(self):
        lc = LoggingConfig(tensorboard=True, tensorboard_log_dir="/tmp/tb")
        self.assertTrue(lc.tensorboard)
        self.assertEqual(lc.tensorboard_log_dir, "/tmp/tb")


# ===========================================================================
# RunningConfig (compute_dtype validation)
# ===========================================================================


class TestRunningConfig(unittest.TestCase):
    def test_compute_dtype_default_is_fp32(self):
        rc = RunningConfig(model_path="dummy")
        self.assertEqual(rc.compute_dtype, torch.float32)

    def test_compute_dtype_string_fp16(self):
        rc = RunningConfig(model_path="dummy", compute_dtype="fp16")
        self.assertEqual(rc.compute_dtype, torch.float16)

    def test_compute_dtype_string_bf16(self):
        rc = RunningConfig(model_path="dummy", compute_dtype="bf16")
        self.assertEqual(rc.compute_dtype, torch.bfloat16)

    def test_compute_dtype_invalid(self):
        with self.assertRaises(ValidationError):
            RunningConfig(model_path="dummy", compute_dtype="float999")

    def test_compute_dtype_torch_dtype_passthrough(self):
        rc = RunningConfig(model_path="dummy", compute_dtype=torch.float16)
        self.assertEqual(rc.compute_dtype, torch.float16)


# ===========================================================================
# EmbeddingsConfig
# ===========================================================================


class TestEmbeddingsConfig(unittest.TestCase):
    def test_defaults(self):
        ec = EmbeddingsConfig()
        self.assertEqual(ec.src_word_vec_size, 512)
        self.assertEqual(ec.tgt_word_vec_size, 512)

    def test_learned_requires_n_positions(self):
        with self.assertRaises(ValidationError):
            EmbeddingsConfig(position_encoding_type=PositionEncodingType.Learned)

    def test_learned_with_n_positions(self):
        ec = EmbeddingsConfig(position_encoding_type=PositionEncodingType.Learned, n_positions=512)
        self.assertEqual(ec.n_positions, 512)

    def test_relative_requires_n_positions(self):
        with self.assertRaises(ValidationError):
            EmbeddingsConfig(position_encoding_type=PositionEncodingType.Relative)

    def test_sinusoidal_no_normalize(self):
        """Sinusoidal encoding cannot be combined with normalization."""
        with self.assertRaises(ValidationError):
            EmbeddingsConfig(
                position_encoding_type=PositionEncodingType.SinusoidalInterleaved,
                normalize=True,
            )

    def test_rotary_allows_normalize(self):
        """Rotary encoding is not affected by the sinusoidal normalization check."""
        ec = EmbeddingsConfig(
            position_encoding_type=PositionEncodingType.Rotary,
            normalize=True,
        )
        self.assertTrue(ec.normalize)

    def test_none_position_encoding_no_constraints(self):
        ec = EmbeddingsConfig(position_encoding_type=None)
        self.assertIsNone(ec.position_encoding_type)


# ===========================================================================
# TransformerConfig
# ===========================================================================


class TestTransformerConfig(unittest.TestCase):
    def test_defaults(self):
        tc = TransformerConfig()
        self.assertEqual(tc.heads, 8)
        self.assertEqual(tc.transformer_ff, 2048)

    def test_rotary_creates_rope_config(self):
        """When position_encoding_type=Rotary, rope_config is auto-created."""
        tc = TransformerConfig(position_encoding_type=PositionEncodingType.Rotary)
        self.assertIsInstance(tc.rope_config, RotaryPositionConfig)

    def test_non_rotary_no_rope_config(self):
        tc = TransformerConfig(position_encoding_type=PositionEncodingType.SinusoidalInterleaved)
        self.assertIsNone(tc.rope_config)

    def test_add_qkvbias_implies_final_linear_bias(self):
        """Setting add_qkvbias=True should auto-set add_final_linear_bias=True."""
        tc = TransformerConfig(add_qkvbias=True)
        self.assertTrue(tc.add_final_linear_bias)

    def test_dim_per_head_computed(self):
        """dim_per_head is hidden_size // heads when head_dim is not set."""
        tc = TransformerConfig()
        # hidden_size is at BaseModelConfig level; use CustomModelConfig
        mc = CustomModelConfig(hidden_size=512, heads=8)
        self.assertEqual(mc.dim_per_head, 64)

    def test_dim_per_head_explicit(self):
        mc = CustomModelConfig(hidden_size=512, heads=8, head_dim=32)
        self.assertEqual(mc.dim_per_head, 32)


# ===========================================================================
# TransformerDecoderConfig
# ===========================================================================


class TestTransformerDecoderConfig(unittest.TestCase):
    def test_hidden_size_must_be_divisible_by_heads(self):
        """hidden_size must be divisible by heads when head_dim is None."""
        with self.assertRaises(ValidationError):
            TransformerDecoderConfig(hidden_size=100, heads=8)

    def test_valid_hidden_heads(self):
        dc = TransformerDecoderConfig(hidden_size=512, heads=8)
        self.assertEqual(dc.hidden_size, 512)
        self.assertEqual(dc.heads, 8)

    def test_explicit_head_dim_skips_divisibility_check(self):
        """With explicit head_dim the divisibility check is skipped."""
        dc = TransformerDecoderConfig(hidden_size=100, heads=8, head_dim=32)
        self.assertEqual(dc.head_dim, 32)


# ===========================================================================
# Model configs
# ===========================================================================


class TestCustomModelConfig(unittest.TestCase):
    def test_architecture_is_custom(self):
        mc = CustomModelConfig()
        self.assertEqual(mc.architecture, "custom")

    def test_model_type_decoder_only(self):
        """Decoder-only model (no encoder) has ModelType.DECODER."""
        from eole.constants import ModelType

        mc = CustomModelConfig(
            decoder={"decoder_type": "transformer", "hidden_size": 512, "heads": 8}
        )
        self.assertEqual(mc.model_type, ModelType.DECODER)

    def test_model_type_encoder_only(self):
        from eole.constants import ModelType

        mc = CustomModelConfig(
            encoder={"encoder_type": "transformer", "hidden_size": 256, "heads": 4}
        )
        self.assertEqual(mc.model_type, ModelType.ENCODER)

    def test_model_type_encoder_decoder(self):
        from eole.constants import ModelType

        mc = CustomModelConfig(
            encoder={"encoder_type": "transformer", "hidden_size": 256, "heads": 4},
            decoder={"decoder_type": "transformer", "hidden_size": 256, "heads": 4},
        )
        self.assertEqual(mc.model_type, ModelType.ENCODER_DECODER)

    def test_encoder_decoder_must_match_hidden_size(self):
        with self.assertRaises(ValidationError):
            CustomModelConfig(
                encoder={"encoder_type": "transformer", "hidden_size": 256, "heads": 4},
                decoder={"decoder_type": "transformer", "hidden_size": 512, "heads": 8},
            )

    def test_share_embeddings_requires_enc_dec(self):
        """share_embeddings is only valid for encoder-decoder models."""
        with self.assertRaises(ValidationError):
            CustomModelConfig(
                share_embeddings=True,
                decoder={"decoder_type": "transformer", "hidden_size": 512, "heads": 8},
            )


class TestRnnModelConfig(unittest.TestCase):
    def test_architecture_is_rnn(self):
        mc = RnnModelConfig()
        self.assertEqual(mc.architecture, "rnn")

    def test_encoder_type_is_rnn(self):
        mc = RnnModelConfig()
        self.assertEqual(mc.encoder.encoder_type, "rnn")

    def test_decoder_type_is_rnn(self):
        mc = RnnModelConfig()
        self.assertEqual(mc.decoder.decoder_type, "rnn")


class TestTransformerModelConfig(unittest.TestCase):
    def test_architecture_is_transformer(self):
        mc = TransformerModelConfig()
        self.assertEqual(mc.architecture, "transformer")

    def test_encoder_type_is_transformer(self):
        mc = TransformerModelConfig()
        self.assertEqual(mc.encoder.encoder_type, "transformer")

    def test_decoder_type_is_transformer(self):
        mc = TransformerModelConfig()
        self.assertEqual(mc.decoder.decoder_type, "transformer")


class TestTransformerLMModelConfig(unittest.TestCase):
    def test_architecture(self):
        mc = TransformerLMModelConfig()
        self.assertEqual(mc.architecture, "transformer_lm")

    def test_encoder_is_none(self):
        """TransformerLM forces encoder=None."""
        mc = TransformerLMModelConfig()
        self.assertIsNone(mc.encoder)

    def test_decoder_is_transformer(self):
        mc = TransformerLMModelConfig()
        self.assertEqual(mc.decoder.decoder_type, "transformer")


class TestTransformerEncoderModelConfig(unittest.TestCase):
    def test_architecture(self):
        mc = TransformerEncoderModelConfig()
        self.assertEqual(mc.architecture, "transformer_encoder")

    def test_decoder_is_none(self):
        mc = TransformerEncoderModelConfig()
        self.assertIsNone(mc.decoder)


class TestBuildModelConfig(unittest.TestCase):
    """Tests for the build_model_config discriminated union factory."""

    def test_custom_architecture(self):
        mc = build_model_config({"architecture": "custom"})
        self.assertIsInstance(mc, CustomModelConfig)

    def test_rnn_architecture(self):
        mc = build_model_config({"architecture": "rnn"})
        self.assertIsInstance(mc, RnnModelConfig)

    def test_transformer_architecture(self):
        mc = build_model_config({"architecture": "transformer"})
        self.assertIsInstance(mc, TransformerModelConfig)

    def test_transformer_lm_architecture(self):
        mc = build_model_config({"architecture": "transformer_lm"})
        self.assertIsInstance(mc, TransformerLMModelConfig)

    def test_transformer_encoder_architecture(self):
        mc = build_model_config({"architecture": "transformer_encoder"})
        self.assertIsInstance(mc, TransformerEncoderModelConfig)

    def test_invalid_architecture_raises(self):
        with self.assertRaises(ValidationError):
            build_model_config({"architecture": "does_not_exist"})


# ===========================================================================
# OptimizerConfig
# ===========================================================================


class TestOptimizerConfig(unittest.TestCase):
    def test_defaults(self):
        oc = OptimizerConfig()
        self.assertEqual(oc.optim, "sgd")
        self.assertAlmostEqual(oc.learning_rate, 1.0)

    def test_invalid_optim(self):
        with self.assertRaises(ValidationError):
            OptimizerConfig(optim="unsupported_optimizer")

    def test_decay_methods(self):
        for method in ["noam", "noamwd", "cosine", "rsqrt", "none"]:
            oc = OptimizerConfig(decay_method=method)
            self.assertEqual(oc.decay_method, method)

    def test_invalid_decay_method(self):
        with self.assertRaises(ValidationError):
            OptimizerConfig(decay_method="magic_decay")


# ===========================================================================
# TrainingConfig
# ===========================================================================


class TestTrainingConfig(unittest.TestCase):
    def test_defaults(self):
        tc = TrainingConfig()
        self.assertEqual(tc.batch_size, 64)
        self.assertEqual(tc.train_steps, 100000)

    def test_dropout_and_accum_must_match_length(self):
        """dropout/attention_dropout list length must match dropout_steps."""
        with self.assertRaises(ValidationError):
            TrainingConfig(dropout=[0.1, 0.2], dropout_steps=[0])  # 2 vs 1

    def test_attention_dropout_must_match_length(self):
        with self.assertRaises(ValidationError):
            TrainingConfig(attention_dropout=[0.1, 0.2], dropout_steps=[0])

    def test_accum_count_must_match_accum_steps(self):
        with self.assertRaises(ValidationError):
            TrainingConfig(accum_count=[1, 2], accum_steps=[0])  # 2 vs 1

    def test_use_ckpting_valid_values(self):
        for val in [[], ["ffn"], ["mha"], ["lora"], ["ffn", "mha"]]:
            tc = TrainingConfig(use_ckpting=val)
            self.assertEqual(sorted(tc.use_ckpting), sorted(val))

    def test_use_ckpting_invalid_values(self):
        with self.assertRaises(ValidationError):
            TrainingConfig(use_ckpting=["invalid_layer"])

    def test_world_size_must_be_gte_gpu_ranks(self):
        with self.assertRaises(ValidationError):
            TrainingConfig(world_size=1, gpu_ranks=[0, 1])

    def test_update_vocab_requires_train_from(self):
        with self.assertRaises(ValidationError):
            TrainingConfig(update_vocab=True)

    def test_update_vocab_requires_valid_reset_optim(self):
        with self.assertRaises(ValidationError):
            TrainingConfig(update_vocab=True, train_from="some_path", reset_optim="none")

    def test_compute_dtype_int8_rejected(self):
        """int8 compute_dtype is only for inference."""
        with self.assertRaises(ValidationError):
            TrainingConfig(compute_dtype="int8")

    def test_storage_dtype_fp32_when_amp(self):
        """With AMP, storage dtype is fp32."""
        tc = TrainingConfig(compute_dtype="fp16", use_amp=True)
        self.assertEqual(tc.storage_dtype, torch.float32)

    def test_storage_dtype_fp16_without_amp(self):
        """Without AMP and fp16, storage dtype is fp16."""
        tc = TrainingConfig(compute_dtype="fp16", use_amp=False)
        self.assertEqual(tc.storage_dtype, torch.float16)

    def test_save_format_values(self):
        for fmt in ["pytorch", "safetensors"]:
            tc = TrainingConfig(save_format=fmt)
            self.assertEqual(tc.save_format, fmt)


# ===========================================================================
# DecodingConfig
# ===========================================================================


class TestDecodingConfig(unittest.TestCase):
    def test_defaults(self):
        dc = DecodingConfig()
        self.assertEqual(dc.beam_size, 5)
        self.assertEqual(dc.temperature, 1.0)
        self.assertEqual(dc.min_length, 0)

    def test_min_length_must_be_non_negative(self):
        with self.assertRaises(ValidationError):
            DecodingConfig(min_length=-1)

    def test_max_length_ratio_non_negative(self):
        with self.assertRaises(ValidationError):
            DecodingConfig(max_length_ratio=-1)

    def test_length_penalty_values(self):
        for val in ["avg", "wu", "none"]:
            dc = DecodingConfig(length_penalty=val)
            self.assertEqual(dc.length_penalty, val)

    def test_invalid_length_penalty(self):
        with self.assertRaises(ValidationError):
            DecodingConfig(length_penalty="bad_penalty")

    def test_no_speech_threshold_bounds(self):
        """no_speech_threshold is bounded [0, 1]."""
        with self.assertRaises(ValidationError):
            DecodingConfig(no_speech_threshold=1.5)

    def test_no_speech_threshold_zero(self):
        dc = DecodingConfig(no_speech_threshold=0.0)
        self.assertEqual(dc.no_speech_threshold, 0.0)


# ===========================================================================
# InferenceConfig
# ===========================================================================


class TestInferenceConfig(unittest.TestCase):
    def test_model_path_string_becomes_list(self):
        """_validate_model_path coerces a plain string to a list."""
        ic = InferenceConfig(model_path="some/model")
        self.assertEqual(ic.model_path, ["some/model"])

    def test_gold_align_requires_report_align(self):
        """gold_align without report_align should raise."""
        with self.assertRaises(ValidationError):
            InferenceConfig(model_path="dummy", gold_align=True)

    def test_storage_dtype_matches_compute_dtype_fp16(self):
        ic = InferenceConfig(model_path="dummy", compute_dtype="fp16")
        self.assertEqual(ic.storage_dtype, torch.float16)

    def test_storage_dtype_matches_compute_dtype_bf16(self):
        ic = InferenceConfig(model_path="dummy", compute_dtype="bf16")
        self.assertEqual(ic.storage_dtype, torch.bfloat16)

    def test_storage_dtype_fp32_for_fp32(self):
        ic = InferenceConfig(model_path="dummy", compute_dtype="fp32")
        self.assertEqual(ic.storage_dtype, torch.float32)

    def test_model_path_list_via_predict_config(self):
        """PredictConfig (which extends InferenceConfig) accepts a list for model_path."""
        pc = PredictConfig(model_path=["a", "b"], src="dummy")
        self.assertEqual(pc.model_path, ["a", "b"])

    def test_get_model_path_returns_first_via_predict_config(self):
        """get_model_path() returns the first entry."""
        pc = PredictConfig(model_path=["a", "b"], src="dummy")
        self.assertEqual(pc.get_model_path(), "a")

    def test_gold_align_requires_tgt_via_predict_config(self):
        """gold_align without tgt raises via PredictConfig."""
        with self.assertRaises(ValidationError):
            PredictConfig(model_path="dummy", src="dummy", gold_align=True, report_align=True)


# ===========================================================================
# BaseVocabConfig / VocabConfig / Dataset
# ===========================================================================


class TestBaseVocabConfig(unittest.TestCase):
    def test_defaults(self):
        from eole.constants import DefaultTokens

        vc = BaseVocabConfig(src_vocab="dummy")
        self.assertEqual(vc.bos_token, DefaultTokens.BOS)
        self.assertEqual(vc.eos_token, DefaultTokens.EOS)
        self.assertIsNone(vc.tgt_vocab)

    def test_share_vocab(self):
        vc = BaseVocabConfig(src_vocab="dummy", share_vocab=True)
        self.assertTrue(vc.share_vocab)


class TestVocabConfig(unittest.TestCase):
    def test_defaults(self):
        vc = VocabConfig(src_vocab="dummy")
        self.assertEqual(vc.src_vocab_size, 32768)
        self.assertEqual(vc.tgt_vocab_size, 32768)
        self.assertEqual(vc.vocab_size_multiple, 8)


class TestDataset(unittest.TestCase):
    def test_defaults(self):
        ds = Dataset()
        self.assertIsNone(ds.path_src)
        self.assertIsNone(ds.path_tgt)
        self.assertEqual(ds.weight, 1)

    def test_path_src_set(self):
        ds = Dataset(path_src="/some/path")
        self.assertEqual(ds.path_src, "/some/path")

    def test_extra_fields_rejected(self):
        with self.assertRaises(ValidationError):
            Dataset(completely_unknown=True)


# ===========================================================================
# TrainConfig
# ===========================================================================


class TestTrainConfig(unittest.TestCase):
    def test_valid_minimal(self):
        tc = TrainConfig(**_minimal_train_config())
        self.assertEqual(tc.src_vocab, "dummy")

    def test_model_required_without_train_from(self):
        """A model must be specified if not loading from checkpoint."""
        with self.assertRaises(ValidationError):
            TrainConfig(src_vocab="dummy", share_vocab=True, data={})

    def test_n_sample_requires_save_data(self):
        with self.assertRaises(ValidationError):
            TrainConfig(
                src_vocab="dummy",
                share_vocab=True,
                data={},
                model={"architecture": "custom"},
                n_sample=100,
            )

    def test_n_sample_with_save_data_passes(self):
        tc = TrainConfig(
            src_vocab="dummy",
            share_vocab=True,
            data={},
            model={"architecture": "custom"},
            n_sample=100,
            save_data="/tmp/some_data",
        )
        self.assertEqual(tc.n_sample, 100)

    def test_model_from_string_dict(self):
        """model field accepts a stringified dict."""
        tc = TrainConfig(
            src_vocab="dummy",
            share_vocab=True,
            data={},
            model="{'architecture': 'custom'}",
        )
        self.assertIsNotNone(tc.model)

    def test_training_is_optional(self):
        tc = TrainConfig(**_minimal_train_config())
        self.assertIsNotNone(tc.training)

    def test_get_defaults_rnn(self):
        defaults = TrainConfig.get_defaults("rnn")
        self.assertIn("model", defaults)

    def test_default_architecture_injected(self):
        """If model dict has no architecture key, 'custom' is injected."""
        tc = TrainConfig(
            src_vocab="dummy",
            share_vocab=True,
            data={},
            model={},
        )
        self.assertEqual(tc.model.architecture, "custom")


# ===========================================================================
# PredictConfig
# ===========================================================================


class TestPredictConfig(unittest.TestCase):
    def test_valid_minimal(self):
        pc = PredictConfig(**_minimal_predict_config())
        self.assertEqual(pc.model_path, ["dummy"])

    def test_model_path_string_to_list(self):
        pc = PredictConfig(model_path="dummy", src="dummy")
        self.assertEqual(pc.model_path, ["dummy"])

    def test_skip_empty_level_values(self):
        for level in ["silent", "warning", "error"]:
            pc = PredictConfig(**_minimal_predict_config(skip_empty_level=level))
            self.assertEqual(pc.skip_empty_level, level)

    def test_invalid_skip_empty_level(self):
        with self.assertRaises(ValidationError):
            PredictConfig(**_minimal_predict_config(skip_empty_level="verbose"))


# ===========================================================================
# BuildVocabConfig
# ===========================================================================


class TestBuildVocabConfig(unittest.TestCase):
    def test_learn_subwords_requires_onmt_tokenize(self):
        """learn_subwords=True requires onmt_tokenize in transforms."""
        with self.assertRaises(ValidationError):
            BuildVocabConfig(
                src_vocab="dummy",
                share_vocab=True,
                data={},
                transforms=[],
                learn_subwords=True,
            )

    def test_learn_subwords_with_onmt_tokenize_passes(self):
        bvc = BuildVocabConfig(
            src_vocab="dummy",
            share_vocab=True,
            data={},
            transforms=["onmt_tokenize"],
            learn_subwords=True,
        )
        self.assertTrue(bvc.learn_subwords)

    def test_defaults(self):
        bvc = BuildVocabConfig(src_vocab="dummy", share_vocab=True, data={})
        self.assertEqual(bvc.n_sample, 5000)
        self.assertFalse(bvc.learn_subwords)


# ===========================================================================
# Utility helpers: recursive_model_fields_set, reorder_fields
# ===========================================================================


class TestRecursiveModelFieldsSet(unittest.TestCase):
    def test_flat_model_returns_set_fields(self):
        mc = MiscConfig(seed=123)
        result = recursive_model_fields_set(mc)
        self.assertIn("seed", result)
        self.assertEqual(result["seed"], 123)

    def test_nested_model_recurses(self):
        tc = TrainConfig(**_minimal_train_config(seed=7))
        result = recursive_model_fields_set(tc)
        # seed was explicitly set
        self.assertEqual(result.get("seed"), 7)

    def test_torch_dtype_becomes_string(self):
        """torch.dtype values are serialized as strings (not directly JSON-safe)."""
        rc = RunningConfig(model_path="dummy", compute_dtype=torch.float16)
        result = recursive_model_fields_set(rc)
        # compute_dtype should be serialized as a string
        dtype_val = result.get("compute_dtype")
        self.assertIsInstance(dtype_val, str)

    def test_dict_input(self):
        data = {"a": 1, "b": {"c": 2}}
        result = recursive_model_fields_set(data)
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"]["c"], 2)


class TestReorderFields(unittest.TestCase):
    def test_non_nested_before_nested(self):
        """Non-nested fields should appear before nested ones."""
        fields = {
            "nested": {"x": 1, "y": 2},
            "flat_a": 10,
            "flat_b": 20,
        }
        result = reorder_fields(fields)
        keys = list(result.keys())
        # flat_a and flat_b must come before 'nested'
        self.assertLess(keys.index("flat_a"), keys.index("nested"))
        self.assertLess(keys.index("flat_b"), keys.index("nested"))

    def test_empty_dict(self):
        self.assertEqual(reorder_fields({}), {})

    def test_all_flat(self):
        fields = {"a": 1, "b": 2}
        result = reorder_fields(fields)
        self.assertEqual(set(result.keys()), {"a", "b"})

    def test_nested_preserved(self):
        fields = {"outer": {"inner": {"deep": 42}}}
        result = reorder_fields(fields)
        self.assertEqual(result["outer"]["inner"]["deep"], 42)
