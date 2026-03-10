#!/usr/bin/env python
"""Convert a GGUF model file to EOLE format.

GGUF (llama.cpp) models often store weights in quantized formats (Q4_K, Q5_K,
Q8_0, …).  This converter follows the same philosophy as the AutoRound/GPTQ
backend: quantized weights are kept in their compact binary representation and
stored as ``uint8`` tensors in the output safetensors shard.  At inference time
:class:`~eole.modules.gguf_linear.GGUFLinear` dequantizes them on-the-fly,
achieving the same on-disk storage savings without requiring a separate
dequantization step during conversion.

Float-typed tensors (F16, BF16, F32) are stored as-is, cast to the target dtype
requested via ``--dtype``.

Requires:
  pip install gguf safetensors pyonmttok

Usage example::

    eole convert GGUF \\
        --gguf_path  /path/to/model-Q4_K_M.gguf \\
        --output     /path/to/output_dir \\
        --dtype      fp16
"""

# Standard Library Imports
import json
import os
import re
from typing import Optional

# Third-Party Library Imports
import torch

# Eole Imports
from eole.bin import BaseBin, register_bin
from eole.config import recursive_model_fields_set
from eole.config.run import TrainConfig
from eole.config.training import TrainingConfig
from eole.constants import DefaultTokens, TORCH_DTYPES, PositionEncodingType
from eole.inputters.inputter import vocabs_to_dict

try:
    import pyonmttok
except ImportError:
    pyonmttok = None

try:
    from safetensors.torch import save_file as safetensors_save_file
except ImportError:
    safetensors_save_file = None


# ---------------------------------------------------------------------------
# Float types – linear layers with these types stay as nn.Linear (no GGUFLinear)
# ---------------------------------------------------------------------------

_FLOAT_TYPE_NAMES = frozenset({"F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64"})


def _is_float_type(tensor_type_name: str) -> bool:
    return tensor_type_name in _FLOAT_TYPE_NAMES


# ---------------------------------------------------------------------------
# GGUF tensor-name → EOLE tensor-name mapping
# ---------------------------------------------------------------------------

# Global tensors (not inside a transformer block)
_GLOBAL_MAP: dict[str, Optional[str]] = {
    "token_embd.weight": "tgt_emb.embeddings.weight",
    "output_norm.weight": "decoder.layer_norm.weight",
    "output_norm.bias": "decoder.layer_norm.bias",
    "output.weight": "generator.weight",
    "output.bias": "generator.bias",
    # RoPE frequency cache – EOLE recomputes this at runtime
    "rope_freqs.weight": None,
    "rope_factors_long.weight": None,
    "rope_factors_short.weight": None,
}

# Per-block suffix → EOLE decoder-layer suffix.
# A value of None means "known tensor but not yet modelled in EOLE – skip silently".
_BLOCK_MAP: dict[str, Optional[str]] = {
    # Norms
    "attn_norm.weight": "input_layernorm.weight",
    "attn_norm.bias": "input_layernorm.bias",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.bias": "post_attention_layernorm.bias",
    # Per-layer norms for Gemma2/3
    "pre_feedforward_layernorm.weight": "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight": "post_feedforward_layernorm.weight",
    # Q-norm / K-norm (Qwen3/Qwen3.5 etc.)
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    # Attention projections (separate Q/K/V)
    "attn_q.weight": "self_attn.linear_query.weight",
    "attn_q.bias": "self_attn.linear_query.bias",
    "attn_k.weight": "self_attn.linear_keys.weight",
    "attn_k.bias": "self_attn.linear_keys.bias",
    "attn_v.weight": "self_attn.linear_values.weight",
    "attn_v.bias": "self_attn.linear_values.bias",
    "attn_output.weight": "self_attn.final_linear.weight",
    "attn_output.bias": "self_attn.final_linear.bias",
    # Combined QKV projection (Phi-2, Phi-3 style)
    "attn_qkv.weight": "self_attn.qkv_proj.weight",
    "attn_qkv.bias": "self_attn.qkv_proj.bias",
    # Feed-forward (SwiGLU: gate + up + down)
    "ffn_gate.weight": "mlp.gate_up_proj.weight",
    "ffn_gate.bias": "mlp.gate_up_proj.bias",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_up.bias": "mlp.up_proj.bias",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_down.bias": "mlp.down_proj.bias",
    # Gate + Up fused (Phi-3 style)
    "ffn_gate_up.weight": "mlp.gate_up_proj.weight",
    "ffn_gate_up.bias": "mlp.gate_up_proj.bias",
    # MoE router gate
    "ffn_gate_inp.weight": "mlp.gate.weight",
    # MoE per-expert weights (stacked tensors)
    "ffn_gate_exps.weight": "mlp.experts_gate_up.weight",
    "ffn_up_exps.weight": "mlp.experts_up.weight",
    "ffn_down_exps.weight": "mlp.experts_down.weight",
    # ------------------------------------------------------------------
    # Known tensors that have no direct EOLE equivalent yet – skip silently.
    # ------------------------------------------------------------------
    # Post-attention output norm (Qwen3.5 and similar models)
    "post_attention_norm.weight": None,
    "post_attention_norm.bias": None,
    # Attention output gate (Qwen3.5 gated-attention mechanism)
    "attn_gate.weight": None,
    "attn_gate.bias": None,
    # Mamba / SSM layer weights (hybrid models: Qwen3.5, Jamba, …)
    "ssm_a": None,
    "ssm_a.weight": None,
    "ssm_d": None,
    "ssm_d.weight": None,
    "ssm_conv1d.weight": None,
    "ssm_conv1d.bias": None,
    "ssm_in_proj.weight": None,
    "ssm_out_proj.weight": None,
    "ssm_dt.weight": None,
    "ssm_dt.bias": None,
    "ssm_norm.weight": None,
    "ssm_alpha.weight": None,
    "ssm_beta.weight": None,
}


def _gguf_to_eole_name(gguf_name: str) -> Optional[str]:
    """Map a GGUF tensor name to the EOLE tensor name.

    Returns ``None`` for tensors that should be skipped (e.g., RoPE cache,
    or architecture-specific tensors not yet modelled in EOLE).
    Returns the empty string ``""`` for unrecognised names (caller should warn).
    """
    if gguf_name in _GLOBAL_MAP:
        return _GLOBAL_MAP[gguf_name]

    m = re.fullmatch(r"blk\.(\d+)\.(.*)", gguf_name)
    if m:
        block_idx = m.group(1)
        suffix = m.group(2)
        if suffix not in _BLOCK_MAP:
            return ""  # unrecognised
        eole_suffix = _BLOCK_MAP[suffix]
        if eole_suffix is None:
            return None  # known but not yet supported in EOLE, skip silently
        return f"decoder.transformer_layers.{block_idx}.{eole_suffix}"

    return ""  # unrecognised


# ---------------------------------------------------------------------------
# GGUF metadata reader
# ---------------------------------------------------------------------------


class GGUFMetadata:
    """Thin wrapper around :class:`gguf.GGUFReader` exposing typed metadata."""

    def __init__(self, path: str):
        try:
            from gguf import GGUFReader
        except ImportError:
            raise ImportError("Install 'gguf': pip install gguf")
        self._reader = GGUFReader(path)
        self._fields = {f.name: f for f in self._reader.fields.values()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_field(self, key: str):
        return self._fields.get(key)

    def _scalar(self, key: str, default=None):
        """Return the Python scalar value of a GGUF metadata field."""
        field = self._get_field(key)
        if field is None or not field.data:
            return default
        # field.data is a list of part indices; the last one holds the value
        raw = field.parts[field.data[-1]]
        val = raw.tolist()
        if isinstance(val, list):
            return val[0] if len(val) == 1 else val
        return val

    def _str(self, key: str, default: str = "") -> str:
        """Return a string metadata field decoded from UTF-8 bytes."""
        field = self._get_field(key)
        if field is None or not field.data:
            return default
        raw = field.parts[field.data[-1]]
        return bytes(raw).decode("utf-8", errors="replace")

    def _str_list(self, key: str) -> list[str]:
        """Return a list-of-strings metadata field."""
        field = self._get_field(key)
        if field is None:
            return []
        return [bytes(field.parts[idx]).decode("utf-8", errors="replace") for idx in field.data]

    def _a(self, template: str) -> str:
        """Substitute ``{arch}`` in a metadata key template."""
        return template.format(arch=self.arch)

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    @property
    def arch(self) -> str:
        return self._str("general.architecture", "llama")

    @property
    def model_name(self) -> str:
        return self._str("general.name", "")

    # ------------------------------------------------------------------
    # Model hyperparameters
    # ------------------------------------------------------------------

    @property
    def block_count(self) -> int:
        return int(self._scalar(self._a("{arch}.block_count"), 0))

    @property
    def context_length(self) -> int:
        return int(self._scalar(self._a("{arch}.context_length"), 2048))

    @property
    def embedding_length(self) -> int:
        return int(self._scalar(self._a("{arch}.embedding_length"), 512))

    @property
    def feed_forward_length(self) -> int:
        return int(self._scalar(self._a("{arch}.feed_forward_length"), 0))

    @property
    def expert_feed_forward_length(self) -> int:
        return int(self._scalar(self._a("{arch}.expert_feed_forward_length"), 0))

    @property
    def head_count(self) -> int:
        return int(self._scalar(self._a("{arch}.attention.head_count"), 8))

    @property
    def head_count_kv(self) -> Optional[int]:
        v = self._scalar(self._a("{arch}.attention.head_count_kv"), None)
        return int(v) if v is not None else None

    @property
    def head_dim(self) -> Optional[int]:
        v = self._scalar(self._a("{arch}.attention.key_length"), None)
        return int(v) if v is not None else None

    @property
    def norm_rms_eps(self) -> float:
        return float(self._scalar(self._a("{arch}.attention.layer_norm_rms_epsilon"), 1e-5))

    @property
    def norm_eps(self) -> float:
        return float(self._scalar(self._a("{arch}.attention.layer_norm_epsilon"), 1e-5))

    @property
    def vocab_size(self) -> int:
        v = self._scalar(self._a("{arch}.vocab_size"), None)
        if v is not None:
            return int(v)
        return max(len(self.tokens), 32000)

    @property
    def rope_freq_base(self) -> float:
        return float(self._scalar(self._a("{arch}.rope.freq_base"), 10000.0))

    @property
    def rope_dim_count(self) -> Optional[int]:
        v = self._scalar(self._a("{arch}.rope.dimension_count"), None)
        return int(v) if v is not None else None

    @property
    def expert_count(self) -> int:
        return int(self._scalar(self._a("{arch}.expert_count"), 0))

    @property
    def expert_used_count(self) -> int:
        return int(self._scalar(self._a("{arch}.expert_used_count"), 0))

    @property
    def use_parallel_residual(self) -> bool:
        v = self._scalar(self._a("{arch}.use_parallel_residual"), None)
        return bool(v) if v is not None else False

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    @property
    def tokenizer_model(self) -> str:
        return self._str("tokenizer.ggml.model", "llama")

    @property
    def tokens(self) -> list[str]:
        return self._str_list("tokenizer.ggml.tokens")

    @property
    def token_types(self) -> list:
        field = self._get_field("tokenizer.ggml.token_type")
        if field is None:
            return []
        return field.parts[field.data[-1]].tolist() if field.data else []

    @property
    def merges(self) -> list[str]:
        return self._str_list("tokenizer.ggml.merges")

    @property
    def bos_token_id(self) -> Optional[int]:
        v = self._scalar("tokenizer.ggml.bos_token_id", None)
        return int(v) if v is not None else None

    @property
    def eos_token_id(self) -> Optional[int]:
        v = self._scalar("tokenizer.ggml.eos_token_id", None)
        return int(v) if v is not None else None

    @property
    def unk_token_id(self) -> Optional[int]:
        v = self._scalar("tokenizer.ggml.unknown_token_id", None)
        return int(v) if v is not None else None

    @property
    def pad_token_id(self) -> Optional[int]:
        v = self._scalar("tokenizer.ggml.padding_token_id", None)
        return int(v) if v is not None else None

    @property
    def add_bos_token(self) -> bool:
        v = self._scalar("tokenizer.ggml.add_bos_token", None)
        return bool(v) if v is not None else True

    @property
    def chat_template(self) -> Optional[str]:
        raw = self._str("tokenizer.chat_template", "")
        return raw if raw else None

    @property
    def hf_tokenizer_json(self) -> Optional[str]:
        """Embedded HuggingFace tokenizer.json (if present in the GGUF)."""
        raw = self._str("tokenizer.huggingface.json", "")
        return raw if raw else None

    # ------------------------------------------------------------------
    # Tensors
    # ------------------------------------------------------------------

    @property
    def tensors(self):
        return self._reader.tensors


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------

_RMS_NORM_ARCHS = frozenset(
    "llama mistral mixtral qwen2 qwen2moe qwen3 qwen3moe qwen35 phi3 deepseek2 "
    "falcon gemma gemma2 gemma3 llama4 deci grok starcoder2".split()
)
_SWIGLU_ARCHS = frozenset(
    "llama mistral mixtral qwen2 qwen2moe qwen3 qwen3moe qwen35 phi3 deepseek2 "
    "gemma gemma2 gemma3 llama4 deci grok starcoder2".split()
)


def build_model_config(meta: GGUFMetadata) -> dict:
    """Build an EOLE ``model_config`` dict from GGUF metadata."""
    arch = meta.arch.lower()
    hidden_size = meta.embedding_length
    layers = meta.block_count
    heads = meta.head_count
    heads_kv = meta.head_count_kv
    head_dim = meta.head_dim
    ff_length = meta.feed_forward_length
    rope_base = meta.rope_freq_base
    rope_dim = meta.rope_dim_count
    num_experts = meta.expert_count
    num_experts_per_tok = meta.expert_used_count
    expert_ff = meta.expert_feed_forward_length

    use_rms = arch in _RMS_NORM_ARCHS
    norm_eps = meta.norm_rms_eps if use_rms else meta.norm_eps
    layer_norm = "rms" if use_rms else "standard"
    mlp_activation_fn = "gated-silu" if arch in _SWIGLU_ARCHS else "gelu"

    model_config: dict = {
        "layers": layers,
        "hidden_size": hidden_size,
        "heads": heads,
        "transformer_ff": ff_length,
        "mlp_activation_fn": mlp_activation_fn,
        "layer_norm": layer_norm,
        "norm_eps": norm_eps,
        "add_qkvbias": False,
        "add_final_linear_bias": False,
        "add_ffnbias": False,
        "shared_layer_norm": False,
        "generator_bias": False,
        "rope_config": {
            "rotary_interleave": False,
            "rotary_theta": rope_base,
        },
        "embeddings": {
            "position_encoding_type": PositionEncodingType.Rotary,
            "n_positions": 0,
        },
    }

    if heads_kv is not None:
        model_config["heads_kv"] = heads_kv

    if head_dim is not None:
        model_config["head_dim"] = head_dim

    # Rotary dimension
    effective_head_dim = head_dim if head_dim is not None else hidden_size // heads
    if rope_dim is not None:
        model_config["rope_config"]["rotary_dim"] = rope_dim
    else:
        model_config["rope_config"]["rotary_dim"] = effective_head_dim

    if num_experts > 0:
        model_config["num_experts"] = num_experts
        model_config["num_experts_per_tok"] = num_experts_per_tok or 2
        if expert_ff:
            model_config["moe_transformer_ff"] = expert_ff

    # Architecture-specific overrides
    if arch in ("phi2",):
        model_config.update(
            parallel_residual=True,
            shared_layer_norm=True,
            add_qkvbias=True,
            add_final_linear_bias=True,
            add_ffnbias=True,
        )
    if arch in ("qwen2", "qwen3", "qwen35"):
        model_config.update(add_qkvbias=True, add_final_linear_bias=False)
    if meta.use_parallel_residual:
        model_config["parallel_residual"] = True

    if arch in ("gemma2", "gemma3"):
        model_config["ffn_layernorm"] = True

    return model_config


# ---------------------------------------------------------------------------
# Tensor-name → EOLE leaf module name (for quant_layers detection)
# ---------------------------------------------------------------------------

# These are the leaf module names (last path component) that correspond to
# linear layers inside a transformer block.
_QUANTIZABLE_SUFFIXES = frozenset(
    "attn_q attn_k attn_v attn_output attn_qkv "
    "ffn_gate ffn_up ffn_down ffn_gate_up "
    "ffn_gate_exps ffn_up_exps ffn_down_exps".split()
)

# Map suffix → EOLE leaf module name  (only the first component of the mapped path)
_SUFFIX_TO_EOLE_LEAF: dict[str, str] = {
    "attn_q": "linear_query",
    "attn_k": "linear_keys",
    "attn_v": "linear_values",
    "attn_output": "final_linear",
    "attn_qkv": "qkv_proj",
    "ffn_gate": "gate_up_proj",
    "ffn_up": "up_proj",
    "ffn_down": "down_proj",
    "ffn_gate_up": "gate_up_proj",
    "ffn_gate_exps": "experts_gate_up",
    "ffn_up_exps": "experts_up",
    "ffn_down_exps": "experts_down",
    "ffn_gate_inp": "gate",
}


def _detect_quant_layers(tensors) -> tuple[list[str], str]:
    """Scan tensors and return (quant_layers, dominant_qtype_name).

    ``quant_layers`` contains the EOLE leaf-module names of all layers that
    carry at least one quantised tensor.
    ``dominant_qtype_name`` is the name of the most-common quantisation type
    across those tensors (e.g. ``"Q4_K"``).
    """
    from collections import Counter

    quant_leaf_set: set[str] = set()
    qtype_counter: Counter = Counter()

    for tensor in tensors:
        m = re.fullmatch(r"blk\.\d+\.(.*?)(?:\.weight|\.bias)?$", tensor.name)
        if not m:
            continue
        raw_suffix = m.group(1)
        # Strip any ".weight" / ".bias" that might still be in raw_suffix
        base_suffix = raw_suffix.rsplit(".", 1)[0] if "." in raw_suffix else raw_suffix
        if base_suffix not in _QUANTIZABLE_SUFFIXES:
            continue
        if _is_float_type(tensor.tensor_type.name):
            continue
        leaf = _SUFFIX_TO_EOLE_LEAF.get(base_suffix)
        if leaf:
            quant_leaf_set.add(leaf)
        qtype_counter[tensor.tensor_type.name] += 1

    dominant_qtype = qtype_counter.most_common(1)[0][0] if qtype_counter else ""
    return sorted(quant_leaf_set), dominant_qtype


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

_TOKEN_TYPE_CONTROL = 3  # ggml token type: control (BOS/EOS/…)


def _extract_vocab_from_gguf(
    meta: GGUFMetadata, vocabs: dict, output_dir: str
) -> tuple:
    """Extract vocab from GGUF and write tokenizer artefacts.

    Returns ``(src_vocab, tokenizer_basename, gpt2_pretok)``.
    """
    if pyonmttok is None:
        raise ImportError("Install pyonmttok: pip install pyonmttok")

    tokens = meta.tokens
    merges = meta.merges
    if not tokens:
        raise ValueError("GGUF file contains no tokenizer.ggml.tokens")

    def _tok(tid):
        return tokens[tid] if tid is not None and 0 <= tid < len(tokens) else None

    bos_str = _tok(meta.bos_token_id)
    eos_str = _tok(meta.eos_token_id)
    unk_str = _tok(meta.unk_token_id)
    pad_str = _tok(meta.pad_token_id)

    if bos_str:
        vocabs["specials"]["bos_token"] = bos_str
    if eos_str:
        vocabs["specials"]["eos_token"] = eos_str
    if unk_str:
        vocabs["specials"]["unk_token"] = unk_str
    if pad_str:
        vocabs["specials"]["pad_token"] = pad_str
    else:
        vocabs["specials"].setdefault("pad_token", DefaultTokens.PAD)

    gpt2_pretok = bool(merges)
    if merges:
        tokenizer_basename = "bpe.model"
        with open(os.path.join(output_dir, tokenizer_basename), "w", encoding="utf-8") as f:
            f.write("v3;false;false;false;Ġ;Ġ\n")
            for merge in merges:
                f.write(merge + "\n")
    else:
        tokenizer_basename = "sentencepiece.model"
        sp_path = os.path.join(output_dir, tokenizer_basename)
        if not os.path.exists(sp_path):
            open(sp_path, "wb").close()  # placeholder

    vocab_list = list(tokens)
    declared = meta.vocab_size
    while len(vocab_list) < declared:
        vocab_list.append(f"{DefaultTokens.VOCAB_PAD}{len(vocab_list)}")

    if gpt2_pretok:
        vocab_list = [DefaultTokens.PAD if t == "Ā" else t for t in vocab_list]
        if DefaultTokens.PAD in vocab_list:
            vocabs["specials"]["pad_token"] = DefaultTokens.PAD

    src_vocab = pyonmttok.build_vocab_from_tokens(vocab_list)
    return src_vocab, tokenizer_basename, gpt2_pretok


# ---------------------------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------------------------

def _tensor_to_torch(tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Convert a single GGUF ReaderTensor to a torch.Tensor.

    For quantised types (Q4_K, Q8_0, …) the raw ``uint8`` block data is
    stored as-is.  The companion ``gguf_qtype`` tensor carries the
    :class:`gguf.GGMLQuantizationType` integer value so that
    :class:`~eole.modules.gguf_linear.GGUFLinear` can dequantize correctly
    at inference time.

    For float types (F16, BF16, F32) the tensor is cast to *target_dtype*.

    Returns ``(tensor, qtype_tensor_or_None)``.
    ``qtype_tensor_or_None`` is a ``torch.int32`` scalar when the tensor is
    quantised; ``None`` for float tensors.
    """
    import numpy as np

    tname = tensor.tensor_type.name
    data = tensor.data  # numpy ndarray

    if _is_float_type(tname):
        t = torch.from_numpy(data.copy())
        if t.is_floating_point():
            t = t.to(target_dtype)
        return t, None
    else:
        # Quantized: keep raw uint8 bytes; data.shape is already
        # (out_features, bytes_per_row) for 2-D weights.
        raw = data.copy()
        if raw.dtype != np.uint8:
            raw = raw.view(np.uint8)
        t = torch.from_numpy(raw)  # dtype=torch.uint8
        qtype_t = torch.tensor([tensor.tensor_type.value], dtype=torch.int32)
        return t, qtype_t


# ---------------------------------------------------------------------------
# Shard builder
# ---------------------------------------------------------------------------

def build_safetensors(
    meta: GGUFMetadata,
    output_dir: str,
    target_dtype: torch.dtype,
) -> tuple[dict, list[str]]:
    """Convert all tensors from the GGUF and write ``model.00.safetensors``.

    Returns ``(written_tensor_dict, quant_layers)`` where *quant_layers* is the
    list of EOLE leaf-module names (e.g. ``"linear_query"``) that carry
    quantised weights and therefore need to be replaced with
    :class:`~eole.modules.gguf_linear.GGUFLinear`.
    """
    if safetensors_save_file is None:
        raise ImportError("Install safetensors: pip install safetensors")

    written: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    quant_layers, dominant_qtype = _detect_quant_layers(meta.tensors)

    for tensor in meta.tensors:
        gguf_name = tensor.name
        eole_name = _gguf_to_eole_name(gguf_name)

        if eole_name is None:
            # Explicitly skipped (e.g. rope_freqs)
            continue
        if eole_name == "":
            skipped.append(gguf_name)
            continue

        t, qtype_t = _tensor_to_torch(tensor, target_dtype)
        written[eole_name] = t

        # For quantised linear weights, also store the per-tensor qtype so
        # GGUFLinear.forward() knows how to dequantize.
        if qtype_t is not None:
            # Derive the companion gguf_qtype key: replace ".weight" → ".gguf_qtype"
            qtype_key = eole_name.rsplit(".weight", 1)[0] + ".gguf_qtype"
            written[qtype_key] = qtype_t

    shard_path = os.path.join(output_dir, "model.00.safetensors")
    safetensors_save_file(written, shard_path)

    n_quant = sum(1 for k in written if k.endswith(".gguf_qtype"))
    n_float = sum(1 for k in written if k.endswith(".weight") and not k.replace(".weight", ".gguf_qtype") in written)
    print(f"Wrote {len(written)} tensors to {shard_path}")
    print(f"  {n_quant} quantised weight tensors (uint8)  +  {n_float} float weight tensors")
    if skipped:
        print(f"  Skipped {len(skipped)} unrecognised tensor(s): {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    return written, quant_layers


# ---------------------------------------------------------------------------
# EOLE config builder
# ---------------------------------------------------------------------------

def build_eole_config(
    meta: GGUFMetadata,
    vocabs: dict,
    src_vocab,
    model_config_dict: dict,
    training_config_dict: dict,
    transforms: list,
    transforms_configs: dict,
    optional_eos: list,
) -> dict:
    """Assemble and return the EOLE ``config.json`` dictionary."""
    from eole.config.models import TransformerLMModelConfig

    config = TrainConfig(
        data=None,
        skip_empty_level="silent",
        save_data=None,
        n_sample=0,
        src_vocab=None,
        tgt_vocab=None,
        share_vocab=True,
        src_vocab_size=meta.vocab_size,
        tgt_vocab_size=meta.vocab_size,
        vocab_size_multiple=8,
        decoder_start_token=vocabs.get("decoder_start_token", ""),
        **{k: v for k, v in vocabs.get("specials", {}).items()},
        transforms=transforms,
        transforms_configs=transforms_configs,
        model=TransformerLMModelConfig(**model_config_dict),
        training=TrainingConfig(**{
            k: v for k, v in training_config_dict.items()
            if k not in ("compute_dtype",)
        }),
    )

    config_dict = recursive_model_fields_set(config)
    config_dict["inference"] = {"optional_eos": optional_eos}
    if meta.chat_template:
        config_dict["inference"]["chat_template"] = meta.chat_template
    return config_dict


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@register_bin(name="GGUF")
class GGUFConverter(BaseBin):
    """Convert a GGUF model file to EOLE format.

    Quantised weights are kept in their native compact binary form and stored as
    ``uint8`` tensors in the output safetensors shard – no loss of quantisation
    precision.  At inference time :class:`~eole.modules.gguf_linear.GGUFLinear`
    dequantizes them on-the-fly exactly as the GPTQ/Marlin backend does for
    its packed int4 weights.
    """

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--gguf_path",
            type=str,
            required=True,
            help="Path to the input GGUF file.",
        )
        parser.add_argument(
            "--output",
            type=str,
            required=True,
            help="Path to the output directory (created if absent).",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default="fp16",
            choices=list(TORCH_DTYPES.keys()),
            help="Dtype to use for non-quantised (float) tensors (default: fp16).",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default="hf",
            choices=["hf", "onmt"],
            help="Tokenizer transform to embed in the EOLE config (default: hf).",
        )
        parser.add_argument(
            "--hf_tokenizer",
            type=str,
            default=None,
            help=(
                "Optional path to a HuggingFace tokenizer.json file (or directory "
                "containing one).  When provided the HF tokenizer is used directly "
                "instead of extracting it from the GGUF metadata."
            ),
        )

    @classmethod
    def run(cls, args):
        if safetensors_save_file is None:
            raise ImportError("Install safetensors: pip install safetensors")
        try:
            import gguf  # noqa: F401
        except ImportError:
            raise ImportError("Install gguf: pip install gguf")

        os.makedirs(args.output, exist_ok=True)
        target_dtype = TORCH_DTYPES.get(args.dtype, torch.float16)

        print(f"Reading GGUF metadata from {args.gguf_path} …")
        meta = GGUFMetadata(args.gguf_path)
        print(f"  Architecture : {meta.arch}")
        print(f"  Layers       : {meta.block_count}")
        print(f"  Hidden size  : {meta.embedding_length}")
        print(f"  Heads        : {meta.head_count}")
        if meta.head_count_kv:
            print(f"  KV heads     : {meta.head_count_kv}")
        if meta.expert_count:
            print(f"  Experts      : {meta.expert_count} (used: {meta.expert_used_count})")

        # Detect whether the output.weight tensor is missing (tied embeddings)
        tensor_names = {t.name for t in meta.tensors}
        share_decoder_embeddings = "output.weight" not in tensor_names
        print(f"  Shared emb   : {share_decoder_embeddings}")

        # ------------------------------------------------------------------
        # Tokenizer
        # ------------------------------------------------------------------
        vocabs: dict = {"specials": {}}
        optional_eos: list = []
        transforms_configs: dict = {}
        transforms: list = []

        hf_tok_path = args.hf_tokenizer
        if hf_tok_path is not None and os.path.isdir(hf_tok_path):
            candidate = os.path.join(hf_tok_path, "tokenizer.json")
            if os.path.exists(candidate):
                hf_tok_path = candidate

        # Helper: build vocab from HF tokenizer.json dict
        def _vocab_from_hf_tok(tok_data: dict, vocabs: dict) -> tuple:
            vlist = list(tok_data.get("model", {}).get("vocab", {}).keys())
            if isinstance(vlist, dict):
                vlist = list(vlist.keys())
            declared = meta.vocab_size
            while len(vlist) < declared:
                vlist.append(f"{DefaultTokens.VOCAB_PAD}{len(vlist)}")
            for tok in tok_data.get("added_tokens", []):
                idx = tok["id"]
                if 0 <= idx < len(vlist):
                    vlist[idx] = tok["content"]
            for field in ["bos_token", "eos_token", "unk_token", "pad_token"]:
                val = tok_data.get(field)
                if isinstance(val, str):
                    vocabs["specials"][field] = val
                elif isinstance(val, dict):
                    vocabs["specials"][field] = val.get("content", "")
            src_v = pyonmttok.build_vocab_from_tokens(vlist) if pyonmttok else None
            return src_v, "tokenizer.json"

        if hf_tok_path is not None and os.path.isfile(hf_tok_path):
            import shutil

            dest_tok = os.path.join(args.output, "tokenizer.json")
            if os.path.normpath(hf_tok_path) != os.path.normpath(dest_tok):
                shutil.copy2(hf_tok_path, dest_tok)
            with open(dest_tok, encoding="utf-8") as fh:
                tok_data = json.load(fh)
            src_vocab, tok_basename = _vocab_from_hf_tok(tok_data, vocabs)
            transforms = ["huggingface_tokenize"]
            transforms_configs["huggingface_tokenize"] = {"path": dest_tok}

        elif meta.hf_tokenizer_json is not None:
            dest_tok = os.path.join(args.output, "tokenizer.json")
            tok_data = json.loads(meta.hf_tokenizer_json)
            with open(dest_tok, "w", encoding="utf-8") as fh:
                json.dump(tok_data, fh, indent=2, ensure_ascii=False)
            print(f"  Extracted embedded tokenizer.json → {dest_tok}")
            src_vocab, tok_basename = _vocab_from_hf_tok(tok_data, vocabs)
            transforms = ["huggingface_tokenize"]
            transforms_configs["huggingface_tokenize"] = {"path": dest_tok}

        else:
            print("  Extracting tokenizer from GGUF …")
            src_vocab, tok_basename, gpt2_pretok = _extract_vocab_from_gguf(meta, vocabs, args.output)
            if args.tokenizer == "hf":
                tok_path = os.path.join(args.output, "tokenizer.json")
                if os.path.exists(tok_path):
                    transforms = ["huggingface_tokenize"]
                    transforms_configs["huggingface_tokenize"] = {"path": tok_path}
                else:
                    subword_type = "bpe" if gpt2_pretok else "sentencepiece"
                    transforms = ["onmt_tokenize", "filtertoolong"]
                    transforms_configs = {
                        "filtertoolong": {"src_seq_length": 512, "tgt_seq_length": 512},
                        "onmt_tokenize": {
                            "src_subword_type": subword_type,
                            "src_subword_model": os.path.join("${MODEL_PATH}", tok_basename),
                            "gpt2_pretok": gpt2_pretok,
                        },
                    }
            else:
                subword_type = "bpe" if gpt2_pretok else "sentencepiece"
                transforms = ["onmt_tokenize", "filtertoolong"]
                transforms_configs = {
                    "filtertoolong": {"src_seq_length": 512, "tgt_seq_length": 512},
                    "onmt_tokenize": {
                        "src_subword_type": subword_type,
                        "src_subword_model": os.path.join("${MODEL_PATH}", tok_basename),
                        "gpt2_pretok": gpt2_pretok,
                    },
                }

        # Optional EOS tokens
        eos_id = meta.eos_token_id
        tokens = meta.tokens
        for tid, ttype in enumerate(meta.token_types):
            if ttype == _TOKEN_TYPE_CONTROL and tid != eos_id and tid < len(tokens):
                optional_eos.append(tokens[tid])

        # Decoder start token
        if meta.add_bos_token and "bos_token" in vocabs["specials"]:
            vocabs["decoder_start_token"] = vocabs["specials"]["bos_token"]
        else:
            vocabs["decoder_start_token"] = ""

        # Write vocab files
        if src_vocab is not None:
            vocabs["src"] = src_vocab
            vocabs["tgt"] = src_vocab
            vocab_dict = vocabs_to_dict(vocabs)
            with open(os.path.join(args.output, "vocab.json"), "w", encoding="utf-8") as fh:
                json.dump(vocab_dict, fh, indent=2, ensure_ascii=False)
            with open(os.path.join(args.output, "vocab.txt"), "w", encoding="utf-8") as fh:
                for tok in vocab_dict["src"]:
                    fh.write(tok + "\n")

        # ------------------------------------------------------------------
        # Convert tensors (keep quantised as uint8, cast floats to target_dtype)
        # ------------------------------------------------------------------
        print("Converting tensors …")
        written, quant_layers = build_safetensors(meta, args.output, target_dtype)

        # ------------------------------------------------------------------
        # Build EOLE config
        # ------------------------------------------------------------------
        model_config_dict = build_model_config(meta)
        model_config_dict["share_decoder_embeddings"] = share_decoder_embeddings

        quant_layers_all = [
            "gate_up_proj",
            "down_proj",
            "up_proj",
            "linear_values",
            "linear_query",
            "linear_keys",
            "final_linear",
        ]

        training_config_dict: dict = {
            "quant_type": "gguf" if quant_layers else "",
            "quant_layers": quant_layers_all if quant_layers else [],
            "w_bit": 0,
            "group_size": 0,
            "compute_dtype": args.dtype,
        }

        print("Building config.json …")
        config_dict = build_eole_config(
            meta=meta,
            vocabs=vocabs,
            src_vocab=src_vocab,
            model_config_dict=model_config_dict,
            training_config_dict=training_config_dict,
            transforms=transforms,
            transforms_configs=transforms_configs,
            optional_eos=optional_eos,
        )

        config_path = os.path.join(args.output, "config.json")
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config_dict, fh, indent=2, ensure_ascii=False)

        print(f"\nConversion complete → {args.output}")
        print("  config.json")
        if src_vocab is not None:
            print("  vocab.json / vocab.txt")
        print(f"  model.00.safetensors  ({len(written)} tensors)")
        if quant_layers:
            print(f"  Quantised layers ({len(quant_layers)}): {', '.join(quant_layers)}")
