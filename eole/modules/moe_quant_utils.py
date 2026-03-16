"""
moe_quant_utils.py
Utilities for detecting quantisation type and stacking per-expert int4
weights into batch tensors compatible with fused_moe_int4.py.

Supported backends
------------------
* AutoGPTQ / AutoRound  – QuantLinear   (K-packed int4, kpacked=True)
* AutoAWQ               – WQLinear_GEMM (N-packed int4, kpacked=False)
* gptqmodel Marlin      – MarlinQuantLinear  (Marlin tiled layout)

Tensor layout after stacking (E = num_experts, gs = group_size)
----------------------------------------------------------------
GPTQ / AutoRound (kpacked=True):
  w1_qweight  (E, K//8,  2*I)     int32
  w1_scales   (E, K//gs, 2*I)     fp16/bf16
  w1_qzeros   (E, K//gs, 2*I//8)  int32
  w2_qweight  (E, I//8,  H)       int32
  w2_scales   (E, I//gs, H)       fp16/bf16
  w2_qzeros   (E, I//gs, H//8)    int32

AWQ (kpacked=False):
  w1_qweight  (E, K,     2*I//8)  int32
  w1_scales   (E, K//gs, 2*I)     fp16/bf16
  w1_qzeros   (E, K//gs, 2*I//8)  int32
  w2_qweight  (E, I,     H//8)    int32
  w2_scales   (E, I//gs, H)       fp16/bf16
  w2_qzeros   (E, I//gs, H//8)    int32

Marlin (after gptqmodel post_init, for use with moe_wna16_marlin_gemm):
  w1_qweight  (E, K//16, 4*I)     int32   – gate+up concatenated along N dim
  w1_scales   (E, K//gs, 2*I)     fp16    – gate+up scales concatenated
  w2_qweight  (E, I//16, 2*K)     int32
  w2_scales   (E, I//gs, K)       fp16
  (qzeros, g_idx, perm are empty tensors for symmetric quantization)

where K = in_features (= H for W1, = I for W2)
      I = per-stream intermediate size (W1 out_features = 2*I for gated)
"""

from __future__ import annotations
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Optional vLLM integration – probed once at import time
# ─────────────────────────────────────────────────────────────────────────────

# Cache the vLLM moe_align_block_size C++ kernel as a direct callable so that
# _moe_align_block_size() can call it without doing an import or attribute
# lookup on every forward pass.
try:
    import vllm._custom_ops as _vllm_custom_ops

    _vllm_moe_align_block_size_fn = getattr(_vllm_custom_ops, "moe_align_block_size", None)
    _vllm_moe_wna16_marlin_gemm_fn = getattr(_vllm_custom_ops, "moe_wna16_marlin_gemm", None)
except (ImportError, ModuleNotFoundError):
    _vllm_custom_ops = None
    _vllm_moe_align_block_size_fn = None
    _vllm_moe_wna16_marlin_gemm_fn = None

# ─────────────────────────────────────────────────────────────────────────────
# Named constants
# ─────────────────────────────────────────────────────────────────────────────

# Fallback SM count used when torch.cuda.get_device_properties() is unavailable.
# 160 covers H100 SXM5 (132 SMs) and future high-end GPUs with headroom; a
# larger value only wastes a few KB of workspace memory so erring high is safe.
_DEFAULT_SM_COUNT_FALLBACK = 160

# Marlin scalar type IDs as used by vLLM / gptqmodel_marlin_kernels.
# These map to the ScalarType enum values inside the C++ extension.
# Must stay in sync with vLLM's scalar_types registry; verify with
# ``vllm.scalar_type.scalar_types.uint4b8.id`` (== 4) and
# ``vllm.scalar_type.scalar_types.uint8b128.id`` (== 5).
MARLIN_UINT4B8_TYPE_ID = 4   # uint4b8: 4-bit unsigned with bias 8 (symmetric int4)
MARLIN_UINT8B128_TYPE_ID = 5  # uint8b128: 8-bit unsigned with bias 128 (symmetric int8)


# ─────────────────────────────────────────────────────────────────────────────
# vLLM Marlin MoE availability
# ─────────────────────────────────────────────────────────────────────────────


def _vllm_moe_marlin_available() -> bool:
    """Return True when vLLM's moe_wna16_marlin_gemm wrapper is importable.

    We probe ``vllm._custom_ops`` rather than ``torch.ops._moe_C`` directly
    because PyTorch op-namespace attributes (``OpOverloadPacket`` /
    ``_OpNamespace``) are lazily registered objects that do **not** pass
    Python's ``callable()`` test before the op has been invoked at least once.
    Importing ``vllm._custom_ops`` both loads the shared-library extension
    *and* exposes a proper Python function for ``moe_wna16_marlin_gemm``.
    """
    return _vllm_moe_wna16_marlin_gemm_fn is not None


# ─────────────────────────────────────────────────────────────────────────────
# Marlin workspace / block-size helpers
# ─────────────────────────────────────────────────────────────────────────────


def _marlin_make_workspace(device: torch.device, max_blocks_per_sm: int = 4) -> torch.Tensor:
    """Allocate the integer workspace required by the Marlin GEMM kernel.

    The workspace size matches gptqmodel's ``marlin_make_workspace_new`` but
    uses a slightly larger default (4 blocks per SM instead of 1) for the
    batched MoE kernel which may spawn more concurrent thread-blocks.
    """
    try:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        # Conservative fallback when device properties cannot be queried.
        # 108 SMs is the A100-SXM4-80GB count; a larger value only wastes a
        # few KB of workspace memory so erring on the high side is safe.
        sms = _DEFAULT_SM_COUNT_FALLBACK
    return torch.zeros(sms * max_blocks_per_sm, dtype=torch.int32, device=device)


def _exclusive_cumsum(t: torch.Tensor) -> torch.Tensor:
    """Return the exclusive prefix sum of 1-D tensor *t*.

    ``result[i] = sum(t[0..i-1])``,  ``result[0] = 0``.
    """
    return torch.cat([torch.zeros(1, dtype=t.dtype, device=t.device), t.cumsum(0)])[:-1]


def _moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    pre_sorted_ids: "torch.Tensor | None" = None,
    pre_expert_ids: "torch.Tensor | None" = None,
    pre_ntpp: "torch.Tensor | None" = None,
    pre_topk_ids_i32: "torch.Tensor | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the routing structures required by the Marlin MoE GEMM kernel.

    Pads each expert's token set to a multiple of *block_size* and returns
    three tensors that drive the Marlin MoE GEMM kernel:

    sorted_token_ids : (total_padded,) int32
        Flat token-expert pair indices (position in topk_ids.flatten()) for
        real slots; value ``M * topk`` for padding slots.
    expert_ids : (num_blocks,) int32
        Expert index for every GEMM block (one block = *block_size* slots).
    num_tokens_post_padded : (1,) int32
        Total number of slots including padding (= len(sorted_token_ids)).

    Optional pre-allocated output buffers
    --------------------------------------
    When the caller provides *pre_sorted_ids*, *pre_expert_ids*, and
    *pre_ntpp* that are large enough (see sizing notes below), those buffers
    are used directly and no GPU memory allocation occurs.  This eliminates
    3–4 small CUDA malloc/free calls per MoE layer per decode step, which at
    M=1 can represent a meaningful fraction of total dispatch overhead.

    Buffer sizing (worst-case pre-allocation for any block_size ≤ MAX_BLOCK):
      pre_sorted_ids.numel()  ≥ M*topk + num_experts * (MAX_BLOCK - 1)
      pre_expert_ids.numel()  ≥ ceil(above / MAX_BLOCK) + 1
      pre_ntpp.numel()        ≥ 1

    Passing *pre_topk_ids_i32* (shape matching topk_ids, dtype int32) avoids
    the per-call ``topk_ids.to(int32)`` allocation when ``topk_ids`` is int64.
    The buffer is filled in-place using ``copy_`` before the kernel call.

    Two implementations are tried in order:

    1. **Fast path** – ``vllm._custom_ops.moe_align_block_size``: the vLLM C++
       kernel runs entirely on the GPU with zero Python-loop or CPU–GPU sync
       overhead.  Output buffers are pre-allocated to the worst-case size
       ``M * topk + num_experts * (block_size - 1)`` so no ``.item()`` call
       is needed before launching the kernel.

    2. **Vectorised fallback** – a pure-PyTorch implementation with **no
       Python loops** and a single ``.item()`` call (vs. the previous
       O(num_experts) CPU–GPU synchronisation round-trips).
    """
    M, topk = topk_ids.shape
    num_tokens = M * topk  # also serves as the padding-sentinel value
    device = topk_ids.device

    # ── Fast path: vLLM C++ kernel ────────────────────────────────────────
    # Use pre-allocated buffers when provided and large enough; otherwise fall
    # back to dynamic allocation.  The C++ kernel only writes the valid prefix
    # [0, num_tokens_post_padded[0]) so passing a larger-than-minimum buffer
    # is always safe.
    if _vllm_moe_align_block_size_fn is not None:
        max_padded = num_tokens + num_experts * (block_size - 1)
        max_blocks = (max_padded + block_size - 1) // block_size

        if pre_sorted_ids is not None and pre_sorted_ids.numel() >= max_padded:
            sorted_token_ids = pre_sorted_ids
        else:
            sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device=device)

        if pre_expert_ids is not None and pre_expert_ids.numel() >= max_blocks:
            expert_ids = pre_expert_ids
        else:
            expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)

        if pre_ntpp is not None:
            num_tokens_post_padded = pre_ntpp
        else:
            num_tokens_post_padded = torch.empty(1, dtype=torch.int32, device=device)

        # Resolve int32 topk_ids: reuse pre-allocated buffer when available to
        # avoid the per-call allocation from .to(torch.int32).
        if topk_ids.dtype == torch.int32:
            topk_ids_i32 = topk_ids
        elif pre_topk_ids_i32 is not None and pre_topk_ids_i32.numel() >= topk_ids.numel():
            pre_topk_ids_i32.view_as(topk_ids).copy_(topk_ids)
            topk_ids_i32 = pre_topk_ids_i32.view_as(topk_ids)
        else:
            topk_ids_i32 = topk_ids.to(torch.int32)

        # The C++ kernel writes all slots [0, num_tokens_post_padded) including
        # padding sentinels, so no pre-fill of sorted_token_ids is needed.
        _vllm_moe_align_block_size_fn(
            topk_ids_i32,
            num_experts,
            block_size,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
        )
        return sorted_token_ids, expert_ids, num_tokens_post_padded

    # ── Vectorised fallback (no Python loops) ────────────────────────────
    flat_ids = topk_ids.reshape(-1).long()  # (M * topk,) – expert IDs

    # Tokens per expert and their padded equivalents
    counts = torch.bincount(flat_ids, minlength=num_experts)  # (E,) int64
    padded_counts = ((counts + block_size - 1) // block_size) * block_size  # (E,)

    # Single .item() to determine output sizes (unavoidable for pre-allocation)
    total_padded = int(padded_counts.sum().item())
    num_blocks = total_padded // block_size

    # Exclusive prefix sums: padded_starts[e] = sum(padded_counts[0..e-1])
    padded_starts = _exclusive_cumsum(padded_counts)  # (E,)
    # Exclusive prefix sums: real_starts[e] = sum(counts[0..e-1])
    real_starts = _exclusive_cumsum(counts)  # (E,)

    # Sort token-expert pair flat indices by expert ID (stable → deterministic)
    sort_order = torch.argsort(flat_ids, stable=True)  # (M * topk,)
    sorted_flat_ids = flat_ids[sort_order]  # expert ID for each sorted slot

    # Compute each sorted element's destination position in the padded output
    arange = torch.arange(num_tokens, dtype=torch.long, device=device)
    output_pos = padded_starts[sorted_flat_ids] + (arange - real_starts[sorted_flat_ids])

    # Place real token indices; padding slots keep the sentinel value
    sorted_token_ids = torch.full((total_padded,), num_tokens, dtype=torch.int32, device=device)
    sorted_token_ids[output_pos] = sort_order.to(torch.int32)

    # expert_ids: one entry per GEMM block (repeat_interleave is fully vectorised)
    block_counts = padded_counts // block_size  # (E,) int64
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=device),
        block_counts,
    )  # (num_blocks,) int32

    num_tokens_post_padded = torch.tensor([total_padded], dtype=torch.int32, device=device)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_expert_quant_type(experts) -> str:
    """
    Inspect the first expert gate_up_proj and return one of:
      'gptq'   – AutoGPTQ / AutoRound  (K-packed int4)
      'awq'    – AutoAWQ               (N-packed int4)
      'marlin' – gptqmodel MarlinQuantLinear when vLLM moe_wna16_marlin_gemm
                 is available (uses fused Marlin MoE kernel)
      'fp16'   – plain nn.Linear / anything else (including Marlin without vLLM)

    Note on gptqmodel Marlin layers
    --------------------------------
    gptqmodel's Marlin backend repacks the standard GPTQ qweight (shape
    ``(in_features//8, out_features)``) into its own tiled layout during
    ``post_init()``.  After repacking the shape no longer matches either
    the GPTQ (K-packed) or AWQ (N-packed) conventions.

    When ``vllm._custom_ops.moe_wna16_marlin_gemm`` is importable we return
    'marlin' so that ``stack_marlin_moe_weights`` + the fused Marlin MoE
    kernel path in moe.py are used.  Otherwise we fall through to 'fp16' so
    that ``vectorized_moe`` calls each expert's own ``forward()`` (which
    already uses the fast per-layer Marlin CUDA kernel).
    """
    if not experts:
        return "fp16"

    layer = experts[0].gate_up_proj
    module_path = type(layer).__module__
    classname = type(layer).__name__

    # gptqmodel Marlin layers live under the "gptqmodel" namespace.
    # Their qweight has been repacked into Marlin tiles and is incompatible
    # with our int4 Triton kernel.
    # When vLLM's moe_wna16_marlin_gemm is available we use the fused Marlin
    # MoE kernel (faster than calling each expert's own forward() separately).
    # Otherwise fall back to per-expert vectorized_moe.
    if "gptqmodel" in module_path:
        if _vllm_moe_marlin_available():
            return "marlin"
        return "fp16"

    # AutoGPTQ / AutoRound: QuantLinear with qweight.
    # Distinguish from AWQ's QuantLinear by packing axis:
    #   GPTQ : qweight.shape[0] == in_features // 8  (K-packed)
    #   AWQ  : qweight.shape[0] == in_features        (N-packed)
    # Any other shape (e.g. Marlin-repacked qweight from a wrapper that does
    # not live under "gptqmodel") is treated as an unrecognised format and
    # falls through to 'fp16'.
    if classname == "QuantLinear" and hasattr(layer, "qweight"):
        in_f = (
            layer.infeatures
            if hasattr(layer, "infeatures")
            else layer.in_features if hasattr(layer, "in_features") else None
        )
        if in_f is not None:
            if layer.qweight.shape[0] == in_f:
                return "awq"
            if layer.qweight.shape[0] == in_f // 8:
                return "gptq"
            # Unrecognised packing (e.g. Marlin-repacked qweight exposed via a
            # non-gptqmodel wrapper) – fall through to per-expert forward().
            return "fp16"
        # Cannot determine in_features; cannot safely validate the weight layout,
        # so fall back to per-expert forward() rather than risk a kernel crash.
        return "fp16"

    if "WQLinear" in classname and hasattr(layer, "qweight"):
        return "awq"

    if "auto_round" in module_path and hasattr(layer, "qweight"):
        return "gptq"

    return "fp16"


# ─────────────────────────────────────────────────────────────────────────────
# Low-level tensor extractors
# ─────────────────────────────────────────────────────────────────────────────


def _gptq_tensors(layer):
    """Return (qweight, scales, qzeros, group_size) from a GPTQ QuantLinear.

    qweight : (K//8, N)      int32
    scales  : (K//gs, N)     fp16
    qzeros  : (K//gs, N//8)  int32
    """
    qw, sc, qz = layer.qweight, layer.scales, layer.qzeros
    gs = (
        layer.group_size
        if (hasattr(layer, "group_size") and layer.group_size > 0)
        else (qw.shape[0] * 8) // sc.shape[0]
    )
    return qw, sc, qz, gs


def _awq_tensors(layer):
    """Return (qweight, scales, qzeros, group_size) from an AWQ WQLinear.

    qweight : (K, N//8)      int32
    scales  : (K//gs, N)     fp16
    qzeros  : (K//gs, N//8)  int32
    """
    qw, sc, qz = layer.qweight, layer.scales, layer.qzeros
    if hasattr(layer, "group_size") and layer.group_size > 0:
        gs = layer.group_size
    else:
        gs = qw.shape[0] // sc.shape[0]
    return qw, sc, qz, gs


def _cat_n(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Concatenate along the last (output / N) dimension."""
    return torch.cat([t1, t2], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Weight stacking
# ─────────────────────────────────────────────────────────────────────────────


def stack_gptq_moe_weights(experts, device: torch.device):
    """
    Stack per-expert GPTQ / AutoRound int4 weights into batched tensors.

    Returns
    -------
    w1_qweight  (E, K//8,  2*I)     int32
    w1_scales   (E, K//gs, 2*I)     fp16
    w1_qzeros   (E, K//gs, 2*I//8)  int32
    w2_qweight  (E, I//8,  H)       int32
    w2_scales   (E, I//gs, H)       fp16
    w2_qzeros   (E, I//gs, H//8)    int32
    group_size  int
    """
    w1_qw, w1_sc, w1_qz = [], [], []
    w2_qw, w2_sc, w2_qz = [], [], []
    group_size = None

    for e in experts:
        qw_g, sc_g, qz_g, gs = _gptq_tensors(e.gate_up_proj)
        if group_size is None:
            group_size = gs

        up = getattr(e, "up_proj", None)
        if up is not None:
            qw_u, sc_u, qz_u, _ = _gptq_tensors(up)
            qw1 = _cat_n(qw_g, qw_u)
            sc1 = _cat_n(sc_g, sc_u)
            qz1 = _cat_n(qz_g, qz_u)
        else:
            qw1, sc1, qz1 = qw_g, sc_g, qz_g

        w1_qw.append(qw1.unsqueeze(0))
        w1_sc.append(sc1.unsqueeze(0))
        w1_qz.append(qz1.unsqueeze(0))

        qw_d, sc_d, qz_d, _ = _gptq_tensors(e.down_proj)
        w2_qw.append(qw_d.unsqueeze(0))
        w2_sc.append(sc_d.unsqueeze(0))
        w2_qz.append(qz_d.unsqueeze(0))

        _free_expert_weights_single(e)

    return (
        torch.cat(w1_qw, dim=0).to(device),
        torch.cat(w1_sc, dim=0).to(device),
        torch.cat(w1_qz, dim=0).to(device),
        torch.cat(w2_qw, dim=0).to(device),
        torch.cat(w2_sc, dim=0).to(device),
        torch.cat(w2_qz, dim=0).to(device),
        group_size,
    )


def stack_awq_moe_weights(experts, device: torch.device):
    """
    Stack per-expert AWQ int4 weights into batched tensors.

    Returns the same 7-tuple as stack_gptq_moe_weights but N-packed:
    w1_qweight  (E, K,     2*I//8)  int32
    w1_scales   (E, K//gs, 2*I)     fp16
    w1_qzeros   (E, K//gs, 2*I//8)  int32
    w2_qweight  (E, I,     H//8)    int32
    w2_scales   (E, I//gs, H)       fp16
    w2_qzeros   (E, I//gs, H//8)    int32
    group_size  int
    """
    w1_qw, w1_sc, w1_qz = [], [], []
    w2_qw, w2_sc, w2_qz = [], [], []
    group_size = None

    for e in experts:
        qw_g, sc_g, qz_g, gs = _awq_tensors(e.gate_up_proj)
        if group_size is None:
            group_size = gs

        up = getattr(e, "up_proj", None)
        if up is not None:
            qw_u, sc_u, qz_u, _ = _awq_tensors(up)
            qw1 = _cat_n(qw_g, qw_u)
            sc1 = _cat_n(sc_g, sc_u)
            qz1 = _cat_n(qz_g, qz_u)
        else:
            qw1, sc1, qz1 = qw_g, sc_g, qz_g

        w1_qw.append(qw1.unsqueeze(0))
        w1_sc.append(sc1.unsqueeze(0))
        w1_qz.append(qz1.unsqueeze(0))

        qw_d, sc_d, qz_d, _ = _awq_tensors(e.down_proj)
        w2_qw.append(qw_d.unsqueeze(0))
        w2_sc.append(sc_d.unsqueeze(0))
        w2_qz.append(qz_d.unsqueeze(0))

        _free_expert_weights_single(e)

    return (
        torch.cat(w1_qw, dim=0).to(device),
        torch.cat(w1_sc, dim=0).to(device),
        torch.cat(w1_qz, dim=0).to(device),
        torch.cat(w2_qw, dim=0).to(device),
        torch.cat(w2_sc, dim=0).to(device),
        torch.cat(w2_qz, dim=0).to(device),
        group_size,
    )


def stack_fp16_moe_weights(experts, device: torch.device):
    """Stack plain fp16/bf16 expert weights for vectorized MoE forward."""
    w1, w2 = [], []

    for e in experts:
        gate_up = e.gate_up_proj.weight.data  # (2I, H)
        down = e.down_proj.weight.data  # (H, I)

        w1.append(gate_up.unsqueeze(0))
        w2.append(down.unsqueeze(0))

        _free_expert_weights_single(e)

    return (
        torch.cat(w1, dim=0).to(device),
        torch.cat(w2, dim=0).to(device),
    )


def _free_expert_weights_single(expert):
    """Free one expert's weight tensors immediately after stacking."""
    for proj_name in ("gate_up_proj", "up_proj", "down_proj"):
        proj = getattr(expert, proj_name, None)
        if proj is None:
            continue
        # quantized backends
        for attr in ("qweight", "scales", "qzeros"):
            if hasattr(proj, attr):
                setattr(proj, attr, None)
        # fp16 plain linear
        if hasattr(proj, "weight") and proj.weight is not None:
            proj.weight = None
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Marlin MoE weight stacking
# ─────────────────────────────────────────────────────────────────────────────


def stack_marlin_moe_weights(experts, device: torch.device):
    """Stack per-expert gptqmodel MarlinQuantLinear weights for moe_wna16_marlin_gemm.

    After ``post_init()`` each ``MarlinQuantLinear`` stores weights in the
    Marlin tile format (``gptq_marlin_repack`` applied).  For int4 symmetric
    quantization the layout per expert is::

        gate_up_proj.qweight  (K//16, 2*I)  int32   – Marlin-tiled K→I weight
        up_proj.qweight       (K//16, 2*I)  int32   – Marlin-tiled K→I weight
        down_proj.qweight     (I//16, 2*K)  int32   – Marlin-tiled I→K weight

    The gate and up projections are concatenated along their last (N) dimension
    before stacking, producing the combined W1 weight expected by
    ``moe_wna16_marlin_gemm`` with ``size_n = 2*I``.  This concatenation is
    valid because Marlin tiling is column-independent: repacking (K, I)
    separately and concatenating along N gives the same result as repacking
    the combined (K, 2I) weight matrix in one shot.

    Returns
    -------
    w1_qweight  (E, K//16, 4*I)   int32   – combined gate+up Marlin weight
    w1_scales   (E, K//gs, 2*I)   fp16    – combined gate+up scales
    w1_qzeros   (E, 0)            int32   – empty (symmetric quantization)
    w1_g_idx    (E, 0)            int32   – empty (no activation reordering)
    w1_perm     (E, 0)            int32   – empty
    w2_qweight  (E, I//16, 2*K)   int32
    w2_scales   (E, I//gs, K)     fp16
    w2_qzeros   (E, 0)            int32   – empty
    w2_g_idx    (E, 0)            int32   – empty
    w2_perm     (E, 0)            int32   – empty
    workspace   (sms * 4,)        int32   – shared Marlin workspace
    num_bits    int               – quantization bit-width (4 or 8)
    scalar_type_id  int           – Marlin scalar type ID (e.g. uint4b8.id)
    group_size  int               – quantization group size
    """
    w1_qw, w1_sc = [], []
    w2_qw, w2_sc = [], []
    num_bits = None
    scalar_type_id = None
    group_size = None

    for e in experts:
        gate_layer = e.gate_up_proj
        up_layer = getattr(e, "up_proj", None)
        down_layer = e.down_proj

        if num_bits is None:
            num_bits = getattr(gate_layer, "bits", 4)
            group_size = getattr(gate_layer, "group_size", 128)
            # Retrieve the Marlin scalar type ID (e.g. uint4b8.id)
            weight_type = getattr(gate_layer, "weight_type", None)
            if weight_type is not None:
                scalar_type_id = int(weight_type.id)
            else:
                # Fallback: derive ID from bits assuming symmetric quantization.
                # Only 4-bit (uint4b8) and 8-bit (uint8b128) are supported;
                # other bit-widths are not used by the Marlin kernel path.
                scalar_type_id = MARLIN_UINT4B8_TYPE_ID if num_bits == 4 else MARLIN_UINT8B128_TYPE_ID

        # ── W1: concatenate gate and up along the N (last) dimension ─────────
        if up_layer is not None:
            # Separate gate and up projections – concatenate Marlin qweights
            # along the last dim (valid since Marlin tiling is column-independent)
            qw1 = torch.cat([gate_layer.qweight, up_layer.qweight], dim=-1)
            sc1 = torch.cat([gate_layer.scales, up_layer.scales], dim=-1)
        else:
            # gate_up_proj already contains the fused gate+up weight
            qw1 = gate_layer.qweight
            sc1 = gate_layer.scales

        w1_qw.append(qw1.unsqueeze(0))
        w1_sc.append(sc1.unsqueeze(0))

        # ── W2: down projection ───────────────────────────────────────────────
        w2_qw.append(down_layer.qweight.unsqueeze(0))
        w2_sc.append(down_layer.scales.unsqueeze(0))

        _free_expert_weights_single(e)

    # Empty tensors for symmetric (no zero-points, no g_idx / perm)
    empty = torch.empty(0, dtype=torch.int32, device=device)

    w1_qweight = torch.cat(w1_qw, dim=0).to(device)
    w1_scales = torch.cat(w1_sc, dim=0).to(device)
    w2_qweight = torch.cat(w2_qw, dim=0).to(device)
    w2_scales = torch.cat(w2_sc, dim=0).to(device)

    workspace = _marlin_make_workspace(device)

    return (
        w1_qweight,
        w1_scales,
        empty,  # w1_qzeros
        empty,  # w1_g_idx
        empty,  # w1_perm
        w2_qweight,
        w2_scales,
        empty,  # w2_qzeros
        empty,  # w2_g_idx
        empty,  # w2_perm
        workspace,
        num_bits,
        scalar_type_id,
        group_size,
    )
