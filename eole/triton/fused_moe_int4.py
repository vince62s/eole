"""
Int4 MoE Triton kernels supporting GPTQ/AutoRound (K-packed) and AWQ (N-packed) formats.

Layout conventions
------------------
GPTQ / AutoRound (KPACKED=True)
  qweight : (E, K//8,          N)  int32  – 8 int4 values packed along K (input) dim
  scales  : (E, K//group_size, N)  fp16
  qzeros  : (E, K//group_size, N//8) int32 – 8 int4 zero-points packed along N dim

AWQ (KPACKED=False)
  qweight : (E, K,             N//8) int32 – 8 int4 values packed along N (output) dim
  scales  : (E, K//group_size, N)   fp16
  qzeros  : (E, K//group_size, N//8) int32 – same as GPTQ zeros

where  K = in_features (H for W1, I for W2)
       N = out_features (2*I for W1, H for W2)

For W1 (gate_up projection) the kernel simultaneously computes
gate = W_gate @ x  and  up = W_up @ x,
then applies gated SiLU/GELU/ReLU in-kernel.
For W2 (down projection) the kernel applies a weighted atomic-add reduce
directly into the output buffer.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Optional vLLM integration – probed once at import time
# ─────────────────────────────────────────────────────────────────────────────
# Cache the vLLM Marlin MoE GEMM wrapper, ScalarType constructor, and fused
# activation kernels so that fused_experts_marlin_impl() can call them without
# per-call imports or attribute lookups.
try:
    import vllm._custom_ops as _vllm_ops
    from vllm.scalar_type import ScalarType as _VllmScalarType

    _vllm_moe_wna16_marlin_gemm = _vllm_ops.moe_wna16_marlin_gemm
    _HAS_VLLM_MARLIN = True
    # Fused gated-activation kernels: silu_and_mul(out, inp) / gelu_and_mul(out, inp)
    # where inp shape is (M, 2*N) and out shape is (M, N).  They replace the
    # two-kernel F.silu + element-wise multiply with a single CUDA kernel and
    # require no intermediate allocation.
    _ops_C = getattr(torch.ops, "_C", None)
    _silu_and_mul_fn = getattr(_ops_C, "silu_and_mul", None)
    _gelu_and_mul_fn = getattr(_ops_C, "gelu_and_mul", None)
except (ImportError, ModuleNotFoundError, AttributeError):
    _vllm_ops = None  # type: ignore[assignment]
    _VllmScalarType = None  # type: ignore[assignment,misc]
    _vllm_moe_wna16_marlin_gemm = None  # type: ignore[assignment]
    _HAS_VLLM_MARLIN = False
    _silu_and_mul_fn = None
    _gelu_and_mul_fn = None

from eole.modules.moe_quant_utils import _moe_align_block_size  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Buffer view helper (mirrors vLLM's _resize_cache)
# ─────────────────────────────────────────────────────────────────────────────


def _view_cache(flat: torch.Tensor, *shape: int) -> torch.Tensor:
    """Return a view of *flat* with the given *shape*.

    The total number of elements in *shape* must be ≤ flat.numel().  This is
    the equivalent of vLLM's ``_resize_cache`` and lets two differently-shaped
    views share the same underlying storage (e.g. the W1 GEMM output and the
    W2 GEMM output both live in the same flat buffer since they are never live
    at the same time).
    """
    n = 1
    for s in shape:
        n *= s
    return flat[:n].view(*shape)



# ---------------------------------------------------------------------------
# W1 kernel: int4 matmul + gated activation (gate and up in one pass)
# ---------------------------------------------------------------------------


@triton.jit
def _w1_int4_act_kernel(
    # ── inputs ──────────────────────────────────────────────────────────────
    X_ptr,  # (M, H)         fp16/bf16  – input tokens
    Qw_ptr,  # see layout doc above
    Sc_ptr,  # (E, K//gs, 2I) fp16       – scales
    Qz_ptr,  # (E, K//gs, 2I//8) int32   – packed zeros
    Y_ptr,  # (num_pairs, I) fp16/bf16  – intermediate output
    expert_ids_ptr,  # (num_pairs,)   int32
    token_ids_ptr,  # (num_pairs,)   int32
    # ── strides ─────────────────────────────────────────────────────────────
    sx_m,
    sx_k,  # X
    sq_e,
    sq_r,
    sq_c,  # Qw  (e / row / col)
    ss_e,
    ss_r,
    ss_c,  # Sc
    sz_e,
    sz_r,
    sz_c,  # Qz
    sy_m,
    sy_n,  # Y
    # ── problem dims (constexpr → compiled-in, no recompile per batch) ──────
    H: tl.constexpr,  # input hidden size
    I: tl.constexpr,  # intermediate size (gate OR up, so W1 width = 2*I)
    group_size: tl.constexpr,  # quantisation group size (e.g. 128)
    # ── layout / activation flags ───────────────────────────────────────────
    KPACKED: tl.constexpr,  # True = GPTQ/AutoRound, False = AWQ
    ACTIVATION: tl.constexpr,  # 0 = SiLU (default), 1 = GELU approx, 2 = ReLU
    # ── tile sizes ──────────────────────────────────────────────────────────
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # ── packing constants ────────────────────────────────────────────────────
    KPACK: tl.constexpr,  # 8  (int4: 8 values per int32)
    NPACK: tl.constexpr,  # 8  (zero-points also packed ×8 along N)
    # ── dynamic scalar ───────────────────────────────────────────────────────
    num_pairs,  # NOT constexpr – avoids recompile per batch size
):
    pid_pair = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_pair >= num_pairs:
        return

    eid = tl.load(expert_ids_ptr + pid_pair).to(tl.int64)
    tid = tl.load(token_ids_ptr + pid_pair).to(tl.int64)

    # Output indices for the gate (first I outputs); up uses offs_n + I
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < I

    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base = X_ptr + tid * sx_m

    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        x = tl.load(x_base + offs_k * sx_k, mask=mask_k, other=0.0).to(tl.float32)

        sc_row = offs_k // group_size  # (BLOCK_K,) – which scale row

        # ── load & unpack int4 weights ───────────────────────────────────────
        if KPACKED:
            # GPTQ: qweight[e, k//8, n] – packed along K
            qw_row = offs_k // KPACK  # (BLOCK_K,)
            k_shift = ((offs_k % KPACK) * 4).to(tl.int32)  # (BLOCK_K,)

            # Load (BLOCK_K, BLOCK_N) int32 for gate outputs
            qw_gate = tl.load(
                Qw_ptr + eid * sq_e + qw_row[:, None] * sq_r + offs_n[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )
            # Load (BLOCK_K, BLOCK_N) int32 for up outputs (starts at column I)
            qw_up = tl.load(
                Qw_ptr + eid * sq_e + qw_row[:, None] * sq_r + (offs_n[None, :] + I) * sq_c,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )

            gate_nib = ((qw_gate >> k_shift[:, None]) & 0xF).to(tl.float32)
            up_nib = ((qw_up >> k_shift[:, None]) & 0xF).to(tl.float32)

        else:
            # AWQ: qweight[e, k, n//8] – packed along N
            qw_col_g = offs_n // NPACK  # (BLOCK_N,)
            qw_col_u = (offs_n + I) // NPACK
            n_shift = ((offs_n % NPACK) * 4).to(tl.int32)  # (BLOCK_N,)

            qw_gate = tl.load(
                Qw_ptr + eid * sq_e + offs_k[:, None] * sq_r + qw_col_g[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )
            qw_up = tl.load(
                Qw_ptr + eid * sq_e + offs_k[:, None] * sq_r + qw_col_u[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )

            gate_nib = ((qw_gate >> n_shift[None, :]) & 0xF).to(tl.float32)
            up_nib = ((qw_up >> n_shift[None, :]) & 0xF).to(tl.float32)

        # ── scales (BLOCK_K, BLOCK_N) ────────────────────────────────────────
        sc_gate = tl.load(
            Sc_ptr + eid * ss_e + sc_row[:, None] * ss_r + offs_n[None, :] * ss_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        sc_up = tl.load(
            Sc_ptr + eid * ss_e + sc_row[:, None] * ss_r + (offs_n[None, :] + I) * ss_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        # ── zero-points (always N-packed, both layouts share this convention) ──
        z_col = offs_n // NPACK  # (BLOCK_N,)
        z_shft = ((offs_n % NPACK) * 4).to(tl.int32)  # (BLOCK_N,)
        z_col_u = (offs_n + I) // NPACK

        qz_gate = tl.load(
            Qz_ptr + eid * sz_e + sc_row[:, None] * sz_r + z_col[None, :] * sz_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        )
        qz_up = tl.load(
            Qz_ptr + eid * sz_e + sc_row[:, None] * sz_r + z_col_u[None, :] * sz_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        )

        z_gate = ((qz_gate >> z_shft[None, :]) & 0xF).to(tl.float32)
        z_up = ((qz_up >> z_shft[None, :]) & 0xF).to(tl.float32)

        # ── dequantize and accumulate ─────────────────────────────────────────
        w_gate = (gate_nib - z_gate) * sc_gate  # (BLOCK_K, BLOCK_N)
        w_up = (up_nib - z_up) * sc_up

        acc_gate += tl.sum(x[:, None] * w_gate, axis=0)
        acc_up += tl.sum(x[:, None] * w_up, axis=0)

    # ── gated activation ─────────────────────────────────────────────────────
    if ACTIVATION == 0:  # SiLU (default for most MoE models)
        act = acc_gate * tl.sigmoid(acc_gate) * acc_up
    elif ACTIVATION == 1:  # GELU approximation
        sqrt_2_over_pi: tl.constexpr = 0.7978845608028654
        g3 = acc_gate * acc_gate * acc_gate
        tanh_arg = sqrt_2_over_pi * (acc_gate + 0.044715 * g3)
        act = 0.5 * acc_gate * (1.0 + tl.math.tanh(tanh_arg)) * acc_up
    else:  # ReLU
        act = tl.where(acc_gate > 0, acc_gate, 0.0) * acc_up

    tl.store(
        Y_ptr + pid_pair * sy_m + offs_n * sy_n,
        act.to(X_ptr.dtype.element_ty),
        mask=mask_n,
    )


# ---------------------------------------------------------------------------
# W2 kernel: int4 matmul + weighted atomic-add reduce into output
# ---------------------------------------------------------------------------


@triton.jit
def _w2_int4_reduce_kernel(
    X_ptr,  # (num_pairs, I) fp16/bf16  – intermediate (from W1 kernel)
    Qw_ptr,  # GPTQ: (E, I//8, H)  AWQ: (E, I, H//8)  int32
    Sc_ptr,  # (E, I//group_size, H)  fp16
    Qz_ptr,  # (E, I//group_size, H//8) int32
    Y_ptr,  # (M, H)         fp16/bf16  – output (atomic accumulation)
    expert_ids_ptr,
    token_ids_ptr,
    weights_ptr,  # (num_pairs,)   fp16/bf16  – routing weights
    sx_m,
    sx_k,
    sq_e,
    sq_r,
    sq_c,
    ss_e,
    ss_r,
    ss_c,
    sz_e,
    sz_r,
    sz_c,
    sy_m,
    sy_n,
    H: tl.constexpr,
    I: tl.constexpr,
    group_size: tl.constexpr,
    KPACKED: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KPACK: tl.constexpr,
    NPACK: tl.constexpr,
    num_pairs,
):
    pid_pair = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_pair >= num_pairs:
        return

    eid = tl.load(expert_ids_ptr + pid_pair).to(tl.int64)
    tid = tl.load(token_ids_ptr + pid_pair).to(tl.int64)
    weight = tl.load(weights_ptr + pid_pair).to(tl.float32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output (hidden) indices
    mask_n = offs_n < H

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_base = X_ptr + pid_pair * sx_m

    for k0 in range(0, I, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < I

        x = tl.load(x_base + offs_k * sx_k, mask=mask_k, other=0.0).to(tl.float32)

        sc_row = offs_k // group_size

        if KPACKED:
            # GPTQ: qweight[e, k//8, n]  shape (E, I//8, H)
            qw_row = offs_k // KPACK
            k_shift = ((offs_k % KPACK) * 4).to(tl.int32)

            qw = tl.load(
                Qw_ptr + eid * sq_e + qw_row[:, None] * sq_r + offs_n[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )
            nib = ((qw >> k_shift[:, None]) & 0xF).to(tl.float32)

        else:
            # AWQ: qweight[e, k, n//8]  shape (E, I, H//8)
            qw_col = offs_n // NPACK
            n_shift = ((offs_n % NPACK) * 4).to(tl.int32)

            qw = tl.load(
                Qw_ptr + eid * sq_e + offs_k[:, None] * sq_r + qw_col[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )
            nib = ((qw >> n_shift[None, :]) & 0xF).to(tl.float32)

        sc = tl.load(
            Sc_ptr + eid * ss_e + sc_row[:, None] * ss_r + offs_n[None, :] * ss_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        z_col = offs_n // NPACK
        z_shft = ((offs_n % NPACK) * 4).to(tl.int32)
        qz = tl.load(
            Qz_ptr + eid * sz_e + sc_row[:, None] * sz_r + z_col[None, :] * sz_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        )
        z = ((qz >> z_shft[None, :]) & 0xF).to(tl.float32)

        w = (nib - z) * sc
        acc += tl.sum(x[:, None] * w, axis=0)

    weighted = acc * weight
    y_ptrs = Y_ptr + tid * sy_m + offs_n * sy_n
    tl.atomic_add(y_ptrs, weighted.to(X_ptr.dtype.element_ty), mask=mask_n)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

_ACTIVATION_MAP = {"silu": 0, "gelu": 1, "relu": 2}


def fused_experts_int4_impl(
    hidden_states: torch.Tensor,
    w1_qweight: torch.Tensor,
    w1_scales: torch.Tensor,
    w1_qzeros: torch.Tensor,
    w2_qweight: torch.Tensor,
    w2_scales: torch.Tensor,
    w2_qzeros: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    group_size: int = 128,
    kpacked: bool = True,
    activation: str = "silu",
) -> torch.Tensor:
    """
    MoE forward pass with int4-quantised weights.

    Parameters
    ----------
    hidden_states : (M, H)
    w1_qweight    : (E, H//8, 2I) if kpacked else (E, H, 2I//8)  – int32
    w1_scales     : (E, H//group_size, 2I)                         – fp16/bf16
    w1_qzeros     : (E, H//group_size, 2I//8)                      – int32
    w2_qweight    : (E, I//8, H)  if kpacked else (E, I, H//8)    – int32
    w2_scales     : (E, I//group_size, H)                          – fp16/bf16
    w2_qzeros     : (E, I//group_size, H//8)                       – int32
    topk_weights  : (M, K)
    topk_ids      : (M, K)
    group_size    : quantisation group size (default 128)
    kpacked       : True = GPTQ/AutoRound layout, False = AWQ layout
    activation    : "silu" | "gelu" | "relu"
    """
    M, H = hidden_states.shape
    K = topk_ids.shape[1]

    # Derive I (intermediate size) from stacked weight shape
    if kpacked:
        double_I = w1_qweight.shape[2]  # (E, H//8, 2I)  → col dim = 2I
    else:
        double_I = w1_qweight.shape[2] * 8  # (E, H, 2I//8)  → col dim = 2I//8

    I = double_I // 2  # noqa: E741

    device = hidden_states.device
    dtype = hidden_states.dtype
    num_pairs = M * K

    expert_ids = topk_ids.flatten().to(torch.int32)
    token_ids = torch.arange(M, device=device, dtype=torch.int32).repeat_interleave(K)
    weights = topk_weights.flatten()

    act_code = _ACTIVATION_MAP.get(activation.lower(), 0)

    BLOCK_N, BLOCK_K = 64, 64

    # ── W1: gate+up projection + activation → intermediate ────────────────
    intermediate = torch.empty((num_pairs, I), device=device, dtype=dtype)

    grid_w1 = (num_pairs, triton.cdiv(I, BLOCK_N))
    _w1_int4_act_kernel[grid_w1](
        hidden_states,
        w1_qweight,
        w1_scales,
        w1_qzeros,
        intermediate,
        expert_ids,
        token_ids,
        # X strides
        hidden_states.stride(0),
        hidden_states.stride(1),
        # Qw strides
        w1_qweight.stride(0),
        w1_qweight.stride(1),
        w1_qweight.stride(2),
        # Sc strides
        w1_scales.stride(0),
        w1_scales.stride(1),
        w1_scales.stride(2),
        # Qz strides
        w1_qzeros.stride(0),
        w1_qzeros.stride(1),
        w1_qzeros.stride(2),
        # Y strides
        intermediate.stride(0),
        intermediate.stride(1),
        # dims
        H=H,
        I=I,
        group_size=group_size,
        KPACKED=kpacked,
        ACTIVATION=act_code,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        KPACK=8,
        NPACK=8,
        num_pairs=num_pairs,
    )

    # ── W2: down projection + weighted reduce → output ────────────────────
    final_output = torch.zeros((M, H), device=device, dtype=dtype)

    grid_w2 = (num_pairs, triton.cdiv(H, BLOCK_N))
    _w2_int4_reduce_kernel[grid_w2](
        intermediate,
        w2_qweight,
        w2_scales,
        w2_qzeros,
        final_output,
        expert_ids,
        token_ids,
        weights,
        # intermediate strides
        intermediate.stride(0),
        intermediate.stride(1),
        # Qw strides
        w2_qweight.stride(0),
        w2_qweight.stride(1),
        w2_qweight.stride(2),
        # Sc strides
        w2_scales.stride(0),
        w2_scales.stride(1),
        w2_scales.stride(2),
        # Qz strides
        w2_qzeros.stride(0),
        w2_qzeros.stride(1),
        w2_qzeros.stride(2),
        # output strides
        final_output.stride(0),
        final_output.stride(1),
        # dims
        H=H,
        I=I,
        group_size=group_size,
        KPACKED=kpacked,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        KPACK=8,
        NPACK=8,
        num_pairs=num_pairs,
    )

    return final_output


# ---------------------------------------------------------------------------
# Marlin MoE path using vLLM's moe_wna16_marlin_gemm kernel
# ---------------------------------------------------------------------------


def fused_experts_marlin_impl(
    hidden_states: torch.Tensor,
    w1_qweight: torch.Tensor,
    w1_scales: torch.Tensor,
    w1_qzeros: torch.Tensor,
    w1_g_idx: torch.Tensor,
    w1_perm: torch.Tensor,
    w2_qweight: torch.Tensor,
    w2_scales: torch.Tensor,
    w2_qzeros: torch.Tensor,
    w2_g_idx: torch.Tensor,
    w2_perm: torch.Tensor,
    workspace: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b_q_type,  # vllm.scalar_type.ScalarType – pre-created once at setup time
    num_experts: int,
    activation: str = "silu",
    # ── Pre-allocated intermediate buffers (grown lazily in moe.py) ─────────
    # cache13: flat buffer used for both the W1 GEMM output (M*topk, 2*I) and
    # the W2 GEMM output (M*topk, K) via non-overlapping views – exactly the
    # ``intermediate_cache13`` strategy from vLLM's _fused_marlin_moe().
    # cache2 : flat buffer for the activation output (M*topk, I).
    # out    : pre-allocated (M, K) output tensor for the final topk sum.
    # When any of these is None the function falls back to dynamic allocation.
    cache13: "torch.Tensor | None" = None,
    cache2: "torch.Tensor | None" = None,
    out: "torch.Tensor | None" = None,
    # ── Pre-allocated routing buffers (grown lazily in moe.py) ──────────────
    # These eliminate 4 small GPU allocations per decode step per MoE layer:
    #   sorted_ids_buf : (M*topk + E*(MAX_BLOCK-1),) int32
    #   expert_ids_buf : (ceil(above / MAX_BLOCK) + 1,) int32
    #   ntpp_buf       : (1,) int32 – num_tokens_post_padded
    #   topk_ids_i32   : (M, topk) int32 – avoids per-call .to(int32) alloc
    # When None, _moe_align_block_size falls back to dynamic allocation.
    sorted_ids_buf: "torch.Tensor | None" = None,
    expert_ids_buf: "torch.Tensor | None" = None,
    ntpp_buf: "torch.Tensor | None" = None,
    topk_ids_i32: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """MoE forward pass using vLLM's ``moe_wna16_marlin_gemm`` fused kernel.

    Requires ``vllm._custom_ops.moe_wna16_marlin_gemm`` (installed with vLLM).
    The vLLM wrapper is used instead of ``torch.ops._moe_C`` directly because
    the wrapper accepts a ``ScalarType`` object for ``b_q_type`` and handles the
    ``.id`` extraction, making it the stable public API for this kernel.

    Parameters
    ----------
    hidden_states : (M, K)               – input tokens (fp16 / bf16)
    w1_qweight    : (E, K//16, 4*I)      – Marlin-tiled gate+up weight (int32)
    w1_scales     : (E, K//gs, 2*I)      – permuted gate+up scales
    w1_qzeros     : (E, 0)               – empty for symmetric quant
    w1_g_idx      : (E, 0)               – empty (no act-ordering)
    w1_perm       : (E, 0)               – empty
    w2_qweight    : (E, I//16, 2*K)      – Marlin-tiled down weight (int32)
    w2_scales     : (E, I//gs, K)        – permuted down scales
    w2_qzeros     : (E, 0)               – empty
    w2_g_idx      : (E, 0)               – empty
    w2_perm       : (E, 0)               – empty
    workspace     : (sms * 4,) int32     – Marlin workspace
    topk_weights  : (M, topk)  float32   – routing weights (sum to 1 per token)
    topk_ids      : (M, topk)  int32/int64 – selected expert indices
    b_q_type      : ScalarType           – Marlin scalar type (pre-built at setup)
    num_experts   : int                  – total number of experts
    activation    : "silu" | "gelu" | "relu"
    cache13       : flat buffer ≥ M*topk*max(2*I, K) elements (optional)
    cache2        : flat buffer ≥ M*topk*I elements (optional)
    out           : pre-allocated (M, K) output tensor (optional)
    sorted_ids_buf: pre-allocated routing buffer, numel ≥ M*topk+E*(MAX_BLOCK-1)
    expert_ids_buf: pre-allocated routing buffer, numel ≥ ceil(above/MAX_BLOCK)+1
    ntpp_buf      : pre-allocated (1,) int32 for num_tokens_post_padded
    topk_ids_i32  : pre-allocated (M, topk) int32 buffer (avoids per-call cast)

    Returns
    -------
    (M, K) output tensor (= *out* when provided)
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    M_topk = M * topk
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Derive intermediate size I from W2 shape: w2_qweight (E, I//16, 2*K)
    # Matches vLLM's marlin_moe_intermediate_size: I = w2.size(1) * 16
    I = w2_qweight.size(1) * 16  # noqa: E741

    # Ensure routing weights are float32 as required by the Marlin kernel.
    # Guard avoids a redundant GPU copy when weights are already float32.
    topk_weights_f32 = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()

    # ── Pick moe_block_size based on heuristic from vLLM ──────────────────
    # Smaller block sizes work better when tokens per expert is low.
    moe_block_size = 8
    for bs in [8, 16, 32, 48, 64]:
        if M * topk / max(num_experts, 1) / bs < 0.9:
            break
        moe_block_size = bs

    # ── Build sorted routing structures ───────────────────────────────────
    # Pass pre-allocated routing buffers to avoid per-call GPU allocations.
    # The pre-allocated buffers are sized at worst-case (block_size=64) by the
    # caller in moe.py, so they are always large enough for any moe_block_size.
    sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size(
        topk_ids.view(M, topk),
        moe_block_size,
        num_experts,
        pre_sorted_ids=sorted_ids_buf,
        pre_expert_ids=expert_ids_buf,
        pre_ntpp=ntpp_buf,
        pre_topk_ids_i32=topk_ids_i32,
    )

    # ── Prepare intermediate buffers ─────────────────────────────────────
    # cache13 is a flat buffer large enough to hold both:
    #   cache_w1  (M_topk, 2*I)  – W1 GEMM output
    #   cache_w2  (M_topk, K)    – W2 GEMM output (reuses same memory)
    # The two views never overlap in time so sharing is safe.
    # cache2 holds the activation output (M_topk, I).
    need13 = M_topk * max(2 * I, K)
    need2 = M_topk * I
    if cache13 is None or cache13.numel() < need13:
        cache13 = torch.empty(need13, device=device, dtype=dtype)
    if cache2 is None or cache2.numel() < need2:
        cache2 = torch.empty(need2, device=device, dtype=dtype)

    cache_w1 = _view_cache(cache13, M_topk, 2 * I)   # W1 output view
    cache_w2 = _view_cache(cache13, M_topk, K)        # W2 output view (same flat buffer)
    cache_act = _view_cache(cache2, M_topk, I)        # activation output

    # ── W1: gate+up projection → cache_w1 (M*topk, 2*I) ──────────────────
    _vllm_moe_wna16_marlin_gemm(
        hidden_states,
        cache_w1,  # write directly into pre-allocated buffer
        w1_qweight,
        None,   # no bias
        w1_scales,
        None,   # no activation scales (fp16 input)
        None,   # no global scale
        w1_qzeros if w1_qzeros.numel() > 0 else None,
        w1_g_idx if w1_g_idx.numel() > 0 else None,
        w1_perm if w1_perm.numel() > 0 else None,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights_f32,
        moe_block_size=moe_block_size,
        top_k=topk,
        mul_topk_weights=False,   # apply weights during W2 instead
        b_q_type=b_q_type,
        size_m=M,
        size_n=2 * I,  # gate + up combined
        size_k=K,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    # ── Gated activation: cache_w1 (M*topk, 2*I) → cache_act (M*topk, I) ─
    # Prefer vLLM's fused silu_and_mul / gelu_and_mul which fuse the gate/up
    # split, activation, and element-wise multiply into a single CUDA kernel,
    # writing directly into the pre-allocated cache_act buffer.
    act_str = activation.lower()
    if act_str == "silu" and _silu_and_mul_fn is not None:
        _silu_and_mul_fn(cache_act, cache_w1)
    elif act_str == "gelu" and _gelu_and_mul_fn is not None:
        _gelu_and_mul_fn(cache_act, cache_w1)
    else:
        # Pure-PyTorch fallback (silu/gelu/relu without fused CUDA op).
        # Use torch.mul(..., out=cache_act) to write the final result directly
        # into the pre-allocated buffer, avoiding an extra intermediate tensor.
        gate = cache_w1[:, :I]
        up = cache_w1[:, I:]
        if act_str == "silu":
            torch.mul(F.silu(gate), up, out=cache_act)
        elif act_str == "gelu":
            torch.mul(F.gelu(gate, approximate="tanh"), up, out=cache_act)
        else:  # relu
            torch.mul(F.relu(gate), up, out=cache_act)

    # ── W2: down projection → cache_w2 (M*topk, K) ───────────────────────
    # cache_w2 reuses the same flat buffer as cache_w1 since cache_w1 data
    # is no longer needed after the activation step.
    _vllm_moe_wna16_marlin_gemm(
        cache_act,
        cache_w2,  # write directly into pre-allocated buffer (reuses cache13)
        w2_qweight,
        None,   # no bias
        w2_scales,
        None,   # no activation scales
        None,   # no global scale
        w2_qzeros if w2_qzeros.numel() > 0 else None,
        w2_g_idx if w2_g_idx.numel() > 0 else None,
        w2_perm if w2_perm.numel() > 0 else None,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights_f32,
        moe_block_size=moe_block_size,
        top_k=1,   # each intermediate slot is an independent token
        mul_topk_weights=True,   # multiply routing weight into W2 output
        b_q_type=b_q_type,
        size_m=M_topk,
        size_n=K,
        size_k=I,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    # ── Reduce over topk: weighted sum → (M, K) ───────────────────────────
    # cache_w2[m * topk + k] already carries routing_weight[m, k] * W2_result.
    # Sum over the topk dimension directly into the pre-allocated output buffer.
    if out is None:
        out = torch.empty((M, K), device=device, dtype=dtype)
    torch.sum(cache_w2.view(M, topk, K), dim=1, out=out)

    return out
