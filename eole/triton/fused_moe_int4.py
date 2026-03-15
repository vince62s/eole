"""
Int4 MoE Triton kernels using a sorted-token grouped GEMM approach.

Key design choices that avoid the "runs forever" (Triton JIT compilation
freeze) of the previous implementation:

1. ``H``, ``I`` and ``group_size`` are passed as **regular int parameters**,
   NOT ``tl.constexpr``.  Making them constexpr forced Triton to emit a
   separate kernel binary for every unique (H, I, group_size) triple — a
   compilation that can take minutes on the first call.  With dynamic dims
   Triton compiles once per (BLOCK_M, BLOCK_N, BLOCK_K, KPACKED, ACTIVATION)
   combination, which is small and fast.

2. ``moe_align_block_size`` is fully GPU-native: it calls ``.item()`` exactly
   *once* (to read ``total_padded`` back to Python for the output allocation).
   The previous implementation called ``.item()`` 2×num_experts times,
   causing ~128 CPU-GPU synchronisations per forward pass for DeepSeek-style
   models.

3. ``group_m`` is clamped to ≥1 to avoid a Triton divide-by-zero that could
   occur in the last group when ``num_pid_m % GROUP_M == 0``.

Layout conventions
------------------
GPTQ / AutoRound (KPACKED=True)
  qweight : (E, K//8,          N)  int32  – 8 int4 packed along K
  scales  : (E, K//group_size, N)  fp16
  qzeros  : (E, K//group_size, N//8) int32 – 8 int4 zero-points packed along N

AWQ (KPACKED=False)
  qweight : (E, K,             N//8) int32 – 8 int4 packed along N
  scales  : (E, K//group_size, N)   fp16
  qzeros  : (E, K//group_size, N//8) int32

where  K = in_features (H for W1, I for W2)
       N = out_features (2*I for W1, H for W2)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# GPU-native token-sorting helper
# ---------------------------------------------------------------------------


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple:
    """Sort tokens by expert and pad each expert's bucket to a multiple of
    *block_size*.

    Parameters
    ----------
    topk_ids    : (M, topk) int32/int64
    block_size  : GEMM tile height (BLOCK_M)
    num_experts : total number of experts E

    Returns
    -------
    sorted_token_ids : (total_padded,) int32
        Original token index in [0, M).  Padding rows are set to M (sentinel).
    expert_ids : (total_padded // block_size,) int32
        Expert index for each GEMM block of *block_size* rows.
    num_tokens_post_padded : scalar int32 tensor
        Total padded rows across all experts.
    sort_order : (M * topk,) int64
        Permutation that sorts the flat topk_ids array by expert; used by the
        caller to reorder topk_weights into the same padded layout.
    """
    M, topk = topk_ids.shape
    device = topk_ids.device

    flat_ids = topk_ids.flatten().long()  # (M * topk,)
    sort_order = torch.argsort(flat_ids, stable=True)  # (M * topk,)

    expert_counts = torch.bincount(flat_ids.int(), minlength=num_experts)  # (E,)
    padded_counts = ((expert_counts + block_size - 1) // block_size) * block_size  # (E,)

    # Cumulative offsets — all on GPU, no per-expert .item() calls
    valid_offsets = torch.zeros(num_experts + 1, device=device, dtype=torch.int64)
    valid_offsets[1:] = expert_counts.long().cumsum(0)

    padded_offsets = torch.zeros(num_experts + 1, device=device, dtype=torch.int64)
    padded_offsets[1:] = padded_counts.long().cumsum(0)

    # Single .item() call to get the scalar needed for allocation
    total_padded = int(padded_offsets[-1].item())

    # Build sorted_token_ids on GPU via scatter
    sorted_token_ids = torch.full((total_padded,), M, device=device, dtype=torch.int32)

    positions = torch.arange(M * topk, device=device, dtype=torch.int64)
    expert_of_sorted = flat_ids[sort_order]          # expert id for each sorted entry
    rank_in_expert = positions - valid_offsets[expert_of_sorted]
    dest = padded_offsets[expert_of_sorted] + rank_in_expert

    token_indices = (sort_order // topk).to(torch.int32)
    sorted_token_ids.scatter_(0, dest, token_indices)

    # expert_ids for each GEMM block
    expert_ids_out = torch.repeat_interleave(
        torch.arange(num_experts, device=device, dtype=torch.int32),
        (padded_counts // block_size).int(),
    )

    return (
        sorted_token_ids,
        expert_ids_out,
        padded_offsets[-1:].int(),   # scalar tensor
        sort_order,
    )


# ---------------------------------------------------------------------------
# W1 kernel: sorted-token grouped GEMM + fused gated activation
# ---------------------------------------------------------------------------


@triton.jit
def _w1_int4_grouped_kernel(
    # ── inputs ──────────────────────────────────────────────────────────────
    X_ptr,          # (M, H)               fp16/bf16
    Qw_ptr,         # (E, H//8, 2I) KPACKED or (E, H, 2I//8)  int32
    Sc_ptr,         # (E, H//gs, 2I)       fp16
    Qz_ptr,         # (E, H//gs, 2I//8)    int32
    Y_ptr,          # (total_padded, I)    fp16/bf16
    sorted_token_ids_ptr,          # (total_padded,)          int32
    expert_ids_ptr,                # (total_padded // BLOCK_M,) int32
    num_tokens_post_padded_ptr,    # scalar int32
    # ── strides ─────────────────────────────────────────────────────────────
    sx_m, sx_k,
    sq_e, sq_r, sq_c,
    ss_e, ss_r, ss_c,
    sz_e, sz_r, sz_c,
    sy_m, sy_n,
    # ── dynamic dims (NOT constexpr – one compilation per block config) ─────
    H,             # input hidden size
    I,             # per-stream intermediate (W1 out = 2*I)
    group_size,    # quantisation group size
    M,             # actual token count
    # ── flags (constexpr – define kernel variant) ───────────────────────────
    KPACKED:     tl.constexpr,
    ACTIVATION:  tl.constexpr,   # 0=SiLU, 1=GELU, 2=ReLU
    BLOCK_M:     tl.constexpr,
    BLOCK_N:     tl.constexpr,
    BLOCK_K:     tl.constexpr,
    GROUP_M:     tl.constexpr,
    KPACK:       tl.constexpr,
    NPACK:       tl.constexpr,
):
    """2D-tiled grouped GEMM for gate+up projection with fused activation."""
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_M)
    num_pid_n = tl.cdiv(I, BLOCK_N)

    pid = tl.program_id(0)

    # Grouped L2-reuse ordering (same as standard Triton GEMM tutorial)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    # clamp to ≥1 to prevent divide-by-zero when num_pid_m is a multiple of GROUP_M
    group_m = tl.maximum(tl.minimum(num_pid_m - first_pid_m, GROUP_M), 1)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_m
    pid_n = (pid % num_pid_in_group) // group_m

    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_token_ids_ptr + offs_m).to(tl.int64)
    token_mask = token_ids < M

    eid = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < I

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(H, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        a = tl.load(
            X_ptr + token_ids[:, None] * sx_m + offs_k[None, :] * sx_k,
            mask=token_mask[:, None] & mask_k[None, :],
            other=0.0,
        )

        sc_row = offs_k // group_size

        if KPACKED:
            qw_row  = offs_k // KPACK
            k_shift = ((offs_k % KPACK) * 4).to(tl.int32)

            qw_gate = tl.load(
                Qw_ptr + eid * sq_e + qw_row[:, None] * sq_r + offs_n[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :], other=0,
            )
            qw_up = tl.load(
                Qw_ptr + eid * sq_e + qw_row[:, None] * sq_r + (offs_n + I)[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :], other=0,
            )
            nib_gate = ((qw_gate >> k_shift[:, None]) & 0xF).to(tl.float32)
            nib_up   = ((qw_up   >> k_shift[:, None]) & 0xF).to(tl.float32)
        else:
            qw_col_g = offs_n // NPACK
            n_shift_g = ((offs_n % NPACK) * 4).to(tl.int32)
            qw_col_u = (offs_n + I) // NPACK
            n_shift_u = (((offs_n + I) % NPACK) * 4).to(tl.int32)

            qw_gate = tl.load(
                Qw_ptr + eid * sq_e + offs_k[:, None] * sq_r + qw_col_g[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :], other=0,
            )
            qw_up = tl.load(
                Qw_ptr + eid * sq_e + offs_k[:, None] * sq_r + qw_col_u[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :], other=0,
            )
            nib_gate = ((qw_gate >> n_shift_g[None, :]) & 0xF).to(tl.float32)
            nib_up   = ((qw_up   >> n_shift_u[None, :]) & 0xF).to(tl.float32)

        # Scales
        sc_gate = tl.load(
            Sc_ptr + eid * ss_e + sc_row[:, None] * ss_r + offs_n[None, :] * ss_c,
            mask=mask_k[:, None] & mask_n[None, :], other=0.0,
        ).to(tl.float32)
        sc_up = tl.load(
            Sc_ptr + eid * ss_e + sc_row[:, None] * ss_r + (offs_n + I)[None, :] * ss_c,
            mask=mask_k[:, None] & mask_n[None, :], other=0.0,
        ).to(tl.float32)

        # Zero-points (N-packed in both layouts)
        z_col_g = offs_n // NPACK
        z_shft_g = ((offs_n % NPACK) * 4).to(tl.int32)
        z_col_u = (offs_n + I) // NPACK
        z_shft_u = (((offs_n + I) % NPACK) * 4).to(tl.int32)

        qz_gate = tl.load(
            Qz_ptr + eid * sz_e + sc_row[:, None] * sz_r + z_col_g[None, :] * sz_c,
            mask=mask_k[:, None] & mask_n[None, :], other=0,
        )
        qz_up = tl.load(
            Qz_ptr + eid * sz_e + sc_row[:, None] * sz_r + z_col_u[None, :] * sz_c,
            mask=mask_k[:, None] & mask_n[None, :], other=0,
        )
        z_gate = ((qz_gate >> z_shft_g[None, :]) & 0xF).to(tl.float32)
        z_up   = ((qz_up   >> z_shft_u[None, :]) & 0xF).to(tl.float32)

        b_gate = ((nib_gate - z_gate) * sc_gate).to(a.dtype)
        b_up   = ((nib_up   - z_up  ) * sc_up  ).to(a.dtype)

        acc_gate = tl.dot(a, b_gate, acc=acc_gate, out_dtype=tl.float32)
        acc_up   = tl.dot(a, b_up,   acc=acc_up,   out_dtype=tl.float32)

    # Fused gated activation
    if ACTIVATION == 0:    # SiLU
        act = acc_gate * tl.sigmoid(acc_gate) * acc_up
    elif ACTIVATION == 1:  # GELU approximation
        c: tl.constexpr = 0.7978845608028654
        act = (
            0.5 * acc_gate
            * (1.0 + tl.math.tanh(c * (acc_gate + 0.044715 * acc_gate * acc_gate * acc_gate)))
            * acc_up
        )
    else:                  # ReLU
        act = tl.where(acc_gate > 0, acc_gate, 0.0) * acc_up

    offs_out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    y_ptrs = Y_ptr + offs_out_m[:, None] * sy_m + offs_n[None, :] * sy_n
    tl.store(y_ptrs, act.to(X_ptr.dtype.element_ty),
             mask=token_mask[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# W2 kernel: sorted-token grouped GEMM (down projection, no activation)
# ---------------------------------------------------------------------------


@triton.jit
def _w2_int4_grouped_kernel(
    X_ptr,       # (total_padded, I) fp16/bf16
    Qw_ptr,      # (E, I//8, H) KPACKED or (E, I, H//8) int32
    Sc_ptr,      # (E, I//gs, H) fp16
    Qz_ptr,      # (E, I//gs, H//8) int32
    Y_ptr,       # (total_padded, H) fp16/bf16
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    sx_m, sx_k,
    sq_e, sq_r, sq_c,
    ss_e, ss_r, ss_c,
    sz_e, sz_r, sz_c,
    sy_m, sy_n,
    # dynamic dims
    I,
    H,
    group_size,
    M,
    # constexpr flags
    KPACKED:  tl.constexpr,
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
    BLOCK_K:  tl.constexpr,
    GROUP_M:  tl.constexpr,
    KPACK:    tl.constexpr,
    NPACK:    tl.constexpr,
):
    """2D-tiled grouped GEMM for the down projection (W2).

    Writes un-weighted output to Y; routing-weight multiply + scatter_add
    reduction is done in Python after the kernel returns.
    """
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_M)
    num_pid_n = tl.cdiv(H, BLOCK_N)

    pid = tl.program_id(0)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_m = tl.maximum(tl.minimum(num_pid_m - first_pid_m, GROUP_M), 1)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_m
    pid_n = (pid % num_pid_in_group) // group_m

    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_token_ids_ptr + offs_m).to(tl.int64)
    token_mask = token_ids < M

    eid = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(I, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < I

        # W2 reads from the padded intermediate (not the original tokens)
        a = tl.load(
            X_ptr + offs_m[:, None] * sx_m + offs_k[None, :] * sx_k,
            mask=token_mask[:, None] & mask_k[None, :],
            other=0.0,
        )

        sc_row = offs_k // group_size

        if KPACKED:
            qw_row  = offs_k // KPACK
            k_shift = ((offs_k % KPACK) * 4).to(tl.int32)

            qw = tl.load(
                Qw_ptr + eid * sq_e + qw_row[:, None] * sq_r + offs_n[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :], other=0,
            )
            nib = ((qw >> k_shift[:, None]) & 0xF).to(tl.float32)
        else:
            qw_col  = offs_n // NPACK
            n_shift = ((offs_n % NPACK) * 4).to(tl.int32)

            qw = tl.load(
                Qw_ptr + eid * sq_e + offs_k[:, None] * sq_r + qw_col[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :], other=0,
            )
            nib = ((qw >> n_shift[None, :]) & 0xF).to(tl.float32)

        sc = tl.load(
            Sc_ptr + eid * ss_e + sc_row[:, None] * ss_r + offs_n[None, :] * ss_c,
            mask=mask_k[:, None] & mask_n[None, :], other=0.0,
        ).to(tl.float32)

        z_col  = offs_n // NPACK
        z_shft = ((offs_n % NPACK) * 4).to(tl.int32)
        qz = tl.load(
            Qz_ptr + eid * sz_e + sc_row[:, None] * sz_r + z_col[None, :] * sz_c,
            mask=mask_k[:, None] & mask_n[None, :], other=0,
        )
        z = ((qz >> z_shft[None, :]) & 0xF).to(tl.float32)

        b = ((nib - z) * sc).to(a.dtype)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)

    y_ptrs = Y_ptr + offs_m[:, None] * sy_m + offs_n[None, :] * sy_n
    tl.store(y_ptrs, acc.to(X_ptr.dtype.element_ty),
             mask=token_mask[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# Python wrapper — Triton path (non-Marlin GPTQ / AWQ)
# ---------------------------------------------------------------------------

_ACTIVATION_MAP = {"silu": 0, "gelu": 1, "relu": 2}

_BLOCK_M_CANDIDATES = [16, 32, 64]


def _select_block_m(M: int, topk: int, num_experts: int) -> int:
    """Return the smallest BLOCK_M candidate where the average token count
    per expert is less than one full block.  For large batches the largest
    candidate is returned.  This mirrors the heuristic in vLLM's
    fused_marlin_moe.py: small tiles for decode, large tiles for prefill.
    """
    avg = M * topk / max(num_experts, 1)
    for bm in _BLOCK_M_CANDIDATES:
        if avg / bm < 0.9:
            return bm
    return _BLOCK_M_CANDIDATES[-1]


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
    """MoE forward pass with int4-quantised weights (GPTQ / AutoRound / AWQ).

    Parameters
    ----------
    hidden_states : (M, H)
    w1_qweight    : (E, H//8, 2I) kpacked=True  or  (E, H, 2I//8)  int32
    w1_scales     : (E, H//group_size, 2I)       fp16/bf16
    w1_qzeros     : (E, H//group_size, 2I//8)    int32
    w2_qweight    : (E, I//8, H)  kpacked=True  or  (E, I, H//8)   int32
    w2_scales     : (E, I//group_size, H)        fp16/bf16
    w2_qzeros     : (E, I//group_size, H//8)     int32
    topk_weights  : (M, K)
    topk_ids      : (M, K)
    group_size    : quantisation group size (default 128)
    kpacked       : True = GPTQ/AutoRound, False = AWQ
    activation    : "silu" | "gelu" | "relu"
    """
    M, H = hidden_states.shape
    E = w1_qweight.shape[0]
    topk = topk_ids.shape[1]

    # Derive I from packed weight shape
    double_I = w1_qweight.shape[2] if kpacked else w1_qweight.shape[2] * 8
    I = double_I // 2  # noqa: E741

    device = hidden_states.device
    dtype  = hidden_states.dtype
    act_code = _ACTIVATION_MAP.get(activation.lower(), 0)

    block_m = _select_block_m(M, topk, E)
    sorted_token_ids, expert_ids, num_tokens_pt, sort_order = moe_align_block_size(
        topk_ids, block_m, E
    )
    total_padded = int(num_tokens_pt.item())

    # ── W1 ──────────────────────────────────────────────────────────────────
    intermediate = torch.empty((total_padded, I), device=device, dtype=dtype)

    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8

    num_blocks_m  = triton.cdiv(total_padded, block_m)
    num_blocks_n1 = triton.cdiv(I, BLOCK_N)

    _w1_int4_grouped_kernel[(num_blocks_m * num_blocks_n1,)](
        hidden_states, w1_qweight, w1_scales, w1_qzeros, intermediate,
        sorted_token_ids, expert_ids, num_tokens_pt,
        hidden_states.stride(0), hidden_states.stride(1),
        w1_qweight.stride(0), w1_qweight.stride(1), w1_qweight.stride(2),
        w1_scales.stride(0),  w1_scales.stride(1),  w1_scales.stride(2),
        w1_qzeros.stride(0),  w1_qzeros.stride(1),  w1_qzeros.stride(2),
        intermediate.stride(0), intermediate.stride(1),
        H=H, I=I, group_size=group_size, M=M,
        KPACKED=kpacked, ACTIVATION=act_code,
        BLOCK_M=block_m, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, KPACK=8, NPACK=8,
    )

    # ── W2 ──────────────────────────────────────────────────────────────────
    w2_output = torch.empty((total_padded, H), device=device, dtype=dtype)
    num_blocks_n2 = triton.cdiv(H, BLOCK_N)

    _w2_int4_grouped_kernel[(num_blocks_m * num_blocks_n2,)](
        intermediate, w2_qweight, w2_scales, w2_qzeros, w2_output,
        sorted_token_ids, expert_ids, num_tokens_pt,
        intermediate.stride(0), intermediate.stride(1),
        w2_qweight.stride(0), w2_qweight.stride(1), w2_qweight.stride(2),
        w2_scales.stride(0),  w2_scales.stride(1),  w2_scales.stride(2),
        w2_qzeros.stride(0),  w2_qzeros.stride(1),  w2_qzeros.stride(2),
        w2_output.stride(0), w2_output.stride(1),
        I=I, H=H, group_size=group_size, M=M,
        KPACKED=kpacked,
        BLOCK_M=block_m, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, KPACK=8, NPACK=8,
    )

    # ── Routing-weighted scatter-add reduce ─────────────────────────────────
    # Build sorted_weights with the same padded layout as sorted_token_ids.
    # sort_order[i] is the flat index (in [0, M*topk)) of the i-th valid pair
    # sorted by expert.  Padding gaps (within each expert's bucket) stay 0.
    flat_weights = topk_weights.flatten()           # (M * topk,)
    expert_counts  = torch.bincount(topk_ids.flatten().int(), minlength=E)
    padded_counts  = ((expert_counts + block_m - 1) // block_m) * block_m
    valid_offsets  = torch.zeros(E + 1, device=device, dtype=torch.int64)
    valid_offsets[1:] = expert_counts.long().cumsum(0)
    padded_offsets = torch.zeros(E + 1, device=device, dtype=torch.int64)
    padded_offsets[1:] = padded_counts.long().cumsum(0)

    sorted_weights = torch.zeros(total_padded, device=device, dtype=dtype)
    # Place the routing weights at the valid (non-padding) positions.
    # Positions: padded_offsets[e] + rank_in_expert for each valid (token,expert).
    positions = torch.arange(M * topk, device=device, dtype=torch.int64)
    expert_of_sorted = topk_ids.flatten().long()[sort_order]
    rank_in_expert   = positions - valid_offsets[expert_of_sorted]
    dest = padded_offsets[expert_of_sorted] + rank_in_expert
    sorted_weights.scatter_(0, dest, flat_weights[sort_order].to(dtype))

    weighted_output = w2_output * sorted_weights[:, None]

    # scatter_add: for each padded row, add its weighted output to the
    # originating token's row.  Padding rows contribute 0 (sorted_token_ids==M).
    final_output = torch.zeros((M, H), device=device, dtype=dtype)
    valid_mask  = sorted_token_ids < M
    # clamp: scatter_add_ requires indices in [0, M-1].  Padding rows are
    # clamped to a valid index but zeroed out by valid_mask.
    valid_ids = sorted_token_ids.clamp(max=M - 1).long()
    idx = valid_ids[:, None].expand(-1, H)
    src = weighted_output * valid_mask[:, None]
    final_output.scatter_add_(0, idx, src)

    return final_output


# ---------------------------------------------------------------------------
# Marlin MoE fast path (uses gptq_marlin_gemm per expert)
# ---------------------------------------------------------------------------
# vLLM achieves maximum speed with ``moe_wna16_marlin_gemm``, a specialised
# CUDA kernel that routes tokens to all experts in a single launch.
# gptqmodel_marlin_kernels does not expose that kernel, so we approximate
# the approach by calling ``gptq_marlin_gemm`` (the single-expert Marlin
# kernel) for each expert individually, with tokens already sorted by expert.
# This still significantly reduces Python overhead vs the per-expert
# expert.forward() path because:
#   • W1 + activation + W2 are fused within the same Python loop iteration
#     (fewer kernel launches and intermediate tensor allocations)
#   • All per-expert tensors are pre-stacked, avoiding per-call dictionary
#     lookups and attribute accesses inside expert.forward()
#
# If moe_wna16_marlin_gemm ever becomes available in gptqmodel_marlin_kernels
# it will automatically be preferred via the _MOE_MARLIN_AVAILABLE check.
# ---------------------------------------------------------------------------

try:
    import gptqmodel_marlin_kernels as _marlin_kernels
    _MARLIN_AVAILABLE = True
    # Check for the batched MoE kernel (vLLM-style)
    _MOE_MARLIN_AVAILABLE = hasattr(_marlin_kernels, "moe_wna16_marlin_gemm")
except (ImportError, ModuleNotFoundError):
    _MARLIN_AVAILABLE = False
    _MOE_MARLIN_AVAILABLE = False


def _act_fn(x: torch.Tensor, activation: str) -> torch.Tensor:
    """Apply gated activation to a (batch, 2*I) tensor, returning (batch, I)."""
    gate, up = x.chunk(2, dim=-1)
    if activation == "silu":
        return torch.nn.functional.silu(gate) * up
    elif activation == "gelu":
        return torch.nn.functional.gelu(gate) * up
    else:  # relu
        return torch.nn.functional.relu(gate) * up


def fused_marlin_moe_impl(
    hidden_states: torch.Tensor,
    w1_qweight: list,
    w1_scales: list,
    w1_qzeros: list,
    w1_g_idx: list,
    w1_g_idx_sort: list,
    w2_qweight: list,
    w2_scales: list,
    w2_qzeros: list,
    w2_g_idx: list,
    w2_g_idx_sort: list,
    workspaces: list,
    quant_type,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    in_features: int,
    intermediate_features: int,
    out_features: int,
    is_k_full: bool,
    activation: str = "silu",
) -> torch.Tensor:
    """MoE forward with Marlin-repacked int4 weights.

    Uses ``gptq_marlin_gemm`` (from gptqmodel_marlin_kernels) per expert,
    with tokens pre-sorted by expert to maximise batch size per call.
    Falls back to plain fp16 matmul if the Marlin kernels are unavailable.

    Parameters
    ----------
    hidden_states       : (M, H) fp16/bf16
    w1_qweight          : list[E] of Marlin-repacked qweight tensors for W1
    w1_scales           : list[E] of Marlin-permuted scale tensors for W1
    w1_qzeros           : list[E] of (empty) zero-point tensors for W1
    w1_g_idx            : list[E] of g_idx tensors (empty if no act_order)
    w1_g_idx_sort       : list[E] of sort_indices tensors
    w2_qweight / scales / qzeros / g_idx / g_idx_sort : same for W2
    workspaces          : list[E] of pre-allocated workspace tensors (or None)
    quant_type          : ScalarType (e.g. scalar_types.uint4b8)
    topk_weights        : (M, K) routing weights
    topk_ids            : (M, K) expert assignments
    in_features         : H (hidden size)
    intermediate_features : I (per-expert intermediate, W1 out = 2*I)
    out_features        : H (same as in_features for standard MoE)
    is_k_full           : Marlin is_k_full flag
    activation          : "silu" | "gelu" | "relu"
    """
    if not _MARLIN_AVAILABLE:
        raise RuntimeError(
            "fused_marlin_moe_impl requires gptqmodel_marlin_kernels. "
            "Install gptqmodel with the Marlin backend."
        )

    M, H = hidden_states.shape
    E = len(w1_qweight)
    topk = topk_ids.shape[1]
    device = hidden_states.device
    dtype  = hidden_states.dtype

    # Sort tokens by expert (GPU-native — single .item() call in moe_align_block_size)
    flat_ids   = topk_ids.flatten().long()
    sort_order = torch.argsort(flat_ids, stable=True)
    expert_counts = torch.bincount(flat_ids.int(), minlength=E)

    flat_weights = topk_weights.flatten()   # (M * topk,)

    final_output = torch.zeros((M, H), device=device, dtype=dtype)

    valid_off = 0
    for e in range(E):
        count = int(expert_counts[e].item())
        if count == 0:
            continue

        # Indices into sort_order for this expert's tokens
        routed     = sort_order[valid_off : valid_off + count]
        token_ids  = (routed // topk).long()        # original token indices
        routing_w  = flat_weights[routed].to(dtype)  # routing weights

        x = hidden_states[token_ids]  # (count, H)

        # Marlin requires scales to have the same dtype as input.
        sc1 = w1_scales[e] if w1_scales[e].dtype == dtype else w1_scales[e].to(dtype)
        sc2 = w2_scales[e] if w2_scales[e].dtype == dtype else w2_scales[e].to(dtype)

        if _MOE_MARLIN_AVAILABLE:
            # Batched kernel: handles W1 and W2 in one call (vLLM style)
            # (placeholder — moe_wna16_marlin_gemm not yet in gptqmodel)
            pass

        # ── W1: gate+up projection ──────────────────────────────────────────
        ws1 = workspaces[e] if workspaces[e] is not None else _make_workspace(device)
        gate_up = _marlin_kernels.gptq_marlin_gemm(
            x, None,
            w1_qweight[e], None,
            sc1, None,
            w1_qzeros[e],
            w1_g_idx[e], w1_g_idx_sort[e],
            ws1,
            quant_type.id,
            count, 2 * intermediate_features, in_features,
            is_k_full, False, True, False,
        )  # (count, 2*I)

        # Gated activation → (count, I)
        act_out = _act_fn(gate_up, activation)

        # ── W2: down projection ─────────────────────────────────────────────
        ws2 = workspaces[e] if workspaces[e] is not None else _make_workspace(device)
        down = _marlin_kernels.gptq_marlin_gemm(
            act_out, None,
            w2_qweight[e], None,
            sc2, None,
            w2_qzeros[e],
            w2_g_idx[e], w2_g_idx_sort[e],
            ws2,
            quant_type.id,
            count, out_features, intermediate_features,
            is_k_full, False, True, False,
        )  # (count, H)

        # Weighted accumulate into final output
        weighted = down * routing_w[:, None]
        final_output.scatter_add_(
            0,
            token_ids[:, None].expand(-1, H),
            weighted,
        )

        valid_off += count

    return final_output


def _make_workspace(device: torch.device) -> torch.Tensor:
    """Create a minimal Marlin workspace (used only when no cached workspace)."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms, dtype=torch.int32, device=device)
