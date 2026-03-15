"""
Int4 MoE Triton kernels with sorted-token grouped GEMM and tensor-core acceleration.

This implementation follows the approach used by vLLM's fused_moe_kernel_gptq_awq:
  1. Tokens are sorted by expert (moe_align_block_size) so that multiple tokens
     going to the same expert are processed as a single batched GEMM tile.
  2. 2D-tiled GEMM using tl.dot provides tensor-core acceleration and much better
     memory-access coalescing than the old per-token-per-block approach.
  3. Gated activation (silu/gelu/relu) is fused into the W1 kernel.
  4. W2 writes to a padded intermediate buffer with regular stores; the routing-
     weighted reduction back to the (M, H) output is done with a single
     scatter_add in Python.

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

For W1 the kernel simultaneously computes gate = W_gate @ x and up = W_up @ x,
then fuses the gated SiLU/GELU/ReLU in-kernel.
For W2 the kernel writes un-reduced expert outputs to a padded buffer; the
caller multiplies by routing weights and scatter_add-reduces to the final output.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Token-sorting helper
# ---------------------------------------------------------------------------


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple:
    """Sort tokens by expert and pad to multiples of *block_size*.

    Parameters
    ----------
    topk_ids    : (M, topk) int32/int64 – expert assignments per token
    block_size  : GEMM tile height (BLOCK_M)
    num_experts : total number of experts E

    Returns
    -------
    sorted_token_ids : (total_padded,) int32
        Original token index in [0, M) for each padded row.
        Padding rows that have no real token are set to M (sentinel).
    expert_ids : (total_padded // block_size,) int32
        Expert index for each GEMM block of *block_size* rows.
    num_tokens_post_padded : (1,) int32 tensor
        Scalar: total number of padded rows across all experts.
    sort_order : (M * topk,) int64
        Permutation that sorts the flat topk_ids array by expert.
        Needed by the caller to correctly order topk_weights with the same
        per-expert padding layout used for sorted_token_ids.
    """
    M, topk = topk_ids.shape
    device = topk_ids.device

    flat_ids = topk_ids.flatten().long()  # (M * topk,)
    sort_order = torch.argsort(flat_ids, stable=True)  # (M * topk,)

    expert_counts = torch.bincount(flat_ids.int(), minlength=num_experts)  # (E,)
    padded_counts = ((expert_counts + block_size - 1) // block_size) * block_size  # (E,)
    total_padded = int(padded_counts.sum().item())

    # sentinel M marks padding rows (no valid token)
    sorted_token_ids = torch.full((total_padded,), M, device=device, dtype=torch.int32)
    expert_ids_out = torch.empty(total_padded // block_size, device=device, dtype=torch.int32)

    offset = 0
    valid_offset = 0
    for e in range(num_experts):
        count = int(expert_counts[e].item())
        padded = int(padded_counts[e].item())
        if count > 0:
            # sort_order[valid_offset:valid_offset+count] are flat indices in [0,M*topk).
            # Dividing by topk gives the original token index in [0, M).
            token_indices = (sort_order[valid_offset : valid_offset + count] // topk).to(torch.int32)
            sorted_token_ids[offset : offset + count] = token_indices
        num_blocks_e = padded // block_size
        expert_ids_out[offset // block_size : offset // block_size + num_blocks_e] = e
        offset += padded
        valid_offset += count

    return (
        sorted_token_ids,
        expert_ids_out,
        torch.tensor([total_padded], device=device, dtype=torch.int32),
        sort_order,
    )


# ---------------------------------------------------------------------------
# W1 kernel: int4 grouped GEMM + fused gated activation
# ---------------------------------------------------------------------------


@triton.jit
def _w1_int4_grouped_kernel(
    # ── inputs ──────────────────────────────────────────────────────────────
    X_ptr,  # (M, H)               fp16/bf16
    Qw_ptr,  # (E, H//8, 2I)  or  (E, H, 2I//8)  int32
    Sc_ptr,  # (E, H//gs, 2I)     fp16
    Qz_ptr,  # (E, H//gs, 2I//8)  int32
    Y_ptr,  # (total_padded, I)  fp16/bf16  output after activation
    sorted_token_ids_ptr,  # (total_padded,)          int32
    expert_ids_ptr,  # (total_padded // BLOCK_M,) int32
    num_tokens_post_padded_ptr,  # (1,)                    int32
    # ── strides ─────────────────────────────────────────────────────────────
    sx_m,
    sx_k,  # X
    sq_e,
    sq_r,
    sq_c,  # Qw
    ss_e,
    ss_r,
    ss_c,  # Sc
    sz_e,
    sz_r,
    sz_c,  # Qz
    sy_m,
    sy_n,  # Y
    # ── problem dims ────────────────────────────────────────────────────────
    H: tl.constexpr,  # input hidden size
    I: tl.constexpr,  # per-stream intermediate size (W1 out = 2*I)
    group_size: tl.constexpr,  # quantisation group size
    M,  # actual number of tokens (dynamic – no recompile)
    # ── flags ────────────────────────────────────────────────────────────────
    KPACKED: tl.constexpr,  # True = GPTQ/AutoRound, False = AWQ
    ACTIVATION: tl.constexpr,  # 0 = SiLU, 1 = GELU, 2 = ReLU
    # ── tile sizes ───────────────────────────────────────────────────────────
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,  # L2-reuse grouping along M
    # ── packing ─────────────────────────────────────────────────────────────
    KPACK: tl.constexpr,  # 8 int4 values per int32 (K-packed)
    NPACK: tl.constexpr,  # 8 int4 zero-points per int32 (N-packed zeros)
):
    """Grouped GEMM for W1 (gate+up) with fused gated activation.

    Grid shape: (num_blocks_m * num_blocks_n,)
    where  num_blocks_m = total_padded // BLOCK_M
           num_blocks_n = I // BLOCK_N   (output after activation has width I)

    Each program computes a (BLOCK_M, BLOCK_N) tile of the gate output AND
    the corresponding (BLOCK_M, BLOCK_N) tile of the up output, then fuses
    the gated activation before writing a single (BLOCK_M, BLOCK_N) tile to Y.
    """
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_M)
    num_pid_n = tl.cdiv(I, BLOCK_N)

    pid = tl.program_id(0)

    # Grouped block ordering for better L2 reuse (same as vLLM / standard Triton GEMM)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_m
    pid_n = (pid % num_pid_in_group) // group_m

    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    # Token indices for this M-block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_token_ids_ptr + offs_m).to(tl.int64)
    token_mask = token_ids < M  # mask out padding rows

    eid = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # N-range: gate columns [n0, n0+BLOCK_N); up columns [I+n0, I+n0+BLOCK_N)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < I

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop – iterate in BLOCK_K steps
    for k0 in range(0, tl.cdiv(H, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        # ── Load input tokens: (BLOCK_M, BLOCK_K) ────────────────────────────
        a = tl.load(
            X_ptr + token_ids[:, None] * sx_m + offs_k[None, :] * sx_k,
            mask=token_mask[:, None] & mask_k[None, :],
            other=0.0,
        )  # fp16 / bf16

        # ── Load & unpack int4 weights ────────────────────────────────────────
        sc_row = offs_k // group_size  # (BLOCK_K,) – scale-row index

        if KPACKED:
            # GPTQ layout: qweight[e, k//8, n]
            qw_row = offs_k // KPACK  # (BLOCK_K,) – int32 row
            k_shift = ((offs_k % KPACK) * 4).to(tl.int32)

            qw_gate = tl.load(
                Qw_ptr + eid * sq_e + qw_row[:, None] * sq_r + offs_n[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )  # (BLOCK_K, BLOCK_N) int32 – rows repeat every KPACK steps
            qw_up = tl.load(
                Qw_ptr + eid * sq_e + qw_row[:, None] * sq_r + (offs_n + I)[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )

            nib_gate = ((qw_gate >> k_shift[:, None]) & 0xF).to(tl.float32)
            nib_up = ((qw_up >> k_shift[:, None]) & 0xF).to(tl.float32)

        else:
            # AWQ layout: qweight[e, k, n//8]
            qw_col_g = offs_n // NPACK
            n_shift_g = ((offs_n % NPACK) * 4).to(tl.int32)
            qw_col_u = (offs_n + I) // NPACK
            n_shift_u = (((offs_n + I) % NPACK) * 4).to(tl.int32)

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

            nib_gate = ((qw_gate >> n_shift_g[None, :]) & 0xF).to(tl.float32)
            nib_up = ((qw_up >> n_shift_u[None, :]) & 0xF).to(tl.float32)

        # ── Scales (BLOCK_K, BLOCK_N) ─────────────────────────────────────────
        sc_gate = tl.load(
            Sc_ptr + eid * ss_e + sc_row[:, None] * ss_r + offs_n[None, :] * ss_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        sc_up = tl.load(
            Sc_ptr + eid * ss_e + sc_row[:, None] * ss_r + (offs_n + I)[None, :] * ss_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        # ── Zero-points (N-packed in both layouts) ────────────────────────────
        z_col_g = offs_n // NPACK
        z_shft_g = ((offs_n % NPACK) * 4).to(tl.int32)
        z_col_u = (offs_n + I) // NPACK
        z_shft_u = (((offs_n + I) % NPACK) * 4).to(tl.int32)

        qz_gate = tl.load(
            Qz_ptr + eid * sz_e + sc_row[:, None] * sz_r + z_col_g[None, :] * sz_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        )
        qz_up = tl.load(
            Qz_ptr + eid * sz_e + sc_row[:, None] * sz_r + z_col_u[None, :] * sz_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        )
        z_gate = ((qz_gate >> z_shft_g[None, :]) & 0xF).to(tl.float32)
        z_up = ((qz_up >> z_shft_u[None, :]) & 0xF).to(tl.float32)

        # ── Dequantize → fp16/bf16 ────────────────────────────────────────────
        b_gate = ((nib_gate - z_gate) * sc_gate).to(a.dtype)  # (BLOCK_K, BLOCK_N)
        b_up = ((nib_up - z_up) * sc_up).to(a.dtype)

        # ── Accumulate with tensor-core GEMM ─────────────────────────────────
        acc_gate = tl.dot(a, b_gate, acc=acc_gate, out_dtype=tl.float32)
        acc_up = tl.dot(a, b_up, acc=acc_up, out_dtype=tl.float32)

    # ── Gated activation (fp32 → output dtype) ────────────────────────────────
    if ACTIVATION == 0:  # SiLU (default for most MoE models)
        act = acc_gate * tl.sigmoid(acc_gate) * acc_up
    elif ACTIVATION == 1:  # GELU approximation
        c: tl.constexpr = 0.7978845608028654
        act = (
            0.5
            * acc_gate
            * (1.0 + tl.math.tanh(c * (acc_gate + 0.044715 * acc_gate * acc_gate * acc_gate)))
            * acc_up
        )
    else:  # ReLU
        act = tl.where(acc_gate > 0, acc_gate, 0.0) * acc_up

    # ── Store (BLOCK_M, BLOCK_N) tile to the padded intermediate ─────────────
    offs_out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    y_ptrs = Y_ptr + offs_out_m[:, None] * sy_m + offs_n[None, :] * sy_n
    tl.store(y_ptrs, act.to(X_ptr.dtype.element_ty), mask=token_mask[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# W2 kernel: int4 grouped GEMM (down projection – no activation, no reduce)
# ---------------------------------------------------------------------------


@triton.jit
def _w2_int4_grouped_kernel(
    # ── inputs ──────────────────────────────────────────────────────────────
    X_ptr,  # (total_padded, I)  fp16/bf16  – intermediate from W1
    Qw_ptr,  # (E, I//8, H)  or  (E, I, H//8)  int32
    Sc_ptr,  # (E, I//gs, H)    fp16
    Qz_ptr,  # (E, I//gs, H//8) int32
    Y_ptr,  # (total_padded, H) fp16/bf16  – output (regular stores, no reduce)
    sorted_token_ids_ptr,  # (total_padded,)          int32  (for token_mask only)
    expert_ids_ptr,  # (total_padded // BLOCK_M,) int32
    num_tokens_post_padded_ptr,  # (1,)                    int32
    # ── strides ─────────────────────────────────────────────────────────────
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
    # ── dims ─────────────────────────────────────────────────────────────────
    I: tl.constexpr,  # intermediate size
    H: tl.constexpr,  # output hidden size
    group_size: tl.constexpr,
    M,  # actual number of tokens (dynamic)
    # ── flags ────────────────────────────────────────────────────────────────
    KPACKED: tl.constexpr,
    # ── tile sizes ───────────────────────────────────────────────────────────
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    KPACK: tl.constexpr,
    NPACK: tl.constexpr,
):
    """Grouped GEMM for W2 (down projection).

    Grid shape: (num_blocks_m * num_blocks_n,)
    where  num_blocks_m = total_padded // BLOCK_M
           num_blocks_n = H // BLOCK_N

    Each program writes a (BLOCK_M, BLOCK_N) tile to Y; the routing-weight
    multiplication and reduction are performed by the caller in Python.
    """
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_M)
    num_pid_n = tl.cdiv(H, BLOCK_N)

    pid = tl.program_id(0)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
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

        # Input from W1 intermediate (indexed by padded position, not token_id)
        a = tl.load(
            X_ptr + offs_m[:, None] * sx_m + offs_k[None, :] * sx_k,
            mask=token_mask[:, None] & mask_k[None, :],
            other=0.0,
        )

        sc_row = offs_k // group_size

        if KPACKED:
            qw_row = offs_k // KPACK
            k_shift = ((offs_k % KPACK) * 4).to(tl.int32)

            qw = tl.load(
                Qw_ptr + eid * sq_e + qw_row[:, None] * sq_r + offs_n[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )
            nib = ((qw >> k_shift[:, None]) & 0xF).to(tl.float32)

        else:
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

        b = ((nib - z) * sc).to(a.dtype)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)

    y_ptrs = Y_ptr + offs_m[:, None] * sy_m + offs_n[None, :] * sy_n
    tl.store(y_ptrs, acc.to(X_ptr.dtype.element_ty), mask=token_mask[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

_ACTIVATION_MAP = {"silu": 0, "gelu": 1, "relu": 2}

# BLOCK_M selection: pick the smallest block that keeps padding overhead low.
# Mirrors the heuristic used by vLLM (fused_marlin_moe.py).
_BLOCK_M_CANDIDATES = [16, 32, 64]


def _select_block_m(M: int, topk: int, num_experts: int) -> int:
    """Choose BLOCK_M so that average valid rows per block ≥ 0.9."""
    avg_tokens_per_expert = M * topk / max(num_experts, 1)
    for bm in _BLOCK_M_CANDIDATES:
        if avg_tokens_per_expert / bm < 0.9:
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
    """MoE forward pass with int4-quantised weights.

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
    E = w1_qweight.shape[0]
    topk = topk_ids.shape[1]

    # Derive I (intermediate size) from stacked weight shape
    if kpacked:
        double_I = w1_qweight.shape[2]  # (E, H//8, 2I) → col dim = 2I
    else:
        double_I = w1_qweight.shape[2] * 8  # (E, H, 2I//8) → col dim = 2I//8
    I = double_I // 2  # noqa: E741

    device = hidden_states.device
    dtype = hidden_states.dtype
    act_code = _ACTIVATION_MAP.get(activation.lower(), 0)

    # ── Sort tokens by expert and compute padded block layout ─────────────
    block_m = _select_block_m(M, topk, E)
    sorted_token_ids, expert_ids, num_tokens_pt, sort_order = moe_align_block_size(topk_ids, block_m, E)
    total_padded = int(num_tokens_pt.item())

    # ── W1: gate+up projection + activation → intermediate ────────────────
    # Output shape: (total_padded, I) – one row per padded (token, expert) slot
    intermediate = torch.empty((total_padded, I), device=device, dtype=dtype)

    BLOCK_N = 64
    BLOCK_K = 32  # must be a multiple of KPACK (8); 32 gives 4 int32s per K-step
    GROUP_M = 8

    num_blocks_m = triton.cdiv(total_padded, block_m)
    num_blocks_n_w1 = triton.cdiv(I, BLOCK_N)

    _w1_int4_grouped_kernel[(num_blocks_m * num_blocks_n_w1,)](
        hidden_states,
        w1_qweight,
        w1_scales,
        w1_qzeros,
        intermediate,
        sorted_token_ids,
        expert_ids,
        num_tokens_pt,
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
        M=M,
        KPACKED=kpacked,
        ACTIVATION=act_code,
        BLOCK_M=block_m,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        KPACK=8,
        NPACK=8,
    )

    # ── W2: down projection → padded output (no routing-weight reduction) ─
    # Output shape: (total_padded, H) – routing weights applied after
    w2_output = torch.empty((total_padded, H), device=device, dtype=dtype)

    num_blocks_n_w2 = triton.cdiv(H, BLOCK_N)

    _w2_int4_grouped_kernel[(num_blocks_m * num_blocks_n_w2,)](
        intermediate,
        w2_qweight,
        w2_scales,
        w2_qzeros,
        w2_output,
        sorted_token_ids,
        expert_ids,
        num_tokens_pt,
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
        w2_output.stride(0),
        w2_output.stride(1),
        # dims
        I=I,
        H=H,
        group_size=group_size,
        M=M,
        KPACKED=kpacked,
        BLOCK_M=block_m,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        KPACK=8,
        NPACK=8,
    )

    # ── Routing-weighted scatter-add reduce ───────────────────────────────
    # For each padded position p:
    #   final_output[sorted_token_ids[p]] += topk_weights[p] * w2_output[p]
    # Padding rows (sorted_token_ids[p] == M) contribute nothing.

    # Build sorted_weights with the same padded layout as sorted_token_ids.
    # sort_order[i] is the flat index in [0, M*topk) of the i-th valid pair
    # after sorting by expert.  We must insert per-expert padding gaps so that
    # sorted_weights[p] aligns with sorted_token_ids[p].
    flat_weights = topk_weights.flatten()  # (M * topk,)
    expert_counts = torch.bincount(topk_ids.flatten().int(), minlength=E)
    padded_counts = ((expert_counts + block_m - 1) // block_m) * block_m

    sorted_weights = torch.zeros(total_padded, device=device, dtype=dtype)
    valid_offset = 0
    offset = 0
    for e in range(E):
        count = int(expert_counts[e].item())
        padded = int(padded_counts[e].item())
        if count > 0:
            sorted_weights[offset : offset + count] = flat_weights[sort_order[valid_offset : valid_offset + count]]
        offset += padded
        valid_offset += count

    # Weighted output: (total_padded, H)
    weighted_output = w2_output * sorted_weights[:, None]

    # Scatter-add into (M, H)
    final_output = torch.zeros((M, H), device=device, dtype=dtype)
    valid_mask = sorted_token_ids < M  # (total_padded,)
    valid_token_ids = sorted_token_ids.clamp(max=M - 1).long()  # (total_padded,)
    idx = valid_token_ids[:, None].expand(-1, H)  # (total_padded, H)
    src = weighted_output * valid_mask[:, None]  # zero out padding rows
    final_output.scatter_add_(0, idx, src)

    return final_output
