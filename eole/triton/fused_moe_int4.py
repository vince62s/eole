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

Key optimizations over the naive logical-K loop
------------------------------------------------
KPACKED (GPTQ/AutoRound) path
  A naive loop ``for k0 in range(0, K, BLOCK_K)`` over logical K with BLOCK_K=64
  computes ``qw_row = k // KPACK``.  Because KPACK=8, every 8 consecutive logical-K
  elements share the same packed row, so a (BLOCK_K=64, BLOCK_N) weight load fetches
  only (BLOCK_K/KPACK=8) unique rows – loading each row 8 times redundantly.  The
  same applies to scales: when BLOCK_K < group_size the tile belongs to one group
  yet a (BLOCK_K, BLOCK_N) load replicates the same scale row BLOCK_K times.

  The fix iterates directly over the PACKED K dimension (H//KPACK rows) with
  BLOCK_KP packed rows per tile.  Each packed row is loaded exactly ONCE.  Scales
  and zero-points are loaded as (BLOCK_N,) vectors (one per group tile).  The eight
  nibbles per packed row are extracted via a statically-unrolled
  ``tl.static_range(KPACK)`` inner loop; for each bit-position b the corresponding
  activation slice (BLOCK_KP elements at stride KPACK in x) is fetched with a small
  strided load – this avoids tl.reshape which is not available in Triton ≤2.1.

  Constraint: BLOCK_KP * KPACK must not exceed group_size so that every element in
  a tile belongs to the same quantisation group.  With the default BLOCK_KP=8 and
  KPACK=8, this requires group_size >= 64 (typical minimum is 128, so fine).

AWQ (N-packed) path
  The N-packed layout already accesses each (k, n//8) weight element once; no
  structural redundancy exists.  The logical-K loop is kept as-is.  Scales are
  loaded per-element because different logical-K values in a tile may belong to
  different groups.

Both paths
  * BLOCK_N (output channels per CTA tile) and BLOCK_KP (packed-K rows per tile)
    are explicit constexpr parameters – callers can tune them.  The defaults
    (BLOCK_N=64, BLOCK_KP=8) are chosen to fit typical GPU register files.
  * Token-expert pairs are pre-sorted by expert ID in the Python wrapper so that
    consecutive CTAs access the same expert's weights, improving L2 cache reuse.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Default tile sizes  (must satisfy BLOCK_KP * KPACK <= group_size)
# ---------------------------------------------------------------------------

_DEFAULT_BLOCK_N = 64  # output channels per CTA tile (power of 2)
_DEFAULT_BLOCK_KP = 8  # packed-K rows per tile; 8 * KPACK(=8) = 64 logical K

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
    # ── problem dims ────────────────────────────────────────────────────────
    H: tl.constexpr,  # input hidden size
    I: tl.constexpr,  # intermediate size (gate OR up; W1 width = 2*I)
    group_size: tl.constexpr,  # quantisation group size (e.g. 128)
    # ── layout / activation flags ───────────────────────────────────────────
    KPACKED: tl.constexpr,  # True = GPTQ/AutoRound, False = AWQ
    ACTIVATION: tl.constexpr,  # 0 = SiLU, 1 = GELU approx, 2 = ReLU
    # ── tile sizes ──────────────────────────────────────────────────────────
    BLOCK_N: tl.constexpr,  # output channels per CTA tile
    BLOCK_KP: tl.constexpr,  # packed-K rows per tile (KPACKED) / logical-K per tile (AWQ)
    # ── packing constants ────────────────────────────────────────────────────
    KPACK: tl.constexpr,  # 8  (int4: 8 values per int32, packed along K or N)
    NPACK: tl.constexpr,  # 8  (zero-points packed ×8 along N)
    # ── dynamic scalar ───────────────────────────────────────────────────────
    num_pairs,  # NOT constexpr – avoids kernel recompile when batch size changes
):
    pid_pair = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_pair >= num_pairs:
        return

    eid = tl.load(expert_ids_ptr + pid_pair).to(tl.int64)
    tid = tl.load(token_ids_ptr + pid_pair).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < I

    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base = X_ptr + tid * sx_m

    if KPACKED:
        # ── GPTQ / AutoRound: iterate over the PACKED K dimension ─────────────
        #
        # Each outer tile covers BLOCK_KP packed rows = BLOCK_KP * KPACK logical K
        # elements.  Packed weights and scales are loaded ONCE per tile; the
        # tl.static_range(KPACK) inner loop is unrolled at compile time so there
        # is zero runtime loop overhead.
        #
        # Tile constraint: BLOCK_KP * KPACK <= group_size  (all elements share
        # one scale group; guaranteed by default values and documented above).

        for kp0 in range(0, H // KPACK, BLOCK_KP):
            offs_kp = kp0 + tl.arange(0, BLOCK_KP)  # (BLOCK_KP,) packed-row indices
            mask_kp = offs_kp < H // KPACK

            # ── Load each packed weight row ONCE (BLOCK_KP rows, not BLOCK_K) ──
            qw_gate = tl.load(
                Qw_ptr + eid * sq_e + offs_kp[:, None] * sq_r + offs_n[None, :] * sq_c,
                mask=mask_kp[:, None] & mask_n[None, :],
                other=0,
            )  # (BLOCK_KP, BLOCK_N) int32
            qw_up = tl.load(
                Qw_ptr + eid * sq_e + offs_kp[:, None] * sq_r + (offs_n[None, :] + I) * sq_c,
                mask=mask_kp[:, None] & mask_n[None, :],
                other=0,
            )  # (BLOCK_KP, BLOCK_N) int32

            # ── Scales & zeros: one (BLOCK_N,) vector per group tile ───────────
            # All BLOCK_KP * KPACK logical K elements share the same scale group
            # (guaranteed by the BLOCK_KP * KPACK <= group_size constraint).
            sc_row = (kp0 * KPACK) // group_size  # scalar
            sc_gate = tl.load(
                Sc_ptr + eid * ss_e + sc_row * ss_r + offs_n * ss_c,
                mask=mask_n,
                other=1.0,
            ).to(tl.float32)  # (BLOCK_N,)
            sc_up = tl.load(
                Sc_ptr + eid * ss_e + sc_row * ss_r + (offs_n + I) * ss_c,
                mask=mask_n,
                other=1.0,
            ).to(tl.float32)  # (BLOCK_N,)

            z_col = offs_n // NPACK
            z_shft = ((offs_n % NPACK) * 4).to(tl.int32)
            z_col_u = (offs_n + I) // NPACK
            qz_gate = tl.load(
                Qz_ptr + eid * sz_e + sc_row * sz_r + z_col * sz_c,
                mask=mask_n,
                other=0,
            )  # (BLOCK_N,)
            qz_up = tl.load(
                Qz_ptr + eid * sz_e + sc_row * sz_r + z_col_u * sz_c,
                mask=mask_n,
                other=0,
            )  # (BLOCK_N,)
            z_gate = ((qz_gate >> z_shft) & 0xF).to(tl.float32)  # (BLOCK_N,)
            z_up = ((qz_up >> z_shft) & 0xF).to(tl.float32)  # (BLOCK_N,)

            # ── Statically-unrolled nibble extraction and accumulation ─────────
            # For each bit-position b, load the BLOCK_KP activation values
            # x[kp * KPACK + b] for kp in 0..BLOCK_KP-1 (strided by KPACK in x).
            # This avoids tl.reshape which is unavailable in Triton ≤2.1.
            for b in tl.static_range(KPACK):
                offs_kb = (kp0 + tl.arange(0, BLOCK_KP)) * KPACK + b
                x_b = tl.load(
                    x_base + offs_kb * sx_k, mask=offs_kb < H, other=0.0
                ).to(tl.float32)  # (BLOCK_KP,)
                gate_nib = ((qw_gate >> (b * 4)) & 0xF).to(tl.float32)  # (BLOCK_KP, BLOCK_N)
                up_nib = ((qw_up >> (b * 4)) & 0xF).to(tl.float32)
                w_gate = (gate_nib - z_gate[None, :]) * sc_gate[None, :]
                w_up = (up_nib - z_up[None, :]) * sc_up[None, :]
                acc_gate += tl.sum(x_b[:, None] * w_gate, axis=0)
                acc_up += tl.sum(x_b[:, None] * w_up, axis=0)

    else:
        # ── AWQ: iterate over logical K dimension (N-packed weights) ──────────
        # Each (k, n//8) weight element is already unique in this layout so no
        # structural redundancy exists.  Scales span multiple K values within a
        # group; they are loaded per-element as a (BLOCK_KP, BLOCK_N) tile.
        for k0 in range(0, H, BLOCK_KP):
            offs_k = k0 + tl.arange(0, BLOCK_KP)
            mask_k = offs_k < H

            x = tl.load(x_base + offs_k * sx_k, mask=mask_k, other=0.0).to(tl.float32)

            sc_row = offs_k // group_size  # (BLOCK_KP,)

            qw_col_g = offs_n // NPACK
            qw_col_u = (offs_n + I) // NPACK
            n_shift = ((offs_n % NPACK) * 4).to(tl.int32)

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

            z_col = offs_n // NPACK
            z_shft = ((offs_n % NPACK) * 4).to(tl.int32)
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

            acc_gate += tl.sum(x[:, None] * (gate_nib - z_gate) * sc_gate, axis=0)
            acc_up += tl.sum(x[:, None] * (up_nib - z_up) * sc_up, axis=0)

    # ── gated activation ─────────────────────────────────────────────────────
    if ACTIVATION == 0:  # SiLU (default for most MoE models)
        act = acc_gate * tl.sigmoid(acc_gate) * acc_up
    elif ACTIVATION == 1:  # GELU approximation
        sqrt_2_over_pi = 0.7978845608028654
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
    BLOCK_KP: tl.constexpr,
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

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < H

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_base = X_ptr + pid_pair * sx_m

    if KPACKED:
        # ── GPTQ / AutoRound: iterate over packed I dimension ─────────────────
        for kp0 in range(0, I // KPACK, BLOCK_KP):
            offs_kp = kp0 + tl.arange(0, BLOCK_KP)
            mask_kp = offs_kp < I // KPACK

            # Load each packed weight row exactly once
            qw = tl.load(
                Qw_ptr + eid * sq_e + offs_kp[:, None] * sq_r + offs_n[None, :] * sq_c,
                mask=mask_kp[:, None] & mask_n[None, :],
                other=0,
            )  # (BLOCK_KP, BLOCK_N) int32

            # Scales & zeros: one (BLOCK_N,) vector per group tile
            sc_row = (kp0 * KPACK) // group_size
            sc = tl.load(
                Sc_ptr + eid * ss_e + sc_row * ss_r + offs_n * ss_c,
                mask=mask_n,
                other=1.0,
            ).to(tl.float32)  # (BLOCK_N,)

            z_col = offs_n // NPACK
            z_shft = ((offs_n % NPACK) * 4).to(tl.int32)
            qz = tl.load(
                Qz_ptr + eid * sz_e + sc_row * sz_r + z_col * sz_c,
                mask=mask_n,
                other=0,
            )  # (BLOCK_N,)
            z = ((qz >> z_shft) & 0xF).to(tl.float32)  # (BLOCK_N,)

            # Statically-unrolled nibble extraction and accumulation
            for b in tl.static_range(KPACK):
                offs_kb = (kp0 + tl.arange(0, BLOCK_KP)) * KPACK + b
                x_b = tl.load(
                    x_base + offs_kb * sx_k, mask=offs_kb < I, other=0.0
                ).to(tl.float32)  # (BLOCK_KP,)
                nib = ((qw >> (b * 4)) & 0xF).to(tl.float32)  # (BLOCK_KP, BLOCK_N)
                w = (nib - z[None, :]) * sc[None, :]
                acc += tl.sum(x_b[:, None] * w, axis=0)

    else:
        # ── AWQ: iterate over logical I dimension ─────────────────────────────
        for k0 in range(0, I, BLOCK_KP):
            offs_k = k0 + tl.arange(0, BLOCK_KP)
            mask_k = offs_k < I

            x = tl.load(x_base + offs_k * sx_k, mask=mask_k, other=0.0).to(tl.float32)

            sc_row = offs_k // group_size
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

            acc += tl.sum(x[:, None] * (nib - z) * sc, axis=0)

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

    # ── Build token–expert index arrays ────────────────────────────────────
    expert_ids = topk_ids.flatten().to(torch.int32)
    token_ids = torch.arange(M, device=device, dtype=torch.int32).repeat_interleave(K)
    weights = topk_weights.flatten()

    # Sort pairs by expert ID so that consecutive CTAs access the same expert's
    # weight tensors, improving L2 cache reuse when multiple tokens share an expert.
    sort_idx = torch.argsort(expert_ids, stable=True)
    expert_ids = expert_ids[sort_idx]
    token_ids = token_ids[sort_idx]
    weights = weights[sort_idx]

    act_code = _ACTIVATION_MAP.get(activation.lower(), 0)

    BLOCK_N = _DEFAULT_BLOCK_N
    BLOCK_KP = _DEFAULT_BLOCK_KP

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
        hidden_states.stride(0),
        hidden_states.stride(1),
        w1_qweight.stride(0),
        w1_qweight.stride(1),
        w1_qweight.stride(2),
        w1_scales.stride(0),
        w1_scales.stride(1),
        w1_scales.stride(2),
        w1_qzeros.stride(0),
        w1_qzeros.stride(1),
        w1_qzeros.stride(2),
        intermediate.stride(0),
        intermediate.stride(1),
        H=H,
        I=I,
        group_size=group_size,
        KPACKED=kpacked,
        ACTIVATION=act_code,
        BLOCK_N=BLOCK_N,
        BLOCK_KP=BLOCK_KP,
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
        intermediate.stride(0),
        intermediate.stride(1),
        w2_qweight.stride(0),
        w2_qweight.stride(1),
        w2_qweight.stride(2),
        w2_scales.stride(0),
        w2_scales.stride(1),
        w2_scales.stride(2),
        w2_qzeros.stride(0),
        w2_qzeros.stride(1),
        w2_qzeros.stride(2),
        final_output.stride(0),
        final_output.stride(1),
        H=H,
        I=I,
        group_size=group_size,
        KPACKED=kpacked,
        BLOCK_N=BLOCK_N,
        BLOCK_KP=BLOCK_KP,
        KPACK=8,
        NPACK=8,
        num_pairs=num_pairs,
    )

    return final_output
