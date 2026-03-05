"""Triton kernels for int4-quantized Mixture of Experts (MoE).

Weight format: GPTQ / AutoRound int4
  - qweight:  (in_features // 8, out_features) int32
              Each int32 packs 8 int4 values along the input-feature (K) dimension.
              Bit layout: bits[3:0] = int4[0], bits[7:4] = int4[1], ...,
                          bits[31:28] = int4[7]
  - scales:   (in_features // group_size, out_features) fp16
  - qzeros:   (in_features // group_size, out_features // 8) int32
              Same packing scheme as qweight, but along the out_features dimension.
              Zero-point for output channel n in group g:
                zero = (qzeros[g, n // 8] >> ((n % 8) * 4)) & 0xF

Stacked (MoE) format – built once and cached by MoE.forward:
  - stacked_qw1:       (E, H // 8, 2*I) int32      W1 for all E experts
  - stacked_scales_w1: (E, H // group_size, 2*I) fp16
  - stacked_qzeros_w1: (E, H // group_size, 2*I // 8) int32
  - stacked_qw2:       (E, I // 8, H) int32         W2 for all E experts
  - stacked_scales_w2: (E, I // group_size, H) fp16
  - stacked_qzeros_w2: (E, I // group_size, H // 8) int32

Key efficiency gains over per-expert TritonLinear calls:
  - Single fused kernel per pass instead of 2 × num_experts × num_calls launches
  - On-the-fly int4 → fp32 dequantization inside each Triton CTA, so the full
    fp16 weight tensor is never materialised
  - Gated activation (gate × silu(gate) or gate × gelu(gate) / relu(gate))
    is fused with the W1 matmul
  - W2 result is atomically accumulated into the output buffer, so there is no
    intermediate (num_pairs, H) tensor
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# W1 KERNEL  (int4 matmul + gated activation → intermediate)
# ---------------------------------------------------------------------------


@triton.jit
def fused_w1_int4_activation_kernel(
    X_ptr,           # [M, H]             fp16/bf16 – input activations
    qW_ptr,          # [E, H//8, 2*I]     int32     – packed W1 weights
    scales_ptr,      # [E, H//gs, 2*I]   fp16      – per-group scales
    qzeros_ptr,      # [E, H//gs, 2*I//8] int32     – packed per-group zeros
    Y_ptr,           # [num_pairs, I]     fp16/bf16 – output after activation
    expert_ids_ptr,  # [num_pairs]        int32
    token_ids_ptr,   # [num_pairs]        int32
    stride_xm,       # stride of X along M (token) dim
    stride_xk,       # stride of X along K (hidden) dim  (==1 for contiguous)
    stride_we,       # stride of qW along E (expert) dim
    stride_wk,       # stride of qW along K_packed dim
    stride_wn,       # stride of qW along N (output channel) dim
    stride_se,       # stride of scales along E dim
    stride_sg,       # stride of scales along group dim
    stride_sn,       # stride of scales along N dim
    stride_ze,       # stride of qzeros along E dim
    stride_zg,       # stride of qzeros along group dim
    stride_zn,       # stride of qzeros along N_packed dim
    stride_ym,       # stride of Y along num_pairs dim
    stride_yn,       # stride of Y along I dim
    num_pairs: tl.constexpr,
    H: tl.constexpr,          # hidden (input) dimension
    I: tl.constexpr,          # half of W1 output = intermediate size per expert
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,    # output channels per tile  (must be ≥ 8 and power-of-2)
    BLOCK_K_PACKED: tl.constexpr,  # packed int32s per K-tile  (= group_size // 8)
    activation: tl.constexpr, # 0 = silu, 1 = gelu, 2 = relu
):
    """Fused W1 int4-GEMM + gated activation.

    Computes  Y[pair, :] = act(gate) * up
    where     gate = X[token, :] @ W1_gate[expert, :, :].T   (output channels [0,   I))
              up   = X[token, :] @ W1_up  [expert, :, :].T   (output channels [I, 2*I))
    and W1 is stored in GPTQ int4 format as qW of shape (E, H//8, 2*I).
    """
    pid_pair = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_pair >= num_pairs:
        return

    expert_id = tl.load(expert_ids_ptr + pid_pair)
    token_id = tl.load(token_ids_ptr + pid_pair)

    # Output channel indices for the gate half [0, I)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < I

    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_N,), dtype=tl.float32)

    H_packed = H // 8  # number of int32 rows in qweight

    for k_packed_start in range(0, H_packed, BLOCK_K_PACKED):
        # ---- load packed weights for gate and up ----
        offs_k_packed = k_packed_start + tl.arange(0, BLOCK_K_PACKED)
        mask_kp = offs_k_packed < H_packed

        # qW shape: (E, H_packed, 2*I)
        #   gate output: columns offs_n        [0, I)
        #   up   output: columns offs_n + I    [I, 2*I)
        qw_gate_ptrs = (
            qW_ptr
            + expert_id * stride_we
            + offs_k_packed[:, None] * stride_wk
            + offs_n[None, :] * stride_wn
        )
        qw_gate = tl.load(qw_gate_ptrs, mask=mask_kp[:, None] & mask_n[None, :], other=0)

        qw_up_ptrs = (
            qW_ptr
            + expert_id * stride_we
            + offs_k_packed[:, None] * stride_wk
            + (offs_n[None, :] + I) * stride_wn
        )
        qw_up = tl.load(qw_up_ptrs, mask=mask_kp[:, None] & mask_n[None, :], other=0)

        # ---- load scales and zeros (constant for the whole K-tile when
        #      BLOCK_K_PACKED * 8 == group_size) ----
        group_id = (k_packed_start * 8) // group_size

        scale_gate_ptrs = (
            scales_ptr
            + expert_id * stride_se
            + group_id * stride_sg
            + offs_n * stride_sn
        )
        scale_gate = tl.load(scale_gate_ptrs, mask=mask_n, other=1.0).to(tl.float32)

        scale_up_ptrs = (
            scales_ptr
            + expert_id * stride_se
            + group_id * stride_sg
            + (offs_n + I) * stride_sn
        )
        scale_up = tl.load(scale_up_ptrs, mask=mask_n, other=1.0).to(tl.float32)

        # zeros are packed: qzeros[e, group, n // 8]
        zero_gate_ptrs = (
            qzeros_ptr
            + expert_id * stride_ze
            + group_id * stride_zg
            + (offs_n // 8) * stride_zn
        )
        zero_gate_packed = tl.load(zero_gate_ptrs, mask=mask_n, other=0)
        zero_gate_shift = (offs_n % 8) * 4
        zero_gate = ((zero_gate_packed >> zero_gate_shift) & 0xF).to(tl.float32)

        # up zeros at output channel (offs_n + I)
        zero_up_ptrs = (
            qzeros_ptr
            + expert_id * stride_ze
            + group_id * stride_zg
            + ((offs_n + I) // 8) * stride_zn
        )
        zero_up_packed = tl.load(zero_up_ptrs, mask=mask_n, other=0)
        zero_up_shift = ((offs_n + I) % 8) * 4
        zero_up = ((zero_up_packed >> zero_up_shift) & 0xF).to(tl.float32)

        # ---- unpack int4, dequantize, and accumulate over 8 bit positions ----
        # For bit position b, the corresponding input channels are:
        #   k_packed_start * 8 + b,  k_packed_start * 8 + b + 8,  ...
        # We load x at those channels for each packed slot.
        for bit_pos in tl.static_range(8):
            shift = bit_pos * 4
            # Extract int4 values: (BLOCK_K_PACKED, BLOCK_N)
            w_gate_int4 = (qw_gate >> shift) & 0xF
            w_up_int4 = (qw_up >> shift) & 0xF

            # Dequantize: w_fp = (w_int4 - zero) * scale
            w_gate_fp = (w_gate_int4.to(tl.float32) - zero_gate[None, :]) * scale_gate[None, :]
            w_up_fp = (w_up_int4.to(tl.float32) - zero_up[None, :]) * scale_up[None, :]

            # Load x at the K positions corresponding to this bit position.
            # The channels are: k_packed_start*8 + bit_pos + k_packed_idx*8
            # for k_packed_idx in [0, BLOCK_K_PACKED).
            offs_k_bit = k_packed_start * 8 + bit_pos + tl.arange(0, BLOCK_K_PACKED) * 8
            mask_k_bit = offs_k_bit < H
            x_bit = tl.load(
                X_ptr + token_id * stride_xm + offs_k_bit * stride_xk,
                mask=mask_k_bit,
                other=0.0,
            ).to(tl.float32)

            # Accumulate: (BLOCK_K_PACKED,) × (BLOCK_K_PACKED, BLOCK_N) → (BLOCK_N,)
            acc_gate += tl.sum(w_gate_fp * x_bit[:, None], axis=0)
            acc_up += tl.sum(w_up_fp * x_bit[:, None], axis=0)

    # ---- Gated activation ----
    if activation == 0:  # silu
        activated = acc_gate * tl.sigmoid(acc_gate) * acc_up
    elif activation == 1:  # gelu (tanh approximation)
        sqrt_2_over_pi = 0.7978845608028654
        gate_cubed = acc_gate * acc_gate * acc_gate
        tanh_arg = sqrt_2_over_pi * (acc_gate + 0.044715 * gate_cubed)
        activated = 0.5 * acc_gate * (1.0 + tl.math.tanh(tanh_arg)) * acc_up
    elif activation == 2:  # relu
        activated = tl.where(acc_gate > 0.0, acc_gate, 0.0) * acc_up
    else:  # default silu
        activated = acc_gate * tl.sigmoid(acc_gate) * acc_up

    y_ptrs = Y_ptr + pid_pair * stride_ym + offs_n * stride_yn
    tl.store(y_ptrs, activated.to(X_ptr.dtype.element_ty), mask=mask_n)


# ---------------------------------------------------------------------------
# W2 KERNEL  (int4 matmul + weighted atomic-add reduce → final output)
# ---------------------------------------------------------------------------


@triton.jit
def w2_int4_reduce_kernel(
    X_ptr,           # [num_pairs, I]     fp16/bf16 – intermediate activations
    qW_ptr,          # [E, I//8, H]       int32     – packed W2 weights
    scales_ptr,      # [E, I//gs, H]      fp16      – per-group scales
    qzeros_ptr,      # [E, I//gs, H//8]   int32     – packed per-group zeros
    Y_ptr,           # [M, H]             fp16/bf16 – output (accumulated)
    expert_ids_ptr,  # [num_pairs]        int32
    token_ids_ptr,   # [num_pairs]        int32
    weights_ptr,     # [num_pairs]        fp32      – routing weights
    stride_xm,       # stride of X (intermediate) along pairs dim
    stride_xk,       # stride of X along I dim
    stride_we,       # stride of qW along E dim
    stride_wk,       # stride of qW along K_packed dim   (K = I here)
    stride_wn,       # stride of qW along N dim           (N = H here)
    stride_se,
    stride_sg,
    stride_sn,
    stride_ze,
    stride_zg,
    stride_zn,
    stride_ym,       # stride of Y along M (token) dim
    stride_yn,       # stride of Y along H dim
    num_pairs: tl.constexpr,
    H: tl.constexpr,          # output (hidden) dimension
    I: tl.constexpr,          # input (intermediate) dimension
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,    # output (H) channels per tile
    BLOCK_K_PACKED: tl.constexpr,  # packed int32s per K-tile (= group_size // 8)
):
    """W2 int4-GEMM + routing-weight scale + atomic-add reduce.

    Computes  Y[token, :] += routing_weight * X[pair, :] @ W2[expert, :, :].T
    where W2 has shape (E, H, I) logically, stored as (E, I//8, H) in GPTQ int4.
    """
    pid_pair = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_pair >= num_pairs:
        return

    expert_id = tl.load(expert_ids_ptr + pid_pair)
    token_id = tl.load(token_ids_ptr + pid_pair)
    weight = tl.load(weights_ptr + pid_pair)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < H

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    I_packed = I // 8

    for k_packed_start in range(0, I_packed, BLOCK_K_PACKED):
        offs_k_packed = k_packed_start + tl.arange(0, BLOCK_K_PACKED)
        mask_kp = offs_k_packed < I_packed

        qw_ptrs = (
            qW_ptr
            + expert_id * stride_we
            + offs_k_packed[:, None] * stride_wk
            + offs_n[None, :] * stride_wn
        )
        qw = tl.load(qw_ptrs, mask=mask_kp[:, None] & mask_n[None, :], other=0)

        group_id = (k_packed_start * 8) // group_size

        scale_ptrs = (
            scales_ptr
            + expert_id * stride_se
            + group_id * stride_sg
            + offs_n * stride_sn
        )
        scale = tl.load(scale_ptrs, mask=mask_n, other=1.0).to(tl.float32)

        zero_ptrs = (
            qzeros_ptr
            + expert_id * stride_ze
            + group_id * stride_zg
            + (offs_n // 8) * stride_zn
        )
        zero_packed = tl.load(zero_ptrs, mask=mask_n, other=0)
        zero_shift = (offs_n % 8) * 4
        zero = ((zero_packed >> zero_shift) & 0xF).to(tl.float32)

        for bit_pos in tl.static_range(8):
            shift = bit_pos * 4
            w_int4 = (qw >> shift) & 0xF
            w_fp = (w_int4.to(tl.float32) - zero[None, :]) * scale[None, :]

            offs_k_bit = k_packed_start * 8 + bit_pos + tl.arange(0, BLOCK_K_PACKED) * 8
            mask_k_bit = offs_k_bit < I
            x_bit = tl.load(
                X_ptr + pid_pair * stride_xm + offs_k_bit * stride_xk,
                mask=mask_k_bit,
                other=0.0,
            ).to(tl.float32)

            acc += tl.sum(w_fp * x_bit[:, None], axis=0)

    weighted_result = acc * weight
    y_ptrs = Y_ptr + token_id * stride_ym + offs_n * stride_yn
    tl.atomic_add(y_ptrs, weighted_result.to(X_ptr.dtype.element_ty), mask=mask_n)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def fused_experts_int4_impl(
    hidden_states: torch.Tensor,
    stacked_qw1: torch.Tensor,
    stacked_scales_w1: torch.Tensor,
    stacked_qzeros_w1: torch.Tensor,
    stacked_qw2: torch.Tensor,
    stacked_scales_w2: torch.Tensor,
    stacked_qzeros_w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    group_size: int = 128,
    activation: str = "silu",
    use_sorted: bool = False,
) -> torch.Tensor:
    """Fused MoE forward pass for int4-quantized expert weights.

    Uses on-the-fly dequantization inside Triton kernels so the full fp16
    weight tensor is never materialised.  All token–expert pairs are processed
    in a single kernel launch, reducing kernel-launch overhead compared to
    calling a per-expert QuantLinear.

    Args:
        hidden_states:      (M, H) fp16/bf16 input tokens.
        stacked_qw1:        (E, H//8, 2*I) int32  – packed W1 for all experts.
        stacked_scales_w1:  (E, H//group_size, 2*I) fp16 – W1 scales.
        stacked_qzeros_w1:  (E, H//group_size, 2*I//8) int32 – W1 packed zeros.
        stacked_qw2:        (E, I//8, H) int32    – packed W2 for all experts.
        stacked_scales_w2:  (E, I//group_size, H) fp16 – W2 scales.
        stacked_qzeros_w2:  (E, I//group_size, H//8) int32 – W2 packed zeros.
        topk_weights:       (M, K) routing weights.
        topk_ids:           (M, K) expert indices.
        group_size:         Quantization group size (default 128).
        activation:         "silu", "gelu", or "relu".
        use_sorted:         Sort tokens by expert before processing (better for
                            large batches; small overhead for tiny batches).

    Returns:
        torch.Tensor: (M, H) output in the same dtype as hidden_states.
    """
    M, H = hidden_states.shape
    _, H_packed, double_I = stacked_qw1.shape
    I = double_I // 2  # noqa: E741
    K = topk_ids.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    num_pairs = M * K

    # ---- build token–expert index arrays ----
    if use_sorted:
        flat_topk_ids = topk_ids.flatten()
        sorted_indices = torch.argsort(flat_topk_ids, stable=True)
        expert_ids = flat_topk_ids[sorted_indices]
        token_ids = (
            torch.arange(M, device=device, dtype=torch.int32)
            .repeat_interleave(K)[sorted_indices]
        )
        weights = topk_weights.flatten()[sorted_indices]
    else:
        expert_ids = topk_ids.flatten()
        token_ids = torch.arange(M, device=device, dtype=torch.int32).repeat_interleave(K)
        weights = topk_weights.flatten()

    # ---- tune block sizes ----
    # BLOCK_K_PACKED = group_size // 8  ensures each K-tile covers exactly one
    # quantisation group, so scales/zeros can be loaded once per tile.
    BLOCK_K_PACKED = group_size // 8  # e.g. 16 for group_size=128
    BLOCK_N = 64

    activation_type = {"silu": 0, "gelu": 1, "relu": 2}.get(
        (activation or "silu").lower(), 0
    )

    # ---- W1 + activation ----
    intermediate = torch.empty((num_pairs, I), device=device, dtype=dtype)

    grid_w1 = (num_pairs, triton.cdiv(I, BLOCK_N))
    fused_w1_int4_activation_kernel[grid_w1](
        hidden_states,
        stacked_qw1,
        stacked_scales_w1,
        stacked_qzeros_w1,
        intermediate,
        expert_ids,
        token_ids,
        hidden_states.stride(0),
        hidden_states.stride(1),
        stacked_qw1.stride(0),
        stacked_qw1.stride(1),
        stacked_qw1.stride(2),
        stacked_scales_w1.stride(0),
        stacked_scales_w1.stride(1),
        stacked_scales_w1.stride(2),
        stacked_qzeros_w1.stride(0),
        stacked_qzeros_w1.stride(1),
        stacked_qzeros_w1.stride(2),
        intermediate.stride(0),
        intermediate.stride(1),
        num_pairs=num_pairs,
        H=H,
        I=I,
        group_size=group_size,
        BLOCK_N=BLOCK_N,
        BLOCK_K_PACKED=BLOCK_K_PACKED,
        activation=activation_type,
    )

    # ---- W2 + weighted reduce ----
    final_output = torch.zeros((M, H), device=device, dtype=dtype)

    # For W2: in_features = I, out_features = H
    # BLOCK_K_PACKED_W2 = group_size // 8 (same as W1 in most configs)
    BLOCK_K_PACKED_W2 = group_size // 8

    grid_w2 = (num_pairs, triton.cdiv(H, BLOCK_N))
    w2_int4_reduce_kernel[grid_w2](
        intermediate,
        stacked_qw2,
        stacked_scales_w2,
        stacked_qzeros_w2,
        final_output,
        expert_ids,
        token_ids,
        weights,
        intermediate.stride(0),
        intermediate.stride(1),
        stacked_qw2.stride(0),
        stacked_qw2.stride(1),
        stacked_qw2.stride(2),
        stacked_scales_w2.stride(0),
        stacked_scales_w2.stride(1),
        stacked_scales_w2.stride(2),
        stacked_qzeros_w2.stride(0),
        stacked_qzeros_w2.stride(1),
        stacked_qzeros_w2.stride(2),
        final_output.stride(0),
        final_output.stride(1),
        num_pairs=num_pairs,
        H=H,
        I=I,
        group_size=group_size,
        BLOCK_N=BLOCK_N,
        BLOCK_K_PACKED=BLOCK_K_PACKED_W2,
    )

    return final_output
