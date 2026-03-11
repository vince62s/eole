"""GGUF quantized linear layer.

Stores weights in their native GGUF quantized format (uint8 packed blocks) and
dequantizes them on-the-fly during the forward pass, following the same
pattern as the autoround/Marlin backend: quantized weights stay on disk and in
CPU/GPU memory in compact form; dequantization happens per forward call.

The quantization type is persisted as a model buffer (``gguf_qtype``, int32)
and stored alongside the quantized weight buffer (``weight``, uint8) in the
safetensors shard.  Both buffers are loaded automatically by
:meth:`~eole.models.model.BaseModel.load_safe_state_dict`.

GPU fast-paths (priority order)
--------------------------------
1. **vLLM CUDA kernels** (``vllm._C``): when vLLM is installed and the weight
   is on a CUDA device, the three kernels ported from ``ggml_quants.cu``
   handle *all* GGUF quantisation types — including Q4_K/M, Q5_K, Q6_K, Q3_K,
   Q2_K and all IQ-family types — without any CPU round-trip:

   * ``ggml_mul_mat_vec_a8`` – fused MMVQ (matrix-vector) for small batches.
   * ``ggml_mul_mat_a8``     – fused MMQ  (matrix-matrix) for larger batches
     (standard + K-quant types only).
   * ``ggml_dequantize``     – GPU dequantise then standard ``x @ W.T`` for
     IQ types with large batch where MMQ is not yet available.

   This replicates the strategy used in
   ``vllm/model_executor/layers/quantization/gguf.py``.

2. **Triton fused kernel** (Q4_0 only): when Triton is installed but vLLM is
   not, dequantises Q4_0 weights in GPU registers, never writing an fp16
   weight tensor to HBM.

3. **PyTorch-native GPU paths** (Q8_0, Q4_0, Q4_1): on-device tensor ops,
   no CPU round-trip.

4. **CPU fallback** (all other types): ``gguf.dequantize`` via numpy, then
   the result is moved back to the compute device.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Float-typed GGUF types that should be loaded as regular tensors (not as
# uint8 quantized blocks).  Models that use these for linear weights don't
# need GGUFLinear – they are handled by normal nn.Linear loading.
_GGUF_FLOAT_TYPES: set[int] = set()

try:
    from gguf import GGMLQuantizationType

    _GGUF_FLOAT_TYPES = {
        GGMLQuantizationType.F32.value,
        GGMLQuantizationType.F16.value,
        GGMLQuantizationType.BF16.value,
        GGMLQuantizationType.F64.value,
    }
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Optional vLLM GGUF CUDA ops.
#
# When ``vllm._C`` is importable the three kernels compiled from
# ggml_quants.cu are available and handle ALL ggml quantisation types on GPU.
# The kernel selection logic mirrors
# ``vllm/model_executor/layers/quantization/gguf.py::_fused_mul_mat_gguf``.
# ---------------------------------------------------------------------------

_VLLM_GGUF_OPS_AVAILABLE = False
_vllm_ggml_dequantize = None
_vllm_ggml_mul_mat_vec_a8 = None
_vllm_ggml_mul_mat_a8 = None

# Type-ID sets that drive kernel selection (integer values of the enum).
# Built from the gguf library; empty frozensets if gguf is not installed.
_VLLM_MMQ_TYPE_IDS: frozenset[int] = frozenset()      # standard + K-quants
_VLLM_IMATRIX_TYPE_IDS: frozenset[int] = frozenset()  # IQ-family

try:
    from gguf import GGMLQuantizationType as _GQT

    _standard_names = {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1"}
    _kquant_names = {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"}
    _imatrix_names = {
        "IQ1_M", "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S",
        "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL",
    }

    def _type_ids(*names: str) -> frozenset:
        return frozenset(_GQT[n].value for n in names if hasattr(_GQT, n))

    _VLLM_MMQ_TYPE_IDS = _type_ids(*(_standard_names | _kquant_names))
    _VLLM_IMATRIX_TYPE_IDS = _type_ids(*_imatrix_names)

    del _type_ids, _standard_names, _kquant_names, _imatrix_names, _GQT
except ImportError:
    pass

try:
    import vllm._C as _vllm_c  # CUDA extension compiled with vLLM

    _vllm_ggml_dequantize = _vllm_c.ggml_dequantize
    _vllm_ggml_mul_mat_vec_a8 = _vllm_c.ggml_mul_mat_vec_a8
    _vllm_ggml_mul_mat_a8 = _vllm_c.ggml_mul_mat_a8
    _VLLM_GGUF_OPS_AVAILABLE = True
except Exception:
    _VLLM_GGUF_OPS_AVAILABLE = False


def _vllm_gguf_linear(
    weight_u8: torch.Tensor,
    x: torch.Tensor,
    qtype_val: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """vLLM-style fused GGUF dequantize+matmul using ``ggml_quants.cu`` kernels.

    Replicates the kernel-selection logic from
    ``vllm/model_executor/layers/quantization/gguf.py::_fused_mul_mat_gguf``:

    * ``ggml_mul_mat_vec_a8`` (MMVQ) for small batch sizes (≤ 2 or 6).
    * ``ggml_mul_mat_a8`` (MMQ) for larger batches — standard + K-quant types.
    * ``ggml_dequantize`` + ``x @ W.T`` for IQ types with large batches.

    Parameters
    ----------
    weight_u8 : torch.Tensor
        Flat ``uint8`` tensor of raw GGUF weight blocks, on a CUDA device.
    x : torch.Tensor
        Input activations, shape ``(*, in_features)``, on the same device.
    qtype_val : int
        Integer value of the :class:`gguf.GGMLQuantizationType` enum.
    out_features, in_features : int
        Logical dimensions of the linear transformation.

    Returns
    -------
    torch.Tensor
        Output of shape ``(*, out_features)``, same dtype as *x*.
    """
    try:
        from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
    except ImportError:
        raise ImportError(
            "Install 'gguf' to run vLLM-backed GGUF inference: pip install gguf"
        )

    block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType(qtype_val)]
    bytes_per_row = in_features // block_size * type_size
    weight_2d = weight_u8.reshape(out_features, bytes_per_row)

    orig_shape = x.shape
    N = x.numel() // in_features
    x_2d = x.contiguous().reshape(N, in_features)

    if N == 0:
        return torch.empty(*orig_shape[:-1], out_features, dtype=x.dtype, device=x.device)

    is_imatrix = qtype_val in _VLLM_IMATRIX_TYPE_IDS

    # MMVQ batch-size thresholds mirror vLLM's _fused_mul_mat_gguf heuristic:
    #   - IQ types tolerate larger MMVQ batch because MMQ is not yet available
    #     for them; the "large model" boundary (> 5120 output rows) drops the
    #     threshold to conserve shared memory.
    #   - Standard / K-quant types use a much lower threshold because MMQ
    #     (ggml_mul_mat_a8) is available and more efficient for any N > 2/6.
    if is_imatrix:
        # IQ types: MMVQ for N ≤ 8 (large model) or N ≤ 16 (small model).
        _IMATRIX_MMVQ_MAX_LARGE = 8   # out_features > 5120
        _IMATRIX_MMVQ_MAX_SMALL = 16  # out_features ≤ 5120
        mmvq_safe = _IMATRIX_MMVQ_MAX_LARGE if out_features > 5120 else _IMATRIX_MMVQ_MAX_SMALL
    else:
        # Standard / K-quant types: MMVQ for N ≤ 2 (large) or N ≤ 6 (small).
        _STANDARD_MMVQ_MAX_LARGE = 2
        _STANDARD_MMVQ_MAX_SMALL = 6
        mmvq_safe = _STANDARD_MMVQ_MAX_LARGE if out_features > 5120 else _STANDARD_MMVQ_MAX_SMALL

    if N <= mmvq_safe:
        # MMVQ: fused quantised mat-vec (all types).
        y = _vllm_ggml_mul_mat_vec_a8(weight_2d, x_2d, qtype_val, out_features)
    elif not is_imatrix:
        # MMQ: fused quantised mat-mat (standard + K-quant types).
        y = _vllm_ggml_mul_mat_a8(weight_2d, x_2d, qtype_val, out_features)
    else:
        # I-quant with large batch: dequantise on GPU then standard matmul.
        weight_dq = _vllm_ggml_dequantize(
            weight_2d, qtype_val, out_features, in_features, x.dtype
        )
        y = x_2d @ weight_dq.T

    return y.reshape(*orig_shape[:-1], out_features)


# ---------------------------------------------------------------------------
# Optional Triton fused kernel for Q4_0 dequantize + matmul.
#
# Used when Triton is installed but vLLM is not available.  Handles Q4_0 only.
#
# Q4_0 block layout (18 bytes per block of 32 weights):
#   bytes 0-1 : scale, float16 little-endian
#   bytes 2-17: 16 uint8 nibble pairs
#               - low  nibble of byte i → weight at offset i   (0..15)
#               - high nibble of byte i → weight at offset i+16 (16..31)
#   weight value = (nibble - 8) * scale
#
# The Triton kernel tiles both the output-row (M) and the batch (N) dimensions
# and iterates over K in steps of 32 (one Q4_0 block per step).  For each
# block it dequantises the 32 weights in registers and accumulates their
# dot product with the corresponding activation slice, avoiding writing any
# intermediate fp16/fp32 weight tensor to HBM.
# ---------------------------------------------------------------------------

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 16}),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 16}),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}),
        ],
        key=["M", "K", "N"],
    )
    @triton.jit
    def _q4_0_dequant_matmul_kernel(
        W_ptr,  # uint8 flat buffer: [M * (K//32) * 18] bytes
        X_ptr,  # float16 or float32 [N, K]
        Y_ptr,  # float32 [N, M]
        M,
        K,
        N,
        stride_xn,
        stride_xk,
        stride_yn,
        stride_ym,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused Q4_0 dequantize + matmul Triton kernel.

        Each program instance handles a BLOCK_M × BLOCK_N tile of the output
        matrix and iterates over all K//32 Q4_0 blocks, dequantising in
        registers and accumulating into a fp32 accumulator.

        Parameters
        ----------
        W_ptr : pointer to uint8
            Flat Q4_0 weight buffer.  For output row ``m`` and K-block
            ``b``, the block starts at byte ``(m * (K // 32) + b) * 18``.
        X_ptr : pointer to float16 or float32
            Activation matrix, shape [N, K].
        Y_ptr : pointer to float32
            Output matrix, shape [N, M].  Written as fp32; the caller casts
            to the required dtype.
        M, K, N : int
            Matrix dimensions.
        stride_xn, stride_xk : int
            Strides (in elements) for X.
        stride_yn, stride_ym : int
            Strides (in elements) for Y.
        BLOCK_M, BLOCK_N : constexpr int
            Tile sizes; must be ≥ 16 for tensor-core usage via ``tl.dot``.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)    # [BLOCK_N]

        row_mask = rows < M
        n_mask = ns < N

        n_blocks_k = K // 32
        # Number of bytes per output row in the quantised weight buffer.
        row_stride_bytes = n_blocks_k * 18

        acc = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)

        for k_block in range(n_blocks_k):
            # Byte offset of this Q4_0 block for each output row: [BLOCK_M]
            block_base = rows * row_stride_bytes + k_block * 18

            # ---- Load scale (2-byte little-endian float16) ----
            b0 = tl.load(W_ptr + block_base, mask=row_mask, other=0).to(tl.uint16)
            b1 = tl.load(W_ptr + block_base + 1, mask=row_mask, other=0).to(tl.uint16)
            # Reconstruct uint16 in little-endian order then bitcast to f16.
            scale_u16 = b0 | (b1 << 8)
            scale = scale_u16.to(tl.float16, bitcast=True).to(tl.float32)  # [BLOCK_M]

            # ---- Load 16 nibble bytes: [BLOCK_M, 16] ----
            nibble_offs = block_base[:, None] + 2 + tl.arange(0, 16)[None, :]
            packed = tl.load(
                W_ptr + nibble_offs, mask=row_mask[:, None], other=0
            )  # [BLOCK_M, 16] uint8

            # ---- Dequantise nibbles: both halves as float16 [BLOCK_M, 16] ----
            # lo: weight offsets 0..15  (low  nibble of each byte)
            # hi: weight offsets 16..31 (high nibble of each byte)
            lo = ((packed & 0x0F).to(tl.int8) - 8).to(tl.float16)
            hi = (((packed >> 4) & 0x0F).to(tl.int8) - 8).to(tl.float16)

            k_base_val = k_block * 32

            # ---- Load activations: [BLOCK_N, 16] for each half ----
            k_lo_idx = k_base_val + tl.arange(0, 16)        # [16]  (constexpr)
            k_hi_idx = k_base_val + 16 + tl.arange(0, 16)   # [16]

            x_lo = tl.load(
                X_ptr + ns[:, None] * stride_xn + k_lo_idx[None, :] * stride_xk,
                mask=n_mask[:, None],
                other=0.0,
            ).to(tl.float16)  # [BLOCK_N, 16]

            x_hi = tl.load(
                X_ptr + ns[:, None] * stride_xn + k_hi_idx[None, :] * stride_xk,
                mask=n_mask[:, None],
                other=0.0,
            ).to(tl.float16)  # [BLOCK_N, 16]

            # ---- Fused dot: [BLOCK_N, 16] @ [16, BLOCK_M] = [BLOCK_N, BLOCK_M] ----
            # contrib[n, m] = Σ_j x_lo[n,j]*lo[m,j] + x_hi[n,j]*hi[m,j]
            contrib = tl.dot(x_lo, tl.trans(lo)) + tl.dot(x_hi, tl.trans(hi))

            # Apply per-row scale (broadcast over batch dim).
            acc += contrib * scale[None, :]

        # ---- Store output tile ----
        y_offs = ns[:, None] * stride_yn + rows[None, :] * stride_ym
        y_mask = n_mask[:, None] & row_mask[None, :]
        tl.store(Y_ptr + y_offs, acc, mask=y_mask)

    def _q4_0_triton_linear(
        weight_u8: torch.Tensor,
        x: torch.Tensor,
        out_features: int,
        in_features: int,
    ) -> torch.Tensor:
        """Python wrapper around :func:`_q4_0_dequant_matmul_kernel`.

        Parameters
        ----------
        weight_u8 : torch.Tensor
            1-D ``uint8`` tensor of length ``out_features * (in_features // 32) * 18``
            holding the raw Q4_0 weight blocks, on a CUDA device.
        x : torch.Tensor
            Input activations of shape ``(*, in_features)``, on the same CUDA device.
            Any leading batch dimensions are collapsed before calling the kernel.
        out_features, in_features : int
            Logical shape of the linear transformation.

        Returns
        -------
        torch.Tensor
            Output of shape ``(*, out_features)``, dtype ``float32``, on the
            same CUDA device.  Cast to the desired dtype by the caller.
        """
        orig_shape = x.shape
        N = x.numel() // in_features
        x_2d = x.contiguous().reshape(N, in_features)

        M = out_features
        K = in_features

        y = torch.empty((N, M), dtype=torch.float32, device=x.device)

        # Grid is a lambda so the autotuner can inject the winning BLOCK_M /
        # BLOCK_N values at runtime without recompiling.
        grid = lambda meta: (  # noqa: E731
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        _q4_0_dequant_matmul_kernel[grid](
            weight_u8,
            x_2d,
            y,
            M,
            K,
            N,
            x_2d.stride(0),
            x_2d.stride(1),
            y.stride(0),
            y.stride(1),
        )

        return y.reshape(*orig_shape[:-1], M)

except Exception:
    # Triton import failed or kernel definition raised – stay on PyTorch path.
    _TRITON_AVAILABLE = False


class GGUFLinear(nn.Module):
    """Linear layer whose weights are stored in GGUF quantized format.

    Parameters
    ----------
    in_features, out_features:
        Logical (float) dimensions of the linear transformation.
    bias:
        Whether the layer has a bias term.
    qtype_val:
        Integer value of the :class:`gguf.GGMLQuantizationType` enum for the
        stored weights.  Used to pre-populate the ``gguf_qtype`` buffer so that
        :meth:`forward` works correctly before any checkpoint is loaded.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        qtype_val: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Quantized weight data – filled by load_safe_state_dict.
        # Initialised to a 1-byte placeholder; the actual size depends on the
        # quantisation type and will be set when the shard is loaded.
        self.register_buffer("weight", torch.zeros(1, dtype=torch.uint8))

        # Quantization type ID (GGMLQuantizationType int value).
        # Stored as a buffer so it is persisted in the safetensors shard.
        self.register_buffer("gguf_qtype", torch.tensor([qtype_val], dtype=torch.int32))

        if bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros(out_features)))
        else:
            self.bias = None

    @staticmethod
    def _decode_f16_col(blocks: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Reinterpret a uint8 byte slice as float16 scalars (little-endian).

        Parameters
        ----------
        blocks:
            2-D uint8 tensor of shape ``(n_blocks, type_size)``.
        start, end:
            Byte slice ``[start:end]`` within each block (must span exactly
            ``(end - start) / 2`` float16 values, so ``end - start`` must be even).

        Returns
        -------
        torch.Tensor
            1-D float32 tensor of shape ``(n_blocks * (end - start) // 2,)``.
        """
        return blocks[:, start:end].contiguous().view(torch.float16).reshape(-1).to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize weight on-the-fly, then apply linear transformation.

        **vLLM CUDA kernels (all types, CUDA + vLLM installed)** — the three
        kernels compiled from ``ggml_quants.cu`` handle every GGUF quantisation
        type, including Q4_K/M, Q5_K, Q6_K, Q3_K, Q2_K, and all IQ-family
        types.  No CPU round-trip, no intermediate fp16 weight tensor.

        **Triton fused kernel (Q4_0, CUDA + Triton, no vLLM)** — dequantises
        Q4_0 weights in GPU registers and accumulates directly into the output.

        **PyTorch-native GPU paths (Q8_0, Q4_0, Q4_1)** — on-device tensor
        ops; no CPU round-trip when the model is on a CUDA device.

        **CPU fallback (all remaining types)** — ``gguf.dequantize`` via numpy;
        the quantised buffer is copied to the CPU and the result is moved back.
        """
        try:
            from gguf import GGMLQuantizationType
        except ImportError:
            raise ImportError("Install 'gguf' to run inference with GGUF quantized models: pip install gguf")

        qtype_val = int(self.gguf_qtype.item())
        qtype = GGMLQuantizationType(qtype_val)
        target_dtype = x.dtype
        device = self.weight.device

        # ------------------------------------------------------------------
        # 1. vLLM CUDA path: handles ALL GGUF types on GPU using the three
        #    kernels from ggml_quants.cu (MMVQ / MMQ / dequantize+matmul).
        #    This covers K-quants and IQ-quants with no CPU round-trip.
        # ------------------------------------------------------------------
        if _VLLM_GGUF_OPS_AVAILABLE and self.weight.is_cuda:
            out = _vllm_gguf_linear(
                self.weight, x, qtype_val, self.out_features, self.in_features
            ).to(dtype=target_dtype)
            if self.bias is not None:
                out = out + self.bias
            return out

        # ------------------------------------------------------------------
        # 2. PyTorch-native GPU paths.
        #    The quantised uint8 weight stays on the GPU; no CPU round-trip.
        #
        # Block layouts follow the GGUF spec (little-endian scalars):
        #   Q8_0 – [d:f16 (2 B), qs:i8[32] (32 B)]          34 B / 32 elems
        #   Q4_0 – [d:f16 (2 B), qs:u8[16] (16 B)]           18 B / 32 elems
        #   Q4_1 – [d:f16 (2 B), m:f16 (2 B), qs:u8[16]]     20 B / 32 elems
        # ------------------------------------------------------------------

        if qtype == GGMLQuantizationType.Q8_0:
            blocks = self.weight.reshape(-1, 34)  # (n_blocks, 34)
            # First 2 bytes of each block = scale (f16, little-endian)
            scales = self._decode_f16_col(blocks, 0, 2)  # (n_blocks,) float32
            # Remaining 32 bytes = quantised values (int8)
            qs = blocks[:, 2:].contiguous().view(torch.int8).to(torch.float32)  # (n_blocks, 32)
            weight = (qs * scales.unsqueeze(1)).reshape(self.out_features, self.in_features).to(dtype=target_dtype)
            return F.linear(x, weight, self.bias)

        if qtype == GGMLQuantizationType.Q4_0:
            # ------------------------------------------------------------------
            # Triton fused path: dequantise in registers, never materialise the
            # fp16 weight matrix.  Requires CUDA + Triton and K divisible by 32.
            # ------------------------------------------------------------------
            if (
                _TRITON_AVAILABLE
                and self.weight.is_cuda
                and self.in_features % 32 == 0
            ):
                out = _q4_0_triton_linear(
                    self.weight, x, self.out_features, self.in_features
                ).to(dtype=target_dtype)
                if self.bias is not None:
                    out = out + self.bias
                return out

            # PyTorch fallback (CPU or Triton unavailable).
            blocks = self.weight.reshape(-1, 18)  # (n_blocks, 18)
            scales = self._decode_f16_col(blocks, 0, 2)  # (n_blocks,) float32
            qs_u8 = blocks[:, 2:]  # (n_blocks, 16) uint8
            # Nibble order: low nibbles of bytes[0..15], then high nibbles
            lo = qs_u8 & 0x0F         # low  nibbles, values 0–15
            hi = (qs_u8 >> 4) & 0x0F  # high nibbles, values 0–15
            qs = torch.cat([lo, hi], dim=1).to(torch.float32) - 8.0  # (n_blocks, 32), center at 0
            weight = (qs * scales.unsqueeze(1)).reshape(self.out_features, self.in_features).to(dtype=target_dtype)
            return F.linear(x, weight, self.bias)

        if qtype == GGMLQuantizationType.Q4_1:
            blocks = self.weight.reshape(-1, 20)  # (n_blocks, 20)
            scales = self._decode_f16_col(blocks, 0, 2)  # (n_blocks,) float32
            mins   = self._decode_f16_col(blocks, 2, 4)  # (n_blocks,) float32
            qs_u8  = blocks[:, 4:]  # (n_blocks, 16) uint8
            lo = qs_u8 & 0x0F
            hi = (qs_u8 >> 4) & 0x0F
            qs = torch.cat([lo, hi], dim=1).to(torch.float32)  # (n_blocks, 32), values 0–15 (no centering)
            weight = (qs * scales.unsqueeze(1) + mins.unsqueeze(1)).reshape(self.out_features, self.in_features).to(dtype=target_dtype)
            return F.linear(x, weight, self.bias)

        # ------------------------------------------------------------------
        # CPU fallback for K-quant and other complex GGUF types.
        # The weight is copied to CPU for numpy-based dequantisation, then
        # moved back to the compute device.
        # ------------------------------------------------------------------
        from gguf import dequantize

        w_np = self.weight.cpu().numpy()
        dq_np = dequantize(w_np, qtype)
        weight = (
            torch.from_numpy(dq_np)
            .reshape(self.out_features, self.in_features)
            .to(dtype=target_dtype, device=device)
        )
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        qtype_name = "unknown"
        try:
            from gguf import GGMLQuantizationType

            qtype_name = GGMLQuantizationType(int(self.gguf_qtype.item())).name
        except Exception:
            pass
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, qtype={qtype_name}"
        )


def replace_gguf_linear(
    model: nn.Module,
    module_to_convert: list[str],
) -> nn.Module:
    """Replace :class:`~torch.nn.Linear` layers with :class:`GGUFLinear`.

    Walks the model recursively and replaces any ``nn.Linear`` module whose
    name appears in *module_to_convert* with a ``GGUFLinear`` placeholder.
    The quantized weight data and ``gguf_qtype`` are populated later by
    :meth:`~eole.models.model.BaseModel.load_safe_state_dict`.

    Parameters
    ----------
    model:
        Root module to patch in-place.
    module_to_convert:
        List of *local* module names (``module.name``) to replace.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_gguf_linear(module, module_to_convert)

        if isinstance(module, nn.Linear) and name in module_to_convert:
            model._modules[name] = GGUFLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                # qtype_val=0 is a placeholder; the real value is loaded from
                # the safetensors shard via the gguf_qtype buffer.
                qtype_val=0,
            )
    return model
