"""GGUF quantized linear layer.

Stores weights in their native GGUF quantized format (uint8 packed blocks) and
dequantizes them on-the-fly during the forward pass, following the same
pattern as the autoround/Marlin backend: quantized weights stay on disk and in
CPU/GPU memory in compact form; dequantization happens per forward call.

The quantization type is persisted as a model buffer (``gguf_qtype``, int32)
and stored alongside the quantized weight buffer (``weight``, uint8) in the
safetensors shard.  Both buffers are loaded automatically by
:meth:`~eole.models.model.BaseModel.load_safe_state_dict`.
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

        For **Q8_0**, **Q4_0**, and **Q4_1** quantisation types the
        dequantisation is performed entirely on the same device as the weight
        tensor (GPU or CPU) using PyTorch tensor operations, which avoids any
        CPU round-trip when the model is on a CUDA device.

        All other GGUF types (Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, IQ-family, …)
        fall back to the ``gguf`` library's CPU path.  In those cases the
        quantised weight buffer is briefly copied to the CPU for
        dequantisation before the result is moved back to the compute device.

        .. note::
            A truly fused dequantise+GEMM CUDA kernel (analogous to Marlin
            for GPTQ) would eliminate the intermediate float16 weight
            allocation for K-quant types as well.  Until such a kernel is
            available the GPU path already covers the most common
            ``llama.cpp``-style Q8_0 / Q4_0 / Q4_1 conversions.
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
        # GPU-native paths: dequantise directly on-device with PyTorch ops.
        # The quantised uint8 weight stays on the GPU; no CPU round-trip.
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
