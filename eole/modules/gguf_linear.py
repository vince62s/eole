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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize weight on-the-fly, then apply linear transformation."""
        try:
            from gguf import dequantize, GGMLQuantizationType
        except ImportError:
            raise ImportError("Install 'gguf' to run inference with GGUF quantized models: pip install gguf")

        qtype = GGMLQuantizationType(int(self.gguf_qtype.item()))

        # Dequantize: weight is stored as (out_features, bytes_per_row) uint8
        # or as a 1-D flat buffer.  Either way, gguf.dequantize produces a
        # float32 numpy array with shape (out_features * in_features,) or
        # (out_features, in_features).
        w_np = self.weight.cpu().numpy()
        dq_np = dequantize(w_np, qtype)  # → float32 ndarray

        weight = (
            torch.from_numpy(dq_np)
            .reshape(self.out_features, self.in_features)
            .to(dtype=x.dtype, device=x.device)
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
