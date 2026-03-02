import torch.nn as nn
from torch.cuda import is_available as cuda_is_available

def replace_autoround_linear(model, module_to_convert=[], w_bit=4, group_size=128, packing_format="auto_round:auto_gptq"):
    """Replace nn.Linear layers with AutoRound QuantLinear for quantized inference.

    The packing_format determines which QuantLinear implementation to use:
    - If 'gptq' is in packing_format, qzeros are stored as (zero_point - 1) per GPTQ convention,
      so qlinear_*_zp is used (which adds +1 during dequantization).
    - Otherwise, qzeros are stored directly and qlinear_* (no zp) is used.

    When CUDA + Triton are available, fast Triton GPU kernels are used (tritonv2/tritonv2_zp).
    Otherwise, pure PyTorch kernels are used as a fallback (torch/torch_zp).
    The Triton kernels avoid re-dequantizing weights on every forward pass, making them
    significantly faster for GPU inference.
    """
    use_gptq_zp = "gptq" in packing_format

    # Prefer Triton kernels on CUDA (faster: avoids per-call weight dequantization),
    # fall back to PyTorch-only kernels when Triton/CUDA is not available.
    QuantLinear = _get_autoround_quant_linear_cls(use_gptq_zp)

    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_autoround_linear(module, module_to_convert, w_bit, group_size, packing_format)

        if isinstance(module, nn.Linear) and name in module_to_convert:
            model._modules[name] = QuantLinear(
                bits=w_bit,
                group_size=group_size,
                infeatures=module.in_features,
                outfeatures=module.out_features,
                bias=module.bias is not None,
            )
    return model


def _get_autoround_quant_linear_cls(use_gptq_zp: bool):
    """Return the best available QuantLinear class for AutoRound inference.

    Preference order:
    1. Triton kernels (requires CUDA + triton) — fastest for GPU inference
    2. PyTorch fallback — works everywhere but dequantizes weights on every forward pass
    """
    if cuda_is_available():
        try:
            if use_gptq_zp:
                from auto_round_extension.triton.qlinear_tritonv2_zp import QuantLinear
            else:
                from auto_round_extension.triton.qlinear_tritonv2 import QuantLinear
            return QuantLinear
        except ImportError:
            pass
    # Fallback to pure PyTorch kernels
    try:
        if use_gptq_zp:
            from auto_round_extension.torch.qlinear_torch_zp import QuantLinear
        else:
            from auto_round_extension.torch.qlinear_torch import QuantLinear
        return QuantLinear
    except ImportError:
        raise ImportError("Install auto-round to use autoround quantized models")
