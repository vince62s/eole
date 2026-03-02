import torch.nn as nn
from torch.cuda import is_available as cuda_is_available

def replace_autoround_linear(model, module_to_convert=[], w_bit=4, group_size=128, packing_format="auto_round:auto_gptq"):
    """Replace nn.Linear layers with AutoRound QuantLinear for quantized inference.

    The packing_format determines which QuantLinear implementation to use:
    - If 'gptq' is in packing_format, qzeros are stored as (zero_point - 1) per GPTQ convention,
      so qlinear_*_zp is used (which adds +1 during dequantization).
    - Otherwise, qzeros are stored directly and qlinear_* (no zp) is used.

    Backend preference order:
    1. CUDA/Marlin kernels — fastest; only available for symmetric (non-GPTQ-zp) format on CUDA.
    2. Triton kernels — fast GPU kernels, avoid re-dequantizing weights on every forward pass.
    3. PyTorch fallback — works everywhere but dequantizes weights on every forward pass.
    """
    use_gptq_zp = "gptq" in packing_format

    # Prefer CUDA/Marlin, then Triton on CUDA, fall back to PyTorch-only kernels.
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
    1. CUDA/Marlin kernels (requires CUDA + gptqmodel_marlin_kernels) — fastest; symmetric only
    2. Triton kernels (requires CUDA + triton) — fast for GPU inference
    3. PyTorch fallback — works everywhere but dequantizes weights on every forward pass

    Marlin only supports symmetric quantization, so it is skipped when use_gptq_zp=True
    (GPTQ zero-point format is asymmetric).
    """
    if cuda_is_available():
        # Try CUDA/Marlin first (fastest) — only supports symmetric quantization
        if not use_gptq_zp:
            try:
                import gptqmodel_marlin_kernels  # noqa: F401 — verify Marlin CUDA kernels are available
                from auto_round_extension.cuda.gptqmodel_marlin import get_marlin_layer

                _MarlinCls = get_marlin_layer()

                # Wrap to normalize the interface: Marlin uses in_features/out_features/sym/desc_act
                # while the rest of our code uses infeatures/outfeatures.
                class MarlinQuantLinear(_MarlinCls):
                    def __init__(self, bits, group_size, infeatures, outfeatures, bias, **kwargs):
                        super().__init__(
                            bits=bits,
                            group_size=group_size,
                            desc_act=False,
                            sym=True,  # Marlin only supports symmetric quantization
                            in_features=infeatures,
                            out_features=outfeatures,
                            bias=bias,
                            **kwargs,
                        )

                return MarlinQuantLinear
            except ImportError:
                pass

        # Try Triton kernels
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
