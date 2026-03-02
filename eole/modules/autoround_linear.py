import torch.nn as nn


def replace_autoround_linear(model, module_to_convert=[], w_bit=4, group_size=128, packing_format="auto_round:auto_gptq"):
    """Replace nn.Linear layers with AutoRound QuantLinear for quantized inference.

    The packing_format determines which QuantLinear implementation to use:
    - If 'gptq' is in packing_format, qzeros are stored as (zero_point - 1) per GPTQ convention,
      so qlinear_torch_zp is used (which adds +1 during dequantization).
    - Otherwise, qzeros are stored directly and qlinear_torch is used.
    """
    try:
        from auto_round_extension.torch.qlinear_torch import QuantLinear as QuantLinearDirect
        from auto_round_extension.torch.qlinear_torch_zp import QuantLinear as QuantLinearGptq
    except ImportError:
        raise ImportError("Install auto-round to use autoround quantized models")
    # Select QuantLinear based on packing_format, not sym:
    # GPTQ packing stores zeros as (zp-1), so inference needs to add +1 (qlinear_torch_zp)
    # Direct packing stores zeros as-is, so no adjustment needed (qlinear_torch)
    QuantLinear = QuantLinearGptq if "gptq" in packing_format else QuantLinearDirect

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
