import torch.nn as nn


def replace_autoround_linear(model, module_to_convert=[], w_bit=4, group_size=128, sym=True):
    try:
        from auto_round_extension.torch.qlinear_torch import QuantLinear as QuantLinearSym
        from auto_round_extension.torch.qlinear_torch_zp import QuantLinear as QuantLinearAsym
    except ImportError:
        raise ImportError("Install auto-round to use autoround quantized models")
    QuantLinear = QuantLinearSym if sym else QuantLinearAsym

    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_autoround_linear(module, module_to_convert, w_bit, group_size, sym)

        if isinstance(module, nn.Linear) and name in module_to_convert:
            model._modules[name] = QuantLinear(
                bits=w_bit,
                group_size=group_size,
                infeatures=module.in_features,
                outfeatures=module.out_features,
                bias=module.bias is not None,
            )
    return model
