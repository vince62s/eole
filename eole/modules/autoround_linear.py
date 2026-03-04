import torch
import torch.nn as nn
from torch.cuda import is_available as cuda_is_available


def replace_autoround_linear(
    model, module_to_convert=[], w_bit=4, group_size=128, packing_format="auto_round:auto_gptq", sym=True,
    module_to_not_convert=[]
):
    """Replace nn.Linear layers with AutoRound QuantLinear for quantized inference.

    The packing_format determines how qzeros are stored:
    - 'gptq' in packing_format: qzeros stored as (zero_point - 1) per GPTQ convention.
    - Otherwise: qzeros stored directly (direct zero-point).

    Backend preference order:
    1. Triton GPU kernels (requires CUDA + triton) — fast GPU kernels
    2. PyTorch fallback — works everywhere

    module_to_not_convert: list of module names (direct children) whose entire subtree
        should be skipped.  Use this for parent modules that were kept in fp16 during
        quantization (e.g. ``shared_experts`` in MoE models).
    """
    use_gptq_zp = "gptq" in packing_format
    QuantLinear = _get_autoround_quant_linear_cls(use_gptq_zp)

    for name, module in model.named_children():
        if name in module_to_not_convert:
            continue  # skip entire subtree — this parent was kept in fp16
        if len(list(module.children())) > 0:
            replace_autoround_linear(module, module_to_convert, w_bit, group_size, packing_format, sym, module_to_not_convert)

        if isinstance(module, nn.Linear) and name in module_to_convert:
            new_module = QuantLinear(
                bits=w_bit,
                group_size=group_size,
                infeatures=module.in_features,
                outfeatures=module.out_features,
                bias=module.bias is not None,
            )
            model._modules[name] = new_module
    return model


def post_init_autoround_linear(model):
    """Call post_init() on all AutoRound QuantLinear modules in the model.

    Must be called after weights have been loaded and the model moved to CUDA.
    """
    for name, module in model.named_children():
        module_cls = type(module)
        module_pkg = getattr(module_cls, "__module__", "") or ""
        if module_pkg.startswith("auto_round_extension") and hasattr(module, "post_init"):
            module.post_init()
        else:
            # Recurse into containers that are not themselves QuantLinear modules.
            post_init_autoround_linear(module)


def _get_autoround_quant_linear_cls(use_gptq_zp: bool):
    """Return the best available QuantLinear class for AutoRound inference.

    Preference order:
    1. Triton kernels (requires CUDA + triton) — fast GPU kernels
    2. PyTorch fallback — works everywhere

    Args:
        use_gptq_zp: use GPTQ-style zero-point packing (affects which variant is imported).

    Returns:
        QuantLinear class
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

