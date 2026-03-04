import contextlib
import logging
import os
import torch
import torch.nn as nn
from torch.cuda import is_available as cuda_is_available


@contextlib.contextmanager
def _suppress_all_output():
    """Redirect stdout and stderr at the file-descriptor level.

    Unlike contextlib.redirect_stdout/stderr, this also suppresses output
    from C extensions that write directly to fd 1/fd 2 — e.g. the PyTorch
    C++ warning "[W302] torch.backends.cuda.preferred_linalg_library is an
    experimental feature" that gptqmodel triggers during initialisation.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(devnull_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


def replace_autoround_linear(
    model, module_to_convert=[], w_bit=4, group_size=128, packing_format="auto_round:auto_gptq", sym=True,
    module_to_not_convert=[]
):
    """Replace nn.Linear layers with AutoRound QuantLinear for quantized inference.

    The packing_format determines how qzeros are stored:
    - 'gptq' in packing_format: qzeros stored as (zero_point - 1) per GPTQ convention.
    - Otherwise: qzeros stored directly (direct zero-point).

    Backend preference order (fastest first):
    1. Marlin CUDA kernels (requires CUDA + gptqmodel, sym=True only) — fastest
    2. Triton GPU kernels (requires CUDA + triton) — fast GPU kernels
    3. PyTorch fallback — works everywhere, but dequantizes weights on every forward pass

    For Marlin layers, call post_init_autoround_linear(model) after loading the weights
    to repack them into the Marlin-optimized layout.

    Marlin validates layer dimensions at construction time and raises NotImplementedError
    for unsupported shapes (e.g. out_features not divisible by 64).  The Triton backend
    shares the same CUDA kernel block-size constraints, so it would also fail at runtime
    with an illegal memory access.  Such layers are therefore replaced with the
    pure-PyTorch backend, which has no CUDA kernel shape requirements.

    module_to_not_convert: list of module names (direct children) whose entire subtree
        should be skipped.  Use this for parent modules that were kept in fp16 during
        quantization (e.g. ``shared_experts`` in MoE models).
    """
    use_gptq_zp = "gptq" in packing_format
    QuantLinear, use_marlin = _get_autoround_quant_linear_cls(use_gptq_zp, sym)

    # Lazily computed fallback class for layers that Marlin rejects (e.g. out_features % 64 != 0).
    # Only populated on the first rejected layer; shared across the whole recursive call tree.
    fallback_cls = [None]

    def _get_fallback():
        if fallback_cls[0] is None:
            # Triton has the same CUDA kernel block-size alignment requirements as
            # Marlin (output dim must be divisible by the kernel's thread-block size).
            # Using Triton for layers that Marlin rejected causes CUDA illegal memory
            # accesses at inference time (asynchronous CUDA error in triton forward).
            # Fall back to the pure-PyTorch backend which works for any layer shape.
            fallback_cls[0], _ = _get_autoround_quant_linear_cls(use_gptq_zp, force_pytorch=True)
        return fallback_cls[0]

    for name, module in model.named_children():
        if name in module_to_not_convert:
            continue  # skip entire subtree — this parent was kept in fp16
        if len(list(module.children())) > 0:
            replace_autoround_linear(module, module_to_convert, w_bit, group_size, packing_format, sym, module_to_not_convert)

        if isinstance(module, nn.Linear) and name in module_to_convert:
            if use_marlin:
                try:
                    new_module = QuantLinear(
                        bits=w_bit,
                        group_size=group_size,
                        desc_act=False,
                        sym=sym,
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                    )
                except NotImplementedError:
                    # Marlin rejected this layer's dimensions — use next-best backend.
                    new_module = _get_fallback()(
                        bits=w_bit,
                        group_size=group_size,
                        infeatures=module.in_features,
                        outfeatures=module.out_features,
                        bias=module.bias is not None,
                    )
            else:
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

    This is required for Marlin layers: it repacks the GPTQ-format weights into Marlin's
    optimized memory layout and pre-allocates the Marlin workspace buffer.
    It is a no-op for Triton and PyTorch backends.
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


def _get_autoround_quant_linear_cls(use_gptq_zp: bool, sym: bool = True, force_pytorch: bool = False):
    """Return the best available QuantLinear class for AutoRound inference.

    Preference order:
    1. Marlin CUDA kernels (requires CUDA + gptqmodel, sym=True only) — fastest
    2. Triton kernels (requires CUDA + triton) — fast GPU kernels
    3. PyTorch fallback — works everywhere but dequantizes weights on every forward pass

    Args:
        use_gptq_zp: use GPTQ-style zero-point packing (affects which variant is imported).
        sym: when True, try the Marlin (fastest) backend first.
        force_pytorch: when True, skip all CUDA-kernel backends (Marlin and Triton) and
            return the pure-PyTorch backend directly.  Use this for layers whose dimensions
            do not meet the CUDA kernel alignment requirements of both Marlin and Triton
            (e.g. out_features not divisible by 64).

    Returns:
        (QuantLinear class, use_marlin: bool)
            use_marlin=True means the Marlin constructor signature must be used.
    """
    if not force_pytorch and cuda_is_available():
        # Marlin is fastest but only supports symmetric quantization
        if sym:
            try:
                # Suppress verbose gptqmodel initialisation output at the
                # file-descriptor level so that both Python-level writes
                # (logbar ASCII banner) and C++ direct fd writes
                # (PyTorch [W302] preferred_linalg_library warning) are
                # silenced.  get_marlin_layer() is called inside the block
                # because it triggers `import gptqmodel` at call-time.
                # After the block, silence the logbar logger for later calls.
                with _suppress_all_output():
                    from auto_round_extension.cuda.gptqmodel_marlin import get_marlin_layer
                    marlin_cls = get_marlin_layer()
                logging.getLogger("logbar").setLevel(logging.ERROR)

                return marlin_cls, True
            except ImportError:
                pass
        # Triton is the next best option on CUDA
        try:
            if use_gptq_zp:
                from auto_round_extension.triton.qlinear_tritonv2_zp import QuantLinear
            else:
                from auto_round_extension.triton.qlinear_tritonv2 import QuantLinear
            return QuantLinear, False
        except ImportError:
            pass
    # Fallback to pure PyTorch kernels
    try:
        if use_gptq_zp:
            from auto_round_extension.torch.qlinear_torch_zp import QuantLinear
        else:
            from auto_round_extension.torch.qlinear_torch import QuantLinear
        return QuantLinear, False
    except ImportError:
        raise ImportError("Install auto-round to use autoround quantized models")

