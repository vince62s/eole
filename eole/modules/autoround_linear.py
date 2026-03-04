import gc
import threading
import torch.nn as nn
from torch.cuda import is_available as cuda_is_available

_marlin_preflight_done = False


def _preflight_marlin_import():
    """Import gptqmodel marlin and retroactively patch any pre-existing triton.Autotuner instances.

    gptqmodel's nogil_patcher patches triton.Autotuner.__init__ to add a _cache_lock
    threading.Lock for thread-safe compilation-cache access.  Instances that already exist
    when gptqmodel is imported (e.g. those created by FLA's @triton.autotune decorators at
    module-load time, via gated_delta_net.py → fla.modules.convolution) are missing this
    attribute and will crash when gptqmodel's second thread calls the patched run().

    This function:
    1. Imports gptqmodel marlin (side effect: patches triton.Autotuner at class level).
    2. Walks all live Python objects via gc and adds _cache_lock to any Autotuner instance
       that was created before the patch and is therefore missing the attribute.

    Because retroactive patching is order-independent — it finds and fixes pre-existing
    instances regardless of when it runs — this function may be called at any point before
    inference begins, even after triton and FLA have already been imported.

    A module-level flag ensures the gc scan runs at most once per process.

    Safe to call unconditionally: silently returns if gptqmodel is not installed or CUDA
    is unavailable.
    """
    global _marlin_preflight_done
    if _marlin_preflight_done:
        return
    if not cuda_is_available():
        return
    try:
        from auto_round_extension.cuda.gptqmodel_marlin import get_marlin_layer  # side effect: patches triton.Autotuner
    except ImportError:
        return

    # Retroactively add _cache_lock to Autotuner instances that predate the patch.
    # These are typically created by FLA's @triton.autotune decorators at module-load time.
    try:
        from triton.runtime.autotuner import Autotuner

        for obj in gc.get_objects():
            if isinstance(obj, Autotuner) and not hasattr(obj, "_cache_lock"):
                obj._cache_lock = threading.Lock()
    except (ImportError, AttributeError):
        pass

    _marlin_preflight_done = True


def replace_autoround_linear(
    model,
    module_to_convert=[],
    w_bit=4,
    group_size=128,
    packing_format="auto_round:auto_gptq",
    sym=True,
    module_to_not_convert=[],
):
    """Replace nn.Linear layers with AutoRound QuantLinear for quantized inference.

    The packing_format determines how qzeros are stored:
    - 'gptq' in packing_format: qzeros stored as (zero_point - 1) per GPTQ convention.
    - Otherwise: qzeros stored directly (direct zero-point).

    Backend preference order:
    1. Marlin CUDA kernels (requires CUDA + gptqmodel, sym=True only) — fastest
    2. Triton GPU kernels (requires CUDA + triton) — fast GPU kernels
    3. PyTorch fallback — works everywhere

    For Marlin layers, call post_init_autoround_linear(model) after loading the weights
    to repack them into the Marlin-optimized layout.

    Marlin validates layer dimensions at construction time and raises NotImplementedError
    for unsupported shapes (e.g. out_features not divisible by 64).  Such layers are
    replaced with the pure-PyTorch backend, which has no CUDA kernel shape requirements.

    module_to_not_convert: list of module names (direct children) whose entire subtree
        should be skipped.  Use this for parent modules that were kept in fp16 during
        quantization (e.g. ``shared_experts`` in MoE models).
    """
    use_gptq_zp = "gptq" in packing_format
    QuantLinear, use_marlin = _get_autoround_quant_linear_cls(use_gptq_zp, sym)

    # Lazily computed fallback class for layers that Marlin rejects.
    fallback_cls = None

    def _get_fallback():
        nonlocal fallback_cls
        if fallback_cls is None:
            fallback_cls, _ = _get_autoround_quant_linear_cls(use_gptq_zp, force_pytorch=True)
        return fallback_cls

    for name, module in model.named_children():
        if name in module_to_not_convert:
            continue  # skip entire subtree — this parent was kept in fp16
        if len(list(module.children())) > 0:
            replace_autoround_linear(
                module, module_to_convert, w_bit, group_size, packing_format, sym, module_to_not_convert
            )

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
                    # Marlin rejected this layer's dimensions — use PyTorch fallback.
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

    Required for Marlin layers: repacks GPTQ-format weights into Marlin's optimized
    memory layout and pre-allocates the workspace buffer.  No-op for other backends.
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
    3. PyTorch fallback — works everywhere

    When Marlin is used, gptqmodel's nogil_patcher patches triton.Autotuner with
    thread-safety locks.  Libraries such as FLA (used in gated_delta_net.py) create
    Autotuner instances at module-load time and will be missing _cache_lock if
    gptqmodel was not yet imported.  Call _preflight_marlin_import() before inference
    to retroactively add _cache_lock to all pre-existing Autotuner instances.

    Args:
        use_gptq_zp: use GPTQ-style zero-point packing.
        sym: when True, try Marlin first.
        force_pytorch: skip all CUDA-kernel backends; return PyTorch directly.

    Returns:
        (QuantLinear class, use_marlin: bool)
    """
    if not force_pytorch and cuda_is_available():
        if sym:
            try:
                from auto_round_extension.cuda.gptqmodel_marlin import get_marlin_layer

                return get_marlin_layer(), True
            except ImportError:
                pass
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
