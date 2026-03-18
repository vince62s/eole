"""
eole/modules/marlin_utils.py

Utility functions for the Marlin quantization backend.
Adapted from gptqmodel/utils/marlin.py (Apache-2.0).
Original source: https://github.com/ModelCloud/GPTQModel
"""

from typing import Callable, List, Optional, Tuple, Union

import torch

from .marlin_scalar_type import ScalarType


def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def marlin_make_workspace(device: torch.device, max_blocks_per_sm: int = 4) -> torch.Tensor:
    """Allocate the Marlin workspace (lock tensor for barrier sync).

    ``max_blocks_per_sm=4`` matches the MoE kernel's worst-case requirement and
    ensures the workspace is large enough for both decode and prefill batches.
    """
    try:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        sms = 160
    return torch.zeros(sms * max_blocks_per_sm, dtype=torch.int32, device=device)


def get_scale_perms() -> Tuple[List[int], List[int]]:
    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: List[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int, group_size: int) -> torch.Tensor:
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def marlin_sort_g_idx(g_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices


def marlin_make_empty_g_idx(device: torch.device) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )


def update_tensor_inplace(dst: torch.Tensor, src: torch.Tensor) -> None:
    assert dst.dtype == src.dtype, "Tensors must have the same dtype"
    with torch.no_grad():
        dst.as_strided_(src.shape, src.stride())
        if dst.data_ptr() != src.data_ptr():
            dst.copy_(src)
            del src


def replace_parameter(
    mod: torch.nn.Module,
    name: str,
    new: Union[torch.Tensor, torch.nn.Parameter],
) -> None:
    """Replace a registered parameter in-place when possible, else re-register."""
    old = getattr(mod, name)
    if (
        type(old) is type(new)
        and old.dtype == new.dtype
        and old.untyped_storage().nbytes() == new.untyped_storage().nbytes()
    ):
        update_tensor_inplace(old, new)
    else:
        if not isinstance(new, torch.nn.Parameter):
            new = torch.nn.Parameter(new, requires_grad=False)
        mod.register_parameter(name, torch.nn.Parameter(new, requires_grad=False))


def _transform_param(
    layer: torch.nn.Module,
    name: Optional[str],
    fn: Callable,
) -> None:
    if name is not None and getattr(layer, name, None) is not None:
        old_param = getattr(layer, name)
        new_param = fn(old_param)
        replace_parameter(
            layer, name, torch.nn.Parameter(new_param.data, requires_grad=False)
        )


def should_use_atomic_add_reduce(
    m: int, n: int, k: int, device: torch.device, dtype: torch.dtype
) -> bool:
    # atomicAdd is faster only for small N (narrow output) with large K.
    _ATOMIC_ADD_MAX_N = 2048  # atomicAdd not beneficial above this output width
    _ATOMIC_ADD_MIN_K = 2048  # need sufficient reduction depth for atomicAdd to win
    if n >= _ATOMIC_ADD_MAX_N or k < _ATOMIC_ADD_MIN_K or device.type != "cuda":
        return False
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        return False
    return True


def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    """Repack GPTQ-packed weights into Marlin tiled layout via eole._ops."""
    from eole import _ops

    return _ops.gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)


def gptq_marlin_gemm(
    a: torch.Tensor,
    c: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_bias: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    b_zeros: Optional[torch.Tensor],
    g_idx: Optional[torch.Tensor],
    perm: Optional[torch.Tensor],
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    """Dense GPTQ-Marlin GEMM using the dedicated dense kernel.

    Calls ``eole._ops.gptq_marlin_gemm`` directly — no MoE routing tensors
    are allocated or passed.

    Args:
        a: Activation tensor of shape ``(M, K)``, half or bfloat16.
        c: Optional pre-allocated output tensor of shape ``(M, N)``.
        b_q_weight: Marlin-repacked quantized weight, shape ``(K//16, N*bits//32)``.
        b_bias: Optional bias tensor of shape ``(N,)``.
        b_scales: Scale tensor, shape ``(num_groups, N)``.
        global_scale: Reserved for nvfp4 (pass ``None`` for GPTQ).
        b_zeros: Optional zero-points tensor (asymmetric quant).
        g_idx: Optional group-index tensor for act-order (shape ``(K,)``).
        perm: Optional permutation tensor for act-order (shape ``(K,)``).
        workspace: Zeros tensor used as lock buffer (see marlin_make_workspace).
        b_q_type: ScalarType describing the weight quantisation (e.g. uint4b8).
        size_m: Batch size (number of rows of A/C).
        size_n: Output dimension (number of columns of C).
        size_k: Reduction dimension (number of columns of A / rows of B).
        is_k_full: Whether K is fully covered by groups (True for GPTQ).
        use_atomic_add: Use atomic-add for small-N cross-block reduction.
        use_fp32_reduce: Use fp32 tmp buffer for cross-block reduction.
        is_zp_float: Whether zero-points are stored as float (HQQ).

    Returns:
        Output tensor of shape ``(M, N)`` in the same dtype as ``a``.
    """
    from eole import _ops

    return _ops.gptq_marlin_gemm(
        a,
        c,
        b_q_weight,
        b_bias,
        b_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )


def apply_gptq_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    wtype: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    is_k_full: bool,
    bias: Optional[torch.Tensor] = None,
    use_fp32_reduce: bool = True,
    use_atomics: bool = False,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomics = use_atomics and should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )

    output = gptq_marlin_gemm(
        reshaped_x,
        None,
        weight,
        bias,
        weight_scale,
        None,  # global_scale
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        wtype,
        size_m=reshaped_x.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        is_k_full=is_k_full,
        use_atomic_add=use_atomics,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )

    return output.reshape(out_shape)


__all__ = [
    "marlin_is_k_full",
    "marlin_make_workspace",
    "get_scale_perms",
    "marlin_permute_scales",
    "marlin_sort_g_idx",
    "marlin_make_empty_g_idx",
    "replace_parameter",
    "_transform_param",
    "should_use_atomic_add_reduce",
    "gptq_marlin_repack",
    "gptq_marlin_gemm",
    "apply_gptq_marlin_linear",
]
