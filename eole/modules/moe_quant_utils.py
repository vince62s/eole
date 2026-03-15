"""
moe_quant_utils.py
Utilities for detecting quantisation type and stacking per-expert int4
weights into batch tensors compatible with fused_moe_int4.py.

Supported backends
------------------
* gptqmodel Marlin    – MarlinQuantLinear  (Marlin-repacked int4, fast path)
* AutoGPTQ / AutoRound  – QuantLinear   (K-packed int4, kpacked=True)
* AutoAWQ               – WQLinear_GEMM (N-packed int4, kpacked=False)

Tensor layout after stacking (E = num_experts, gs = group_size)
----------------------------------------------------------------
GPTQ / AutoRound (kpacked=True):
  w1_qweight  (E, K//8,  2*I)     int32
  w1_scales   (E, K//gs, 2*I)     fp16/bf16
  w1_qzeros   (E, K//gs, 2*I//8)  int32
  w2_qweight  (E, I//8,  H)       int32
  w2_scales   (E, I//gs, H)       fp16/bf16
  w2_qzeros   (E, I//gs, H//8)    int32

AWQ (kpacked=False):
  w1_qweight  (E, K,     2*I//8)  int32
  w1_scales   (E, K//gs, 2*I)     fp16/bf16
  w1_qzeros   (E, K//gs, 2*I//8)  int32
  w2_qweight  (E, I,     H//8)    int32
  w2_scales   (E, I//gs, H)       fp16/bf16
  w2_qzeros   (E, I//gs, H//8)    int32

where K = in_features (= H for W1, = I for W2)
      I = per-stream intermediate size (W1 out_features = 2*I)
"""

from __future__ import annotations
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_expert_quant_type(experts) -> str:
    """
    Inspect the first expert gate_up_proj and return one of:
      'marlin' – gptqmodel Marlin-repacked int4 (uses gptq_marlin_gemm)
      'gptq'   – AutoGPTQ / AutoRound  (K-packed int4, Triton path)
      'awq'    – AutoAWQ               (N-packed int4, Triton path)
      'fp16'   – plain nn.Linear / anything else

    Note on gptqmodel Marlin layers
    --------------------------------
    gptqmodel's Marlin backend repacks the standard GPTQ qweight during
    ``post_init()`` into a tiled layout incompatible with our Triton kernel.
    We detect these layers via the "gptqmodel" namespace and return 'marlin'
    so that ``fused_marlin_moe_impl`` (using ``gptq_marlin_gemm``) is used
    instead of the Triton int4 path.
    """
    if not experts:
        return "fp16"

    layer = experts[0].gate_up_proj
    module_path = type(layer).__module__
    classname = type(layer).__name__

    # gptqmodel Marlin layers live under the "gptqmodel" namespace.
    # They expose qweight, scales, g_idx, g_idx_sort_indices, workspace.
    if "gptqmodel" in module_path and hasattr(layer, "qweight"):
        return "marlin"

    # AutoGPTQ / AutoRound: QuantLinear with qweight.
    # Distinguish from AWQ's QuantLinear by packing axis:
    #   GPTQ : qweight.shape[0] == in_features // 8  (K-packed)
    #   AWQ  : qweight.shape[0] == in_features        (N-packed)
    # Any other shape (e.g. Marlin-repacked qweight from a wrapper that does
    # not live under "gptqmodel") is treated as an unrecognised format and
    # falls through to 'fp16'.
    if classname == "QuantLinear" and hasattr(layer, "qweight"):
        in_f = (
            layer.infeatures
            if hasattr(layer, "infeatures")
            else layer.in_features if hasattr(layer, "in_features") else None
        )
        if in_f is not None:
            if layer.qweight.shape[0] == in_f:
                return "awq"
            if layer.qweight.shape[0] == in_f // 8:
                return "gptq"
            # Unrecognised packing – fall through to per-expert forward().
            return "fp16"
        return "fp16"

    if "WQLinear" in classname and hasattr(layer, "qweight"):
        return "awq"

    if "auto_round" in module_path and hasattr(layer, "qweight"):
        return "gptq"

    return "fp16"


# ─────────────────────────────────────────────────────────────────────────────
# Low-level tensor extractors
# ─────────────────────────────────────────────────────────────────────────────


def _gptq_tensors(layer):
    """Return (qweight, scales, qzeros, group_size) from a GPTQ QuantLinear.

    qweight : (K//8, N)      int32
    scales  : (K//gs, N)     fp16
    qzeros  : (K//gs, N//8)  int32
    """
    qw, sc, qz = layer.qweight, layer.scales, layer.qzeros
    gs = (
        layer.group_size
        if (hasattr(layer, "group_size") and layer.group_size > 0)
        else (qw.shape[0] * 8) // sc.shape[0]
    )
    return qw, sc, qz, gs


def _awq_tensors(layer):
    """Return (qweight, scales, qzeros, group_size) from an AWQ WQLinear.

    qweight : (K, N//8)      int32
    scales  : (K//gs, N)     fp16
    qzeros  : (K//gs, N//8)  int32
    """
    qw, sc, qz = layer.qweight, layer.scales, layer.qzeros
    if hasattr(layer, "group_size") and layer.group_size > 0:
        gs = layer.group_size
    else:
        gs = qw.shape[0] // sc.shape[0]
    return qw, sc, qz, gs


def _cat_n(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Concatenate along the last (output / N) dimension."""
    return torch.cat([t1, t2], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Weight stacking
# ─────────────────────────────────────────────────────────────────────────────


def stack_gptq_moe_weights(experts, device: torch.device):
    """
    Stack per-expert GPTQ / AutoRound int4 weights into batched tensors.

    Returns
    -------
    w1_qweight  (E, K//8,  2*I)     int32
    w1_scales   (E, K//gs, 2*I)     fp16
    w1_qzeros   (E, K//gs, 2*I//8)  int32
    w2_qweight  (E, I//8,  H)       int32
    w2_scales   (E, I//gs, H)       fp16
    w2_qzeros   (E, I//gs, H//8)    int32
    group_size  int
    """
    w1_qw, w1_sc, w1_qz = [], [], []
    w2_qw, w2_sc, w2_qz = [], [], []
    group_size = None

    for e in experts:
        qw_g, sc_g, qz_g, gs = _gptq_tensors(e.gate_up_proj)
        if group_size is None:
            group_size = gs

        up = getattr(e, "up_proj", None)
        if up is not None:
            qw_u, sc_u, qz_u, _ = _gptq_tensors(up)
            qw1 = _cat_n(qw_g, qw_u)
            sc1 = _cat_n(sc_g, sc_u)
            qz1 = _cat_n(qz_g, qz_u)
        else:
            qw1, sc1, qz1 = qw_g, sc_g, qz_g

        w1_qw.append(qw1.unsqueeze(0))
        w1_sc.append(sc1.unsqueeze(0))
        w1_qz.append(qz1.unsqueeze(0))

        qw_d, sc_d, qz_d, _ = _gptq_tensors(e.down_proj)
        w2_qw.append(qw_d.unsqueeze(0))
        w2_sc.append(sc_d.unsqueeze(0))
        w2_qz.append(qz_d.unsqueeze(0))

        _free_expert_weights_single(e)

    return (
        torch.cat(w1_qw, dim=0).to(device),
        torch.cat(w1_sc, dim=0).to(device),
        torch.cat(w1_qz, dim=0).to(device),
        torch.cat(w2_qw, dim=0).to(device),
        torch.cat(w2_sc, dim=0).to(device),
        torch.cat(w2_qz, dim=0).to(device),
        group_size,
    )


def stack_awq_moe_weights(experts, device: torch.device):
    """
    Stack per-expert AWQ int4 weights into batched tensors.

    Returns the same 7-tuple as stack_gptq_moe_weights but N-packed:
    w1_qweight  (E, K,     2*I//8)  int32
    w1_scales   (E, K//gs, 2*I)     fp16
    w1_qzeros   (E, K//gs, 2*I//8)  int32
    w2_qweight  (E, I,     H//8)    int32
    w2_scales   (E, I//gs, H)       fp16
    w2_qzeros   (E, I//gs, H//8)    int32
    group_size  int
    """
    w1_qw, w1_sc, w1_qz = [], [], []
    w2_qw, w2_sc, w2_qz = [], [], []
    group_size = None

    for e in experts:
        qw_g, sc_g, qz_g, gs = _awq_tensors(e.gate_up_proj)
        if group_size is None:
            group_size = gs

        up = getattr(e, "up_proj", None)
        if up is not None:
            qw_u, sc_u, qz_u, _ = _awq_tensors(up)
            qw1 = _cat_n(qw_g, qw_u)
            sc1 = _cat_n(sc_g, sc_u)
            qz1 = _cat_n(qz_g, qz_u)
        else:
            qw1, sc1, qz1 = qw_g, sc_g, qz_g

        w1_qw.append(qw1.unsqueeze(0))
        w1_sc.append(sc1.unsqueeze(0))
        w1_qz.append(qz1.unsqueeze(0))

        qw_d, sc_d, qz_d, _ = _awq_tensors(e.down_proj)
        w2_qw.append(qw_d.unsqueeze(0))
        w2_sc.append(sc_d.unsqueeze(0))
        w2_qz.append(qz_d.unsqueeze(0))

        _free_expert_weights_single(e)

    return (
        torch.cat(w1_qw, dim=0).to(device),
        torch.cat(w1_sc, dim=0).to(device),
        torch.cat(w1_qz, dim=0).to(device),
        torch.cat(w2_qw, dim=0).to(device),
        torch.cat(w2_sc, dim=0).to(device),
        torch.cat(w2_qz, dim=0).to(device),
        group_size,
    )


def stack_fp16_moe_weights(experts, device: torch.device):
    """Stack plain fp16/bf16 expert weights for vectorized MoE forward."""
    w1, w2 = [], []

    for e in experts:
        gate_up = e.gate_up_proj.weight.data  # (2I, H)
        down = e.down_proj.weight.data  # (H, I)

        w1.append(gate_up.unsqueeze(0))
        w2.append(down.unsqueeze(0))

        _free_expert_weights_single(e)

    return (
        torch.cat(w1, dim=0).to(device),
        torch.cat(w2, dim=0).to(device),
    )


def _free_expert_weights_single(expert):
    """Free one expert's weight tensors immediately after stacking."""
    for proj_name in ("gate_up_proj", "up_proj", "down_proj"):
        proj = getattr(expert, proj_name, None)
        if proj is None:
            continue
        # quantized backends
        for attr in ("qweight", "scales", "qzeros"):
            if hasattr(proj, attr):
                setattr(proj, attr, None)
        # fp16 plain linear
        if hasattr(proj, "weight") and proj.weight is not None:
            proj.weight = None
    torch.cuda.empty_cache()


def _marlin_tensors(layer):
    """Return the Marlin-specific tensors from a gptqmodel MarlinQuantLinear.

    After ``post_init()`` the layer stores:
      qweight          – Marlin-repacked int4 weights
      scales           – Marlin-permuted scales
      qzeros           – empty tensor (Marlin uses symmetric quant)
      g_idx            – empty or sorted g_idx (activation reordering)
      g_idx_sort_indices – permutation for g_idx
      workspace        – pre-allocated Marlin workspace
      weight_type      – ScalarType (e.g. scalar_types.uint4b8)
      is_k_full        – whether K dimension is fully populated
    """
    return (
        layer.qweight,
        layer.scales,
        layer.qzeros,
        getattr(layer, "g_idx", None),
        getattr(layer, "g_idx_sort_indices", None),
        getattr(layer, "workspace", None),
    )


def stack_marlin_moe_weights(experts, device: torch.device):
    """Collect per-expert Marlin weight tensors into lists for batched inference.

    Unlike the GPTQ/AWQ stacking functions, Marlin weights cannot simply be
    cat'd into (E, ...) tensors because ``gptq_marlin_gemm`` operates on
    individual expert weights.  We therefore return lists of per-expert
    tensors plus the shared metadata needed to call the kernel.

    Returns
    -------
    w1_qweight   list[E]  Marlin qweight for W1 (gate+up stacked)
    w1_scales    list[E]  Marlin scales for W1
    w1_qzeros    list[E]  (empty) qzeros for W1
    w1_g_idx     list[E]  g_idx for W1
    w1_g_idx_sort list[E] sort_indices for W1
    w2_qweight   list[E]  Marlin qweight for W2
    w2_scales    list[E]  Marlin scales for W2
    w2_qzeros    list[E]  (empty) qzeros for W2
    w2_g_idx     list[E]  g_idx for W2
    w2_g_idx_sort list[E] sort_indices for W2
    workspaces   list[E]  Marlin workspace tensors (one per expert)
    quant_type          ScalarType from the first expert
    in_features         int  (H – hidden size)
    intermediate_features int (I – per-expert intermediate)
    out_features        int  (H – hidden size, same as in_features for MoE)
    is_k_full           bool
    """
    w1_qw, w1_sc, w1_qz, w1_gi, w1_gs = [], [], [], [], []
    w2_qw, w2_sc, w2_qz, w2_gi, w2_gs = [], [], [], [], []
    workspaces = []

    quant_type = None
    in_features = intermediate_features = out_features = None
    is_k_full = True

    for e in experts:
        gate_up = e.gate_up_proj
        down    = e.down_proj

        qw1, sc1, qz1, gi1, gs1, ws1 = _marlin_tensors(gate_up)
        qw2, sc2, qz2, gi2, gs2, ws2 = _marlin_tensors(down)

        if quant_type is None:
            quant_type        = getattr(gate_up, "weight_type", None)
            in_features       = gate_up.in_features
            # w1_out_features is the actual output of W1 (gate+up combined for
            # gated MoE, or a single projection for non-gated MoE).
            # intermediate_features is the actual K fed into W2 (= down.in_features),
            # which equals gate_up.out_features // 2 for gated MoE and
            # gate_up.out_features for non-gated MoE.
            w1_out_features   = gate_up.out_features         # actual W1 output: 2*I (gated) or I (non-gated)
            intermediate_features = down.in_features         # K for W2 = I always
            out_features      = down.out_features
            is_k_full         = getattr(gate_up, "is_k_full", True)

        w1_qw.append(qw1.to(device) if qw1 is not None else qw1)
        w1_sc.append(sc1.to(device) if sc1 is not None else sc1)
        w1_qz.append(qz1.to(device) if qz1 is not None else qz1)
        w1_gi.append(gi1.to(device) if gi1 is not None else gi1)
        w1_gs.append(gs1.to(device) if gs1 is not None else gs1)

        w2_qw.append(qw2.to(device) if qw2 is not None else qw2)
        w2_sc.append(sc2.to(device) if sc2 is not None else sc2)
        w2_qz.append(qz2.to(device) if qz2 is not None else qz2)
        w2_gi.append(gi2.to(device) if gi2 is not None else gi2)
        w2_gs.append(gs2.to(device) if gs2 is not None else gs2)

        # Keep workspace on its device; create one if missing
        workspaces.append(ws1.to(device) if ws1 is not None else None)

    return (
        w1_qw, w1_sc, w1_qz, w1_gi, w1_gs,
        w2_qw, w2_sc, w2_qz, w2_gi, w2_gs,
        workspaces,
        quant_type,
        in_features,
        w1_out_features,
        intermediate_features,
        out_features,
        is_k_full,
    )
