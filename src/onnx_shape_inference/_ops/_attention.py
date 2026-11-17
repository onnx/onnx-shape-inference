# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for attention operators."""

from __future__ import annotations

__all__ = [
    "infer_attention",
    "infer_flex_attention",
    "infer_linear_attention",
    "infer_linear_attention_29",
    "infer_rotary_embedding",
]

import onnx_ir as ir

from onnx_shape_inference import _context, _registry
from onnx_shape_inference._ops import _utils

_reg = _registry.registry.register


def _attr_int(node: ir.Node, name: str) -> int | None:
    """Read an INT attribute, or ``None`` when absent."""
    attr = node.attributes.get(name)
    return attr.as_int() if attr is not None else None


def _head_size(
    ctx: _context.ShapeInferenceContext,
    hidden: int | ir.SymbolicDim | None,
    num_heads: int | None,
) -> int | ir.SymbolicDim:
    """Split a packed hidden size into per-head size.

    Returns ``hidden // num_heads`` when the packed hidden size and head count
    are available, preserving a symbolic expression when ``hidden`` is symbolic.
    A fresh symbolic dim is used when the split is not statically resolvable.
    """
    if isinstance(num_heads, int) and num_heads > 0 and hidden is not None:
        if isinstance(hidden, int) and hidden % num_heads != 0:
            return ctx.new_symbolic_dim()
        return _utils.floor_div_dim(hidden, num_heads)
    return ctx.new_symbolic_dim()


@_reg("", "Attention", since_version=23)
def infer_attention(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Attention operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Attention.html

    3D: Q=[B,Sq,Hq], K=[B,Sk,Hk], V=[B,Sk,Hv] -> Y=[B,Sq,Hq]
        (Hq = q_num_heads * head_size, output preserves Q's hidden dim)
    4D: Q=[B,Nh,Sq,Hd], K=[B,Nkv,Sk,Hd], V=[B,Nkv,Sk,Hdv] -> Y=[B,Nh,Sq,Hdv]
    present_key/value: if past is provided, present is past shape with seq extended
    """
    (q, k, v) = _context.check_inputs(node, "Q", "K", "V")

    output_dtype = q.dtype

    # Output[0]: Y shape
    # 3D: Y=[B, Sq, q_num_heads * v_head_size]
    # 4D: Y=[B, q_num_heads, Sq, v_head_size]
    if len(node.outputs) > 0:
        output_shape: ir.Shape | None = None
        if q.shape is not None:
            rank = q.shape.rank()
            if rank == 3 and v.shape is not None:
                # Output hidden = q_num_heads * v_head_size
                q_nh_attr = node.attributes.get("q_num_heads")
                kv_nh_attr = node.attributes.get("kv_num_heads")
                q_nh = q_nh_attr.as_int() if q_nh_attr is not None else None
                kv_nh = kv_nh_attr.as_int() if kv_nh_attr is not None else None
                v_hidden = v.shape[2]
                if q_nh is not None and kv_nh is not None:
                    v_head_size = _head_size(ctx, v_hidden, kv_nh)
                    out_hidden: int | ir.SymbolicDim = q_nh * v_head_size
                else:
                    out_hidden = v_hidden
                output_shape = ir.Shape([q.shape[0], q.shape[1], out_hidden])
            elif rank == 4 and v.shape is not None:
                output_shape = ir.Shape([q.shape[0], q.shape[1], q.shape[2], v.shape[3]])
            else:
                output_shape = q.shape
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)

    # present_key and present_value
    past_key = node.inputs[4] if len(node.inputs) > 4 and node.inputs[4] is not None else None
    past_value = (
        node.inputs[5] if len(node.inputs) > 5 and node.inputs[5] is not None else None
    )

    if len(node.outputs) > 1 and node.outputs[1] is not None:
        pk_shape = _attention_present_shape(k, past_key)
        ctx.set_shape_and_dtype(node.outputs[1], pk_shape, output_dtype)

    if len(node.outputs) > 2 and node.outputs[2] is not None:
        pv_shape = _attention_present_shape(v, past_value)
        ctx.set_shape_and_dtype(node.outputs[2], pv_shape, output_dtype)

    # Output[3]: qk_matmul_output = [B, q_num_heads, Sq, total_seq_length]
    if len(node.outputs) > 3 and node.outputs[3] is not None:
        qk_shape: ir.Shape | None = None
        if q.shape is not None:
            rank = q.shape.rank()
            batch = q.shape[0]

            # Get q_num_heads: from attribute (3D) or Q shape dim 1 (4D)
            qk_nh_attr = node.attributes.get("q_num_heads")
            qk_q_nh: int | ir.SymbolicDim | None = None
            if qk_nh_attr is not None:
                qk_q_nh = qk_nh_attr.as_int()
            elif rank == 4:
                qk_q_nh = q.shape[1]

            sq = q.shape[1] if rank == 3 else q.shape[2]

            # total_seq_length: past_seq + current_seq (from K)
            total_seq: int | ir.SymbolicDim = ctx.new_symbolic_dim()
            pk_shape = _attention_present_shape(k, past_key)
            if pk_shape is not None:
                total_seq = pk_shape[pk_shape.rank() - 2]
            elif k.shape is not None:
                k_seq = k.shape[k.shape.rank() - 2]
                total_seq = k_seq

            if qk_q_nh is not None:
                qk_shape = ir.Shape([batch, qk_q_nh, sq, total_seq])
        ctx.set_shape_and_dtype(node.outputs[3], qk_shape, output_dtype)


def _attention_present_shape(current: ir.Value, past: ir.Value | None) -> ir.Shape | None:
    """Compute present_key/present_value shape.

    When past is provided, present matches past shape with seq dim extended.
    When past is None, present = current shape.
    """
    if past is not None and past.shape is not None:
        past_rank = past.shape.rank()
        seq_idx = past_rank - 2
        past_seq = past.shape[seq_idx]
        if current.shape is not None:
            cur_rank = current.shape.rank()
            cur_seq = current.shape[cur_rank - 2]
        else:
            return None
        if isinstance(past_seq, int) and isinstance(cur_seq, int):
            total_seq: int | ir.SymbolicDim = past_seq + cur_seq
        else:
            total_seq = past_seq + cur_seq  # type: ignore[assignment]
        dims = list(past.shape.dims)
        dims[seq_idx] = total_seq
        return ir.Shape(dims)
    if current.shape is None:
        return None
    return current.shape


@_reg("", "RotaryEmbedding", since_version=23)
def infer_rotary_embedding(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for RotaryEmbedding operator.

    Spec: https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html
    """
    (input_val,) = _context.check_inputs(node, "input")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, input_val.dtype)

    # Optional second output (output_position_ids)
    if len(node.outputs) > 1 and node.outputs[1] is not None:
        position_ids_shape: ir.Shape | None = None
        if len(node.inputs) > 1 and node.inputs[1] is not None:
            position_ids_shape = node.inputs[1].shape
        ctx.set_shape_and_dtype(node.outputs[1], position_ids_shape, ir.DataType.INT64)


@_reg("", "LinearAttention", since_version=27)
def infer_linear_attention(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for LinearAttention operator.

    Spec: https://onnx.ai/onnx/operators/onnx__LinearAttention.html

    query/key/value are 3-D packed ``[B, T, H*D]``; ``q_num_heads`` and
    ``kv_num_heads`` are required.  Outputs:

    * ``output``        ``[B, T, q_num_heads * d_v]`` where
      ``d_v = value_hidden // kv_num_heads`` (T type = query dtype).
    * ``present_state`` ``[B, kv_num_heads, d_k, d_v]`` with
      ``d_k = key_hidden // kv_num_heads``; equal to ``past_state`` when that
      optional input is provided.
    """
    (q, k, v) = _context.check_inputs(node, "query", "key", "value")
    past_state = (
        node.inputs[3] if len(node.inputs) > 3 and node.inputs[3] is not None else None
    )

    q_num_heads = _attr_int(node, "q_num_heads")
    kv_num_heads = _attr_int(node, "kv_num_heads")
    output_dtype = q.dtype

    # Output 0: [B, T, q_num_heads * d_v]
    if len(node.outputs) > 0:
        output_shape: ir.Shape | None = None
        if q.shape is not None and q.shape.rank() == 3:
            v_hidden = v.shape[2] if v.shape is not None else None
            d_v = _head_size(ctx, v_hidden, kv_num_heads)
            if q_num_heads is not None:
                out_hidden: int | ir.SymbolicDim = q_num_heads * d_v
            else:
                out_hidden = ctx.new_symbolic_dim()
            output_shape = ir.Shape([q.shape[0], q.shape[1], out_hidden])
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)

    # Output 1: present_state [B, kv_num_heads, d_k, d_v]
    if len(node.outputs) > 1 and node.outputs[1] is not None:
        present_dtype = (
            past_state.dtype
            if past_state is not None and past_state.dtype is not None
            else output_dtype
        )
        present_shape: ir.Shape | None = None
        if (
            past_state is not None
            and past_state.shape is not None
            and past_state.shape.rank() == 4
        ):
            # past_state is ground truth for the recurrent state buffer.
            present_shape = past_state.shape
        elif q.shape is not None and q.shape.rank() == 3 and kv_num_heads is not None:
            k_hidden = k.shape[2] if k.shape is not None else None
            v_hidden = v.shape[2] if v.shape is not None else None
            d_k = _head_size(ctx, k_hidden, kv_num_heads)
            d_v = _head_size(ctx, v_hidden, kv_num_heads)
            present_shape = ir.Shape([q.shape[0], kv_num_heads, d_k, d_v])
        ctx.set_shape_and_dtype(node.outputs[1], present_shape, present_dtype)


@_reg("", "LinearAttention", since_version=29)
def infer_linear_attention_29(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer compact Q/K, expanded value heads and optional state trajectories.

    Spec: https://onnx.ai/onnx/operators/onnx__LinearAttention.html

    Zero/omitted v_num_heads retains query GQA. A positive value expands Q/K
    adjacently and makes both output and state use that many heads.
    """
    q, k, v = _context.check_inputs(node, "query", "key", "value")
    q_heads = _context.require_attr(node, "q_num_heads").as_int()
    k_heads = _context.require_attr(node, "kv_num_heads").as_int()
    v_heads = _attr_int(node, "v_num_heads") or 0
    if q_heads <= 0 or k_heads <= 0 or q_heads % k_heads:
        ctx.record_error(
            node,
            f"q_num_heads ({q_heads}) must be a positive multiple of "
            f"kv_num_heads ({k_heads}); set both head counts to positive compatible values.",
        )
        return
    if v_heads < 0 or (v_heads > 0 and (q_heads != k_heads or v_heads % k_heads)):
        ctx.record_error(
            node,
            f"v_num_heads ({v_heads}) must be zero, or a positive multiple of "
            f"kv_num_heads ({k_heads}) with q_num_heads ({q_heads}) == kv_num_heads.",
        )
        return
    state_heads = v_heads or k_heads
    output_heads = v_heads or q_heads
    past = node.inputs[3] if len(node.inputs) > 3 else None
    decay = node.inputs[4] if len(node.inputs) > 4 else None
    beta = node.inputs[5] if len(node.inputs) > 5 else None
    rule_attr = node.attributes.get("update_rule")
    rule = rule_attr.as_string() if rule_attr is not None else "gated_delta"
    if rule not in ("linear", "gated", "delta", "gated_delta"):
        ctx.record_error(
            node, f"Unknown update_rule {rule!r}; use linear, gated, delta, or gated_delta."
        )
        return
    if (decay is not None) != (rule in ("gated", "gated_delta")) or (beta is not None) != (
        rule in ("delta", "gated_delta")
    ):
        ctx.record_error(
            node,
            f"update_rule={rule!r} has incompatible decay/beta inputs; supply decay "
            "only for gated/gated_delta and beta only for delta/gated_delta.",
        )
        return

    dims: dict[str, int | ir.SymbolicDim] = {"heads": state_heads}

    def unify(name: str, dim: int | ir.SymbolicDim) -> bool:
        existing = dims.get(name)
        if isinstance(existing, int) and isinstance(dim, int) and existing != dim:
            ctx.record_error(
                node,
                f"Inconsistent {name} dimensions: {existing} and {dim}; "
                "align query/key/value, state, and gate shapes.",
            )
            return False
        if existing is None or isinstance(dim, int):
            dims[name] = dim
        return True

    for value, names in (
        (q, ("B", "T", "Q")),
        (k, ("B", "T", "K")),
        (v, ("B", "T", "V")),
        (past, ("B", "heads", "Dk", "Dv")),
        (decay, ("B", "T", "decay")),
        (beta, ("B", "T", "beta")),
    ):
        if value is None or value.shape is None:
            continue
        if value.shape.rank() != len(names):
            ctx.record_error(
                node,
                f"Input {value.name!r} has rank {value.shape.rank()}; "
                f"LinearAttention requires rank {len(names)} for dimensions {names}.",
            )
            return
        for name, dim in zip(names, value.shape, strict=True):
            if not unify(name, dim):
                return

    for packed, heads, name in (
        ("Q", q_heads, "Dk"),
        ("K", k_heads, "Dk"),
        ("V", state_heads, "Dv"),
    ):
        hidden = dims.get(packed)
        if hidden is None:
            continue
        if isinstance(hidden, int) and (hidden <= 0 or hidden % heads):
            ctx.record_error(
                node,
                f"Packed {packed} width ({hidden}) must be positive and divisible "
                f"by its head count ({heads}); use positive integral head widths.",
            )
            return
        if not unify(name, _utils.floor_div_dim(hidden, heads)):
            return
    for name in ("Dk", "Dv"):
        dim = dims.get(name)
        if isinstance(dim, int) and dim <= 0:
            ctx.record_error(node, f"{name} must be positive, got {dim}; fix the state shape.")
            return
    dk = dims.get("Dk")
    decay_width = dims.get("decay")
    if isinstance(decay_width, int) and (
        decay_width <= 0
        or (
            decay_width not in (state_heads, state_heads * dk)
            if isinstance(dk, int)
            else decay_width % state_heads != 0
        )
    ):
        ctx.record_error(
            node,
            f"decay width ({decay_width}) must be {state_heads} or "
            f"{state_heads} * Dk; match the value/state head layout.",
        )
        return
    beta_width = dims.get("beta")
    if isinstance(beta_width, int) and beta_width not in (1, state_heads):
        ctx.record_error(
            node,
            f"beta width ({beta_width}) must be 1 or {state_heads}; "
            "match the value/state head layout.",
        )
        return
    for name in ("B", "T", "Dk", "Dv"):
        if name not in dims:
            dims[name] = ctx.new_symbolic_dim()
    batch, time, dk, dv = (dims[name] for name in ("B", "T", "Dk", "Dv"))
    state_dtype = past.dtype if past is not None else q.dtype
    shapes = (
        ([batch, time, output_heads * dv], q.dtype),
        ([batch, state_heads, dk, dv], state_dtype),
        ([batch, time, state_heads, dk, dv], state_dtype),
    )
    for output, (shape, dtype) in zip(node.outputs, shapes):
        if output.name != "":
            ctx.set_shape_and_dtype(output, ir.Shape(shape), dtype)


@_reg("ai.onnx.preview", "FlexAttention", since_version=1)
def infer_flex_attention(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for ai.onnx.preview FlexAttention operator.

    Inputs are rank-4:

    * Q ``[B, q_num_heads, q_seq, head_size]``
    * K ``[B, kv_num_heads, kv_seq, head_size]``
    * V ``[B, kv_num_heads, kv_seq, v_head_size]``

    Output ``Y = [B, q_num_heads, q_seq, v_head_size]`` (same element type as Q).
    """
    (q, _k, v) = _context.check_inputs(node, "Q", "K", "V")

    if len(node.outputs) > 0:
        output_shape: ir.Shape | None
        if (
            q.shape is not None
            and q.shape.rank() == 4
            and v.shape is not None
            and v.shape.rank() == 4
        ):
            output_shape = ir.Shape([q.shape[0], q.shape[1], q.shape[2], v.shape[3]])
        else:
            output_shape = q.shape
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, q.dtype)
