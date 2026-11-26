# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Cast operator."""

from __future__ import annotations

__all__ = [
    "infer_bitcast",
    "infer_cast",
    "infer_cast_like",
]

import onnx_ir as ir

from onnx_shape_inference import _context, _registry


def _propagate_integer_values(
    ctx: _context.ShapeInferenceContext,
    data: ir.Value,
    output: ir.Value,
    output_dtype: ir.DataType | None,
) -> None:
    """Keep shape-tensor data only when the cast provably preserves its values."""
    input_dtype = data.dtype
    if (
        input_dtype is None
        or output_dtype is None
        or not input_dtype.is_integer()
        or not output_dtype.is_integer()
    ):
        ctx.clear_symbolic_value(output)
        return
    values = ctx.get_symbolic_value(data)
    if values is None:
        ctx.clear_symbolic_value(output)
        return

    def bounds(dtype: ir.DataType) -> tuple[int, int]:
        if dtype.is_signed():
            limit = 1 << (dtype.bitwidth - 1)
            return -limit, limit - 1
        return 0, (1 << dtype.bitwidth) - 1

    source_min, source_max = bounds(input_dtype)
    target_min, target_max = bounds(output_dtype)
    # Symbolic integers may take any value representable by the input dtype.
    # Concrete values permit a narrower conversion when each value still fits.
    if (target_min <= source_min and source_max <= target_max) or all(
        isinstance(value, int) and target_min <= value <= target_max for value in values
    ):
        ctx.set_symbolic_value(output, values)
    else:
        ctx.clear_symbolic_value(output)


@_registry.registry.register("", "Cast", since_version=1)
def infer_cast(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Cast operator.

    Shape is identical to the input; dtype comes from the ``to`` attribute.

    Spec: https://onnx.ai/onnx/operators/onnx__Cast.html
    """
    (data,) = _context.check_inputs(node, "input")
    to_attr = _context.require_attr(node, "to")

    output_dtype = ir.DataType(to_attr.as_int())

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], data.shape, output_dtype)

        _propagate_integer_values(ctx, data, node.outputs[0], output_dtype)


@_registry.registry.register("", "BitCast", since_version=26)
def infer_bitcast(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for BitCast operator.

    Same shape as the input; dtype comes from the ``to`` attribute.
    The target type must have the same bit-width as the input type.

    Spec: https://onnx.ai/onnx/operators/onnx__BitCast.html
    """
    (data,) = _context.check_inputs(node, "input")
    to_attr = _context.require_attr(node, "to")

    output_dtype = ir.DataType(to_attr.as_int())

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], data.shape, output_dtype)


@_registry.registry.register("", "CastLike", since_version=15)
def infer_cast_like(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for CastLike operator.

    Shape is identical to input[0]; dtype comes from input[1].

    Spec: https://onnx.ai/onnx/operators/onnx__CastLike.html
    """
    (data, target) = _context.check_inputs(node, "input", "target_type")

    output_dtype = target.dtype
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], data.shape, output_dtype)
        _propagate_integer_values(ctx, data, node.outputs[0], output_dtype)
