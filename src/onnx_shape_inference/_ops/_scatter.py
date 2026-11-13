# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for ScatterElements and ScatterND operators."""

from __future__ import annotations

__all__ = [
    "infer_scatter",
    "infer_tensor_scatter",
]

import onnx_ir as ir

from onnx_shape_inference import _context, _registry


@_registry.registry.register("", "Scatter", since_version=9)
@_registry.registry.register("", "ScatterElements", since_version=11)
@_registry.registry.register("", "ScatterND", since_version=11)
def infer_scatter(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Scatter, ScatterElements, and ScatterND.

    Output shape and dtype match the data (first) input.
    """
    (data, _indices, _updates) = _context.check_inputs(node, "data", "indices", "updates")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], data.shape, data.dtype)


@_registry.registry.register("", "TensorScatter", since_version=24)
def infer_tensor_scatter(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for TensorScatter."""
    (past_cache, _update) = _context.check_inputs(node, "past_cache", "update")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], past_cache.shape, past_cache.dtype)


@_registry.registry.register("", "ScatterND", since_version=29)
@_registry.registry.register("", "ScatterElements", since_version=29)
def infer_scatter_29(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Validate the new mean reduction while preserving the data shape and type."""
    (data, _indices, _updates) = _context.check_inputs(node, "data", "indices", "updates")
    reduction = node.attributes.get_string("reduction", "none")
    if reduction not in ("none", "add", "mul", "max", "min", "mean"):
        ctx.record_error(
            node,
            f"Invalid scatter reduction {reduction!r}; use none, add, mul, max, min, mean.",
        )
        return
    if reduction == "mean" and data.dtype in (
        ir.DataType.BOOL,
        ir.DataType.STRING,
        ir.DataType.COMPLEX64,
        ir.DataType.COMPLEX128,
    ):
        ctx.record_error(
            node,
            f"Scatter mean does not support {data.dtype}; cast to integer or floating point.",
        )
        return
    infer_scatter(ctx, node)
