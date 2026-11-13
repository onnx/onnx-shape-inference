# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Gemm operator."""

from __future__ import annotations

__all__ = [
    "infer_gemm",
]

import onnx_ir as ir

from onnx_shape_inference import _context, _registry


@_registry.registry.register("", "Gemm", since_version=7)
def infer_gemm(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Gemm operator.

    Computes Y = alpha * A' * B' + beta * C, where A' and B' are optionally
    transposed.  Output is always 2-D: (M, N).

    Spec: https://onnx.ai/onnx/operators/onnx__Gemm.html
    """
    (input_a, input_b) = _context.check_inputs(node, "A", "B")
    output_dtype = input_a.dtype or input_b.dtype
    _gemm_shape(ctx, node, input_a.shape, input_b.shape, output_dtype)


def _gemm_shape(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    shape_a: ir.Shape | None,
    shape_b: ir.Shape | None,
    output_dtype: ir.DataType | None,
) -> None:
    """Infer rank-2 Gemm geometry with an operator-specific output dtype."""
    if shape_a is None or shape_b is None:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        return

    if shape_a.rank() != 2:
        ctx.record_error(node, f"Gemm input A must be rank 2, got {shape_a.rank()}")
        return
    if shape_b.rank() != 2:
        ctx.record_error(node, f"Gemm input B must be rank 2, got {shape_b.rank()}")
        return

    trans_a_attr = node.attributes.get("transA")
    trans_a = trans_a_attr.as_int() if trans_a_attr is not None else 0
    trans_b_attr = node.attributes.get("transB")
    trans_b = trans_b_attr.as_int() if trans_b_attr is not None else 0

    m_dim = shape_a[1] if trans_a else shape_a[0]
    n_dim = shape_b[0] if trans_b else shape_b[1]
    k_dim_a = shape_a[0] if trans_a else shape_a[1]
    k_dim_b = shape_b[1] if trans_b else shape_b[0]

    output_shape = ir.Shape([m_dim, n_dim])
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)
    if isinstance(k_dim_a, int) and isinstance(k_dim_b, int) and k_dim_a != k_dim_b:
        ctx.record_error(
            node,
            f"Incompatible Gemm contraction dimensions: {k_dim_a} vs {k_dim_b}",
        )


@_registry.registry.register("", "Gemm", since_version=29)
def infer_gemm_29(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer Gemm with rank >= 1 A and a rank-2 B."""
    (a, b) = _context.check_inputs(node, "A", "B")
    if b.shape is not None and b.shape.rank() != 2:
        ctx.record_error(node, f"Gemm B has rank {b.shape.rank()}; reshape B to rank 2.")
        return
    if a.shape is None:
        ctx.set_shape_and_dtype(node.outputs[0], None, a.dtype or b.dtype)
        return
    dims = list(a.shape)
    if not dims:
        ctx.record_error(node, "Gemm A is scalar; reshape A to rank >= 1.")
        return
    if node.attributes.get_int("transA", 0) and len(dims) > 1:
        dims[-2], dims[-1] = dims[-1], dims[-2]
    n = (
        ctx.new_symbolic_dim()
        if b.shape is None
        else b.shape[0 if node.attributes.get_int("transB", 0) else 1]
    )
    if b.shape is not None:
        k = b.shape[1 if node.attributes.get_int("transB", 0) else 0]
        if isinstance(k, int) and isinstance(dims[-1], int) and k != dims[-1]:
            ctx.record_error(
                node, f"Gemm contraction dimensions {dims[-1]} and {k} differ; align A and B."
            )
            return
    result = [*dims[:-1], n]
    if len(node.inputs) > 2 and (c := node.inputs[2]) is not None and c.shape is not None:
        if c.shape.rank() > len(result) or any(
            isinstance(cd, int) and isinstance(yd, int) and cd not in (1, yd)
            for cd, yd in zip(reversed(c.shape), reversed(result))
        ):
            ctx.record_error(
                node, f"Gemm C shape {c.shape} cannot broadcast to {result}; reshape C."
            )
            return
    ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(result), a.dtype or b.dtype)


@_registry.registry.register("", "GroupedMatMul", since_version=29)
def infer_grouped_matmul(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer [M, S, N] and unify shared token, group, and contraction dimensions."""
    (data, weights, indices) = _context.check_inputs(node, "input", "weights", "group_indices")
    dimensions: dict[str, int | ir.SymbolicDim] = {}
    trans_b = node.attributes.get_int("transB", 0)
    inputs = [
        (data, ("M", "K")),
        (weights, ("G", "N", "K") if trans_b else ("G", "K", "N")),
        (indices, ("M", "S")),
    ]
    if len(node.inputs) > 3 and node.inputs[3] is not None:
        inputs.append((node.inputs[3], ("G", "N")))
    for value, labels in inputs:
        if value.shape is None:
            continue
        if value.shape.rank() != len(labels):
            ctx.record_error(
                node, f"GroupedMatMul {value.name} has shape {value.shape}; expected {labels}."
            )
            return
        for label, dim in zip(labels, value.shape):
            previous = dimensions.get(label)
            if isinstance(previous, int) and isinstance(dim, int) and previous != dim:
                ctx.record_error(
                    node,
                    f"GroupedMatMul {label} dimensions {previous} and {dim} differ; align inputs.",
                )
                return
            if previous is None or isinstance(dim, int):
                dimensions[label] = dim
    shape = ir.Shape(
        [
            dimensions[label] if label in dimensions else ctx.new_symbolic_dim()
            for label in ("M", "S", "N")
        ]
    )
    ctx.set_shape_and_dtype(node.outputs[0], shape, data.dtype)
