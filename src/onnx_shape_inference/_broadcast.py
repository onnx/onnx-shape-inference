# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Broadcasting utilities for shape inference."""

from __future__ import annotations

__all__ = [
    "broadcast_shapes",
]

import onnx_ir as ir

from onnx_shape_inference import _context


def broadcast_shapes(
    shape1: ir.Shape | None,
    shape2: ir.Shape | None,
) -> ir.Shape | None:
    """Compute a broadcast shape following NumPy broadcasting rules.

    Equal dimensions and singleton axes determine the result. Different
    concrete non-singleton dimensions are incompatible. Unrelated symbolic
    axes remain unresolved because either can be a runtime singleton.

    Args:
        shape1: First shape (can be None if unknown).
        shape2: Second shape (can be None if unknown).

    Returns:
        The broadcasted shape, or None for incompatible or unresolved shapes.
        Inference handlers use the context-aware helper to retain known axes
        and allocate unique symbols for the unresolved axes.

    Example::

        >>> broadcast_shapes(ir.Shape([3, 1, 5]), ir.Shape([1, 4, 5]))
        Shape([3, 4, 5])
        >>> broadcast_shapes(ir.Shape(["batch", 1]), ir.Shape([1, "width"]))
        Shape([SymbolicDim('batch'), SymbolicDim('width')])
        >>> broadcast_shapes(ir.Shape(["N", 4]), ir.Shape(["M", 4])) is None
        True
    """
    return _broadcast_shapes(shape1, shape2)


def _broadcast_shapes(
    shape1: ir.Shape | None,
    shape2: ir.Shape | None,
    *,
    ctx: _context.ShapeInferenceContext | None = None,
) -> ir.Shape | None:
    """Retain unresolved axes using the inference context's symbol allocator."""
    if shape1 is None or shape2 is None:
        return None

    # Get dimensions, padding shorter shape with 1s on the left
    dims1 = list(shape1.dims)
    dims2 = list(shape2.dims)

    max_rank = max(len(dims1), len(dims2))

    # Pad with 1s on the left
    while len(dims1) < max_rank:
        dims1.insert(0, 1)
    while len(dims2) < max_rank:
        dims2.insert(0, 1)

    result_dims: list[int | ir.SymbolicDim] = []
    equalities = (
        {frozenset(pair) for pair in ctx.symbolic_equalities} if ctx is not None else set()
    )

    for d1, d2 in zip(dims1, dims2):
        broadcasted = _broadcast_dim(d1, d2)
        if broadcasted is None:
            if ctx is None or (isinstance(d1, int) and isinstance(d2, int)):
                return None
            if frozenset((str(d1), str(d2))) in equalities:
                # Prefer a model author's name over a generated placeholder.
                broadcasted = (
                    d2
                    if ctx.is_generated_symbol(str(d1))
                    and not ctx.is_generated_symbol(str(d2))
                    else d1
                )
            else:
                broadcasted = ctx.new_symbolic_dim()
        result_dims.append(broadcasted)

    return ir.Shape(result_dims)


def _broadcast_dim(
    dim1: int | ir.SymbolicDim,
    dim2: int | ir.SymbolicDim,
) -> int | ir.SymbolicDim | None:
    """Broadcast two dimensions.

    Returns the broadcasted dimension, or None if incompatible.
    """
    # Both concrete integers
    if isinstance(dim1, int) and isinstance(dim2, int):
        if dim1 == dim2:
            return dim1
        if dim1 == 1:
            return dim2
        if dim2 == 1:
            return dim1
        # Different non-1 values - incompatible
        return None

    # One is concrete 1 - broadcasts to the other
    if dim1 == 1:
        return dim2
    if dim2 == 1:
        return dim1

    # One concrete, one symbolic
    if isinstance(dim1, int) and isinstance(dim2, ir.SymbolicDim):
        # Concrete non-1 wins over symbolic if we assume they're compatible
        return dim1
    if isinstance(dim2, int) and isinstance(dim1, ir.SymbolicDim):
        return dim2

    # A named symbol denotes one runtime extent. Independent symbols may each
    # be 1, so compatibility alone does not establish equality with either.
    if isinstance(dim1, ir.SymbolicDim) and isinstance(dim2, ir.SymbolicDim):
        if dim1.value is not None and dim2.value is not None and dim1 == dim2:
            return dim1
    return None
