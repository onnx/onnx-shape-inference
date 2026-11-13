# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for packed codes and block-quantized tensors."""

from __future__ import annotations

__all__ = ["infer_pack", "infer_unpack", "infer_dequantize_packed"]

import math

import onnx_ir as ir

from onnx_shape_inference import _context, _registry
from onnx_shape_inference._ops import _utils


def _parameters(ctx: _context.ShapeInferenceContext, node: ir.Node) -> tuple[int, int] | None:
    bits = node.attributes.get_int("bits", None)
    radix = node.attributes.get_int("radix", None)
    group = node.attributes.get_int("group_size", None)
    if bits is not None:
        if radix is not None or group is not None or not 1 <= bits <= 8:
            ctx.record_error(node, "Use bits in [1, 8], without radix or group_size.")
            return None
        return 1, bits
    if radix is None or group is None or not 2 <= radix <= 256 or not 1 <= group <= 62:
        ctx.record_error(
            node,
            "Specify bits, or radix in [2, 256] and positive group_size with radix^group_size < 2^63.",
        )
        return None
    limit = radix**group
    if limit >= 2**63:
        ctx.record_error(node, "radix^group_size exceeds INT64; reduce radix or group_size.")
        return None
    return group, (limit - 1).bit_length()


def _infer_packing(ctx: _context.ShapeInferenceContext, node: ir.Node, unpack: bool) -> None:
    if unpack:
        data, count_input = _context.check_inputs(node, "X", "count")
    else:
        (data,) = _context.check_inputs(node, "X")
        count_input = None
    parameters = _parameters(ctx, node)
    if parameters is None:
        return
    group, width = parameters
    storage = (
        data.dtype
        if unpack
        else node.attributes.get_int("output_type", int(ir.DataType.UINT8))
    )
    if storage is not None and storage not in (ir.DataType.UINT8, ir.DataType.UINT32):
        ctx.record_error(
            node, f"Packed storage type {storage} is invalid; use UINT8 or UINT32."
        )
        return
    word_bits = 32 if storage == ir.DataType.UINT32 else 8
    dtype = ir.DataType.UINT8 if unpack else ir.DataType(storage)
    count: int | ir.SymbolicDim | None = None
    if count_input is not None:
        if count_input.dtype not in (None, ir.DataType.INT64) or (
            count_input.shape is not None and count_input.shape.rank() != 0
        ):
            ctx.record_error(
                node, "Unpack count must be a scalar INT64 tensor; reshape or cast count."
            )
            return
        values = _utils.get_known_dim_values(ctx, count_input)
        if values is not None:
            if len(values) != 1 or (isinstance(values[0], int) and values[0] < 0):
                ctx.record_error(node, "Unpack count must contain one nonnegative integer.")
                return
            count = values[0]
    if data.shape is None:
        ctx.set_shape_and_dtype(node.outputs[0], None, dtype)
        return
    rank = data.shape.rank()
    axis = node.attributes.get_int("axis", -1)
    if rank == 0 or not -rank <= axis < rank:
        ctx.record_error(
            node,
            f"Packing axis {axis} is invalid for rank {rank}; use rank >= 1 and a valid axis.",
        )
        return
    axis %= rank
    dims = list(data.shape)
    if unpack:
        if count is None:
            count = (
                dims[axis]
                if storage is not None and width == word_bits and group == 1
                else ctx.new_symbolic_dim()
            )
        if isinstance(count, int) and isinstance(dims[axis], int) and storage is not None:
            expected = _utils.ceil_div_dim(
                _utils.ceil_div_dim(count, group) * width, word_bits
            )
            if dims[axis] != expected:
                ctx.record_error(
                    node,
                    f"Unpack axis has {dims[axis]} words but count requires {expected}; correct count or input shape.",
                )
                return
        dims[axis] = count
    else:
        dims[axis] = _utils.ceil_div_dim(
            _utils.ceil_div_dim(dims[axis], group) * width, word_bits
        )
        if isinstance(dims[axis], int) and dims[axis] >= 2**63:
            ctx.record_error(
                node, "Packed dimension exceeds INT64; reduce input length or packing width."
            )
            return
    ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(dims), dtype)


@_registry.registry.register("", "Pack", since_version=29)
def infer_pack(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    _infer_packing(ctx, node, False)


@_registry.registry.register("", "Unpack", since_version=29)
def infer_unpack(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    _infer_packing(ctx, node, True)


_BLOCKS = {
    "q4_0": (32, 18),
    "q4_1": (32, 20),
    "q5_0": (32, 22),
    "q5_1": (32, 24),
    "q8_0": (32, 34),
    "q8_1": (32, 36),
    "q2_k": (256, 84),
    "q3_k": (256, 110),
    "q4_k": (256, 144),
    "q5_k": (256, 176),
    "q6_k": (256, 210),
    "q8_k": (256, 292),
    "iq2_xxs": (256, 66),
    "iq2_xs": (256, 74),
    "iq3_xxs": (256, 98),
    "iq1_s": (256, 50),
    "iq4_nl": (32, 18),
    "iq3_s": (256, 110),
    "iq2_s": (256, 82),
    "iq4_xs": (256, 136),
    "iq1_m": (256, 56),
    "tq1_0": (256, 54),
    "tq2_0": (256, 66),
    "mxfp4": (32, 17),
    "nvfp4": (64, 36),
    "q1_0": (128, 18),
    "q2_0": (64, 18),
}


@_registry.registry.register("", "DequantizePacked", since_version=29)
def infer_dequantize_packed(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    (data,) = _context.check_inputs(node, "x")
    format_ = _context.require_attr(node, "format").as_string()
    dims = list(_context.require_attr(node, "shape").as_ints())
    block = _BLOCKS.get(format_[5:]) if format_.startswith("gguf:") else None
    if block is None:
        ctx.record_error(
            node, f"Unsupported quantization format {format_!r}; use a specified gguf: format."
        )
        return
    dtype = node.attributes.get_int("output_dtype", int(ir.DataType.FLOAT))
    if dtype not in (
        ir.DataType.FLOAT,
        ir.DataType.FLOAT16,
        ir.DataType.BFLOAT16,
        ir.DataType.DOUBLE,
    ):
        ctx.record_error(
            node, f"Invalid output_dtype {dtype}; use FLOAT, FLOAT16, BFLOAT16, or DOUBLE."
        )
        return
    if not dims or any(d < 0 for d in dims) or dims[-1] % block[0]:
        ctx.record_error(
            node,
            f"Invalid logical shape {dims}; use rank >= 1, nonnegative dimensions, and last dimension divisible by {block[0]}.",
        )
        return
    size = math.prod(dims[:-1]) * (dims[-1] // block[0]) * block[1]
    if size >= 2**63:
        ctx.record_error(node, "Packed byte count exceeds INT64; reduce logical shape.")
        return
    if data.shape is not None and (
        data.shape.rank() != 1 or (isinstance(data.shape[0], int) and data.shape[0] != size)
    ):
        ctx.record_error(
            node,
            f"Packed input shape {data.shape} is invalid; provide a vector of exactly {size} bytes.",
        )
        return
    ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(dims), ir.DataType(dtype))
