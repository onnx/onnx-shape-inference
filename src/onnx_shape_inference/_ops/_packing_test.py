# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Pack, Unpack, and DequantizePacked shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
import parameterized

from onnx_shape_inference import OpUsageError, ShapeInferenceError, registry
from onnx_shape_inference._ops._testing import (
    const_value,
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class PackingOpset29Test(unittest.TestCase):
    @parameterized.parameterized.expand([("Pack",), ("Unpack",), ("DequantizePacked",)])
    def test_new_operator_version_boundary(self, op):
        self.assertIsNone(registry.get("", op, 28))
        self.assertIsNotNone(registry.get("", op, 29))

    def test_unpack_unknown_storage_type_does_not_assume_word_width(self):
        data = ir.Value(shape=ir.Shape([3]))
        count = const_value(8)
        self.assertEqual(
            run_shape_inference_with_values(
                "",
                "Unpack",
                [data, count],
                {"bits": ir.AttrInt64("bits", 8)},
                opset_version=29,
            ),
            [ts(ir.DataType.UINT8, [8])],
        )

    @parameterized.parameterized.expand(
        [
            ("bits3", [2, 8], {"bits": 3}, ir.DataType.UINT8, [2, 3]),
            (
                "word32",
                [2, 7],
                {"bits": 5, "output_type": int(ir.DataType.UINT32)},
                ir.DataType.UINT32,
                [2, 2],
            ),
            ("grouped", [2, 11], {"radix": 3, "group_size": 5}, ir.DataType.UINT8, [2, 3]),
            ("axis", [7, 2], {"bits": 3, "axis": 0}, ir.DataType.UINT8, [3, 2]),
            ("zero", [0, 0], {"bits": 7}, ir.DataType.UINT8, [0, 0]),
            ("identity", ["B", "N"], {"bits": 8}, ir.DataType.UINT8, ["B", "N"]),
            ("symbolic", ["B", "N"], {"bits": 4}, ir.DataType.UINT8, ["B", "-floor(-N/2)"]),
            ("unknown", None, {"bits": 3}, ir.DataType.UINT8, None),
        ]
    )
    def test_pack(self, _name, shape, attributes, dtype, expected):
        self.assertEqual(
            run_shape_inference(
                "",
                "Pack",
                [ts(ir.DataType.UINT8, shape)],
                {attr.name: attr for attr in ir.convenience.convert_attributes(attributes)},
                opset_version=29,
            ),
            [ts(dtype, expected)],
        )

    @parameterized.parameterized.expand(
        [
            ("bits", ir.DataType.UINT8, [2, 3], 8, {"bits": 3}, [2, 8]),
            ("grouped", ir.DataType.UINT8, [2, 3], 11, {"radix": 3, "group_size": 5}, [2, 11]),
            ("word32", ir.DataType.UINT32, [2, 2], 7, {"bits": 5}, [2, 7]),
            ("zero", ir.DataType.UINT8, [2, 0], 0, {"bits": 3}, [2, 0]),
            ("runtime", ir.DataType.UINT8, ["B", 3], None, {"bits": 3}, ["B", "_d0"]),
        ]
    )
    def test_unpack(self, _name, dtype, shape, count, attributes, expected):
        data = ir.Value(type=ir.TensorType(dtype), shape=ir.Shape(shape))
        count_input = (
            const_value(count)
            if count is not None
            else ir.Value(type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([]))
        )
        self.assertEqual(
            run_shape_inference_with_values(
                "",
                "Unpack",
                [data, count_input],
                {attr.name: attr for attr in ir.convenience.convert_attributes(attributes)},
                opset_version=29,
            ),
            [ts(ir.DataType.UINT8, expected)],
        )

    def test_symbolic_unpack_count(self):
        data = ir.Value(type=ir.TensorType(ir.DataType.UINT8), shape=ir.Shape(["B", "bytes"]))
        count = ir.Value(type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([]))
        self.assertEqual(
            run_shape_inference_with_values(
                "",
                "Unpack",
                [data, count],
                {"bits": ir.AttrInt64("bits", 3)},
                opset_version=29,
                symbolic_values={1: [ir.SymbolicDim("N")]},
            ),
            [ts(ir.DataType.UINT8, ["B", "N"])],
        )

    @parameterized.parameterized.expand(
        [
            ("q4", "gguf:q4_0", [2, 32], 36),
            ("iq", "gguf:iq1_m", [2, 256], 112),
            ("nv", "gguf:nvfp4", [64], 36),
            ("empty", "gguf:q4_0", [0, 32], 0),
        ]
    )
    def test_dequantize(self, _name, format_, shape, size):
        self.assertEqual(
            run_shape_inference(
                "",
                "DequantizePacked",
                [ts(ir.DataType.UINT8, [size])],
                {
                    "format": ir.AttrString("format", format_),
                    "shape": ir.AttrInt64s("shape", shape),
                    "output_dtype": ir.AttrInt64("output_dtype", int(ir.DataType.BFLOAT16)),
                },
                opset_version=29,
            ),
            [ts(ir.DataType.BFLOAT16, shape)],
        )

    @parameterized.parameterized.expand(
        [
            ("Pack", [ts(ir.DataType.UINT8, [7])], {"bits": 0}),
            ("Pack", [ts(ir.DataType.UINT8, [7])], {"bits": 2, "radix": 3}),
            ("Pack", [ts(ir.DataType.UINT8, [7])], {"radix": 256, "group_size": 8}),
            ("Pack", [ts(ir.DataType.UINT8, [])], {"bits": 3}),
            ("Pack", [ts(ir.DataType.UINT8, [7])], {"bits": 3, "axis": 1}),
            ("Pack", [ts(ir.DataType.UINT8, [7])], {"bits": 3, "output_type": int(FLOAT)}),
            (
                "DequantizePacked",
                [ts(ir.DataType.UINT8, [36])],
                {"format": "GGUF:q4_0", "shape": [2, 32]},
            ),
            (
                "DequantizePacked",
                [ts(ir.DataType.UINT8, [36])],
                {"format": "gguf:q4_0", "shape": [2, 31]},
            ),
            (
                "DequantizePacked",
                [ts(ir.DataType.UINT8, [35])],
                {"format": "gguf:q4_0", "shape": [2, 32]},
            ),
        ]
    )
    def test_errors(self, op, inputs, attributes):
        for policy in ("override", "skip"):
            expected = (
                self.assertRaises(ShapeInferenceError)
                if policy == "override"
                else self.assertLogs("onnx_shape_inference._context", level="WARNING")
            )
            with expected:
                run_shape_inference(
                    "",
                    op,
                    inputs,
                    {
                        attr.name: attr
                        for attr in ir.convenience.convert_attributes(attributes)
                    },
                    opset_version=29,
                    policy=policy,
                )

    def test_unpack_invalid_count_and_storage(self):
        data = ir.Value(type=ir.TensorType(ir.DataType.UINT8), shape=ir.Shape([3]))
        for count in (const_value(-1), const_value([8]), const_value(2)):
            with self.assertRaises(ShapeInferenceError):
                run_shape_inference_with_values(
                    "",
                    "Unpack",
                    [data, count],
                    {"bits": ir.AttrInt64("bits", 3)},
                    opset_version=29,
                )
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "", "Unpack", [data, None], {"bits": ir.AttrInt64("bits", 3)}, opset_version=29
            )
