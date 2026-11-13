# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ScatterElements, ScatterND, and TensorScatter shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
import parameterized

from onnx_shape_inference import OpUsageError, ShapeInferenceError
from onnx_shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class ScatterTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("scatter_elements", "ScatterElements", [3, 4]),
            ("scatter_nd", "ScatterND", [4, 5, 6]),
        ]
    )
    def test_basic(self, _name, op_type, input_shape):
        actual = run_shape_inference(
            "",
            op_type,
            [ts(FLOAT, input_shape), ts(INT64, [2, 3]), ts(FLOAT, [2, 3])],
            opset_version=18,
        )
        self.assertEqual(actual, [ts(FLOAT, input_shape)])

    @parameterized.parameterized.expand(
        [
            ("scatter_elements", "ScatterElements", ["N", "C"]),
            ("scatter_nd", "ScatterND", ["N", "C", "D"]),
        ]
    )
    def test_symbolic_dims(self, _name, op_type, input_shape):
        actual = run_shape_inference(
            "",
            op_type,
            [ts(FLOAT, input_shape), ts(INT64, [2, 3]), ts(FLOAT, [2, 3])],
            opset_version=18,
        )
        self.assertEqual(actual, [ts(FLOAT, input_shape)])

    @parameterized.parameterized.expand(
        [
            ("scatter_elements", "ScatterElements"),
            ("scatter_nd", "ScatterND"),
        ]
    )
    def test_none_input_raises(self, _name, op_type):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", op_type, [None], opset_version=18)

    @parameterized.parameterized.expand(
        [
            ("scatter_elements", "ScatterElements"),
            ("scatter_nd", "ScatterND"),
        ]
    )
    def test_missing_indices_raises(self, _name, op_type):
        """Missing indices/updates inputs should raise OpUsageError."""
        with self.assertRaises(OpUsageError):
            run_shape_inference("", op_type, [ts(FLOAT, [3, 4])], opset_version=18)


class TensorScatterTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "TensorScatter",
            [ts(FLOAT, [5, 3]), ts(INT64, [2, 1]), ts(FLOAT, [2, 3])],
            opset_version=24,
        )
        self.assertEqual(actual, [ts(FLOAT, [5, 3])])

    def test_optional_write_indices_omitted(self):
        actual = run_shape_inference(
            "",
            "TensorScatter",
            [ts(FLOAT, ["batch", 5, 3]), ts(FLOAT, [2, 3])],
            opset_version=24,
        )
        self.assertEqual(actual, [ts(FLOAT, ["batch", 5, 3])])


class ScatterOpset29Test(unittest.TestCase):
    @parameterized.parameterized.expand([("ScatterND",), ("ScatterElements",)])
    def test_scatter_mean(self, op):
        self.assertEqual(
            run_shape_inference(
                "",
                op,
                [
                    ts(ir.DataType.BFLOAT16, ["B", 3]),
                    ts(ir.DataType.INT32, [2, 2]),
                    ts(ir.DataType.BFLOAT16, [2]),
                ],
                {"reduction": ir.AttrString("reduction", "mean")},
                opset_version=29,
            ),
            [ts(ir.DataType.BFLOAT16, ["B", 3])],
        )
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                "",
                op,
                [
                    ts(ir.DataType.BOOL, [3]),
                    ts(ir.DataType.INT64, [1]),
                    ts(ir.DataType.BOOL, [1]),
                ],
                {"reduction": ir.AttrString("reduction", "mean")},
                opset_version=29,
            )


if __name__ == "__main__":
    unittest.main()
