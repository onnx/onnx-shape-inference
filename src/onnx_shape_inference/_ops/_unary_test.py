# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for unary element-wise shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
import parameterized

from onnx_shape_inference import OpUsageError, ShapeInferenceError, registry
from onnx_shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
BOOL = ir.DataType.BOOL

_UNARY_OPS = [
    "Identity",
    "Neg",
    "Abs",
    "Ceil",
    "Floor",
    "Round",
    "Reciprocal",
    "Sqrt",
    "Exp",
    "Log",
    "Sigmoid",
    "Relu",
    "Tanh",
    "Erf",
    "Sign",
    "Sin",
    "Cos",
]


class UnaryTest(unittest.TestCase):
    """Tests for unary passthrough shape inference."""

    @parameterized.parameterized.expand([(op,) for op in _UNARY_OPS])
    def test_shape_passthrough(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT, ["batch", 128])],
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT, ["batch", 128])])

    @parameterized.parameterized.expand([(op,) for op in _UNARY_OPS])
    def test_concrete_shape(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT, [3, 4, 5])],
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4, 5])])

    @parameterized.parameterized.expand([(op,) for op in _UNARY_OPS])
    def test_missing_shape(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT)],
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT)])

    def test_relu_symbolic_dims(self):
        """Relu on ["N", "C"] → same shape ["N", "C"]."""
        actual = run_shape_inference(
            "",
            "Relu",
            [ts(FLOAT, ["N", "C"])],
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT, ["N", "C"])])

    @parameterized.parameterized.expand(
        [
            ("abs", "Abs", 1),
            ("mish", "Mish", 18),
            ("prelu", "PRelu", 1),
            ("cumsum", "CumSum", 11),
        ]
    )
    def test_earliest_shape_compatible_opset(self, _name, op_type, opset_version):
        inputs = [ts(FLOAT, [3, 4])]
        if op_type == "PRelu":
            inputs.append(ts(FLOAT, [1]))
        elif op_type == "CumSum":
            inputs.append(ts(ir.DataType.INT32, []))
        actual = run_shape_inference("", op_type, inputs, opset_version=opset_version)
        self.assertEqual(actual, [ts(FLOAT, [3, 4])])

    @parameterized.parameterized.expand(
        [
            ("not", "Not", BOOL, [3, 4], BOOL),
            ("isnan", "IsNaN", FLOAT, [2, 3], BOOL),
            ("isinf", "IsInf", FLOAT, [4, 5], BOOL),
        ]
    )
    def test_bool_output(self, _name, op_type, input_dtype, input_shape, output_dtype):
        actual = run_shape_inference(
            "", op_type, [ts(input_dtype, input_shape)], opset_version=17
        )
        self.assertEqual(actual, [ts(output_dtype, input_shape)])

    def test_unary_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Abs", [], opset_version=17)

    def test_unary_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Relu",
                [None],
                opset_version=17,
            )

    def test_logical_unary_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Not", [], opset_version=17)

    def test_logical_unary_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "IsNaN",
                [None],
                opset_version=17,
            )


class UnaryOpset29Test(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("Expm1",),
            ("Log1p",),
            ("LogCumSumExp",),
        ]
    )
    def test_new_shape_preserving_ops(self, op):
        for shape in (["B", 0, 7], None):
            self.assertEqual(
                run_shape_inference(
                    "", op, [ts(ir.DataType.BFLOAT16, shape)], opset_version=29
                ),
                [ts(ir.DataType.BFLOAT16, shape)],
            )
        self.assertIsNone(registry.get("", op, 28))

    def test_existing_mvn(self):
        self.assertEqual(
            run_shape_inference(
                "",
                "MeanVarianceNormalization",
                [ts(ir.DataType.BFLOAT16, ["B", 3, 7, 7])],
                {
                    "axes": ir.AttrInt64s("axes", [-1]),
                    "epsilon": ir.AttrFloat32("epsilon", 1e-05),
                },
                opset_version=29,
            ),
            [ts(ir.DataType.BFLOAT16, ["B", 3, 7, 7])],
        )

    @parameterized.parameterized.expand(
        [
            ("LogCumSumExp", [ts(FLOAT, [])], {}),
            ("LogCumSumExp", [ts(FLOAT, [3])], {"axis": 1}),
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


if __name__ == "__main__":
    unittest.main()
