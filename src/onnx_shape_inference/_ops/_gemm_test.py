# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Gemm shape inference."""

from __future__ import annotations

import unittest

import onnx
import onnx_ir as ir
import parameterized

from onnx_shape_inference import OpUsageError, ShapeInferenceError, registry
from onnx_shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class GemmTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (dtype.name, dtype, shape)
            for dtype in (ir.DataType.FLOAT16, ir.DataType.BFLOAT16)
            for shape in ([7, 5], None)
        ]
    )
    def test_preserves_input_dtype(self, _name, dtype, shape):
        actual = run_shape_inference(
            "", "Gemm", [ts(dtype, shape), ts(dtype, [5, 11])], opset_version=17
        )
        self.assertEqual(actual, [ts(dtype, [7, 11] if shape is not None else None)])

    @parameterized.parameterized.expand(
        [
            ("basic", [7, 5], [5, 11], 0, 0, [7, 11]),
            ("trans_a", [5, 7], [5, 11], 1, 0, [7, 11]),
            ("trans_b", [7, 5], [11, 5], 0, 1, [7, 11]),
            ("both_trans", [5, 7], [11, 5], 1, 1, [7, 11]),
            ("symbolic", ["M", 64], [64, "N"], 0, 0, ["M", "N"]),
            ("symbolic_trans_a", [64, "M"], [64, "N"], 1, 0, ["M", "N"]),
            ("symbolic_trans_b", ["M", 64], ["N", 64], 0, 1, ["M", "N"]),
        ]
    )
    def test_gemm(self, _name, shape_a, shape_b, trans_a, trans_b, expected_shape):
        attrs = {}
        if trans_a:
            attrs["transA"] = ir.Attr("transA", ir.AttributeType.INT, trans_a)
        if trans_b:
            attrs["transB"] = ir.Attr("transB", ir.AttributeType.INT, trans_b)
        actual = run_shape_inference(
            "",
            "Gemm",
            [ts(FLOAT, shape_a), ts(FLOAT, shape_b)],
            attrs or None,
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_no_bias(self):
        """From ONNX test_gemm_no_bias: C is optional."""
        actual = run_shape_inference(
            "",
            "Gemm",
            [ts(FLOAT, [7, 5]), ts(FLOAT, [5, 11])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [7, 11])])

    def test_with_bias(self):
        """Gemm with bias (C) — does not affect output shape."""
        actual = run_shape_inference(
            "",
            "Gemm",
            [ts(FLOAT, [7, 5]), ts(FLOAT, [5, 11]), ts(FLOAT, [11])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [7, 11])])

    def test_missing_shape(self):
        actual = run_shape_inference(
            "",
            "Gemm",
            [ts(FLOAT), ts(FLOAT, [5, 11])],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_gemm_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Gemm", [], opset_version=17)

    def test_gemm_none_input(self):
        v = ir.Value(name="a", type=ir.TensorType(FLOAT), shape=ir.Shape([3, 4]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Gemm",
                [v, None],
                opset_version=17,
            )

    def test_gemm_wrong_rank_a(self):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                "",
                "Gemm",
                [ts(FLOAT, [2, 3, 4]), ts(FLOAT, [4, 5])],
                opset_version=17,
            )

    def test_gemm_wrong_rank_b(self):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                "",
                "Gemm",
                [ts(FLOAT, [2, 3]), ts(FLOAT, [3])],
                opset_version=17,
            )

    @parameterized.parameterized.expand(
        [
            ("none", [2, 3], [5, 4], 0, 0),
            ("trans_a", [3, 2], [5, 4], 1, 0),
            ("trans_b", [2, 3], [4, 5], 0, 1),
            ("both", [3, 2], [4, 5], 1, 1),
        ]
    )
    def test_contraction_dimension_mismatch_records_error(
        self, _name, shape_a, shape_b, trans_a, trans_b
    ):
        attributes = {
            "transA": ir.AttrInt64("transA", trans_a),
            "transB": ir.AttrInt64("transB", trans_b),
        }
        with self.assertRaisesRegex(ShapeInferenceError, "contraction dimensions: 3 vs 5"):
            run_shape_inference(
                "",
                "Gemm",
                [ts(FLOAT, shape_a), ts(FLOAT, shape_b)],
                attributes,
                opset_version=17,
            )

    @parameterized.parameterized.expand([(17,), (29,)])
    def test_contraction_dimension_mismatch_preserves_best_effort_shape(self, opset):
        actual = run_shape_inference(
            "",
            "Gemm",
            [ts(FLOAT, [2, 3]), ts(FLOAT, [5, 4])],
            opset_version=opset,
            policy="skip",
        )

        self.assertEqual(actual, [ts(FLOAT, [2, 4])])

    def test_symbolic_contraction_dimensions_are_not_rejected(self):
        actual = run_shape_inference(
            "",
            "Gemm",
            [ts(FLOAT, [2, "K1"]), ts(FLOAT, ["K2", 4])],
            opset_version=17,
        )

        self.assertEqual(actual, [ts(FLOAT, [2, 4])])


class GemmOpset29Test(unittest.TestCase):
    @parameterized.parameterized.expand([("GroupedMatMul",)])
    def test_new_operator_version_boundary(self, op):
        self.assertIsNone(registry.get("", op, 28))
        self.assertIsNotNone(registry.get("", op, 29))

    @parameterized.parameterized.expand(
        [
            ("vector", [4], [4, 7], {}, [7]),
            ("vector_transpose_noop", [4], [7, 4], {"transA": 1, "transB": 1}, [7]),
            ("batched", ["B", 3, 4], [4, 7], {}, ["B", 3, 7]),
            ("transpose", ["B", 4, 3], [7, 4], {"transA": 1, "transB": 1}, ["B", 3, 7]),
            ("unknown_weight", [2, 4], None, {}, [2, "_d0"]),
        ]
    )
    def test_gemm(self, _name, a, b, attributes, expected):
        self.assertEqual(
            run_shape_inference(
                "",
                "Gemm",
                [ts(ir.DataType.BFLOAT16, a), ts(ir.DataType.BFLOAT16, b)],
                {attr.name: attr for attr in ir.convenience.convert_attributes(attributes)},
                opset_version=29,
            ),
            [ts(ir.DataType.BFLOAT16, expected)],
        )

    def test_gemm_old_rank_semantics(self):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                "", "Gemm", [ts(FLOAT, [4]), ts(FLOAT, [4, 7])], opset_version=28
            )

    @parameterized.parameterized.expand(
        [
            ("standard", [5, 4], [3, 4, 7], [5, 2], None, 0, [5, 2, 7]),
            ("transpose_bias", ["M", 4], [3, 7, 4], [5, "S"], [3, 7], 1, [5, "S", 7]),
            ("bias_refines", None, None, [5, 2], [3, 7], 0, [5, 2, 7]),
            ("unknown", None, None, None, None, 0, ["_d0", "_d1", "_d2"]),
        ]
    )
    def test_grouped_matmul(self, _name, data, weights, indices, bias, trans, expected):
        self.assertEqual(
            run_shape_inference(
                "",
                "GroupedMatMul",
                [
                    ts(ir.DataType.BFLOAT16, data),
                    ts(ir.DataType.BFLOAT16, weights),
                    ts(ir.DataType.INT32, indices),
                    ts(ir.DataType.BFLOAT16, bias) if bias is not None else None,
                ],
                {"transB": ir.AttrInt64("transB", trans)},
                opset_version=29,
            ),
            [ts(ir.DataType.BFLOAT16, expected)],
        )

    @parameterized.parameterized.expand(
        [
            ("Gemm", [ts(FLOAT, [4]), ts(FLOAT, [4, 7]), ts(FLOAT, [2, 7])], {}),
            (
                "GroupedMatMul",
                [ts(FLOAT, [5, 4]), ts(FLOAT, [3, 8, 7]), ts(ir.DataType.INT64, [5, 2])],
                {},
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

    def test_matches_available_upstream_gemm(self):
        if onnx.defs.onnx_opset_version() < 29:
            self.skipTest("Installed ONNX predates opset 29")
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Gemm", ["a", "b"], ["y"])],
            "gemm29",
            [
                onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, ["B", 3, 4]),
                onnx.helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [4, 7]),
            ],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None)],
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 29)])
        out = ir.serde.deserialize_model(
            onnx.shape_inference.infer_shapes(model)
        ).graph.outputs[0]
        self.assertEqual(
            run_shape_inference(
                "", "Gemm", [ts(FLOAT, ["B", 3, 4]), ts(FLOAT, [4, 7])], opset_version=29
            ),
            [ir.TypeAndShape(out.type, out.shape)],
        )


if __name__ == "__main__":
    unittest.main()
