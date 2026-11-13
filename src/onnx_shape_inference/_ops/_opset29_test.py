# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Regressions for all schemas in upstream ONNX's opset 29 inventory."""

from __future__ import annotations

import unittest

import onnx
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
U8 = ir.DataType.UINT8
U32 = ir.DataType.UINT32
I64 = ir.DataType.INT64
BF16 = ir.DataType.BFLOAT16


def attrs(**values):
    result = {}
    for name, value in values.items():
        if isinstance(value, str):
            result[name] = ir.AttrString(name, value)
        elif isinstance(value, list):
            result[name] = ir.AttrInt64s(name, value)
        elif isinstance(value, float):
            result[name] = ir.AttrFloat32(name, value)
        else:
            result[name] = ir.AttrInt64(name, value)
    return result


class Opset29Test(unittest.TestCase):
    @parameterized.parameterized.expand(
        [("Pack",), ("Unpack",), ("GroupedMatMul",), ("DequantizePacked",)]
    )
    def test_new_operator_version_boundary(self, op):
        self.assertIsNone(registry.get("", op, 28))
        self.assertIsNotNone(registry.get("", op, 29))

    def test_causal_previous_opset_keeps_unit_dilation(self):
        self.assertEqual(
            run_shape_inference(
                "",
                "CausalConvWithState",
                [ts(BF16, [2, 3, 7]), ts(BF16, [3, 1, 4])],
                opset_version=27,
                num_outputs=2,
            ),
            [ts(BF16, [2, 3, 7]), ts(BF16, [2, 3, 3])],
        )

    def test_unpack_unknown_storage_type_does_not_assume_word_width(self):
        data = ir.Value(shape=ir.Shape([3]))
        count = const_value(8)
        self.assertEqual(
            run_shape_inference_with_values(
                "", "Unpack", [data, count], attrs(bits=8), opset_version=29
            ),
            [ts(U8, [8])],
        )

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
                run_shape_inference("", op, [ts(BF16, shape)], opset_version=29),
                [ts(BF16, shape)],
            )
        self.assertIsNone(registry.get("", op, 28))

    def test_existing_mvn(self):
        self.assertEqual(
            run_shape_inference(
                "",
                "MeanVarianceNormalization",
                [ts(BF16, ["B", 3, 7, 7])],
                attrs(axes=[-1], epsilon=1e-5),
                opset_version=29,
            ),
            [ts(BF16, ["B", 3, 7, 7])],
        )

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
                "", "Gemm", [ts(BF16, a), ts(BF16, b)], attrs(**attributes), opset_version=29
            ),
            [ts(BF16, expected)],
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
                    ts(BF16, data),
                    ts(BF16, weights),
                    ts(ir.DataType.INT32, indices),
                    ts(BF16, bias) if bias is not None else None,
                ],
                attrs(transB=trans),
                opset_version=29,
            ),
            [ts(BF16, expected)],
        )

    @parameterized.parameterized.expand(
        [
            ("bits3", [2, 8], {"bits": 3}, U8, [2, 3]),
            ("word32", [2, 7], {"bits": 5, "output_type": int(U32)}, U32, [2, 2]),
            ("grouped", [2, 11], {"radix": 3, "group_size": 5}, U8, [2, 3]),
            ("axis", [7, 2], {"bits": 3, "axis": 0}, U8, [3, 2]),
            ("zero", [0, 0], {"bits": 7}, U8, [0, 0]),
            ("identity", ["B", "N"], {"bits": 8}, U8, ["B", "N"]),
            ("symbolic", ["B", "N"], {"bits": 4}, U8, ["B", "-floor(-N/2)"]),
            ("unknown", None, {"bits": 3}, U8, None),
        ]
    )
    def test_pack(self, _name, shape, attributes, dtype, expected):
        self.assertEqual(
            run_shape_inference(
                "", "Pack", [ts(U8, shape)], attrs(**attributes), opset_version=29
            ),
            [ts(dtype, expected)],
        )

    @parameterized.parameterized.expand(
        [
            ("bits", U8, [2, 3], 8, {"bits": 3}, [2, 8]),
            ("grouped", U8, [2, 3], 11, {"radix": 3, "group_size": 5}, [2, 11]),
            ("word32", U32, [2, 2], 7, {"bits": 5}, [2, 7]),
            ("zero", U8, [2, 0], 0, {"bits": 3}, [2, 0]),
            ("runtime", U8, ["B", 3], None, {"bits": 3}, ["B", "_d0"]),
        ]
    )
    def test_unpack(self, _name, dtype, shape, count, attributes, expected):
        data = ir.Value(type=ir.TensorType(dtype), shape=ir.Shape(shape))
        count_input = (
            const_value(count)
            if count is not None
            else ir.Value(type=ir.TensorType(I64), shape=ir.Shape([]))
        )
        self.assertEqual(
            run_shape_inference_with_values(
                "", "Unpack", [data, count_input], attrs(**attributes), opset_version=29
            ),
            [ts(U8, expected)],
        )

    def test_symbolic_unpack_count(self):
        data = ir.Value(type=ir.TensorType(U8), shape=ir.Shape(["B", "bytes"]))
        count = ir.Value(type=ir.TensorType(I64), shape=ir.Shape([]))
        self.assertEqual(
            run_shape_inference_with_values(
                "",
                "Unpack",
                [data, count],
                attrs(bits=3),
                opset_version=29,
                symbolic_values={1: [ir.SymbolicDim("N")]},
            ),
            [ts(U8, ["B", "N"])],
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
                [ts(U8, [size])],
                attrs(format=format_, shape=shape, output_dtype=int(BF16)),
                opset_version=29,
            ),
            [ts(BF16, shape)],
        )

    @parameterized.parameterized.expand(
        [
            ("dilated", [2, 3, 0], [3, 1, 4], None, 3, [2, 3, 9]),
            ("kernel_one", [2, 3, 7], [3, 1, 1], None, 4, [2, 3, 0]),
            ("symbolic", ["B", 3, "L"], [3, 1, "K"], None, 2, ["B", 3, "2*K - 2"]),
            ("past_refines", None, None, [2, 3, 6], 2, [2, 3, 6]),
        ]
    )
    def test_causal(self, _name, shape, weight, past, dilation, state):
        self.assertEqual(
            run_shape_inference(
                "",
                "CausalConvWithState",
                [ts(BF16, shape), ts(BF16, weight), None, ts(BF16, past) if past else None],
                attrs(dilation=dilation),
                opset_version=29,
                num_outputs=2,
            ),
            [ts(BF16, shape), ts(BF16, state)],
        )

    def test_existing_gather_int32(self):
        self.assertEqual(
            run_shape_inference(
                "",
                "GatherND",
                [ts(BF16, [2, 3, 4]), ts(ir.DataType.INT32, [2, 5, 1])],
                attrs(batch_dims=1),
                opset_version=29,
            ),
            [ts(BF16, [2, 5, 4])],
        )

    @parameterized.parameterized.expand([("ScatterND",), ("ScatterElements",)])
    def test_scatter_mean(self, op):
        self.assertEqual(
            run_shape_inference(
                "",
                op,
                [ts(BF16, ["B", 3]), ts(ir.DataType.INT32, [2, 2]), ts(BF16, [2])],
                attrs(reduction="mean"),
                opset_version=29,
            ),
            [ts(BF16, ["B", 3])],
        )
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                "",
                op,
                [ts(ir.DataType.BOOL, [3]), ts(I64, [1]), ts(ir.DataType.BOOL, [1])],
                attrs(reduction="mean"),
                opset_version=29,
            )

    @parameterized.parameterized.expand(
        [
            ("LogCumSumExp", [ts(FLOAT, [])], {}),
            ("LogCumSumExp", [ts(FLOAT, [3])], {"axis": 1}),
            ("Pack", [ts(U8, [7])], {"bits": 0}),
            ("Pack", [ts(U8, [7])], {"bits": 2, "radix": 3}),
            ("Pack", [ts(U8, [7])], {"radix": 256, "group_size": 8}),
            ("Pack", [ts(U8, [])], {"bits": 3}),
            ("Pack", [ts(U8, [7])], {"bits": 3, "axis": 1}),
            ("Pack", [ts(U8, [7])], {"bits": 3, "output_type": int(FLOAT)}),
            ("DequantizePacked", [ts(U8, [36])], {"format": "GGUF:q4_0", "shape": [2, 32]}),
            ("DequantizePacked", [ts(U8, [36])], {"format": "gguf:q4_0", "shape": [2, 31]}),
            ("DequantizePacked", [ts(U8, [35])], {"format": "gguf:q4_0", "shape": [2, 32]}),
            ("Gemm", [ts(FLOAT, [4]), ts(FLOAT, [4, 7]), ts(FLOAT, [2, 7])], {}),
            ("GroupedMatMul", [ts(FLOAT, [5, 4]), ts(FLOAT, [3, 8, 7]), ts(I64, [5, 2])], {}),
            (
                "CausalConvWithState",
                [ts(FLOAT, [2, 3, 7]), ts(FLOAT, [3, 1, 4])],
                {"dilation": 0},
            ),
            ("CausalConvWithState", [ts(FLOAT, [2, 3, 7]), ts(FLOAT, [3, 2, 4])], {}),
            (
                "CausalConvWithState",
                [ts(FLOAT, [2, 3, 7]), ts(FLOAT, [3, 1, 4]), None, ts(FLOAT, [2, 3, 3])],
                {"dilation": 2},
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
                    "", op, inputs, attrs(**attributes), opset_version=29, policy=policy
                )

    def test_unpack_invalid_count_and_storage(self):
        data = ir.Value(type=ir.TensorType(U8), shape=ir.Shape([3]))
        for count in (const_value(-1), const_value([8]), const_value(2)):
            with self.assertRaises(ShapeInferenceError):
                run_shape_inference_with_values(
                    "", "Unpack", [data, count], attrs(bits=3), opset_version=29
                )
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "", "Unpack", [data, None], attrs(bits=3), opset_version=29
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
