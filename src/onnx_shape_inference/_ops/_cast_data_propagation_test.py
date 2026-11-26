# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Soundness tests for shape-tensor data crossing dtype conversions."""

from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx_ir as ir
import parameterized
from onnx.reference import ReferenceEvaluator

from onnx_shape_inference import _context, infer_symbolic_shapes
from onnx_shape_inference._ops import _cast


class CastDataPropagationTest(unittest.TestCase):
    def _cast(self, ctx, data, dtype, *, like=False):
        if like:
            target = ir.Value(type=ir.TensorType(dtype))
            node = ir.Node("", "CastLike", [data, target], num_outputs=1)
            _cast.infer_cast_like(ctx, node)
        else:
            node = ir.Node(
                "", "Cast", [data], attributes=[ir.AttrInt64("to", dtype)], num_outputs=1
            )
            _cast.infer_cast(ctx, node)
        return node.outputs[0]

    @parameterized.parameterized.expand(
        [
            (f"{source.name}_{target.name}_{like}", source, target, safe, like)
            for source, target, safe in [
                (ir.DataType.INT32, ir.DataType.INT64, True),
                (ir.DataType.UINT8, ir.DataType.INT16, True),
                (ir.DataType.UINT32, ir.DataType.INT64, True),
                (ir.DataType.INT64, ir.DataType.INT64, True),
                (ir.DataType.INT64, ir.DataType.INT32, False),
                (ir.DataType.INT8, ir.DataType.UINT8, False),
                (ir.DataType.UINT64, ir.DataType.INT64, False),
                (ir.DataType.INT64, ir.DataType.FLOAT, False),
                (ir.DataType.FLOAT, ir.DataType.INT64, False),
                (ir.DataType.INT64, ir.DataType.BOOL, False),
                (ir.DataType.BOOL, ir.DataType.INT64, False),
            ]
            for like in (False, True)
        ]
    )
    def test_symbolic_values_require_range_preserving_integer_cast(
        self, _name, source, target, safe, like
    ):
        ctx = _context.ShapeInferenceContext({"": 21})
        data = ir.Value(type=ir.TensorType(source), shape=ir.Shape([2]))
        values = [ir.SymbolicDim("N"), 3]
        ctx.set_symbolic_value(data, values)
        out = self._cast(ctx, data, target, like=like)
        self.assertEqual(out.dtype, target)
        self.assertEqual(out.shape, data.shape)
        self.assertEqual(ctx.get_symbolic_value(out), values if safe else None)
        if not safe:
            self.assertNotIn(_context.SYM_DATA_KEY, out.metadata_props)
            self.assertIsNone(out.const_value)

    @parameterized.parameterized.expand(
        [
            (f"{name}_{like}", source, target, values, safe, like)
            for name, source, target, values, safe in [
                ("signed_boundary", ir.DataType.INT64, ir.DataType.INT8, [-128, 127], True),
                ("signed_overflow", ir.DataType.INT64, ir.DataType.INT8, [128], False),
                ("signed_underflow", ir.DataType.INT64, ir.DataType.INT8, [-129], False),
                ("unsigned_boundary", ir.DataType.INT64, ir.DataType.UINT8, [0, 255], True),
                ("unsigned_overflow", ir.DataType.INT64, ir.DataType.UINT8, [257], False),
                ("negative_unsigned", ir.DataType.INT64, ir.DataType.UINT8, [-1], False),
                ("positive_signed", ir.DataType.INT8, ir.DataType.UINT16, [127], True),
                ("negative_signed", ir.DataType.INT8, ir.DataType.UINT16, [-1], False),
                ("uint64_fits", ir.DataType.UINT64, ir.DataType.INT64, [2**63 - 1], True),
                ("uint64_overflow", ir.DataType.UINT64, ir.DataType.INT64, [2**63], False),
            ]
            for like in (False, True)
        ]
    )
    def test_concrete_values_must_fit_target_range(
        self, _name, source, target, values, safe, like
    ):
        ctx = _context.ShapeInferenceContext({"": 21})
        data = ir.Value(
            type=ir.TensorType(source),
            shape=ir.Shape([len(values)]),
            const_value=ir.tensor(values, dtype=source),
        )
        out = self._cast(ctx, data, target, like=like)
        self.assertEqual(ctx.get_symbolic_value(out), values if safe else None)
        if safe:
            np.testing.assert_array_equal(out.const_value.numpy(), values)
        else:
            self.assertIsNone(out.const_value)
            self.assertNotIn(_context.SYM_DATA_KEY, out.metadata_props)

    @parameterized.parameterized.expand(
        [
            (f"{name}_{like}", value, intermediate, expected, like)
            for name, value, intermediate, expected in [
                ("wrap", 257, ir.DataType.UINT8, 1),
                ("bool", 2, ir.DataType.BOOL, 1),
                ("round_float", 16777217, ir.DataType.FLOAT, 16777216),
            ]
            for like in (False, True)
        ]
    )
    def test_lossy_cast_roundtrip_does_not_declare_original_data(
        self, _name, value, intermediate, expected, like
    ):
        data = ir.Value(
            name="data",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape([1]),
            const_value=ir.tensor([value], dtype=ir.DataType.INT64),
        )
        initializers = [data]
        if like:
            target = ir.Value(
                name="target",
                type=ir.TensorType(intermediate),
                shape=ir.Shape([]),
                const_value=ir.tensor(0, dtype=intermediate),
            )
            initializers.append(target)
            cast = ir.Node("", "CastLike", [data, target], num_outputs=1)
        else:
            cast = ir.Node(
                "",
                "Cast",
                [data],
                attributes=[ir.AttrInt64("to", intermediate)],
                num_outputs=1,
            )
        restored = ir.Node(
            "",
            "Cast",
            [cast.outputs[0]],
            attributes=[ir.AttrInt64("to", ir.DataType.INT64)],
            num_outputs=1,
        )
        model = ir.Model(
            ir.Graph(
                [],
                [cast.outputs[0], restored.outputs[0]],
                nodes=[cast, restored],
                initializers=initializers,
                name="cast_roundtrip",
                opset_imports={"": 21},
            ),
            ir_version=10,
        )
        infer_symbolic_shapes(model)
        for output in model.graph.outputs:
            self.assertIsNone(output.const_value)
            self.assertNotIn(_context.SYM_DATA_KEY, output.metadata_props)
        proto = ir.to_proto(model)
        onnx.checker.check_model(proto, full_check=True)
        outputs = ReferenceEvaluator(proto).run(None, {})
        np.testing.assert_array_equal(outputs[-1], [expected])

    def test_reinference_drops_previously_propagated_constant(self):
        ctx = _context.ShapeInferenceContext({"": 21}, policy="override")
        data = ir.Value(
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape([1]),
            const_value=ir.tensor([257], dtype=ir.DataType.INT64),
        )
        node = ir.Node(
            "",
            "Cast",
            [data],
            attributes=[ir.AttrInt64("to", ir.DataType.INT64)],
            num_outputs=1,
        )
        _cast.infer_cast(ctx, node)
        out = node.outputs[0]
        self.assertEqual(ctx.get_symbolic_value(out), [257])
        node.attributes["to"] = ir.AttrInt64("to", ir.DataType.UINT8)
        _cast.infer_cast(ctx, node)
        self.assertEqual(out.dtype, ir.DataType.UINT8)
        self.assertIsNone(ctx.get_symbolic_value(out))
        self.assertIsNone(out.const_value)
        self.assertNotIn(_context.SYM_DATA_KEY, out.metadata_props)

    def test_cast_like_unknown_dtype_does_not_copy_shape_data(self):
        ctx = _context.ShapeInferenceContext({"": 21})
        data = ir.Value(type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([1]))
        ctx.set_symbolic_value(data, [ir.SymbolicDim("N")])
        target = ir.Value()
        node = ir.Node("", "CastLike", [data, target], num_outputs=1)
        _cast.infer_cast_like(ctx, node)
        self.assertEqual(node.outputs[0].shape, data.shape)
        self.assertIsNone(ctx.get_symbolic_value(node.outputs[0]))


if __name__ == "__main__":
    unittest.main()
