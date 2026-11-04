# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bound partial data propagation while preserving large-tensor shapes."""

from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir
import parameterized

from onnx_shape_inference import _context, _engine


class _CountingTensor(ir.Tensor):
    def __init__(self, array):
        super().__init__(array)
        self.reads = 0

    def numpy(self):
        self.reads += 1
        return super().numpy()


class SymbolicDataBudgetTest(unittest.TestCase):
    def test_large_constant_is_not_materialized(self):
        tensor = _CountingTensor(np.zeros(_context.MAX_SYMBOLIC_ELEMENTS + 1, np.uint8))
        value = ir.val("packed", const_value=tensor)
        ctx = _context.ShapeInferenceContext()
        self.assertIsNone(ctx.get_symbolic_value(value))
        self.assertEqual(tensor.reads, 0)
        self.assertIs(value.const_value, tensor)

    @parameterized.parameterized.expand(
        [(_context.MAX_SYMBOLIC_ELEMENTS,), (_context.MAX_SYMBOLIC_ELEMENTS + 1,)]
    )
    def test_limit_boundary(self, count):
        tensor = _CountingTensor(np.arange(count, dtype=np.int64))
        value = ir.val("shape", const_value=tensor)
        result = _context.ShapeInferenceContext().get_symbolic_value(value)
        if count <= _context.MAX_SYMBOLIC_ELEMENTS:
            self.assertEqual(result, list(range(count)))
            self.assertGreater(tensor.reads, 0)
        else:
            self.assertIsNone(result)
            self.assertEqual(tensor.reads, 0)

    def test_oversized_results_clear_stale_metadata(self):
        ctx = _context.ShapeInferenceContext()
        value = ir.val("shape", dtype=ir.DataType.INT64, shape=[1])
        ctx.set_symbolic_value(value, [ir.SymbolicDim("a")])
        self.assertIn(_context.SYM_DATA_KEY, value.metadata_props)
        count = _context.MAX_SYMBOLIC_ELEMENTS + 1
        value.shape = ir.Shape([count])
        ctx.set_symbolic_value(value, list(range(count)))
        self.assertNotIn(_context.SYM_DATA_KEY, value.metadata_props)
        self.assertIsNone(ctx.get_symbolic_value(value))
        self.assertIsNone(value.const_value)

    def test_scalars_and_empty_data(self):
        ctx = _context.ShapeInferenceContext()
        for array, expected in [(np.array(1, np.int64), [1]), (np.empty(0, np.int64), [])]:
            with self.subTest(shape=array.shape):
                value = ir.val("value", const_value=ir.tensor(array))
                self.assertEqual(ctx.get_symbolic_value(value), expected)

    def test_cast_keeps_shape_and_bounds_metadata(self):
        count = _context.MAX_SYMBOLIC_ELEMENTS * 2
        tensor = _CountingTensor(np.zeros(count, np.uint8))
        packed = ir.val("packed", const_value=tensor)
        tape = ir.tape.Tape()
        output = tape.op("Cast", [packed], {"to": int(ir.DataType.INT64)})
        model = ir.Model(
            ir.Graph(
                [], [output], nodes=tape.nodes, initializers=[packed], opset_imports={"": 18}
            ),
            ir_version=10,
        )
        _engine.infer_symbolic_shapes(model)
        self.assertEqual(output.shape, [count])
        self.assertEqual(output.dtype, ir.DataType.INT64)
        self.assertEqual(tensor.reads, 0)
        self.assertNotIn(_context.SYM_DATA_KEY, output.metadata_props)

    def test_function_budget_preserves_shape_program(self):
        for count in (64, _context.MAX_SYMBOLIC_ELEMENTS * 2):
            with self.subTest(count=count):
                arg = ir.val("packed", dtype=ir.DataType.UINT8, shape=[count])
                tape = ir.tape.Tape()
                indices = tape.op("Cast", [arg], {"to": int(ir.DataType.INT64)})
                table = tape.op(
                    "Constant", [], {"value": ir.tensor(np.arange(16, dtype=np.float32))}
                )
                decoded = tape.op("Gather", [table, indices])
                shape = tape.op(
                    "Constant", [], {"value": ir.tensor(np.array([count // 2, 2], np.int64))}
                )
                output = tape.op("Reshape", [decoded, shape])
                function = ir.Function(
                    "test",
                    "Decode",
                    graph=ir.Graph([arg], [output], nodes=tape.nodes, opset_imports={"": 18}),
                    attributes=[],
                )
                tensor = _CountingTensor(np.zeros(count, np.uint8))
                packed = ir.val("weights", const_value=tensor)
                outer = ir.tape.Tape()
                result = outer.op("Decode", [packed], domain="test")
                model = ir.Model(
                    ir.Graph(
                        [],
                        [result],
                        nodes=outer.nodes,
                        initializers=[packed],
                        opset_imports={"": 18, "test": 1},
                    ),
                    functions=[function],
                    ir_version=10,
                )
                _engine.infer_symbolic_shapes(model)
                self.assertEqual(result.shape, [count // 2, 2])
                self.assertEqual(result.dtype, ir.DataType.FLOAT)
                self.assertEqual(tensor.reads > 0, count <= _context.MAX_SYMBOLIC_ELEMENTS)
                self.assertIs(packed.const_value, tensor)
