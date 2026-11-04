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
    @parameterized.parameterized.expand([(-1,), (True,), (1.5,), ("1024",)])
    def test_invalid_limit(self, limit):
        with self.assertRaisesRegex(ValueError, "max_symbolic_elements"):
            _context.ShapeInferenceContext(max_symbolic_elements=limit)

    def test_large_constant_is_not_materialized(self):
        tensor = _CountingTensor(np.zeros(2048, np.uint8))
        value = ir.val("packed", const_value=tensor)
        ctx = _context.ShapeInferenceContext()
        self.assertIsNone(ctx.get_symbolic_value(value))
        self.assertEqual(tensor.reads, 0)
        self.assertIs(value.const_value, tensor)

    def test_limit_boundary_and_unlimited(self):
        tensor = _CountingTensor(np.arange(9, dtype=np.int64))
        value = ir.val("shape", const_value=tensor)
        self.assertIsNone(
            _context.ShapeInferenceContext(max_symbolic_elements=8).get_symbolic_value(value)
        )
        for limit in (9, None):
            ctx = _context.ShapeInferenceContext(max_symbolic_elements=limit)
            self.assertEqual(ctx.get_symbolic_value(value), list(range(9)))
            self.assertEqual(ctx.create_child().max_symbolic_elements, limit)

    def test_oversized_results_clear_stale_metadata(self):
        ctx = _context.ShapeInferenceContext(max_symbolic_elements=2)
        value = ir.val("shape", dtype=ir.DataType.INT64, shape=[3])
        ctx.set_symbolic_value(value, [ir.SymbolicDim("a")])
        self.assertIn(_context.SYM_DATA_KEY, value.metadata_props)
        ctx.set_symbolic_value(value, [1, 2, 3])
        self.assertNotIn(_context.SYM_DATA_KEY, value.metadata_props)
        self.assertIsNone(ctx.get_symbolic_value(value))
        self.assertIsNone(value.const_value)

    def test_empty_data_with_zero_budget(self):
        ctx = _context.ShapeInferenceContext(max_symbolic_elements=0)
        value = ir.val("empty", const_value=ir.tensor(np.empty(0, np.int64)))
        self.assertEqual(ctx.get_symbolic_value(value), [])
        scalar = ir.val("scalar", const_value=ir.tensor(np.array(1, np.int64)))
        self.assertIsNone(ctx.get_symbolic_value(scalar))

    def test_cast_keeps_shape_and_bounds_metadata(self):
        for limit in (1024, None):
            with self.subTest(limit=limit):
                tensor = _CountingTensor(np.zeros(2048, np.uint8))
                packed = ir.val("packed", const_value=tensor)
                tape = ir.tape.Tape()
                output = tape.op("Cast", [packed], {"to": int(ir.DataType.INT64)})
                model = ir.Model(
                    ir.Graph(
                        [],
                        [output],
                        nodes=tape.nodes,
                        initializers=[packed],
                        opset_imports={"": 18},
                    ),
                    ir_version=10,
                )
                _engine.infer_symbolic_shapes(model, max_symbolic_elements=limit)
                self.assertEqual(output.shape, [2048])
                self.assertEqual(output.dtype, ir.DataType.INT64)
                self.assertEqual(tensor.reads > 0, limit is None)
                self.assertEqual(_context.SYM_DATA_KEY in output.metadata_props, limit is None)

    def test_function_budget_preserves_shape_program(self):
        for limit in (1024, None):
            with self.subTest(limit=limit):
                arg = ir.val("packed", dtype=ir.DataType.UINT8, shape=[4096])
                tape = ir.tape.Tape()
                indices = tape.op("Cast", [arg], {"to": int(ir.DataType.INT64)})
                table = tape.op(
                    "Constant", [], {"value": ir.tensor(np.arange(16, dtype=np.float32))}
                )
                decoded = tape.op("Gather", [table, indices])
                shape = tape.op(
                    "Constant", [], {"value": ir.tensor(np.array([64, 64], np.int64))}
                )
                output = tape.op("Reshape", [decoded, shape])
                function = ir.Function(
                    "test",
                    "Decode",
                    graph=ir.Graph([arg], [output], nodes=tape.nodes, opset_imports={"": 18}),
                    attributes=[],
                )
                tensor = _CountingTensor(np.zeros(4096, np.uint8))
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
                _engine.infer_symbolic_shapes(model, max_symbolic_elements=limit)
                self.assertEqual(result.shape, [64, 64])
                self.assertEqual(result.dtype, ir.DataType.FLOAT)
                self.assertEqual(tensor.reads > 0, limit is None)
                self.assertIs(packed.const_value, tensor)
