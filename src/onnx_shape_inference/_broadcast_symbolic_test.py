# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Broadcast annotations preserve independent runtime dimension assignments."""

from __future__ import annotations

import unittest

import numpy as np
import onnx.reference
import onnx_ir as ir
import parameterized

from onnx_shape_inference import _broadcast, _engine


class SymbolicBroadcastTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("named", ["N", 4], ["M", 4]),
            ("anonymous_named", [None, 4], ["M", 4]),
            ("named_anonymous", ["N", 4], [None, 4]),
            ("anonymous", [None, 4], [None, 4]),
        ]
    )
    def test_context_free_ambiguous_shape(self, _name, left, right):
        self.assertIsNone(_broadcast.broadcast_shapes(ir.Shape(left), ir.Shape(right)))

    def test_context_free_matching_symbols(self):
        self.assertEqual(
            _broadcast.broadcast_shapes(ir.Shape(["N", 4]), ir.Shape(["N", 1])),
            ir.Shape(["N", 4]),
        )

    def _check_bindings(self, model, feeds):
        actual = onnx.reference.ReferenceEvaluator(ir.to_proto(model)).run(None, feeds)
        bindings = {}
        for value, array in [
            *((value, feeds[value.name]) for value in model.graph.inputs),
            *zip(model.graph.outputs, actual, strict=True),
        ]:
            self.assertEqual(len(value.shape), array.ndim)
            for declared, extent in zip(value.shape, array.shape, strict=True):
                if isinstance(declared, int):
                    self.assertEqual(declared, extent)
                elif declared.value is not None:
                    self.assertEqual(bindings.setdefault(declared.value, extent), extent)
        return actual

    @parameterized.parameterized.expand(
        [
            ("left_singletons", (1, 3), (2, 1)),
            ("right_singletons", (2, 1), (1, 3)),
            ("left_zero", (0, 3), (1, 3)),
            ("right_zero", (1, 3), (0, 3)),
        ]
    )
    def test_elementwise_independent_symbols(self, _name, left, right):
        x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["N", "K"])
        y = ir.val("y", dtype=ir.DataType.FLOAT, shape=["M", "L"])
        tape = ir.tape.Tape()
        output = tape.op("Mul", [x, y])
        model = ir.Model(
            ir.Graph(
                [x, y], [output], nodes=tape.nodes, opset_imports={"": 18}, name="broadcast"
            ),
            ir_version=10,
        )
        _engine.infer_symbolic_shapes(model)
        self.assertEqual(len(output.shape), 2)
        self.assertNotIn(output.shape[0], (x.shape[0], y.shape[0]))
        self.assertNotIn(output.shape[1], (x.shape[1], y.shape[1]))
        self._check_bindings(
            model, {"x": np.ones(left, np.float32), "y": np.ones(right, np.float32)}
        )

    @parameterized.parameterized.expand([(1, 5), (5, 1), (0, 1), (1, 0)])
    def test_matmul_independent_batch_symbols(self, left, right):
        x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["N", 2, 3])
        y = ir.val("y", dtype=ir.DataType.FLOAT, shape=["M", 3, 4])
        tape = ir.tape.Tape()
        output = tape.op("MatMul", [x, y])
        model = ir.Model(
            ir.Graph(
                [x, y], [output], nodes=tape.nodes, opset_imports={"": 18}, name="matmul"
            ),
            ir_version=10,
        )
        _engine.infer_symbolic_shapes(model)
        self.assertEqual(list(output.shape)[1:], [2, 4])
        self.assertNotIn(output.shape[0], (x.shape[0], y.shape[0]))
        self._check_bindings(
            model,
            {"x": np.ones((left, 2, 3), np.float32), "y": np.ones((right, 3, 4), np.float32)},
        )

    def test_where_independent_symbols(self):
        condition = ir.val("condition", dtype=ir.DataType.BOOL, shape=["C", "D"])
        x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["N", "K"])
        y = ir.val("y", dtype=ir.DataType.FLOAT, shape=[])
        tape = ir.tape.Tape()
        output = tape.op("Where", [condition, x, y])
        model = ir.Model(
            ir.Graph(
                [condition, x, y],
                [output],
                nodes=tape.nodes,
                opset_imports={"": 18},
                name="where",
            ),
            ir_version=10,
        )
        _engine.infer_symbolic_shapes(model)
        self._check_bindings(
            model,
            {
                "condition": np.ones((2, 1), bool),
                "x": np.ones((1, 3), np.float32),
                "y": np.array(0, np.float32),
            },
        )

    def test_fresh_dimensions_do_not_alias_inputs_or_other_broadcasts(self):
        inputs = [
            ir.val(name, dtype=ir.DataType.FLOAT, shape=[dim, 4])
            for name, dim in zip(("x", "y", "z", "w"), ("_d0", "_d1", "P", "Q"), strict=True)
        ]
        tape = ir.tape.Tape()
        outputs = [tape.op("Add", inputs[:2]), tape.op("Mul", inputs[2:])]
        model = ir.Model(
            ir.Graph(
                inputs, outputs, nodes=tape.nodes, opset_imports={"": 18}, name="independent"
            ),
            ir_version=10,
        )
        _engine.infer_symbolic_shapes(model)
        known = {v.shape[0] for v in inputs}
        self.assertNotIn(outputs[0].shape[0], known)
        self.assertNotIn(outputs[1].shape[0], known)
        self.assertNotEqual(outputs[0].shape[0], outputs[1].shape[0])
        self.assertEqual(list(outputs[0].shape)[1:], [4])
        self.assertEqual(list(outputs[1].shape)[1:], [4])
