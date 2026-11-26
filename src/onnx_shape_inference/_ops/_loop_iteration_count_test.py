# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Loop iteration bounds must not become unsound static output dimensions."""

from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir
import parameterized
from onnx.reference import ReferenceEvaluator

from onnx_shape_inference import infer_symbolic_shapes


def _constant(name, value, dtype):
    tensor = ir.tensor(value, dtype=dtype)
    return ir.Value(
        name=name, type=ir.TensorType(dtype), shape=tensor.shape, const_value=tensor
    )


def _scan_loop(limit, initial, continuation):
    iteration = ir.Value(
        name="iteration", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
    )
    condition = ir.Value(
        name="condition", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
    )
    item = _constant("item", [7], ir.DataType.INT64)
    body = ir.Graph([iteration, condition], [], nodes=[], name="body", opset_imports={"": 21})
    body.initializers[item.name] = item
    if continuation == "identity":
        cond_out = condition
    elif continuation == "dynamic":
        threshold = _constant("threshold", 1, ir.DataType.INT64)
        body.initializers[threshold.name] = threshold
        compare = ir.Node("", "Less", [iteration, threshold], num_outputs=1)
        body.append(compare)
        cond_out = compare.outputs[0]
    else:
        cond_out = _constant("continue", continuation, ir.DataType.BOOL)
        body.initializers[cond_out.name] = cond_out
    body.outputs.extend([cond_out, item, item])
    maximum = _constant("maximum", limit, ir.DataType.INT64)
    cond = (
        _constant("initial", initial, ir.DataType.BOOL)
        if initial is not None
        else ir.Value(name="initial", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))
    )
    loop = ir.Node(
        "", "Loop", [maximum, cond], attributes=[ir.AttrGraph("body", body)], num_outputs=2
    )
    graph = ir.Graph(
        [cond] if initial is None else [],
        list(loop.outputs),
        nodes=[loop],
        name="scan_loop",
        opset_imports={"": 21},
    )
    graph.initializers[maximum.name] = maximum
    if initial is not None:
        graph.initializers[cond.name] = cond
    return ir.Model(graph, ir_version=10), loop


class LoopIterationCountTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("stop_after_one", 3, True, False, 1),
            ("initial_false", 3, False, True, 0),
            ("zero_limit", 0, True, True, 0),
            ("negative_limit", -2, True, True, 0),
            ("always_true", 3, True, True, 3),
            ("preserved_condition", 3, True, "identity", 3),
            ("unknown_initial", 3, None, True, None),
            ("data_dependent_exit", 3, True, "dynamic", None),
        ]
    )
    def test_scan_iteration_count(self, _name, limit, initial, continuation, expected):
        model, loop = _scan_loop(limit, initial, continuation)
        infer_symbolic_shapes(model)
        first = loop.outputs[0].shape[0]
        if expected is None:
            self.assertIsInstance(first, ir.SymbolicDim)
        else:
            self.assertEqual(first, expected)
        self.assertEqual(loop.outputs[1].shape[0], first)
        self.assertEqual(loop.outputs[0].shape[1], 1)
        # ReferenceEvaluator cannot retain trailing scan dimensions on an empty
        # loop, so exercise nonempty and data-dependent cases against execution.
        if initial is True and limit > 0:
            actual = ReferenceEvaluator(ir.to_proto(model)).run(None, {})
            for value, data in zip(loop.outputs, actual):
                self.assertEqual(data.shape[1:], (1,))
                if isinstance(value.shape[0], int):
                    self.assertEqual(value.shape[0], data.shape[0])

    @parameterized.parameterized.expand([(False, 3, 2), (True, 1, 3), (True, 3, None)])
    def test_growing_carried_state(self, initial, limit, expected):
        iteration = ir.Value(
            name="iteration", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
        )
        condition = ir.Value(
            name="condition", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
        )
        carried = ir.Value(
            name="carried", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([2])
        )
        append = _constant("append", [7], ir.DataType.INT64)
        concat = ir.Node(
            "",
            "Concat",
            [carried, append],
            attributes=[ir.AttrInt64("axis", 0)],
            num_outputs=1,
        )
        body = ir.Graph(
            [iteration, condition, carried],
            [condition, concat.outputs[0]],
            nodes=[concat],
            name="growing",
            opset_imports={"": 21},
        )
        body.initializers[append.name] = append
        maximum = _constant("maximum", limit, ir.DataType.INT64)
        cond = _constant("initial", initial, ir.DataType.BOOL)
        x = _constant("x", [1, 2], ir.DataType.INT64)
        loop = ir.Node(
            "",
            "Loop",
            [maximum, cond, x],
            attributes=[ir.AttrGraph("body", body)],
            num_outputs=1,
        )
        graph = ir.Graph(
            [], list(loop.outputs), nodes=[loop], name="growing_loop", opset_imports={"": 21}
        )
        for value in (maximum, cond, x):
            graph.initializers[value.name] = value
        model = ir.Model(graph, ir_version=10)
        infer_symbolic_shapes(model)
        if expected is None:
            self.assertIsInstance(loop.outputs[0].shape[0], ir.SymbolicDim)
        else:
            self.assertEqual(loop.outputs[0].shape, ir.Shape([expected]))
        (actual,) = ReferenceEvaluator(ir.to_proto(model)).run(None, {})
        np.testing.assert_array_equal(actual, [1, 2] + [7] * (limit if initial else 0))
