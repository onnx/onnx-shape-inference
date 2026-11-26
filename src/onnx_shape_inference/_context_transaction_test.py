# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Atomic inference metadata and data-propagation journal tests."""

from __future__ import annotations

import unittest

import onnx_ir as ir

from onnx_shape_inference import _context


class _NoHistoryScan(dict):
    def __iter__(self):
        raise AssertionError("transaction scanned graph history")

    def items(self):
        raise AssertionError("transaction scanned graph history")

    def values(self):
        raise AssertionError("transaction scanned graph history")

    def copy(self):
        raise AssertionError("transaction copied graph history")


class TransactionTest(unittest.TestCase):
    def setUp(self):
        self.context = _context.ShapeInferenceContext({"": 21}, policy="override")
        self.type = ir.TensorType(ir.DataType.FLOAT)
        self.x = ir.Value(name="x", type=self.type, shape=ir.Shape([None, 3]))
        self.y = ir.Value(name="y", type=self.type, shape=ir.Shape([1]))
        self.node = ir.Node("", "Identity", [self.x], outputs=[self.y])

    def test_failure_restores_metadata_constant_ownership_and_input_shape(self):
        context = self.context
        context.set_symbolic_value(self.y, [1])
        original_constant = self.y.const_value
        original_shape = self.x.shape
        self.y.metadata_props["other"] = "retained"
        with (
            self.assertRaisesRegex(ValueError, "inference failed"),
            context.transaction(self.node),
        ):
            context.name_anonymous_dims(self.x)
            context.set_shape(self.y, ir.Shape([1]))
            context.set_dtype(self.y, ir.DataType.INT32)
            context.set_symbolic_value(self.y, [999])
            self.y.shape[0] = 999
            raise ValueError("inference failed")
        self.assertIs(self.y.type, self.type)
        self.assertEqual(self.x.dtype, ir.DataType.FLOAT)
        self.assertEqual(self.y.dtype, ir.DataType.FLOAT)
        self.assertIs(self.x.shape, original_shape)
        self.assertIsNone(self.x.shape[0].value)
        self.assertEqual(self.y.shape, ir.Shape([1]))
        self.assertIs(self.y.const_value, original_constant)
        self.assertEqual(context.get_symbolic_value(self.y), [1])
        self.assertEqual(self.y.metadata_props[_context.SYM_DATA_KEY], "[1]")
        self.assertEqual(self.y.metadata_props["other"], "retained")
        # Restored ownership must still invalidate the restored constant later.
        context.clear_symbolic_value(self.y)
        self.assertIsNone(self.y.const_value)
        self.assertIsNone(context.get_symbolic_value(self.y))
        # Anonymous names from aborted inference are intentionally not reused.
        self.assertEqual(context.new_symbolic_dim(), ir.SymbolicDim("_d1"))

    def test_direct_node_value_writes_are_rolled_back(self):
        original_shape = self.y.shape
        with self.assertRaises(ValueError), self.context.transaction(self.node):
            self.y.type = ir.TensorType(ir.DataType.INT64)
            self.y.shape = ir.Shape([999])
            self.y.const_value = ir.tensor([999], dtype=ir.DataType.INT64)
            self.y.metadata_props[_context.SYM_DATA_KEY] = "[999]"
            raise ValueError("failed")
        self.assertIs(self.y.type, self.type)
        self.assertIs(self.y.shape, original_shape)
        self.assertIsNone(self.y.const_value)
        self.assertNotIn(_context.SYM_DATA_KEY, self.y.metadata_props)

    def test_setter_touched_values_outside_node_are_rolled_back(self):
        other = ir.Value(type=ir.TensorType(ir.DataType.INT64))
        with self.assertRaises(ValueError), self.context.transaction(self.node):
            self.context.set_shape(other, ir.Shape([1]))
            self.context.set_symbolic_value(other, [42])
            raise ValueError("failed")
        self.assertIsNone(other.shape)
        self.assertIsNone(other.const_value)
        self.assertIsNone(self.context.get_symbolic_value(other))
        self.assertNotIn(_context.SYM_DATA_KEY, other.metadata_props)

    def test_replaced_type_alias_is_restored_without_copying_types(self):
        replacement = ir.TensorType(ir.DataType.DOUBLE)
        alias = ir.Value(type=replacement)
        with self.assertRaises(ValueError), self.context.transaction(self.node):
            self.context.set_type(self.y, replacement)
            self.context.set_dtype(self.y, ir.DataType.INT64)
            raise ValueError("failed")
        self.assertIs(self.y.type, self.type)
        self.assertIs(alias.type, replacement)
        self.assertEqual(alias.dtype, ir.DataType.DOUBLE)

    def test_sparse_tensor_dtype_alias_is_restored(self):
        sparse_type = ir.SparseTensorType(ir.DataType.FLOAT)
        self.y.type = sparse_type
        alias = ir.Value(type=sparse_type)
        with self.assertRaises(ValueError), self.context.transaction(self.node):
            self.context.set_dtype(self.y, ir.DataType.DOUBLE)
            raise ValueError("failed")
        self.assertIs(self.y.type, sparse_type)
        self.assertIs(alias.type, sparse_type)
        self.assertEqual(alias.dtype, ir.DataType.FLOAT)

    def test_nested_input_dtype_alias_is_restored(self):
        for outer_type, inner_type in (
            (ir.SequenceType, ir.OptionalType),
            (ir.OptionalType, ir.SequenceType),
        ):
            for tensor_type in (ir.TensorType, ir.SparseTensorType):
                with self.subTest(outer=outer_type, leaf=tensor_type):
                    leaf = tensor_type(ir.DataType.FLOAT)
                    inner = inner_type(leaf)
                    outer = outer_type(inner)
                    self.x.type = outer
                    alias = ir.Value(type=leaf)
                    with self.assertRaises(ValueError), self.context.transaction(self.node):
                        self.x.dtype = ir.DataType.DOUBLE
                        raise ValueError("failed")
                    self.assertIs(self.x.type, outer)
                    self.assertIs(outer.elem_type, inner)
                    self.assertIs(inner.elem_type, leaf)
                    self.assertIs(alias.type, leaf)
                    self.assertEqual(alias.dtype, ir.DataType.FLOAT)
                    self.assertEqual(self.x.dtype, ir.DataType.FLOAT)

    def test_replaced_nested_type_dtype_alias_is_restored(self):
        for outer_type, inner_type in (
            (ir.SequenceType, ir.OptionalType),
            (ir.OptionalType, ir.SequenceType),
        ):
            for tensor_type in (ir.TensorType, ir.SparseTensorType):
                with self.subTest(outer=outer_type, leaf=tensor_type):
                    leaf = tensor_type(ir.DataType.FLOAT)
                    inner = inner_type(leaf)
                    replacement = outer_type(inner)
                    alias = ir.Value(type=leaf)
                    with self.assertRaises(ValueError), self.context.transaction(self.node):
                        self.context.set_type(self.y, replacement)
                        self.context.set_dtype(self.y, ir.DataType.DOUBLE)
                        raise ValueError("failed")
                    self.assertIs(self.y.type, self.type)
                    self.assertIs(replacement.elem_type, inner)
                    self.assertIs(inner.elem_type, leaf)
                    self.assertIs(alias.type, leaf)
                    self.assertEqual(alias.dtype, ir.DataType.FLOAT)
                    self.assertEqual(self.y.dtype, ir.DataType.FLOAT)

    def test_failure_restores_constraints_and_recorded_errors(self):
        context = _context.ShapeInferenceContext({"": 21}, policy="skip")
        context.add_symbolic_equality("a", "b")
        context.add_upper_bound("a", "limit")
        context.record_error(self.node, "existing")
        original_errors = list(context.errors)
        with self.assertRaises(ValueError), context.transaction(self.node):
            context.add_symbolic_equality("bad", "a")
            context.add_upper_bound("bad", "a")
            context.record_error(self.node, "new error")
            raise ValueError("failed")
        self.assertEqual(context.symbolic_equalities, [("a", "b")])
        self.assertEqual(context.symbolic_upper_bounds, [("a", "limit")])
        self.assertEqual(context.errors, original_errors)

    def test_success_commits_metadata_and_constraints(self):
        with self.context.transaction(self.node):
            self.context.set_shape(self.y, ir.Shape([1]))
            self.context.set_dtype(self.y, ir.DataType.INT64)
            self.context.set_symbolic_value(self.y, [42])
            self.context.add_symbolic_equality("a", "b")
            self.context.add_upper_bound("a", "limit")
        self.assertIs(self.y.type, self.type)
        self.assertEqual(self.y.dtype, ir.DataType.INT64)
        self.assertEqual(self.context.get_symbolic_value(self.y), [42])
        self.assertEqual(self.y.const_value.numpy().tolist(), [42])
        self.assertEqual(self.context.symbolic_equalities, [("a", "b")])
        self.assertEqual(self.context.symbolic_upper_bounds, [("a", "limit")])

    def test_nested_failure_restores_inner_state_but_outer_can_commit(self):
        context = self.context
        with context.transaction(self.node):
            context.set_shape(self.y, ir.Shape([1]))
            context.set_dtype(self.y, ir.DataType.INT64)
            context.set_symbolic_value(self.y, [2])
            outer_constant = self.y.const_value
            with self.assertRaises(ValueError), context.transaction(self.node):
                context.set_dtype(self.y, ir.DataType.INT32)
                context.set_symbolic_value(self.y, [3])
                raise ValueError("inner failed")
            self.assertEqual(self.y.dtype, ir.DataType.INT64)
            self.assertEqual(context.get_symbolic_value(self.y), [2])
            self.assertIs(self.y.const_value, outer_constant)
        self.assertEqual(context.get_symbolic_value(self.y), [2])
        context.clear_symbolic_value(self.y)
        self.assertIsNone(self.y.const_value)

    def test_outer_failure_restores_successful_inner_changes(self):
        other = ir.Value(type=ir.TensorType(ir.DataType.INT64))
        with self.assertRaises(ValueError), self.context.transaction(self.node):
            with self.context.transaction(self.node):
                self.context.set_shape(other, ir.Shape([1]))
                self.context.set_symbolic_value(other, [3])
                self.context.set_dtype(self.y, ir.DataType.INT64)
            raise ValueError("outer failed")
        self.assertEqual(self.y.dtype, ir.DataType.FLOAT)
        self.assertIsNone(other.shape)
        self.assertIsNone(self.context.get_symbolic_value(other))

    def test_transaction_does_not_scan_previous_propagation_history(self):
        self.context._symbolic_values = _NoHistoryScan(self.context._symbolic_values)
        self.context._propagated_constants = _NoHistoryScan(self.context._propagated_constants)
        # Nonempty maps must not be copied or traversed to infer a fresh node.
        old = ir.Value(type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([1]))
        self.context.set_symbolic_value(old, [4])
        old_constant = old.const_value
        with self.assertRaises(ValueError), self.context.transaction(self.node):
            self.context.set_symbolic_value(self.y, [5])
            raise ValueError("failed")
        self.assertIs(old.const_value, old_constant)
        self.assertEqual(self.context.get_symbolic_value(old), [4])

    def test_in_place_shape_and_symbolic_list_mutations_are_restored(self):
        self.context.set_symbolic_value(self.y, [1])
        original_shape = self.y.shape
        alias = ir.Value(shape=original_shape)
        original_data = self.context.get_symbolic_value(self.y)
        with self.assertRaises(ValueError), self.context.transaction(self.node):
            self.y.shape[0] = 999
            original_data[0] = 999
            raise ValueError("failed")
        self.assertIs(self.y.shape, original_shape)
        self.assertIs(alias.shape, original_shape)
        self.assertEqual(alias.shape, ir.Shape([1]))
        self.assertIs(self.context.get_symbolic_value(self.y), original_data)
        self.assertEqual(original_data, [1])

    def test_base_exception_rolls_back_and_context_remains_usable(self):
        with self.assertRaises(KeyboardInterrupt), self.context.transaction(self.node):
            self.context.set_dtype(self.y, ir.DataType.INT64)
            raise KeyboardInterrupt
        self.assertEqual(self.y.dtype, ir.DataType.FLOAT)
        with self.context.transaction(self.node):
            self.context.set_shape(self.y, ir.Shape([2]))
        self.assertEqual(self.y.shape, ir.Shape([2]))

    def test_clear_preserves_caller_replacement_constant(self):
        self.context.set_symbolic_value(self.y, [1])
        replacement = ir.tensor([8], dtype=ir.DataType.INT64)
        self.y.const_value = replacement
        self.context.clear_symbolic_value(self.y)
        self.assertIs(self.y.const_value, replacement)
        self.assertNotIn(_context.SYM_DATA_KEY, self.y.metadata_props)

    def test_clear_rolls_back_data_and_owned_constant(self):
        self.context.set_symbolic_value(self.y, [1])
        original = self.y.const_value
        with self.assertRaises(ValueError), self.context.transaction(self.node):
            self.context.clear_symbolic_value(self.y)
            raise ValueError("failed")
        self.assertIs(self.y.const_value, original)
        self.assertEqual(self.context.get_symbolic_value(self.y), [1])
        self.context.clear_symbolic_value(self.y)
        self.assertIsNone(self.y.const_value)


if __name__ == "__main__":
    unittest.main()
