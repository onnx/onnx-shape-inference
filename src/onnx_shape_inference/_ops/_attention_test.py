# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Attention and RotaryEmbedding shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
import parameterized

from onnx_shape_inference import OpUsageError, ShapeInferenceError, _context
from onnx_shape_inference._ops import _attention
from onnx_shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class AttentionTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [ts(FLOAT, [1, 10, 64]), ts(FLOAT, [1, 10, 64]), ts(FLOAT, [1, 10, 64])],
            opset_version=23,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 10, 64])])

    def test_q_shape_propagated(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [ts(FLOAT, [2, 8, 32]), ts(FLOAT, [2, 8, 32]), ts(FLOAT, [2, 8, 32])],
            opset_version=23,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 32])])

    def test_missing_q_shape_falls_back_to_v(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [ts(FLOAT), ts(FLOAT, [1, 10, 64]), ts(FLOAT, [1, 10, 64])],
            opset_version=23,
        )
        # Q has no shape → output shape is None (can't determine without Q)
        self.assertIsNone(actual[0].shape)

    def test_missing_all_shapes(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [ts(FLOAT), ts(FLOAT), ts(FLOAT)],
            opset_version=23,
        )
        self.assertIsNone(actual[0].shape)

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Attention", [], opset_version=23)

    def test_none_input(self):
        v = ir.Value(name="q", type=ir.TensorType(FLOAT), shape=ir.Shape([1, 10, 64]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Attention",
                [v, None, None],
                opset_version=23,
            )

    def test_present_key_value_outputs(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [ts(FLOAT, [1, 10, 64]), ts(FLOAT, [1, 10, 64]), ts(FLOAT, [1, 10, 64])],
            opset_version=23,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [1, 10, 64]))
        self.assertEqual(actual[1], ts(FLOAT, [1, 10, 64]))
        self.assertEqual(actual[2], ts(FLOAT, [1, 10, 64]))


class RotaryEmbeddingTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "RotaryEmbedding",
            [ts(FLOAT, [1, 10, 64])],
            opset_version=24,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 10, 64])])

    def test_missing_shape(self):
        actual = run_shape_inference(
            "",
            "RotaryEmbedding",
            [ts(FLOAT)],
            opset_version=24,
        )
        self.assertIsNone(actual[0].shape)

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "RotaryEmbedding", [], opset_version=24)

    def test_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "RotaryEmbedding",
                [None],
                opset_version=24,
            )

    def test_rotary_embedding_with_position_ids_two_outputs(self):
        input_val = ir.Value(
            name="input", shape=ir.Shape([1, 10, 64]), type=ir.TensorType(FLOAT)
        )
        position_ids = ir.Value(
            name="position_ids", shape=ir.Shape([1, 10]), type=ir.TensorType(INT64)
        )
        actual = run_shape_inference_with_values(
            "",
            "RotaryEmbedding",
            [input_val, position_ids],
            opset_version=24,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [1, 10, 64]))
        self.assertEqual(actual[1], ts(INT64, [1, 10]))


class AttentionSymbolicDimsTest(unittest.TestCase):
    def test_symbolic_dims(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [
                ts(FLOAT, ["B", "S", "D"]),
                ts(FLOAT, ["B", "S", "D"]),
                ts(FLOAT, ["B", "S", "D"]),
            ],
            opset_version=23,
        )
        self.assertEqual(actual, [ts(FLOAT, ["B", "S", "D"])])

    def test_symbolic_with_concrete_embed_dim(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [
                ts(FLOAT, ["B", "S", 64]),
                ts(FLOAT, ["B", "S", 64]),
                ts(FLOAT, ["B", "S", 64]),
            ],
            opset_version=23,
        )
        self.assertEqual(actual, [ts(FLOAT, ["B", "S", 64])])

    def test_symbolic_value_hidden_derives_gqa_output_hidden(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [
                ts(FLOAT, ["B", "S", "Q"]),
                ts(FLOAT, ["B", "S", "K"]),
                ts(FLOAT, ["B", "S", "V"]),
            ],
            attributes=_attrs(q_num_heads=16, kv_num_heads=8),
            opset_version=23,
        )
        self.assertEqual(actual, [ts(FLOAT, ["B", "S", "16*floor(V/8)"])])

    def test_4d_concrete(self):
        """4D inputs: [batch, num_heads, seq_len, head_size]."""
        actual = run_shape_inference(
            "",
            "Attention",
            [
                ts(FLOAT, [2, 8, 10, 64]),
                ts(FLOAT, [2, 8, 10, 64]),
                ts(FLOAT, [2, 8, 10, 64]),
            ],
            opset_version=23,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 10, 64]))
        self.assertEqual(actual[1], ts(FLOAT, [2, 8, 10, 64]))
        self.assertEqual(actual[2], ts(FLOAT, [2, 8, 10, 64]))

    def test_4d_symbolic(self):
        """4D inputs with symbolic batch and seq dims."""
        actual = run_shape_inference(
            "",
            "Attention",
            [
                ts(FLOAT, ["B", 8, "S", 64]),
                ts(FLOAT, ["B", 8, "Sk", 64]),
                ts(FLOAT, ["B", 8, "Sk", 64]),
            ],
            opset_version=23,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, ["B", 8, "S", 64]))
        self.assertEqual(actual[1], ts(FLOAT, ["B", 8, "Sk", 64]))
        self.assertEqual(actual[2], ts(FLOAT, ["B", 8, "Sk", 64]))


def _attrs(**kw) -> dict:
    out = {}
    for name, val in kw.items():
        if isinstance(val, int):
            out[name] = ir.Attr(name, ir.AttributeType.INT, val)
        elif isinstance(val, str):
            out[name] = ir.Attr(name, ir.AttributeType.STRING, val)
        elif isinstance(val, float):
            out[name] = ir.Attr(name, ir.AttributeType.FLOAT, val)
    return out


class LinearAttentionTest(unittest.TestCase):
    def test_prefill_no_past(self):
        actual = run_shape_inference(
            "",
            "LinearAttention",
            [ts(FLOAT, [2, 4, 32]), ts(FLOAT, [2, 4, 32]), ts(FLOAT, [2, 4, 32])],
            attributes=_attrs(q_num_heads=4, kv_num_heads=4),
            opset_version=27,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 4, 32]))
        # present_state [B, kv_num_heads, d_k, d_v] = [2, 4, 8, 8]
        self.assertEqual(actual[1], ts(FLOAT, [2, 4, 8, 8]))

    def test_gqa_output_hidden(self):
        """GQA: output hidden = q_num_heads * (value_hidden // kv_num_heads)."""
        actual = run_shape_inference(
            "",
            "LinearAttention",
            [ts(FLOAT, [2, 4, 64]), ts(FLOAT, [2, 4, 32]), ts(FLOAT, [2, 4, 32])],
            attributes=_attrs(q_num_heads=8, kv_num_heads=4),
            opset_version=27,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 4, 64]))
        self.assertEqual(actual[1], ts(FLOAT, [2, 4, 8, 8]))

    def test_mqa_present_state(self):
        actual = run_shape_inference(
            "",
            "LinearAttention",
            [ts(FLOAT, [2, 4, 64]), ts(FLOAT, [2, 4, 8]), ts(FLOAT, [2, 4, 8])],
            attributes=_attrs(q_num_heads=8, kv_num_heads=1),
            opset_version=27,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 4, 64]))
        self.assertEqual(actual[1], ts(FLOAT, [2, 1, 8, 8]))

    def test_past_state_is_ground_truth(self):
        """present_state copies past_state when provided."""
        actual = run_shape_inference_with_values(
            "",
            "LinearAttention",
            [
                ir.Value(name="q", shape=ir.Shape([2, 4, 32]), type=ir.TensorType(FLOAT)),
                ir.Value(name="k", shape=ir.Shape([2, 4, 32]), type=ir.TensorType(FLOAT)),
                ir.Value(name="v", shape=ir.Shape([2, 4, 32]), type=ir.TensorType(FLOAT)),
                ir.Value(name="ps", shape=ir.Shape([2, 4, 8, 8]), type=ir.TensorType(FLOAT)),
            ],
            attributes=_attrs(q_num_heads=4, kv_num_heads=4),
            opset_version=27,
            num_outputs=2,
        )
        self.assertEqual(actual[1], ts(FLOAT, [2, 4, 8, 8]))

    def test_symbolic_hidden_symbolic_heads(self):
        """Symbolic packed hidden -> output hidden and head dims symbolic."""
        actual = run_shape_inference(
            "",
            "LinearAttention",
            [
                ts(FLOAT, ["B", "T", "Dq"]),
                ts(FLOAT, ["B", "T", "Dk"]),
                ts(FLOAT, ["B", "T", "Dv"]),
            ],
            attributes=_attrs(q_num_heads=4, kv_num_heads=4),
            opset_version=27,
            num_outputs=2,
        )
        out0 = actual[0].shape
        self.assertEqual(str(out0[0]), "B")
        self.assertEqual(str(out0[1]), "T")
        self.assertIsInstance(out0[2], ir.SymbolicDim)
        present = actual[1].shape
        self.assertEqual(present[1], 4)
        self.assertIsInstance(present[2], ir.SymbolicDim)
        self.assertIsInstance(present[3], ir.SymbolicDim)


class LinearAttention29Test(unittest.TestCase):
    @parameterized.parameterized.expand([("", "states"), ("present", ""), ("", "")])
    def test_empty_output_slots(self, present_name, states_name):
        inputs = [
            ir.Value(name=name, type=ir.TensorType(FLOAT), shape=ir.Shape([2, 4, 32]))
            for name in ("q", "k", "v")
        ]
        outputs = [ir.Value(name=name) for name in ("output", present_name, states_name)]
        node = ir.Node(
            "",
            "LinearAttention",
            inputs,
            outputs=outputs,
            attributes={
                **_attrs(q_num_heads=4, kv_num_heads=4),
                "update_rule": ir.AttrString("update_rule", "linear"),
            },
        )
        _attention.infer_linear_attention_29(_context.ShapeInferenceContext({"": 29}), node)
        expected = [ts(FLOAT, [2, 4, 32]), ts(FLOAT, [2, 4, 8, 8]), ts(FLOAT, [2, 4, 4, 8, 8])]
        for value, spec in zip(outputs, expected):
            self.assertEqual(
                ir.TypeAndShape(value.type, value.shape), spec if value.name else ts()
            )

    @parameterized.parameterized.expand(
        [
            ("expanded", 4, 4, 12, 4, 24, 12, 24),
            ("equal_explicit", 4, 4, 4, 4, 24, 4, 24),
            ("default_gqa", 8, 4, None, 4, 24, 4, 48),
            ("zero_gqa", 8, 4, 0, 4, 24, 4, 48),
            ("empty_chunk", 4, 4, 12, 0, 24, 12, 24),
        ]
    )
    def test_head_layouts(self, _name, qh, kh, vh, time, width, sh, out):
        attrs = _attrs(q_num_heads=qh, kv_num_heads=kh)
        if vh is not None:
            attrs.update(_attrs(v_num_heads=vh))
        attrs["update_rule"] = ir.AttrString("update_rule", "linear")
        actual = run_shape_inference(
            "",
            "LinearAttention",
            [
                ts(FLOAT, [2, time, qh * 8]),
                ts(FLOAT, [2, time, kh * 8]),
                ts(FLOAT, [2, time, width]),
            ],
            attributes=attrs,
            opset_version=29,
            num_outputs=3,
        )
        self.assertEqual(
            actual,
            [
                ts(FLOAT, [2, time, out]),
                ts(FLOAT, [2, sh, 8, width // sh]),
                ts(FLOAT, [2, time, sh, 8, width // sh]),
            ],
        )

    @parameterized.parameterized.expand(
        [
            ("linear", None, None),
            ("gated", 12, None),
            ("gated_per_key", 96, None),
            ("delta", None, 12),
            ("gated_delta", 12, 1),
        ]
    )
    def test_rules_and_mixed_state_dtype(self, name, decay, beta):
        rule = "gated" if name == "gated_per_key" else name
        actual = run_shape_inference(
            "",
            "LinearAttention",
            [
                ts(ir.DataType.FLOAT16, [2, 4, 32]),
                ts(ir.DataType.FLOAT16, [2, 4, 32]),
                ts(ir.DataType.FLOAT16, [2, 4, 24]),
                ts(FLOAT, [2, 12, 8, 2]),
                ts(ir.DataType.FLOAT16, [2, 4, decay]) if decay else None,
                ts(ir.DataType.FLOAT16, [2, 4, beta]) if beta else None,
            ],
            attributes={
                **_attrs(q_num_heads=4, kv_num_heads=4, v_num_heads=12),
                "update_rule": ir.AttrString("update_rule", rule),
            },
            opset_version=29,
            num_outputs=3,
        )
        self.assertEqual(
            actual,
            [
                ts(ir.DataType.FLOAT16, [2, 4, 24]),
                ts(FLOAT, [2, 12, 8, 2]),
                ts(FLOAT, [2, 4, 12, 8, 2]),
            ],
        )

    def test_symbolic_shapes(self):
        actual = run_shape_inference(
            "",
            "LinearAttention",
            [
                ts(FLOAT, ["B", "T", "K"]),
                ts(FLOAT, ["B", "T", "K"]),
                ts(FLOAT, ["B", "T", "V"]),
            ],
            attributes={
                **_attrs(q_num_heads=4, kv_num_heads=4, v_num_heads=12),
                "update_rule": ir.AttrString("update_rule", "linear"),
            },
            opset_version=29,
            num_outputs=3,
        )
        self.assertEqual(
            actual,
            [
                ts(FLOAT, ["B", "T", "12*floor(V/12)"]),
                ts(FLOAT, ["B", 12, "floor(K/4)", "floor(V/12)"]),
                ts(FLOAT, ["B", "T", 12, "floor(K/4)", "floor(V/12)"]),
            ],
        )

    def test_unknown_inputs_inferred_from_state_and_gate(self):
        actual = run_shape_inference(
            "",
            "LinearAttention",
            [
                ts(FLOAT),
                ts(FLOAT),
                ts(FLOAT),
                ts(None, [2, 12, 8, 2]),
                ts(FLOAT, [2, "T", 12]),
            ],
            attributes={
                **_attrs(q_num_heads=4, kv_num_heads=4, v_num_heads=12),
                "update_rule": ir.AttrString("update_rule", "gated"),
            },
            opset_version=29,
            num_outputs=3,
        )
        self.assertEqual(
            actual,
            [
                ts(FLOAT, [2, "T", 24]),
                ts(None, [2, 12, 8, 2]),
                ts(None, [2, "T", 12, 8, 2]),
            ],
        )

    @parameterized.parameterized.expand(
        [
            ("negative_heads", {"v_num_heads": -1}, None),
            ("nonmultiple", {"v_num_heads": 6}, None),
            ("expanded_gqa", {"q_num_heads": 8}, None),
            ("zero_heads", {"kv_num_heads": 0}, None),
            ("zero_key", {}, (1, [2, 4, 0])),
            ("zero_value", {}, (2, [2, 4, 0])),
            ("indivisible_value", {}, (2, [2, 4, 25])),
            ("key_width_mismatch", {}, (1, [2, 4, 16])),
            ("batch_mismatch", {}, (2, [3, 4, 24])),
            ("wrong_rank", {}, (1, [2, 4])),
            ("wrong_state_heads", {}, (3, [2, 4, 8, 2])),
            ("wrong_decay", {}, (4, [2, 4, 4])),
            ("zero_decay", {}, (4, [2, 4, 0])),
            ("wrong_beta", {}, (5, [2, 4, 4])),
        ]
    )
    def test_invalid_layout(self, _name, attrs, replacement):
        inputs = [
            ts(FLOAT, [2, 4, 32]),
            ts(FLOAT, [2, 4, 32]),
            ts(FLOAT, [2, 4, 24]),
            None,
            ts(FLOAT, [2, 4, 12]),
            ts(FLOAT, [2, 4, 12]),
        ]
        if replacement is not None:
            index, shape = replacement
            inputs[index] = ts(FLOAT, shape)
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                "",
                "LinearAttention",
                inputs,
                attributes=_attrs(q_num_heads=4, kv_num_heads=4, v_num_heads=12)
                | _attrs(**attrs),
                opset_version=29,
                num_outputs=3,
            )

    @parameterized.parameterized.expand([("missing_gate", "gated_delta"), ("bad_rule", "bad")])
    def test_invalid_rule_inputs(self, _name, rule):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                "",
                "LinearAttention",
                [ts(FLOAT)] * 3,
                attributes={
                    **_attrs(q_num_heads=4, kv_num_heads=4),
                    "update_rule": ir.AttrString("update_rule", rule),
                },
                opset_version=29,
            )

    @parameterized.parameterized.expand([(27,), (28,)])
    def test_legacy_zero_width(self, version):
        actual = run_shape_inference(
            "",
            "LinearAttention",
            [ts(FLOAT, [2, 4, 0])] * 3,
            attributes=_attrs(q_num_heads=4, kv_num_heads=4),
            opset_version=version,
            num_outputs=2,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 4, 0]), ts(FLOAT, [2, 4, 0, 0])])


class FlexAttentionTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "ai.onnx.preview",
            "FlexAttention",
            [ts(FLOAT, [1, 2, 4, 8]), ts(FLOAT, [1, 2, 4, 8]), ts(FLOAT, [1, 2, 4, 8])],
            opset_version=1,
        )
        self.assertEqual(actual[0], ts(FLOAT, [1, 2, 4, 8]))

    def test_diff_head_sizes(self):
        """Y v_head_size comes from V's last dim; q_seq from Q."""
        actual = run_shape_inference(
            "ai.onnx.preview",
            "FlexAttention",
            [ts(FLOAT, [2, 4, 8, 16]), ts(FLOAT, [2, 4, 6, 16]), ts(FLOAT, [2, 4, 6, 32])],
            opset_version=1,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 4, 8, 32]))

    def test_gqa(self):
        actual = run_shape_inference(
            "ai.onnx.preview",
            "FlexAttention",
            [ts(FLOAT, [2, 8, 4, 16]), ts(FLOAT, [2, 2, 6, 16]), ts(FLOAT, [2, 2, 6, 16])],
            opset_version=1,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 4, 16]))

    def test_symbolic(self):
        actual = run_shape_inference(
            "ai.onnx.preview",
            "FlexAttention",
            [
                ts(FLOAT, ["B", "Hq", "Sq", "D"]),
                ts(FLOAT, ["B", "Hkv", "Sk", "D"]),
                ts(FLOAT, ["B", "Hkv", "Sk", "Dv"]),
            ],
            opset_version=1,
        )
        self.assertEqual(actual[0], ts(FLOAT, ["B", "Hq", "Sq", "Dv"]))

    def test_non_rank4_falls_back_to_q(self):
        actual = run_shape_inference(
            "ai.onnx.preview",
            "FlexAttention",
            [ts(FLOAT, [2, 4, 8]), ts(FLOAT, [2, 4, 8]), ts(FLOAT, [2, 4, 8])],
            opset_version=1,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 4, 8]))


if __name__ == "__main__":
    unittest.main()
