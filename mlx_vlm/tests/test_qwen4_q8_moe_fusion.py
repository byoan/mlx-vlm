"""Q8 router dispatch, ties and reduction correctness without large experts."""

import os
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.qwen4_exp import language as ql


class RouterFixture(nn.Module):
    def __init__(
        self, bits=8, group_size=64, top_k=10, dense_shared=False, float32_scales=False
    ):
        super().__init__()
        self.gate = nn.QuantizedLinear(
            2560, 512, bias=False, bits=bits, group_size=group_size, mode="affine"
        )
        self.shared_expert_gate = nn.QuantizedLinear(
            2560, 1, bias=False, bits=bits, group_size=group_size, mode="affine"
        )
        if not float32_scales:
            self.gate.set_dtype(mx.bfloat16)
            self.shared_expert_gate.set_dtype(mx.bfloat16)
        if dense_shared:
            self.shared_expert_gate = nn.Linear(2560, 1, bias=False)
            self.shared_expert_gate.set_dtype(mx.bfloat16)
        self.shared_expert = nn.Identity()
        self.switch_mlp = nn.Identity()
        self.top_k = top_k


def routed(_module, x, indices):
    # Expert-ID-dependent outputs detect selection/order mistakes as well as
    # score normalization and shared-expert gating errors. No giant expert bank.
    basis = mx.arange(2560).astype(mx.bfloat16) / 2560
    values = indices.astype(mx.bfloat16)[..., None] / 512
    return values * basis + x[..., None, :] * mx.array(0.125, dtype=mx.bfloat16)


class TestQwen4Q8MoEFusion(unittest.TestCase):
    def setUp(self):
        mx.random.seed(917)
        self.verifier = ql._QWEN4_EXACT_SPECULATIVE_VERIFIER
        self.flags = {
            "MLX_VLM_QWEN4_COMBINED_MOE_GATE_PROJECTION": "1",
            "MLX_VLM_QWEN4_FUSED_MOE_ROUTE": "1",
            "MLX_VLM_QWEN4_FUSED_MOE_COMBINE": "1",
        }

    def compare(self, module, x, *, enabled):
        original_route = ql.exact_moe_route

        def checked_route(projection):
            result = original_route(projection)
            self.assertIsNotNone(result)
            return result

        with (
            patch.dict(os.environ, self.flags),
            patch.object(self.verifier, "_switch_glu", side_effect=routed),
            patch.object(ql, "exact_moe_route", side_effect=checked_route) as route,
        ):
            with patch.dict(os.environ, {"MLX_VLM_QWEN4_Q8_MOE_FUSION": "0"}):
                expected = self.verifier._feed_forward(module, x)
                mx.eval(expected)
            self.assertEqual(route.call_count, 0)
            with patch.dict(os.environ, {"MLX_VLM_QWEN4_Q8_MOE_FUSION": "1"}):
                actual = self.verifier._feed_forward(module, x)
                mx.eval(actual)
            self.assertEqual(route.call_count, int(enabled))
            self.assertTrue(mx.array_equal(actual, expected).item())

    def test_all_supported_widths_and_tied_gates(self):
        for dense_shared in (False, True):
            module = RouterFixture(dense_shared=dense_shared)
            for width in range(2, 9):
                for tied in (False, True):
                    with self.subTest(
                        width=width, tied=tied, dense_shared=dense_shared
                    ):
                        x = (
                            mx.zeros((1, width, 2560))
                            if tied
                            else mx.random.normal((1, width, 2560))
                        ).astype(mx.bfloat16)
                        self.compare(module, x, enabled=True)

    def test_unsupported_layouts_keep_existing_path(self):
        cases = [
            ((1, 1, 2560), mx.bfloat16, {}),
            ((2, 4, 2560), mx.bfloat16, {}),
            ((1, 9, 2560), mx.bfloat16, {}),
            ((1, 4, 2560), mx.float16, {}),
            ((1, 4, 2560), mx.bfloat16, {"bits": 4}),
            ((1, 4, 2560), mx.bfloat16, {"group_size": 32}),
            ((1, 4, 2560), mx.bfloat16, {"top_k": 8}),
            ((1, 4, 2560), mx.bfloat16, {"float32_scales": True}),
            (
                (1, 4, 2560),
                mx.bfloat16,
                {"dense_shared": True, "float32_scales": True},
            ),
        ]
        for shape, dtype, kw in cases:
            with self.subTest(shape=shape, dtype=dtype, kwargs=kw):
                self.compare(
                    RouterFixture(**kw),
                    mx.random.normal(shape).astype(dtype),
                    enabled=False,
                )


if __name__ == "__main__":
    unittest.main()
