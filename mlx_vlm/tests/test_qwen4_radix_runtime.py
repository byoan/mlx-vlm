"""Runtime-length selection must preserve the static kernel's ordered IDs."""

import os
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx_vlm.models.qwen4_exp import exact_sparse_qsa as qsa

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Radix QSA requires Metal"
)


@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize("blocks", [16384, 16385, 18730, 18731])
@pytest.mark.parametrize("kind", ["random", "ties", "zero", "masked"])
def test_runtime_radix_preserves_ordered_selection(rows, blocks, kind):
    scores = mx.random.uniform(shape=(1, rows, blocks), key=mx.random.key(1732))
    if kind == "ties":
        scores = mx.floor(scores * 8)
    elif kind == "zero":
        scores = mx.zeros_like(scores)
    elif kind == "masked":
        scores = mx.where(mx.arange(blocks) < 600, mx.zeros_like(scores), -mx.inf)
    with (
        patch.dict(os.environ, {"MLX_VLM_QWEN4_RADIX_QSA_TOPK": "1"}),
        patch.object(qsa, "enabled", return_value=True),
    ):
        actual = qsa.select_blocks(scores, 512)
    reference = qsa._topk_kernel()(
        inputs=[scores.reshape(rows, blocks)],
        template=[("N", blocks), ("K", 512)],
        grid=(256, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, 512)],
        output_dtypes=[mx.int32],
    )[0].reshape(1, rows, 512)
    mx.eval(actual, reference)
    assert mx.array_equal(actual, reference).item()
    if kind == "masked":
        assert mx.all(actual < 600).item()


def test_decode_specialization_is_independent_of_context_length():
    calls = []

    def kernel(**kwargs):
        calls.append(kwargs["template"])
        return [mx.zeros(kwargs["output_shapes"][0], dtype=mx.int32)]

    with (
        patch.dict(os.environ, {"MLX_VLM_QWEN4_RADIX_QSA_TOPK": "1"}),
        patch.object(qsa, "enabled", return_value=True),
        patch.object(qsa, "_topk_kernel", return_value=kernel) as factory,
    ):
        for blocks in (16384, 16385, 18730):
            qsa.select_blocks(mx.zeros((1, 4, blocks)), 512)
        assert all(c.kwargs == {"runtime_length": True} for c in factory.call_args_list)
        assert calls == [[("K", 512)]] * 3


def test_prefill_keeps_static_length_specialization():
    def kernel(**kwargs):
        assert kwargs["template"] == [("N", 8192), ("K", 512)]
        return [mx.zeros(kwargs["output_shapes"][0], dtype=mx.int32)]

    with (
        patch.dict(os.environ, {"MLX_VLM_QWEN4_PREFILL_RADIX_QSA_TOPK": "1"}),
        patch.object(qsa, "_is_m3_ultra", return_value=True),
        patch.object(qsa, "_topk_kernel", return_value=kernel) as factory,
    ):
        qsa.select_blocks(mx.zeros((1, 16, 8192)), 512)
        factory.assert_called_once_with(runtime_length=False)
