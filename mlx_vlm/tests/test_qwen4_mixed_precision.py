"""Synthetic operator and dispatch tests; no checkpoint is loaded."""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_vlm.models.qwen4_exp.language import (
    _QWEN4_BATCH_INVARIANT_FORWARD,
    LanguageModel,
    Qwen4ExpBatchInvariantForward,
    Qwen4ExpGatedResidual,
)
from mlx_vlm.models.qwen4_exp.mixed_precision import (
    Pending,
    Qwen4MixedVerifier,
    kernel,
    route_pack,
    validate_hyper,
)


@pytest.fixture
def hc():
    mx.random.seed(192)
    args = SimpleNamespace(
        hidden_size=2560, hc_count=4, hc_lowrank=320, rms_norm_eps=1e-6
    )
    layer = Qwen4ExpGatedResidual(args)
    layer.set_dtype(mx.bfloat16)
    layer.input_mix_weight_down = layer.input_mix_weight_down.to_quantized(
        bits=8, group_size=64
    )
    layer.input_mix_weight_up = layer.input_mix_weight_up.to_quantized(
        bits=8, group_size=64
    )
    mx.eval(layer.parameters())
    return layer


@pytest.mark.parametrize("width", [2, 4, 8])
def test_hyper_and_deferred_residual(hc, width):
    validate_hyper(hc)
    v = Qwen4MixedVerifier()
    base = Qwen4ExpBatchInvariantForward()
    x = mx.random.normal((1, width, 10240)).astype(mx.bfloat16)
    branch = mx.random.normal((1, width, 2560)).astype(mx.bfloat16) * 0.1
    injection = mx.random.uniform(shape=(1, width, 4)).astype(mx.bfloat16)
    for value in (x, Pending(x, branch, injection)):
        source = value.materialize() if isinstance(value, Pending) else value
        expected = base._hyper_connection(hc, source)
        actual = v._hyper_connection(hc, value)
        mx.eval(expected, actual)
        assert mx.array_equal(actual[1], source).item()
        for a, b in zip(expected, actual):
            assert mx.all(mx.isfinite(b)).item()
            np.testing.assert_allclose(
                np.array(a.astype(mx.float32)),
                np.array(b.astype(mx.float32)),
                atol=0.025,
                rtol=0.025,
            )


@pytest.mark.parametrize("width", [2, 4, 8])
def test_route_pack_and_weighted_reduction(width):
    ids = (mx.arange(width * 10) % 7).reshape(1, width, 10)
    inverse, sorted_ids, lhs = route_pack(ids)
    down = mx.random.normal((width * 10, 2560)).astype(mx.bfloat16)
    scores = mx.softmax(mx.random.normal((1, width, 10))).astype(mx.bfloat16)
    out = kernel("verify_expert_reduce_source")(
        inputs=[down, inverse, scores],
        template=[("T", mx.bfloat16)],
        grid=(width * 2560, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, width, 2560)],
        output_dtypes=[mx.bfloat16],
    )[0]
    expected = (down[inverse].reshape(1, width, 10, 2560) * scores[..., None]).sum(
        axis=-2
    )
    mx.eval(out, expected, sorted_ids, lhs)
    assert mx.array_equal(sorted_ids[inverse], ids.reshape(-1)).item()
    assert mx.array_equal(lhs[inverse], mx.arange(width * 10) // 10).item()
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=0.015,
        rtol=0.015,
    )


def test_dispatch_is_model_local_and_keeps_unsupported_batches():
    v = Qwen4MixedVerifier()
    lm = SimpleNamespace(_mixed_verifier=v)
    assert LanguageModel._verification_forward(lm, 1, 4) is v
    for shape in ((2, 4), (1, 1), (1, 9)):
        assert (
            LanguageModel._verification_forward(lm, *shape)
            is _QWEN4_BATCH_INVARIANT_FORWARD
        )
    other = SimpleNamespace()
    assert (
        LanguageModel._verification_forward(other, 1, 4)
        is _QWEN4_BATCH_INVARIANT_FORWARD
    )
    LanguageModel.configure_qwen4_optimizations(lm, "off")
    assert lm._mixed_verifier is None
    with pytest.raises(ValueError):
        LanguageModel.configure_qwen4_optimizations(lm, "unknown")


def test_reject_wrong_hyper_quantization(hc):
    hc.input_mix_weight_down.group_size = 32
    with pytest.raises(ValueError):
        validate_hyper(hc)
