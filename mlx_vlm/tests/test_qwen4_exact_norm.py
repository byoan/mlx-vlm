import os
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx_vlm.models.qwen4_exp.exact_norm import exact_norm
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNorm


@pytest.mark.parametrize("group", [None, 2560])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 8, 9])
@pytest.mark.parametrize("weight_dtype", [mx.bfloat16, mx.float32])
def test_exact_norm_preserves_rounding_and_strided_inputs(group, rows, weight_dtype):
    if (
        not mx.metal.is_available()
        or mx.device_info().get("device_name") != "Apple M3 Ultra"
    ):
        pytest.skip("Exact norm is validated on M3 Ultra")

    module = Qwen4ExpRMSNorm(10240, group_size=group)
    # These seeds expose differences in a whole-norm compiled implementation.
    for seed in (5, 7, 13):
        module.weight = (
            mx.random.normal((10240,), key=mx.random.key(seed)) * 0.1
        ).astype(weight_dtype)
        inputs = mx.random.normal(
            (1, rows * 2, 10240), key=mx.random.key(1000 + seed)
        ).astype(mx.bfloat16)[:, ::2]
        with patch.dict(os.environ, MLX_VLM_QWEN4_EXACT_NORM="0"):
            expected = module(inputs)
        actual = exact_norm(inputs, module.weight, group, 1e-6)
        assert actual is not None
        assert mx.array_equal(expected, actual).item()
        with patch.dict(os.environ, MLX_VLM_QWEN4_EXACT_NORM="1"):
            assert mx.array_equal(expected, module(inputs)).item()


@pytest.mark.parametrize(
    "shape,dtype,group,eps",
    [
        ((2, 4, 10240), mx.bfloat16, None, 1e-6),
        ((1, 10, 10240), mx.bfloat16, None, 1e-6),
        ((1, 4, 2560), mx.bfloat16, None, 1e-6),
        ((1, 4, 10240), mx.float16, None, 1e-6),
        ((1, 4, 10240), mx.float32, None, 1e-6),
        ((1, 4, 10240), mx.bfloat16, 1280, 1e-6),
        ((1, 4, 10240), mx.bfloat16, None, 1e-5),
    ],
)
def test_exact_norm_unsupported_layout_falls_back(shape, dtype, group, eps):
    module = Qwen4ExpRMSNorm(shape[-1], group_size=group, eps=eps)
    inputs = mx.ones(shape, dtype=dtype)
    assert exact_norm(inputs, module.weight, group, eps) is None
    with patch.dict(os.environ, MLX_VLM_QWEN4_EXACT_NORM="0"):
        expected = module(inputs)
    with patch.dict(os.environ, MLX_VLM_QWEN4_EXACT_NORM="1"):
        assert mx.array_equal(expected, module(inputs)).item()
