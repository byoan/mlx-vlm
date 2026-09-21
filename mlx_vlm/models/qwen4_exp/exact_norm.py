"""Qwen4 normalization tail with the original FP32 rounding boundaries."""

from functools import lru_cache

import mlx.core as mx

_APPLY_NORM = (
    mx.fast.metal_kernel(
        name="qwen4_exact_norm_apply",
        input_names=["x", "variance", "weight"],
        output_names=["output"],
        source=r"""
        #pragma clang fp contract(off)
        uint index = thread_position_in_grid.x;
        if (index >= COUNT) return;

        float mean_square = variance[index / GROUP];
        float inv_rms = metal::precise::rsqrt(mean_square + 1e-6f);
        float normalized = float(x[index]) * inv_rms;
        float scale = 1.0f + float(weight[index % 10240]);
        output[index] = bfloat(normalized * scale);
        """,
        ensure_row_contiguous=True,
    )
    if mx.metal.is_available()
    else None
)


@lru_cache(maxsize=1)
def _supported_device():
    return (
        mx.metal.is_available()
        and mx.device_info().get("device_name") == "Apple M3 Ultra"
    )


def exact_norm(x, weight, group_size, eps):
    """Return normalized BF16 values, or ``None`` outside the validated layout."""
    if (
        _APPLY_NORM is None
        or x.ndim != 3
        or x.shape[0] != 1
        or not 1 <= x.shape[1] <= 9
        or x.shape[2] != 10240
        or x.dtype != mx.bfloat16
        or weight.shape != (10240,)
        or weight.dtype not in (mx.bfloat16, mx.float32)
        or group_size not in (None, 2560)
        or eps != 1e-6
        or not _supported_device()
    ):
        return None

    # Keep MLX's square/mean reduction and its rounded FP32 output. Compiling
    # the entire norm can change intermediate rounding and BF16 outputs, even
    # when a particular greedy continuation still matches.
    values = x.astype(mx.float32)
    if group_size is not None:
        values = values.reshape(*x.shape[:-1], -1, group_size)
    variance = mx.mean(mx.square(values), axis=-1, keepdims=True)
    return _APPLY_NORM(
        inputs=[x, variance, weight],
        template=[("COUNT", x.size), ("GROUP", group_size or 10240)],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.bfloat16],
    )[0]
