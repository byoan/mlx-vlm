"""Exact fused MoE reduction and shared-expert add for Qwen4 verification."""

import mlx.core as mx

_EXACT_MOE_COMBINE = (
    mx.fast.metal_kernel(
        name="qwen4_exact_moe_combine",
        input_names=["routed", "shared", "scores", "shared_gate"],
        output_names=["output"],
        source=r"""
        uint index = thread_position_in_grid.x;
        if (index >= ROWS * 2560) return;
        uint row = index / 2560;
        uint column = index - row * 2560;

        bfloat products[10];
        #pragma clang loop unroll(full)
        for (uint slot = 0; slot < 10; ++slot) {
            products[slot] = bfloat(
                float(routed[(row * 10 + slot) * 2560 + column]) *
                float(scores[row * 10 + slot]));
        }

        // Match MLX col_reduce_small for ten BF16 rows: its eight y lanes
        // first fold rows 8 and 9, then lane zero folds lanes 1 through 7.
        bfloat routed_sum = bfloat(float(products[0]) + float(products[8]));
        bfloat second = bfloat(float(products[1]) + float(products[9]));
        routed_sum = bfloat(float(second) + float(routed_sum));
        #pragma clang loop unroll(full)
        for (uint slot = 2; slot < 8; ++slot) {
            routed_sum = bfloat(float(products[slot]) + float(routed_sum));
        }

        bfloat gated_shared = bfloat(
            float(shared_gate[row]) * float(shared[row * 2560 + column]));
        output[index] = bfloat(float(routed_sum) + float(gated_shared));
        """,
        ensure_row_contiguous=True,
    )
    if mx.metal.is_available()
    else None
)


def exact_moe_combine(routed, shared, scores, shared_gate):
    """Return the exact fused MoE output, or ``None`` when unsupported."""

    if (
        _EXACT_MOE_COMBINE is None
        or routed.ndim != 4
        or shared.ndim != 3
        or scores.ndim != 3
        or shared_gate.ndim != 3
        or routed.shape[0] != 1
        or not 1 < routed.shape[1] <= 8
        or routed.shape[2:] != (10, 2560)
        or shared.shape != routed.shape[:2] + (2560,)
        or scores.shape != routed.shape[:3]
        or shared_gate.shape != routed.shape[:2] + (1,)
        or any(
            value.dtype != mx.bfloat16
            for value in (routed, shared, scores, shared_gate)
        )
    ):
        return None

    rows = routed.shape[0] * routed.shape[1]
    return _EXACT_MOE_COMBINE(
        inputs=[routed, shared, scores, shared_gate],
        template=[("ROWS", rows)],
        grid=(rows * 2560, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shared.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


__all__ = ["exact_moe_combine"]
