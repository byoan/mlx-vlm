"""Exact fused routing tail for Qwen3.8 Flash Next verification."""

import mlx.core as mx

_EXACT_MOE_ROUTE = (
    mx.fast.metal_kernel(
        name="qwen4_exact_moe_route",
        input_names=["projection"],
        output_names=["expert_ids", "route_scores", "shared_factor"],
        header=r"""
        #include <metal_simdgroup>
        #include <metal_stdlib>
        using namespace metal;

        """,
        source=r"""
        constexpr int N_EXPERTS = 512;
        constexpr int TOP_K = 10;
        constexpr int N_THREADS = 128;
        constexpr int N_SIMDGROUPS = N_THREADS / 32;
        constexpr int READS = 4;

        const uint row = threadgroup_position_in_grid.x;
        const uint lid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint simdgroup = simdgroup_index_in_threadgroup;

        threadgroup float local_max[32];
        threadgroup float local_norm[32];
        threadgroup bfloat gate_values[N_EXPERTS];
        threadgroup float partial_values[N_SIMDGROUPS];
        threadgroup uint partial_indices[N_SIMDGROUPS];
        threadgroup uint selected[TOP_K];

        const device bfloat* input = projection + row * 513 + lid * READS;
        float values[READS];
        #pragma clang loop unroll(full)
        for (int i = 0; i < READS; ++i) values[i] = float(input[i]);

        if (simdgroup == 0) {
            local_max[lane] = -metal::numeric_limits<float>::infinity();
            local_norm[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float max_value = -metal::numeric_limits<float>::max();
        #pragma clang loop unroll(full)
        for (int i = 0; i < READS; ++i) {
            max_value = max_value < values[i] ? values[i] : max_value;
        }
        max_value = simd_max(max_value);
        if (lane == 0) local_max[simdgroup] = max_value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup == 0) {
            max_value = simd_max(local_max[lane]);
            if (lane == 0) local_max[0] = max_value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_value = local_max[0];

        float normalizer = 0.0f;
        #pragma clang loop unroll(full)
        for (int i = 0; i < READS; ++i) {
            const float exponent = fast::exp(values[i] - max_value);
            values[i] = exponent;
            normalizer += exponent;
        }
        normalizer = simd_sum(normalizer);
        if (lane == 0) local_norm[simdgroup] = normalizer;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup == 0) {
            normalizer = simd_sum(local_norm[lane]);
            if (lane == 0) local_norm[0] = normalizer;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        normalizer = 1.0f / local_norm[0];

        #pragma clang loop unroll(full)
        for (int i = 0; i < READS; ++i) {
            const uint index = lid * READS + i;
            const bfloat gate = bfloat(values[i] * normalizer);
            gate_values[index] = gate;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int pick = 0; pick < TOP_K; ++pick) {
            float best_value = -metal::numeric_limits<float>::infinity();
            uint best_index = 0;
            #pragma clang loop unroll(full)
            for (int i = 0; i < READS; ++i) {
                const uint index = lid * READS + i;
                bool used = false;
                for (int prior = 0; prior < pick; ++prior) {
                    used = used || selected[TOP_K - 1 - prior] == index;
                }
                const float value = used
                    ? -metal::numeric_limits<float>::infinity()
                    : float(gate_values[index]);
                if (
                    value > best_value ||
                    (value == best_value && index > best_index)
                ) {
                    best_value = value;
                    best_index = index;
                }
            }
            for (ushort offset = 16; offset >= 1; offset >>= 1) {
                const float other_value = simd_shuffle_down(best_value, offset);
                const uint other_index = simd_shuffle_down(best_index, offset);
                if (
                    other_value > best_value ||
                    (other_value == best_value && other_index > best_index)
                ) {
                    best_value = other_value;
                    best_index = other_index;
                }
            }
            if (lane == 0) {
                partial_values[simdgroup] = best_value;
                partial_indices[simdgroup] = best_index;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid == 0) {
                float winner_value = partial_values[0];
                uint winner_index = partial_indices[0];
                #pragma clang loop unroll(full)
                for (int group = 1; group < N_SIMDGROUPS; ++group) {
                    const float value = partial_values[group];
                    const uint index = partial_indices[group];
                    if (
                        value > winner_value ||
                        (value == winner_value && index > winner_index)
                    ) {
                        winner_value = value;
                        winner_index = index;
                    }
                }
                selected[TOP_K - 1 - pick] = winner_index;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid == 0) {
            bfloat picked[TOP_K];
            #pragma clang loop unroll(full)
            for (int index = 0; index < TOP_K; ++index) {
                const uint expert = selected[index];
                expert_ids[row * TOP_K + index] = expert;
                picked[index] = gate_values[expert];
            }
            bfloat total = bfloat(0.0f);
            #pragma clang loop unroll(full)
            for (int index = 0; index < TOP_K; ++index) {
                total = picked[index] + total;
            }
            #pragma clang loop unroll(full)
            for (int index = 0; index < TOP_K; ++index) {
                route_scores[row * TOP_K + index] = picked[index] / total;
            }

            const bfloat shared = projection[row * 513 + N_EXPERTS];
            auto sigmoid_y = 1 / (1 + metal::exp(metal::abs(shared)));
            shared_factor[row] = shared < bfloat(0.0f)
                ? bfloat(sigmoid_y)
                : bfloat(1 - sigmoid_y);
        }
        """,
        ensure_row_contiguous=True,
    )
    if mx.metal.is_available()
    else None
)


def exact_moe_route(projection: mx.array):
    """Return exact top-10 routing outputs, or ``None`` when unsupported."""

    if (
        _EXACT_MOE_ROUTE is None
        or projection.ndim != 3
        or projection.shape[0] != 1
        or not 1 < projection.shape[1] <= 8
        or projection.shape[2] != 513
        or projection.dtype != mx.bfloat16
    ):
        return None

    batch, length, _ = projection.shape
    rows = batch * length
    expert_ids, route_scores, shared_factor = _EXACT_MOE_ROUTE(
        inputs=[projection],
        grid=(128 * rows, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, 10), (rows, 10), (rows,)],
        output_dtypes=[mx.uint32, mx.bfloat16, mx.bfloat16],
    )
    return (
        expert_ids.reshape(batch, length, 10),
        route_scores.reshape(batch, length, 10),
        shared_factor.reshape(batch, length, 1),
    )


__all__ = ["exact_moe_route"]
