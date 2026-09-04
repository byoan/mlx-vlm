import os
from functools import lru_cache

import mlx.core as mx

_PASS1_SOURCE = r"""
    uint kv_head = threadgroup_position_in_grid.x;
    uint batch = threadgroup_position_in_grid.y;
    uint block = threadgroup_position_in_grid.z;
    uint q_group = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    constexpr int BD = 32;
    constexpr int QK_PER_THREAD = D_SIZE / BD;
    constexpr int V_PER_THREAD = D_SIZE / BD;
    uint q_head = kv_head * GQA_FACTOR + q_group;

    const device T* qptr =
        queries + (batch * NUM_Q_HEADS + q_head) * D_SIZE +
        lane * QK_PER_THREAD;
    int begin = int(offsets[block]);
    int end = int(offsets[block + 1]);
    int khs = int(k_head_stride[0]);
    int vhs = int(v_head_stride[0]);

    float q[QK_PER_THREAD];
    float out_acc[V_PER_THREAD] = {0};
    float s = float(scale[0]);
    for (int i = 0; i < QK_PER_THREAD; ++i) {
        q[i] = s * float(qptr[i]);
    }

    float max_score = -3.4028234663852886e38f;
    float sum_exp = 0.0f;
    for (int cursor = begin; cursor < end; ++cursor) {
        int key_pos = int(positions[cursor]);
        const device T* kptr =
            keys + batch * NUM_KV_HEADS * khs + kv_head * khs +
            key_pos * D_SIZE + lane * QK_PER_THREAD;
        float score = 0.0f;
        for (int i = 0; i < QK_PER_THREAD; ++i) {
            score += q[i] * float(kptr[i]);
        }
        score = simd_sum(score);

        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;

        const device T* vptr =
            values + batch * NUM_KV_HEADS * vhs + kv_head * vhs +
            key_pos * D_SIZE + lane * V_PER_THREAD;
        for (int i = 0; i < V_PER_THREAD; ++i) {
            out_acc[i] = out_acc[i] * factor + exp_score * float(vptr[i]);
        }
    }

    uint partial_offset =
        ((batch * NUM_Q_HEADS + q_head) * 1024 + block) * D_SIZE +
        lane * V_PER_THREAD;
    for (int i = 0; i < V_PER_THREAD; ++i) {
        partials[partial_offset + i] = T(out_acc[i]);
    }
    if (lane == 0) {
        uint scalar_offset = (batch * NUM_Q_HEADS + q_head) * 1024 + block;
        sums[scalar_offset] = sum_exp;
        maxs[scalar_offset] = max_score;
    }
"""


_PASS2_SOURCE = r"""
    uint q_head = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int ELEM_PER_THREAD = D_SIZE / BD;
    threadgroup float outputs[BN * BD];
    float out_acc[ELEM_PER_THREAD] = {0};

    const device T* partial =
        partials + (q_head * 1024 + simd_gid) * D_SIZE +
        lane * ELEM_PER_THREAD;
    const device float* local_sums = sums + q_head * 1024;
    const device float* local_maxs = maxs + q_head * 1024;

    float max_score = -3.4028234663852886e38f;
    for (int b = 0; b < 1024 / BN; ++b) {
        max_score = max(max_score, local_maxs[lane + BN * b]);
    }
    max_score = simd_max(max_score);

    float sum_exp = 0.0f;
    for (int b = 0; b < 1024 / BN; ++b) {
        float factor = fast::exp(local_maxs[lane + BN * b] - max_score);
        sum_exp += factor * local_sums[lane + BN * b];
    }
    sum_exp = simd_sum(sum_exp);

    for (int b = 0; b < 1024 / BN; ++b) {
        float factor = fast::exp(local_maxs[simd_gid] - max_score);
        for (int i = 0; i < ELEM_PER_THREAD; ++i) {
            out_acc[i] += factor * float(partial[i]);
        }
        local_maxs += BN;
        partial += BN * D_SIZE;
    }

    for (int i = 0; i < ELEM_PER_THREAD; ++i) {
        outputs[lane * BD + simd_gid] = out_acc[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        out_acc[i] = simd_sum(outputs[simd_gid * BD + lane]);
        out_acc[i] = sum_exp == 0 ? out_acc[i] : out_acc[i] / sum_exp;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        device T* optr = out + q_head * D_SIZE + simd_gid * ELEM_PER_THREAD;
        for (int i = 0; i < ELEM_PER_THREAD; ++i) {
            optr[i] = T(out_acc[i]);
        }
    }
"""


@lru_cache(maxsize=None)
def _kernels(dtype, head_dim, q_heads, kv_heads):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    prefix = (
        f"qwen4_exact_sparse_qsa_{dtype_name}_d{head_dim}_" f"qh{q_heads}_kh{kv_heads}"
    )
    header = "#include <metal_simdgroup>\nusing namespace metal;\n"
    return (
        mx.fast.metal_kernel(
            name=f"{prefix}_pass1",
            input_names=[
                "queries",
                "keys",
                "values",
                "positions",
                "offsets",
                "scale",
                "k_head_stride",
                "v_head_stride",
            ],
            output_names=["partials", "sums", "maxs"],
            header=header,
            source=_PASS1_SOURCE,
        ),
        mx.fast.metal_kernel(
            name=f"{prefix}_pass2",
            input_names=["partials", "sums", "maxs"],
            output_names=["out"],
            header=header,
            source=_PASS2_SOURCE,
        ),
    )


@lru_cache(maxsize=32)
def _scalars(scale, k_head_stride, v_head_stride):
    return (
        mx.array([scale], dtype=mx.float32),
        mx.array([k_head_stride], dtype=mx.int32),
        mx.array([v_head_stride], dtype=mx.int32),
    )


@lru_cache(maxsize=1)
def _is_m3_ultra():
    if not mx.metal.is_available():
        return False
    info = mx.device_info()
    return info.get("device_name") == "Apple M3 Ultra" and str(
        info.get("architecture", "")
    ).endswith("d")


def enabled():
    blocks = os.environ.get("MLX_SDPA_BLOCKS", "0")
    return (
        os.environ.get("MLX_VLM_QWEN4_EXACT_SPARSE_QSA") == "1"
        and blocks in ("", "0", "1024")
        and _is_m3_ultra()
    )


def _selected_positions(selection, query_index):
    block_size = selection.block_size
    blocks = selection.selected_blocks[0, query_index].astype(mx.int32)
    positions = (
        blocks[:, None] * block_size + mx.arange(block_size, dtype=mx.int32)[None]
    ).reshape(-1)

    tail_capacity = block_size - 1
    tail_start = selection.complete_counts[0, query_index] * block_size
    tail = tail_start + mx.arange(tail_capacity, dtype=mx.int32)
    tail_valid = tail < selection.query_ends[query_index]
    positions = mx.concatenate([positions, tail])
    valid = mx.concatenate(
        [mx.ones((blocks.size * block_size,), dtype=mx.bool_), tail_valid]
    )

    partitions = mx.where(valid, positions % 1024, 1024)
    safe_positions = mx.where(valid, positions, 0)
    order = mx.argsort(partitions * (selection.key_length + 1) + safe_positions)
    positions = positions[order]
    partitions = partitions[order]
    offsets = mx.searchsorted(partitions, mx.arange(1025, dtype=mx.int32))
    return positions, offsets


def _attention(
    query,
    keys,
    values,
    positions,
    offsets,
    scale,
    k_head_stride,
    v_head_stride,
):
    if (
        query.shape != (1, 24, 1, 256)
        or keys.ndim != 4
        or values.shape != keys.shape
        or keys.shape[0] != 1
        or keys.shape[1] != 2
        or keys.shape[-1] != 256
        or keys.shape[2] < 65_536
        or query.dtype not in (mx.bfloat16, mx.float16)
        or keys.dtype != query.dtype
        or values.dtype != query.dtype
        or k_head_stride < keys.shape[2] * 256
        or v_head_stride < values.shape[2] * 256
    ):
        return None

    pass1, pass2 = _kernels(query.dtype, 256, 24, 2)
    scale_value, k_stride, v_stride = _scalars(
        float(scale), int(k_head_stride), int(v_head_stride)
    )
    first = pass1(
        inputs=[
            mx.contiguous(query),
            keys,
            values,
            positions,
            offsets,
            scale_value,
            k_stride,
            v_stride,
        ],
        template=[
            ("T", query.dtype),
            ("D_SIZE", 256),
            ("NUM_Q_HEADS", 24),
            ("NUM_KV_HEADS", 2),
            ("GQA_FACTOR", 12),
        ],
        grid=(64, 12, 1024),
        threadgroup=(32, 12, 1),
        output_shapes=[
            (1, 24, 1024, 256),
            (1, 24, 1024),
            (1, 24, 1024),
        ],
        output_dtypes=[query.dtype, mx.float32, mx.float32],
    )
    return pass2(
        inputs=first,
        template=[("T", query.dtype), ("D_SIZE", 256)],
        grid=(1024, 24, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[query.shape],
        output_dtypes=[query.dtype],
    )[0]


class Qwen4ExactSparseSelection:
    def __init__(
        self,
        selected_blocks,
        complete_counts,
        query_ends,
        key_length,
        block_size,
    ):
        self.selected_blocks = selected_blocks
        self.complete_counts = complete_counts
        self.query_ends = query_ends
        self.key_length = key_length
        self.block_size = block_size

    def apply(self, queries, keys, values, scale, cache):
        storage = getattr(cache, "kv_cache", cache)
        if (
            storage is None
            or getattr(storage, "keys", None) is None
            or getattr(storage, "values", None) is None
        ):
            return None
        k_head_stride = int(storage.keys.shape[-2]) * keys.shape[-1]
        v_head_stride = int(storage.values.shape[-2]) * values.shape[-1]
        outputs = []
        for query_index in range(queries.shape[2]):
            positions, offsets = _selected_positions(self, query_index)
            output = _attention(
                queries[:, :, query_index : query_index + 1],
                storage.keys,
                storage.values,
                positions,
                offsets,
                scale,
                k_head_stride,
                v_head_stride,
            )
            if output is None:
                return None
            outputs.append(output)
        return mx.concatenate(outputs, axis=2)

    def dense_mask(self):
        batch, query_length, _ = self.selected_blocks.shape
        token_indices = (
            self.selected_blocks[..., None] * self.block_size
            + mx.arange(self.block_size, dtype=mx.int32)[None, None, None]
        ).reshape(batch, query_length, -1)
        selected = mx.put_along_axis(
            mx.zeros((batch, query_length, self.key_length + 1), dtype=mx.bool_),
            mx.minimum(token_indices, self.key_length),
            token_indices < self.key_length,
            axis=-1,
        )[..., : self.key_length]
        tokens = mx.arange(self.key_length, dtype=mx.int32)
        tail_start = self.complete_counts * self.block_size
        tail = (tokens[None, None] >= tail_start[..., None]) & (
            tokens[None, None] < self.query_ends[None, :, None]
        )
        return (selected | tail)[:, None]


__all__ = ["Qwen4ExactSparseSelection", "enabled"]
