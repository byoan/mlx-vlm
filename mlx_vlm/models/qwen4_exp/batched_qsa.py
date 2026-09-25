"""Multi-request QSA reads separate cache buffers without prefix stacking.

Only buffer selection changes from the qualified singleton shader. Preserve its
pairwise query widths, reduction order, and per-request projection shapes. The
16K..64K gate avoids short-context regressions and the distinct 64K+ reduction.
"""

from functools import lru_cache
import mlx.core as mx
from .qsa_kernel import _QSA_SPARSE_ATTENTION_SOURCE, _strided_qsa_device_supported
from .batched_verifier import RowCaches
from .language import QSAKVCache


@lru_cache(maxsize=4)
def kernel(batch):
    source = _QSA_SPARSE_ATTENTION_SOURCE.replace(
        "int key_length = int(k_size[0]);", "int key_length;"
    )
    anchor = "int batch_idx = batch_head_idx / NUM_Q_HEADS;"
    select = []
    for i in range(batch):
        select.append(f"""{'if' if i==0 else 'else if'} (batch_idx == {i}) {{
            keys = keys{i}; values = values{i};
            khs = keys{i}_strides[1]; kts = keys{i}_strides[2]; kds = keys{i}_strides[3];
            vhs = values{i}_strides[1]; vts = values{i}_strides[2]; vds = values{i}_strides[3];
        }}""")
    source = source.replace(
        anchor,
        anchor + """
        const device T* keys = nullptr;
        const device T* values = nullptr;
        int khs=0, kts=0, kds=0, vhs=0, vts=0, vds=0;
    """ + "\n".join(select) + """\nkey_length = int(k_size[batch_idx]);""",
    )
    source = source.replace("batch_idx * keys_strides[0] +", "").replace(
        "batch_idx * values_strides[0] +", ""
    )
    for name, alias in [("keys", "k"), ("values", "v")]:
        for axis, label in [(1, "hs"), (2, "ts"), (3, "ds")]:
            source = source.replace(f"{name}_strides[{axis}]", alias + label)
    return mx.fast.metal_kernel(
        name=f"qwen4_multi_request_qsa_b{batch}",
        input_names=["queries", "block_indices", "query_ends", "scale", "k_size"]
        + [name + str(i) for i in range(batch) for name in ("keys", "values")],
        output_names=["out"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source="constexpr bool STRIDED_KV = true;\n" + source,
        ensure_row_contiguous=False,
    )


def attend(queries, keys, values, blocks, ends, scale):
    batch = len(queries)
    width = queries[0].shape[2]
    q = mx.contiguous(mx.concatenate(queries))
    blocks = mx.contiguous(mx.sort(mx.concatenate(blocks).astype(mx.int32), axis=-1))
    ends = mx.contiguous(mx.concatenate(ends).astype(mx.int32))
    inputs = [
        q,
        blocks,
        ends,
        mx.array([scale], dtype=mx.float32),
        mx.array([k.shape[2] for k in keys], dtype=mx.int32),
    ]
    inputs.extend(value for pair in zip(keys, values) for value in pair)
    return kernel(batch)(
        inputs=inputs,
        template=[
            ("T", q.dtype),
            ("D_SIZE", 256),
            ("Q_LEN", width),
            ("NUM_Q_HEADS", 24),
            ("NUM_KV_HEADS", 2),
            ("GQA_FACTOR", 12),
            ("BLOCK_SIZE", 4),
            ("TOPK_BLOCKS", 512),
            ("SELECTED_LENGTH", 2048),
        ],
        grid=(1024, batch * 24 * width, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


def forward(self, attention, x, cache, positions):
    if not isinstance(cache, RowCaches) or not 2 <= len(cache.rows) <= 4:
        return None
    if (
        mx.default_device() != mx.gpu
        or not _strided_qsa_device_supported()
        or attention.training
    ):
        return None
    if any(type(c) is not QSAKVCache for c in cache.rows):
        return None
    width = x.shape[1] // len(cache.rows)
    if not 2 <= width <= 8 or any(
        c.index_keys is None or c.index_keys.shape[1] != c.offset for c in cache.rows
    ):
        return None
    if (
        x.dtype != mx.bfloat16
        or attention.num_attention_heads != 24
        or attention.num_key_value_heads != 2
        or attention.head_dim != 256
        or attention.indexer.compress_ratio != 4
        or attention.indexer.block_topk != 512
        or any(not 16384 <= c.offset or c.offset + width >= 65536 for c in cache.rows)
    ):
        return None
    self.batched_qsa_calls = getattr(self, "batched_qsa_calls", 0) + 1
    outputs = [[] for _ in cache.rows]
    for start in range(0, width, 2):
        qs, ks, vs, blocks, ends, gates = [], [], [], [], [], []
        n = min(2, width - start)
        for i, c in enumerate(cache.rows):
            lo = i * width + start
            h = x[:, lo : lo + n]
            pos = None if positions is None else positions[..., lo : lo + n]
            projected = self._linear(attention.indexer.index_qk_proj, h)
            selection = attention.indexer.select_from_projected(projected, c, pos)
            q, k, v = self._linears(
                (attention.q_proj, attention.k_proj, attention.v_proj), h
            )
            q, k, v, gate, _ = attention._prepare_projected_qkv(
                q, k, v, c, pos, None, None
            )
            qs.append(q)
            ks.append(k)
            vs.append(v)
            blocks.append(selection.selected_blocks)
            ends.append(selection.query_ends)
            gates.append(gate)
        out = attend(qs, ks, vs, blocks, ends, attention.scale)
        parts = []
        for i, gate in enumerate(gates):
            value = out[i : i + 1].transpose(0, 2, 1, 3).reshape(1, n, -1)
            value = self._linear(attention.o_proj, value * mx.sigmoid(gate))
            outputs[i].append(value)
            parts.append(value)
        mx.async_eval(parts)
    return mx.concatenate([mx.concatenate(parts, axis=1) for parts in outputs], axis=1)
