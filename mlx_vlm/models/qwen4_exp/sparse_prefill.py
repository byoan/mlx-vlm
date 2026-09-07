"""Opt-in QSA prefill using selected KV rows and matrix attention.

Grouping sibling GQA heads as matrix rows changes floating-point accumulation
relative to dense masked SDPA. It preserves the attention selection and model
weights, but does not promise identical logits or greedy continuations.
"""

import os

import mlx.core as mx

from ..qwen3_5.language import _qwen3_5_left_padding_info
from .exact_sparse_qsa import _is_m3_ultra
from .fused_sparse_prefill import attention as fused_attention


def enabled():
    return (
        os.environ.get("MLX_VLM_QWEN4_SPARSE_PREFILL") == "1"
        and mx.default_device() == mx.gpu
        and _is_m3_ultra()
    )


def fused_enabled():
    return enabled() and os.environ.get("MLX_VLM_QWEN4_FUSED_SPARSE_PREFILL") == "1"


def supports(module, x, cache, mask):
    if not (
        enabled()
        and not module.training
        and x.dtype == mx.bfloat16
        and x.shape[0] == 1
        and x.shape[1] > 8
        and module.num_attention_heads == 24
        and module.num_key_value_heads == 2
        and module.head_dim == 256
        and module.indexer.compress_ratio == 4
        and module.indexer.block_topk == 512
        and cache is not None
        and not hasattr(cache, "bits")
        and (
            mask is None
            or isinstance(mask, str)
            and mask == "causal"
            or isinstance(mask, mx.array)
            and mask.dtype == mx.bool_
            and 1 <= mask.ndim <= 4
        )
    ):
        return False
    # MTP uses a batch cache even for one unpadded request. Its physical
    # length is host-side _idx; offset is a per-row MLX array.
    length = getattr(cache, "_idx", cache.offset)
    padding = _qwen3_5_left_padding_info(cache)
    if isinstance(mask, mx.array):
        shape = (1,) * (4 - mask.ndim) + tuple(mask.shape)
        if (
            shape[:2] != (1, 1)
            or shape[2] not in (1, x.shape[1])
            or not isinstance(length, int)
            or shape[3] < length + x.shape[1]
        ):
            return False
    return (
        isinstance(length, int)
        # With fewer complete blocks than the top-k budget, the selector can
        # also return invisible blocks. Keep the original dense/causal path
        # for that initial chunk so a partial block cannot duplicate the tail.
        and length >= module.indexer.block_topk * module.indexer.compress_ratio
        and length + x.shape[1] >= 32768
        and (padding is None or len(padding[0]) == 1 and padding[1] == 0)
    )


def attention(queries, keys, values, selection, scale, mask=None):
    """Bound temporary KV gathers to 128 query positions per matrix batch."""
    length = queries.shape[2]
    heads = queries.shape[1]
    kv_heads = keys.shape[1]
    dim = queries.shape[-1]
    group = heads // kv_heads
    block_size = selection.block_size
    blocks = mx.sort(selection.selected_blocks[0], axis=-1)
    indices = (
        blocks[..., None] * block_size + mx.arange(block_size)[None, None]
    ).reshape(length, -1)
    tail = (
        selection.complete_counts[0, :, None] * block_size + mx.arange(block_size)[None]
    )
    indices = mx.concatenate([indices, tail], axis=-1)
    valid = indices < selection.query_ends[:, None]
    indices = mx.minimum(indices, keys.shape[2] - 1)
    if isinstance(mask, mx.array):
        mask = mx.broadcast_to(
            mask[..., : keys.shape[2]], (1, 1, length, keys.shape[2])
        )
        valid = valid & mx.take_along_axis(mask[0, 0], indices, axis=-1)
    if fused_enabled():
        return fused_attention(queries, keys, values, blocks, valid, scale)
    selected_length = indices.shape[-1]
    outputs = []
    for start in range(0, length, 128):
        end = min(length, start + 128)
        batch = end - start
        query = (
            queries[:, :, start:end]
            .reshape(1, kv_heads, group, batch, dim)
            .transpose(0, 3, 1, 2, 4)
            .reshape(batch * kv_heads, 1, group, dim)
        )
        key = (
            mx.take(keys[0], indices[start:end], axis=1)
            .transpose(1, 0, 2, 3)
            .reshape(batch * kv_heads, 1, selected_length, dim)
        )
        value = (
            mx.take(values[0], indices[start:end], axis=1)
            .transpose(1, 0, 2, 3)
            .reshape(batch * kv_heads, 1, selected_length, dim)
        )
        mask = mx.broadcast_to(
            valid[start:end, None, None, None],
            (batch, kv_heads, 1, 1, selected_length),
        ).reshape(batch * kv_heads, 1, 1, selected_length)
        output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=scale, mask=mask
        )
        outputs.append(
            output.reshape(batch, kv_heads, group, dim)
            .transpose(1, 2, 0, 3)
            .reshape(1, heads, batch, dim)
        )
    return mx.concatenate(outputs, axis=2)
