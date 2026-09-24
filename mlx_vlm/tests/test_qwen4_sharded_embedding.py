import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.models.qwen4_exp.language import ShardedEmbedding


@pytest.mark.parametrize("shape", [(0,), (128,), (129,), (2, 97), (2, 3, 64)])
@pytest.mark.parametrize("quantized", [False, True])
def test_sharded_gather_matches_independent_shard_lookup(shape, quantized):
    embedding = ShardedEmbedding(257, 64, 7)
    if quantized:
        nn.quantize(embedding, group_size=64, bits=4)
    size = 1
    for dim in shape:
        size *= dim
    # Repeats and every shard boundary, including the uneven final shard.
    values = [0, 36, 37, 73, 74, 110, 111, 147, 148, 184, 185, 220, 221, 256, 0]
    ids = mx.array((values * (size // len(values) + 1))[:size], dtype=mx.int32).reshape(
        shape
    )
    full = mx.concatenate(
        [
            shard(mx.arange(n))
            for shard, n in zip(embedding.shards, embedding.shard_sizes)
        ]
    )
    assert mx.array_equal(embedding(ids), full[ids]).item()


@pytest.mark.parametrize("size", [128, 129])
@pytest.mark.parametrize("invalid", [-1, 257])
def test_sharded_gather_rejects_invalid_ids(size, invalid):
    embedding = ShardedEmbedding(257, 64, 7)
    ids = mx.array([0] * (size - 1) + [invalid])
    with pytest.raises(IndexError):
        embedding(ids)
