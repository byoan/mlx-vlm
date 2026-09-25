import mlx.core as mx
import pytest
from mlx_vlm.models.qwen4_exp.batched_qsa import attend
from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.qwen4_exp.qsa_kernel import qsa_sparse_attention
from mlx_vlm.models.qwen4_exp.batched_verifier import _IndependentTemporalRows
from mlx_vlm.tests.test_qwen4_batched_mtp import (
    test_shared_verifier_matches_independent_cache_transactions as check_transactions,
)


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("width", [2, 4, 8])
def test_gdn_transaction_parity(batch, width, monkeypatch):
    monkeypatch.setenv("MLX_VLM_QWEN4_BATCHED_GDN", "1")
    check_transactions(batch, width)


@pytest.mark.parametrize("batch", [2, 3, 4])
@pytest.mark.parametrize("width", [1, 2])
def test_qsa_separate_strided_buffers(batch, width):
    qs, ks, vs, blocks, ends, refs = [], [], [], [], [], []
    for i in range(batch):
        length = 32768 - i * 117
        q = mx.random.normal((1, 24, width, 256)).astype(mx.bfloat16)
        key = mx.random.normal((1, 2, 33024, 256)).astype(mx.bfloat16)
        value = mx.random.normal((1, 2, 33024, 256)).astype(mx.bfloat16)
        c = KVCache()
        c.keys = key
        c.values = value
        c.offset = length
        k, v = key[:, :, :length], value[:, :, :length]
        b = mx.broadcast_to(
            mx.random.permutation(length // 4 - 1)[:512], (1, width, 512)
        )
        e = mx.arange(length - width + 1, length + 1)[None]
        ref = qsa_sparse_attention(
            q,
            k,
            v,
            b,
            e,
            scale=256**-0.5,
            block_size=4,
            allow_sparse_decode=True,
            cache=c,
        )
        qs.append(q)
        ks.append(k)
        vs.append(v)
        blocks.append(b)
        ends.append(e)
        refs.append(ref)
    actual = attend(qs, ks, vs, blocks, ends, 256**-0.5)
    expected = mx.concatenate(refs)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("batch", [5, 8])
def test_temporal_groups_beyond_four(batch, monkeypatch):
    monkeypatch.setenv("MLX_VLM_QWEN4_BATCHED_GDN", "1")
    monkeypatch.setenv("MLX_VLM_QWEN4_BATCHED_QSA", "1")
    check_transactions(batch, 4)


def test_unsupported_qsa_returns_without_mutation():
    from types import SimpleNamespace
    from mlx_vlm.models.qwen4_exp.batched_qsa import forward
    from mlx_vlm.models.qwen4_exp.batched_verifier import RowCaches

    cache = RowCaches([KVCache(), KVCache()])
    attention = SimpleNamespace(training=False)
    x = mx.zeros((1, 8, 64), dtype=mx.bfloat16)
    assert forward(None, attention, x, cache, None) is None
    assert all(c.offset == 0 for c in cache.rows)
