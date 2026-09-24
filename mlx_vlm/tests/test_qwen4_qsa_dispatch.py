from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.qwen4_exp import qsa_kernel as qsa
from mlx_vlm.models.qwen4_exp.language import QSAKVCache

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="QSA kernels require Metal"
)


def inputs(cache, length, width=2):
    fresh = mx.random.normal((1, 2, length, 256), key=mx.random.key(length)).astype(
        mx.bfloat16
    )
    keys, values = cache.update_and_fetch(fresh, fresh * 0.5)
    query = mx.random.normal((1, 24, width, 256), key=mx.random.key(9)).astype(
        mx.bfloat16
    )
    # Deliberately unsorted, with a sentinel. Every selected block is complete
    # for both queries; the incomplete causal tail is handled by the kernel.
    blocks = mx.broadcast_to(mx.arange(511, -1, -1, dtype=mx.int32), (1, width, 512))
    blocks = mx.where(blocks == 0, -1, blocks)
    ends = mx.arange(cache.offset - width + 1, cache.offset + 1, dtype=mx.int32)[None]
    return query, keys, values, blocks, ends


def assert_dispatch_exact(cache, args, expected):
    query, keys, values, blocks, ends = args
    with patch.object(qsa, "_strided_qsa_device_supported", return_value=True):
        assert qsa._use_strided_qsa(query, keys, values, cache, 4, 512) is expected
        offset = cache.offset
        actual = qsa.dispatch_qsa_attention(
            *args,
            cache=cache,
            scale=256**-0.5,
            block_size=4,
            causal=True,
            mask=None,
            mask_factory=lambda: pytest.fail("unexpected dense fallback"),
            allow_sparse_decode=True,
        )
        with patch.object(qsa, "_use_strided_qsa", return_value=False):
            reference = qsa.qsa_sparse_attention(
                *args, scale=256**-0.5, block_size=4, allow_sparse_decode=True
            )
        mx.eval(actual, reference)
        assert mx.array_equal(actual, reference).item()
        assert cache.offset == offset


@pytest.mark.parametrize("length", [4096, 16383, 16384, 32808])
@pytest.mark.parametrize("padding", [0, 200])
@pytest.mark.parametrize("width", [1, 2])
def test_qsa_dispatch_threshold_layout_and_exactness(length, padding, width):
    cache = QSAKVCache()
    cache.step = length + padding
    args = inputs(cache, length, width)
    assert_dispatch_exact(cache, args, length >= 16384 and padding > 0)


def test_qsa_cache_growth_rejection_and_restore():
    cache = QSAKVCache()
    # Fill capacity exactly, cross an allocation boundary, reject all newly
    # verified tokens, append a shorter accepted prefix, then restore a view.
    args = inputs(cache, 16384)
    assert_dispatch_exact(cache, args, False)
    prefix = cache.state
    args = inputs(cache, 4)
    assert_dispatch_exact(cache, args, True)
    cache.trim(4)
    args = inputs(cache, 2)
    assert_dispatch_exact(cache, args, True)
    cache.state = prefix
    assert cache.offset == 16384
    args = inputs(cache, 3)
    assert_dispatch_exact(cache, args, True)
    # State restoration of a padded view must not invent visible capacity.
    cache.state = cache.state
    assert not qsa._use_strided_qsa(args[0], cache.keys, cache.values, cache, 4, 512)
    args = inputs(cache, 1)
    assert_dispatch_exact(cache, args, True)


def test_qsa_dispatch_falls_back_for_unqualified_inputs():
    cache = QSAKVCache()
    query, keys, values, _, _ = inputs(cache, 16386)
    with patch.object(qsa, "_strided_qsa_device_supported", return_value=False):
        assert not qsa._use_strided_qsa(query, keys, values, cache, 4, 512)
    with patch.object(qsa, "_strided_qsa_device_supported", return_value=True):
        for c, q, block_size, topk in [
            (None, query, 4, 512),
            (cache, query.astype(mx.float16), 4, 512),
            (cache, mx.concatenate([query, query], axis=2), 4, 512),
            (cache, query, 8, 512),
            (cache, query, 4, 256),
        ]:
            assert not qsa._use_strided_qsa(q, keys, values, c, block_size, topk)

        class CustomCache(KVCache):
            def update_and_fetch(self, *args):
                return super().update_and_fetch(*args)

        custom = CustomCache()
        custom.keys, custom.values, custom.offset = (
            cache.keys,
            cache.values,
            cache.offset,
        )
        assert not qsa._use_strided_qsa(query, keys, values, custom, 4, 512)
        cache.offset = mx.array(cache.offset)
        assert not qsa._use_strided_qsa(query, keys, values, cache, 4, 512)


def test_qsa_dispatch_crosses_threshold_after_rejection():
    cache = QSAKVCache()
    assert_dispatch_exact(cache, inputs(cache, 16382), False)
    assert_dispatch_exact(cache, inputs(cache, 2), False)  # full capacity
    assert_dispatch_exact(cache, inputs(cache, 2), True)  # larger allocation
    cache.trim(5)
    assert_dispatch_exact(cache, inputs(cache, 2), False)  # back below cutoff
    assert_dispatch_exact(cache, inputs(cache, 2), True)
