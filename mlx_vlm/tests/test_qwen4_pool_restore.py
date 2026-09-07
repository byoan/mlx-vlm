import os
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache, LanguageModel
from mlx_vlm.tests.test_qwen4_mtp import _outer_config, _tiny_text_config


@pytest.mark.parametrize("batch_cache", [False, True])
@pytest.mark.parametrize("raw_preallocated", ["0", "1"])
def test_pool_reuse_rollback_and_state_replacement(batch_cache, raw_preallocated):
    language = LanguageModel(_tiny_text_config(), _outer_config())

    def caches():
        return [
            (
                BatchQSAKVCache([0])
                if batch_cache and not isinstance(c, ArraysCache)
                else c
            )
            for c in language.make_cache()
        ]

    control, candidate = caches(), caches()
    for start, length in ((0, 16), (16, 4), (20, 1), (21, 7)):
        tokens = (mx.arange(start, start + length) % 60)[None]
        outputs = []
        for enabled, cache in (("0", control), ("1", candidate)):
            with patch.dict(
                os.environ,
                {
                    "MLX_VLM_QWEN4_PREALLOCATED_QSA_POOL": enabled,
                    "MLX_VLM_QWEN4_PREALLOCATED_INDEX_CACHE": raw_preallocated,
                },
            ):
                out = language(tokens, cache=cache).logits
                mx.eval(out, [c.state for c in cache])
                outputs.append(out)
        assert mx.array_equal(*outputs).item()
        a, b = control[1], candidate[1]
        assert mx.array_equal(a.index_block_keys, b.index_block_keys).item()
        assert b.index_block_keys.shape[2] == (start + length) // 4
        assert b._index_block_keys_buffer.shape[2] == 256
    a, b = control[1], candidate[1]
    buffer = b._index_block_keys_buffer
    for cache in (a, b):
        LanguageModel._trim_speculative_attention_cache(cache, 7)
    assert b._index_block_keys_buffer is buffer
    assert b.index_block_keys.shape[2] == 5
    indexer = language.layers[1].self_attn.indexer
    x = mx.ones((1, 7, language.args.hidden_size))
    for enabled, cache in (("0", a), ("1", b)):
        with patch.dict(os.environ, {"MLX_VLM_QWEN4_PREALLOCATED_QSA_POOL": enabled}):
            sel = indexer.select(x, cache, mx.arange(21, 28)[None])
            mx.eval(sel.selected_blocks, cache.index_block_keys)
    assert mx.array_equal(a.index_block_keys, b.index_block_keys).item()
    state = b.state
    b.state = state
    assert b._index_block_keys_buffer is None
    assert mx.array_equal(b.index_block_keys, a.index_block_keys).item()
    b.clear_index_blocks()
    assert b.index_block_keys is None and b._index_block_keys_buffer is None


@pytest.mark.parametrize("batch_cache", [False, True])
def test_pool_capacity_growth_and_row_filter(batch_cache):
    language = LanguageModel(_tiny_text_config(), _outer_config())
    cache = BatchQSAKVCache([0]) if batch_cache else language.make_cache()[1]
    indexer = language.layers[1].self_attn.indexer
    with patch.dict(os.environ, {"MLX_VLM_QWEN4_PREALLOCATED_QSA_POOL": "1"}):
        capacities = []
        for start, length in ((0, 1020), (1020, 8), (1028, 4)):
            x = mx.ones((1, length, language.args.hidden_size))
            selection = indexer.select(x, cache, mx.arange(start, start + length)[None])
            # The indexer is isolated here: keep the attention offset in step.
            kv = mx.zeros((1, 1, length, 16))
            cache.update_and_fetch(kv, kv)
            mx.eval(selection.selected_blocks, cache.index_block_keys)
            assert cache.index_block_keys.shape[2] == (start + length) // 4
            capacity = cache._index_block_keys_buffer.shape[2]
            logical = cache.index_block_keys.shape[2]
            assert logical <= capacity < logical + 256
            capacities.append(capacity)
        assert capacities[1] > capacities[0]
        assert capacities[2] == capacities[1]
        cache.filter(mx.array([0]))
        assert cache._index_block_keys_buffer is None
