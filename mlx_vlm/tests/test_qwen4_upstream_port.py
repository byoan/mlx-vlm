"""Integration boundaries between custom Qwen4 features and upstream caches."""

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx_vlm.fp8 import make_quantization_config, transform_fp8_weights
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen3_5 import language as qwen35
from mlx_vlm.models.qwen4_exp.language import (
    BatchQSAKVCache,
    QSAKVCache,
    Qwen4ExpBatchInvariantForward,
)
from mlx_vlm.models.qwen4_exp.mixed_precision import Pending, Qwen4MixedVerifier
from mlx_vlm.models.qwen4_exp.qwen4_exp import Model
from mlx_vlm.speculative.cache_state import SpeculativeCacheTransaction
from mlx_vlm.speculative.drafters.qwen4_exp_mtp import (
    ModelConfig,
    Qwen4ExpMTPDraftModel,
)
from mlx_vlm.speculative.mtp import _mtp_rounds, _mtp_rounds_batch
from mlx_vlm.speculative.sampled_mtp import SampledMTPSampler
from mlx_vlm.tests.test_qwen4_mtp_custom import _assert_cache_equal, _rollback_language


@pytest.mark.parametrize("transformed", [False, True])
def test_shared_fp8_loading_preserves_native_experts_and_ple(transformed):
    config = {
        "model_type": "qwen4_exp",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        },
    }
    assert make_quantization_config(config)["mode"] == "mxfp8"
    prefix = "model.language_model.layers.0"
    weights = {}
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"{prefix}.mlp.experts.{expert}.{projection}.weight"
            weights[name] = mx.to_fp8(mx.full((128, 128), expert + 1.0))
            weights[name + "_scale_inv"] = mx.ones((1, 1))
    ple = f"{prefix}.ple.ple_embedding.ngram_embedding"
    weights[ple + ".shard_0.weight"] = mx.to_fp8(mx.ones((4, 32)))
    weights[ple + ".weight_scale"] = mx.array([0.5], dtype=mx.bfloat16)
    if transformed:
        weights, _ = transform_fp8_weights(weights, config)
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                ple_storage=None,
                tie_word_embeddings=False,
                num_hidden_layers=1,
            )
        )
    )
    actual = Model.sanitize(model, weights)
    for projection in ("gate_proj", "up_proj", "down_proj"):
        name = f"language_model.model.layers.0.mlp.switch_mlp.{projection}"
        restored = mx.dequantize(
            actual[name + ".weight"],
            actual[name + ".scales"],
            group_size=32,
            bits=8,
            mode="mxfp8",
        )
        assert restored.shape == (2, 128, 128)
        assert mx.array_equal(restored[:, 0, 0], mx.array([1.0, 2.0])).item()
    name = "language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shards.0"
    restored = mx.dequantize(
        actual[name + ".weight"],
        actual[name + ".scales"],
        group_size=32,
        bits=8,
        mode="mxfp8",
    )
    assert mx.all(restored == 0.5).item()


@pytest.mark.parametrize("batched", [False, True])
def test_preallocated_qsa_memory_and_transaction_abort(batched, monkeypatch):
    monkeypatch.setenv("MLX_VLM_QWEN4_PREALLOCATED_INDEX_CACHE", "1")
    c = BatchQSAKVCache([0]) if batched else QSAKVCache()
    keys = mx.ones((1, 1, 3, 32))
    c.update_and_fetch(keys, keys)
    c.update_indexer(mx.ones((1, 3, 8)), mx.arange(3)[None])
    storage = c.kv_cache if batched else c
    assert c.nbytes == (
        storage.keys.nbytes
        + storage.values.nbytes
        + c._index_keys_buffer.nbytes
        + c._index_position_ids_buffer.nbytes
    )
    profile = c.memory_profile(3)
    assert profile.source_bytes == c.nbytes
    assert profile.footprint(3) >= c.nbytes
    from mlx_vlm.speculative.cache_state import start_speculative_cache

    txn = start_speculative_cache([c], 2)
    c.update_and_fetch(keys[:, :, :2], keys[:, :, :2])
    c.update_indexer(mx.full((1, 2, 8), 99), mx.array([[3, 4]]))
    txn.abort()
    c.update_and_fetch(keys[:, :, :1], keys[:, :, :1])
    c.update_indexer(mx.full((1, 1, 8), 7), mx.array([[3]]))
    assert mx.array_equal(c.index_keys[0, :, 0], mx.array([1.0, 1.0, 1.0, 7.0])).item()
    assert c.index_position_ids.tolist() == [[0, 1, 2, 3]]


@pytest.mark.parametrize("width", [2, 4, 8])
def test_mixed_layer_adapter_uses_transactional_gdn_and_ple(width):
    lm = _rollback_language()
    verifier = Qwen4MixedVerifier()
    base = Qwen4ExpBatchInvariantForward()
    # Exercise the real mixed layer/deferred residual lifecycle with tiny
    # projections; the actual fixed-size kernels have separate operator tests.
    verifier.eligible = lambda x: True
    verifier._hyper_connection = lambda module, x: base._hyper_connection(
        module, x.materialize() if isinstance(x, Pending) else x
    )
    verifier._feed_forward = base._feed_forward
    object.__setattr__(lm, "_mixed_verifier", verifier)
    cache, expected = lm.make_cache(), lm.make_cache()
    prompt = mx.array([[2, 3, 4]])
    lm(prompt, cache=cache)
    lm(prompt, cache=expected)
    block = (mx.arange(width) + 5)[None]
    _, _, txn = lm.speculative_verify_hidden(block, cache)
    assert isinstance(txn, SpeculativeCacheTransaction)
    lm.rollback_speculative_cache(cache, txn, 0, width)
    lm(block[:, :1], cache=expected)
    _assert_cache_equal(cache, expected)
    assert not txn.active


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("greedy", [False, True])
def test_dedicated_draft_sampler_runs_through_upstream_rounds(batched, greedy):
    lm = _rollback_language()
    draft = Qwen4ExpMTPDraftModel(
        ModelConfig(
            text_config=lm.args,
            private_draft_io=True,
            draft_head_strategy="q3_top32_q8",
            block_size=4,
        )
    )
    draft.set_dtype(mx.bfloat16)
    draft.eval()

    def shortlist(self, hidden, sampler, greedy):
        logits = self.draft_lm_head(hidden).reshape(-1)
        if callable(getattr(sampler, "sample_draft", None)):
            return sampler.sample_draft(logits, mx.arange(logits.size)).reshape(1, 1)
        assert greedy
        return mx.argmax(logits).reshape(1, 1)

    object.__setattr__(draft, "_sample_shortlist", MethodType(shortlist, draft))
    sampler = (
        (lambda logits: mx.argmax(logits, axis=-1))
        if greedy
        else SampledMTPSampler(seed=9)
    )
    if not greedy:
        sampler.set_vocabulary(lm.args.vocab_size)
    cache = [
        BatchQSAKVCache([0]) if batched and not isinstance(c, ArraysCache) else c
        for c in lm.make_cache()
    ]
    prompt = mx.array([[2, 3, 4]])
    output = lm(prompt, cache=cache, return_hidden=True)
    bonus = int(sampler(output.logits[:, -1]).item())
    rounds = _mtp_rounds_batch if batched else _mtp_rounds
    result = list(
        rounds(
            lm,
            draft,
            cache,
            output.hidden_states[-1],
            {},
            first_bonus=mx.array([bonus]) if batched else bonus,
            max_tokens=12,
            sampler=sampler,
            draft_block_size=4,
            greedy_sampling=greedy,
            prompt_tokens=prompt,
        )
    )
    assert len(result) == 11
    assert draft.draft_lens
    assert all(not getattr(c, "is_speculating", False) for c in cache)
    assert draft._cache[0].offset < prompt.shape[1] + 12


def test_server_prefill_hooks_do_not_intercept_speculative_linear_calls():
    lm = _rollback_language()
    layer = qwen35.Qwen3_5GatedDeltaNet(lm.args)
    layer.set_dtype(mx.bfloat16)
    layer.eval()
    x = mx.ones((1, 3, lm.args.hidden_size), dtype=mx.bfloat16)
    with patch.object(
        qwen35, "_target_verify_linears", wraps=qwen35._target_verify_linears
    ) as hook:
        layer(x, cache=ArraysCache(size=2))
        assert hook.call_count == 2
        assert all(call.args[-1] is False for call in hook.call_args_list)
    with patch.object(
        qwen35, "_target_verify_linears", side_effect=AssertionError("prefill hook")
    ):
        _, _, txn = lm.speculative_verify_hidden(mx.array([[2, 3]]), lm.make_cache())
        txn.abort()
