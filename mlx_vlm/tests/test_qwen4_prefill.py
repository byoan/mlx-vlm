import os
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx_vlm.models.qwen4_exp import exact_sparse_qsa, sparse_prefill
from mlx_vlm.models.qwen4_exp.language import (
    BatchQSAKVCache,
    LanguageModel,
    _create_qwen3_5_attention_mask,
)
from mlx_vlm.models.qwen4_exp.qwen4_exp import Model


@pytest.mark.parametrize("kind", ["zero", "ties", "random"])
def test_prefill_radix_keeps_future_blocks_below_zero_scores(kind):
    if not exact_sparse_qsa._is_m3_ultra():
        pytest.skip("Prefill radix dispatch is specific to M3 Ultra")
    shape = (1, 16, 8192)
    if kind == "zero":
        scores = mx.zeros(shape)
    else:
        scores = mx.random.uniform(shape=shape, key=mx.random.key(49))
        if kind == "ties":
            scores = mx.floor(scores * 8)
    visible = mx.arange(1024, 1040)[None, :, None]
    scores = mx.where(mx.arange(8192)[None, None] < visible, scores, -mx.inf)
    reference = mx.argpartition(scores, kth=-512, axis=-1)[..., -512:]
    with patch.dict(
        os.environ, {"MLX_VLM_QWEN4_PREFILL_RADIX_QSA_TOPK": "1"}, clear=True
    ):
        actual = exact_sparse_qsa.select_blocks(scores, 512)
    assert actual is not None
    assert mx.array_equal(mx.sort(reference, axis=-1), mx.sort(actual, axis=-1)).item()
    assert mx.all(actual < visible).item()


@pytest.mark.parametrize(
    "shape,topk,device,flag",
    [
        ((1, 8, 8192), 512, True, "1"),
        ((1, 16, 4096), 512, True, "1"),
        ((2, 16, 8192), 512, True, "1"),
        ((1, 16, 8192), 256, True, "1"),
        ((1, 16, 8192), 512, False, "1"),
        ((1, 16, 8192), 512, True, "0"),
    ],
)
def test_prefill_radix_falls_back_outside_dispatch(shape, topk, device, flag):
    with (
        patch.dict(
            os.environ, {"MLX_VLM_QWEN4_PREFILL_RADIX_QSA_TOPK": flag}, clear=True
        ),
        patch.object(exact_sparse_qsa, "_is_m3_ultra", return_value=device),
    ):
        assert exact_sparse_qsa.select_blocks(mx.zeros(shape), topk) is None


@pytest.mark.parametrize("extra_mask", [False, True])
@pytest.mark.parametrize("use_fused", [False, True])
def test_sparse_prefill_matches_selected_attention_with_query_and_kv_tails(
    extra_mask, use_fused
):
    if not mx.metal.is_available():
        pytest.skip("Metal prefill attention")
    length, key_length, dim = 129, 4099, 256
    queries = mx.random.normal((1, 24, length, dim), key=mx.random.key(61)).astype(
        mx.bfloat16
    )
    keys = mx.random.normal((1, 2, key_length, dim), key=mx.random.key(62)).astype(
        mx.bfloat16
    )
    values = mx.random.normal((1, 2, key_length, dim), key=mx.random.key(63)).astype(
        mx.bfloat16
    )
    blocks = mx.argsort(
        mx.random.uniform(shape=(1, length, 768), key=mx.random.key(64)), axis=-1
    )[..., :512]
    ends = mx.arange(key_length - length + 1, key_length + 1)
    selection = exact_sparse_qsa.Qwen4ExactSparseSelection(
        blocks, ends[None] // 4, ends, key_length, 4
    )
    mask = None
    reference_mask = selection.dense_mask()
    if extra_mask:
        mask = (
            mx.random.uniform(shape=(1, 1, length, key_length), key=mx.random.key(65))
            > 0.1
        )
        reference_mask = reference_mask & mask
    with patch.object(sparse_prefill, "fused_enabled", return_value=use_fused):
        actual = sparse_prefill.attention(
            queries, keys, values, selection, dim**-0.5, mask
        )
    reference = mx.fast.scaled_dot_product_attention(
        queries.astype(mx.float32),
        keys.astype(mx.float32),
        values.astype(mx.float32),
        scale=dim**-0.5,
        mask=reference_mask,
    )
    baseline = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=dim**-0.5, mask=reference_mask
    )
    mx.eval(reference, baseline, actual)
    assert actual.shape == queries.shape
    assert actual.dtype == queries.dtype
    assert mx.all(mx.isfinite(actual)).item()
    # The original BF16 kernel also differs from FP32 attention. Check the
    # selection/head mapping against that kernel, and separately require that
    # regrouping does not increase its mean squared error against FP32.
    if use_fused:
        # Fused attention keeps FP32 scores and probabilities; check its final
        # BF16 rounding against FP32 instead of matching BF16 intermediates.
        assert mx.allclose(
            actual.astype(mx.float32), reference, rtol=0.004, atol=1e-5
        ).item()
    else:
        assert mx.allclose(actual, baseline, rtol=0.01, atol=0.001).item()
    candidate_error = mx.mean(mx.square(actual.astype(mx.float32) - reference))
    baseline_error = mx.mean(mx.square(baseline.astype(mx.float32) - reference))
    assert (candidate_error <= 1.01 * baseline_error).item()


def test_fused_prefill_returns_zero_for_fully_masked_rows():
    if not mx.metal.is_available():
        pytest.skip("Metal prefill attention")
    length, key_length = 16, 4099
    q = mx.zeros((1, 24, length, 256), dtype=mx.bfloat16)
    k = mx.zeros((1, 2, key_length, 256), dtype=mx.bfloat16)
    v = mx.ones(k.shape, dtype=mx.bfloat16)
    ends = mx.arange(key_length - length + 1, key_length + 1)
    selection = exact_sparse_qsa.Qwen4ExactSparseSelection(
        mx.broadcast_to(mx.arange(512)[None, None], (1, length, 512)),
        ends[None] // 4,
        ends,
        key_length,
        4,
    )
    mask = mx.broadcast_to(mx.arange(length)[:, None] != 0, (length, key_length))
    with patch.object(sparse_prefill, "fused_enabled", return_value=True):
        out = sparse_prefill.attention(q, k, v, selection, 256**-0.5, mask)
    assert mx.array_equal(out[:, :, :1], mx.zeros_like(out[:, :, :1])).item()
    assert mx.array_equal(out[:, :, 1:], mx.ones_like(out[:, :, 1:])).item()


@pytest.mark.parametrize(
    "failure", ["disabled", "training", "padding", "mask", "short", "first_chunk"]
)
def test_sparse_prefill_fallback_guards(failure):
    module = SimpleNamespace(
        training=failure == "training",
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        indexer=SimpleNamespace(compress_ratio=4, block_topk=512),
    )
    cache = SimpleNamespace(offset=0 if failure == "first_chunk" else 32768)
    if failure == "padding":
        cache.left_padding = mx.array([1])
    length = 32768 if failure == "first_chunk" else 8 if failure == "short" else 32
    x = mx.zeros((1, length, 2560), dtype=mx.bfloat16)
    mask = mx.zeros((32, 32800), dtype=mx.float32) if failure == "mask" else "causal"
    with patch.object(sparse_prefill, "enabled", return_value=failure != "disabled"):
        assert not sparse_prefill.supports(module, x, cache, mask)


def test_sparse_prefill_supports_single_unpadded_mtp_batch_cache():
    module = SimpleNamespace(
        training=False,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        indexer=SimpleNamespace(compress_ratio=4, block_topk=512),
    )
    cache = BatchQSAKVCache([0])
    cache.kv_cache._idx = 32768
    cache.offset = mx.array([32768])
    x = mx.zeros((1, 32, 2560), dtype=mx.bfloat16)
    mask = _create_qwen3_5_attention_mask(x, cache)
    with patch.object(sparse_prefill, "enabled", return_value=True):
        assert sparse_prefill.supports(module, x, cache, "causal")
        assert sparse_prefill.supports(module, x, cache, mask)
        cache.left_padding = mx.array([1])
        assert not sparse_prefill.supports(module, x, cache, "causal")


def test_sparse_prefill_has_a_separate_apc_semantic_key():
    from mlx_vlm.apc import semantic_extra_hash

    language = SimpleNamespace(
        apc_key_dependencies=lambda: LanguageModel.apc_key_dependencies(None)
    )
    model = SimpleNamespace(
        language_model=language,
        apc_key_dependencies=lambda: Model.apc_key_dependencies(model),
    )
    with patch(
        "mlx_vlm.models.qwen4_exp.language.sparse_prefill_enabled", return_value=False
    ):
        baseline = semantic_extra_hash(model=language)
        assert baseline == semantic_extra_hash()
        assert semantic_extra_hash(model=model) == baseline
    with patch(
        "mlx_vlm.models.qwen4_exp.language.sparse_prefill_enabled", return_value=True
    ):
        candidate = semantic_extra_hash(model=language)
        assert candidate != baseline
        assert semantic_extra_hash(model=model) == candidate
    with patch(
        "mlx_vlm.models.qwen4_exp.language.fused_sparse_prefill_enabled",
        return_value=True,
    ):
        fused = semantic_extra_hash(model=language)
        assert fused not in (baseline, candidate)
        assert semantic_extra_hash(model=model) == fused
