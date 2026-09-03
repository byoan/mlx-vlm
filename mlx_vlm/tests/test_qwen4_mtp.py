import json
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.qwen4_exp.language import (
    BatchQSAKVCache,
    LanguageModel,
    Qwen4ExpDecoderLayer,
)
from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter, get_mtp_splitter
from mlx_vlm.speculative.drafters.qwen4_exp_mtp import (
    ModelConfig,
    Qwen4ExpMTPDraftModel,
)
from mlx_vlm.speculative.drafters.qwen4_exp_mtp.split import split_qwen4_exp_mtp
from mlx_vlm.speculative.mtp import _mtp_next_block_size, _mtp_rounds, _mtp_rounds_batch


def _tiny_text_config():
    return TextConfig.from_dict(
        {
            "model_type": "qwen4_exp_text",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "shared_expert_intermediate_size": 16,
            "moe_intermediate_size": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 64,
            "num_key_value_heads": 1,
            "max_position_embeddings": 128,
            "hc_count": 2,
            "hc_lowrank": 8,
            "head_dim": 16,
            "layer_types": ["linear_attention", "full_attention"],
            "ple_layer_ids": [],
            "indexer_n_heads": 1,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 16,
            "indexer_budget": 8,
            "indexer_compress_ratio": 4,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [1, 1, 0],
                "rope_theta": 10_000,
                "partial_rotary_factor": 0.25,
            },
            "mtp_num_hidden_layers": 1,
        }
    )


def _outer_config():
    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=59,
    )


def test_qwen4_decoder_layers_expose_normalized_layer_types_for_mtp():
    config = _tiny_text_config()

    assert Qwen4ExpDecoderLayer(config, 0).layer_type == "linear_attention"
    assert Qwen4ExpDecoderLayer(config, 1).layer_type == "qwen_sparse_attention"


def test_qwen4_mtp_fusion_matches_released_equations():
    config = _tiny_text_config()
    drafter = Qwen4ExpMTPDraftModel(ModelConfig(text_config=config))
    drafter.fc_embedding.weight = mx.eye(config.hidden_size)
    drafter.fc_hidden.weight = mx.eye(config.hidden_size)
    embedding_weight = mx.linspace(-0.8, -0.2, config.hidden_size)
    hidden_weight = mx.linspace(-0.6, 0.3, config.hc_count * config.hidden_size)
    drafter.pre_fc_norm_embedding.weight = embedding_weight
    drafter.pre_fc_norm_hidden.weight = hidden_weight
    embedding = mx.arange(1, 33, dtype=mx.float32).reshape(1, 1, 32)
    hidden = mx.arange(1, 65, dtype=mx.float32).reshape(1, 1, 64)

    actual = drafter.fuse_inputs(embedding, hidden)
    expected_embedding = embedding * mx.rsqrt(
        mx.mean(embedding * embedding, axis=-1, keepdims=True) + config.rms_norm_eps
    )
    expected_embedding = expected_embedding * (1 + embedding_weight)
    expected_hidden = hidden * mx.rsqrt(
        mx.mean(hidden * hidden, axis=-1, keepdims=True) + config.rms_norm_eps
    )
    expected_hidden = expected_hidden * (1 + hidden_weight)
    expected_hidden = expected_hidden.reshape(1, 1, config.hc_count, 32)
    expected = (expected_embedding[..., None, :] + expected_hidden).reshape(1, 1, 64)

    assert mx.allclose(actual, expected, atol=2e-5).item()


def test_qwen4_mtp_compiles_single_token_input_fusion_after_bind():
    config = _tiny_text_config()
    drafter = Qwen4ExpMTPDraftModel(ModelConfig(text_config=config))
    drafter.set_dtype(mx.bfloat16)
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            args=config,
            model=SimpleNamespace(embed_tokens=nn.Embedding(64, 32)),
            lm_head=nn.Linear(32, 64, bias=False),
        )
    )
    embedding = mx.random.normal((1, 1, 32)).astype(mx.bfloat16)
    hidden = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)
    expected = drafter._fuse_inputs_eager(embedding, hidden)

    drafter.bind(target)
    actual = drafter.fuse_inputs(embedding, hidden)
    mx.eval(expected, actual)

    assert drafter._compiled_input_fusion is not None
    assert mx.array_equal(actual, expected).item()


def test_qwen4_mtp_keeps_multi_token_input_fusion_eager():
    config = _tiny_text_config()
    drafter = Qwen4ExpMTPDraftModel(ModelConfig(text_config=config))
    drafter.set_dtype(mx.bfloat16)
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            args=config,
            model=SimpleNamespace(embed_tokens=nn.Embedding(64, 32)),
            lm_head=nn.Linear(32, 64, bias=False),
        )
    )
    embedding = mx.random.normal((1, 3, 32)).astype(mx.bfloat16)
    hidden = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    drafter.bind(target)

    with patch.object(
        drafter, "_fuse_inputs_eager", wraps=drafter._fuse_inputs_eager
    ) as eager:
        actual = drafter.fuse_inputs(embedding, hidden)
    expected = drafter._fuse_inputs_eager(embedding, hidden)
    mx.eval(expected, actual)

    eager.assert_called_once()
    assert mx.array_equal(actual, expected).item()


def test_qwen4_mtp_uses_requested_block_size_as_adaptive_ceiling():
    drafter = Qwen4ExpMTPDraftModel(ModelConfig(text_config=_tiny_text_config()))

    assert _mtp_next_block_size(drafter, 4, 2, 32) == 2

    drafter.accept_lens.extend([1] * 8)
    assert _mtp_next_block_size(drafter, 4, 2, 32) == 4

    drafter.accept_lens.extend([0] * 16)
    assert _mtp_next_block_size(drafter, 4, 2, 32) == 2


def test_qwen4_mtp_can_use_private_quantized_draft_head():
    config = _tiny_text_config()
    drafter = Qwen4ExpMTPDraftModel(ModelConfig(text_config=config))
    target_head = nn.Linear(32, 64, bias=False)
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            args=config,
            model=SimpleNamespace(embed_tokens=nn.Embedding(64, 32)),
            lm_head=target_head,
        )
    )

    drafter.configure_draft_lm_head(bits=4)
    drafter.bind(target)
    first_draft_head = drafter._draft_lm_head

    assert isinstance(first_draft_head, nn.QuantizedLinear)
    assert first_draft_head.bits == 4
    assert drafter._lm_head_fn is first_draft_head
    assert target.language_model.lm_head is target_head

    hidden = mx.random.normal((1, 1, 32)).astype(mx.bfloat16)
    expected = mx.argmax(first_draft_head(hidden), axis=-1)
    actual = drafter._sample_hidden(hidden, None, greedy=True)
    mx.eval(expected, actual)
    assert mx.array_equal(actual, expected).item()

    drafter.bind(target)
    assert drafter._draft_lm_head is first_draft_head


@pytest.mark.parametrize("bits, mode", [(4, "affine"), (8, "mxfp8")])
@pytest.mark.parametrize("greedy", [True, False])
def test_qwen4_mtp_can_use_ranked_private_draft_vocabulary(bits, mode, greedy):
    config = _tiny_text_config()
    drafter = Qwen4ExpMTPDraftModel(ModelConfig(text_config=config))
    target_head = nn.Linear(32, 64, bias=False)
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            args=config,
            model=SimpleNamespace(embed_tokens=nn.Embedding(64, 32)),
            lm_head=target_head,
        )
    )
    vocab_ids = [1, 7, 19, 42]

    drafter.configure_draft_lm_head(bits=bits, mode=mode, vocab_ids=vocab_ids)
    drafter.bind(target)

    assert drafter._draft_lm_head.weight.shape[0] == len(vocab_ids)
    hidden = mx.random.normal((1, 1, 32)).astype(mx.bfloat16)
    compact_logits = drafter._draft_lm_head(hidden)
    if greedy:
        expected_local = mx.argmax(compact_logits, axis=-1)
        sampler = None
    else:
        expected_local = mx.array([[2]])
        sampler = lambda _: expected_local
    expected = mx.take(mx.array(vocab_ids), expected_local)
    actual = drafter._sample_hidden(hidden, sampler, greedy=greedy)
    mx.eval(expected, actual)
    assert mx.array_equal(actual, expected).item()


def test_qwen4_mtp_rejects_invalid_ranked_draft_vocabulary():
    drafter = Qwen4ExpMTPDraftModel(ModelConfig(text_config=_tiny_text_config()))

    with pytest.raises(ValueError, match="must not be empty"):
        drafter.configure_draft_lm_head(bits=4, vocab_ids=[])
    with pytest.raises(ValueError, match="unique and sorted"):
        drafter.configure_draft_lm_head(bits=4, vocab_ids=[2, 1])
    with pytest.raises(ValueError, match="non-negative"):
        drafter.configure_draft_lm_head(bits=4, vocab_ids=[-1, 2])


def test_qwen4_mtp_rejects_invalid_draft_head_quantization():
    drafter = Qwen4ExpMTPDraftModel(ModelConfig(text_config=_tiny_text_config()))

    with pytest.raises(ValueError, match="bits must be between 2 and 8"):
        drafter.configure_draft_lm_head(bits=1)
    with pytest.raises(ValueError, match="must divide the model hidden size"):
        drafter.configure_draft_lm_head(bits=4, group_size=24)


def test_qwen4_mtp_draft_block_uses_hyper_connection_hidden():
    config = _tiny_text_config()
    drafter = Qwen4ExpMTPDraftModel(ModelConfig(text_config=config))
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            args=config,
            model=SimpleNamespace(embed_tokens=nn.Embedding(64, 32)),
            lm_head=nn.Linear(32, 64, bias=False),
        )
    )
    drafter.reset(target)
    drafter.set_shared_kv({}, kv_offset=4, position=3, kv_valid_len=4)
    tokens = drafter.draft_block(
        7,
        mx.zeros((1, 1, 64)),
        None,
        2,
        lambda logits: mx.argmax(logits, axis=-1),
        mx.int32,
        greedy=True,
    )
    mx.eval(tokens)

    assert tokens.shape == (1, 1)
    assert drafter._cache[0].offset == 1


@pytest.mark.parametrize("accepted", [0, 1])
def test_qwen4_target_exposes_pre_mixer_hidden_and_restores_rejection_exactly(
    accepted,
):
    config = _tiny_text_config()
    language = LanguageModel(config, _outer_config())
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    verify = mx.array([[4, 5, 6]], dtype=mx.int32)

    speculative_cache = language.make_cache()
    prefill = language(prompt, cache=speculative_cache, return_hidden=True)
    hidden, _, rollback = language.speculative_verify_hidden(verify, speculative_cache)
    with patch.object(LanguageModel, "__call__", side_effect=AssertionError("replay")):
        language.rollback_speculative_cache(
            speculative_cache, rollback, accepted=accepted, block_size=3
        )

    reference_cache = language.make_cache()
    language(prompt, cache=reference_cache)
    for index in range(accepted + 1):
        language(verify[:, index : index + 1], cache=reference_cache)
    probe = mx.array([[7]], dtype=mx.int32)
    speculative_logits = language(probe, cache=speculative_cache).logits
    reference_logits = language(probe, cache=reference_cache).logits
    mx.eval(prefill.hidden_states, hidden, speculative_logits, reference_logits)

    assert prefill.hidden_states[-1].shape == (1, 3, 64)
    assert hidden.shape == (1, 3, 64)
    assert mx.array_equal(speculative_logits, reference_logits).item()


def _assert_cache_equal(actual, expected):
    def compare(a, b):
        if isinstance(a, mx.array):
            assert isinstance(b, mx.array)
            if mx.issubdtype(a.dtype, mx.floating):
                assert a.dtype == b.dtype
            assert mx.array_equal(a, b).item()
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                compare(x, y)
        else:
            assert a == b

    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        compare(a.state, b.state)
        compare(a.meta_state, b.meta_state)
        if isinstance(a, ArraysCache):
            compare(a.left_padding, b.left_padding)
            compare(a.lengths, b.lengths)


def _rollback_language(dtype=mx.bfloat16, head_dim=16):
    mx.random.seed(17)
    config = _tiny_text_config()
    config.linear_key_head_dim = config.linear_value_head_dim = 32
    config.head_dim = head_dim
    config.num_hidden_layers = 3
    config.layer_types = [
        "linear_attention",
        "qwen_sparse_attention",
        "linear_attention",
    ]
    config.ple_layer_ids = [3]
    config.heads_per_ngram = 2
    config.ngram_vocab_size_base = 17
    config.make_ngram_vocab_size_divisible_by = 4
    config.split_ngram_parts = 4
    config.eos_token_id = 1
    language = LanguageModel(config, _outer_config())
    language.set_dtype(dtype)
    language.eval()
    return language


@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_qwen4_mtp_restores_ple_and_gdn_cache_exactly(accepted, dtype, batch_size):
    language = _rollback_language(dtype)
    prompt = (mx.arange(batch_size * 17).reshape(batch_size, 17) % 13 + 1).astype(
        mx.int32
    )
    verify = mx.array([[4, 1, 6, 7], [8, 9, 1, 10]][:batch_size], dtype=mx.int32)

    def make_cache():
        return [
            (
                cache
                if isinstance(cache, ArraysCache)
                else BatchQSAKVCache([0] * batch_size)
            )
            for cache in language.make_cache()
        ]

    actual = make_cache()
    reference = make_cache()
    language(prompt, cache=actual)
    language(prompt, cache=reference)
    _, _, rollback = language.speculative_verify_hidden(verify, actual)
    forward = LanguageModel.__call__
    replay_calls = []

    def count_forward(*args, **kwargs):
        replay_calls.append(args[1].shape[1])
        return forward(*args, **kwargs)

    with patch.object(LanguageModel, "__call__", count_forward):
        language.rollback_speculative_cache(
            actual, rollback, accepted=[accepted] * batch_size, block_size=4
        )
    assert replay_calls == ([] if batch_size == 1 else [1] * (accepted + 1))
    for i in range(accepted + 1):
        language(verify[:, i : i + 1], cache=reference)
    _assert_cache_equal(actual, reference)

    # Repeated decode must consume the restored PLE history, recurrent state,
    # and QSA indexer at the same offsets as ordinary tokenwise generation.
    for token in (11, 12, 13):
        inputs = mx.full((batch_size, 1), token, dtype=mx.int32)
        logits = language(inputs, cache=actual).logits
        expected_logits = language(inputs, cache=reference).logits
        assert mx.array_equal(logits, expected_logits).item()
        _assert_cache_equal(actual, reference)


@pytest.mark.parametrize("bits", [None, 4, 8])
@pytest.mark.parametrize("prefix_length", [0, 1, 17])
def test_qwen4_mtp_repeated_rollback_preserves_rolling_state(bits, prefix_length):
    language = _rollback_language()
    if bits is not None:
        nn.quantize(
            language,
            group_size=32,
            bits=bits,
            class_predicate=lambda _, module: (
                isinstance(module, nn.Linear) and module.weight.shape[-1] % 32 == 0
            ),
        )
    actual = language.make_cache()
    reference = language.make_cache()
    if prefix_length:
        prompt = mx.arange(2, prefix_length + 2, dtype=mx.int32)[None]
        language(prompt, cache=actual)
        language(prompt, cache=reference)

    # Cross both the n-gram history and dilated-convolution window lengths,
    # including EOS in accepted and rejected positions and QSA pool boundaries.
    for size, accepted in ((4, 0), (4, 2), (2, 0), (8, 5), (3, 1)):
        inputs = mx.arange(1, size + 1, dtype=mx.int32)[None]
        _, _, rollback = language.speculative_verify_hidden(inputs, actual)
        with patch.object(
            LanguageModel, "__call__", side_effect=AssertionError("replay")
        ):
            language.rollback_speculative_cache(
                actual, rollback, accepted=mx.array([accepted]), block_size=size
            )
        for i in range(accepted + 1):
            language(inputs[:, i : i + 1], cache=reference)
        _assert_cache_equal(actual, reference)
        probe = mx.array([[11]], dtype=mx.int32)
        logits = language(probe, cache=actual).logits
        expected = language(probe, cache=reference).logits
        assert mx.array_equal(logits, expected).item()


@pytest.mark.parametrize("accepted", [-1, 3, [0, 1], []])
def test_qwen4_mtp_rejects_invalid_acceptance_without_mutating_cache(accepted):
    language = _rollback_language()
    cache = language.make_cache()
    language(mx.array([[2, 3]]), cache=cache)
    _, _, rollback = language.speculative_verify_hidden(mx.array([[4, 5, 6]]), cache)
    before = [list(entry.state) for entry in cache]
    with pytest.raises(ValueError):
        language.rollback_speculative_cache(cache, rollback, accepted, block_size=3)
    after = [entry.state for entry in cache]
    for a, b in zip(before, after):
        for x, y in zip(a, b):
            if isinstance(x, mx.array):
                assert mx.array_equal(x, y).item()


def test_qwen4_mtp_rollback_restores_recurrent_padding_metadata():
    language = _rollback_language()
    actual = language.make_cache()
    reference = language.make_cache()
    prompt = mx.array([[2, 3, 4]], dtype=mx.int32)
    for cache in (actual, reference):
        language(prompt, cache=cache)
        for entry in cache:
            if isinstance(entry, ArraysCache):
                entry.left_padding = mx.array([-3])
                entry.lengths = mx.array([20])
    block = mx.array([[5, 6, 7, 8]], dtype=mx.int32)
    _, _, rollback = language.speculative_verify_hidden(block, actual)
    with patch.object(LanguageModel, "__call__", side_effect=AssertionError("replay")):
        language.rollback_speculative_cache(actual, rollback, 1, block_size=4)
    for i in range(2):
        language(block[:, i : i + 1], cache=reference)
    _assert_cache_equal(actual, reference)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("mrope", [False, True])
def test_qwen4_mtp_rollback_preserves_quantized_qsa_and_positions(bits, mrope):
    language = _rollback_language(head_dim=32)
    prompt = mx.arange(2, 19, dtype=mx.int32)[None]
    positions = mx.arange(prompt.shape[1], dtype=mx.int64)[None]
    if mrope:
        positions = mx.stack([positions, positions + 2, positions + 5])
    language._position_ids = positions
    language._rope_deltas = mx.array([[5 if mrope else 0]], dtype=mx.int64)

    def make_cache():
        cache = language.make_cache()
        language(prompt, cache=cache, position_ids=positions)
        return [
            (
                entry
                if isinstance(entry, ArraysCache)
                else entry.to_quantized(group_size=32, bits=bits)
            )
            for entry in cache
        ]

    actual, reference = make_cache(), make_cache()
    block = mx.array([[3, 1, 5, 6]], dtype=mx.int32)
    _, _, rollback = language.speculative_verify_hidden(block, actual)
    with patch.object(LanguageModel, "__call__", side_effect=AssertionError("replay")):
        language.rollback_speculative_cache(actual, rollback, 1, block_size=4)
    for i in range(2):
        language(block[:, i : i + 1], cache=reference)
    _assert_cache_equal(actual, reference)
    for token in (7, 8, 9):
        inputs = mx.array([[token]], dtype=mx.int32)
        logits = language(inputs, cache=actual).logits
        expected = language(inputs, cache=reference).logits
        assert mx.array_equal(logits, expected).item()
        _assert_cache_equal(actual, reference)


@pytest.mark.parametrize("batched", [False, True])
def test_qwen4_mtp_rounds_match_serial_without_target_replay(batched):
    language = _rollback_language()
    drafter = Qwen4ExpMTPDraftModel(
        ModelConfig(text_config=language.args, block_size=4)
    )
    drafter.set_dtype(mx.bfloat16)
    drafter.eval()
    nn.quantize(
        drafter,
        group_size=32,
        bits=3,
        class_predicate=lambda path, module: (
            isinstance(module, nn.Linear)
            and module.weight.shape[-1] % 32 == 0
            and not path.endswith("mlp.gate")
        ),
    )
    prompt = mx.array([[2, 3, 4, 5]], dtype=mx.int32)
    reference = language.make_cache()
    logits = language(prompt, cache=reference).logits
    expected = []
    for _ in range(16):
        token = int(mx.argmax(logits[:, -1], axis=-1).item())
        expected.append(token)
        logits = language(mx.array([[token]]), cache=reference).logits

    cache = language.make_cache()
    if batched:
        cache = [
            entry if isinstance(entry, ArraysCache) else BatchQSAKVCache([0])
            for entry in cache
        ]
    output = language(prompt, cache=cache, return_hidden=True)
    bonus = int(mx.argmax(output.logits[:, -1], axis=-1).item())
    kwargs = dict(
        first_bonus=mx.array([bonus]) if batched else bonus,
        max_tokens=16,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        draft_block_size=4,
        greedy_sampling=True,
    )
    if not batched:
        kwargs["prompt_tokens"] = prompt
    rounds = _mtp_rounds_batch if batched else _mtp_rounds
    with patch.object(LanguageModel, "__call__", side_effect=AssertionError("replay")):
        emitted = list(
            rounds(language, drafter, cache, output.hidden_states[-1], {}, **kwargs)
        )
    actual = [bonus] + [tokens[0] if batched else tokens for tokens, _ in emitted]
    assert actual == expected
    assert any(
        accepted < drafted
        for accepted, drafted in zip(drafter.accept_lens, drafter.draft_lens)
    )


def test_qwen4_speculative_verifier_matches_tokenwise_hidden_and_logits():
    config = _tiny_text_config()
    language = LanguageModel(config, _outer_config())
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    verify = mx.array([[4, 5, 6]], dtype=mx.int32)

    batched_cache = language.make_cache()
    language(prompt, cache=batched_cache)
    batched_hidden, _, _, batched_logits = language.speculative_verify_logits(
        verify, batched_cache, lambda logits: logits
    )

    tokenwise_cache = language.make_cache()
    language(prompt, cache=tokenwise_cache)
    tokenwise_hidden = []
    tokenwise_logits = []
    for index in range(verify.shape[1]):
        output = language(
            verify[:, index : index + 1],
            cache=tokenwise_cache,
            return_hidden=True,
        )
        tokenwise_hidden.append(output.hidden_states[-1])
        tokenwise_logits.append(output.logits)
    tokenwise_hidden = mx.concatenate(tokenwise_hidden, axis=1)
    tokenwise_logits = mx.concatenate(tokenwise_logits, axis=1)
    mx.eval(batched_hidden, batched_logits, tokenwise_hidden, tokenwise_logits)

    assert mx.allclose(batched_hidden, tokenwise_hidden, rtol=0, atol=1e-6).item()
    assert mx.allclose(batched_logits, tokenwise_logits, rtol=0, atol=1e-6).item()
    assert mx.array_equal(
        mx.argmax(batched_logits, axis=-1),
        mx.argmax(tokenwise_logits, axis=-1),
    ).item()


def test_qwen4_fused_greedy_mixes_captured_hyper_state_before_lm_head(monkeypatch):
    from mlx_vlm.models.qwen4_exp import language as qwen4_language

    config = _tiny_text_config()
    language = LanguageModel(config, _outer_config())
    verifier = qwen4_language._QWEN4_EXACT_SPECULATIVE_VERIFIER
    monkeypatch.setattr(verifier, "can_quantized_head", lambda linear: True)
    monkeypatch.setattr(
        verifier,
        "quantized_argmax",
        lambda linear, hidden, token_mask=None: mx.argmax(linear(hidden), axis=-1),
    )
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
    expected = mx.argmax(language(inputs, cache=language.make_cache()).logits, axis=-1)
    language._position_ids = None
    language._rope_deltas = None
    actual = language.fused_greedy_decode(inputs, cache=language.make_cache())
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


def test_qwen4_mtp_splitter_maps_fused_experts_and_quantizes(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    source.mkdir()
    text_config = _tiny_text_config().to_dict()
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "text_config": text_config})
    )
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {
            "mtp.pre_fc_norm_hidden.weight": mx.zeros((64,)),
            "mtp.fc_hidden.weight": mx.ones((32, 32)),
            "mtp.layers.0.mlp.experts.gate_up_proj": mx.ones((4, 32, 32)),
            "mtp.layers.0.mlp.experts.down_proj": mx.ones((4, 32, 16)),
            "mtp.layers.0.mlp.gate.weight": mx.ones((4, 32)),
        },
    )

    splitter = detect_mtp_splitter(source)
    assert splitter is not None
    assert splitter.output_model_type == "qwen4_exp_mtp"
    assert get_mtp_splitter("qwen4_exp").output_model_type == "qwen4_exp_mtp"

    split_qwen4_exp_mtp(str(source), str(output), q_bits=3, q_group_size=32)
    weights = mx.load(str(output / "model.safetensors"))
    config = json.loads((output / "config.json").read_text())

    assert "layers.0.mlp.switch_mlp.gate_proj.weight" in weights
    assert "layers.0.mlp.switch_mlp.up_proj.weight" in weights
    assert "layers.0.mlp.switch_mlp.down_proj.weight" in weights
    assert "fc_hidden.scales" in weights
    assert "layers.0.mlp.gate.scales" not in weights
    assert config["model_type"] == "qwen4_exp_mtp"
    assert config["block_size"] == 2
    assert config["quantization"] == {
        "group_size": 32,
        "bits": 3,
        "mode": "affine",
    }


def test_qwen4_mtp_splitter_converts_official_fp8_experts(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    source.mkdir()
    text_config = _tiny_text_config().to_dict()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": text_config,
                "quantization_config": {
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "weight_block_size": [128, 128],
                },
            }
        )
    )
    weights = {"mtp.pre_fc_norm_hidden.weight": mx.zeros((64,))}
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            key = f"mtp.layers.0.mlp.experts.{expert}.{projection}.weight"
            weights[key] = mx.to_fp8(mx.ones((128, 128)) * (expert + 1))
            weights[f"{key}_scale_inv"] = mx.ones((1, 1))
    mx.save_safetensors(str(source / "model.safetensors"), weights)

    split_qwen4_exp_mtp(str(source), str(output))

    split_weights = mx.load(str(output / "model.safetensors"))
    config = json.loads((output / "config.json").read_text())
    gate = split_weights["layers.0.mlp.switch_mlp.gate_proj.weight"]
    up = split_weights["layers.0.mlp.switch_mlp.up_proj.weight"]
    down = split_weights["layers.0.mlp.switch_mlp.down_proj.weight"]
    gate_scales = split_weights["layers.0.mlp.switch_mlp.gate_proj.scales"]
    up_scales = split_weights["layers.0.mlp.switch_mlp.up_proj.scales"]
    down_scales = split_weights["layers.0.mlp.switch_mlp.down_proj.scales"]
    mx.eval(gate, up, down, gate_scales, up_scales, down_scales)
    assert gate.shape == (2, 128, 32)
    assert up.shape == (2, 128, 32)
    assert down.shape == (2, 128, 32)
    assert gate.dtype == mx.uint32
    assert up.dtype == mx.uint32
    assert down.dtype == mx.uint32
    for weight, scales in (
        (gate, gate_scales),
        (up, up_scales),
        (down, down_scales),
    ):
        restored = mx.dequantize(weight, scales, group_size=32, bits=8, mode="mxfp8")
        assert restored.shape == (2, 128, 128)
    assert not any(key.endswith("weight_scale_inv") for key in split_weights)
    assert config["quantization"] == {
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
    }
