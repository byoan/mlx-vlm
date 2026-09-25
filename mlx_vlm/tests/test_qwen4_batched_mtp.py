import copy
from dataclasses import replace
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen4_exp.batched_verifier import RowCaches, verify_requests
from mlx_vlm.models.qwen4_exp.language import LanguageModel
from mlx_vlm.speculative.drafters.qwen4_exp_mtp import (
    ModelConfig,
    Qwen4ExpMTPDraftModel,
)
from mlx_vlm.speculative.mtp import _mtp_rounds
from mlx_vlm.speculative.qwen4_batch import rounds
from mlx_vlm.speculative.sampled_mtp import SampledMTPSampler
from mlx_vlm.tests.test_qwen4_mtp_custom import (
    _rollback_language,
    _tiny_text_config,
    _outer_config,
    _assert_cache_equal,
)


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("width", [2, 4, 8])
def test_shared_verifier_matches_independent_cache_transactions(batch, width):
    lm = _rollback_language()
    caches, refs = [], []
    for i in range(batch):
        prompt = mx.array([[2 + j % 11 for j in range(17 + i)]])
        a, b = lm.make_cache(), lm.make_cache()
        lm(prompt, cache=a)
        lm(prompt, cache=b)
        caches.append(a)
        refs.append(b)
    inputs = mx.arange(batch * width).reshape(batch, width) % 11 + 2
    actual, transactions = verify_requests(lm, inputs, caches)
    expected = []
    for i in range(batch):
        h, _, txn = lm.speculative_verify_hidden(inputs[i : i + 1], refs[i])
        expected.append(h)
        retained = 1 + i % width
        txn.commit([retained])
        transactions[i].commit([retained])
    assert mx.array_equal(actual, mx.concatenate(expected)).item()
    for a, b in zip(caches, refs):
        _assert_cache_equal(a, b)


def make_pair():
    config = replace(_tiny_text_config(), hidden_size=64, vocab_size=16416)
    lm = LanguageModel(config, _outer_config())
    lm.set_dtype(mx.bfloat16)
    draft = Qwen4ExpMTPDraftModel(
        ModelConfig(
            text_config=config,
            private_draft_io=True,
            norm_weights_folded=True,
            draft_head_strategy="q3_top32_q8",
        )
    )
    draft.set_dtype(mx.bfloat16)
    draft.draft_lm_head = draft.draft_lm_head.to_quantized(bits=8, group_size=64)
    draft.configure_tokenizer_vocab_size(16391)
    draft.prepare_draft_readout()
    lm.configure_tokenizer_vocab_size(16391)
    return lm, draft


@pytest.mark.parametrize("batch", [2, 4, 9])
def test_sampled_mtp_batch_matches_singletons(batch, monkeypatch):
    monkeypatch.setenv("MLX_VLM_QWEN4_MTP_BATCH_SIZE", str(batch))
    mx.random.seed(80)
    lm, draft = make_pair()
    caches, hidden, first, expected = [], [], [], []
    for i in range(batch):
        prompt = mx.array([[3 + j for j in range(5 + i)]])
        c = lm.make_cache()
        out = lm(prompt, cache=c, return_hidden=True)
        s = SampledMTPSampler(seed=1732)
        s.set_vocabulary(16391)
        bonus = int(
            s.sample_target(out.logits[:, -1], row_ids=[0], positions=[0]).item()
        )
        caches.append(c)
        hidden.append(out.hidden_states[0][:, -1:])
        first.append(bonus)
        ref = copy.deepcopy(c)
        d = copy.copy(draft)
        tokens = [
            t
            for t, _ in _mtp_rounds(
                lm,
                d,
                ref,
                hidden[-1],
                {},
                first_bonus=bonus,
                max_tokens=16,
                sampler=s,
                draft_block_size=4,
            )
        ]
        expected.append(tokens)
    bank = [RowCaches([c[i] for c in caches]) for i in range(len(caches[0]))]
    s = SampledMTPSampler(seed=1732)
    s.set_vocabulary(16391)
    actual = [[] for _ in range(batch)]
    for tokens, _ in rounds(
        lm,
        draft,
        bank,
        mx.concatenate(hidden),
        first_bonus=mx.array(first),
        max_tokens=16,
        sampler=s,
        draft_block_size=4,
        row_ids=[0] * batch,
    ):
        for i, token in enumerate(tokens):
            if token is not None:
                actual[i].append(token)
    assert actual == expected
    assert len(draft.batch_accept_lens) == batch
    assert all(len(tokens) == 15 for tokens in actual)


@pytest.mark.parametrize("rows", [8, 16, 32])
def test_mixed_route_pack_supports_larger_verification_groups(rows):
    from mlx_vlm.models.qwen4_exp.mixed_precision import route_pack

    ids = (mx.arange(rows * 10).reshape(1, rows, 10) * 17 % 19).astype(mx.uint32)
    inverse, sorted_ids, lhs = route_pack(ids)
    order = mx.argsort(ids.reshape(-1).astype(mx.int64) * 1000 + mx.arange(ids.size))
    assert mx.array_equal(sorted_ids, ids.reshape(-1)[order]).item()
    assert mx.array_equal(lhs, order // 10).item()
    assert mx.array_equal(inverse[order], mx.arange(ids.size)).item()


def test_group_verification_aborts_every_request_on_failure():
    lm = _rollback_language()
    caches = []
    for _ in range(2):
        c = lm.make_cache()
        lm(mx.array([[2, 3, 4]]), cache=c)
        caches.append(c)
    refs = copy.deepcopy(caches)
    from mlx_vlm.models.qwen4_exp.batched_verifier import IndependentExactVerifier

    original = IndependentExactVerifier._layer
    calls = 0

    def fail(self, *args, **kwargs):
        nonlocal calls
        result = original(self, *args, **kwargs)
        calls += 1
        if calls == 2:
            raise RuntimeError("injected verifier failure")
        return result

    with patch.object(IndependentExactVerifier, "_layer", fail):
        with pytest.raises(RuntimeError, match="injected"):
            verify_requests(lm, mx.array([[5, 6], [7, 8]]), caches)
    for actual, reference in zip(caches, refs):
        _assert_cache_equal(actual, reference)


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize(
    "scenario", ["ordinary", "limits", "warm", "cancel_prefill", "cancel_decode"]
)
def test_cohort_prefill_and_generation_match_singleton_scheduler(
    batch, scenario, monkeypatch
):
    from types import SimpleNamespace
    from mlx_vlm.generate import BatchGenerator

    mx.random.seed(18)
    lm, draft = make_pair()

    class Stop:
        def __call__(self, token):
            return False

        def add_eos_token_ids(self, tokens):
            pass

    processor = SimpleNamespace(
        get_vocab=lambda: {"last": 16390}, stopping_criteria=Stop()
    )
    prompts = [[3 + i + j % 20 for j in range(33 + i)] for i in range(batch)]

    def run(indices, limit):
        monkeypatch.setenv("MLX_VLM_QWEN4_MTP_BATCH_SIZE", str(limit))
        from mlx_vlm.apc import APCManager

        manager = APCManager(num_blocks=8, block_size=4) if scenario == "warm" else None
        gen = BatchGenerator(
            lm,
            processor,
            max_tokens=12,
            sampler=SampledMTPSampler(seed=1732),
            draft_model=draft,
            draft_kind="mtp",
            draft_block_size=4,
            compute_logprobs=False,
            completion_batch_size=limit,
            prefill_batch_size=limit,
            prefill_step_size=16,
            apc_manager=manager,
        )
        kwargs = [
            {
                "inputs_embeds": lm.model.embed_tokens(mx.array([prompts[i]])),
                "_apc_semantic_hash": 0,
            }
            for i in indices
        ]
        if manager is not None:
            prefix = lm.make_cache()
            lm(mx.array([prompts[0][:16]]), cache=prefix)
            assert gen.apc.store_checkpoint(prompts[0][:16], prefix, extra_hash=0)
        limits = (
            [1 + 3 * i for i in indices]
            if scenario == "limits"
            else [12] * len(indices)
        )
        uids = gen.insert(
            [prompts[i] for i in indices], max_tokens=limits, prompt_kwargs=kwargs
        )
        canceled = False
        results = {uid: [] for uid in uids}
        while gen.has_work:
            _, responses = gen.next()
            if limit > 1 and not canceled:
                if (
                    scenario == "cancel_prefill"
                    and gen._prompt_batch is not None
                    or scenario == "cancel_decode"
                    and responses
                ):
                    assert gen.remove(uids[0])
                    canceled = True
            for r in responses:
                if r.token is not None:
                    results[r.uid].append(r.token)
        if scenario == "limits" and limit > 1:
            cache = gen._generation_batch.prompt_cache[lm.model.fa_idx]
            for slot, uid in enumerate(gen._generation_batch._all_uids):
                original = indices[uids.index(uid)]
                assert (
                    cache.extract(slot).offset
                    == len(prompts[original]) + len(results[uid]) - 1
                )
        if manager is not None and 0 in indices:
            assert manager.stats.exact_hits > 0
        gen.close()
        return [results[uid] for uid in uids]

    expected = [run([i], 1)[0] for i in range(batch)]
    actual = run(list(range(batch)), batch)
    if scenario.startswith("cancel"):
        assert actual[1:] == expected[1:]
        assert len(actual[0]) < len(expected[0])
    else:
        assert actual == expected
