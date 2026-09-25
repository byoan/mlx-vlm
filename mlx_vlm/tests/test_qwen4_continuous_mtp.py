import copy
from types import SimpleNamespace
import mlx.core as mx
import pytest
from mlx_vlm.generate import BatchGenerator
from mlx_vlm.tests.test_qwen4_batched_mtp import make_pair
from mlx_vlm.speculative.sampled_mtp import SampledMTPSampler
from mlx_vlm.speculative.drafters.qwen4_exp_mtp.readout import top32


class Stop:
    def __call__(self, token):
        return False

    def add_eos_token_ids(self, tokens):
        pass


@pytest.mark.parametrize("batch", [1, 2, 4, 9])
def test_top32_rows(batch):
    scores = mx.random.normal((batch, 16419)).astype(mx.bfloat16)
    scores = scores.at[:, 120:190].add(6)
    actual = top32(scores)
    expected = mx.stack([top32(scores[i]) for i in range(batch)])
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("batch", [2, 4])
def test_readout_retains_selected_q8_scores(batch):
    _, draft = make_pair()
    hidden = mx.random.normal((batch, 1, 64)).astype(mx.bfloat16)
    scores, ids = draft._coarse_readout.logits_batch(hidden)
    head = draft.draft_lm_head
    for i in range(batch):
        expected = mx.quantized_matmul(
            hidden[i],
            head.weight[ids[i]],
            head.scales[ids[i]],
            head.biases[ids[i]],
            transpose=True,
            bits=8,
            group_size=64,
        ).reshape(-1)
        assert mx.array_equal(scores[i], expected).item()
        sampler = SampledMTPSampler(seed=i)
        token = sampler.sample_draft(scores[i], ids[i])
        q, retained_ids, retained_token = sampler.proposals[0]
        assert mx.array_equal(ids[i], retained_ids).item()
        assert token.item() == retained_token.item()
        assert abs(q.sum().item() - 1) < 1e-5


@pytest.mark.parametrize("prefill", ["0", "1"])
@pytest.mark.parametrize("draft_mode", ["off", "exact", "readout", "full"])
@pytest.mark.parametrize("short_limit", [1, 8])
def test_continuous_admission(prefill, draft_mode, short_limit, monkeypatch):
    monkeypatch.setenv("MLX_VLM_QWEN4_BATCHED_PREFILL", prefill)
    mx.random.seed(42)
    lm, draft = make_pair()
    processor = SimpleNamespace(
        get_vocab=lambda: {"last": 16390}, stopping_criteria=Stop()
    )
    prompts = [[3 + i + j % 20 for j in range(35 + i * 7)] for i in range(5)]
    limits = [64, short_limit, short_limit, short_limit, short_limit]

    def build(batch):
        return BatchGenerator(
            lm,
            processor,
            max_tokens=64,
            sampler=SampledMTPSampler(seed=1732),
            draft_model=draft,
            draft_kind="mtp",
            draft_block_size=4,
            compute_logprobs=False,
            completion_batch_size=batch,
            prefill_batch_size=batch,
            prefill_step_size=16,
        )

    def insert(gen, indices):
        return gen.insert(
            [prompts[i] for i in indices],
            max_tokens=[limits[i] for i in indices],
            prompt_kwargs=[
                {
                    "inputs_embeds": lm.model.embed_tokens(mx.array([prompts[i]])),
                    "position_ids": mx.broadcast_to(
                        (mx.arange(len(prompts[i])) + 3 + i * 7)[None, None],
                        (3, 1, len(prompts[i])),
                    ),
                    "rope_deltas": mx.array([[3 + i * 7]], dtype=mx.int32),
                }
                for i in indices
            ],
        )

    monkeypatch.setenv("MLX_VLM_QWEN4_MTP_BATCH_SIZE", "1")
    expected = []
    for i in range(5):
        gen = build(1)
        uids = insert(gen, [i])
        tokens = []
        while gen.has_work:
            _, resp = gen.next()
            tokens.extend(r.token for r in resp if r.token is not None)
        expected.append(tokens)
        gen.close()
    monkeypatch.setenv("MLX_VLM_QWEN4_MTP_BATCH_SIZE", "2")
    monkeypatch.setenv("MLX_VLM_QWEN4_CONTINUOUS_MTP", "1")
    monkeypatch.setenv("MLX_VLM_QWEN4_BATCHED_DRAFT", draft_mode)
    gen = build(2)
    uids = insert(gen, [0])
    out = {uids[0]: []}
    started = {}
    finished = {}
    added = False
    for step in range(1000):
        if not gen.has_work:
            break
        _, resp = gen.next()
        assert len(getattr(gen._generation_batch, "_all_uids", [])) <= 2
        for r in resp:
            if r.token is not None:
                out[r.uid].append(r.token)
                started.setdefault(r.uid, step)
            if r.finish_reason:
                finished[r.uid] = step
        if not added and len(out[uids[0]]) >= 3:
            more = insert(gen, [1, 2, 3, 4])
            uids.extend(more)
            out.update({u: [] for u in more})
            added = True
    else:
        pytest.fail("scheduler did not drain")
    assert [len(out[u]) for u in uids] == limits
    assert started[uids[1]] < finished[uids[0]]
    assert started[uids[2]] < finished[uids[0]]
    if draft_mode in ("off", "exact"):
        assert [out[u] for u in uids] == expected
    gen.close()


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("explicit", [False, True])
def test_joint_prefill_exact(batch, explicit, monkeypatch):
    from mlx_vlm.models.qwen4_exp.batched_prefill import step, supported

    monkeypatch.setenv("MLX_VLM_QWEN4_MTP_BATCH_SIZE", str(batch))
    mx.random.seed(42)
    lm, draft = make_pair()
    processor = SimpleNamespace(
        get_vocab=lambda: {"last": 16390}, stopping_criteria=Stop()
    )
    gen = BatchGenerator(
        lm,
        processor,
        max_tokens=8,
        sampler=SampledMTPSampler(seed=1732),
        draft_model=draft,
        draft_kind="mtp",
        draft_block_size=4,
        compute_logprobs=False,
        completion_batch_size=batch,
        prefill_batch_size=batch,
        prefill_step_size=16,
    )
    ids = [[3 + i + j % 20 for j in range(35 + i * 7)] for i in range(batch)]
    gen.insert(
        ids,
        prompt_kwargs=[
            {
                "inputs_embeds": lm.model.embed_tokens(mx.array([t])),
                **(
                    {
                        "position_ids": mx.broadcast_to(
                            (mx.arange(len(t)) + 7)[None, None], (3, 1, len(t))
                        ),
                        "rope_deltas": mx.full((1, 1), 7, dtype=mx.int32),
                    }
                    if explicit
                    else {}
                ),
            }
            for t in ids
        ],
    )
    from mlx_vlm.speculative.qwen4_batch import CohortPromptBatch

    cohort = CohortPromptBatch(gen, gen._unprocessed_sequences)
    refs = copy.deepcopy(cohort.children)
    assert supported(gen, cohort.children)
    from mlx_vlm.tests.test_qwen4_mtp_custom import _assert_cache_equal

    while all(c.needs_processing() for c in cohort.children):
        for child in refs:
            child.prompt_step()
        step(gen, cohort.children)
        for a, b in zip(cohort.children, refs):
            _assert_cache_equal(
                [c.extract(0) for c in a.prompt_cache],
                [c.extract(0) for c in b.prompt_cache],
            )
            assert a._processed_prompt_columns == b._processed_prompt_columns
    gen.close()


@pytest.mark.parametrize("joint", ["0", "1"])
@pytest.mark.parametrize("cancel", [False, True])
def test_short_prefill_starts_before_long_prefill_finishes(joint, cancel, monkeypatch):
    monkeypatch.setenv("MLX_VLM_QWEN4_MTP_BATCH_SIZE", "2")
    monkeypatch.setenv("MLX_VLM_QWEN4_CONTINUOUS_MTP", "1")
    monkeypatch.setenv("MLX_VLM_QWEN4_BATCHED_PREFILL", joint)
    lm, draft = make_pair()
    processor = SimpleNamespace(
        get_vocab=lambda: {"last": 16390}, stopping_criteria=Stop()
    )
    gen = BatchGenerator(
        lm,
        processor,
        max_tokens=16,
        sampler=SampledMTPSampler(seed=1732),
        draft_model=draft,
        draft_kind="mtp",
        draft_block_size=4,
        compute_logprobs=False,
        completion_batch_size=2,
        prefill_batch_size=2,
        prefill_step_size=16,
    )
    ids = [[3 + j % 20 for j in range(n)] for n in [241, 8]]
    uids = gen.insert(
        ids,
        prompt_kwargs=[
            {"inputs_embeds": lm.model.embed_tokens(mx.array([t]))} for t in ids
        ],
    )
    started = {}
    counts = {u: 0 for u in uids}
    canceled = False
    for tick in range(200):
        if not gen.has_work:
            break
        _, resp = gen.next()
        for row in resp:
            if row.token is not None:
                started.setdefault(row.uid, tick)
                counts[row.uid] += 1
        if cancel and not canceled and uids[1] in started:
            assert uids[0] not in started
            assert gen.remove(uids[0])
            canceled = True
    else:
        pytest.fail("did not drain")
    assert counts[uids[1]] == 16
    if cancel:
        assert counts[uids[0]] == 0
    else:
        assert counts[uids[0]] == 16
        assert started[uids[1]] < started[uids[0]]
    gen.close()


@pytest.mark.parametrize(
    "scenario", ["ordinary", "warm", "cancel_prefill", "cancel_decode"]
)
def test_continuous_existing_scheduler_contracts(scenario, monkeypatch):
    from mlx_vlm.tests.test_qwen4_batched_mtp import (
        test_cohort_prefill_and_generation_match_singleton_scheduler as check,
    )

    monkeypatch.setenv("MLX_VLM_QWEN4_CONTINUOUS_MTP", "1")
    monkeypatch.setenv("MLX_VLM_QWEN4_BATCHED_DRAFT", "exact")
    monkeypatch.setenv("MLX_VLM_QWEN4_BATCHED_PREFILL", "1")
    check(4, scenario, monkeypatch)


def test_large_exact_head_cohort(monkeypatch):
    from mlx_vlm.tests.test_qwen4_batched_mtp import (
        test_sampled_mtp_batch_matches_singletons as check,
    )

    monkeypatch.setenv("MLX_VLM_QWEN4_BATCHED_DRAFT", "exact")
    monkeypatch.setenv("MLX_VLM_QWEN4_BATCHED_GDN", "1")
    monkeypatch.setenv("MLX_VLM_QWEN4_BATCHED_QSA", "1")
    check(9, monkeypatch)
