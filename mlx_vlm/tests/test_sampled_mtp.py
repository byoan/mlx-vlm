import contextlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.generate.ar import BatchGenerator, _limit_sampler_vocab
from mlx_vlm.speculative.sampled_mtp import SampledMTPSampler, prepare_mtp_sampler


def verify(sampler, probs, tokens, budget=10):
    logits = mx.log(mx.array([probs]))
    lm = SimpleNamespace(speculative_logits_from_hidden=lambda _: logits)
    return sampler.speculative_accept(
        lm, None, mx.array([tokens], dtype=mx.int32), budget, 0, 1
    )


@pytest.mark.parametrize(
    "budget,expected",
    [(0, (0, [])), (1, (1, [2])), (2, (2, [2, 1])), (3, (2, [2, 1, 0]))],
)
def test_full_accept_and_budget_with_permuted_shortlist(budget, expected):
    s = SampledMTPSampler(top_k=0, top_p=1, seed=7)
    s.set_vocabulary(3)
    s.proposals = [(mx.array([1.0]), mx.array([i]), mx.array(i)) for i in (2, 1)]
    assert verify(s, [[0, 0, 1], [0, 1, 0], [1, 0, 0]], [2, 1], budget) == expected
    assert not s.proposals


def test_rejection_recovers_probability_outside_draft_shortlist():
    s = SampledMTPSampler(top_k=0, top_p=1, seed=7)
    s.set_vocabulary(3)
    s.proposals = [(mx.array([1.0]), mx.array([2]), mx.array(2))]
    assert verify(s, [[1, 0, 0], [0, 1, 0]], [2]) == (0, [0])


def test_retained_probability_alignment_is_checked():
    s = SampledMTPSampler(seed=7)
    s.set_vocabulary(3)
    with pytest.raises(ValueError, match="alignment"):
        verify(s, [[1, 0, 0], [1, 0, 0]], [0])
    s.proposals = [(mx.array([1.0]), mx.array([2]), mx.array(2))]
    with pytest.raises(ValueError, match="differs"):
        verify(s, [[1, 0, 0], [1, 0, 0]], [0])


def test_sampling_law_recovers_target_mass():
    # q concentrates on token 0; rejection must restore p's missing token 1 mass.
    outcomes = []
    s = SampledMTPSampler(top_k=0, top_p=1, seed=42)
    s.set_vocabulary(2)
    lm = SimpleNamespace(
        speculative_logits_from_hidden=lambda _: mx.log(
            mx.array([[[0.25, 0.75], [0.25, 0.75]]])
        )
    )
    for row in range(512):
        s.proposals = [(mx.array([1.0]), mx.array([0]), mx.array(0))]
        _, out = s.speculative_accept(lm, None, mx.array([[0]]), 1, row, 1)
        outcomes.append(out[0])
    assert abs(np.mean(outcomes) - 0.75) < 0.06


def test_reset_and_vocab_wrapper_preserve_sampler_protocol():
    s = SampledMTPSampler(top_k=0, top_p=1, seed=18)
    wrapped = _limit_sampler_vocab(s, 3)
    logits = mx.array([2.0, 1.0, 0.0])
    ids = mx.array([2, 0, 1])
    first = wrapped.sample_draft(logits, ids).item()
    wrapped.reset_draft()
    assert wrapped.sample_draft(logits, ids).item() == first
    assert len(s.proposals) == 1 and s.vocab_size == 3
    assert wrapped(mx.array([[-100.0, -100.0, 1.0, 1e6]])).item() == 2
    assert (
        wrapped.sample_target(
            mx.array([[-100.0, -100.0, 1.0, 1e6]]), row_ids=[0], positions=[5]
        ).item()
        == 2
    )


def test_batch_generator_limits_new_drafter_to_one_request():
    drafter = SimpleNamespace(requires_sampled_residual=True)
    sampler = SimpleNamespace(temperature=1.0, top_k=20, top_p=0.95, seed=1)
    converted = prepare_mtp_sampler(drafter, sampler, False)
    assert isinstance(converted, SampledMTPSampler)
    assert prepare_mtp_sampler(drafter, None, True) is None
    with pytest.raises(ValueError, match="positioned"):
        prepare_mtp_sampler(drafter, lambda x: x, False)
    model = SimpleNamespace()
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"a": 0, "b": 1}, stopping_criteria=Mock()
    )
    with patch(
        "mlx_vlm.generate.ar.wired_limit", return_value=contextlib.nullcontext()
    ):
        gen = BatchGenerator(
            model,
            tokenizer,
            draft_model=drafter,
            draft_kind="mtp",
            sampler=sampler,
            completion_batch_size=8,
            prefill_batch_size=4,
        )
    assert gen.completion_batch_size == gen.prefill_batch_size == 1
    with pytest.raises(ValueError, match="processors"):
        BatchGenerator(
            model,
            tokenizer,
            draft_model=drafter,
            sampler=sampler,
            logits_processors=[lambda x, y: y],
        )
