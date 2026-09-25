"""Opt-in fixed-cohort MTP with independent request state and shared verification."""

import copy
import os
from collections import defaultdict
from dataclasses import dataclass

import mlx.core as mx


def batch_limit(model, drafter):
    """Default to one; never route other MTP implementations through this path."""
    limit = int(os.environ.get("MLX_VLM_QWEN4_MTP_BATCH_SIZE", "1"))
    if limit < 1:
        raise ValueError("MLX_VLM_QWEN4_MTP_BATCH_SIZE must be positive")
    if limit == 1 or drafter is None:
        return 1
    from ..models.qwen4_exp.language import LanguageModel
    from .drafters.qwen4_exp_mtp import Qwen4ExpMTPDraftModel

    lm = getattr(model, "language_model", model)
    if (
        type(lm) is not LanguageModel
        or type(drafter) is not Qwen4ExpMTPDraftModel
        or not drafter.config.private_draft_io
        or not drafter.requires_sampled_residual
    ):
        return 1
    return limit


@dataclass
class _Request:
    index: int
    row_id: int
    draft: object
    sampler: object
    cache: list
    hidden: object
    bonus: int
    offset: int
    rope_delta: int = 0
    limit: int = 0
    emitted: int = 1
    finished: bool = False


def rounds(
    model,
    draft_model,
    prompt_cache,
    hidden,
    *,
    first_bonus,
    max_tokens,
    sampler,
    draft_block_size=None,
    token_dtype=mx.int32,
    stop_check=None,
    eos_token_ids=None,
    greedy_sampling=False,
    row_ids=None,
    max_tokens_per_row=None,
):
    from ..models.qwen4_exp.batched_verifier import RowCaches, verify_requests
    from .common import (
        _dflash_block_total,
        _record_speculative_round,
        generation_stream,
    )
    from .mtp import (
        _MTPVerifyResult,
        _mtp_acceptance_walk,
        _mtp_draft_hidden,
        _mtp_draft_kwargs,
        _mtp_next_block_size,
    )

    lm = getattr(model, "language_model", model)
    batch = first_bonus.shape[0]
    if batch > batch_limit(lm, draft_model):
        raise ValueError("Qwen4 MTP batch exceeds the configured request limit")
    if not greedy_sampling and not callable(getattr(sampler, "fork", None)):
        raise ValueError(
            "Batched Qwen4 sampled MTP requires request-local sampler forks"
        )
    # Extract once at the prefill/decode boundary, never copy/merge long prefixes
    # every round. Each row continues to own a normal QSA/temporal cache.
    caches = [[c.extract(i) for c in prompt_cache] for i in range(batch)]
    if any(hasattr(c, "bits") for row in caches for c in row):
        raise ValueError("Batched Qwen4 MTP requires unquantized caches")
    prompt_cache[:] = [
        RowCaches([row[i] for row in caches]) for i in range(len(caches[0]))
    ]
    bonus = first_bonus.tolist()
    row_ids = list(range(batch)) if row_ids is None else list(row_ids)
    if len(row_ids) != batch:
        raise ValueError("Expected one sampling row ID per request")
    deltas = getattr(lm, "_rope_deltas", None)
    deltas = [0] * batch if deltas is None else deltas.reshape(-1).tolist()
    if len(deltas) == 1:
        deltas *= batch
    if len(deltas) != batch:
        raise ValueError("Expected one RoPE delta per request")
    limits = (
        [max_tokens] * batch if max_tokens_per_row is None else list(max_tokens_per_row)
    )
    if len(limits) != batch or any(n < 1 or n > max_tokens for n in limits):
        raise ValueError("Invalid per-request token limits")
    states = []
    for i in range(batch):
        draft = copy.copy(draft_model)
        draft.reset(lm)
        child_sampler = sampler if greedy_sampling else sampler.fork()
        offset = caches[i][lm.model.fa_idx].offset
        draft.set_shared_kv({}, offset, kv_valid_len=offset)
        states.append(
            _Request(
                i,
                row_ids[i],
                draft,
                child_sampler,
                caches[i],
                _mtp_draft_hidden(lm, hidden[i : i + 1, -1:]),
                int(bonus[i]),
                offset,
                int(deltas[i]),
                limit=limits[i],
            )
        )
    draft_model.accept_lens = []
    draft_model.draft_lens = []
    draft_model.batch_accept_lens = [s.draft.accept_lens for s in states]
    draft_model.batch_draft_lens = [s.draft.draft_lens for s in states]
    block_total = _dflash_block_total(draft_model, draft_block_size)
    configured = int(getattr(draft_model.config, "block_size", block_total))
    while True:
        active = []
        for s in states:
            if (
                s.emitted >= s.limit
                or (eos_token_ids and s.bonus in eos_token_ids)
                or (stop_check and stop_check(s.index, s.bonus))
            ):
                s.finished = True
            if not s.finished:
                active.append(s)
        if not active:
            return
        groups = defaultdict(list)
        for s in active:
            width = _mtp_next_block_size(
                s.draft, block_total, configured, s.limit - s.emitted + 1
            )
            if width <= 1:
                s.finished = True
                continue
            with mx.stream(generation_stream):
                proposals = s.draft.draft_block(
                    s.bonus,
                    s.hidden,
                    None,
                    width,
                    s.sampler,
                    token_dtype,
                    **_mtp_draft_kwargs(s.draft, greedy_sampling, s.sampler),
                )
                mx.async_eval(proposals, s.draft.draft_eval_state())
            groups[width].append((s, proposals))
        outputs = {}
        for width, group in groups.items():
            # Scale beyond B=2 without exceeding the Metal route-pack scratch.
            group_size = max(1, 32 // width)
            for start in range(0, len(group), group_size):
                chunk = group[start : start + group_size]
                inputs = mx.concatenate(
                    [
                        mx.concatenate(
                            [mx.array([[s.bonus]], dtype=token_dtype), p], axis=1
                        )
                        for s, p in chunk
                    ],
                    axis=0,
                )
                transactions = []
                try:
                    with mx.stream(generation_stream):
                        verified, transactions = verify_requests(
                            lm,
                            inputs,
                            [s.cache for s, _ in chunk],
                            [s.rope_delta for s, _ in chunk],
                        )
                    for i, (s, proposals) in enumerate(chunk):
                        h = verified[i : i + 1]
                        result = _MTPVerifyResult(h, {}, rollback_state=transactions[i])
                        accepted, tokens = _mtp_acceptance_walk(
                            lm,
                            result,
                            proposals,
                            s.sampler,
                            s.limit - s.emitted,
                            row_id=s.row_id,
                            base_position=s.emitted,
                        )
                        # Commit only the emitted prefix. A terminal token is
                        # the next bonus, so it need not be consumed by target KV.
                        for position, token in enumerate(tokens):
                            if (eos_token_ids and token in eos_token_ids) or (
                                stop_check and stop_check(s.index, token)
                            ):
                                tokens = tokens[: position + 1]
                                break
                        accepted = min(accepted, max(0, len(tokens) - 1))
                        _record_speculative_round(s.draft, accepted, width - 1)
                        _record_speculative_round(draft_model, accepted, width - 1)
                        with mx.stream(generation_stream):
                            s.draft.accept_verified_tokens(
                                h,
                                proposals,
                                accepted,
                                tokens,
                                s.sampler,
                                token_dtype,
                                **_mtp_draft_kwargs(
                                    s.draft, greedy_sampling, s.sampler
                                ),
                            )
                        result.commit(lm, s.cache, accepted, width)
                        s.hidden = _mtp_draft_hidden(lm, h[:, accepted : accepted + 1])
                        s.offset += accepted + 1
                        s.draft.set_shared_kv({}, s.offset, kv_valid_len=s.offset)
                        outputs[s.index] = tokens
                except BaseException:
                    for transaction in transactions:
                        transaction.abort()
                    raise
        count = max(map(len, outputs.values()), default=0)
        if not count:
            return
        for pos in range(count):
            tokens_out = [None] * batch
            for s in active:
                tokens = outputs.get(s.index, [])
                if s.finished or pos >= len(tokens):
                    continue
                token = tokens[pos]
                tokens_out[s.index] = token
                s.bonus = token
                s.emitted += 1
                if (
                    s.emitted >= s.limit
                    or (eos_token_ids and token in eos_token_ids)
                    or (stop_check and stop_check(s.index, token))
                ):
                    s.finished = True
            yield tokens_out, {"round_pos": pos, "round_len": count}


class CohortPromptBatch:
    """Prefill each request with its singleton chunk boundaries and APC path.

    Padding changes Qwen4 prefill reductions. Keep prefill request-local and
    combine only the generation boundary, without copying prefix buffers.
    """

    def __init__(self, owner, sequences):
        self.owner = owner
        self.uids = [s[0] for s in sequences]
        self.children = [
            owner._build_mixed_prompt_batch([s]) or owner._build_cold_prompt_batch([s])
            for s in sequences
        ]
        self.completed = []
        self.rope_deltas = []
        self.index = 0
        self._timed_child = None
        self.total_prompt_tokens = sum(c.total_prompt_tokens for c in self.children)

    def __len__(self):
        return len(self.uids)

    def needs_processing(self):
        return self.index < len(self.children)

    def prompt_step(self):
        child = self.children[self.index]
        self._timed_child = child
        if child.needs_processing():
            return child.prompt_step()
        result = child.generate(
            self.owner.sampler,
            self.owner.tokenizer.stopping_criteria,
            compute_logprobs=False,
            top_logprobs_k=0,
        )
        self.completed.append(result)
        lm = getattr(self.owner.model, "language_model", self.owner.model)
        delta = getattr(lm, "_rope_deltas", None)
        self.rope_deltas.append(
            0 if delta is None else int(delta.reshape(-1)[0].item())
        )
        self.index += 1
        return 0

    def record_prompt_time(self, elapsed):
        if self._timed_child is not None:
            self._timed_child.record_prompt_time(elapsed)

    def prompt_progress(self):
        return [p for child in self.children for p in child.prompt_progress()]

    def remove(self, uid):
        index = self.uids.index(uid)
        child = self.children.pop(index)
        child._release_apc_meta_blocks()
        self.uids.pop(index)
        if index < self.index:
            self.completed.pop(index)
            self.rope_deltas.pop(index)
            self.index -= 1

    def generate(self, sampler, stop_criteria, **kwargs):
        from ..generate.ar import SpeculativeGenerationBatch
        from ..models.qwen4_exp.batched_verifier import RowCaches

        if self.needs_processing():
            raise RuntimeError("Cohort prefill is incomplete")
        rows = self.completed
        lm = getattr(self.owner.model, "language_model", self.owner.model)
        lm._rope_deltas = mx.array(self.rope_deltas)[:, None]
        # Cancellation can leave a singleton: preserve its ordinary cache and
        # generation path instead of handing row banks to the singleton loop.
        if len(rows) == 1:
            return rows[0]
        caches = [
            RowCaches([r.prompt_cache[i].extract(0) for r in rows])
            for i in range(len(rows[0].prompt_cache))
        ]
        return SpeculativeGenerationBatch(
            model=self.owner.model,
            draft_model=self.owner.draft_model,
            draft_kind="mtp",
            uids=[r.uids[0] for r in rows],
            first_tokens=mx.concatenate([r.first_tokens for r in rows]),
            prompt_cache=caches,
            sampler=sampler,
            stop_criteria=stop_criteria,
            max_tokens=[r.max_tokens[0] for r in rows],
            hidden=mx.concatenate([r.hidden[:, -1:] for r in rows]),
            shared_kv_states={},
            prompt_tokens=mx.zeros((len(rows), 0), dtype=mx.int32),
            draft_block_size=self.owner.draft_block_size,
            greedy_sampling=self.owner.greedy_sampling,
        )
