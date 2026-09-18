"""Request-local sampled MTP with retained proposals and residual correction.

Standard speculative acceptance. No Typical/TokenV3 policy is enabled.
"""

import math
import secrets

import mlx.core as mx

from ..sample_utils import apply_top_k, apply_top_p


def distribution(logits, temperature, top_k, top_p):
    scores = logits.astype(mx.float32) / temperature
    if top_k and top_k < scores.shape[-1]:
        scores = apply_top_k(scores, top_k)
    scores = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
    if top_p < 1:
        scores = apply_top_p(scores, top_p)
    return scores - mx.logsumexp(scores, axis=-1, keepdims=True)


def prepare_mtp_sampler(drafter, sampler, greedy):
    if not getattr(drafter, "requires_sampled_residual", False) or greedy:
        return sampler
    if isinstance(sampler, SampledMTPSampler):
        return sampler
    if not all(hasattr(sampler, k) for k in ("temperature", "top_k", "top_p", "seed")):
        raise ValueError(
            "Qwen4 two-stage sampled drafting requires SampledMTPSampler or a positioned temperature/top-k/top-p sampler"
        )
    return SampledMTPSampler(
        temperature=sampler.temperature,
        top_k=sampler.top_k,
        top_p=sampler.top_p,
        seed=sampler.seed,
    )


class SampledMTPSampler:
    """One sequence; target temperature/top-k/top-p, donor draft T1/k20/p.95."""

    def __init__(self, temperature=1.0, top_k=20, top_p=0.95, seed=None):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Positive finite target temperature required")
        if type(top_k) is not int or top_k < 0 or not 0 < top_p <= 1:
            raise ValueError("Invalid target top-k/top-p")
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.seed = secrets.randbits(32) if seed is None else int(seed)
        self.vocab_size = None
        self.reset_draft()

    def set_vocabulary(self, size):
        if int(size) <= 0:
            raise ValueError("Vocabulary must be positive")
        self.vocab_size = int(size)

    def reset_draft(self):
        self.proposals = []
        self._draft_draw = 0

    def distribution_logprobs(self, logits):
        if self.vocab_size is not None:
            logits = logits[..., : self.vocab_size]
        return distribution(logits, self.temperature, self.top_k, self.top_p)

    def __call__(self, logits):
        return self.sample_target(
            logits, row_ids=[0] * logits.shape[0], positions=[0] * logits.shape[0]
        )

    def sample_target(self, logits, *, row_ids, positions):
        if logits.shape[0] != len(row_ids) or len(row_ids) != len(positions):
            raise ValueError("Sampling row/position mismatch")
        scores = self.distribution_logprobs(logits)
        return mx.stack(
            [
                mx.random.categorical(
                    scores[i],
                    key=mx.random.key(
                        (self.seed + 1000003 * int(row) + 9176 * int(pos)) & 0xFFFFFFFF
                    ),
                )
                for i, (row, pos) in enumerate(zip(row_ids, positions))
            ]
        )

    def sample_draft(self, logits, ids):
        q = mx.exp(distribution(logits, 1.0, 20, 0.95))
        # Separate draft randomness from first-token and verification draws.
        key = mx.random.key(
            ((self.seed ^ 0x0DFA5202) + 104729 * self._draft_draw) & 0xFFFFFFFF
        )
        self._draft_draw += 1
        index = mx.random.categorical(mx.log(q), key=key)
        token = ids[index]
        self.proposals.append((q, ids, token))
        return token

    def speculative_accept(self, lm, hidden, drafts, budget, row_id, base_position):
        proposals, self.proposals = self.proposals, []
        count = drafts.size
        if drafts.shape[0] != 1 or len(proposals) != count:
            raise ValueError("Proposal alignment failed")
        if self.vocab_size is None:
            raise ValueError("Set tokenizer vocabulary before verification")
        logits = lm.speculative_logits_from_hidden(hidden)[
            0, :, : self.vocab_size
        ].astype(mx.float32)
        p = mx.exp(self.distribution_logprobs(logits))
        if p.shape[0] != count + 1:
            raise ValueError("Verification rows must include bonus")
        if not count:
            key = mx.random.key(
                (self.seed + 1000003 * row_id + 9176 * base_position) & 0xFFFFFFFF
            )
            _, correction_key = mx.random.split(key)
            token = mx.random.categorical(mx.log(p[0]), key=correction_key)
            return 0, ([int(token.item())] if budget else [])
        rows = []
        for q, ids, token in proposals:
            row = mx.zeros((self.vocab_size,), dtype=mx.float32)
            row = row.at[: q.size].add(q) if ids is None else row.at[ids].add(q)
            rows.append(row)
        qfull = mx.stack(rows)
        tokens = drafts.reshape(-1)
        proposal_tokens = mx.stack([v[2].reshape(()) for v in proposals])
        qx = mx.take_along_axis(qfull, tokens[:, None], axis=-1).squeeze(-1)
        px = mx.take_along_axis(p[:count], tokens[:, None], axis=-1).squeeze(-1)
        ratios = mx.minimum(1, px / mx.maximum(qx, 1e-30))
        corrected = mx.maximum(p[:count] - qfull, 0)
        mass = mx.sum(corrected, axis=-1, keepdims=True)
        corrected = mx.where(mass > 0, corrected / mx.maximum(mass, 1e-30), p[:count])
        coins = []
        samples = []
        for pos in range(count + 1):
            key = mx.random.key(
                (self.seed + 1000003 * row_id + 9176 * (base_position + pos))
                & 0xFFFFFFFF
            )
            coin_key, correction_key = mx.random.split(key)
            if pos < count:
                coins.append(mx.random.uniform(key=coin_key) < ratios[pos])
                samples.append(
                    mx.random.categorical(mx.log(corrected[pos]), key=correction_key)
                )
            else:
                samples.append(
                    mx.random.categorical(mx.log(p[pos]), key=correction_key)
                )
        keeps = mx.stack(coins)
        samples = mx.stack(samples)
        mx.eval(keeps, samples, tokens, proposal_tokens)
        token_list = tokens.tolist()
        if proposal_tokens.tolist() != token_list:
            raise ValueError("Proposal token differs from retained q")
        keep_list = keeps.tolist()
        corrections = samples.tolist()
        out = []
        accepted = 0
        for pos in range(min(count + 1, budget)):
            if pos == count or not keep_list[pos]:
                out.append(corrections[pos])
                break
            accepted += 1
            out.append(token_list[pos])
        return accepted, out
