"""Request-owned MTP caches with shared draft compute and retained proposal q.

Only the approximate drafter changes arithmetic when requests are packed.
Target verification retains its existing exact kernels and rejection sampler.
"""

import os
import mlx.core as mx
from ....models.qwen4_exp.language import _create_qwen4_exp_attention_mask


def mode():
    value = os.environ.get("MLX_VLM_QWEN4_BATCHED_DRAFT", "off")
    if value not in ("off", "exact", "readout", "full"):
        raise ValueError(
            "MLX_VLM_QWEN4_BATCHED_DRAFT must be off, exact, readout or full"
        )
    return value


def local(fn, rows, shared=True):
    if not shared:
        return [fn(x) for x in rows]
    lengths = [x.shape[1] for x in rows]
    out = fn(mx.concatenate(rows, axis=1))
    edges = []
    for n in lengths[:-1]:
        edges.append(n + (edges[-1] if edges else 0))
    if isinstance(out, tuple):
        parts = [mx.split(x, edges, axis=1) for x in out]
        return list(zip(*parts))
    return mx.split(out, edges, axis=1)


def layer_forward(layer, rows, ids, caches, positions, share_hc=True, share_moe=True):
    if "ple" in layer:
        rows = [x + layer.ple(x, t, c, None) for x, t, c in zip(rows, ids, caches)]
    triples = local(layer.attn_hyper_connection, rows, share_hc)
    output = []
    for (mixed, base, weights), c, pos in zip(triples, caches, positions):
        if layer.is_linear:
            branch = layer.linear_attn(mixed, mask=None, cache=c)
        else:
            branch = layer.self_attn(
                mixed,
                mask=_create_qwen4_exp_attention_mask(mixed, c),
                cache=c,
                position_ids=pos,
            )
        output.append(
            base + (branch[..., None, :] * weights[..., None]).reshape(base.shape)
        )
    triples = local(layer.mlp_hyper_connection, output, share_hc)
    branches = local(layer.mlp, [x[0] for x in triples], share_moe)
    return [
        base + (branch[..., None, :] * weights[..., None]).reshape(base.shape)
        for (_, base, weights), branch in zip(triples, branches)
    ]


def draft_forward(drafts, tokens, hidden, share_hc=True, share_moe=True):
    owner = drafts[0]
    embeds = [d._input_embed(t) for d, t in zip(drafts, tokens)]
    lengths = [t.shape[1] for t in tokens]
    edges = []
    for n in lengths[:-1]:
        edges.append(n + (edges[-1] if edges else 0))
    fused = owner.fuse_inputs(
        mx.concatenate(embeds, axis=1), mx.concatenate(hidden, axis=1)
    )
    rows = mx.split(fused, edges, axis=1)
    positions = [d._position_ids(t.shape[1]) for d, t in zip(drafts, tokens)]
    for i, layer in enumerate(owner.layers):
        rows = layer_forward(
            layer,
            rows,
            tokens,
            [d._cache[i] for d in drafts],
            positions,
            share_hc,
            share_moe,
        )
    mixed = local(owner.hyper_connection_mixer, rows, share_hc)
    for d, t in zip(drafts, tokens):
        d._next_position += t.shape[1]
    return mixed, rows


def forward(drafts, tokens, hidden, token_dtype):
    if len(drafts) == 1 or mode() != "full":
        results = [
            d._forward_tokens(t, h, token_dtype)
            for d, t, h in zip(drafts, tokens, hidden)
        ]
        return [x[0] for x in results], [x[1] for x in results]
    return draft_forward(drafts, tokens, hidden)


def sample(states, hiddens):
    owner = states[0].draft
    owner.prepare_draft_readout()
    scores, ids = owner._coarse_readout.logits_batch(
        mx.concatenate(hiddens), exact=mode() == "exact"
    )
    # Retain each row's actual q and draw counter, including the seed for the
    # following round. The acceptance sampler consumes these in proposal order.
    return [
        s.sampler.sample_draft(scores[i], ids[i]).reshape(1, 1)
        for i, s in enumerate(states)
    ]


def draft_blocks(states, width, token_dtype):
    tokens, hidden, proposals = [], [], [[] for _ in states]
    for i, s in enumerate(states):
        d = s.draft
        d._round_appended = 0
        if d._seed_token is not None and d._seed_hidden is not None:
            tokens.append(d._seed_token.astype(token_dtype))
            hidden.append(d._seed_hidden)
            proposals[i].append(tokens[-1])
            d._seed_token = d._seed_hidden = None
        else:
            tokens.append(mx.array([[s.bonus]], dtype=token_dtype))
            hidden.append(s.hidden)
    while True:
        active = [i for i in range(len(states)) if len(proposals[i]) < width - 1]
        if not active:
            break
        selected = [states[i] for i in active]
        logits_hidden, next_hidden = forward(
            [s.draft for s in selected],
            [tokens[i] for i in active],
            [hidden[i] for i in active],
            token_dtype,
        )
        next_tokens = sample(selected, logits_hidden)
        for j, i in enumerate(active):
            states[i].draft._round_appended += 1
            tokens[i], hidden[i] = next_tokens[j], next_hidden[j]
            proposals[i].append(tokens[i])
    for s in states:
        s.draft._draft_round += 1
    return [mx.concatenate(row, axis=1) for row in proposals]


def accept_batch(jobs, token_dtype):
    states, tokens, hidden = [], [], []
    for s, verify_hidden, proposals, accepted, new_tokens in jobs:
        d = s.draft
        retained = min(accepted, d._round_appended)
        trim = d._round_appended - retained
        if trim:
            for c in d._cache:
                c.trim(trim)
            d._next_position -= trim
        token_parts = [proposals[:, j : j + 1] for j in range(retained, accepted)]
        hidden_parts = [verify_hidden[:, j : j + 1] for j in range(retained, accepted)]
        if new_tokens:
            token_parts.append(mx.array([[new_tokens[-1]]], dtype=token_dtype))
            hidden_parts.append(verify_hidden[:, accepted : accepted + 1])
        if token_parts:
            states.append(s)
            tokens.append(mx.concatenate(token_parts, axis=1))
            hidden.append(mx.concatenate(hidden_parts, axis=1))
        d._round_appended = 0
    if states:
        output, pre_hc = forward([s.draft for s in states], tokens, hidden, token_dtype)
        seeds = sample(states, [x[:, -1:] for x in output])
        for s, token, h in zip(states, seeds, pre_hc):
            s.draft._seed_token = token
            s.draft._seed_hidden = h[:, -1:]
