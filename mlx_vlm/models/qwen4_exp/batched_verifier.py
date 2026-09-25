"""Shared token-local verification with independent, unpadded request caches.

Only token-local operations see the flattened batch. Attention and temporal
operations retain the qualified singleton kernels and cache transactions.
"""

import mlx.core as mx

from .language import Qwen4ExpBatchInvariantForward, _create_qwen3_5_attention_mask
from .mixed_precision import Pending, Qwen4MixedVerifier


class RowCaches:
    """Request-owned cache rows; extraction never pads or merges their buffers."""

    def __init__(self, rows):
        self.rows = list(rows)

    @property
    def state(self):
        return [c.state for c in self.rows]

    def extract(self, index):
        return self.rows[index]

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.rows)


class _IndependentTemporalRows:
    def _rows(self, x, cache, fn):
        if not isinstance(cache, RowCaches):
            return fn(x, cache, 0, x.shape[1])
        width = x.shape[1] // len(cache.rows)
        return mx.concatenate(
            [
                fn(x[:, i * width : (i + 1) * width], c, i * width, width)
                for i, c in enumerate(cache.rows)
            ],
            axis=1,
        )

    def _ple(self, module, x, ids, cache, mask):
        parent = super()._ple
        return self._rows(
            x,
            cache,
            lambda row, c, start, width: parent(
                module, row, ids[:, start : start + width], c, None
            ),
        )

    def _gated_delta(self, layer, x, mask, cache):
        parent = super()._gated_delta
        return self._rows(
            x, cache, lambda row, c, start, width: parent(layer, row, None, c)
        )

    def _qsa_attention(self, attention, x, cache, positions, mask):
        parent = super()._qsa_attention
        if not isinstance(cache, RowCaches):
            return parent(attention, x, cache, positions, mask)
        return self._rows(
            x,
            cache,
            lambda row, c, start, width: parent(
                attention,
                row,
                c,
                None if positions is None else positions[..., start : start + width],
                _create_qwen3_5_attention_mask(row, c),
            ),
        )

    def _model(self, model, inputs, cache, inputs_embeds, position_ids):
        hidden = model.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        hidden = mx.tile(hidden, (1, 1, model.args.hc_count))
        for layer, c in zip(model.layers, cache):
            hidden = self._layer(layer, hidden, inputs, None, c, position_ids)
            if isinstance(hidden, Pending):
                mx.async_eval(hidden.x, hidden.branch, hidden.injection)
            else:
                mx.async_eval(hidden)
        return hidden.materialize() if isinstance(hidden, Pending) else hidden


class IndependentExactVerifier(_IndependentTemporalRows, Qwen4ExpBatchInvariantForward):
    pass


class IndependentMixedVerifier(_IndependentTemporalRows, Qwen4MixedVerifier):
    @staticmethod
    def eligible(x):
        # Route packing has 320 slots (10 experts per token). Larger request
        # cohorts are split into groups; there is no two-request ceiling.
        return (
            x.ndim == 3
            and x.shape[0] == 1
            and 2 <= x.shape[1] <= 32
            and x.dtype == mx.bfloat16
        )


def verify_requests(lm, inputs, caches, rope_deltas=None):
    """Return hidden states and independent transactions for equal-width rows."""
    from ...speculative.cache_state import start_speculative_cache

    batch, width = inputs.shape
    if not 2 <= width <= 8 or batch * width > 32 or len(caches) != batch:
        raise ValueError(
            "Group verification requires width 2..8 and at most 32 token rows"
        )
    transactions = []
    try:
        for row in caches:
            mx.async_eval([c.state for c in row])
            transactions.append(start_speculative_cache(row, width))
        offsets = [row[lm.model.fa_idx].offset for row in caches]
        positions = mx.array(offsets, dtype=mx.int64)[:, None] + mx.arange(width)[None]
        if rope_deltas is not None:
            positions += mx.array(rope_deltas, dtype=mx.int64)[:, None]
        positions = positions.reshape(1, -1)
        if lm._position_ids is not None and lm._position_ids.ndim == 3:
            positions = mx.broadcast_to(positions[None], (3, 1, batch * width))
        mixed = getattr(lm, "_mixed_verifier", None) is not None
        name = "_independent_mixed_verifier" if mixed else "_independent_exact_verifier"
        forward = getattr(lm, name, None)
        if forward is None:
            forward = (
                IndependentMixedVerifier() if mixed else IndependentExactVerifier()
            )
            object.__setattr__(lm, name, forward)
        layer_caches = [
            RowCaches([row[i] for row in caches]) for i in range(len(caches[0]))
        ]
        result = forward(
            lm,
            inputs.reshape(1, -1),
            cache=layer_caches,
            position_ids=positions,
            skip_logits=True,
        )
        return result.hidden_states[-1].reshape(batch, width, -1), transactions
    except BaseException:
        for transaction in transactions:
            transaction.abort()
        raise
