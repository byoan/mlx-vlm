"""Batch the bounded recurrent transition, preserving projection shapes."""

import mlx.core as mx
import mlx.nn as nn
from ..qwen3_5.gated_delta import gated_delta_update
from .batched_verifier import RowCaches
from .qsa_kernel import _strided_qsa_device_supported


def forward(self, layer, x, cache):
    if not isinstance(cache, RowCaches) or len(cache.rows) != 4:
        return None
    if (
        mx.default_device() != mx.gpu
        or not _strided_qsa_device_supported()
        or layer.training
    ):
        return None
    length = x.shape[1] // len(cache.rows)
    if not 2 <= length <= 8 or any(
        getattr(c, "history_capacity", 0) != length for c in cache.rows
    ):
        return None
    self.batched_gdn_calls = getattr(self, "batched_gdn_calls", 0) + 1
    values = []
    for i, c in enumerate(cache.rows):
        row = x[:, i * length : (i + 1) * length]
        mixed, z, b, a = self._gated_delta_projections(layer, row)
        conv_state = c[0]
        if conv_state is None:
            conv_state = mx.zeros(
                (1, layer.conv_kernel_size - 1, layer.conv_dim), dtype=x.dtype
            )
        conv = mx.concatenate([conv_state, mixed], axis=1)
        c.update_window(0, conv, layer.conv_kernel_size - 1, lengths=c.lengths)
        conv = nn.silu(layer.conv1d(conv))
        q, k, v = [
            value.reshape(1, length, heads, width)
            for value, heads, width in zip(
                mx.split(conv, [layer.key_dim, 2 * layer.key_dim], -1),
                [layer.num_k_heads, layer.num_k_heads, layer.num_v_heads],
                [layer.head_k_dim, layer.head_k_dim, layer.head_v_dim],
            )
        ]
        q, k = self._normalize_gated_delta_qk(layer, q, k)
        state = c[1]
        if state is None:
            state = mx.zeros(
                (1, layer.num_v_heads, layer.head_v_dim, layer.head_k_dim),
                dtype=mx.float32,
            )
        values.append(
            (q, k, v, a, b, state, z.reshape(1, length, -1, layer.head_v_dim))
        )
    q, k, v, a, b, state = [
        mx.concatenate([row[i] for row in values]) for i in range(6)
    ]
    y, final, history = gated_delta_update(
        q,
        k,
        v,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        state=state,
        use_kernel=not layer.training,
        state_steps=length - 1,
    )
    result = []
    helpers = self._helpers()
    for i, c in enumerate(cache.rows):
        out, _ = c.update_recurrent(
            1,
            length,
            lambda initial, steps, i=i: (
                y[i : i + 1],
                final[i : i + 1],
                history[i : i + 1],
            ),
        )
        if hasattr(c, "advance"):
            c.advance(length)
            helpers._qwen3_5_advance_left_padding_info(c, length)
            helpers._qwen3_5_advance_lengths_info(c, length)
        out = layer.norm(out, values[i][-1])
        result.append(self._linear(layer.out_proj, out.reshape(1, length, -1)))
    return mx.concatenate(result, axis=1)
