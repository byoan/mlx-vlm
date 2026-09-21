"""Qwen partial recurrent-state storage across batch sizes and cache commits."""

import mlx.core as mx
import pytest

import mlx_vlm.models.qwen3_5.gated_delta as qwen_gated_delta
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.speculative.cache_state import start_speculative_cache


@pytest.mark.parametrize("B", [1, 2, 3, 4, 5, 6, 7, 8, 16, 32])
@pytest.mark.parametrize("S", [1, 2, 3, 8])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("save_states", [False, True])
def test_qwen_gdn_verify_can_omit_the_live_final_state(B, S, masked, save_states):
    mx.random.seed(27)
    Hk, D, Hv, Dv = 2, 64, 4, 16
    q = mx.random.normal((B, S, Hk, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, S, Hk, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, S, Hv, Dv)).astype(mx.bfloat16)
    a = mx.random.normal((B, S, Hv)).astype(mx.bfloat16)
    b = mx.random.normal((B, S, Hv)).astype(mx.bfloat16)
    A_log = mx.random.normal((Hv,)).astype(mx.bfloat16)
    dt_bias = mx.ones((Hv,), dtype=mx.bfloat16)
    state = mx.zeros((B, Hv, Dv, D), dtype=mx.float32)
    mask = (
        mx.array([[step >= row % S for step in range(S)] for row in range(B)])
        if masked
        else None
    )
    state_steps = S - 1 if save_states else 0

    full = qwen_gated_delta.gated_delta_update_with_states(
        q, k, v, a, b, A_log, dt_bias, state, mask, state_steps=S
    )
    shortened = qwen_gated_delta.gated_delta_update_with_states(
        q, k, v, a, b, A_log, dt_bias, state, mask, state_steps=state_steps
    )
    mx.eval(*full, *shortened)

    assert bool(mx.array_equal(full[0], shortened[0]).item())
    assert bool(mx.array_equal(full[1], shortened[1]).item())
    assert shortened[2].shape == (B, state_steps, Hv, Dv, D)
    assert bool(mx.array_equal(full[2][:, :state_steps], shortened[2]).item())


def _qwen_gdn_batch_inputs(B, S, dtype, masked, head_dims=(2, 64, 4, 16)):
    Hk, D, Hv, Dv = head_dims
    # Match the normalized q/k used by the model and start with nonzero state.
    q = mx.random.normal((B, S, Hk, D))
    k = mx.random.normal((B, S, Hk, D))
    q = (q / mx.linalg.norm(q, axis=-1, keepdims=True)).astype(dtype)
    k = (k / mx.linalg.norm(k, axis=-1, keepdims=True)).astype(dtype)
    v = mx.random.normal((B, S, Hv, Dv)).astype(dtype)
    a = mx.random.normal((B, S, Hv)).astype(dtype)
    b = mx.random.normal((B, S, Hv)).astype(dtype)
    A_log = mx.random.normal((Hv,)).astype(dtype)
    dt_bias = mx.ones((Hv,), dtype=dtype)
    state = mx.random.normal((B, Hv, Dv, D)) * 0.1
    # Include an entirely masked row, left padding, and interleaved valid steps.
    mask = (
        mx.array(
            [
                [
                    row % 3 != 0
                    and (step >= row % S if row % 3 == 1 else step % 2 == 0)
                    for step in range(S)
                ]
                for row in range(B)
            ]
        )
        if masked
        else None
    )
    return (q, k, v, a, b, A_log, dt_bias), state, mask


def _qwen_gdn_cpu_reference(inputs, state, mask):
    q, k, v, a, b, A_log, dt_bias = inputs
    # Share gate values with the Metal path: CPU/GPU BF16 sigmoid/softplus
    # rounding can differ. Only the recurrent update is under test here.
    g, beta = qwen_gated_delta._compute_g_beta(A_log, a, b, dt_bias)
    mx.eval(g, beta)
    with mx.stream(mx.cpu):
        reference = qwen_gated_delta._gated_delta_with_states_ops(
            q, k, v, g, beta, state, mask
        )
        mx.eval(*reference)
    return reference


def _assert_qwen_gdn_reference_close(actual, expected):
    # BF16/FP16 output rounding and different reduction orders are expected;
    # recurrent states remain FP32 and must agree to much tighter tolerances.
    atol = {mx.bfloat16: 2e-3, mx.float16: 2e-4, mx.float32: 2e-6}[actual.dtype]
    assert bool(mx.all(mx.isfinite(actual)).item())
    assert bool(mx.allclose(actual, expected, rtol=1e-4, atol=atol).item())


@pytest.mark.parametrize("B", [1, 2, 3, 6, 8])
@pytest.mark.parametrize("S", [1, 3, 8])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.skipif(
    not mx.metal.is_available() or mx.default_device() != mx.gpu,
    reason="Requires Metal kernels",
)
def test_qwen_gdn_saved_states_match_cpu_and_independent_rows(B, S, dtype, masked):
    mx.random.seed(29)
    inputs, state, mask = _qwen_gdn_batch_inputs(B, S, dtype, masked)
    reference = _qwen_gdn_cpu_reference(inputs, state, mask)
    ordinary = qwen_gated_delta.gated_delta_update(*inputs, state, mask)
    mx.eval(*ordinary)

    for state_steps in sorted({0, 1, S - 1, S}):
        actual = qwen_gated_delta.gated_delta_update_with_states(
            *inputs, state, mask, state_steps=state_steps
        )
        mx.eval(*actual)
        for got, expected in zip(actual[:2], ordinary):
            assert bool(mx.array_equal(got, expected).item())
        for got, expected in zip(
            actual, (*reference[:2], reference[2][:, :state_steps])
        ):
            _assert_qwen_gdn_reference_close(got, expected)

        # Each batched row must be bit-for-bit identical to its singleton call.
        for row in range(B):
            row_inputs = tuple(x[row : row + 1] for x in inputs[:5]) + inputs[5:]
            single = qwen_gated_delta.gated_delta_update_with_states(
                *row_inputs,
                state[row : row + 1],
                None if mask is None else mask[row : row + 1],
                state_steps=state_steps,
            )
            mx.eval(*single)
            for got, expected in zip(actual, single):
                assert bool(mx.array_equal(got[row : row + 1], expected).item())


@pytest.mark.parametrize("B", [1, 2, 6])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.skipif(
    not mx.metal.is_available() or mx.default_device() != mx.gpu,
    reason="Requires Metal kernels",
)
def test_qwen_gdn_partial_states_at_qwen35b_head_dimensions(B, masked):
    mx.random.seed(37)
    inputs, state, mask = _qwen_gdn_batch_inputs(
        B, 3, mx.bfloat16, masked, head_dims=(16, 128, 32, 128)
    )
    reference = _qwen_gdn_cpu_reference(inputs, state, mask)
    actual = qwen_gated_delta.gated_delta_update_with_states(
        *inputs, state, mask, state_steps=2
    )
    mx.eval(*actual)
    for got, expected in zip(actual, (*reference[:2], reference[2][:, :2])):
        _assert_qwen_gdn_reference_close(got, expected)


@pytest.mark.parametrize("B", [1, 2, 3, 6, 8, 16])
@pytest.mark.parametrize("S", [1, 3, 8])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.skipif(
    not mx.metal.is_available() or mx.default_device() != mx.gpu,
    reason="Requires Metal kernels",
)
def test_qwen_gdn_partial_state_rollback_with_shrinking_batch(B, S, masked):
    mx.random.seed(31)
    state = reference_state = None
    previous_batch = None
    for round_index, batch in enumerate(
        sorted({B, max(1, B - 1), max(1, B // 2), 1}, reverse=True)
    ):
        inputs, initial_state, mask = _qwen_gdn_batch_inputs(
            batch, S, mx.bfloat16, masked
        )
        if state is None:
            state = reference_state = initial_state
        else:
            # Keep non-prefix rows in reverse order, as finished rows are removed.
            keep = mx.array(
                list(range(previous_batch - 1, previous_batch - batch - 1, -1))
            )
            state, reference_state = state[keep], reference_state[keep]
        actual = qwen_gated_delta.gated_delta_update_with_states(
            *inputs, state, mask, state_steps=S - 1
        )
        reference = _qwen_gdn_cpu_reference(inputs, reference_state, mask)
        mx.eval(*actual)
        for got, expected in zip(actual, (*reference[:2], reference[2][:, : S - 1])):
            _assert_qwen_gdn_reference_close(got, expected)

        # Exercise the current transaction API at every retained length,
        # including zero (initial state) and S (the separate live state).
        conv_input = mx.random.normal((batch, S + 3, 8)).astype(mx.bfloat16)
        input_state = state
        reference_input_state = reference_state
        for offset in range(S + 1):
            retained = [(row + round_index + offset) % (S + 1) for row in range(batch)]
            cache = ArraysCache(2)
            cache[0], cache[1] = conv_input[:, :3], input_state
            with start_speculative_cache([cache], S) as transaction:
                cache.update_window(0, conv_input, 3)
                output, live_state = qwen_gated_delta.gated_delta_update(
                    *inputs, mask=mask, cache=cache
                )
                mx.eval(output, live_state)
                assert bool(mx.array_equal(output, actual[0]).item())
                assert bool(mx.array_equal(live_state, actual[1]).item())
                transaction.commit(retained)
            state, conv = cache[1], cache[0]
            reference_state = mx.stack(
                [
                    reference[2][row, count - 1]
                    if count
                    else reference_input_state[row]
                    for row, count in enumerate(retained)
                ]
            )
            reference_conv = mx.concatenate(
                [
                    conv_input[row : row + 1, count : count + 3]
                    for row, count in enumerate(retained)
                ]
            )
            mx.eval(state, conv, reference_state, reference_conv)
            _assert_qwen_gdn_reference_close(state, reference_state)
            assert bool(mx.array_equal(conv, reference_conv).item())
            assert not cache.is_speculating and cache.history_capacity == 0
        previous_batch = batch
