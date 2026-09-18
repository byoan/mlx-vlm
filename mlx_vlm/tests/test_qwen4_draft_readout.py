import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNorm
from mlx_vlm.speculative.drafters.qwen4_exp_mtp import (
    ModelConfig,
    Qwen4ExpMTPDraftModel,
)
from mlx_vlm.speculative.drafters.qwen4_exp_mtp.readout import (
    FoldedRMSNorm,
    Qwen4DraftReadout,
    top32,
)
from mlx_vlm.tests.test_qwen4_mtp import _tiny_text_config


@pytest.mark.parametrize("size", [16384, 16391])
def test_top32_ties_and_padding(size):
    values = mx.arange(size) % 71
    actual = top32(values.astype(mx.float32))
    expected = np.lexsort((np.arange(size), np.array(values)))[-32:]
    np.testing.assert_array_equal(np.array(actual), expected)


@pytest.mark.parametrize("group", [None, 32])
def test_folded_norm_does_not_add_one(group):
    original = Qwen4ExpRMSNorm(64, group_size=group)
    original.weight = mx.full((64,), 2, dtype=mx.bfloat16)
    folded = FoldedRMSNorm(original)
    x = mx.random.normal((1, 2, 64)).astype(mx.bfloat16)
    shape = x.shape if group is None else (1, 2, 2, 32)
    expected = mx.fast.rms_norm(
        x.reshape(shape), mx.full((shape[-1],), 2, dtype=mx.bfloat16), original.eps
    ).reshape(x.shape)
    np.testing.assert_allclose(
        np.array(folded(x).astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=0.04,
        rtol=0.02,
    )


def test_readout_rescores_q8_and_preserves_head():
    head = nn.Linear(64, 16416, bias=False)
    head.set_dtype(mx.bfloat16)
    head = head.to_quantized(bits=8, group_size=64)
    before = mx.array(head.weight)
    readout = Qwen4DraftReadout(head, 16391)
    x = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)
    logits, ids = readout.logits(x)
    expected = mx.quantized_matmul(
        x.reshape(1, -1),
        head.weight[ids],
        head.scales[ids],
        head.biases[ids],
        transpose=True,
        bits=8,
        group_size=64,
    ).reshape(-1)
    assert mx.array_equal(logits, expected).item()
    assert mx.array_equal(head.weight, before).item()
    assert int(mx.max(ids).item()) < 16391
    with pytest.raises(ValueError):
        readout.logits(mx.zeros((2, 1, 64)))


def test_dedicated_binding_survives_reset_without_target_head_changes():
    config = ModelConfig(
        text_config=_tiny_text_config(), private_draft_io=True, norm_weights_folded=True
    )
    draft = Qwen4ExpMTPDraftModel(config)
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=nn.Embedding(64, 32)),
        lm_head=nn.Linear(32, 64, bias=False),
    )
    original = target.lm_head
    draft.bind(target)
    draft.reset(target)
    assert draft._input_embed is draft.draft_embed_tokens
    assert draft._lm_head_fn is draft.draft_lm_head
    assert target.lm_head is original
    assert isinstance(draft.pre_fc_norm_hidden, FoldedRMSNorm)
    assert draft._compiled_input_fusion is None
    with pytest.raises(ValueError, match="clear draft_head_bits"):
        draft.configure_draft_lm_head(8, mode="mxfp8")
    with pytest.raises(ValueError):
        ModelConfig(draft_head_strategy="q3_top32_q8")
