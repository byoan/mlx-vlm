"""Joint prefill submission with singleton-shaped request-local operations.

Packing the quantized HC/MoE matrices changes target numerics. This path keeps
their original shapes and combines scheduling rather than changing arithmetic.
"""

import mlx.core as mx
from ..base import LanguageModelOutput
from .language import (
    LanguageModel,
    _create_qwen4_exp_attention_mask,
    _create_qwen3_5_ssm_mask,
)


def supported(owner, children):
    if type(owner.model) is not LanguageModel or len(children) < 2:
        return False
    for child in children:
        if len(child.uids) != 1 or any(child._left_padding_per_row):
            return False
        kw = child._prompt_kwargs
        # Multimodal position construction remains on the established wrapper.
        if any(
            kw.get(k) is not None
            for k in (
                "pixel_values",
                "image_grid_thw",
                "video_grid_thw",
                "capture_layer_ids",
            )
        ):
            return False
        if any(hasattr(c, "bits") for c in child.prompt_cache):
            return False
    return True


def step(owner, children):
    jobs = [(c, c.prepare_prompt_step()) for c in children]
    if any(not prepared for _, prepared in jobs):
        return None
    lm = owner.model
    ids, rows, positions, fa_masks, ssm_masks = [], [], [], [], []
    for child, (n, kw) in jobs:
        tokens = child._input_ids[:, :n]
        h = child._inputs_embeds[:, :n]
        cache = child.prompt_cache[lm.model.fa_idx]
        offset = cache._idx if hasattr(cache, "_idx") else cache.offset
        position = kw.get("position_ids")
        if position is not None:
            if position.shape[-1] > n:
                position = position[..., offset : offset + n]
        else:
            position = (mx.arange(n, dtype=mx.int64) + offset)[None]
            delta = kw.get("rope_deltas")
            if delta is not None:
                position = mx.broadcast_to((position + delta)[None], (3, 1, n))
        positions.append(position)
        ids.append(tokens)
        fa_masks.append(_create_qwen4_exp_attention_mask(h, cache))
        ssm_masks.append(
            _create_qwen3_5_ssm_mask(h, child.prompt_cache[lm.model.ssm_idx])
        )
        rows.append(mx.tile(h, (1, 1, lm.args.hc_count)))
    for index, layer in enumerate(lm.model.layers):
        rows = [
            layer(
                h,
                t,
                mask=sm if layer.is_linear else fm,
                cache=child.prompt_cache[index],
                position_ids=pos,
            )
            for h, t, pos, fm, sm, (child, _) in zip(
                rows, ids, positions, fa_masks, ssm_masks, jobs
            )
        ]
        mx.async_eval(rows)
    total = 0
    for h, (child, (n, _)) in zip(rows, jobs):
        out = lm.model.hyper_connection_mixer(h)
        logits = (
            lm.model.embed_tokens.as_linear(out)
            if lm.args.tie_word_embeddings
            else lm.lm_head(out)
        )
        total += child.finish_prompt_step(n, LanguageModelOutput(logits=logits))
    # Subsequent text-only final-token calls use their own cache offset.
    lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int64)
    lm._position_ids = None
    owner._qwen4_joint_prefill_steps = (
        getattr(owner, "_qwen4_joint_prefill_steps", 0) + 1
    )
    mx.clear_cache()
    return total
