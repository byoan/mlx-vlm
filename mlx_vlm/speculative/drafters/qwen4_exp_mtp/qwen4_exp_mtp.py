from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from ....models.qwen3_5.language import _create_qwen3_5_attention_mask
from ....models.qwen4_exp.language import (
    _QWEN4_EXACT_SPECULATIVE_VERIFIER,
    QSAKVCache,
    Qwen4ExpDecoderLayer,
    Qwen4ExpGatedResidual,
    Qwen4ExpRMSNorm,
)
from ..deepseek_v4_mtp.deepseek_v4_mtp import DeepseekV4MTPDraftModel
from .config import Qwen4ExpMTPConfig


class Qwen4ExpMTPDraftModel(DeepseekV4MTPDraftModel):
    """Standalone runtime for Qwen4's native hyper-connection MTP head.

    The draft lifecycle is shared with the DeepSeek-V4 hyper-connection head,
    while input fusion and the decoder block follow Qwen4's released tensors.
    """

    supports_greedy_draft_argmax = True
    # A caller-provided block size is an adaptive ceiling. Longer
    # autoregressive tails are useful only after the native one-token prefix
    # has demonstrated enough acceptance to amortize them.
    prefer_requested_block_size = False
    requires_uniform_batch_acceptance = True

    def __init__(self, config: Qwen4ExpMTPConfig):
        nn.Module.__init__(self)
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("Qwen4ExpMTPConfig.text_config must be set")

        self.args = text_config
        hidden_size = text_config.hidden_size
        hc_hidden_size = text_config.hc_count * hidden_size
        self.pre_fc_norm_embedding = Qwen4ExpRMSNorm(
            hidden_size, eps=text_config.rms_norm_eps
        )
        # The released head applies one global RMS normalization across all
        # hyper-connection streams before projecting them independently.
        self.pre_fc_norm_hidden = Qwen4ExpRMSNorm(
            hc_hidden_size, eps=text_config.rms_norm_eps
        )
        self.fc_embedding = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size, bias=False)

        layer_config = replace(
            text_config,
            num_hidden_layers=1,
            layer_types=["qwen_sparse_attention"],
            full_attention_interval=1,
            ple_layer_ids=[],
        )
        self.layers = [Qwen4ExpDecoderLayer(layer_config, layer_idx=0)]
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(
            layer_config, use_combine=False
        )

        self._input_embed = None
        self._lm_head_fn = None
        self._draft_lm_head = None
        self._draft_lm_head_key = None
        self._draft_lm_head_quantization = None
        self._draft_vocab_ids = None
        object.__setattr__(self, "_draft_vocab_ids_array", None)
        self._compile_input_fusion = True
        object.__setattr__(self, "_compiled_input_fusion", None)
        self._cache: List[QSAKVCache] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._next_position = 0
        self._round_appended = 0
        self._kv_valid_len = 0
        self._position = 0
        self._draft_round = 0

        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def configure_draft_lm_head(
        self,
        bits: int,
        group_size: int = 32,
        mode: str = "affine",
        vocab_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """Use a private quantized copy of target-head rows for drafting."""
        if bits not in range(2, 9):
            raise ValueError("draft LM-head bits must be between 2 and 8")
        if group_size <= 0 or self.args.hidden_size % group_size:
            raise ValueError(
                "draft LM-head group size must divide the model hidden size"
            )
        if vocab_ids is not None:
            vocab_ids = tuple(int(token) for token in vocab_ids)
            if not vocab_ids:
                raise ValueError("draft vocabulary must not be empty")
            if any(token < 0 for token in vocab_ids):
                raise ValueError("draft vocabulary token IDs must be non-negative")
            if any(a >= b for a, b in zip(vocab_ids, vocab_ids[1:])):
                raise ValueError("draft vocabulary token IDs must be unique and sorted")
        self._draft_lm_head_quantization = (group_size, bits, mode)
        self._draft_vocab_ids = vocab_ids
        object.__setattr__(
            self,
            "_draft_vocab_ids_array",
            None if vocab_ids is None else mx.array(vocab_ids, dtype=mx.uint32),
        )
        self._draft_lm_head = None
        self._draft_lm_head_key = None

    def bind(self, target_model) -> "Qwen4ExpMTPDraftModel":
        super().bind(target_model)
        quantization = self._draft_lm_head_quantization
        if quantization is None:
            self._install_compiled_input_fusion()
            return self

        target_head = self._lm_head_fn
        if not isinstance(target_head, nn.Linear):
            raise ValueError(
                "Qwen4 draft LM-head quantization requires a dense target LM head"
            )
        group_size, bits, mode = quantization
        vocab_ids = self._draft_vocab_ids
        if vocab_ids is not None and vocab_ids[-1] >= target_head.weight.shape[0]:
            raise ValueError("draft vocabulary token ID exceeds target vocabulary size")
        key = (id(target_head.weight), group_size, bits, mode, id(vocab_ids))
        if self._draft_lm_head is None or self._draft_lm_head_key != key:
            draft_head = target_head
            if vocab_ids is not None:
                draft_head = nn.Linear(
                    target_head.weight.shape[1],
                    len(vocab_ids),
                    bias=getattr(target_head, "bias", None) is not None,
                )
                vocab_array = self._draft_vocab_ids_array
                draft_head.weight = mx.take(target_head.weight, vocab_array, axis=0)
                if getattr(target_head, "bias", None) is not None:
                    draft_head.bias = mx.take(target_head.bias, vocab_array, axis=0)
            self._draft_lm_head = draft_head.to_quantized(
                group_size=group_size,
                bits=bits,
                mode=mode,
            )
            mx.eval(self._draft_lm_head.parameters())
            self._draft_lm_head_key = key
        self._lm_head_fn = self._draft_lm_head
        self._install_compiled_input_fusion()
        return self

    def configure_compiled_input_fusion(self, enabled: bool = True) -> None:
        """Select compiled batch-one, single-token input fusion."""
        self._compile_input_fusion = bool(enabled)
        if not enabled:
            object.__setattr__(self, "_compiled_input_fusion", None)

    def _install_compiled_input_fusion(self) -> None:
        if not self._compile_input_fusion or self._compiled_input_fusion is not None:
            return
        width = self.args.hc_count * self.args.hidden_size
        dtype = self.pre_fc_norm_hidden.weight.dtype
        if dtype not in (mx.bfloat16, mx.float16):
            return
        hidden = mx.arange(width, dtype=mx.float32).reshape(1, 1, width).astype(dtype)
        token_embed = (
            mx.arange(self.args.hidden_size, dtype=mx.float32)
            .reshape(1, 1, self.args.hidden_size)
            .astype(dtype)
        )
        expected = self._fuse_inputs_eager(token_embed, hidden)
        compiled = mx.compile(self._fuse_inputs_eager)
        actual = compiled(token_embed, hidden)
        mx.eval(expected, actual)
        if not mx.array_equal(actual, expected).item():
            raise RuntimeError("compiled Qwen4 MTP input fusion failed exact parity")
        object.__setattr__(self, "_compiled_input_fusion", compiled)

    def _sample_hidden(self, hidden: mx.array, sampler, greedy: bool) -> mx.array:
        token = None
        if greedy and self._draft_lm_head is not None:
            token = _QWEN4_EXACT_SPECULATIVE_VERIFIER.quantized_argmax(
                self._draft_lm_head, hidden
            )
        if token is None:
            logits = self._lm_head_fn(hidden)
            token = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
        if self._draft_vocab_ids is not None:
            token = mx.take(self._draft_vocab_ids_array, token)
        return token

    @property
    def quant_predicate(self):
        def predicate(path, _):
            return not path.endswith("mlp.gate")

        return predicate

    def validate_target_compatibility(self, target_model) -> None:
        target = getattr(target_model, "language_model", target_model)
        args = getattr(target, "args", None)
        model_type = getattr(args, "model_type", "")
        if not str(model_type).startswith("qwen4_exp"):
            raise ValueError(
                "Qwen4-Exp MTP requires a Qwen4-Exp target model, got "
                f"model_type={model_type!r}."
            )
        if getattr(args, "hc_count", None) != self.args.hc_count:
            raise ValueError("Qwen4-Exp target and MTP hc_count do not match.")

    def make_cache(self) -> List[QSAKVCache]:
        return [QSAKVCache() for _ in self.layers]

    def _target_hidden(self, hidden: mx.array) -> mx.array:
        expected = self.args.hc_count * self.args.hidden_size
        if hidden.ndim == 4:
            hidden = hidden.reshape(*hidden.shape[:-2], expected)
        if hidden.ndim != 3 or hidden.shape[-1] != expected:
            raise ValueError(
                "Qwen4-Exp MTP expects target hidden shape "
                "[batch, tokens, hc_count * hidden_size]."
            )
        return hidden

    def _fuse_inputs_eager(
        self,
        token_embed: mx.array,
        hidden: mx.array,
    ) -> mx.array:
        hidden = self._target_hidden(hidden)
        projected_embedding = self.fc_embedding(self.pre_fc_norm_embedding(token_embed))
        hidden_streams = self.pre_fc_norm_hidden(hidden).reshape(
            *hidden.shape[:-1], self.args.hc_count, self.args.hidden_size
        )
        projected_hidden = self.fc_hidden(hidden_streams)
        return (projected_embedding[..., None, :] + projected_hidden).reshape(
            hidden.shape
        )

    def fuse_inputs(
        self,
        token_embed: mx.array,
        hidden: mx.array,
    ) -> mx.array:
        hidden = self._target_hidden(hidden)
        if (
            self._compiled_input_fusion is not None
            and hidden.shape[:2] == (1, 1)
            and token_embed.shape[:2] == (1, 1)
        ):
            return self._compiled_input_fusion(token_embed, hidden)
        return self._fuse_inputs_eager(token_embed, hidden)

    def _forward_hidden(
        self,
        token_embed: mx.array,
        hidden: mx.array,
        tokens: mx.array,
        cache: Optional[List[QSAKVCache]],
    ) -> Tuple[mx.array, mx.array]:
        hidden = self.fuse_inputs(token_embed, hidden)
        if cache is None:
            cache = [None] * len(self.layers)
        position_ids = self._position_ids(length=tokens.shape[1])
        mask = _create_qwen3_5_attention_mask(hidden, cache[0])
        for layer, layer_cache in zip(self.layers, cache):
            hidden = layer(
                hidden,
                tokens,
                mask=mask,
                cache=layer_cache,
                position_ids=position_ids,
            )
        return self.hyper_connection_mixer(hidden), hidden

    def filter_batch(self, keep) -> None:
        if not isinstance(keep, mx.array):
            keep = mx.array(keep, dtype=mx.int32)
        for cache in self._cache:
            cache.filter(keep)
        if self._seed_token is not None:
            self._seed_token = self._seed_token[keep]
        if self._seed_hidden is not None:
            self._seed_hidden = self._seed_hidden[keep]
        for attr in ("_next_position", "_kv_valid_len", "_position"):
            value = getattr(self, attr)
            if isinstance(value, mx.array) and value.ndim > 0 and value.size > 1:
                setattr(self, attr, value[keep])

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        weights = dict(weights)
        stripped = {}
        for key, value in weights.items():
            for prefix in ("language_model.mtp.", "model.mtp.", "mtp."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            stripped[key] = value

        gate_up_key = "layers.0.mlp.experts.gate_up_proj"
        down_key = "layers.0.mlp.experts.down_proj"
        if gate_up_key in stripped:
            gate_up = stripped.pop(gate_up_key)
            gate, up = mx.split(gate_up, 2, axis=-2)
            stripped["layers.0.mlp.switch_mlp.gate_proj.weight"] = gate
            stripped["layers.0.mlp.switch_mlp.up_proj.weight"] = up
            gate_up_scales_key = f"{gate_up_key}_scales"
            if gate_up_scales_key in stripped:
                gate_scales, up_scales = mx.split(
                    stripped.pop(gate_up_scales_key), 2, axis=-2
                )
                stripped["layers.0.mlp.switch_mlp.gate_proj.scales"] = gate_scales
                stripped["layers.0.mlp.switch_mlp.up_proj.scales"] = up_scales
        if down_key in stripped:
            stripped["layers.0.mlp.switch_mlp.down_proj.weight"] = stripped.pop(
                down_key
            )
            down_scales_key = f"{down_key}_scales"
            if down_scales_key in stripped:
                stripped["layers.0.mlp.switch_mlp.down_proj.scales"] = stripped.pop(
                    down_scales_key
                )
        return stripped
