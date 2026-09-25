"""Dedicated Qwen4 MTP normalization and coarse-to-fine readout."""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

ROOT = Path(__file__).with_name("kernels")


@lru_cache(None)
def kernels():
    partial = mx.fast.metal_kernel(
        name="qwen4_donor_top32_partial",
        input_names=["logits"],
        output_names=["cand_ord", "cand_idx"],
        header=(ROOT / "donor_top32_header.metal").read_text(),
        source=(ROOT / "donor_top32_partial.metal").read_text(),
    )
    final = mx.fast.metal_kernel(
        name="qwen4_donor_top32_final",
        input_names=["cand_ord", "cand_idx"],
        output_names=["token_ids"],
        source=(ROOT / "donor_top32_final.metal").read_text(),
    )
    return partial, final


def top32(logits):
    """Select each row independently, preserving the singleton tie order."""
    if logits.ndim not in (1, 2) or not 16384 <= logits.shape[-1] <= 524288:
        raise ValueError("Donor top32 expects rows of 16384..524288 scores")
    batch = 1 if logits.ndim == 1 else logits.shape[0]
    partial, final = kernels()
    intermediate = partial(
        inputs=[logits],
        template=[("RC", logits.shape[-1])],
        grid=(16384, batch, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, 2048), (batch, 2048)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    result = final(
        inputs=intermediate,
        grid=(256, batch, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, 32)],
        output_dtypes=[mx.uint32],
    )[0]
    return result[0] if logits.ndim == 1 else result


class FoldedRMSNorm(nn.Module):
    """A checkpoint that already stores gamma=1+w must not add one again."""

    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.eps = original.eps
        self.group_size = original.group_size
        object.__setattr__(
            self,
            "_ones",
            (
                None
                if self.group_size is None
                else mx.ones(self.group_size, dtype=mx.bfloat16)
            ),
        )

    def __call__(self, x):
        if self.group_size is None:
            return mx.fast.rms_norm(x, self.weight, self.eps)
        grouped = x.reshape(*x.shape[:-1], -1, self.group_size)
        return (
            mx.fast.rms_norm(grouped, self._ones, self.eps)
            * self.weight.reshape(-1, self.group_size)
        ).reshape(x.shape)


class Qwen4DraftReadout:
    """Q3/g64 over the valid vocabulary, then Q8 scores for the best32 IDs."""

    def __init__(self, head, vocab_size):
        if not (
            isinstance(head, nn.QuantizedLinear)
            and head.bits == 8
            and head.group_size == 64
            and head.mode == "affine"
            and "bias" not in head
        ):
            raise ValueError("q3_top32_q8 requires a bias-free affine Q8/group64 head")
        if not 16384 <= vocab_size <= min(524288, head.weight.shape[0]):
            raise ValueError("Unsupported Qwen4 draft vocabulary size")
        self.head = head
        self.vocab_size = vocab_size
        parts = []
        for start in range(0, head.weight.shape[0], 32768):
            value = mx.dequantize(
                head.weight[start : start + 32768],
                head.scales[start : start + 32768],
                head.biases[start : start + 32768],
                group_size=64,
                bits=8,
                mode="affine",
                dtype=mx.bfloat16,
            )
            q = mx.quantize(value, group_size=64, bits=3, mode="affine")
            mx.eval(q)
            parts.append(q)
        self.weight, self.scales, self.biases = [
            mx.concatenate([p[i] for p in parts], axis=0) for i in range(3)
        ]
        mx.eval(self.weight, self.scales, self.biases)

    def logits(self, x):
        if x.shape[:-1] != (1, 1):
            raise ValueError("Qwen4 top32 drafting requires one sequence and one token")
        row = x.reshape(1, -1)
        coarse = mx.quantized_matmul(
            row,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            bits=3,
            group_size=64,
            mode="affine",
        ).reshape(-1)[: self.vocab_size]
        ids = top32(coarse)
        h = self.head
        logits = mx.quantized_matmul(
            row,
            h.weight[ids],
            h.scales[ids],
            h.biases[ids],
            transpose=True,
            bits=8,
            group_size=64,
            mode="affine",
        ).reshape(-1)
        return logits, ids

    def logits_batch(self, x, *, exact=False):
        """Batched approximate Q3 proposals; retain Q8 scores for rejection sampling.

        The larger Q3 matrix multiply can round differently from singleton
        QMV. It changes the proposal distribution, not target verification.
        """
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[0] < 1:
            raise ValueError("Expected one hidden token per request")
        if x.shape[0] == 1:
            scores, ids = self.logits(x)
            return scores[None], ids[None]
        batch = x.shape[0]
        if exact:
            coarse = mx.concatenate(
                [
                    mx.quantized_matmul(
                        x[i].reshape(1, -1),
                        self.weight,
                        self.scales,
                        self.biases,
                        transpose=True,
                        bits=3,
                        group_size=64,
                        mode="affine",
                    )[:, : self.vocab_size]
                    for i in range(batch)
                ]
            )
        else:
            coarse = mx.quantized_matmul(
                x.reshape(batch, -1),
                self.weight,
                self.scales,
                self.biases,
                transpose=True,
                bits=3,
                group_size=64,
                mode="affine",
            )[:, : self.vocab_size]
        ids = top32(mx.contiguous(coarse))
        h = self.head
        # Per-request selected rows use batched QMV rather than flattening
        # unrelated shortlist IDs into a common vocabulary matrix.
        logits = mx.quantized_matmul(
            x,
            h.weight[ids],
            h.scales[ids],
            h.biases[ids],
            transpose=True,
            bits=8,
            group_size=64,
            mode="affine",
        ).reshape(batch, 32)
        return logits, ids
