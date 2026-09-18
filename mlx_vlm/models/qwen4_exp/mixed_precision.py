"""Opt-in Q4/g64 expert and Q8/g64 verifier kernels for Qwen4.

Only singleton verification widths 2..8 use these approximate kernels.
The target pack keeps zero-centered norms and unfused HC scale factors.
"""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .language import Qwen4ExpBatchInvariantForward

ROOT = Path(__file__).with_name("mixed_kernels")


@lru_cache(None)
def kernel(name):
    inputs, outputs = {
        "hc_fused_n_source": (
            ["x_in", "nw", "iw", "eps", "wo_in", "wi_in"],
            ["xn_out", "ipart_out", "xs_out"],
        ),
        "hc_fused_d_source": (
            ["xn_in", "dw_q", "dw_s", "dw_b", "ipart_in"],
            ["act_out", "inj_out"],
        ),
        "hc_fused_u_source": (
            ["xn_in", "act_in", "uw_q", "uw_s", "uw_b"],
            ["mixed_out"],
        ),
        "verify_expert_reuse_source": (["x", "w", "sc", "bi", "ids"], ["y"]),
        "verify_expert_reduce_source": (["down", "inverse", "scores"], ["y"]),
    }[name]
    return mx.fast.metal_kernel(
        name="qwen4_mixed_" + name,
        input_names=inputs,
        output_names=outputs,
        source=(ROOT / (name + ".metal")).read_text(),
    )


@lru_cache(None)
def hc_function(kind, inj, eps):
    def fn(x, nw, dw, ds, db, uw, us, ub, iw, ev, wo, wi):
        rows = x.shape[1]
        pending = kind == "deferred"
        xn, ip, xs = kernel("hc_fused_n_source")(
            inputs=[x, nw, iw, ev, wo, wi],
            template=[
                ("T", x.dtype),
                ("HC", 4),
                ("H", 2560),
                ("INJ", int(inj)),
                ("WR", int(pending)),
            ],
            grid=(1024, rows, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[
                (rows * 10240,),
                (rows * 16,),
                (rows * 10240 if pending else 1,),
            ],
            output_dtypes=[x.dtype, mx.float32, x.dtype],
        )
        act, injection = kernel("hc_fused_d_source")(
            inputs=[xn, dw, ds, db, ip],
            template=[
                ("T", x.dtype),
                ("HC", 4),
                ("H", 2560),
                ("R", 320),
                ("BITS", 8),
                ("GS", 64),
            ],
            grid=(256, 320 + 4 * int(inj), rows),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows * 320,), (rows * 4,)],
            output_dtypes=[x.dtype, x.dtype],
        )
        mixed = kernel("hc_fused_u_source")(
            inputs=[xn, act, uw, us, ub],
            template=[
                ("T", x.dtype),
                ("HC", 4),
                ("H", 2560),
                ("R", 320),
                ("BITS", 8),
                ("GS", 64),
            ],
            grid=(32, 2560, rows),
            threadgroup=(32, 8, 1),
            output_shapes=[(1, rows, 2560)],
            output_dtypes=[x.dtype],
        )[0]
        return (
            mixed,
            injection.reshape(1, rows, 4),
            xs.reshape(x.shape) if pending else x,
        )

    return fn


@lru_cache(None)
def compiled_hc(kind, inj, eps):
    return mx.compile(hc_function(kind, inj, eps))


class Pending:
    def __init__(self, x, branch, injection):
        self.x = x
        self.branch = branch
        self.injection = injection

    def materialize(self):
        return self.x + (self.branch[..., None, :] * self.injection[..., None]).reshape(
            self.x.shape
        )


@lru_cache(None)
def route_kernel():
    return mx.fast.metal_kernel(
        name="qwen4_mixed_route_pack",
        input_names=["ids"],
        output_names=["inverse", "sorted", "lhs"],
        source=(ROOT / "verify_route_pack_source.metal").read_text(),
    )


def route_pack(ids):
    ids = ids.reshape(-1).astype(mx.uint32)
    if not 20 <= ids.size <= 80 or ids.size % 10:
        raise ValueError("Unsupported routed block")
    return route_kernel()(
        inputs=[ids],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[ids.shape] * 3,
        output_dtypes=[mx.uint32] * 3,
    )


def quantized(module, bits):
    return (
        getattr(module, "mode", None) == "affine"
        and getattr(module, "bits", None) == bits
        and getattr(module, "group_size", None) == 64
        and "bias" not in module
        and module.scales.dtype == mx.bfloat16
        and module.biases.dtype == mx.bfloat16
    )


def validate_model(lm):
    a = lm.args
    if (a.hidden_size, a.hc_count) != (2560, 4):
        raise ValueError("mixed_q4_q8 requires Qwen4 hidden_size2560/hc_count4")
    for layer in lm.model.layers:
        for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
            validate_hyper(hc)
        ff = layer.mlp
        if getattr(ff, "top_k", None) != 10 or not hasattr(ff, "switch_mlp"):
            raise ValueError("mixed_q4_q8 requires top10 routed experts")
        for name, shape in (
            ("gate_proj", (512, 640, 320)),
            ("up_proj", (512, 640, 320)),
            ("down_proj", (512, 2560, 80)),
        ):
            p = getattr(ff.switch_mlp, name)
            if not quantized(p, 4) or p.weight.shape != shape:
                raise ValueError("Unsupported Q4 expert layout: " + name)
        if not quantized(ff.gate, 8):
            raise ValueError("mixed_q4_q8 requires Q8 router")
    validate_hyper(lm.model.hyper_connection_mixer)


def validate_hyper(module):
    down, up = module.input_mix_weight_down, module.input_mix_weight_up
    if not (
        quantized(down, 8)
        and quantized(up, 8)
        and down.weight.shape == (320, 2560)
        and up.weight.shape == (10240, 80)
        and module.hc_norm.weight.shape == (10240,)
    ):
        raise ValueError("Unsupported Q8 hyper-connection layout")


class Qwen4MixedVerifier(Qwen4ExpBatchInvariantForward):
    def __init__(self):
        self._params = {}

    @staticmethod
    def eligible(x):
        return (
            x.ndim == 3
            and x.shape[0] == 1
            and 2 <= x.shape[1] <= 8
            and x.dtype == mx.bfloat16
        )

    def _hyper_connection(self, module, value):
        pending = isinstance(value, Pending)
        x = value.x if pending else value
        if not self.eligible(x):
            return super()._hyper_connection(
                module, value.materialize() if pending else value
            )
        down, up = module.input_mix_weight_down, module.input_mix_weight_up
        has_inj = "block_inject_weight" in module
        key = (
            id(module),
            id(module.hc_norm.weight),
            id(module.block_inject_weight.weight) if has_inj else None,
        )
        if key not in self._params:
            iw = (
                mx.contiguous(module.block_inject_weight.weight.T)
                if has_inj
                else module.hc_norm.weight
            )
            ev = mx.array([module.hc_norm.eps], dtype=mx.float32)
            mx.eval(iw, ev)
            self._params[key] = (iw, ev)
        iw, ev = self._params[key]
        fn = compiled_hc(
            "deferred" if pending else "fused", has_inj, module.hc_norm.eps
        )
        mixed, injection, stream = fn(
            x,
            module.hc_norm.weight,
            down.weight,
            down.scales,
            down.biases,
            up.weight,
            up.scales,
            up.biases,
            iw,
            ev,
            value.branch if pending else x,
            value.injection if pending else x,
        )
        return (mixed, stream, injection) if has_inj else mixed

    def _layer(self, layer, hidden, input_ids, mask, cache, position_ids, gdn_sink):
        x = hidden.x if isinstance(hidden, Pending) else hidden
        if not self.eligible(x):
            return super()._layer(
                layer,
                hidden.materialize() if isinstance(hidden, Pending) else hidden,
                input_ids,
                mask,
                cache,
                position_ids,
                gdn_sink,
            )
        ple_state = None
        if "ple" in layer:
            if isinstance(hidden, Pending):
                hidden = hidden.materialize()
            ple_output, ple_state = self._ple(layer.ple, hidden, input_ids, cache, mask)
            hidden = hidden + ple_output
        mixed, hyper_input, injection = self._hyper_connection(
            layer.attn_hyper_connection, hidden
        )
        if layer.is_linear:
            branch = self._gated_delta(layer.linear_attn, mixed, mask, cache, gdn_sink)
            gdn_sink[-1] += (ple_state,)
        else:
            attn_mask = self._qsa_mask(
                layer.self_attn, mixed, cache, position_ids, mask
            )
            branch = self._attention(
                layer.self_attn, mixed, attn_mask, cache, position_ids, None
            )
        mixed, hyper_input, injection = self._hyper_connection(
            layer.mlp_hyper_connection, Pending(hyper_input, branch, injection)
        )
        branch = self._feed_forward(layer.mlp, mixed)
        return Pending(hyper_input, branch, injection)

    def _model(self, *args, **kw):
        value = super()._model(*args, **kw)
        return value.materialize() if isinstance(value, Pending) else value

    def _expert(self, module, x, indices):
        m = x.shape[1]
        top = indices.shape[-1]
        inv, ids, lhs = route_pack(indices)
        flat = x.reshape(m, 2560)
        gu = [
            mx.gather_qmm(
                flat[:, None, :],
                p.weight,
                p.scales,
                p.biases,
                lhs_indices=lhs,
                rhs_indices=ids,
                transpose=True,
                bits=4,
                group_size=64,
                mode="affine",
                sorted_indices=True,
            ).squeeze(-2)
            for p in (module.gate_proj, module.up_proj)
        ]
        act = module.activation(gu[1], gu[0])
        d = module.down_proj
        out = kernel("verify_expert_reuse_source")(
            inputs=[act[:, None, :], d.weight, d.scales, d.biases, ids],
            template=[("T", x.dtype), ("K", 640), ("N", 2560)],
            grid=(32, 640, m * top),
            threadgroup=(32, 2, 1),
            output_shapes=[(m * top, 2560)],
            output_dtypes=[x.dtype],
        )[0]
        return out, inv

    def _feed_forward(self, module, x):
        if not self.eligible(x) or not hasattr(module, "switch_mlp"):
            return super()._feed_forward(module, x)
        gates = mx.softmax(self._linear(module.gate, x), axis=-1, precise=True)
        ids = mx.argpartition(gates, kth=-10, axis=-1)[..., -10:]
        scores = mx.take_along_axis(gates, ids, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        down, inv = self._expert(module.switch_mlp, x, ids)
        output = kernel("verify_expert_reduce_source")(
            inputs=[down, inv, scores],
            template=[("T", x.dtype)],
            grid=(x.shape[1] * 2560, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]
        shared = super()._feed_forward(module.shared_expert, x)
        return output + mx.sigmoid(self._linear(module.shared_expert_gate, x)) * shared

    def _linear(self, p, x):
        if self.eligible(x) and isinstance(p, nn.QuantizedLinear) and quantized(p, 8):
            return p(x)
        return super()._linear(p, x)

    def _linears(self, ps, x):
        if self.eligible(x):
            return tuple(self._linear(p, x) for p in ps)
        return super()._linears(ps, x)
