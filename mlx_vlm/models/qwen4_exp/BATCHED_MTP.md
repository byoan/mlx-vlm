# Batched and continuous Qwen4 MTP

This opt-in path targets Flash-Next with the dedicated private-I/O MTP checkpoint,
`q3_top32_q8` draft readout, unquantized caches, and request-local sampled-MTP
samplers. Keep the target's existing optimization profile and kernel settings.

```sh
export MLX_VLM_QWEN4_MTP_BATCH_SIZE=4
export MLX_VLM_QWEN4_CONTINUOUS_MTP=1
export MLX_VLM_QWEN4_BATCHED_DRAFT=exact
export MLX_VLM_QWEN4_BATCHED_QSA=1
export MLX_VLM_QWEN4_BATCHED_GDN=1
```

The scheduler's completion capacity must also be at least the desired batch size.
A server-level concurrency cap can lower it. Only requests with compatible model,
drafter and sampling configuration should share a generator. These controls do
not change checkpoint files. Their defaults preserve the previous behavior:
capacity one, fixed cohorts, and the original per-request draft/temporal paths.

## Scheduling and state

Continuous mode admits requests between committed speculative rounds. It
interleaves existing chunked prefill with decode, starts finished short prefills
without waiting for longer cohort members, and reuses completed request slots.
Each request owns its target caches, draft cache, sampler, token limit and RoPE
delta. Cancellation and rejection cannot trim another request's cache. Completed
cache states are released; aggregate diagnostic history is bounded.

This is round-level continuous admission, not a unified packed prefill/decode
kernel or a paged KV allocator. One prefill chunk can delay the next decode
round. `prefill_step_size` controls this latency/throughput tradeoff. Request
state remains independently allocated, so memory still limits useful concurrency.

The verifier shares token-local compute in groups of at most 32 token rows.
Temporal kernels use groups of up to four requests. Larger cohorts are split
into groups rather than rejected at a two-request ceiling. When only one request
is active, the established singleton verifier and drafter are used within the
continuous scheduler. Setting batch capacity to one uses the original MTP loop.

## Native temporal kernels

- `MLX_VLM_QWEN4_BATCHED_QSA=1`: on M3 Ultra, BF16 QSA with 24 query heads,
  two KV heads, head dimension 256, and 512 selected four-token blocks reads
  two to four independent cache buffers in one Metal launch. Cache offsets
  must be at least 16,384 and the verification block must end below 65,536.
  The kernel keeps the singleton pairwise query windows, projection shapes and
  attention reduction order. It does not stack or copy long cache prefixes.
- `MLX_VLM_QWEN4_BATCHED_GDN=1`: on M3 Ultra, groups of four requests share
  the bounded recurrent transition and rollback history. Projections,
  convolution and normalization retain their original per-request shapes.
  Smaller groups use the original path because batching this part did not help.

Both paths fall back before mutating caches when eligibility checks fail.
Existing short-context and 64K+ attention paths remain available. Other devices
have not been performance-qualified for these kernels.

## Draft head modes

`MLX_VLM_QWEN4_BATCHED_DRAFT` accepts:

| Mode | Shared work | Numerical contract |
| --- | --- | --- |
| `off` (default) | Original per-request drafter | Original behavior |
| `exact` | Metal top-32 selection and Q8 shortlist rescoring | Retains individual Q3 matvec shapes and singleton tie ordering |
| `readout` | Q3 head multiply, selection and Q8 rescoring | Q3 rounding and seeded proposals can change |
| `full` | Readout plus eligible draft projections, hyper-connections and MoE | Draft rounding and seeded proposals can change |

All modes retain each request's actual proposal probabilities for the rejection
sampler. `readout` and `full` change the approximate proposal, not the target
verification or target sampling rule. They are experimental performance choices;
a faster isolated head did not consistently produce faster full-model generation.
Use `exact` for the measured long-context configuration.

## Joint prefill experiment

`MLX_VLM_QWEN4_BATCHED_PREFILL=1` submits eligible text prefill chunks layer by
layer across requests. It preserves request-local operation shapes, chunk/APC
boundaries, position metadata and caches. It shares graph submission rather than
packing target matrix multiplies. Multimodal construction, padding and unsupported
cache types retain ordinary prefill.

Flattening target hyper-connection/MoE prefill matrices changed numerical results
in the experiment, so that candidate is not enabled by this implementation.
Joint submission is optional and should be benchmarked separately from admission:
continuous admission also works with ordinary round-robin prefill.
