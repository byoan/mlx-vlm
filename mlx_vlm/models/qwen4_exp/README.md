# Qwen3.8-Flash-Next

Qwen3.8-Flash-Next is a large multimodal mixture-of-experts model from Qwen for
text, image, and video understanding. The checkpoint uses the experimental
`qwen4_exp` architecture, combining Gated DeltaNet layers, Qwen Sparse
Attention (QSA), hashed n-gram PLE embeddings, hyper-connections, and a
Qwen3-style vision encoder.

## Model

- Hugging Face ID: `Qwen/Qwen3.8-Flash-Next`
- Modalities: text, image, and video
- Architecture: 48-layer hybrid DeltaNet/QSA MoE with 512 experts
- Best for: multimodal chat, visual reasoning, document and image analysis,
  and video understanding

## CLI

Text generation:

```sh
mlx_vlm.generate \
  --model Qwen/Qwen3.8-Flash-Next \
  --prompt "Explain sparse attention in one paragraph." \
  --max-tokens 256
```

Image understanding:

```sh
mlx_vlm.generate \
  --model Qwen/Qwen3.8-Flash-Next \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

Video understanding:

```sh
mlx_vlm.generate \
  --model Qwen/Qwen3.8-Flash-Next \
  --video ./video.mp4 \
  --fps 1.0 \
  --prompt "Summarize this video." \
  --max-tokens 256
```

## Python

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model_path = "Qwen/Qwen3.8-Flash-Next"
model, processor = load(model_path)

images = ["./image.jpg"]
prompt = apply_chat_template(
    processor,
    model.config,
    "What is happening in this image?",
    num_images=len(images),
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    image=images,
    max_tokens=256,
    temperature=0.0,
)
print(result.text)
```

## MTP speculative decoding

The official checkpoint also contains a native multi-token prediction (MTP)
head. Extract it into a standalone draft model, then use it for speculative
decoding with the official model:

```sh
python -m mlx_vlm.split_mtp \
  --model Qwen/Qwen3.8-Flash-Next \
  --output ./Qwen3.8-Flash-Next-MTP

mlx_vlm.generate \
  --model Qwen/Qwen3.8-Flash-Next \
  --draft-model ./Qwen3.8-Flash-Next-MTP \
  --draft-kind mtp \
  --prompt "Explain speculative decoding in one paragraph." \
  --max-tokens 128
```

The released checkpoint contains one MTP layer. The runtime adaptively chains
that head with a default ceiling of four total verification tokens (one seed
plus up to three proposals). `--draft-block-size` overrides this ceiling;
the best value depends on the prompt and hardware.

Indexed QSA automatically avoids packing padded K/V cache prefixes on Apple
M3 Ultra for single-request BF16 generation at 16,384 tokens and above. This
applies to one- or two-token query windows with 24 query heads, 2 KV heads,
256-wide heads, and 512 selected four-token blocks. The stride-aware kernel
preserves the indexed attention calculation and reads the existing cache layout
directly. Selection uses standard KV-cache capacity metadata without evaluating
arrays. Short contexts, buffers without spare capacity, other cache
implementations, and unqualified devices or layouts retain contiguous addressing.
No additional environment setting is required.

On Apple M3 Ultra, long-context QSA verification can use an exact sparse
attention kernel by setting `MLX_VLM_QWEN4_EXACT_SPARSE_QSA=1`. The kernel is
limited to the checkpoint's 24 query heads, 2 KV heads, and 256-wide BF16/FP16
attention layout. It preserves MLX's 1,024-partition accumulation order and
falls back to the regular attention path for unsupported shapes, devices,
cache layouts, or `MLX_SDPA_BLOCKS` overrides.

The same M3 Ultra path can replace generic QSA block partitioning with a
fixed top-512 radix selector by also setting
`MLX_VLM_QWEN4_RADIX_QSA_TOPK=1`. It keeps the existing FP32 scores and MLX
cutoff-tie behavior, and falls back to `argpartition` outside the checkpoint's
long-context single-request verification layout.

Verification can also combine the hyper-connection mix and injection
projections by setting `MLX_VLM_QWEN4_COMBINED_HYPER_PROJECTION=1`. This keeps
the singleton evaluation order used by exact verification while sharing the
input read across both BF16 projections. The combined weights use about 0.6 GB
for the 48-layer checkpoint and are held only for the lifetime of the model.

The MoE router and shared-expert gate can similarly share their BF16 input
projection by setting `MLX_VLM_QWEN4_COMBINED_MOE_GATE_PROJECTION=1`. This
adds about 0.13 GB for the 48-layer checkpoint. Both combined-projection flags
only affect multi-token exact verification and retain the normal fallback for
other dtypes and shapes.

On Metal, `MLX_VLM_QWEN4_PADDED_MOE_GATE_KERNEL=1` can additionally pad the
checkpoint's 513-row combined router projection to 516 rows and use the exact
multi-token BF16 verification kernel. The padded rows are discarded. This
requires `MLX_VLM_QWEN4_COMBINED_MOE_GATE_PROJECTION=1`, preserves the
singleton accumulation and rounding order, and falls back for unsupported
dtypes or shapes.

The checkpoint's precise softmax, stable top-10 selection, BF16 score
renormalization, and shared-gate sigmoid can then be fused into one Metal
dispatch with `MLX_VLM_QWEN4_FUSED_MOE_ROUTE=1`. This requires both combined
MoE projection flags above and only applies to the checkpoint's 512-expert,
top-10, BF16 verifier layout. Unsupported layouts retain the standard path.
The split projection/tail organization was informed by
[MTPLX PR #391](https://github.com/youssofal/MTPLX/pull/391); the mlx-vlm
kernel supports its dynamic verification widths and MXFP8 checkpoint.

Finally, the two 48-value gated-delta control projections can share a 96-row
BF16 projection by setting `MLX_VLM_QWEN4_COMBINED_GDN_AB_PROJECTION=1`. This
adds about 18 MB for the checkpoint's 36 gated-delta layers and leaves the
large QKV and output-gate projections separate to avoid cache-pressure losses.

On M3 Ultra, `MLX_VLM_QWEN4_EXACT_NORM=1` fuses the normalization steps after
the FP32 mean-square reduction. It preserves the original reduction and
rounding boundaries, including the checkpoint's zero-centered norm weights.
The path supports single-request BF16 inputs with 1–9 tokens, width 10,240,
epsilon `1e-6`, and either full-width or 2,560-wide group normalization.
Other layouts use the original implementation. Both normal decoding and MTP
verification can use this flag; it does not change checkpoint storage.

Performance measurements below were collected on the previous custom branch
(`dev-qwen4-mtp-no-replay`). They have not been repeated on the upstream 0.7.2
port; see [port notes](../../../UPSTREAM_PORT.md) for validation and remaining
hardware checks.

## Experimental prefill controls

On M3 Ultra, `MLX_VLM_QWEN4_PREFILL_RADIX_QSA_TOPK=1` enables the radix
block selector for prefill batches with more than eight queries and at least
8,192 compressed key blocks. Selection preserves the existing cutoff-tie
behavior and ranks masked future blocks below valid zero-score blocks.
It can be used independently of the decode attention flags.

`MLX_VLM_QWEN4_SPARSE_PREFILL=1` also enables matrix attention over the
selected KV rows. The supported layout is an unpadded, single-request BF16
prefill on M3 Ultra: 24 query heads, two KV heads, head width 256, and a
512-block QSA budget with four tokens per block. It requires at least 2,048
cached tokens and 32,768 cached plus new tokens. It falls back for training,
quantized KV caches, and additive or per-head attention masks. Shared Boolean
masks, including the single-row MTP batch-cache mask, are intersected with the
selected positions.
Temporary KV gathers are bounded to 128 query positions.

Sparse prefill groups sibling query heads into matrix rows. This changes
floating-point accumulation and can change logits and generated text, despite
using the same weights and QSA selection rule. Treat it as an optional
speed/quality tradeoff, rather than an exact replacement for dense masked SDPA.
APC semantic keys distinguish this mode from the default prefill path.

With unchanged MXFP8 weights and 2,048-token chunks, three cold 164,802-token
MTP runs on M3 Ultra measured median prefill times of 452.9 seconds for the
default path and 264.4 seconds for sparse prefill (364 versus 623 tokens/s).
Generated responses differed. A small 192-token continuation-likelihood probe
measured 2.3% higher perplexity; this does not establish broad quality
equivalence. Evaluate the option on your own tasks before adopting it.

Additionally setting `MLX_VLM_QWEN4_FUSED_SPARSE_PREFILL=1` selects a fused
Metal implementation adapted from [mlx-serve's `gatherQsa256`](https://github.com/ddalcu/mlx-serve/blob/9dd536a1ef860b08c9677c5f1d739ed04b33515e/src/transformer.zig).
It reads the
selected keys directly, keeps scores and probabilities in FP32, and supports
the same shared Boolean masks. This implementation has its own APC semantic
key because its numerical results differ from the matrix implementation.
The flag requires `MLX_VLM_QWEN4_SPARSE_PREFILL=1`; both default to off.

Two integrated runs measured 203.15 and 203.08 seconds on the same cold
164,802-token workload (811 tokens/s), with a 264.43-second matrix control
between them. A preceding causal-only prototype measured 203.27 seconds.
The small continuation probe measured 1.7% lower perplexity than baseline;
this is encouraging but does not establish broad quality equivalence.

Larger `--prefill-step-size` values, such as 8192, are a separate control.
They can improve matrix utilization but also change numerical results and
increase working memory. Evaluate chunk size and sparse attention separately
on representative long-context tasks. Neither flag changes the default chunk
size or checkpoint quantization.

## Optional quantization

The official BF16 checkpoint is approximately 360 GB. Depending on the
available memory and desired quality/performance tradeoff, it can also be
converted to a lower-bit MLX checkpoint. For example:

```sh
mlx_vlm.convert \
  --hf-path Qwen/Qwen3.8-Flash-Next \
  --mlx-path ~/Qwen3.8-Flash-Next-3bit \
  --quantize \
  --q-group-size 32 \
  --q-bits 3
```

Group size 32 allows the PLE embedding dimensions to be quantized. The bit
width and output path can be adjusted for the target hardware.

The extracted MTP head can be quantized independently:

```sh
python -m mlx_vlm.split_mtp \
  --model Qwen/Qwen3.8-Flash-Next \
  --output ./Qwen3.8-Flash-Next-MTP-3bit \
  --q-group-size 32 \
  --q-bits 3
```

## Notes

- The base conditional-generation runtime ignores the embedded `mtp.*`
  tensors. `mlx_vlm.split_mtp` extracts them into the standalone draft model
  used by speculative decoding.
- QSA maintains an auxiliary index-key cache in addition to the normal KV
  cache. Single-request generation, chunked prefill, continuous batching, and
  uniform KV-cache quantization for single requests are supported.
- KV-cache quantization is unsupported with continuous QSA batching;
  requesting both raises an explicit error to preserve the QSA indexer state.
- Long image or video prompts may benefit from a smaller
  `--prefill-step-size` to reduce peak memory.

## Optional pooled QSA buffer

`MLX_VLM_QWEN4_PREALLOCATED_QSA_POOL=1` grows completed QSA block summaries
in 256-block steps, avoiding a full concatenation on every append. It defaults
to off and applies to single unpadded requests with unquantized KV caches.
Attention and serialized state see only the logical prefix. Speculative
rollback retains capacity while trimming visible summaries; state replacement
and row transformations discard the backing buffer. This uses upstream's
`index_block_keys` cache rather than the older pooled-cache representation.

On M3 Ultra with the MXFP8 target and native MTP drafter, three interleaved
164,802-token cached-prefix / 1,024-generated-token pairs measured median
decode throughput of 34.21 tokens/s without this option and 34.56 with it
(about 1.0%). All six runs had identical tokens and acceptance traces. This
is a decode measurement; it does not establish a cold-prefill speedup.

### Singleton batch-cache QSA dispatch

The stride-aware M3 Ultra QSA path also recognizes the standard `BatchKVCache`
behind `BatchQSAKVCache` for one active sequence. It uses the physical prefix
index without evaluating tensor offsets. The existing 16K context, dtype,
geometry and spare-capacity checks still apply. Restored caches without visible
spare capacity, custom cache updates and multiple active sequences retain the
contiguous path. This covers the singleton cache layout used by `BatchGenerator`.

### Runtime-length radix selection

With `MLX_VLM_QWEN4_RADIX_QSA_TOPK=1`, eligible decode and MTP verification
calls (up to eight query rows) read the block count at runtime. They reuse one
Metal specialization as context grows, avoiding compilation for each new block
count. The radix algorithm, ordered selected IDs, top-512 threshold and existing
hardware/environment eligibility remain unchanged. Larger prefill calls keep
the static-length kernel. This targets stalls on previously unseen long-context
lengths, rather than increasing already-warmed steady-state throughput.

## Opt-in fixed-cohort MTP generation

`MLX_VLM_QWEN4_MTP_BATCH_SIZE=2` allows `BatchGenerator` to admit up to two
compatible dedicated Qwen4 MTP requests together. The default is `1`, retaining
the existing singleton route. Set the generator's `completion_batch_size` and
`prefill_batch_size` to at least the desired cohort size. Larger values are
supported by grouping verification work into at most 32 token rows per kernel
invocation; there is no hard-coded two-request limit.

This path targets the dedicated `q3_top32_q8` drafter with unquantized target KV.
Keep `mixed_q4_q8` and the existing optimization environment settings enabled.
Each request owns its sampler, draft cache, target cache, accepted length and
position. Mixed HC/MoE verification combines token-independent work; attention,
GDN and PLE history stay request-local, preserving the existing stride-aware
QSA, runtime-length radix, pooled-cache and rollback paths. Draft readout retains
its optimized singleton implementation per request.

Prefill uses each request's singleton chunk boundaries and APC lookup. Padded
prefill can change Qwen4 hidden states and seeded continuations, so cohorts join
at the generation boundary. All cohort members therefore wait for cohort
prefill to finish before generation starts. The intended benefit is aggregate
throughput, not improved per-request decode latency.

This is fixed-cohort batching: new arrivals wait while a speculative cohort is
active. Different sampling configurations continue to follow the serving
owner's compatibility grouping. Adding requests to an active cohort and batching
the draft readout itself are separate future optimizations.
