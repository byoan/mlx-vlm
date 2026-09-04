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

The released checkpoint contains one MTP layer, so the default draft block is
one speculative token. `--draft-block-size` can chain the head for additional
draft tokens; the best value depends on the prompt and hardware.

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

Finally, the two 48-value gated-delta control projections can share a 96-row
BF16 projection by setting `MLX_VLM_QWEN4_COMBINED_GDN_AB_PROJECTION=1`. This
adds about 18 MB for the checkpoint's 36 gated-delta layers and leaves the
large QKV and output-gate projections separate to avoid cache-pressure losses.

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
