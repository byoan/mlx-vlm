# Qwen4-Exp MTP

This drafter runs the native multi-token prediction head embedded in Qwen4-Exp
checkpoints. Extract it with the generic splitter, then pass the resulting
folder as an MTP draft model:

```bash
python -m mlx_vlm.split_mtp \
  --model Qwen/Qwen3.8-Flash-Next \
  --output ./Qwen3.8-Flash-Next-MTP

mlx_vlm.generate \
  --model Qwen/Qwen3.8-Flash-Next \
  --draft-model ./Qwen3.8-Flash-Next-MTP \
  --draft-kind mtp \
  --max-tokens 128 \
  --prompt "Explain speculative decoding in one paragraph."
```

The native checkpoint contains one MTP layer, so the default draft block is
one speculative token. A larger `--draft-block-size` chains the same head
autoregressively; whether that is faster depends on prompt and hardware.

## Optional quantization

The head can be quantized independently when it is extracted:

```bash
python -m mlx_vlm.split_mtp \
  --model Qwen/Qwen3.8-Flash-Next \
  --output ./Qwen3.8-Flash-Next-MTP-3bit \
  --q-bits 3 \
  --q-group-size 32
```

The MTP checkpoint does not contain a vocabulary projection, so drafting uses
the target model's full-precision LM head by default. For greedy decoding, a
private quantized copy can reduce the cost of each autoregressive proposal
without changing the target verifier or the final generated tokens:

```bash
mlx_vlm.generate \
  --model Qwen/Qwen3.8-Flash-Next \
  --draft-model ./Qwen3.8-Flash-Next-MTP-3bit \
  --draft-kind mtp \
  --draft-block-size 4 \
  --draft-head-bits 4 \
  --max-tokens 128 \
  --prompt "Explain speculative decoding in one paragraph."
```

The copy uses affine quantization with group size 32 and is held only in
memory; the target model and checkpoint files are not modified. Quantization
can change which tokens the drafter proposes, so acceptance should be measured
on the intended workload even though target verification remains lossless.

For large vocabularies, drafting can project only a ranked subset while target
verification continues to use the complete vocabulary. The JSON file must be a
list of token IDs (or an object with an `ids` list); IDs are deduplicated and
sorted by the CLI. MXFP8 heads use 8-bit, group-size-32 quantization:

```bash
mlx_vlm.generate \
  --model ./Qwen3.8-Flash-Next-MXFP8 \
  --draft-model ./Qwen3.8-Flash-Next-MTP-MXFP8 \
  --draft-kind mtp \
  --draft-block-size 7 \
  --draft-head-bits 8 \
  --draft-head-mode mxfp8 \
  --draft-vocab ./ranked-draft-vocab.json \
  --max-tokens 128 \
  --prompt "Explain speculative decoding in one paragraph."
```

Here `--draft-block-size 7` means one verified seed token plus at most six MTP
proposals. Restricting the draft vocabulary can change proposal tokens and
acceptance, especially with sampling; benchmark the intended temperature and
prompt distribution.

The batch-one, single-token input-fusion graph used by chained proposals is
compiled automatically after binding the drafter. Prompt/history fusion and
other shapes remain on the eager path.

The exact Qwen4 verifier also has opt-in kernels for the released model's
BF16 verification intermediates. After enabling the combined MoE projection
and fused routing path, the weighted routed-expert reduction and gated shared
expert add can be fused with:

```bash
export MLX_VLM_QWEN4_FUSED_MOE_COMBINE=1
```

This kernel is limited to batch-one speculative widths 2 through 8 and the
released 10-expert, 2560-wide output layout. It preserves MLX's BF16 ten-row
reduction order and falls back for other shapes or dtypes.

A locally converted target model can be supplied to `--model` in the
generation command as well.
