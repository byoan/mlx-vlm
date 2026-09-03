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

A locally converted target model can be supplied to `--model` in the
generation command as well.
