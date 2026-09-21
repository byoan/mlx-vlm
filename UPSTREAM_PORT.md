# Custom Qwen4 port onto upstream 0.7.2

Local branch: `dev-qwen4-upstream-main`.

- Upstream base: `a74c7de9` (`upstream/main`, fetched 2026-09-22).
- Custom source: `dev-qwen4-mtp-no-replay` at `bd2899c3`.
- The original branch is unchanged. This branch consolidates the applicable
  custom changes on the new upstream architecture; it does not replay the old
  cache implementation over upstream's fixes.

## Preserved features

| Custom work | Treatment |
| --- | --- |
| Native MXFP8 target/drafter loading (`41835b74`, `1c4c1a0f`) | Uses upstream's shared FP8 conversion and parameter layout; preserves native expert and PLE storage and old packed expert aliases. |
| Private quantized/ranked draft heads (`b0561cda`, `94bf66af`, `33bd9206`) | Preserved, including fused greedy argmax and exclusion of padded tokenizer rows. |
| Compiled draft input fusion (`762d3f5f`) | Preserved for supported singleton inputs with its eager fallback and parity check. |
| QSA index/pool preallocation (`6f133f13`, `fe041bd7`, `662208b1`) | Adapted to cache transaction commit/abort, trim, state replacement and row changes; resident byte reporting includes retained capacity. |
| Exact sparse verification and combined HC/MoE/GDN projections (`df8ad8f9` through `22a7aee5`, `b4267b09`) | Preserved behind their existing flags and shape/device guards, using upstream's QSA dispatch and compiled pointwise operations. |
| Optional sparse/fused prefill (`6eef4de1`) | Preserved with separate APC semantic keys. Historical speed/quality measurements in the model README have not been repeated on this branch. |
| Mixed Q4/Q8 verifier (`27e4e2e3`, `4db48e24`) | Preserved and adapted to transactional GDN/PLE windows and deferred residuals. Still restricted to the validated checkpoint layout and singleton verification widths 2–8. |
| Dedicated private-I/O MTP (`5d494716`, `1e71aa60`, `5cd361f1`) | Preserves folded norms, checkpoint embedding/head, Q3 shortlist/Q8 rescore, retained-proposal residual sampling, singleton admission, and neutral repetition-penalty handling. |

The dedicated private-I/O drafter still starts its cache at the generation
boundary. Ordinary native MTP follows upstream's prompt-history prefill and
adaptive runtime ceiling of four total verification tokens by default.
An explicit `draft_block_size` overrides the ceiling.

## Superseded fixes and compatibility adaptations

- Upstream `bf4b8612` already fixes the GDN saved-state stride. The fixed
  `StateT` expression stays unchanged; the source-layout and GPU state/rollback
  regressions are retained and adapted to upstream's cache API.
- The custom snapshot/replay paths (`5748713f`, `e7ced50c`) are replaced by
  upstream's bounded cache transactions. Both single-row and batched rejection
  restore recurrent and PLE windows without replaying the target model.
- Quantized QSA verification now processes attention positions individually.
  The upstream two-position reduction produced different Q4 KV-cache results
  from serial decode in the carried regression tests. The narrow fallback
  restores exact equality for the tested Q4/Q8 text and MRoPE cases.
- Upstream's compiled multirow FP32 forward has small rounding differences from
  sequential decode, reproduced on an untouched upstream snapshot. The carried
  FP32 multirow regression uses `rtol=atol=1e-6`; BF16/FP16 and singleton FP32
  cache/output comparisons remain exact.
- The new transaction API allows retaining zero verified positions. Tests use
  its current semantics rather than treating that operation as invalid.
- The legacy prefill-linear hooks remain available for mlx-server's optional
  native Q4/Q8 acceleration. Verification uses upstream's separate speculative
  operators and cannot be intercepted by those prefill hooks.
- Upstream APC's `resident_bytes` now includes exact-prefix caches. The companion
  change in mlx-server's `state/cache_accounting.py` avoids adding them twice
  and remains compatible with older mlx-vlm counters.

## Validation

Tests use MLX 0.32.2 in a disposable local environment and synthetic tensors or
tiny randomly initialized models. No full checkpoint is downloaded, and no
request is sent to the Mac Studio.

The combined suite covers upstream speculative decoding, cache lifecycles,
generation, model operators, custom kernels, dedicated greedy/sampled drafting,
mixed-verifier transactions, FP8 conversion, and the original GDN regression.
Results: **1,360 passed, 413 subtests passed, 35 skipped** in the combined
fork suite; **160 passed** in mlx-server's VLM integration suite against this
checkout. The skips are all M3 Ultra-specific device guards. The companion
mlx-server accounting change and its regression test are left uncommitted in
the parent repository. The final capacity-accounting follow-up passed 334
focused cache/integration tests. Wheel packaging also succeeded, including all
custom Metal sources and license files.

M3 Ultra-only tests retain their device guards and must be run on the Mac Studio.
Full-model throughput, acceptance rates and the original concurrent workload
still require deployment testing. The previous branch remains the validated
full-model fallback.
