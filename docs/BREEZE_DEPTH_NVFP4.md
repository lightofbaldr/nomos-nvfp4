# Breeze depth NVFP4 recipe vehicle

**Opt-in, default OFF. No recipe or deployment promotion.** Set
`NOMOS_BREEZE_DEPTH_NVFP4=1` before `nomos_breeze_model_init`. The flag is read
once per handle and is independent of `NOMOS_BREEZE_NVFP4_FUSED` (backbone).

Only the 84 depth layer projections (12 layers x q/k/v/o/gate/up/down) may
select packed weights. Stems are the converter's exact dot-to-underscore names:
`depth_decoder_model_layers_N_self_attn_q_proj_weight.nvfp4`, etc.
Heads, norms, input projector, embeddings and backbone selection are unchanged.

## Loading

- Flag OFF: load sibling `.bf16` only. It must exist with the expected size.
- Flag ON: select `.nvfp4` independently per tensor when present, otherwise
  load sibling `.bf16`. Missing BOTH is fatal.
- A present malformed NVFP4 file is fatal, **not** a BF16 fallback: check header
  element/block counts, payload length and finite positive global scale.
- NVFP4-only recipe entries therefore cannot initialize with the flag OFF.
  Use canonical `model-blobs` as the OFF baseline; no implicit cross-directory
  search, conversion, or BF16 materialization is performed.
- Init prints the actual selected NVFP4 projection count out of 84. Absent
  optional depth NVFP4 files are silent. Calibration scales are retained but
  unused by the inline FP32-activation GEMV.

## All depth call shapes are covered

| Entry / cadence | Projection rows | NVFP4 dispatch |
|---|---:|---|
| `depth_begin` / `depth_advance` | 2 / 1 | Per-row fused GEMV |
| `depth_begin2` / `depth_advance2` | **4** / 2 | Per-row fused GEMV |
| `depth_lane`, combined `step` | Call-local 2 then 1 | Per-row fused GEMV |
| `depth(codes, backbone_hidden, out)` | 16 | Per-row fused GEMV |

Both `_depth_pass` and the full-prefix `depth()` route through `_ddmm`.
No depth NVFP4 path uses whole-weight BF16 scratch. Each row uses byte offsets
`src + row*K*4`, `dst + row*N*4`; attention/KV/cache positions remain per lane.
BF16 projections still use the original batched cuBLAS call, with no arithmetic
change. This does not batch AR-dependent codebooks or reuse weights across rows.

The `depth()` explicit backbone hidden is **post-final-norm**, exactly what
`_run` saves into `last_hidden`. A pre-final-norm residual is the wrong oracle
input. Full-prefix and persistent cadences are not assumed byte-identical.

## Gate and handoff

Build `./refresh_breeze_model_build.sh`. Example on a converter-provided recipe:

```sh
NOMOS_BREEZE_DEPTH_NVFP4=1 python3 tools/breeze_depth_route_gate.py \
  --so ./libnomos_model-breeze.so \
  --blobs /path/to/mb-depth-q \
  --bench /path/to/private-hf-goldens --output /path/to/results
```

The tool gates all 35 frames through combined-step, persistent and paired APIs
against HF, plus full-prefix frame zero (S=16). It banks all LM/depth arrays.
`--equal-to <baseline-output-dir>` asserts per-cadence byte equality for BF16
regressions. `--backbone-equal-to <bf16-output-dir>` asserts that depth-only
quantization cannot change backbone logits under identical forced codes.
Nonzero exit on any quality-gate failure is intentional. Zero fat-margin
misses (HF margin >=0.1), backbone 36/36 per lane and the existing relL2 envelope
remain required—no threshold changes for a quantized recipe.

`tools/breeze_depth_loader_probe.mojo` independently exercises the same loader
without adding shipping FFI symbols. Its args are `stem K N enabled expected_nvfp4`.
Missing or corrupt selected data must raise, even if a valid BF16 sibling exists.

Kvasir owns the per-class quality spectrum and its independent verification.
Only after a quality-safe subset exists should GB10/sm_120 timing adjudicate
its performance. Neither lower byte counts nor a successful build establish
fidelity, effective bandwidth, or an end-to-end speedup.
