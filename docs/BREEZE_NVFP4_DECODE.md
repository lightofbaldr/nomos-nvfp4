# Breeze small-row NVFP4 route

This candidate is **opt-in**, not a change to the BF16 deployment recipe:
`NOMOS_BREEZE_NVFP4_FUSED=1`, read once at model initialization. Unset/0 keeps
the scratch baseline. No ABI changes; the 512-position safety guards remain.

Only backbone projections carrying NVFP4 weights (`gs != 0`) change:

| Rows | Flag=1 dispatch |
|---|---|
| 1 | One existing inline-dequant FP32 GEMV |
| 2 | Two GEMVs, input stride K floats and output stride N floats |
| >2 | Existing dequant-to-BF16 scratch + batched cuBLAS |
| Any, BF16 weights | Existing BF16 dispatch |

Depth, embeddings, norms, text projection and heads stay BF16. The scratch
allocation remains in `_run`, but the small-row fused branch never materializes
weights into it. No shared-engine dispatch or GPU kernel arithmetic changes.

## Numerical contract

The old shared GEMV docstring overstated equivalence. Scratch rounds dequantized
weights and activations to BF16 before FP32 accumulation; fused uses FP32
dequantized weights and activations. Thus scratch/fused need not be byte- or
greedy-identical. Each must be independently scored against HF goldens.
Do not confuse recipe quality with routing: the full-seven-class NVFP4 recipe
already failed the M3 zero-fat-margin gate before this change. Q-only is a
separate, previously passing recipe.

## Reproduction

Build with `./refresh_breeze_model_build.sh`. Run the same FFI replay and timings
on GB10 and sm_120, selecting an idle GPU and recording the built library SHA:

```sh
python3 tools/breeze_nvfp4_route_gate.py \
  --so ./libnomos_model-breeze.so \
  --bf16-blobs /path/to/model-blobs \
  --nvfp4-blobs /path/to/recipe \
  --bench /path/to/private-hf-goldens \
  --output /path/to/results --repeats 3
```

Optional `--baseline-so` adds the pre-change BF16 binary. The test asserts
byte-identical large-prefill logits and BF16 route invariance, banks all frame
logits, reports each route's HF gate separately, and exits nonzero if any HF
gate fails (expected for the rejected full-seven-class recipe). Sequential
means two independent S=1 lanes, not a single-lane clone benchmark. Paired
means the two-row CFG route. Timings exclude init, prefill, first-frame warmup
and codec; no end-to-end service or PyTorch speed claim follows from them.

Discovery credit: Opus's sm_120 slowdown report and Kvasir's `_bbmm` dispatch
trace. Removing the scratch penalty affects backbone only, not the dominant
BF16 depth pass. A speedup is not permission to ship a quality-failing recipe.
