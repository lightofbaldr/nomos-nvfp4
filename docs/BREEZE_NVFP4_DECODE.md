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

## Measured result (2026-09-09, implementation 4c6da29)

**Do not promote the fleet default.** The requested dispatch is implemented and
removes most of the scratch penalty, but it neither makes full-seven-class
NVFP4 quality-green nor gives q-only a speed advantage over BF16. Q-only fused
passes on GB10; its paired path fails one fat-margin check on sm_120.

Three alternating-order repetitions, 35 forced frames each, real cond/uncond
prefixes (23/10 positions). Below are paired model ms/frame, excluding codec,
prefill/init and first-frame warmup (sum of median per-phase times):

| Hardware / recipe | BF16 | NVFP4 scratch | NVFP4 fused |
|---|---:|---:|---:|
| GB10 / q-only | 124.21 | 140.38 | 127.17 |
| PRO 4000 sm_120 / q-only | 81.28 | 88.37 | 82.60 |
| PRO 4000 sm_120 / full-seven | 81.31 | 160.67 | 93.89 |

For full-seven on the sm_120 host, backbone alone is 97.57 -> 30.81 ms paired
(BF16 18.37); two sequential S=1 lanes are 192.08 -> 41.21 ms backbone
(BF16 32.21), with model total 297.50 -> 146.55 ms (BF16 137.44).
This is a **41.6% paired latency reduction versus the bad NVFP4 route**, not a
win over BF16: paired full-seven still costs 15.5% more than BF16.

The sm_120 host's GPU 2 was selected from live inventory and had no competing GPU compute
process. Host load was high; q-only fused paired totals were 82.60, 95.06,
82.56 ms. The middle run's depth phase was slower despite unchanged code.
That outlier is retained in the report, not discarded. No small q-only versus
BF16 performance advantage is claimed.

### Quality and regression results

Every q-only/BF16 run has backbone 36/36 top1 **in each lane**. Summed depth
fat-margin misses (HF margin >=0.1) are:

| Hardware / API | BF16 | q-only scratch | q-only fused |
|---|---:|---:|---:|
| GB10 / two sequential S=1 lanes | 0 | 0 | 0 |
| GB10 / paired S=2 | 0 | 1 | 0 |
| sm_120 / two sequential S=1 lanes | 0 | 1 | 0 |
| sm_120 / paired S=2 | 0 | 0 | **1** |

All three repetitions reproduce these counts. The sm_120 fused paired miss is
frame index 17, depth-logit index 11 (predicting codebook 12), cond lane 0:
HF token 1754, kernel 1093, HF margin **0.1202378273**. It is the same position
that GB10 scratch paired misses. sm_120 fused paired output arrays repeat
byte-identically; this is not sampling noise. The unchanged bar rejects it.
Mean q-only depth relL2 remains about 0.015, but that does not waive the miss.

Full-seven fails with either route on both tested architectures, consistent
with its prior rejected recipe. On the sm_120 host fused: backbone 35/36 cond, 36/36
uncond; depth fat misses 17/19, for both S=1 and paired paths.

BF16 flag-off/on LM and depth arrays are byte-identical on both architectures.
GB10 also matches a freshly built pre-change main binary byte-for-byte.
Prefill frame-0 logits remain exact between scratch/fused (S=23/10 uses the
unchanged batched path). GB10 all-arm repeats are byte-identical. Scratch/fused
NVFP4 logits themselves differ at roughly 0.004 relative L2, not FP32 reduction
noise; neither that difference nor a shared docstring is the acceptance gate.

### Artifact provenance

- Implementation branch: `codex/breeze-nvfp4-decode`, code commit `4c6da29`.
- GB10 library SHA256:
  `5c1d11ef95b50e37d0b8e81636540ac766bb61a3a389ce2273877cedbccfd502`.
  Preserved under `results/breeze-nvfp4-routing/4c6da29/` in the Breeze
  worktree; `qonly/report.json` and all per-frame arrays are beside it.
- sm_120 host library SHA256:
  `e3fd56477e552b466b59ff20cd5fed438db76711330af6b2a0fa24afd2890c75`.
  Isolated source archive/build in a scratch directory on the sm_120 host,
  with `results/{qonly,full7}/report.json`, all logits, and build/run logs.
  Source checksum matched GB10; the existing sm_120 host checkout/services were not
  changed. Both builds used `refresh_breeze_model_build.sh` successfully.

No ABI, depth precision, active service, or deployed artifact was changed.
Further recipe/default decisions require an explicit quality decision; the
candidate remains available behind its flag for independent verification.
