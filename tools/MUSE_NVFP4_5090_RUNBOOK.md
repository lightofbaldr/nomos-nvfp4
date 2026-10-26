# Muse-Glimmer-30B — NVFP4 / 5090 arm run-book (#79)

Staged by Kvasir (on the GB10 box) for Prime (RTX 5090 Laptop, sm_120). This is the
**discrete/NVFP4 arm** of the Muse text-tower port. The GB10/Q4 arm's lossless (kernel-vs-kernel) + perf claims HOLD, but a sharper margin instrument
(Prime, 2026-08-10) later found a small SHARED decode-quality gap vs HF the aggregate-sigma test missed
(decode cos ~0.01-0.015 under prefill on BOTH arches, a few genuine HF-confident misses, worse on NVFP4)
-- so "verified in-family" was overstated; lossless is unaffected. See G1. The Q4 arm otherwise:
default Q4 route, **E=2.83, 6/6 own-base identity (lossless), spec 17.06 vs base 12.12 tok/s (1.41×)**.
This arm needs its own gates because NVFP4 W4A4 is a separate precision path — nothing here inherits the
Q4 numbers.

## What I verified from source so you don't re-derive it

1. **Compile-a-profile-per-model.** The Muse profile is on branch `kvasir/muse-glimmer-port` @ `8da87a6`
   (VOCAB=202048, the Muse deltas). Build it on your box — never copy a `.so` across arches. sm_120.
2. **The drafter body is Q4 on BOTH arches — there is NO NVFP4 drafter body.** `dflash_draft_block_nvfp4`
   runs the *same* `dflash_forward_block_cached` body and only swaps the lm-head readout to
   `_dflash_readout_argmax_nvfp4`, which reads through the **target's** already-loaded W4A4 head
   (`engine_ptr[0].d_w4a4_packed` …). So the drafter artifacts you need are the **Q4 assistant** files —
   the same `muse-assistant-flat-q4/` used on GB10. No NVFP4 assistant converter exists or is needed.
3. **`load_to_gpu_q4` is device-resident** (`cuda_malloc` + `cuda_upload_u8`, strips the 16-byte header) —
   discrete-safe. That's why Gemma's Q4 drafter body already runs on Gold/5090; Muse's will too. The
   Q4_0-MMU-fault (Xid 31) rule is about the *target's* unified-memory Q4 path, not this.
4. **Path selection is automatic:** `use_nvfp4 = engine_ptr[0].d_embed_nvfp4 != 0`. Load the target as
   NVFP4 and the drafter follows — **no env flag** for the draft route.
5. **theta transfers; the Q4 readout fix does NOT.** The RoPE θ=500000 fix is in-kernel (`log(theta)`,
   arch-independent) — it carries. The `8da87a6` "route Q4 draft readout through validated v4" fix is
   Q4-only; the NVFP4 readout (`_dflash_readout_argmax_nvfp4`) is a *separate* path and must be gated on
   its own (that's G2 below — Codex flagged this explicitly).

## Artifacts (all pre-built on the GB10 box; nothing to convert)

| what | GB10 path | R2 | size |
|---|---|---|---|
| NVFP4 target flat | `~/nomos_data/muse-glimmer-flat-nvfp4/` (417 `.nvfp4` + 211 `.bin`) | `r2:lofb-data/muse-nvfp4-5090/target/` | 24G |
| Q4 assistant (drafter body) | `~/nomos_data/muse-assistant-flat-q4/` (36 `.q4` + 22 `.bin`) | `r2:lofb-data/muse-nvfp4-5090/assistant/` | 1.4G |
| HF goldens (per-layer + logits) | `~/kvasir/muse-ref/goldens/` | `r2:lofb-data/muse-nvfp4-5090/goldens/` | 71M |

Pull: `rclone copy r2:lofb-data/muse-nvfp4-5090/target ~/nomos_data/muse-glimmer-flat-nvfp4` (etc.).
**Alternative (if you'd rather regenerate):** `tools/convert_muse_glimmer.py --format nvfp4 <base> <out>`
and `tools/convert_muse_assistant_q4.py <assistant> <out>` from the HF base weights (needs the
Muse-aware transformers ≥5.16.0.dev0 in an isolated venv — see below). The pre-built flats are byte-for-
byte what those converters emit.

## Build (sm_120, Muse profile)

> **Build-env prerequisite (Prime, 2026-08-10 — a clean box exposes this).** `pixi.toml` on this
> branch declares only `mojo` + `max-core` and **NO CUDA** (libcublas rode in as an undeclared
> transitive; zero `libcudart`, zero `cuda_runtime.h`, zero `nvcc`). It only ever built where a
> system CUDA sits at `/usr/local/cuda`. On a card-only laptop you must supply CUDA yourself:
> add `cudart`/`cublas`/headers as **target-scoped** `[target.linux-64.dependencies]` (so
> aarch64/GB10 keeps its sbsa path untouched), and set `MODULAR_HOME=<env>/share/max` or Mojo
> can't find its stdlib (`unknown declaration 'Int32'` / `unable to locate module 'std'`).
> `pixi install`/`pixi run` currently panic `Tokio executor failed` in this repo — **pre-existing**
> (reproduces on pristine `8da87a6`), tracked as a task; Prime has a working fix staged.

```bash
git fetch && git checkout kvasir/muse-glimmer-port    # kernel @ 8da87a6
# the tools/ (converters + harnesses + this run-book) are on origin/main, NOT this branch:
git checkout origin/main -- tools/
# build per the discrete SOP (SOP_GOLD.md pattern, sm_120, cuda lib64), then VERIFY it's in:
nm -D libnomos_kernel.so | grep nomos_dflash_draft_block   # exported FFI entry — must print
nm    libnomos_kernel.so | grep -c nvfp4                   # nvfp4 code compiled in — expect >0 (11)
# NB: dflash_draft_block_nvfp4 is INTERNAL (dispatched inside nomos_dflash_draft_block via
# use_nvfp4), NOT a dynamic symbol — `nm -D | grep ..._nvfp4` returns 0 ALWAYS. Do not "fix" that
# by exporting it. (Corrected by Prime, 2026-08-10.)
```

## Discrete NVFP4 env (the 5090 profile)

```
NOMOS_WEIGHT_NVFP4=1 NOMOS_W4A4=1 NOMOS_PRECISION_BITS=16
NOMOS_KV_QUANT=1 NOMOS_KV_INT4=1 NOMOS_KV_I4_BLOCK=32 NOMOS_KV_SWA=0
NOMOS_MAX_SEQ=4096 NOMOS_BATCHED_PREFILL=1 CUBLAS_WORKSPACE_CONFIG=":4096:8"
# (NOMOS_KV_REUSE is retired: the KV session store replaced it, see docs/RUNBOOK.md §7. Setting it does nothing.)
# CRITICAL, do NOT inherit ./nomos's default here:
NOMOS_VERIFY_BLOCK_ATTN=0     # =1 is LOSSY on discrete/NVFP4 (#65) — it will break the lossless gate
```

## Gate sequence — fail-closed, in order (each blocks the next)

- **G1 — NVFP4 target parity, KV quant OFF to isolate the weight path (Prime, 2026-08-10).** Run with
  `NOMOS_KV_QUANT=0 NOMOS_KV_INT4=0` — the goldens use HF full-precision KV, so INT4 KV on top measures
  weights+KV against a weights-only reference and conflates two lossy axes.
  `muse_parity.py --so <so> --weights <nvfp4 target> --golden <golden> --dump-dir <d>` (both prompts),
  then **`muse_margin.py --dump-dir <d>`**. PASS = smooth per-layer slope (no cliff), prefill
  prompt-logits cos > 0.98, AND the margin split shows misses are near-ties, NOT genuine (HF-confident)
  errors — muse_parity's agree/total CANNOT tell those apart; muse_margin uses HF's own per-token
  confidence and can. **KNOWN & UNRESOLVED (shared, both arches): decode cos ~0.01-0.015 below prefill,
  a few genuine misses, worse on NVFP4 — a prefill-clean/decode-degraded split. Report G1 with the
  margin split, never prompt-logits cos alone.** L51 cos~0 is the instrumentation seam (pre-final-norm
  tap vs zero-mean final norm), not a defect.
- **G1b — INT4 KV as its OWN gate.** With NVFP4 weights fixed, turn KV quant back on and measure the
  delta vs G1. INT4 KV demonstrably sinks NVFP4 prefill 0.9735->0.9914 and flips token 0 — lossy by
  construction; the gate is "how much, acceptable for the demo?", its own axis and tolerance, not
  byte-parity.
- **G2 — drafter readout through the NVFP4 head (Codex's named gate; does NOT inherit the Q4 v4 fix).**
  Capture the HF drafter surfaces once (`~/kvasir/muse-ref/hf_drafter_surfaces.py`, isolated venv), then
  `python3 tools/muse_drafter_diff.py --so <so> --weights <nvfp4 target> --dflash <q4 assistant>`.
  PASS = final→logits cos high and proposals match HF. The harness teacher-forces the anchor to HF's and
  asserts the manifest anchor (fails loud on a stale reference). If this diverges, it is the W4A4 readout
  route specifically — pair with me/Codex; do not chase the body (body is the validated Q4 path).
- **G3 — acceptance + lossless (auto-routes NVFP4).**
  `python3 tools/muse_spec_e2e.py --so <so> --weights <nvfp4 target> --dflash <q4 assistant> --tokens tools/muse_bar_tokens.json --ntok 96 --vb 9`.
  PASS = all 6 fail-closed gates green, **6/6 identity=EXACT** (spec ≡ own greedy, kernel-vs-kernel — the
  real lossless bar, NOT the cross-precision HF greedy), mean E in-family (~2.8 expected; Q4-vs-NVFP4 may
  move it — measure, don't assume).
- **G4 — VB curve + tok/s.**
  `python3 tools/muse_vb_sweep.py … --vbs 5,7,9,11,13`. On GB10 the fast body plateaus VB 5–9 (verify-
  bound, 83%); the 5090's balance differs — read the cycle decomposition, don't port the GB10 optimum.

## Caveats to carry in

- `tools/convert_redhat_nvfp4.py` carries a 2026-06-29 note that NVFP4 correctness on **the GB10 box**
  was unsettled; that's a GB10-NVFP4 caveat, but re-read it before trusting any NVFP4 build — the 5090 is
  the shipping path for NVFP4, so this is the arch where it should be *most* correct, but verify G1 first.
- Prove the code-under-test is IN the binary: `nm -D | grep nomos_dflash_draft_block` (exported entry)
  AND `nm | grep -c nvfp4` > 0 (internal nvfp4 code), `.so` newer than source, and the build SHA —
  before believing any number. (#72 is still open: the SHA isn't embedded in the `.so` yet, so check
  mtime-vs-commit.)
- Every greedy "DIVERGENCE" the harness prints is its any-miss label; the real verdict is the near-tie
  margin statistics + the kernel-vs-kernel identity gate. Don't carry the label into the result.
- **Export `LD_LIBRARY_PATH` for the borrowed CUDA** or the `.so` won't load (`libcudart.so.13`) —
  building against a sibling env's CUDA doesn't put it on the loader path (Prime, 2026-08-10).
- **Watch for `cudaMalloc rc=2` mid-run that looks like a kernel defect but is a memory race.** On the
  5090 box ollama re-loads qwen3-embedding (~10.7G) in ~4-min bursts; a 14.6 MB alloc landing during a
  load OOMs and reads exactly like a kernel bug. the host that owns that process being down means it can't be stopped — retry
  the race, don't debug the kernel. (Prime, 2026-08-10.)
- **Gate-design law (2026-08-10, learned twice in one day):** a gate that asks "is this difference big
  enough to matter?" MUST key on a **magnitude** (spread, prob-gap, max_abs) — NEVER an ordinal/shape
  proxy (HF-rank, correlation-r): both clean base runs had drift r~-0.9, and HF-rank 2 can be 0.98 vs 0.01.
  And **validate every gate against >=2 known-verdict fixtures** (one must-pass, one must-fail) before
  trusting it. G7 (per-prompt speed floor) and G8 (base-decay spread canary) embody this; G2 still lacks a
  numeric threshold = a gate gap. Full principle: memory `magnitude-not-shape-gates-2026-08-10`.

Kvasir holds the goldens, the HF reference env recipe, and the GB10 baseline for A/B. Ping me on any gate.
