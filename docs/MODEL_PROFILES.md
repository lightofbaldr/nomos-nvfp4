# Model profiles — one repo, one main, either model

## The problem this solves

Until 2026-08-15, serving a second model meant a **branch per model**.
`kvasir/muse-glimmer-port` did not add Muse support; it *replaced* Gemma-4:

| alias | Gemma-4-31B | Muse-Glimmer-30B | OLMo-3.1-32B |
|---|---|---|---|
| `D` | 5376 | 6656 | 5120 |
| `NKV` / `HD` | 16 / 256 | 2 / 128 | 8 / 128 |
| `FF` | 21504 | 19968 | 27648 |
| `VOCAB` | 262144 | 202048 | 100278 |
| `TOTAL_LAYERS` | 60 | 52 | 64 |
| `SLIDING_WINDOW` | 1024 | 2048 | 4096 |

These are `alias` — **compile-time**. They size ABI-visible buffers, KV caches and
argmax storage, and several appear as *static layout shapes* (`[NKV, KMAX, HD]`,
`[VOCAB, D]`). So **one `.so` serves exactly one model geometry**, and the branch could
never merge: merging it would have stopped Gemma-4 loading.

The fix is not to make the branch mergeable. It is to stop encoding the model in the
*branch* and encode it in the *build*.

## How it works

- `lib/model_profiles/<name>.mojo` — one tracked file per model. All geometry, plus
  `MODEL_ID`, `MODEL_NAME`, `PROFILE_COMPLETE`.
- `lib/model_config.mojo` — **generated**, gitignored, rewritten on every build as a
  copy of the chosen profile. The engine imports only this.
- `bash refresh_build.sh --model <name>` — writes the config, builds
  `libnomos_kernel-<name>.so`, and publishes `libnomos_kernel.so` as a copy **after**
  the symbol gate passes.

```bash
bash refresh_build.sh --model gemma4 --tokenizer-dir /path/to/gemma4-tokenizer
bash refresh_build.sh --model muse --tokenizer-dir /path/to/muse-tokenizer
```

Publishing the generic name last is deliberate: building straight to
`libnomos_kernel.so` means a **failed build deletes the working kernel**, which happened
on 2026-08-15.

## Build-time special-token gate

Every `refresh_build.sh` invocation now requires a local `--tokenizer-dir`, or
`NOMOS_TOKENIZER_DIR` in the environment. The CLI path takes precedence. There is
no guessed model directory, network download, warning-only mode, or bypass via
`FORCE_INCOMPLETE`. Missing or contradictory metadata fails before generated config
publication, CUDA discovery, environment installation, or compilation. Existing build
automation must provide this new input explicitly.

```bash
# Safe CPU-only preflight: does not rewrite config or build/load a library.
bash refresh_build.sh --model olmo3 \
  --tokenizer-dir ~/nomos_data/olmo-3.1-32b-instruct-tok --check-profile

# Same gate, followed by the normal approved build.
bash refresh_build.sh --model olmo3 \
  --tokenizer-dir ~/nomos_data/olmo-3.1-32b-instruct-tok

# Offline tests; no transformers, weights, compiler, shared library, or GPU.
python3 tools/test_check_profile_tokens.py
python3 tools/test_check_profile_tokens.py \
  --local-tokenizer ~/nomos_data/olmo-3.1-32b-instruct-tok
```

`tools/check_profile_tokens.py` uses the Python standard library (default `python3`,
overridable with `NOMOS_BUILD_PYTHON`). It reads literal `VOCAB`, `TARGET_BOS_ID`,
`TARGET_EOS_ID_0`, and `TARGET_EOS_ID_1` from the selected profile. It requires
`tokenizer.json` and `tokenizer_config.json`; optional `special_tokens_map.json`,
`config.json` (including `text_config`), and `generation_config.json` corroborate
the identity when present. Token strings must resolve in the actual tokenizer
vocabulary. Conflicting maps, duplicate JSON keys, invalid/out-of-range IDs, or
absence of authoritative BOS/EOS declarations are errors, not skipped comparisons.

BOS must match exactly across supplied metadata. EOS needs different semantics:
the tokenizer's canonical EOS must be in the profile's stop set, and every supplied
EOS-ID list must overlap that set. Additional profile stop IDs must be declared
special tokens. A generation config may contain extra turn terminators, so equality
of complete EOS lists is deliberately **not** required. This checks identity and
compatibility, not completeness of every serving protocol's stop policy. For example,
Muse's generation configuration includes an additional eot marker beyond canonical
EOS; Gemma's profile includes both eos and turn-end. This gate does not rewrite either.

On a build, the checker atomically publishes the exact profile snapshot it checked
as `lib/model_config.mojo`. Rejections leave the existing config and binaries alone.
The known-bad OLMo `TARGET_BOS_ID=1` must fail with a message showing that its token
is a literal quote and artifact BOS is 100257; a temporary corrected profile must pass.
The tests exercise both against the local artifact without modifying the live profile.

**Boundary:** this is a source-input build gate, not proof about an already-loaded
binary. A deployment check must bind expected token IDs to that binary (compiled
metadata or a hash-verified build manifest), not infer them from current profile source
or a model name. It must also validate its actual tokenizer directory. Direct compiler
invocations bypass this build wrapper; they are not covered by its preflight.

## The identity check

One `.so` per model means loading the wrong weights *cannot* work. The only question is
whether it fails usefully. Before this existed it failed like this:

```
[engine] model = Muse-Glimmer-30B-text  D = 6656 ... vocab = 202048     <- COMPILED-IN
[WARN] Empty or missing nvfp4: .../gemma-4-31b/lm_head_weight.nvfp4
AssertionError: nomos_init failed
```

Nothing there says *wrong kernel for these weights*. Now:

```bash
python3 tools/check_model_identity.py --so libnomos_kernel.so --weights ~/nomos_data/<dir>
```

Two design choices worth keeping:

1. **Ask the kernel, not the filename.** The profile is read from the compiled `.so` via
   the `nomos_model_id()` export. Filenames are the thing most likely to be stale.
2. **Compare geometry, not names.** Names are cosmetic and inconsistent across converters
   (`Muse-Glimmer-30B-text` / `muse-glimmer-30b` / `gemma-4-31b-it`), and the Muse
   manifest carries no `model` key at all — a name-based check would have been
   INCONCLUSIVE on the exact pair it exists to separate. The `id -> geometry` map is
   parsed from the tracked profiles, so it cannot drift from the compiled kernel.

Anything it cannot fully compare reports **INCONCLUSIVE** (exit 2), never OK. A check
that reports success when it could not actually check is decoration.

## Adding a profile

1. Copy an existing `lib/model_profiles/*.mojo`, set the constants, take the next
   `MODEL_ID`, and set `PROFILE_COMPLETE = 0`.
2. Port whatever *architecture* the model needs (see below) behind the profile.
3. Flip `PROFILE_COMPLETE = 1` only once the model passes its gates.

`PROFILE_COMPLETE` exists because geometry is the easy half. A profile that is
geometrically right and architecturally incomplete builds, runs, and emits **wrong
tokens** — a failure that manufactures plausible output, which is the worst shape we
have. The build refuses those unless `FORCE_INCOMPLETE=1`.

## Status

| profile | complete | notes |
|---|---|---|
| `gemma4` | yes | 12/12 lossless, base 22.9 tok/s, weighted spec 52.1 |
| `muse` | yes | complete for the kernel: `PROFILE_COMPLETE = 1`, and the header of `lib/model_profiles/muse.mojo` records it verified on 2026-08-15 on the 5090 (6/6 identity=EXACT, base ~28.9 tok/s). The serve policy (the ATEM protocol) was completed on 2026-08-16. The KV session store supports this profile (below). |
| `olmo3` | yes | NVFP4 serve path (profile 4, standard attention, no GDN). Rep-penalty 1.15 serve default. Q4 arm blocked on a missing `lm_head_weight.q4`, so GB10 also serves it NVFP4. |

**Muse and the KV session store: this profile only.** The tiered KV session store (GPU → pinned RAM →
NVMe; `docs/KV_SESSION_STORE_DESIGN.md`, `docs/RUNBOOK.md` §7) supports the `muse` profile
(`MODEL_ID == 2`) and no other. It is gated on `MODEL_ID` at compile time: a `gemma4`, `olmo3` or
`qwen3_5` build answers `nomos_session_supported() == -10` and refuses `NOMOS_KVSTORE=1` at init with
code -10. `tools/test_kv_session_guard.py` checks this against a real OLMo build. Other models would
need their own state saved first (Qwen's GDN pools, for one), which this version does not do. A kernel
built before the store existed has no store at all and serves as it always did.

What Muse needed beyond geometry has all landed on main. The header of `lib/model_profiles/muse.mojo` says
"ALL of it has landed (61c0538)" (that commit is in this branch's history):

- the **attention gate** (`attn *= sigmoid(gate_proj(layer_input))`): `GemmaEngine.d_attn_gw`, `lib/gemma4_engine.mojo:136`
- **NoPE** on full layers: `FULL_LAYERS_NOPE` in the profile
- the DFlash drafter's own comptime shape: `DRAFTER_MLP` / `DRAFTER_TAPS` / `DRAFTER_Q_HEADS`
- logit softcapping and the output multiplier: `TARGET_SOFTCAP` / `TARGET_OUTPUT_MULTIPLIER`

This section used to list the branch's **+7.7% decode work** (`LMHEAD_R4`, grouped Q/K/V, chunked verify
readout) as blocked on those. It is in the tree and on by default: `NOMOS_LMHEAD_R4` and
`NOMOS_FP4_GROUPED_QKV` in `lib/engine_decode.mojo`, `NOMOS_VERIFY_BATCH_READOUT` in `lib/engine_prefill.mojo`,
and the `d_decode_token` field of `GemmaEngine` (`lib/gemma4_engine.mojo:223`). The +7.7% is the old branch's
figure; it was not re-measured here.

## Why not one `.so` for both

Runtime geometry is the obvious next step and may well be right — the weights manifest
already carries `hidden` / `layers` / `vocab`, so the data is there. But the constants
that appear as static layout shapes become dynamic, and that is a **perf question, not a
design question**. Measure before committing: the gate is Gemma-4's acceptance rates,
which are deterministic under greedy + deterministic cuBLAS and therefore prove whether
the arithmetic moved (see below).

## Verifying a change here is inert

A refactor in this area will not show up in a losslessness check — that compares
spec-decode against base-decode, and a change to shared arithmetic moves **both**. Use a
gate whose reference sits *outside* the change:

```
L1 lossless   12/12
acc_rate      agentic .317 | code .534 | prose .298 | structured .613
tok/cyc       2.82 | 4.06 | 2.71 | 4.50
E             3.21
```

Identical acceptance means the numerics did not move. That is how the profile refactor
was verified — including that rewriting Gemma-4's full-layer rule from `layer % 6 == 5`
to the shared backward form `(TOTAL_LAYERS - 1 - layer) % 6 == 0` selects the same layers.
