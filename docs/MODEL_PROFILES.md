# Model profiles — one repo, one main, either model

## The problem this solves

Until 2026-08-15, serving a second model meant a **branch per model**.
`an earlier branch` did not add Muse support; it *replaced* Gemma-4:

| alias | Gemma-4-31B | Muse-Glimmer-30B |
|---|---|---|
| `D` | 5376 | 6656 |
| `NKV` / `HD` | 16 / 256 | 2 / 128 |
| `FF` | 21504 | 19968 |
| `VOCAB` | 262144 | 202048 |
| `TOTAL_LAYERS` | 60 | 52 |
| `SLIDING_WINDOW` | 1024 | 2048 |

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
bash refresh_build.sh --model gemma4      # default
bash refresh_build.sh --model muse        # refused while PROFILE_COMPLETE = 0
```

Publishing the generic name last is deliberate: building straight to
`libnomos_kernel.so` means a **failed build deletes the working kernel**, which happened
on 2026-08-15.

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
| `muse` | **no** | geometry only on main; architecture still on `an earlier branch` |

Muse still needs, all currently on the branch:

- the **attention gate** (`attn *= sigmoid(gate_proj(layer_input))`, `GemmaEngine.d_attn_gw`)
- **NoPE** on full layers (`layer_rope_theta = 0`)
- the DFlash drafter's own comptime shape (`DFLASH_HIDDEN` / `MLP` / `VOCAB`)
- logit softcapping / `output_multiplier` differences

Those also block the branch's **+7.7% decode work** (`LMHEAD_R4`, grouped Q/K/V, chunked
verify readout) from reaching main: `engine_decode` / `engine_prefill` reference
`d_attn_gw` and `d_decode_token`, fields that exist only on the Muse engine.

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
