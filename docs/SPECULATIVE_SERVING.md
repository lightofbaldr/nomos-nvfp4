# Speculative decoding on the serve endpoint

The OpenAI-compatible serve (`pyhost/serve.py`) can drive **DFlash block
speculative decoding** directly — the same lossless spec path the benchmark
harness uses, now on `/v1/chat/completions`. It is **off by default**; base
greedy/sampled decode is unchanged.

## Enabling it

Set two environment variables before starting the serve:

```bash
export NOMOS_SERVE_SPEC=1
export DFLASH_DIR=/path/to/dflash-gemma-4-31b-flat/   # TRAILING SLASH REQUIRED
export SPEC_VB=8                                        # verify-block rows (K = VB-1 drafts/cycle)
pyhost/serve_nomos.sh up
```

On startup the serve logs `DFlash speculative decode enabled (vb=…)`. If the
drafter fails to load, the endpoint stays up on **base decode** and logs the
degraded mode — it never silently serves a broken drafter.

## What runs spec vs base

A request uses the speculative path only when **all** hold:

- `NOMOS_SERVE_SPEC=1` and a drafter loaded at startup,
- `temperature <= 0` (greedy — spec losslessness is defined for greedy), and
- no grammar/tool constraint on the request.

Sampling (`temperature > 0`) and grammar-constrained / tool-call requests
transparently fall back to base decode. Thinking-channel suppression, tool-call
parsing, and streaming format are shared — spec only changes the token *source*.

## Losslessness

The speculative stream is **bit-identical** to base greedy decode by
construction: each cycle drafts a block, the target verifies every row in one
fused pass, the longest draft prefix matching the target's own argmax is
accepted, and the target's correction token continues the stream. Verify equals
decode because the Q4 matmul epilogue accumulates in order-independent Q14
fixed-point (see *Correctness by construction* in the README).

Two gates enforce this:

- `tools/spec_serve_lossless_gate.py` — in-process: `spec_generate_stream` ==
  base greedy, token-for-token, across the bar prompts.
- `tools/serve_spec_http_gate.py` — end-to-end: base vs spec through
  `/v1/chat/completions`, byte-identical assembled SSE content.

Both must pass byte-exact for any change to the spec path.

## Config notes

- **`DFLASH_DIR` must end with `/`** — the loader asserts it.
- **`NOMOS_KV_REUSE=0`** — spec re-anchors the KV cache each cycle; reused INT4
  KV is not byte-identical and corrupts multi-turn context.
- `SPEC_VB` caps drafts verified per cycle (`K = VB-1`); the drafter always
  proposes a 16-row block. Start at 8 and sweep up where acceptance is high.
- Per-architecture: GB10 (Q4_0) and discrete Blackwell (NVFP4 W4A4) both drive
  the same serve spec path; the precision env differs (see `BUILD.md`).
