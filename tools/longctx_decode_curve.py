#!/usr/bin/env python3
"""Decode-vs-context curve: LINEAR-CACHE vs RING-CACHE cost at depth.

WHY (2026-08-04). Every long-context number we own is 24GB-shaped, so this measures the curve on a
97GB card and compares the two KV *storage* strategies at depth.

READ THIS BEFORE INTERPRETING THE ARMS. `NOMOS_KV_SWA` does NOT change attention semantics. In
Gemma-4, sliding layers (5 of every 6) attend only the last SLIDING_WINDOW=1024 keys and full
layers (layer%6==5) attend everything -- that is the MODEL (from the HF config `sliding_window`),
and `gemma4_engine.mojo:57-62` states it as an architectural bound, not a toggle. `kv_swa` gates
exactly two storage quantities:
    layer_cache_cap  = SLIDING_WINDOW if kv_swa else max_seq   (engine:375)
    deq_slide_pitch  = SLIDING_WINDOW if kv_swa else max_seq   (engine:334)
So both arms compute the SAME attention mask. SWA=0 merely stores and dequantizes sliding-layer
history that the mask can never read; SWA=1 keeps exactly the reachable window in a ring.

Therefore:
  - SWA=0 slower  => the price of REDUNDANT storage/dequant, NOT "the cost of exact attention".
  - equal speed   => the ring still buys memory capacity; it is not "buying nothing" on a big card.
  - TOKEN DIVERGENCE BETWEEN ARMS IS A BUG, not a fidelity trade. Semantics are identical, so this
    script records decoded IDs per row and the arms must match. (Corrected by kvasir_codex after an
    earlier revision of this file wrongly called SWA=0 "full fidelity".)

SECOND CONFOUND, and it is the bigger one: SWA ALSO SWITCHES THE PREFILL ALGORITHM.
    long_swa = self.kv_swa and S > SLIDING_WINDOW            (engine_prefill.mojo:141)
    d_scores_big = cuda_malloc(NH*S*S*4) if (use_batched and not long_swa)   (:145)
With SWA=0 prefill materializes a full [NH,S,S] fp32 score matrix and does one square batched
attention. With SWA=1 at S>1024, `long_swa` disables block scratch and forces per-token attention.
So an SWA A/B at any S>1024 compares TWO PREFILL IMPLEMENTATIONS as well as two storage layouts,
and cannot attribute a difference to either on its own. Prefill time delta is square-batched vs
per-token ROUTING, not ring write-pattern cost.

Why the deep point dies under SWA=0 -- NOT the dequant scratch:
    32 * 24668^2 * 4 = 77,889,308,672 bytes = 72.5 GiB   (S = prompt 24669 minus final token)
matches the abort byte-for-byte. It is `d_scores_big`, QUADRATIC IN PROMPT LENGTH and independent
of max_seq -- PROVEN by re-running at MAX_SEQ=25600 and getting the identical byte count. The
sliding dequant scratch is only ~256 MiB (16*32768*256*2), so it is not the culprit and this run
says nothing about task #48. `cuda.mojo:26` aborts on cudaMalloc failure, which surfaces as SIGILL
rather than a clean OOM. Tightening max_seq cannot fix an NH*S^2 buffer; the fix is a chunked/
prefix prefill route.
(An earlier revision of this file blamed a "32x deq overallocation pitched at max_seq". That was
wrong -- asserted from a byte count without reading the allocation site. Corrected here.)

What it answers, per context length: prefill (TTFT) cost, base decode tok/s, VRAM actually used,
decoded token IDs for cross-arm identity, and whether output stays coherent at depth (the long bar
exists precisely because window-boundary defects are structurally unreachable below 1024 tokens).

Reuses the FFI drive pattern of tools/gold_cuda_budget.py verbatim -- same provenance discipline,
same reset/prefill/decode loop -- so the two tools are comparable by construction.

Every row is flushed as it completes: an OOM at the deepest point must not cost the shallower rows.

  EXPECTED_SHA=$(git rev-parse HEAD) SO_PATH=... WEIGHTS=... NOMOS_MAX_SEQ=32768 NOMOS_KV_SWA=0 \
  python3 tools/longctx_decode_curve.py
"""
import ctypes as c, json, os, pathlib, subprocess, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SO = pathlib.Path(os.environ.get("SO_PATH", ROOT / "libnomos_kernel.so")).resolve()
WTS = os.environ.get("WEIGHTS", os.path.expanduser("~/nomos_data/gemma-4-31b/"))
NTOK = int(os.environ.get("PERF_NTOK", "48"))
VOCAB = int(os.environ.get("NOMOS_VOCAB", "262144"))
BAR = os.environ.get("LONGBAR", str(ROOT / "tools/gold_bar_tokens_long.json"))

# Provenance, same gate as the budget tool: a number is only real for a known SHA + .so.
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
expected = os.environ.get("EXPECTED_SHA")
if expected and expected != head:
    sys.exit(f"provenance FAIL: HEAD {head} != EXPECTED_SHA {expected}")
# No shell: SO comes from the environment, and the pipe bought nothing over counting here.
nsym = sum(1 for ln in subprocess.run(["nm", "-D", str(SO)], capture_output=True, text=True)
           .stdout.splitlines() if " T nomos_" in ln)
print(f"provenance PASS sha={head} so={SO} symbols={nsym}", flush=True)
print(f"config NOMOS_KV_SWA={os.environ.get('NOMOS_KV_SWA','?')} "
      f"NOMOS_MAX_SEQ={os.environ.get('NOMOS_MAX_SEQ','?')} ntok={NTOK}", flush=True)


def vram_mb():
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True).strip().split("\n")[0])
    except Exception:
        return -1


def health(ids):
    """Cheap degeneracy metrics on decoded token IDs -- the long bar exists to catch depth defects."""
    if not ids:
        return 0, 0.0
    run = best = 1
    for a, b in zip(ids, ids[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    grams = list(zip(ids, ids[1:]))
    return best, (len(set(grams)) / len(grams) if grams else 0.0)


lib = c.CDLL(str(SO))
for name, result, args in [
    ("nomos_init", c.c_void_p, [c.c_char_p]),
    ("nomos_shutdown", None, [c.c_void_p]),
    ("nomos_prefill", c.c_int32, [c.c_void_p, c.c_void_p, c.c_int32, c.c_void_p]),
    ("nomos_decode_step_token", c.c_int32, [c.c_void_p, c.c_int32, c.c_void_p]),
    ("nomos_reset_kv", c.c_int32, [c.c_void_p]),
]:
    fn = getattr(lib, name)
    fn.restype = result
    fn.argtypes = args

bar = json.loads(pathlib.Path(BAR).read_text())
prompts = sorted(bar["prompts"], key=lambda p: len(p["ids"]))
logits = np.zeros(VOCAB, np.float32)
out_token = c.c_int32()

base_vram = vram_mb()
handle = lib.nomos_init(WTS.encode())
assert handle, "nomos_init failed"
load_vram = vram_mb()
print(f"weights resident: {load_vram - base_vram} MB (total {load_vram} MB)", flush=True)
print(f"{'ctx':>7} {'prefill_s':>10} {'tok/s':>8} {'ms/tok':>8} {'VRAM_MB':>8} {'dVRAM':>7} "
      f"{'maxrun':>7} {'d2g':>6}  id", flush=True)

rows = []
for p in prompts:
    ids = np.asarray(p["ids"], dtype=np.int32)
    n = len(ids)
    try:
        assert lib.nomos_reset_kv(handle) == 0
        t0 = time.perf_counter()
        rc = lib.nomos_prefill(handle, ids.ctypes.data, n, logits.ctypes.data)
        prefill_s = time.perf_counter() - t0
        if rc != 0:
            print(f"{n:>7} PREFILL FAILED rc={rc} -- ceiling reached", flush=True)
            break
        token = int(np.argmax(logits))
        seq = [token]
        t0 = time.perf_counter()
        for _ in range(NTOK - 1):
            if lib.nomos_decode_step_token(handle, token, c.addressof(out_token)) != 0:
                break
            token = int(out_token.value)
            seq.append(token)
        dec_s = time.perf_counter() - t0
        v = vram_mb()
        mr, d2g = health(seq)
        tps = (len(seq) - 1) / dec_s if dec_s > 0 else 0.0
        print(f"{n:>7} {prefill_s:>10.3f} {tps:>8.3f} {1000*dec_s/max(len(seq)-1,1):>8.3f} "
              f"{v:>8} {v-load_vram:>7} {mr:>7} {d2g:>6.2f}  {p.get('id','?')}", flush=True)
        # Full decoded IDs: the arms have IDENTICAL attention semantics, so cross-arm token
        # identity is a correctness GATE, not a quality comparison. Without this the run cannot
        # tell a storage win from a storage bug.
        rows.append(dict(ctx=n, prefill_s=prefill_s, tok_s=tps, vram_mb=v,
                         dvram_mb=v - load_vram, max_run=mr, distinct2=d2g,
                         id=p.get("id"), tokens=seq))
    except Exception as e:
        print(f"{n:>7} EXCEPTION {type(e).__name__}: {e} -- ceiling reached", flush=True)
        break

lib.nomos_shutdown(handle)
out = pathlib.Path(os.environ.get("LONGCTX_OUT", "/root/longctx_curve.json"))
out.write_text(json.dumps({"sha": head, "swa": os.environ.get("NOMOS_KV_SWA"),
                           "max_seq": os.environ.get("NOMOS_MAX_SEQ"), "rows": rows}, indent=2))
print(f"LONGCTX COMPLETE  rows={len(rows)}  -> {out}", flush=True)
