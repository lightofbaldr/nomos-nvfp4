#!/usr/bin/env python3
"""D3 spec-loop CYCLE PROFILER (host-side per-phase timers).

Wraps the exact spec_generate cycle from eagle3_spec_loop.py, adding
perf_counter_ns timers around each FFI phase to decompose the ~84ms cycle:

  draft      -> nomos_eagle3_draft (k eagle3 draft-head forwards)
  verify     -> nomos_eagle3_verify_fused (M=k+1 target forward + serial readout)
  accept     -> host argmax over (k+1)xVOCAB + longest-prefix accept
  kvset      -> nomos_kv_set_len (host rollback)
  seltap     -> nomos_eagle3_select_verify_tap
  gettaps    -> nomos_eagle3_get_verify_taps loop (D2H tap copies)
  commit     -> nomos_eagle3_commit (drafter KV rollback + replay)

Aggregated across all cycles of a steady warm+scored run on the 12-prompt bar.
Reports ms/cycle + %/cycle per phase, plus tok/s. Keep NOMOS_VERIFY_PROFILE=0
here (clean host timing); run separately with it =1 for the verify-internal split.
"""
import ctypes, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.environ.get("SO_PATH", os.path.join(os.path.dirname(HERE), "libnomos_kernel.so"))
WTS = os.environ["WEIGHTS"]
EAGLE_DIR = os.environ["EAGLE_DIR"]
NTOK = int(os.environ.get("PERF_NTOK", "96"))
REPEAT = int(os.environ.get("PERF_REPEAT", "2"))
K = int(os.environ.get("SPEC_K", "3"))
VOCAB = 262144
D = 5376
TAPS = 3 * D

assert WTS.endswith("/") and EAGLE_DIR.endswith("/"), "WEIGHTS/EAGLE_DIR need trailing slash"

lib = ctypes.CDLL(SO)
c = ctypes
sigs = {
    "nomos_init": (c.c_void_p, [c.c_char_p]),
    "nomos_prefill": (c.c_int32, [c.c_void_p, c.c_void_p, c.c_int32, c.c_void_p]),
    "nomos_decode_step": (c.c_int32, [c.c_void_p, c.c_int32, c.c_void_p]),
    "nomos_reset_kv": (c.c_int32, [c.c_void_p]),
    "nomos_kv_cache_len": (c.c_int32, [c.c_void_p]),
    "nomos_kv_set_len": (c.c_int32, [c.c_void_p, c.c_int32]),
    "nomos_eagle3_load": (c.c_int32, [c.c_void_p, c.c_char_p]),
    "nomos_eagle3_reset": (c.c_int32, [c.c_void_p]),
    "nomos_eagle3_draft": (c.c_int32, [c.c_void_p, c.c_int32, c.c_int32, c.c_void_p]),
    "nomos_eagle3_verify_fused": (c.c_int32, [c.c_void_p, c.c_void_p, c.c_int32, c.c_int32, c.c_void_p]),
    "nomos_eagle3_select_verify_tap": (c.c_int32, [c.c_void_p, c.c_int32]),
    "nomos_eagle3_commit": (c.c_int32, [c.c_void_p, c.c_void_p, c.c_void_p, c.c_int32, c.c_int32, c.c_void_p]),
    "nomos_eagle3_get_taps": (c.c_int32, [c.c_void_p, c.c_void_p]),
    "nomos_eagle3_get_verify_taps": (c.c_int32, [c.c_void_p, c.c_int32, c.c_void_p]),
}
for n, (r, a) in sigs.items():
    f = getattr(lib, n); f.restype = r; f.argtypes = a

bar = json.load(open(os.path.join(HERE, "gold_bar_tokens.json")))
t0 = time.time()
ht = lib.nomos_init(WTS.encode())
assert ht, "nomos_init failed"
rc = lib.nomos_eagle3_load(ht, EAGLE_DIR.encode())
assert rc == 0, f"nomos_eagle3_load rc={rc}"
print(f"init {time.time()-t0:.0f}s | K={K} NTOK={NTOK}", flush=True)

logits = np.zeros(VOCAB, np.float32)
vlogits = np.zeros((K + 1, VOCAB), np.float32)
draft_ids = np.zeros(K, np.int32)
seed_tap = np.zeros(TAPS, np.float32)
next_tap = np.zeros(TAPS, np.float32)
vtaps = np.zeros((K + 1, TAPS), np.float32)

ns = time.perf_counter_ns
PH = ["draft", "verify", "accept", "kvset", "seltap", "gettaps", "commit"]


def spec_generate_prof(ids, ntok, k=K, acc=None):
    lib.nomos_reset_kv(ht)
    lib.nomos_eagle3_reset(ht)
    lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
    tok = int(np.argmax(logits))
    out = [tok]
    L = lib.nomos_kv_cache_len(ht)
    lib.nomos_eagle3_get_taps(ht, seed_tap.ctypes.data)
    cycles = 0; accepted = 0
    while len(out) < ntok:
        r = ntok - len(out)
        k_eff = min(k, r - 1)
        if k_eff <= 0:
            lib.nomos_decode_step(ht, tok, logits.ctypes.data)
            out.append(int(np.argmax(logits)))
            break
        t = ns(); nd = lib.nomos_eagle3_draft(ht, tok, k_eff, draft_ids.ctypes.data); acc["draft"] += ns() - t
        drafts = [int(x) for x in draft_ids[:nd]]
        rows = np.array([tok] + drafts, np.int32)
        n_rows = nd + 1
        t = ns(); lib.nomos_eagle3_verify_fused(ht, rows.ctypes.data, n_rows, L, vlogits.ctypes.data); acc["verify"] += ns() - t
        t = ns()
        tgt = [int(np.argmax(vlogits[i])) for i in range(n_rows)]
        num_acc = 0
        while num_acc < nd and drafts[num_acc] == tgt[num_acc]:
            num_acc += 1
        next_tok = tgt[num_acc]
        committed = num_acc + 1
        acc["accept"] += ns() - t
        t = ns(); lib.nomos_kv_set_len(ht, L + 1 + num_acc); acc["kvset"] += ns() - t
        t = ns(); lib.nomos_eagle3_select_verify_tap(ht, num_acc); acc["seltap"] += ns() - t
        t = ns()
        for i in range(num_acc):
            lib.nomos_eagle3_get_verify_taps(ht, i, vtaps[i].ctypes.data)
        lib.nomos_eagle3_get_verify_taps(ht, num_acc, next_tap.ctypes.data)
        acc["gettaps"] += ns() - t
        prefix_tokens = np.array([tok] + drafts[:num_acc], np.int32)
        prefix_taps = np.ascontiguousarray(np.concatenate([seed_tap[None, :], vtaps[:num_acc]], axis=0))
        t = ns(); lib.nomos_eagle3_commit(ht, prefix_tokens.ctypes.data, prefix_taps.ctypes.data,
                                          committed, next_tok, next_tap.ctypes.data); acc["commit"] += ns() - t
        out.extend(drafts[:num_acc]); out.append(next_tok)
        cycles += 1; accepted += num_acc
        L = L + 1 + num_acc; tok = next_tok; seed_tap[:] = next_tap
    return out[:ntok], cycles, accepted


acc = {p: 0 for p in PH}
tot_cycles = 0; tot_accepted = 0; tot_dec_s = 0.0; tot_toks = 0
for rep in range(REPEAT):
    scored = rep == REPEAT - 1
    if scored:
        acc = {p: 0 for p in PH}; tot_cycles = 0; tot_accepted = 0; tot_dec_s = 0.0; tot_toks = 0
    for p in bar["prompts"]:
        ids = np.array(p["ids"], dtype=np.int32)
        wall0 = ns()
        out, cyc, na = spec_generate_prof(ids, NTOK, acc=acc)
        wall = ns() - wall0
        if scored:
            tot_cycles += cyc; tot_accepted += na
            tot_dec_s += wall / 1e9; tot_toks += len(out) - 1  # minus prefill seed
    tag = "SCORED" if scored else "warm"
    print(f"[{tag}] done", flush=True)

phase_ns = sum(acc.values())
print("\n== HOST PHASE BREAKDOWN (scored, %d prompts, %d cycles) ==" % (len(bar["prompts"]), tot_cycles))
print(f"{'phase':>10} {'ms/cycle':>10} {'% cycle':>9} {'total_s':>9}")
cyc = max(tot_cycles, 1)
for p in PH:
    mspc = acc[p] / cyc / 1e6
    pct = 100 * acc[p] / max(phase_ns, 1)
    print(f"{p:>10} {mspc:10.3f} {pct:8.1f}% {acc[p]/1e9:9.3f}")
print(f"{'SUM':>10} {phase_ns/cyc/1e6:10.3f} {100.0:8.1f}% {phase_ns/1e9:9.3f}")
mean_acc = tot_accepted / cyc
tps = tot_toks / max(tot_dec_s, 1e-9)
print(f"\ncycle wall (host-measured) ~ {phase_ns/cyc/1e6:.2f} ms | mean_acc/cycle {mean_acc:.3f} "
      f"| tok/cycle {(tot_accepted+tot_cycles)/cyc:.3f} | overall tok/s {tps:.2f}")
