#!/usr/bin/env python3
"""SPEC_K sweep driver — loads the 31B NVFP4 model + eagle adapter ONCE, then
runs the 12-prompt gold bar at each K in SPEC_KS (comma list), so we pay the
model-load cost a single time. spec_generate() is copied VERBATIM from
tools/eagle3_spec_loop.py (the validated fast-verify loop); only the buffers
that depend on K are reallocated per K.

Per K it records per-prompt spec-decode tok/s (SCORED_REPS scored samples after
WARM_REPS warmups), per-bucket median tok/s + acceptance + mean-accepted-len,
weighted tok/s, and L1 lossless (bit-identical vs BASE_IDS base-decode ids).

Env: WEIGHTS, EAGLE_DIR, SO_PATH, BASE_IDS (L1 ref), PERF_NTOK (default 96),
     SPEC_KS (default "2,3,4,5"), WARM_REPS (default 1), SCORED_REPS (default 2),
     RESULTS_OUT (json dump path).
"""
import ctypes, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.environ.get("SO_PATH", os.path.join(os.path.dirname(HERE), "libnomos_kernel.so"))
WTS = os.environ["WEIGHTS"]
EAGLE_DIR = os.environ["EAGLE_DIR"]
NTOK = int(os.environ.get("PERF_NTOK", "96"))
KS = [int(x) for x in os.environ.get("SPEC_KS", "2,3,4,5").split(",")]
WARM_REPS = int(os.environ.get("WARM_REPS", "1"))
SCORED_REPS = int(os.environ.get("SCORED_REPS", "2"))
BASE_IDS = os.environ.get("BASE_IDS", "")
RESULTS_OUT = os.environ.get("RESULTS_OUT", "")
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
assert rc == 0, f"nomos_eagle3_load rc={rc} (NOMOS_EAGLE=1? blobs present?)"
INIT_S = time.time() - t0
print(f"init {INIT_S:.0f}s | KS={KS} NTOK={NTOK} warm={WARM_REPS} scored={SCORED_REPS} "
      f"eagle_mode={os.environ.get('NOMOS_EAGLE_MODE','1(nvfp4)')} SO={os.path.basename(SO)}",
      flush=True)

# K-independent buffers
logits = np.zeros(VOCAB, np.float32)
seed_tap = np.zeros(TAPS, np.float32)
next_tap = np.zeros(TAPS, np.float32)
# K-dependent buffers (reallocated per K in the sweep loop)
vlogits = None
draft_ids = None
vtaps = None


def softcap(x, cap=30.0):
    return (cap * np.tanh(x.astype(np.float64) / cap)).astype(np.float32)


# ── spec_generate: VERBATIM from tools/eagle3_spec_loop.py (buffers are globals,
#    resized per K before each batch of calls) ────────────────────────────────
def spec_generate(ids, ntok, k, trace=False):
    """Free-running spec decode: returns (out_tokens[ntok], cycle stats, decode_seconds)."""
    lib.nomos_reset_kv(ht)
    lib.nomos_eagle3_reset(ht)
    lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
    tok = int(np.argmax(logits))
    out = [tok]
    L = lib.nomos_kv_cache_len(ht)
    assert L == len(ids), f"cache_len {L} != prompt {len(ids)}"
    rc = lib.nomos_eagle3_get_taps(ht, seed_tap.ctypes.data)  # tap of the row that produced tok
    assert rc == 0
    stats = {"cycles": 0, "drafted": 0, "accepted": 0, "acc_hist": [0] * (k + 1),
             "slot_acc": [0] * k, "slot_n": [0] * k, "fallback_steps": 0}
    td0 = time.time()
    while len(out) < ntok:
        r = ntok - len(out)
        k_eff = min(k, r - 1)
        if k_eff <= 0:
            lib.nomos_decode_step(ht, tok, logits.ctypes.data)
            out.append(int(np.argmax(logits)))
            stats["fallback_steps"] += 1
            break
        nd = lib.nomos_eagle3_draft(ht, tok, k_eff, draft_ids.ctypes.data)
        assert nd == k_eff, f"draft rc={nd}"
        drafts = [int(x) for x in draft_ids[:nd]]
        rows = np.array([tok] + drafts, np.int32)
        n_rows = nd + 1
        rc = lib.nomos_eagle3_verify_fused(ht, rows.ctypes.data, n_rows, L, vlogits.ctypes.data)
        assert rc == 0, f"verify rc={rc}"
        tgt = [int(np.argmax(vlogits[i])) for i in range(n_rows)]
        num_acc = 0
        while num_acc < nd and drafts[num_acc] == tgt[num_acc]:
            num_acc += 1
        next_tok = tgt[num_acc]
        committed = num_acc + 1
        assert committed == num_acc + 1 <= n_rows and committed <= r, \
            f"commit invariant broke: num_acc={num_acc} n_rows={n_rows} r={r}"
        lib.nomos_kv_set_len(ht, L + 1 + num_acc)
        rc = lib.nomos_eagle3_select_verify_tap(ht, num_acc)
        assert rc == 0
        for i in range(num_acc):
            rc = lib.nomos_eagle3_get_verify_taps(ht, i, vtaps[i].ctypes.data)
            assert rc == 0
        rc = lib.nomos_eagle3_get_verify_taps(ht, num_acc, next_tap.ctypes.data)
        assert rc == 0
        prefix_tokens = np.array([tok] + drafts[:num_acc], np.int32)
        prefix_taps = np.ascontiguousarray(
            np.concatenate([seed_tap[None, :], vtaps[:num_acc]], axis=0))
        rc = lib.nomos_eagle3_commit(ht, prefix_tokens.ctypes.data, prefix_taps.ctypes.data,
                                     committed, next_tok, next_tap.ctypes.data)
        assert rc == 0, f"commit rc={rc}"
        if trace:
            print(f"  cycle {stats['cycles']:>3}: L={L} drafts={drafts} tgt={tgt[:nd]} "
                  f"acc={num_acc} next={next_tok}", flush=True)
        out.extend(drafts[:num_acc])
        out.append(next_tok)
        for i in range(nd):
            stats["slot_n"][i] += 1
            if i < num_acc:
                stats["slot_acc"][i] += 1
        stats["cycles"] += 1
        stats["drafted"] += nd
        stats["accepted"] += num_acc
        stats["acc_hist"][num_acc] += 1
        L = L + 1 + num_acc
        tok = next_tok
        seed_tap[:] = next_tap
    td = time.time() - td0
    return out[:ntok], stats, td


base_ref = json.load(open(BASE_IDS)) if BASE_IDS else None
sweep = {}  # K -> report

# ── per-bucket-K measured mode (adaptive K): PER_BUCKET_K='{"agentic":3,...}' ──
PBK = os.environ.get("PER_BUCKET_K", "")
if PBK:
    pbk = json.loads(PBK)
    kmax = max(pbk.values())
    vlogits = np.zeros((kmax + 1, VOCAB), np.float32)
    draft_ids = np.zeros(kmax, np.int32)
    vtaps = np.zeros((kmax + 1, TAPS), np.float32)
    print(f"\n########## PER-BUCKET-K (measured) {pbk} ##########", flush=True)
    tps_samples = {p["id"]: [] for p in bar["prompts"]}
    last_ids, last_stats = {}, {}
    for rep in range(WARM_REPS + SCORED_REPS):
        scored = rep >= WARM_REPS
        tag = "SCORED" if scored else "warm"
        for p in bar["prompts"]:
            k = pbk[p["bucket"]]
            ids = np.array(p["ids"], dtype=np.int32)
            out, st, td = spec_generate(ids, NTOK, k=k)
            dec_tps = (NTOK - 1) / td
            if scored:
                tps_samples[p["id"]].append(dec_tps)
                last_ids[p["id"]] = out
                last_stats[p["id"]] = st
            print(f"[PBK k{k} {tag}] {p['id']:>8} [{p['bucket']:>12}] "
                  f"spec-decode={dec_tps:6.2f} tok/s", flush=True)
    pid2bucket = {p["id"]: p["bucket"] for p in bar["prompts"]}
    prompt_tps = {}
    for pid, s in tps_samples.items():
        ss = sorted(s)
        prompt_tps[pid] = ss[len(ss)//2] if len(ss) % 2 else 0.5*(ss[len(ss)//2-1]+ss[len(ss)//2])
    n_match = 0
    for pid, seq in last_ids.items():
        n_match += int(seq == base_ref[pid][:len(seq)]) if base_ref else 0
    print(f"\n-- PER-BUCKET-K L1: {n_match}/{len(last_ids)} bit-identical vs base")
    weighted = 0.0
    print(f"{'bucket':>12} {'K':>2} {'med tok/s':>10} (weight)")
    for b in bar["weights"]:
        pids = [pid for pid in pid2bucket if pid2bucket[pid] == b]
        tvals = sorted(prompt_tps[pid] for pid in pids)
        med = tvals[len(tvals)//2] if len(tvals) % 2 else 0.5*(tvals[len(tvals)//2-1]+tvals[len(tvals)//2])
        weighted += bar["weights"][b] * med
        print(f"{b:>12} {pbk[b]:>2} {med:10.2f}  ({bar['weights'][b]})")
    print(f"{'WEIGHTED(PBK)':>15} {weighted:10.2f} tok/s  | clears 35? "
          f"{'YES' if weighted >= 35 else 'NO (short %.2f)' % (35-weighted)}", flush=True)
    sys.exit(0)

for k in KS:
    # reallocate K-dependent buffers
    vlogits = np.zeros((k + 1, VOCAB), np.float32)
    draft_ids = np.zeros(k, np.int32)
    vtaps = np.zeros((k + 1, TAPS), np.float32)

    print(f"\n########## SPEC_K = {k} ##########", flush=True)
    # per-prompt: list of scored tok/s samples; last scored rep's ids/stats kept for L1/L2
    tps_samples = {p["id"]: [] for p in bar["prompts"]}
    last_ids = {}
    last_stats = {}
    total_reps = WARM_REPS + SCORED_REPS
    for rep in range(total_reps):
        scored = rep >= WARM_REPS
        tag = "SCORED" if scored else "warm"
        for p in bar["prompts"]:
            ids = np.array(p["ids"], dtype=np.int32)
            out, st, td = spec_generate(ids, NTOK, k=k)
            dec_tps = (NTOK - 1) / td
            acc_rate = st["accepted"] / max(st["drafted"], 1)
            mean_acc = st["accepted"] / max(st["cycles"], 1)
            if scored:
                tps_samples[p["id"]].append(dec_tps)
                last_ids[p["id"]] = out
                last_stats[p["id"]] = st
            print(f"[k{k} {tag}] {p['id']:>8} [{p['bucket']:>12}] "
                  f"spec-decode={dec_tps:6.2f} tok/s | acc {st['accepted']}/{st['drafted']} "
                  f"({acc_rate:.2f}) mean_acc/cycle={mean_acc:.2f}", flush=True)

    # per-prompt scored median (of SCORED_REPS)
    pid2bucket = {p["id"]: p["bucket"] for p in bar["prompts"]}
    prompt_tps = {}
    for pid, s in tps_samples.items():
        ss = sorted(s)
        prompt_tps[pid] = ss[len(ss) // 2] if len(ss) % 2 else 0.5 * (ss[len(ss) // 2 - 1] + ss[len(ss) // 2])

    # L1 lossless (scored ids vs base decode)
    l1_match = None
    l1_detail = []
    if base_ref is not None:
        n_match = 0
        for pid, seq in last_ids.items():
            refseq = base_ref[pid][:len(seq)]
            m = seq == refseq
            n_match += int(m)
            if not m:
                di = next(i for i in range(min(len(seq), len(refseq))) if seq[i] != refseq[i])
                l1_detail.append(f"{pid}@{di}:spec={seq[di]} base={refseq[di]}")
        l1_match = n_match

    # per-bucket aggregation
    buckets = sorted(set(pid2bucket.values()))
    per_bucket = {}
    for b in buckets:
        pids = [pid for pid in pid2bucket if pid2bucket[pid] == b]
        tvals = sorted(prompt_tps[pid] for pid in pids)
        med = tvals[len(tvals) // 2] if len(tvals) % 2 else 0.5 * (tvals[len(tvals) // 2 - 1] + tvals[len(tvals) // 2])
        acc = float(np.mean([last_stats[pid]["accepted"] / max(last_stats[pid]["drafted"], 1) for pid in pids]))
        mal = float(np.mean([last_stats[pid]["accepted"] / max(last_stats[pid]["cycles"], 1) for pid in pids]))
        per_bucket[b] = {"median_tps": med, "acc_rate": acc, "mean_acc_len": mal,
                         "prompt_tps": {pid: prompt_tps[pid] for pid in pids}}

    weighted = sum(bar["weights"][b] * per_bucket[b]["median_tps"] for b in bar["weights"])
    sweep[k] = {"per_bucket": per_bucket, "weighted": weighted,
                "l1_match": l1_match, "l1_detail": l1_detail, "n_prompts": len(last_ids)}

    print(f"\n-- k={k} L1: {l1_match}/{len(last_ids)} bit-identical vs base "
          + ("(" + "; ".join(l1_detail) + ")" if l1_detail else "LOSSLESS"))
    print(f"{'bucket':>12} {'med tok/s':>10} {'acc_rate':>9} {'mean_acc':>9} (weight)")
    for b in bar["weights"]:
        pb = per_bucket[b]
        print(f"{b:>12} {pb['median_tps']:10.2f} {pb['acc_rate']:9.3f} {pb['mean_acc_len']:9.3f}  ({bar['weights'][b]})")
    print(f"{'WEIGHTED':>12} {weighted:10.2f} tok/s   | base NVFP4 ref 21.45-21.7", flush=True)

# ── cross-K summary + per-bucket-optimal-K projection ────────────────────────
print("\n\n==================== SWEEP SUMMARY ====================")
buckets_order = list(bar["weights"].keys())
hdr = f"{'K':>3} | " + " ".join(f"{b[:9]:>9}" for b in buckets_order) + f" | {'WEIGHT':>7} {'L1':>5}"
print(hdr)
print("-" * len(hdr))
for k in KS:
    row = f"{k:>3} | " + " ".join(f"{sweep[k]['per_bucket'][b]['median_tps']:9.2f}" for b in buckets_order)
    row += f" | {sweep[k]['weighted']:7.2f} {str(sweep[k]['l1_match'])+'/'+str(sweep[k]['n_prompts']):>5}"
    print(row)

# best global K (L1-lossless only)
valid = [k for k in KS if sweep[k]["l1_match"] == sweep[k]["n_prompts"]]
if valid:
    best_global = max(valid, key=lambda k: sweep[k]["weighted"])
    print(f"\nBest GLOBAL K (L1-lossless): K={best_global}  weighted={sweep[best_global]['weighted']:.2f} tok/s")

# per-bucket optimal K (projection: pick K maximizing each bucket's median tok/s, L1-lossless K only)
print("\nPer-bucket-optimal-K (PROJECTION, among L1-lossless K):")
opt = {}
proj_weighted = 0.0
for b in buckets_order:
    cand = [(k, sweep[k]["per_bucket"][b]["median_tps"]) for k in valid]
    bk, btps = max(cand, key=lambda x: x[1])
    opt[b] = (bk, btps)
    proj_weighted += bar["weights"][b] * btps
    print(f"  {b:>12}: K={bk}  {btps:.2f} tok/s  (weight {bar['weights'][b]})")
print(f"  => PROJECTED per-bucket-optimal weighted = {proj_weighted:.2f} tok/s")
print(f"  clears 35? {'YES' if proj_weighted >= 35 else 'NO (short by %.2f)' % (35 - proj_weighted)}")

if RESULTS_OUT:
    json.dump(sweep, open(RESULTS_OUT, "w"), indent=2)
    print("\nwrote", RESULTS_OUT, flush=True)
