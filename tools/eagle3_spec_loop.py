#!/usr/bin/env python3
"""D3 — EAGLE-3 spec-decode loop on the discrete NVFP4 path (propose k=3 -> batched
verify M=k+1 -> greedy longest-prefix accept -> commit num_acc+1).

Port of the GB10 lineage loop semantics (b1ec92d / 25641de / 573faf0 / the
the eagle3-commit-ffi seam), driven from the host over the existing FFI:

  cycle (seed token `tok`, target KV len L, armed tap = row that produced tok):
    1. nomos_eagle3_draft(tok, k)           -> drafts d[0..k-1] (target-vocab ids)
    2. nomos_eagle3_verify_fused([tok, d0..d_{k-1}], k+1 rows, start_pos=L)
         -> logits[k+1, VOCAB]; target KV left at L+k+1; per-row taps cached
    3. accept: num_acc = longest prefix with d_i == argmax(row i);
       next_tok = argmax(row num_acc)  (correction at first disagreement,
       bonus if all accepted). COMMITTED THIS CYCLE = num_acc + 1.
    4. nomos_kv_set_len(L + 1 + num_acc)    -> target rollback: consumed rows =
       seed + accepted drafts ONLY (prefill-INTEGRATED append: the verify itself
       wrote those rows' KV in-context; rejected rows are trimmed, never reused)
    5. nomos_eagle3_select_verify_tap(num_acc)  -> arm the accepted-final row
    6. nomos_eagle3_commit(prefix=[tok, d0..d_{num_acc-1}],
                           prefix_taps=[seed_tap, vtap_0..vtap_{num_acc-1}],
                           next_token=next_tok, next_taps=vtap_num_acc)
       -> drafter KV rollback + exact replay of the accepted prefix from RAW
          target taps (the drafter's speculative KV is never trusted across
          cycles — KV_REUSE=0 semantics, #429 class)
    7. emit d[0..num_acc-1] + [next_tok]; tok=next_tok; L += 1+num_acc;
       seed_tap = vtap_num_acc

  VB truncation (landmine #4): k_eff = min(K, remaining-1); the final token of
  a budget-capped run falls back to a plain nomos_decode_step (exact path).

Modes (MODE env or --mode):
  smoke  — 1 prompt, few cycles, verbose per-cycle accept trace.
  seam   — the verify bit-exactness gate (landmine #5): (a) M=1-vs-M=4 fused-
           verify row logits BYTE-identical; (b) verify rows vs sequential
           decode steps: argmax identical + softcapped |diff|.
  run    — full 12-prompt bar, warm+scored: ids json (L1), per-bucket
           acceptance (L2), per-bucket weighted tok/s (L3).

Env: WEIGHTS (trailing slash), EAGLE_DIR (trailing slash), SO_PATH,
     PERF_NTOK (default 96), PERF_REPEAT (default 2), SPEC_K (default 3),
     IDS_OUT (ids json), BASE_IDS (compare target for L1),
     NOMOS_EAGLE=1 (+ NOMOS_EAGLE_MODE 1=nvfp4/2=gold) required for load.
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
IDS_OUT = os.environ.get("IDS_OUT", "")
BASE_IDS = os.environ.get("BASE_IDS", "")
MODE = os.environ.get("MODE", (sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "run"))
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
print(f"init {time.time()-t0:.0f}s | mode={MODE} K={K} NTOK={NTOK} "
      f"eagle_mode={os.environ.get('NOMOS_EAGLE_MODE','1(nvfp4)')}", flush=True)

logits = np.zeros(VOCAB, np.float32)
vlogits = np.zeros((K + 1, VOCAB), np.float32)
draft_ids = np.zeros(K, np.int32)
seed_tap = np.zeros(TAPS, np.float32)
next_tap = np.zeros(TAPS, np.float32)
vtaps = np.zeros((K + 1, TAPS), np.float32)


def softcap(x, cap=30.0):
    return (cap * np.tanh(x.astype(np.float64) / cap)).astype(np.float32)


def spec_generate(ids, ntok, k=K, trace=False):
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
        # landmine #4 invariant: tokens committed per cycle == num_acc+1, never
        # beyond the verify block or the remaining budget.
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


def base_generate(ids, ntok):
    lib.nomos_reset_kv(ht)
    lib.nomos_eagle3_reset(ht)
    lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
    tok = int(np.argmax(logits))
    out = [tok]
    td0 = time.time()
    for _ in range(ntok - 1):
        lib.nomos_decode_step(ht, tok, logits.ctypes.data)
        tok = int(np.argmax(logits))
        out.append(tok)
    return out, time.time() - td0


# ── mode: smoke ──────────────────────────────────────────────────────────────
if MODE == "smoke":
    p = bar["prompts"][0]
    ids = np.array(p["ids"], dtype=np.int32)
    n = int(os.environ.get("SMOKE_NTOK", "17"))
    out, st, td = spec_generate(ids, n, trace=True)
    print(f"smoke [{p['id']}]: {n} tokens in {td:.2f}s | cycles={st['cycles']} "
          f"accepted/drafted={st['accepted']}/{st['drafted']} hist={st['acc_hist']}")
    print("ids:", out)
    ref, tdb = base_generate(ids, n)
    print("base:", ref)
    print("MATCH" if out == ref else "MISMATCH", flush=True)
    sys.exit(0 if out == ref else 1)

# ── mode: seam (landmine #5: verify bit-exactness vs decode) ────────────────
if MODE == "seam":
    fails = 0
    for p in bar["prompts"][:3]:
        ids = np.array(p["ids"], dtype=np.int32)
        # decode chain: 4 sequential decode steps from the prefill state
        lib.nomos_reset_kv(ht); lib.nomos_eagle3_reset(ht)
        lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
        tok = int(np.argmax(logits))
        L = lib.nomos_kv_cache_len(ht)
        chain = [tok]
        dec_logits = np.zeros((4, VOCAB), np.float32)
        for i in range(4):
            lib.nomos_decode_step(ht, chain[-1], logits.ctypes.data)
            dec_logits[i] = logits
            chain.append(int(np.argmax(logits)))
        # fused verify M=4 over the SAME rows (rollback first)
        lib.nomos_kv_set_len(ht, L)
        rows = np.array(chain[:4], np.int32)
        v4 = np.zeros((4, VOCAB), np.float32)
        rc = lib.nomos_eagle3_verify_fused(ht, rows.ctypes.data, 4, L, v4.ctypes.data)
        assert rc == 0
        # fused verify M=1 x4 (sequential, KV left in place between rows)
        lib.nomos_kv_set_len(ht, L)
        v1 = np.zeros((4, VOCAB), np.float32)
        for i in range(4):
            row = np.array([chain[i]], np.int32)
            rc = lib.nomos_eagle3_verify_fused(ht, row.ctypes.data, 1, L + i, v1[i:i + 1].ctypes.data)
            assert rc == 0
        lib.nomos_kv_set_len(ht, L)
        byte_eq = bool(np.array_equal(v4, v1))
        am_v = [int(np.argmax(v4[i])) for i in range(4)]
        am_d = [int(np.argmax(dec_logits[i])) for i in range(4)]
        sc = softcap(v4)
        # decode logits are post-softcap; verify rows are raw -> softcap on host.
        maxdiff = float(np.max(np.abs(sc - dec_logits)))
        ok = byte_eq and am_v == am_d
        print(f"[seam] {p['id']:>8}: M4-vs-M1x1 bytes {'IDENTICAL' if byte_eq else 'DIFFER'} | "
              f"verify-vs-decode argmax {'4/4' if am_v == am_d else f'{sum(a==b for a,b in zip(am_v,am_d))}/4 MISMATCH'} | "
              f"softcap-domain max|diff|={maxdiff:.3e}", flush=True)
        if not ok:
            fails += 1
    print("SEAM", "PASS" if fails == 0 else f"FAIL({fails})", flush=True)
    sys.exit(0 if fails == 0 else 1)

# ── mode: run (L1 ids + L2 acceptance + L3 perf) ─────────────────────────────
results = {}
ids_out = {}
acc_out = {}
for rep in range(REPEAT):
    tag = "warm" if rep < REPEAT - 1 else "SCORED"
    for p in bar["prompts"]:
        ids = np.array(p["ids"], dtype=np.int32)
        tp0 = time.time()
        out, st, td = spec_generate(ids, NTOK)
        tp = time.time() - tp0 - td
        dec_tps = (NTOK - 1) / td
        mean_acc = st["accepted"] / max(st["cycles"], 1)
        acc_rate = st["accepted"] / max(st["drafted"], 1)
        if rep == REPEAT - 1:
            results[p["id"]] = {"bucket": p["bucket"], "dec": dec_tps,
                                "acc_rate": acc_rate, "mean_acc": mean_acc,
                                "cycles": st["cycles"], "hist": st["acc_hist"],
                                "slot": [st["slot_acc"][i] / max(st["slot_n"][i], 1) for i in range(K)],
                                "tok_per_cycle": (st["accepted"] + st["cycles"]) / max(st["cycles"], 1)}
            ids_out[p["id"]] = out
            acc_out[p["id"]] = st
        print(f"[{tag}] {p['id']:>8} [{p['bucket']:>12}] prefix={len(ids):>4} "
              f"spec-decode={dec_tps:6.2f} tok/s | acc {st['accepted']}/{st['drafted']} "
              f"({acc_rate:.2f}) mean_acc/cycle={mean_acc:.2f} hist={st['acc_hist']}",
              flush=True)

if IDS_OUT:
    json.dump(ids_out, open(IDS_OUT, "w"))
    print("wrote", IDS_OUT, flush=True)

# L1: bit-identical ids vs base decode
if BASE_IDS:
    ref = json.load(open(BASE_IDS))
    n_match = 0
    for pid, seq in ids_out.items():
        refseq = ref[pid][:len(seq)]
        m = seq == refseq
        n_match += int(m)
        if not m:
            di = next(i for i in range(min(len(seq), len(refseq))) if seq[i] != refseq[i])
            print(f"L1 MISMATCH {pid}: first diff at tok {di}: spec={seq[di]} base={refseq[di]}")
    print(f"L1 LOSSLESS: {n_match}/{len(ids_out)} prompts bit-identical to base decode", flush=True)

# L2: per-bucket acceptance
print("\n== L2 ACCEPTANCE (free-running, 12-prompt bar) ==")
buckets = sorted(set(r["bucket"] for r in results.values()))
print(f"{'bucket':>12} {'acc_rate':>9} {'mean_acc':>9} {'tok/cycle':>9}  slot0/1/2")
tot_a = tot_d = 0
for b in buckets:
    rs = [r for r in results.values() if r["bucket"] == b]
    ar = float(np.mean([r["acc_rate"] for r in rs]))
    ma = float(np.mean([r["mean_acc"] for r in rs]))
    tc = float(np.mean([r["tok_per_cycle"] for r in rs]))
    sl = np.mean([r["slot"] for r in rs], axis=0)
    print(f"{b:>12} {ar:9.3f} {ma:9.3f} {tc:9.3f}  " + "/".join(f"{s:.2f}" for s in sl))
for pid, st in acc_out.items():
    tot_a += st["accepted"]; tot_d += st["drafted"]
print(f"{'OVERALL':>12} {tot_a}/{tot_d} = {tot_a/max(tot_d,1):.3f} accepted, "
      f"mean accepted len {np.mean([r['mean_acc'] for r in results.values()]):.3f}")

# L3: weighted tok/s (base-script convention: per-bucket median, weighted)
print("\n== L3 SPEC DECODE (greedy, k=%d) — per-bucket median, weighted ==" % K)
wsum = 0.0
for b, wgt in bar["weights"].items():
    v = sorted(r["dec"] for r in results.values() if r["bucket"] == b)
    med = v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
    wsum += wgt * med
    print(f"{b:>12}: median {med:6.2f} tok/s  (weight {wgt})")
print(f"WEIGHTED: {wsum:.2f} tok/s   | base NVFP4 reference: 21.45-21.7")
