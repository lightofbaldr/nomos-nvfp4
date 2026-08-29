#!/usr/bin/env python3
"""Profile-aware DFlash block speculative-decode correctness/perf gate.

DFlash drafts one profile-sized synthetic block in a single forward, then the
target verifies seed+candidate rows. Geometry is resolved from nomos_model_id:
Gemma/Muse use 15 candidates; Qwen DSpark uses 7. Never infer vocabulary or
block size from the weights directory.

Gates (must pass before the tok/s headline is trusted):
  L1  LOSSLESS: spec-decode ids match base through its first EOS inclusive (12/12).
  L0  decode non-regress: base M=1 tok/s vs the ~21.7 reference.
Measures: per-bucket + weighted tok/s (vs eagle3 ~33, target 35), per-bucket
  acceptance, and the DFlash draft cost (ms/cycle) + verify cost (ms/cycle).

Seam (code-verified):
  draft_block(seed, start_pos, out[candidates]) -> candidates
  verify_fused(toks[vb], vb, start_pos, rows[vb*VOCAB]) -> 0
  append_verify_context(0, num_acc+1) -> new dflash cache_len
  vb counts the seed row: drafts verified = vb-1, so K = vb-1 (vb=5 => K=4).

Run: tools/dflash_gold.sh tools/dflash_spec_loop.py [--mode run|smoke]
"""
import ctypes, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.environ.get("SO_PATH", os.path.join(os.path.dirname(HERE), "libnomos_kernel.so"))
lib = ctypes.CDLL(SO)
lib.nomos_model_id.restype = ctypes.c_int32
lib.nomos_model_id.argtypes = []
MODEL_ID = int(lib.nomos_model_id())
MODEL_GEOMETRY = {
    1: ("gemma4", 262144, 15, "~/nomos_data/gemma-4-31b/",
        "~/nomos_data/dflash-gemma-4-31b-flat/"),
    2: ("muse", 202048, 15, "~/nomos_data/muse-glimmer-flat/",
        "~/nomos_data/muse-assistant-flat/"),
    3: ("qwen3_5", 248320, 7, "~/nomos_data/qwen3_8_27b-deploy-nvfp4/",
        "~/nomos_data/qwen3_8_27b-dspark-blobs/"),
}
assert MODEL_ID in MODEL_GEOMETRY, f"unsupported nomos_model_id={MODEL_ID}"
MODEL_NAME, VOCAB, DRAFT_CANDIDATES, _WTS_DEFAULT, _DFLASH_DEFAULT = MODEL_GEOMETRY[MODEL_ID]
MODEL_EOS_IDS = {
    1: (1, 106),       # Gemma config eos_token_id
    2: (200001,),      # Muse text_config.eos_token_id
    3: (248044,),      # Qwen text_config.eos_token_id
}
EOS_IDS = set(MODEL_EOS_IDS[MODEL_ID])
WTS = os.environ.get("WEIGHTS", os.path.expanduser(_WTS_DEFAULT))
DFLASH_DIR = os.environ.get("DFLASH_DIR", os.path.expanduser(_DFLASH_DEFAULT))
NTOK = int(os.environ.get("PERF_NTOK", "96"))
VB = int(os.environ.get("SPEC_VB", "8" if MODEL_ID == 3 else "7"))
# Default 7 = the swept, lossless-gated champion (docs/the runbook §3). Was 5, which silently
# disagreed with every runner/serve default — the entry-point class of bug. Harnesses that need
# a different VB (e.g. kv_block32_matrix.sh pins 5 for cross-card comparability) set it explicitly.
REPEAT = int(os.environ.get("PERF_REPEAT", "2"))   # warm... last = scored
BASE_IDS = os.environ.get("BASE_IDS", "")          # optional cross-check json
IDS_OUT = os.environ.get("IDS_OUT", "")
MODE = os.environ.get("MODE", "run")
if "--mode" in sys.argv:
    MODE = sys.argv[sys.argv.index("--mode") + 1]

assert WTS.endswith("/"), "WEIGHTS must end with '/'"
assert DFLASH_DIR.endswith("/"), "DFLASH_DIR must end with '/'"
assert VB >= 2, "VB must be >= 2 (>=1 draft)"
assert VB - 1 <= DRAFT_CANDIDATES, (
    f"SPEC_VB={VB} requests {VB-1} drafts, profile {MODEL_NAME} exposes "
    f"only {DRAFT_CANDIDATES}"
)

c = ctypes
sigs = {
    "nomos_init":                         (c.c_void_p, [c.c_char_p]),
    "nomos_prefill":                      (c.c_int32, [c.c_void_p, c.c_void_p, c.c_int32, c.c_void_p]),
    "nomos_decode_step":                  (c.c_int32, [c.c_void_p, c.c_int32, c.c_void_p]),
    "nomos_reset_kv":                     (c.c_int32, [c.c_void_p]),
    "nomos_kv_cache_len":                 (c.c_int32, [c.c_void_p]),
    "nomos_kv_set_len":                   (c.c_int32, [c.c_void_p, c.c_int32]),
    "nomos_dflash_load":                  (c.c_int32, [c.c_void_p, c.c_char_p]),
    "nomos_dflash_reset":                 (c.c_int32, [c.c_void_p]),
    "nomos_dflash_cache_len":             (c.c_int32, [c.c_void_p]),
    "nomos_dflash_draft_block":           (c.c_int32, [c.c_void_p, c.c_int32, c.c_int32, c.c_void_p]),
    "nomos_dflash_verify_fused":          (c.c_int32, [c.c_void_p, c.c_void_p, c.c_int32, c.c_int32, c.c_void_p]),
    "nomos_dflash_append_verify_context": (c.c_int32, [c.c_void_p, c.c_int32, c.c_int32]),
}
for n, (r, a) in sigs.items():
    f = getattr(lib, n); f.restype = r; f.argtypes = a

# BAR_TOKENS selects the prompt set (repo-relative or absolute). Default = the standard 12-prompt
# bar. tools/gold_bar_tokens_long.json is the LONG-CONTEXT tier (1.2K-20K prompt tokens): the
# standard bar tops out at position ~280, so the 1024 sliding window NEVER fills there and every
# window-boundary behavior (ring, PV block anchoring, acceptance at depth) is unmeasured by it.
_BAR = os.environ.get("BAR_TOKENS", "gold_bar_tokens.json")
bar = json.load(open(_BAR if os.path.isabs(_BAR) else os.path.join(HERE, _BAR)))
assert bar.get("prompts"), f"bar {_BAR} has no prompts"
for p in bar["prompts"]:
    assert p.get("ids"), f"bar prompt {p.get('id', '<unnamed>')} has no token ids"
    bad = [int(t) for t in p["ids"] if int(t) < 0 or int(t) >= VOCAB]
    assert not bad, (
        f"bar prompt {p.get('id', '<unnamed>')} contains ids outside "
        f"profile vocab={VOCAB}: {bad[:4]}"
    )
print(f"bar set: {_BAR} ({len(bar['prompts'])} prompts, weights {bar['weights']})", flush=True)

t0 = time.time()
ht = lib.nomos_init(WTS.encode())
assert ht, "nomos_init failed"
rc = lib.nomos_dflash_load(ht, DFLASH_DIR.encode())
assert rc == 0, f"nomos_dflash_load rc={rc}"
print(f"engine init {time.time()-t0:.0f}s | model={MODEL_NAME} id={MODEL_ID} "
      f"vocab={VOCAB} candidates={DRAFT_CANDIDATES} | SO={os.path.basename(SO)} "
      f"| VB={VB} (K={VB-1}) NTOK={NTOK}", flush=True)

logits = np.zeros(VOCAB, np.float32)
draft_out = np.zeros(DRAFT_CANDIDATES, np.int32)
vlogits = np.zeros((VB, VOCAB), np.float32)


def base_generate(ids, ntok):
    """M=1 greedy base decode — the lossless reference and the decode-tok/s probe."""
    lib.nomos_reset_kv(ht)
    lib.nomos_dflash_reset(ht)
    lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
    tok = int(np.argmax(logits)); out = [tok]
    td0 = time.time()
    for _ in range(ntok - 1):
        lib.nomos_decode_step(ht, tok, logits.ctypes.data)
        tok = int(np.argmax(logits)); out.append(tok)
    td = time.time() - td0
    return out, td


def spec_generate(ids, ntok, vb):
    """DFlash block spec decode. Returns (ids[ntok], stats, decode_seconds)."""
    lib.nomos_reset_kv(ht)
    lib.nomos_dflash_reset(ht)
    lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
    seed = int(np.argmax(logits)); out = [seed]
    L = lib.nomos_kv_cache_len(ht)
    assert L == len(ids), f"cache_len {L} != prompt {len(ids)}"
    K = vb - 1
    st = {"cycles": 0, "drafted": 0, "accepted": 0, "acc_hist": [0] * (K + 1),
          "slot_acc": [0] * K, "slot_n": [0] * K, "fallback": 0,
          "draft_ms": 0.0, "verify_ms": 0.0, "commit_ms": 0.0,
          "partial_cycles": 0}
    td0 = time.time()
    while len(out) < ntok:
        r = ntok - len(out)
        k = min(K, r - 1)                 # drafts to verify this cycle (budget-capped)
        if k <= 0:                        # 1 token left -> exact M=1 step
            lib.nomos_decode_step(ht, seed, logits.ctypes.data)
            out.append(int(np.argmax(logits))); st["fallback"] += 1
            break
        vb_eff = k + 1
        t1 = time.time()
        nd = lib.nomos_dflash_draft_block(ht, seed, L, draft_out.ctypes.data)
        st["draft_ms"] += (time.time() - t1) * 1000.0
        assert nd == DRAFT_CANDIDATES, (
            f"draft_block rc={nd}, expected profile candidates={DRAFT_CANDIDATES}"
        )
        drafts = [int(x) for x in draft_out[:k]]
        rows = np.array([seed] + drafts, np.int32)
        t2 = time.time()
        rc = lib.nomos_dflash_verify_fused(ht, rows.ctypes.data, vb_eff, L, vlogits.ctypes.data)
        st["verify_ms"] += (time.time() - t2) * 1000.0
        assert rc == 0, f"verify rc={rc}"
        tgt = [int(np.argmax(vlogits[i])) for i in range(vb_eff)]
        num_acc = 0
        while num_acc < k and drafts[num_acc] == tgt[num_acc]:
            num_acc += 1
        nxt = tgt[num_acc]
        lib.nomos_kv_set_len(ht, L + 1 + num_acc)
        t3 = time.time()
        newcl = lib.nomos_dflash_append_verify_context(ht, 0, num_acc + 1)
        st["commit_ms"] += (time.time() - t3) * 1000.0
        if num_acc < k:
            st["partial_cycles"] += 1
        assert newcl > 0, f"append_verify_context rc={newcl}"
        out.extend(drafts[:num_acc]); out.append(nxt)
        for i in range(k):
            st["slot_n"][i] += 1
            if i < num_acc:
                st["slot_acc"][i] += 1
        st["cycles"] += 1; st["drafted"] += k; st["accepted"] += num_acc
        st["acc_hist"][num_acc] += 1
        L = L + 1 + num_acc; seed = nxt
    td = time.time() - td0
    return out[:ntok], st, td


def lossless_through_base_eos(spec, base):
    """Compare generated IDs through the base stream's natural stop, inclusive.

    Tokens after base EOS are undefined forced-tail padding. A spec EOS at any
    other position remains a real failure; truncation must never hide an early
    or late stop.
    """
    base_eos = next((i for i, tok in enumerate(base) if tok in EOS_IDS), None)
    if base_eos is None:
        if len(spec) != len(base):
            return False, None, f"LENGTH spec={len(spec)} base={len(base)}"
        if spec == base:
            return True, None, ""
        first = next(i for i, (s, b) in enumerate(zip(spec, base)) if s != b)
        return False, None, f"first diff @tok {first}: spec={spec[first]} base={base[first]}"

    spec_eos = next((i for i, tok in enumerate(spec) if tok in EOS_IDS), None)
    if spec_eos != base_eos:
        return False, base_eos, f"EOS position spec={spec_eos} base={base_eos}"
    if len(spec) <= base_eos:
        return False, base_eos, f"LENGTH spec={len(spec)} cannot include base EOS @{base_eos}"
    for i in range(base_eos + 1):
        if spec[i] != base[i]:
            return False, base_eos, f"first diff @tok {i}: spec={spec[i]} base={base[i]}"
    return True, base_eos, ""


# ─────────────────── smoke: 1-prompt lossless ───────────────────
if MODE == "smoke":
    p = bar["prompts"][0]
    ids = np.array(p["ids"], np.int32)
    b, _ = base_generate(ids, NTOK)
    s, stt, td = spec_generate(ids, NTOK, VB)
    ok, base_eos, why = lossless_through_base_eos(s, b)
    print(f"[smoke] {p['id']} lossless={'MATCH' if ok else 'MISMATCH'} "
          f"acc={stt['accepted']}/{stt['drafted']} cyc={stt['cycles']} "
          f"draft={stt['draft_ms']/max(1,stt['cycles']):.2f}ms/cyc "
          f"commit={stt['commit_ms']/max(1,stt['cycles']):.2f}ms/cyc "
          f"tok/s={(NTOK-1)/td:.2f}", flush=True)
    if not ok:
        print(f"  {why}")
    elif base_eos is not None:
        print(f"  compared through base EOS @tok {base_eos} inclusive; forced tail ignored")
    sys.exit(0 if ok else 1)

# ─────────────────── run: full 12-prompt bar ───────────────────
# base reference (once): L1 truth + decode non-regress
ref, base_dec = {}, {}
for p in bar["prompts"]:
    ids = np.array(p["ids"], np.int32)
    seq, td = base_generate(ids, NTOK)
    ref[p["id"]] = seq
    base_dec[p["id"]] = (NTOK - 1) / td
    print(f"[base]  {p['id']:>8} [{p['bucket']:>12}] decode={base_dec[p['id']]:6.2f} tok/s", flush=True)

if BASE_IDS and os.path.exists(BASE_IDS):
    saved = json.load(open(BASE_IDS))
    nmatch = sum(1 for pid in ref if pid in saved and ref[pid] == saved[pid][:len(ref[pid])])
    print(f"[base] cross-check vs {os.path.basename(BASE_IDS)}: {nmatch}/{len(ref)} match", flush=True)

results, ids_out = {}, {}
for rep in range(REPEAT):
    tag = "warm" if rep < REPEAT - 1 else "SCORED"
    for p in bar["prompts"]:
        ids = np.array(p["ids"], np.int32)
        seq, st, td = spec_generate(ids, NTOK, VB)
        dec = (NTOK - 1) / td
        cyc = max(1, st["cycles"])
        if rep == REPEAT - 1:
            results[p["id"]] = {"bucket": p["bucket"], "dec": dec, "st": st}
            ids_out[p["id"]] = seq
        print(f"[{tag}] {p['id']:>8} [{p['bucket']:>12}] spec={dec:6.2f} tok/s "
              f"acc={st['accepted']}/{st['drafted']}={st['accepted']/max(1,st['drafted']):.3f} "
              f"tok/cyc={(st['accepted']+cyc)/cyc:.2f} "
              f"draft={st['draft_ms']/cyc:.2f}ms verify={st['verify_ms']/cyc:.2f}ms "
              f"commit={st['commit_ms']/cyc:.2f}ms/cyc "
              f"partial={st['partial_cycles']}/{cyc}", flush=True)

if IDS_OUT:
    json.dump(ids_out, open(IDS_OUT, "w"))

# ── L1 LOSSLESS ──
nm = 0
for pid, seq in ids_out.items():
    rseq = ref[pid]
    m, base_eos, why = lossless_through_base_eos(seq, rseq)
    nm += int(m)
    if not m:
        print(f"L1 MISMATCH {pid}: {why}")
    elif base_eos is not None:
        print(f"L1 MATCH {pid}: equal through base EOS @tok {base_eos} inclusive "
              f"(ignored {len(rseq) - base_eos - 1} forced-tail tokens)")
print(f"\nL1 LOSSLESS: {nm}/{len(ids_out)} prompts match base through natural stop "
      f"{'-- PASS' if nm == len(ids_out) else '-- FAIL'}", flush=True)

# ── L0 decode non-regress ──
bw = 0.0
for b, wgt in bar["weights"].items():
    v = sorted(base_dec[pid] for pid in base_dec if results.get(pid, {}).get("bucket") == b)
    if v:
        med = v[len(v)//2] if len(v) % 2 else 0.5*(v[len(v)//2-1]+v[len(v)//2])
        bw += wgt * med
# The base-decode reference is PER-ARCH and the hardcoded ~21.7 was an NVFP4 discrete number.
# Printed against a GB10 run it reads as a 2x regression that does not exist: GB10/Q4_0 base decode
# is ~10.2 (Stage 1 close, ~83% of achievable), so a healthy 10.7 looked like half of 21.7 and cost
# real time chasing it on 2026-08-02. A reference that is wrong for the box is worse than none.
_BASE_REF = os.environ.get("NOMOS_BASE_REF") or (
    "~10.2 GB10/Q4_0" if os.uname().machine == "aarch64" else "~21.7 discrete/NVFP4")
print(f"L0 BASE DECODE weighted: {bw:.2f} tok/s (reference {_BASE_REF})", flush=True)

# ── L2 per-bucket acceptance ──
print("\n== L2 ACCEPTANCE (scored) ==", flush=True)
for b in bar["weights"]:
    ids_b = [r for r in results.values() if r["bucket"] == b]
    if not ids_b:
        continue
    dd = sum(r["st"]["drafted"] for r in ids_b)
    aa = sum(r["st"]["accepted"] for r in ids_b)
    cc = sum(r["st"]["cycles"] for r in ids_b)
    print(f"{b:>12}: acc_rate={aa/max(1,dd):.3f}  tok/cyc={(aa+cc)/max(1,cc):.2f}  "
          f"(cycles={cc})", flush=True)

# ── L3 per-bucket + weighted spec tok/s ──
print("\n== L3 DFLASH SPEC DECODE — per-bucket median, weighted ==", flush=True)
wsum = 0.0
for b, wgt in bar["weights"].items():
    v = sorted(r["dec"] for r in results.values() if r["bucket"] == b)
    med = v[len(v)//2] if len(v) % 2 else 0.5*(v[len(v)//2-1]+v[len(v)//2])
    wsum += wgt * med
    print(f"{b:>12}: median {med:6.2f} tok/s  (weight {wgt})", flush=True)
draft_ms_all = sum(r["st"]["draft_ms"] for r in results.values())
cyc_all = sum(r["st"]["cycles"] for r in results.values())
ver_ms_all = sum(r["st"]["verify_ms"] for r in results.values())
commit_ms_all = sum(r["st"]["commit_ms"] for r in results.values())
partial_all = sum(r["st"]["partial_cycles"] for r in results.values())
print(f"WEIGHTED: {wsum:.2f} tok/s   | eagle3 ~33 | target 35  "
      f"=> clears 35? {'YES' if wsum >= 35 else 'NO (short %.2f)' % (35-wsum)}", flush=True)
print(f"DFlash draft cost: {draft_ms_all/max(1,cyc_all):.2f} ms/cycle "
      f"| verify cost: {ver_ms_all/max(1,cyc_all):.2f} ms/cycle "
      f"| commit/rollback cost: {commit_ms_all/max(1,cyc_all):.2f} ms/cycle "
      f"| partial cycles: {partial_all}/{cyc_all} "
      f"(eagle3 draft ~3ms, MTP ~30ms)", flush=True)

# ── L4 draft-depth decay ──
# slot_acc[i]/slot_n[i] is UNCONDITIONAL: acceptance is a strict prefix, so slot i
# accepted implies every earlier slot accepted. That ratio is therefore the survival
# curve S(i) = P(>= i+1 drafts accepted). The conditional hazard S(i)/S(i-1) is what
# says whether one more draft slot pays for the verify row it costs.
print("\n== L4 DRAFT-DEPTH DECAY (survival + conditional) ==", flush=True)
KK = VB - 1
s_acc = [0] * KK
s_n = [0] * KK
for r in results.values():
    for i in range(KK):
        s_acc[i] += r["st"]["slot_acc"][i]
        s_n[i] += r["st"]["slot_n"][i]
prev = 1.0
exp_tok = 1.0  # the free target token every cycle yields
for i in range(KK):
    surv = s_acc[i] / max(1, s_n[i])
    cond = surv / prev if prev > 1e-9 else 0.0
    exp_tok += surv
    print(f"  depth {i+1:>2}: survival={surv:.3f}  conditional={cond:.3f}  "
          f"n={s_n[i]:>5}", flush=True)
    prev = surv
print(f"  E[tokens/cycle] = {exp_tok:.2f}  (1 free + sum of survival)", flush=True)

# ── DEAD-DRAFTER GUARD ────────────────────────────────────────────────────────────────────
# E == 1.00 means EVERY draft was rejected at EVERY depth: the drafter contributed nothing and
# the run is speculative decode in name only. It is a FAILED RUN, not a slow one.
#
# It has to be called out explicitly because the lossless gate CANNOT catch it. With zero
# acceptance the speculative path degenerates to base decode one token at a time, so it matches
# base decode trivially and "L1 LOSSLESS 12/12 PASS" prints in green. The failure moved BOTH
# sides of the comparison, which is exactly the condition under which a self-referential gate
# goes blind (see docs/the measurement notes).
#
# This signature appeared THREE times on 2026-08-02, each time initially misread:
#   - block-32 KV written as fp16 and read as fp32   -> corrupt KV, drafter taps garbage
#   - the same bug on the demo laptop (Prime)        -> 46.58 -> 17.55 tok/s, E 2.69 -> 1.00
#   - a SPEC_VB sweep whose anchor arm silently ran a different drafter-load path
# In all three the throughput number was ~40% of expected and the gate said PASS.
if exp_tok < 1.05:
    print("", flush=True)
    print("  *** DEAD DRAFTER — THIS RUN IS INVALID, DO NOT QUOTE ITS tok/s ***", flush=True)
    print(f"  E[tokens/cycle]={exp_tok:.2f}: every draft rejected at every depth. The drafter", flush=True)
    print("  contributed NOTHING, so this is base decode wearing a speculative harness.", flush=True)
    print("  A LOSSLESS PASS above does NOT clear this: with zero acceptance the spec path", flush=True)
    print("  degenerates to base decode and matches it trivially. The gate is blind here.", flush=True)
    print("  Usual causes: drafter failed to load / wrong DFLASH_DIR / drafter-target mismatch /", flush=True)
    print("  corrupt KV that the drafter taps (check NOMOS_KV_I4_BLOCK vs the read path).", flush=True)
elif exp_tok < 1.30:
    print(f"  WARNING: E={exp_tok:.2f} is very low — acceptance is nearly dead. Verify the drafter"
          " before quoting this run.", flush=True)
