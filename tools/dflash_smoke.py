#!/usr/bin/env python3
"""DFlash cheapest-gate smoke on discrete Blackwell (arbiter test).

Does the DFlash Q4 block-draft path FAULT (MMU/ILLEGAL_ADDRESS) or WORK on the
RTX PRO 4000 discrete VRAM? Loads the drafter, prefills one short prompt, then
exercises the full spec seam ONCE: draft_block -> verify_fused -> longest-prefix
accept -> kv_set_len rollback -> append_verify_context.

Run via: tools/dflash_gold.sh tools/dflash_smoke.py
"""
import ctypes, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.environ.get("SO_PATH", os.path.join(os.path.dirname(HERE), "libnomos_kernel.so"))
WTS = os.environ.get("WEIGHTS", os.path.expanduser("~/nomos_data/gemma-4-31b/"))
DFLASH_DIR = os.environ.get("DFLASH_DIR", os.path.expanduser("~/nomos_data/dflash/dflash-gemma-4-31b-flat/"))
VOCAB = 262144
VB = int(os.environ.get("SPEC_VB", "5"))   # verify-block rows; vb-1 drafts verified

assert WTS.endswith("/"), "WEIGHTS must end with '/'"
assert DFLASH_DIR.endswith("/"), "DFLASH_DIR must end with '/'"

c = ctypes
lib = ctypes.CDLL(SO)
sigs = {
    "nomos_init":                       (c.c_void_p, [c.c_char_p]),
    "nomos_prefill":                    (c.c_int32, [c.c_void_p, c.c_void_p, c.c_int32, c.c_void_p]),
    "nomos_decode_step":                (c.c_int32, [c.c_void_p, c.c_int32, c.c_void_p]),
    "nomos_reset_kv":                   (c.c_int32, [c.c_void_p]),
    "nomos_kv_cache_len":               (c.c_int32, [c.c_void_p]),
    "nomos_kv_set_len":                 (c.c_int32, [c.c_void_p, c.c_int32]),
    "nomos_dflash_load":                (c.c_int32, [c.c_void_p, c.c_char_p]),
    "nomos_dflash_reset":               (c.c_int32, [c.c_void_p]),
    "nomos_dflash_cache_len":           (c.c_int32, [c.c_void_p]),
    "nomos_dflash_set_len":             (c.c_int32, [c.c_void_p, c.c_int32]),
    "nomos_dflash_draft_block":         (c.c_int32, [c.c_void_p, c.c_int32, c.c_int32, c.c_void_p]),
    "nomos_dflash_verify_fused":        (c.c_int32, [c.c_void_p, c.c_void_p, c.c_int32, c.c_int32, c.c_void_p]),
    "nomos_dflash_append_verify_context": (c.c_int32, [c.c_void_p, c.c_int32, c.c_int32]),
}
for n, (r, a) in sigs.items():
    f = getattr(lib, n); f.restype = r; f.argtypes = a

bar = json.load(open(os.path.join(HERE, "gold_bar_tokens.json")))
p = bar["prompts"][0]
ids = np.array(p["ids"], np.int32)
print(f"SO={os.path.basename(SO)} prompt={p['id']} nids={len(ids)} VB={VB}", flush=True)

t0 = time.time()
ht = lib.nomos_init(WTS.encode())
assert ht, "nomos_init failed"
print(f"engine init {time.time()-t0:.0f}s", flush=True)

rc = lib.nomos_dflash_load(ht, DFLASH_DIR.encode())
print(f"nomos_dflash_load rc={rc}  (0=ok)", flush=True)
assert rc == 0, f"dflash_load failed rc={rc}"

logits = np.zeros(VOCAB, np.float32)
lib.nomos_reset_kv(ht)
lib.nomos_dflash_reset(ht)
lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
seed = int(np.argmax(logits))
L = lib.nomos_kv_cache_len(ht)
dcl = lib.nomos_dflash_cache_len(ht)
print(f"after prefill: kv_cache_len={L} dflash_cache_len={dcl} seed_tok={seed}", flush=True)

# --- THE ARBITER: one draft_block on discrete ---
out15 = np.zeros(15, np.int32)
tb0 = time.time()
nd = lib.nomos_dflash_draft_block(ht, seed, L, out15.ctypes.data)
tb = (time.time() - tb0) * 1000.0
print(f"draft_block rc={nd} (expect 15)  cost={tb:.2f}ms  drafts={out15.tolist()}", flush=True)
assert nd == 15, f"draft_block rc={nd} -- if negative see EXC above (dflash not loaded / no embed / fault)"

# --- exercise verify + accept + re-anchor once ---
drafts = [int(x) for x in out15[:VB - 1]]
rows = np.array([seed] + drafts, np.int32)
vlogits = np.zeros((VB, VOCAB), np.float32)
tv0 = time.time()
rc = lib.nomos_dflash_verify_fused(ht, rows.ctypes.data, VB, L, vlogits.ctypes.data)
tv = (time.time() - tv0) * 1000.0
print(f"verify_fused rc={rc} (0=ok)  cost={tv:.2f}ms", flush=True)
assert rc == 0, f"verify rc={rc}"
tgt = [int(np.argmax(vlogits[i])) for i in range(VB)]
num_acc = 0
while num_acc < len(drafts) and drafts[num_acc] == tgt[num_acc]:
    num_acc += 1
nxt = tgt[num_acc]
print(f"tgt={tgt}  drafts={drafts}  num_acc={num_acc}  next={nxt}", flush=True)

lib.nomos_kv_set_len(ht, L + 1 + num_acc)
newcl = lib.nomos_dflash_append_verify_context(ht, 0, num_acc + 1)
print(f"kv_set_len({L+1+num_acc}) ok ; append_verify_context(0,{num_acc+1}) -> dflash_cache_len={newcl}", flush=True)

# --- lossless cross-check: does verify row0 argmax == base greedy next? ---
lib.nomos_reset_kv(ht)
lib.nomos_dflash_reset(ht)
lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
base_seed = int(np.argmax(logits))
lib.nomos_decode_step(ht, base_seed, logits.ctypes.data)
base_next = int(np.argmax(logits))
print(f"lossless spot-check: verify_tgt[0]={tgt[0]} vs base_decode_next={base_next} "
      f"({'MATCH' if tgt[0]==base_next else 'MISMATCH'})", flush=True)
print("SMOKE OK: DFlash Q4 block-draft path RUNS on the RTX PRO 4000 discrete (no MMU fault).", flush=True)
