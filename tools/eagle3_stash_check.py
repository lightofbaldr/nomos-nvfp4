#!/usr/bin/env python3
"""D2 G1b — aux-tap stash CONTENT check via an independent dual-path probe.

The live stash (d_taps_buf, captured by the decode-loop tap seam at decoder
layers [1,29,56] during the parity run) is compared against:
  (a) the DECODE-path layer_out (stage 7) replayed through the pre-existing
      nomos_debug_omlp_stage_ab probe — expected BYTE-EXACT (same math, an
      independent invocation path: step_logits replay + debug stage dump);
  (b) the BATCHED-prefill-path layer_out for the same position — independent
      kernels (batched GEMM/attention), expected cosine ~1 but not byte-exact.

Env: WEIGHTS, SO_PATH, EAGLE_DIR (NOMOS_EAGLE=1 required), TAPS_NPZ (from
eagle3_parity_ids.py TAPS_OUT), STEPS (default 12).
"""
import ctypes, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.environ.get("SO_PATH", os.path.join(os.path.dirname(HERE), "libnomos_kernel.so"))
WTS = os.environ["WEIGHTS"]
EAGLE_DIR = os.environ["EAGLE_DIR"]
TAPS_NPZ = os.environ["TAPS_NPZ"]
STEPS = int(os.environ.get("STEPS", "12"))
VOCAB = 262144
D = 5376
TAP_LAYERS = [1, 29, 56]   # engine decoder layers = HF hidden_states[2,30,57] (bc2daf6)

lib = ctypes.CDLL(SO)
lib.nomos_init.restype = ctypes.c_void_p
lib.nomos_init.argtypes = [ctypes.c_char_p]
lib.nomos_prefill.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
lib.nomos_reset_kv.argtypes = [ctypes.c_void_p]
lib.nomos_eagle3_load.restype = ctypes.c_int32
lib.nomos_eagle3_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.nomos_debug_omlp_stage_ab.restype = ctypes.c_int32
lib.nomos_debug_omlp_stage_ab.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p]

bar = json.load(open(os.path.join(HERE, "gold_bar_tokens.json")))
p0 = bar["prompts"][0]
ids = np.array(p0["ids"], dtype=np.int32)
n = len(ids)
live = np.load(TAPS_NPZ)
parity = json.load(open(os.environ.get("PARITY_JSON",
    "~/nomos-refs/drafter-study/reference/d2_on_parity_ids.json")))
seq = parity[p0["id"]]

ht = lib.nomos_init(WTS.encode())
assert ht
rc = lib.nomos_eagle3_load(ht, EAGLE_DIR.encode())
assert rc == 0, f"eagle3_load rc={rc}"
logits = np.zeros(VOCAB, np.float32)
lib.nomos_reset_kv(ht)
lib.nomos_prefill(ht, ids.ctypes.data, n, logits.ctypes.data)
assert int(np.argmax(logits)) == seq[0], "replay diverged at prefill"

out_i32 = np.zeros(16, np.int32)
out_f32 = np.zeros(4, np.float32)
dec = np.zeros(D, np.float32)
bat = np.zeros(D, np.float32)

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

print(f"prompt={p0['id']} n={n} steps={STEPS}")
print(f"{'step':>4} {'L':>3} {'byte_eq':>8} {'cos(live,dec)':>14} {'maxabs':>10} {'cos(dec,bat)':>13}")
worst = 1.0
all_byte_eq = True
for s in range(STEPS):
    toks = np.array(seq[: s + 1], dtype=np.int32)
    stash = live[f"step_{s}"]           # [3, D] captured during the live parity run
    for li, L in enumerate(TAP_LAYERS):
        rc = lib.nomos_debug_omlp_stage_ab(
            ht, toks.ctypes.data, s + 1, n, s, L, 7,
            out_i32.ctypes.data, out_f32.ctypes.data, dec.ctypes.data, bat.ctypes.data)
        assert rc == 0, f"probe rc={rc} step={s} L={L}"
        c_live = cos(stash[li], dec)
        byte_eq = bool(np.array_equal(stash[li], dec))
        all_byte_eq &= byte_eq
        worst = min(worst, c_live)
        print(f"{s:>4} {L:>3} {str(byte_eq):>8} "
              f"{c_live:>14.6f} {np.abs(stash[li]-dec).max():>10.3e} {out_f32[1]:>13.6f}")
print(f"\nG1b: all_byte_eq={all_byte_eq}  worst cos(live_stash, decode_replay)={worst:.6f}")
