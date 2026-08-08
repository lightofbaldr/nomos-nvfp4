#!/usr/bin/env python3
"""verify_busgap_prof — pin WHY eagle3 verify forward (M=4) is ~1.7x base decode.

Isolates ONE base decode step vs ONE M=4 verify forward, back to back, on GPU0.
Host wall-clock per call (each FFI forces a D2H sync so wall == true forward time).

Modes (MODE env):
  both   (default) : warm, then time N decode + N verify, print means + ratio
  decode           : warm, then ONLY N decode steps (for a clean nsys slice)
  verify           : warm, then ONLY N verify forwards (for a clean nsys slice)

cudaProfilerStart/Stop bracket the TIMED window only, so:
  nsys profile --capture-range=cudaProfilerApi ... captures just the loop
  (excludes the ~17GB weight load). Then nsys stats gives per-kernel GPU time;
  sum(kernel GPU time) vs wall(window) == the busy-vs-gap discriminator.

Env: WEIGHTS (trailing /), EAGLE_DIR (trailing /), SO_PATH, SPEC_K (3),
     PERF_N (30 timed iters), MODE.
"""
import ctypes, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.environ.get("SO_PATH", os.path.join(os.path.dirname(HERE), "libnomos_kernel.so"))
WTS = os.environ["WEIGHTS"]
EAGLE_DIR = os.environ["EAGLE_DIR"]
K = int(os.environ.get("SPEC_K", "3"))
N = int(os.environ.get("PERF_N", "30"))
MODE = os.environ.get("MODE", "both")
VOCAB = 262144
D = 5376
TAPS = 3 * D
assert WTS.endswith("/") and EAGLE_DIR.endswith("/"), "need trailing slash"

# cudaProfilerStart/Stop for nsys capture-range
cudart = ctypes.CDLL("libcudart.so.13")
cudart.cudaProfilerStart.restype = ctypes.c_int
cudart.cudaProfilerStop.restype = ctypes.c_int

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
    "nomos_eagle3_get_taps": (c.c_int32, [c.c_void_p, c.c_void_p]),
}
for n, (r, a) in sigs.items():
    f = getattr(lib, n); f.restype = r; f.argtypes = a

bar = json.load(open(os.path.join(HERE, "gold_bar_tokens.json")))
t0 = time.time()
ht = lib.nomos_init(WTS.encode())
assert ht, "init failed"
assert lib.nomos_eagle3_load(ht, EAGLE_DIR.encode()) == 0, "eagle load failed"
print(f"init {time.time()-t0:.0f}s | K={K} N={N} MODE={MODE}", flush=True)

logits = np.zeros(VOCAB, np.float32)
vlogits = np.zeros((K + 1, VOCAB), np.float32)
draft_ids = np.zeros(K, np.int32)
seed_tap = np.zeros(TAPS, np.float32)
ns = time.perf_counter_ns

# ---- prefill one prompt, build a real M=K+1 verify row set ----
ids = np.array(bar["prompts"][0]["ids"], dtype=np.int32)
lib.nomos_reset_kv(ht); lib.nomos_eagle3_reset(ht)
lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
tok = int(np.argmax(logits))
L = lib.nomos_kv_cache_len(ht)
lib.nomos_eagle3_get_taps(ht, seed_tap.ctypes.data)
nd = lib.nomos_eagle3_draft(ht, tok, K, draft_ids.ctypes.data)
drafts = [int(x) for x in draft_ids[:nd]]
rows = np.array([tok] + drafts, np.int32)
n_rows = nd + 1
print(f"prefill L={L} rows(M)={n_rows} tok={tok} drafts={drafts}", flush=True)


def one_decode(t):
    lib.nomos_decode_step(ht, t, logits.ctypes.data)
    return int(np.argmax(logits))


def one_verify():
    # fixed start_pos L; leaves KV at L+n_rows but we never advance L (overwrite each call)
    lib.nomos_eagle3_verify_fused(ht, rows.ctypes.data, n_rows, L, vlogits.ctypes.data)


# ---- warm (both paths, let clocks boost + kernels autotune) ----
wt = tok
for _ in range(12):
    one_verify()
    wt = one_decode(wt)
# restore KV len to L (decode advanced it); verify uses start_pos=L explicitly so only need cache sane
lib.nomos_kv_set_len(ht, L)


def clocks():
    try:
        import subprocess
        o = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], text=True).strip()
        return o
    except Exception:
        return "n/a"


print(f"[clocks pre-timed] sm,temp,pw = {clocks()}", flush=True)
cudart.cudaProfilerStart()

dec_ns = []
ver_ns = []
if MODE in ("both", "decode"):
    wt = tok
    for _ in range(N):
        t = ns(); wt = one_decode(wt); dec_ns.append(ns() - t)
    lib.nomos_kv_set_len(ht, L)
if MODE in ("both", "verify"):
    for _ in range(N):
        t = ns(); one_verify(); ver_ns.append(ns() - t)

cudart.cudaProfilerStop()
print(f"[clocks post-timed] sm,temp,pw = {clocks()}", flush=True)


def stats(xs):
    a = np.array(xs, float) / 1e6  # ms
    return f"n={len(a)} mean={a.mean():.3f} med={np.median(a):.3f} min={a.min():.3f} max={a.max():.3f} ms"


if dec_ns:
    print(f"[DECODE]  {stats(dec_ns)}")
if ver_ns:
    print(f"[VERIFY]  {stats(ver_ns)}")
if dec_ns and ver_ns:
    dm = np.median(dec_ns) / 1e6
    vm = np.median(ver_ns) / 1e6
    print(f"[RATIO] verify/decode (median) = {vm/dm:.3f}x  | extra = {vm-dm:.2f} ms/forward")
