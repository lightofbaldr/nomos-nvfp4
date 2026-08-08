#!/usr/bin/env python3
"""#430 footprint probe — load the engine at the given NOMOS_MAX_SEQ + precision env,
report the unified-memory delta (the kernel's cudaMalloc footprint), and do one short
coherence generation. On GB10 unified memory, `free` used-delta == the GPU allocation.

Run twice for an A/B: once NOMOS_KV_SWA=0, once =1, both NOMOS_MAX_SEQ=262144.
"""
import os, sys, ctypes, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4_parity_harness as H


def used_mb():
    out = subprocess.check_output(["free", "-m"]).decode()
    for line in out.splitlines():
        if line.startswith("Mem:"):
            f = line.split()
            return int(f[2]), int(f[6])   # used, available
    return -1, -1


def main():
    swa = os.environ.get("NOMOS_KV_SWA", "0")
    maxseq = os.environ.get("NOMOS_MAX_SEQ", "?")
    u0, a0 = used_mb()
    print(f"=== FOOTPRINT PROBE  max_seq={maxseq}  kv_swa={swa} ===", flush=True)
    print(f"  MEM before: used={u0/1024:.1f}G avail={a0/1024:.1f}G", flush=True)
    lib = H.load_lib()
    h = H.init_engine(lib)
    u1, a1 = used_mb()
    print(f"  MEM after load: used={u1/1024:.1f}G avail={a1/1024:.1f}G", flush=True)
    print(f"  >>> FOOTPRINT (used delta) = {(u1-u0)/1024:.1f} GB  ({u1-u0} MB)", flush=True)
    # one short coherence gen (in-process tokenization; no tokenizer-svc needed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(H.TOK_DIR)
    enc = tok.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        add_generation_prompt=True, return_tensors="np", return_dict=True)
    ids = np.asarray(enc["input_ids"][0], dtype=np.int32)
    outb = np.zeros(12, dtype=np.int32)
    nout = lib.nomos_generate(h, ids.ctypes.data, len(ids), 12, outb.ctypes.data, 12,
                              0.0, -1.0, 1.0)
    txt = tok.decode(outb[:max(int(nout), 0)].tolist(), skip_special_tokens=False)
    print(f"  COHERENCE @ {maxseq}: {txt!r}", flush=True)


if __name__ == "__main__":
    main()
