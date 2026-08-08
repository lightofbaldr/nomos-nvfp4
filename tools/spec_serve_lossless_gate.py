#!/usr/bin/env python3
"""Lossless gate for the serve-side DFlash spec wiring.

Proves KernelLM.spec_generate_stream (the new serve spec path) yields tokens
BYTE-IDENTICAL to base greedy generate() over the bar prompts. This is the gate
that makes wiring spec into the serve safe: verify==decode by construction.
Run with the Q4 GB10 env + DFLASH_DIR (trailing slash).
"""
import os, sys, json
import numpy as np

REPO = os.path.expanduser("~/nomos-nvfp4")
sys.path.insert(0, REPO)
from pyhost.host import KernelLM

NTOK = int(os.environ.get("NTOK", "48"))
VB = int(os.environ.get("SPEC_VB", "8"))
bar = json.load(open(os.path.join(REPO, "tools", "gold_bar_tokens.json")))
prompts = bar["prompts"]

lm = KernelLM()
lm.dflash_load(os.environ["DFLASH_DIR"])

allok = True
for p in prompts:
    ids = list(p["ids"])
    base = lm.generate(ids, max_new_tokens=NTOK, temperature=0.0)
    spec = list(lm.spec_generate_stream(ids, max_new_tokens=NTOK, vb=VB))
    ok = base == spec
    allok = allok and ok
    tag = "MATCH" if ok else "MISMATCH"
    print(f"  {p['id']:>10} [{p.get('bucket','?'):>10}] {tag}  n_base={len(base)} n_spec={len(spec)}")
    if not ok:
        # first divergence
        for i,(a,b) in enumerate(zip(base, spec)):
            if a != b:
                print(f"     first diff @tok {i}: base={a} spec={b}")
                break
print("LOSSLESS — spec stream == base greedy" if allok else "FAIL — spec diverges from greedy")
sys.exit(0 if allok else 1)
