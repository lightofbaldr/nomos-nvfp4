#!/usr/bin/env python3
"""Muse margin analyzer — the sub-criterion muse_parity.py never computes (Prime, 2026-08-10).

muse_parity prints agree/total, which CANNOT separate "HF was ambivalent (near-tie)"
from "we were genuinely wrong". This uses HF's OWN per-token confidence to make that
split, and reports prefill-vs-decode cosine separately. ONE instrument so GB10 and
NVFP4 results compare like-for-like — the whole point after the aggregate-sigma
margin test proved too coarse and washed out real genuine misses.

Inputs: kernel per-step logit dumps from `muse_parity.py --dump-dir` and the HF
goldens' step_logits. Alignment (kernel step k <-> golden step k) is found
empirically by max argmax-agreement, then reported.

  genuine := HF p(top1) - HF p(our_pick) > GAP (default 0.25).  Keys on the actual
             probability mass given up, NOT ordinal HF-rank — because rank 2 can be
             0.98 vs 0.01, so a rank cutoff smuggles the aggregate-sigma coarseness
             back in (Prime, 2026-08-10). p(top1) and p(our_pick) print per miss so
             the threshold stays auditable.

Also reports the decode-cos TREND vs step index (first/second-half means + Pearson r):
a prose prompt that DEGRADES with position but a code prompt that stays FLAT is not
decode-cache accumulation (that hits both) — it's a prompt-dependent effect.
"""
import argparse
import os
import numpy as np


def cos(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else 0.0


def softmax(x):
    x = x.astype(np.float64) - x.max()
    e = np.exp(x)
    return e / e.sum()


def analyze(name, dump_path, goldens_dir, gap):
    z = np.load(os.path.join(goldens_dir, f"golden_{name}.npz"))
    g_step = z["step_logits"]
    k_step = np.load(dump_path)
    n = min(len(g_step), len(k_step))
    best = (0, -1.0)
    for off in (-1, 0, 1):
        agree = tot = 0
        for k in range(n):
            j = k + off
            if 0 <= j < len(g_step):
                tot += 1
                agree += int(np.argmax(k_step[k]) == np.argmax(g_step[j]))
        if tot and agree / tot > best[1]:
            best = (off, agree / tot)
    off = best[0]
    coss, steps, misses, genuine = [], [], [], 0
    for k in range(n):
        j = k + off
        if not (0 <= j < len(g_step)):
            continue
        coss.append(cos(k_step[k], g_step[j]))
        steps.append(k)
        ka, ga = int(np.argmax(k_step[k])), int(np.argmax(g_step[j]))
        if ka != ga:
            p = softmax(g_step[j])
            ptop, ppick = float(p[ga]), float(p[ka])
            gen = (ptop - ppick) > gap
            genuine += gen
            misses.append((k, ptop, ppick, gen))
    coss = np.array(coss)
    h = len(coss) // 2
    r = float(np.corrcoef(steps, coss)[0, 1]) if len(coss) > 2 else 0.0
    trend = "DEGRADES" if r < -0.15 else ("FLAT" if r > -0.15 else "?")
    print(f"\n=== {name}  (align off={off}, argmax-agree {best[1]*100:.0f}%) ===")
    print(f"  mean DECODE cos: {coss.mean():.4f}  (min {coss.min():.4f} @step {steps[int(coss.argmin())]})")
    print(f"  trend vs step: first-half {coss[:h].mean():.4f} -> second-half {coss[h:].mean():.4f}  "
          f"r={r:+.2f}  {trend}")
    print(f"  misses {len(misses)}   GENUINE (p(top1)-p(pick) > {gap}): {genuine}   near-tie {len(misses)-genuine}")
    for k, ptop, ppick, gen in misses:
        print(f"    step {k:2d}: HF p(top1)={ptop:.4f}  p(our_pick)={ppick:.4f}  gap={ptop-ppick:+.4f}  "
              f"{'GENUINE' if gen else 'near-tie'}")
    return genuine, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True, help="dir with kernel_step_logits_<name>.npy")
    ap.add_argument("--goldens", default="~/nomos-ref/muse-ref/goldens")
    ap.add_argument("--prompts", default="en_short,code_short")
    ap.add_argument("--gap", type=float, default=0.25, help="p(top1)-p(pick) above this = genuine")
    a = ap.parse_args()
    tot = 0
    for nm in a.prompts.split(","):
        g, _ = analyze(nm, os.path.join(a.dump_dir, f"kernel_step_logits_{nm}.npy"), a.goldens, a.gap)
        tot += g
    print(f"\nTOTAL genuine misses: {tot}")


if __name__ == "__main__":
    main()
