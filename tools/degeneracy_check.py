#!/usr/bin/env python3
"""Degeneracy check on generated ids (IDS_OUT json from dflash_spec_loop).

WHY (Nano's finding, 2026-08-03): W4A4 Gemma can collapse into a repetition loop on long
prose-then-code prompts. A looping model is trivially predictable, so acceptance-at-depth (E)
INFLATES while the lossless gate passes on both-sides-identical garbage. No depth-E number gets
quoted until its outputs pass this check.

Metrics per prompt (64-token generations):
  max_run    longest identical-token run          (>=8  -> DEGENERATE)
  d2r        distinct-2gram ratio                  (<0.45 -> DEGENERATE; loops repeat bigrams)
  tail_loop  last 24 tokens are a cycle of period <=6 -> DEGENERATE
Verdict per prompt + overall. Optionally decodes text with --decode (needs transformers).

Usage: degeneracy_check.py <ids.json> [--decode]
"""
import json, sys

def max_run(seq):
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best

def distinct_2gram(seq):
    if len(seq) < 2: return 1.0
    grams = list(zip(seq, seq[1:]))
    return len(set(grams)) / len(grams)

def tail_loop(seq, tail=24, max_p=6):
    t = seq[-tail:]
    if len(t) < tail: return False
    for p in range(1, max_p + 1):
        if all(t[i] == t[i - p] for i in range(p, len(t))):
            return True
    return False

def main():
    ids = json.load(open(sys.argv[1]))
    decode = "--decode" in sys.argv
    tok = None
    if decode:
        import warnings; warnings.filterwarnings("ignore")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/models/gemma-4-31b-it"))
    bad = 0
    for pid, seq in ids.items():
        mr, d2, tl = max_run(seq), distinct_2gram(seq), tail_loop(seq)
        degen = mr >= 8 or d2 < 0.45 or tl
        bad += degen
        print(f"{pid:>16}  max_run={mr:<3} d2r={d2:.2f} tail_loop={tl}  "
              f"{'*** DEGENERATE — E for this prompt is NOT acceptance ***' if degen else 'coherent'}")
        if decode and tok:
            print(f"    text: {tok.decode(seq, skip_special_tokens=True)[:160]!r}")
    print(f"\nVERDICT: {len(ids)-bad}/{len(ids)} coherent — "
          + ("depth-E quotable" if bad == 0 else f"{bad} prompt(s) degenerate; EXCLUDE from E claims"))
    sys.exit(1 if bad else 0)

main()
