#!/usr/bin/env python3
"""Build a frequency-ranked draft-vocab subset + coverage curve from a real corpus.

WHY. Our DFlash drafter projects 5376 -> 262144 every draft step: 1.41B params, **48% of total
draft compute** (vs 6% for RedHatAI's DSpark, which uses draft_vocab_size=32000). The drafter only
has to PROPOSE; the target still verifies against the full vocabulary, so a restricted proposal
head is lossless-by-construction on the verify side and costs only the positions whose correct
token falls outside the subset.

**THE TRAP THIS AVOIDS.** "Take the first N token IDs" is WRONG — Gemma's vocab is not
frequency-ordered. Measured on 49,647 tokens of real text, a 32K ID PREFIX covers only 55.5% of
positions, and the single most-used token is 236779 (top of the range). Selection must be by
FREQUENCY, with an explicit draft_idx -> full_id remap.

CORPUS CHOICE. Built from this repo's own Mojo sources + docs, because that is the ModCon
workload (Mojo-writing questions) and it is local. Stated plainly so the bias is visible: a subset
selected on repo text will over-serve repo text. It must NOT then be validated only on the repo bar
-- that would be the self-referential gate. Hold out a disjoint slice, and treat prose/QA buckets
as the adversarial case.

Emits: ranked token list, coverage curve, and a uint8 mask per N for the GPU A/B.
"""
import collections, json, os, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(os.environ.get("DRAFT_VOCAB_OUT", pathlib.Path.home() / "an earlier branch/draft_vocab"))
OUT.mkdir(parents=True, exist_ok=True)
TOK_DIR = os.environ.get("TOK_DIR", os.path.expanduser("~/models/gemma-4-31b-it"))
VOCAB = 262144
HOLDOUT_FRAC = 0.2

import warnings; warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOK_DIR)

# ── corpus ────────────────────────────────────────────────────────────────────
PATTERNS = ("lib/*.mojo", "*.mojo", "docs/*.md", "tools/*.py", "pyhost/*.py", "perf/**/*.py")
files = []
for pat in PATTERNS:
    files.extend(sorted(ROOT.glob(pat)))
files = [f for f in files if f.is_file() and f.stat().st_size < 2_000_000]
print(f"  corpus files: {len(files)}")

# Deterministic split BEFORE counting, so the holdout never informs selection.
files_sorted = sorted(files, key=lambda p: str(p))
hold_n = max(1, int(len(files_sorted) * HOLDOUT_FRAC))
holdout = files_sorted[::int(1/HOLDOUT_FRAC)][:hold_n]      # every 5th file
train = [f for f in files_sorted if f not in set(holdout)]
print(f"  train {len(train)} files / holdout {len(holdout)} files (disjoint, deterministic)")

def toks(fs):
    out = []
    for f in fs:
        try: txt = f.read_text(errors="ignore")
        except Exception: continue
        out.extend(tok(txt, add_special_tokens=False)["input_ids"])
    return out

train_ids, hold_ids = toks(train), toks(holdout)
print(f"  train {len(train_ids):,} tokens ({len(set(train_ids)):,} distinct)")
print(f"  holdout {len(hold_ids):,} tokens ({len(set(hold_ids)):,} distinct)")

cnt = collections.Counter(train_ids)
ranked = [t for t, _ in cnt.most_common()]

# Specials are mandatory regardless of frequency — a drafter that cannot propose EOS is broken.
specials = {getattr(tok, a) for a in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")}
specials = {s for s in specials if isinstance(s, int) and 0 <= s < VOCAB}
print(f"  mandatory specials: {sorted(specials)}")

# ── coverage curve, measured on the HELD-OUT slice ────────────────────────────
hold_cnt = collections.Counter(hold_ids)
print(f"\n  {'N':>8} {'train cov':>10} {'HOLDOUT cov':>12}   {'head params':>12} {'vs 262144':>10}")
rows = []
for N in (512, 1024, 2048, 4096, 8192, 16384, 32768):
    keep = set(ranked[:N]) | specials
    tr = sum(c for t, c in cnt.items() if t in keep) / max(len(train_ids), 1)
    ho = sum(c for t, c in hold_cnt.items() if t in keep) / max(len(hold_ids), 1)
    p = len(keep) * 5376
    print(f"  {N:>8,} {100*tr:>9.3f}% {100*ho:>11.3f}%   {p/1e9:>10.3f}B {100*p/(VOCAB*5376):>9.1f}%")
    rows.append(dict(N=N, kept=len(keep), train_cov=tr, holdout_cov=ho, head_params=p))
    m = np.zeros(VOCAB, np.uint8)
    m[list(keep)] = 1
    m.tofile(OUT / f"mask_{N}.u8")

(OUT / "ranked.json").write_text(json.dumps(
    dict(ranked_top=ranked[:40000], specials=sorted(specials),
         train_tokens=len(train_ids), holdout_tokens=len(hold_ids), rows=rows)))
print(f"\n  masks + ranked list -> {OUT}")
print("  NOTE: holdout coverage is the honest number — train coverage is fit to the selection.")
