#!/usr/bin/env python3
"""Refuse to run a kernel against weights it was not built for.

One .so serves exactly ONE model: the geometry in lib/model_profiles/*.mojo is
compile-time (it sizes ABI-visible buffers and appears as static layout shapes). So
pointing a Muse-built kernel at Gemma-4 weights cannot work. The question is only
whether it fails at the front door with a useful sentence, or deep inside nomos_init
with a useless one.

It used to be the second. On 2026-08-15 a Muse-built .so was pointed at gemma-4-31b/
and the engine printed

    [engine] model = Muse-Glimmer-30B-text  D = 6656 ... vocab = 202048

-- the COMPILED-IN identity, while WEIGHTS said gemma-4-31b -- then died with
"nomos_init failed" and a stray "[WARN] Empty or missing nvfp4: .../lm_head_weight.nvfp4".
Nothing in that output says "wrong kernel for these weights". This tool says exactly that.

We ask the KERNEL what it is (nomos_model_id, exported from the compiled profile)
rather than trusting the filename or a build log, because the filename is the thing
most likely to be stale.

    python3 tools/check_model_identity.py --so libnomos_kernel.so --weights ~/nomos_data/gemma-4-31b
"""
from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
from pathlib import Path

# THE CHECK IS STRUCTURAL, NOT NOMINAL. Compare GEOMETRY, because geometry is what
# actually has to match -- the kernel's D/VOCAB/TOTAL_LAYERS are compile-time and size
# real buffers. Model NAMES are cosmetic and inconsistent across converters
# ("Muse-Glimmer-30B-text" vs "muse-glimmer-30b" vs "gemma-4-31b-it"), and the Muse
# manifest carries no `model` key at all -- so a name-based check would have been
# INCONCLUSIVE on the exact pair it exists to separate.
#
# The id -> geometry map is PARSED FROM THE TRACKED PROFILES rather than duplicated
# here, so it cannot drift from what the kernel was compiled with.
PROFILES_DIR = Path(__file__).resolve().parent.parent / "lib" / "model_profiles"

# manifest key -> profile alias. Converters disagree on names; accept both spellings.
GEOMETRY_KEYS = {
    "hidden": "D", "hidden_size": "D",
    "vocab": "VOCAB", "vocab_size": "VOCAB",
    "layers": "TOTAL_LAYERS", "num_layers": "TOTAL_LAYERS",
}


def load_profiles() -> dict[int, dict]:
    """id -> {name, D, VOCAB, TOTAL_LAYERS} parsed from lib/model_profiles/*.mojo."""
    out = {}
    for f in sorted(PROFILES_DIR.glob("*.mojo")):
        vals, name, pid = {}, None, None
        for line in f.read_text().splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            m = re.match(r'alias\s+(\w+)\s*=\s*"?([^"#]+?)"?\s*(?:#.*)?$', line)
            if not m:
                continue
            k, v = m.group(1), m.group(2).strip()
            if k == "MODEL_ID":
                pid = int(v)
            elif k == "MODEL_NAME":
                name = v
            elif k in ("D", "VOCAB", "TOTAL_LAYERS") and v.isdigit():
                vals[k] = int(v)
        if pid is not None:
            out[pid] = {"name": name or f.stem, "profile": f.stem, **vals}
    return out


def manifest_geometry(weights: Path) -> tuple[dict, str | None, str]:
    """Return (geometry dict, model string or None, which file it came from)."""
    for name in ("manifest.json", "convert_manifest.json", "nomos_manifest.json"):
        p = weights / name
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            return {}, None, f"{p} (unparseable: {e})"
        geo = {}
        for mk, alias in GEOMETRY_KEYS.items():
            v = d.get(mk)
            if isinstance(v, int) and alias not in geo:
                geo[alias] = v
        model = next((d[k] for k in ("model", "model_name", "architecture")
                      if isinstance(d.get(k), str)), None)
        return geo, model, str(p)
    return {}, None, f"{weights}/(manifest.json|convert_manifest.json) (absent)"


def kernel_profile(so: Path) -> tuple[int | None, str]:
    # dlopen needs a resolvable path: a bare "libnomos_kernel.so" is searched on the
    # loader path, not the cwd, and fails with a misleading "No such file".
    try:
        lib = ctypes.CDLL(str(so.resolve()))
    except OSError as e:
        return None, f"cannot dlopen: {e}"
    fn = getattr(lib, "nomos_model_id", None)
    if fn is None:
        # A kernel built before this export existed. Do not guess from the filename —
        # say so, and let the caller decide.
        return None, "no nomos_model_id export (kernel predates the identity check)"
    fn.restype = ctypes.c_int32
    return int(fn()), "nomos_model_id()"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--warn-only", action="store_true",
                    help="report but exit 0 (for wiring into a harness incrementally)")
    a = ap.parse_args()

    so, weights = Path(a.so).expanduser(), Path(a.weights).expanduser()
    profiles = load_profiles()
    kid, ksrc = kernel_profile(so)
    geo, mstr, msrc = manifest_geometry(weights)
    prof = profiles.get(kid) if kid is not None else None

    print(f"  kernel  : {so}")
    print(f"            profile = {prof['profile'] if prof else 'UNKNOWN'}"
          f"{'  (' + prof['name'] + ')' if prof else ''}  (via {ksrc})")
    print(f"  weights : {weights}")
    print(f"            model = {mstr or '(unnamed)'}  (via {msrc})")

    # Compare only the fields BOTH sides actually carry. A manifest that omits `vocab`
    # should narrow the check, never silently widen it to a pass.
    if prof is None or not geo:
        print("  VERDICT : INCONCLUSIVE — could not read both sides.")
        return 0 if a.warn_only else 2

    compared = {k: (geo[k], prof.get(k)) for k in geo if prof.get(k) is not None}
    if not compared:
        print("  VERDICT : INCONCLUSIVE — no overlapping geometry fields to compare.")
        return 0 if a.warn_only else 2

    bad = {k: v for k, v in compared.items() if v[0] != v[1]}
    for k, (want, got) in sorted(compared.items()):
        print(f"            {k:<13} weights={want:<8} kernel={got:<8} {'MISMATCH' if want != got else 'ok'}")

    if bad:
        other = next((q for q in profiles.values()
                      if all(geo.get(k) == q.get(k) for k in geo if q.get(k) is not None)), None)
        print(f"  VERDICT : MISMATCH — this kernel is compiled for '{prof['profile']}' and "
              f"CANNOT load these weights.")
        if other:
            print(f"            Rebuild with:  bash refresh_build.sh --model {other['profile']}")
        else:
            print("            No profile in lib/model_profiles/ matches these weights.")
        return 0 if a.warn_only else 1

    print(f"  VERDICT : OK — geometry matches profile '{prof['profile']}' on "
          f"{len(compared)} field(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
