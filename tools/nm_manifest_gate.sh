#!/bin/bash
# Exact ABI manifest gate. Asserts the .so's exported `nomos_` symbol SET matches an expected
# manifest EXACTLY — and that nomos_model_id is present.
#
# WHY A MANIFEST, NOT A COUNT (Codex, 2026-08-17): a count-only check (even "== 86") passes when
# two symbols trade places — one dropped, one added, count unchanged. The dropped-nomos_model_id
# class of regression (2026-08-15) is exactly a symbol vanishing; a set diff catches it, a count
# does not. refresh_build.sh today only asserts NOMOS_SYMS >= 1, which catches almost nothing.
#
# Regenerate the expected manifest deliberately (never auto) when the ABI intentionally changes:
#   nm -D libnomos_kernel.so | awk '$2=="T" && $3 ~ /^nomos_/ {print $3}' | sort > tools/nm_manifest_<profile>.txt
# and record WHY in the commit / PORT_LOG. Baseline `tools/nm_manifest_baseline.txt` = the 86-symbol
# gemma4/muse ABI (2026-08-17). When Codex lands nomos_debug_final_norm + nomos_debug_gdn_state the
# common ABI becomes 88 for ALL profiles (Option A) — regen the manifest to 88 and re-gate all three.
#
#   tools/nm_manifest_gate.sh <libnomos_kernel.so> <expected_manifest.txt>
set -u
SO="${1:?usage: nm_manifest_gate.sh <so> <expected_manifest>}"
EXPECT="${2:?usage: nm_manifest_gate.sh <so> <expected_manifest>}"
[ -f "$SO" ]     || { echo "FAIL: no .so at $SO"; exit 1; }
[ -f "$EXPECT" ] || { echo "FAIL: no expected manifest at $EXPECT"; exit 1; }

ACTUAL="$(mktemp)"
nm -D "$SO" 2>/dev/null | awk '$2=="T" && $3 ~ /^nomos_/ {print $3}' | sort > "$ACTUAL"
na="$(wc -l < "$ACTUAL")"; ne="$(wc -l < "$EXPECT")"

if ! grep -qx nomos_model_id "$ACTUAL"; then
  echo "FAIL: nomos_model_id NOT exported — the front-door identity check (tools/check_model_identity.py) is dead"
  rm -f "$ACTUAL"; exit 1
fi
if ! diff -q "$EXPECT" "$ACTUAL" >/dev/null; then
  echo "FAIL: exported nomos_ symbol set != expected ($na actual vs $ne expected)"
  echo "  ONLY in the build (unexpected new / renamed):"
  comm -13 "$EXPECT" "$ACTUAL" | sed 's/^/    +/'
  echo "  ONLY in expected (DROPPED — the regression this gate exists to catch):"
  comm -23 "$EXPECT" "$ACTUAL" | sed 's/^/    -/'
  rm -f "$ACTUAL"; exit 1
fi
echo "PASS: exact ABI match — $na nomos_ symbols, set-identical to $(basename "$EXPECT"), nomos_model_id present"
rm -f "$ACTUAL"; exit 0
