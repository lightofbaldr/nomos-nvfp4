#!/bin/bash
# Provision a rented discrete box (Vast PRO 6000 / B200 / any x86 Blackwell) from R2.
#
# WHY THIS IS GATED SO HARD — the silent-corruption trap:
#   gemma4_engine mmaps embed_tokens_weight.bin (5.6 GB, fp32) for INPUT token embedding; only
#   the LM-HEAD readout uses embed_tokens_weight.nvfp4. If the .bin is absent or truncated, every
#   input embeds to ZERO -> all-zero logits -> every token id = 0. That does NOT crash. It
#   presents as ~100% draft acceptance AND an L1-self-lossless PASS, because base and spec both
#   degenerate identically, while being 0/12 against any real reference.
#   A PARTIAL SYNC IS WORSE THAN NO SYNC: it yields a confident, fully "gated", fake result.
# So this script verifies against a manifest and refuses to leave a half-provisioned box behind.
#
# Usage:  provision_from_r2.sh [dest_root]        (default ~/nomos_data)
# Needs:  rclone with an `r2:` remote configured.
set -uo pipefail
DEST="${1:-$HOME/nomos_data}"
R2="r2:lofb-data/nomos-deploy"
W="$DEST/gemma-4-31b"
D="$DEST/dflash/dflash-gemma-4-31b-flat"
T="${TOK_DEST:-$HOME/models/gemma-4-31b-it}"

command -v rclone >/dev/null || { echo "ABORT: rclone not installed"; exit 1; }
rclone listremotes 2>/dev/null | grep -q '^r2:' || { echo "ABORT: no r2: remote configured"; exit 1; }

echo "═══ 1. PULL ═══"
mkdir -p "$W" "$D" "$T"
rclone copy "$R2/gemma-4-31b-nvfp4/"       "$W" --transfers 16 --checkers 32 --stats 30s --stats-one-line
rclone copy "$R2/dflash-gemma-4-31b-flat/" "$D" --transfers 16 --checkers 32 --stats 30s --stats-one-line
rclone copy "$R2/tokenizer-gemma-4-31b-it/" "$T" --transfers 8

echo "═══ 2. VERIFY against MANIFEST ═══"
M="$W/MANIFEST.nvfp4.txt"
[ -f "$M" ] || { echo "  ABORT: manifest missing — cannot certify this set"; exit 1; }
MISSING=0; BADSIZE=0; N=0
while IFS=$'\t' read -r sha bytes name; do
  case "$sha" in \#*|"") continue;; esac
  N=$((N+1))
  if [ ! -f "$W/$name" ]; then echo "  MISSING: $name"; MISSING=$((MISSING+1))
  elif [ "$(stat -c%s "$W/$name")" != "$bytes" ]; then
    echo "  SIZE MISMATCH: $name (have $(stat -c%s "$W/$name"), want $bytes)"; BADSIZE=$((BADSIZE+1))
  fi
done < "$M"
echo "  checked $N files: $MISSING missing, $BADSIZE wrong size"

echo "═══ 3. CRITICAL-FILE GATE ═══"
FAIL=0
for c in embed_tokens_weight.bin embed_tokens_weight.nvfp4; do
  want=$(grep -P "\t$c\$" "$M" | cut -f2)
  if [ ! -f "$W/$c" ]; then
    echo "  *** $c ABSENT — would give all-zero logits that FAKE a lossless pass ***"; FAIL=1
  elif [ "$(stat -c%s "$W/$c")" != "$want" ]; then
    echo "  *** $c TRUNCATED ($(stat -c%s "$W/$c") vs $want) — same failure mode ***"; FAIL=1
  else
    printf "  OK  %-28s %.2f GB\n" "$c" "$(echo "$(stat -c%s "$W/$c")/1e9" | bc -l)"
  fi
done
[ -f "$T/chat_template.jinja" ] && echo "  OK  chat_template.jinja" || { echo "  *** chat_template.jinja ABSENT — serve will not format prompts ***"; FAIL=1; }
DQ8=$(ls "$D"/*.q8 2>/dev/null | wc -l)
echo "  drafter .q8 files: $DQ8 (discrete boxes need q8; .q4 MMU-faults on separate VRAM)"
[ "$DQ8" -gt 0 ] || { echo "  *** no q8 drafter — spec-decode cannot run on a discrete card ***"; FAIL=1; }

if [ "$MISSING" -ne 0 ] || [ "$BADSIZE" -ne 0 ] || [ "$FAIL" -ne 0 ]; then
  echo ""
  echo "PROVISION FAILED — do NOT run the kernel against this set. Re-run to resume the pull."
  exit 1
fi

cat > "$HOME/.nomos_host_env" <<EOF
export WEIGHTS="$W/"
export TOK_DIR="$T"
export DFLASH_DIR="$D/"
export NOMOS_PORT=8090
EOF
echo ""
echo "PROVISION OK — $N files verified, critical gate passed."
echo "  wrote ~/.nomos_host_env (WEIGHTS has the REQUIRED trailing slash)"
echo "  next: build on THIS box (the kernel bakes one GPU arch; never copy a .so between arches)"
