#!/usr/bin/env bash
# The KV session store's directories under NOMOS_KVSTORE_DIR: list them, and remove the ones you name (docs/RUNBOOK.md §7).
#
#   tools/kvstore_dirs.sh [list]                       every muse-* directory under the root: size, last write, which is live.
#                                                      Changes nothing.
#   tools/kvstore_dirs.sh remove <dir>...              removes the directories named (a muse-* name under the root, or a
#                                                      path). Refuses the live one, anything that is not a muse-* directory
#                                                      directly under the root, and a directory whose store is open.
# Options (before or after the command):
#   --root DIR   the store root (NOMOS_KVSTORE_DIR). Default: the live directory's parent, else $NOMOS_KVSTORE_DIR, else
#                ~/nomos_data/kvstore, the launcher's default.
#   --port N     the serve's port, for /health. Default: $NOMOS_PORT, else 8091.
#   --log FILE   the daemon's log. Default: daemon.log in the repo, which `serve_nomos.sh up` writes.
#
# Which directory is live comes from the running process, never from a spelling: the serve's /health (its kvstore.dir, a
# canonical path) when the serve answers; otherwise the newest "[kvstore] dir=" line of the daemon's log, resolved against
# the repo (the daemon's working directory under serve_nomos.sh). Every path is compared canonically (realpath), so a
# trailing slash, a relative or a symlinked NOMOS_KVSTORE_DIR names the live directory one way. With neither source,
# `remove` refuses. And a directory is removed only while holding its .lock, which an open store holds: a store that is
# running, live by any source or not, is never removed.
# There is deliberately no "remove everything but the live one": another directory may belong to another build, weights or
# numerics you still want (a session there is used again when that configuration runs).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "kvstore_dirs: $*" >&2; exit 2; }
usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

CMD="" ROOT_ARG="" PORT="${NOMOS_PORT:-8091}" LOG="$REPO/daemon.log"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --root) [ $# -ge 2 ] || die "--root needs a directory"; ROOT_ARG="$2"; shift 2 ;;
    --port) [ $# -ge 2 ] || die "--port needs a number"; PORT="$2"; shift 2 ;;
    --log)  [ $# -ge 2 ] || die "--log needs a file"; LOG="$2"; shift 2 ;;
    -h|--help) usage ;;
    list|remove) [ -z "$CMD" ] || die "one command at a time"; CMD="$1"; shift ;;
    --) shift; ARGS+=("$@"); break ;;
    -*) die "unknown option $1" ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
CMD="${CMD:-list}"
[ "$CMD" = list ] && [ ${#ARGS[@]} -gt 0 ] && die "list takes no directories (did you mean remove?)"
[ "$CMD" = remove ] && [ ${#ARGS[@]} -eq 0 ] && die "remove needs the directories to remove, by name; it never picks them"

# ── which directory is live ──────────────────────────────────────────────────────────────────────────────────────────
LIVE="" SOURCE=""
health="$(curl -s --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null)"
if [ -n "$health" ]; then
  # "status<TAB>dir" from /health's kvstore object; nothing when the serve predates it.
  kv="$(printf '%s' "$health" | python3 -c '
import json, sys
try:
    kv = json.load(sys.stdin).get("kvstore")
except Exception:
    kv = None
if isinstance(kv, dict):
    print(kv.get("status") or "", kv.get("dir") or "", sep="\t")' 2>/dev/null)"
  if [ -n "$kv" ]; then
    SOURCE="the serve's /health on :$PORT (store ${kv%%$'\t'*})"
    d="${kv#*$'\t'}"
    [ -n "$d" ] && LIVE="$(realpath -m -- "$d")"
  fi
fi
if [ -z "$SOURCE" ] && [ -f "$LOG" ]; then
  d="$(sed -n 's/^\[kvstore\] dir=//p' "$LOG" | tail -n 1)"
  if [ -n "$d" ]; then
    SOURCE="the newest [kvstore] dir= line of $LOG"
    LIVE="$(cd "$REPO" && realpath -m -- "$d")"
  fi
fi

# ── the root ─────────────────────────────────────────────────────────────────────────────────────────────────────────
if [ -n "$ROOT_ARG" ]; then
  ROOT="$(realpath -m -- "$ROOT_ARG")"
elif [ -n "$LIVE" ]; then
  ROOT="$(dirname -- "$LIVE")"
else
  ROOT="$(cd "$REPO" && realpath -m -- "${NOMOS_KVSTORE_DIR:-$HOME/nomos_data/kvstore}")"
fi

# ── list ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
if [ "$CMD" = list ]; then
  echo "root: $ROOT"
  if [ -n "$LIVE" ]; then
    echo "live: $LIVE   (from $SOURCE)"
    [ "$(dirname -- "$LIVE")" = "$ROOT" ] || echo "      (not under this root: nothing listed below is live)"
  elif [ -n "$SOURCE" ]; then
    echo "live: none   ($SOURCE: no store is open)"
  else
    echo "live: UNKNOWN: the serve on :$PORT did not answer /health and $LOG names no store directory. 'remove' refuses until one does."
  fi
  n=0
  for d in "$ROOT"/muse-*; do
    [ -d "$d" ] || continue
    n=$((n + 1))
    c="$(realpath -m -- "$d")"
    bytes="$(command find "$c" -type f -printf '%s\n' 2>/dev/null | awk '{s += $1} END {print s + 0}')"
    newest="$(command find "$c" -maxdepth 1 -type f -name '*.nkvs' -printf '%T@\n' 2>/dev/null | sort -n | tail -n 1)"
    newest="${newest:-$(stat -c %Y -- "$c")}"
    mark=""
    [ "$c" = "$LIVE" ] && mark="   LIVE"
    [ "$c" = "$ROOT/$(basename -- "$d")" ] || mark="$mark   (a link to $c)"
    printf '%10s  %s  %s%s\n' "$(numfmt --to=iec --suffix=B "$bytes")" "$(date -d "@${newest%.*}" '+%Y-%m-%d %H:%M')" \
      "$(basename -- "$d")" "$mark"
  done
  [ "$n" -gt 0 ] || echo "(no muse-* directories under $ROOT)"
  echo "Nothing was changed. To remove directories, name them: tools/kvstore_dirs.sh remove <muse-...>..."
  exit 0
fi

# ── remove ───────────────────────────────────────────────────────────────────────────────────────────────────────────
[ -n "$SOURCE" ] || die "refusing to remove anything: no authoritative live directory (the serve on :$PORT did not answer
/health, and $LOG names no store directory). Start the serve, or point --port / --log at the running one."
targets=() bad=0
for a in "${ARGS[@]}"; do
  p="$a"
  case "$a" in */*) ;; muse-*) p="$ROOT/$a" ;; esac           # a bare name is a directory under the root
  if ! c="$(realpath -e -- "$p" 2>/dev/null)"; then
    echo "refused: $a: no such directory" >&2; bad=1; continue
  fi
  if [ ! -d "$c" ]; then
    echo "refused: $a: not a directory" >&2; bad=1
  elif [ "$(dirname -- "$c")" != "$ROOT" ]; then
    echo "refused: $a ($c): not directly under the store root $ROOT" >&2; bad=1
  elif [[ "$(basename -- "$c")" != muse-* ]]; then
    echo "refused: $a ($c): not a muse-* store directory" >&2; bad=1
  elif [ -n "$LIVE" ] && [ "$c" = "$LIVE" ]; then
    echo "refused: $a ($c): the LIVE store directory, from $SOURCE" >&2; bad=1
  elif [ -n "$LIVE" ] && [[ "$LIVE/" == "$c/"* ]]; then
    echo "refused: $a ($c): contains the LIVE store directory $LIVE, from $SOURCE" >&2; bad=1
  else
    targets+=("$c")
  fi
done
[ "$bad" = 0 ] || die "nothing was removed"
status=0
for c in "${targets[@]}"; do
  # Under the directory's own lock, which a store open on it holds while it runs: that store's directory is never removed
  # (a directory containing the live one is refused above). A store that starts on it mid-removal is not stopped by this.
  ( exec 9>>"$c/.lock" || exit 3; flock -n 9 || exit 75; rm -rf -- "$c" ) 2>/dev/null
  case $? in
    0)  echo "removed: $c" ;;
    75) echo "refused: $c: a running store holds it (its .lock is taken); nothing removed there" >&2; status=1 ;;
    *)  echo "failed: $c: could not be removed (see its permissions); it may be partly removed" >&2; status=1 ;;
  esac
done
exit "$status"
