#!/bin/bash
# set -o pipefail is LOAD-BEARING: every build command below pipes to `tail`, and without
# pipefail a failed `mojo build` exits 0 through the pipe. Prime hit exactly this on the 5090
# laptop (2026-08-04): the script printed its own failure signal ("perf code present? 0") and
# still exited 0 with the .so untouched. A build script that reports success on failure is the
# same defect family as the BLOCK_ATTN trap.
set -eo pipefail
export PATH=$HOME/.pixi/bin:$PATH
cd "$(cd "$(dirname "$0")" && pwd)"
# CUDA is RESOLVED, not assumed. The old code hardcoded ${CUDA_HOME:-/usr/local/cuda} and
# died on any box without a SYSTEM CUDA install — which is the normal case here, because pixi
# provides CUDA inside the environment (cuda-cudart-dev / libcublas-dev in pixi.toml). That
# made `pixi install && ./refresh_build.sh`, the two commands in the README quick-start, fail
# on a clean clone for exactly the discrete/x86 audience this repo targets.
#
# Probe for the ARTIFACT (libcudart), never for a directory name: a path can exist and be
# empty, and "directory is present" is not "the library is here".
case "$(uname -m)" in
  aarch64) CUDA_TRIPLE="sbsa-linux" ;;
  x86_64)  CUDA_TRIPLE="x86_64-linux" ;;
  *) echo "unsupported host architecture: $(uname -m)" >&2; exit 1 ;;
esac
PIXI_ENV="$(pwd)/.pixi/envs/default"
CUDA_TARGET_LIB=""
for _c in "${CUDA_HOME:+$CUDA_HOME/targets/$CUDA_TRIPLE/lib}" "${CUDA_HOME:+$CUDA_HOME/lib64}" \
          "$PIXI_ENV/targets/$CUDA_TRIPLE/lib" "$PIXI_ENV/lib" \
          "/usr/local/cuda/targets/$CUDA_TRIPLE/lib" "/usr/local/cuda/lib64"; do
  [ -n "$_c" ] || continue
  if [ -f "$_c/libcudart.so" ] || ls "$_c"/libcudart.so.* >/dev/null 2>&1; then
    CUDA_TARGET_LIB="$_c"; break
  fi
done
if [ -z "$CUDA_TARGET_LIB" ]; then
  echo "libcudart not found. Looked under CUDA_HOME (${CUDA_HOME:-unset}), the pixi env" >&2
  echo "($PIXI_ENV), and /usr/local/cuda. Run \`pixi install\` first — pixi.toml declares" >&2
  echo "cuda-cudart-dev/libcublas-dev, so a synced env is normally enough." >&2
  exit 1
fi
# Headers for the gcc shim: same probe, on the artifact (cuda_runtime.h).
CUDA_INCLUDE=""
for _i in "${CUDA_HOME:+$CUDA_HOME/targets/$CUDA_TRIPLE/include}" "${CUDA_HOME:+$CUDA_HOME/include}" \
          "$PIXI_ENV/targets/$CUDA_TRIPLE/include" "$PIXI_ENV/include" \
          "/usr/local/cuda/targets/$CUDA_TRIPLE/include" "/usr/local/cuda/include"; do
  [ -n "$_i" ] && [ -f "$_i/cuda_runtime.h" ] && { CUDA_INCLUDE="$_i"; break; }
done
[ -n "$CUDA_INCLUDE" ] || { echo "cuda_runtime.h not found (looked beside libcudart)" >&2; exit 1; }
echo "cuda lib: $CUDA_TARGET_LIB"
echo "cuda inc: $CUDA_INCLUDE"
# Rebuilds the .so on the CURRENT checkout. Does NOT switch branches or hard-reset
# (the old `git checkout fork-refactor && git reset --hard` was a footgun that silently
#  discarded local work — removed 2026-06-25). Check out the branch you want yourself.
echo "=== refresh build on $(git rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "=== pixi env sync ==="
pixi install 2>&1 | tail -3
echo "=== gcc shims ==="
gcc -c -fPIC nomos_libc_shim.c -o nomos_libc_shim.o && gcc -c -fPIC nomos_token_cb_stub.c -o nomos_token_cb_stub.o && gcc -c -fPIC -fvisibility=hidden -I"$CUDA_INCLUDE" nomos_cuda_budget.c -o nomos_cuda_budget.o && echo "shims ok"
echo "=== build libnomos_kernel for current GPU ($(date +%T)) ==="
pixi run mojo build --emit shared-lib nomos_ffi.mojo -o libnomos_kernel.so \
  -Xlinker nomos_libc_shim.o -Xlinker nomos_token_cb_stub.o -Xlinker nomos_cuda_budget.o \
  -Xlinker "-L$CUDA_TARGET_LIB" \
  -Xlinker -lcudart -Xlinker -lcublas -Xlinker -lm 2>&1 | tail -8
echo "=== verify ==="
ls -la libnomos_kernel.so
readelf -d libnomos_kernel.so 2>/dev/null | grep -E "NEEDED.*(cublas|cudart)" || echo "WARN cuda link?"
# HARD GATE, not telemetry: zero exported nomos_ symbols == the build did not produce a usable
# .so, and this script must say so with its exit code (the old `|| true` here swallowed it).
NOMOS_SYMS=$(pixi run nm -D libnomos_kernel.so 2>/dev/null | grep -c nomos_ || true)
echo "perf code present? $NOMOS_SYMS"
if [ "${NOMOS_SYMS:-0}" -lt 1 ]; then
  echo "BUILD FAILED: no nomos_ symbols exported from libnomos_kernel.so" >&2
  exit 1
fi
echo "=== REFRESH BUILD DONE $(date) ==="
