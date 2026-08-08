#!/bin/bash
# set -o pipefail is LOAD-BEARING: every build command below pipes to `tail`, and without
# pipefail a failed `mojo build` exits 0 through the pipe. Prime hit exactly this on the 5090
# laptop (2026-08-04): the script printed its own failure signal ("perf code present? 0") and
# still exited 0 with the .so untouched. A build script that reports success on failure is the
# same defect family as the BLOCK_ATTN trap.
set -eo pipefail
export PATH=$HOME/.pixi/bin:$PATH
cd "$(cd "$(dirname "$0")" && pwd)"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
case "$(uname -m)" in
  aarch64) CUDA_TARGET_LIB="$CUDA_ROOT/targets/sbsa-linux/lib" ;;
  x86_64) CUDA_TARGET_LIB="$CUDA_ROOT/targets/x86_64-linux/lib" ;;
  *) echo "unsupported host architecture: $(uname -m)" >&2; exit 1 ;;
esac
if [ ! -d "$CUDA_TARGET_LIB" ]; then
  echo "CUDA target library directory not found: $CUDA_TARGET_LIB" >&2
  exit 1
fi
# Rebuilds the .so on the CURRENT checkout. Does NOT switch branches or hard-reset
# (the old `git checkout fork-refactor && git reset --hard` was a footgun that silently
#  discarded local work — removed 2026-06-25). Check out the branch you want yourself.
echo "=== refresh build on $(git rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "=== pixi env sync ==="
pixi install 2>&1 | tail -3
echo "=== gcc shims ==="
gcc -c -fPIC nomos_libc_shim.c -o nomos_libc_shim.o && gcc -c -fPIC nomos_token_cb_stub.c -o nomos_token_cb_stub.o && gcc -c -fPIC -fvisibility=hidden -I"$CUDA_ROOT/include" nomos_cuda_budget.c -o nomos_cuda_budget.o && echo "shims ok"
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
