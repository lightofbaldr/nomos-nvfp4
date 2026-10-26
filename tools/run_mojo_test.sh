#!/bin/bash
# Build and run one Mojo test, linked exactly as refresh_build.sh links the kernel (its "gcc shims" and
# "build libnomos_kernel" steps): the libc shim, the token-callback stub and the CUDA budget object, plus
# libcudart/libcublas from the same CUDA root it discovers.
# Usage: bash tools/run_mojo_test.sh tools/<name>_test.mojo   (run from anywhere; works in the repo root)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PATH=$HOME/.pixi/bin:$PATH
TEST="$1"
case "$(uname -m)" in
  aarch64) CUDA_ARCH_DIR="sbsa-linux" ;;
  x86_64)  CUDA_ARCH_DIR="x86_64-linux" ;;
  *) echo "unsupported host architecture: $(uname -m)" >&2; exit 1 ;;
esac
CUDA_ROOT=""
for c in "${CUDA_HOME:-}" "$REPO/.pixi/envs/default" /usr/local/cuda; do
  [ -n "$c" ] || continue
  if ls "$c/targets/$CUDA_ARCH_DIR/lib"/libcudart.so* >/dev/null 2>&1; then CUDA_ROOT="$c"; break; fi
done
[ -n "$CUDA_ROOT" ] || { echo "FATAL: no libcudart for $CUDA_ARCH_DIR (same search as refresh_build.sh)" >&2; exit 1; }
CUDA_TARGET_LIB="$CUDA_ROOT/targets/$CUDA_ARCH_DIR/lib"
CUDA_INC=""
for i in "$CUDA_ROOT/targets/$CUDA_ARCH_DIR/include" "$CUDA_ROOT/include"; do
  [ -f "$i/cuda_runtime.h" ] && { CUDA_INC="$i"; break; }
done
[ -n "$CUDA_INC" ] || { echo "FATAL: cuda_runtime.h not found under $CUDA_ROOT" >&2; exit 1; }
mkdir -p build/mojo_test
gcc -c -fPIC nomos_libc_shim.c -o build/mojo_test/nomos_libc_shim.o
gcc -c -fPIC nomos_token_cb_stub.c -o build/mojo_test/nomos_token_cb_stub.o
gcc -c -fPIC -fvisibility=hidden -I"$CUDA_INC" nomos_cuda_budget.c -o build/mojo_test/nomos_cuda_budget.o
OUT="build/mojo_test/$(basename "$TEST" .mojo)"
pixi run --no-install mojo build -I . "$TEST" -o "$OUT" \
  -Xlinker build/mojo_test/nomos_libc_shim.o -Xlinker build/mojo_test/nomos_token_cb_stub.o \
  -Xlinker build/mojo_test/nomos_cuda_budget.o \
  -Xlinker "-L$CUDA_TARGET_LIB" -Xlinker "-L$CUDA_ROOT/lib" \
  -Xlinker -rpath -Xlinker "$CUDA_TARGET_LIB" -Xlinker -rpath -Xlinker "$CUDA_ROOT/lib" \
  -Xlinker -lcudart -Xlinker -lcublas -Xlinker -lm
"./$OUT"
