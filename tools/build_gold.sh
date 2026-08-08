#!/bin/bash
# Gold (RTX PRO 4000 Blackwell, x86, sm_120a) build of libnomos_kernel.so.
# The GB10 scripts (refresh_build.sh/step0_build.sh) hardcode the sbsa-linux (ARM)
# CUDA lib path; on Gold the x86 libs live at /usr/local/cuda/lib64. Kernels are
# JIT-compiled for the present device via ctx.compile_function at runtime, so the
# .so link itself needs no --target-accelerator (that is only for standalone
# tools/*.mojo probe binaries).
set -e
export PATH="$HOME/.pixi/bin:$PATH"
cd "$(cd "$(dirname "$0")/.." && pwd)"
echo "=== gold build on $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD) ($(date +%T)) ==="
gcc -c -fPIC nomos_libc_shim.c -o nomos_libc_shim.o
gcc -c -fPIC nomos_token_cb_stub.c -o nomos_token_cb_stub.o
echo "shims ok"
OUT="${1:-libnomos_kernel.so}"
pixi run mojo build --emit shared-lib nomos_ffi.mojo -o "$OUT" \
  -Xlinker nomos_libc_shim.o -Xlinker nomos_token_cb_stub.o \
  -Xlinker -L/usr/local/cuda/lib64 \
  -Xlinker -lcudart -Xlinker -lcublas -Xlinker -lm 2>&1 | tail -20
echo "BUILD_EXIT=${PIPESTATUS[0]}"
ls -la "$OUT"
readelf -d "$OUT" 2>/dev/null | grep -E "NEEDED.*(cublas|cudart)" && echo "LINK_OK" || echo "LINK_MISSING"
echo "=== gold build done $(date +%T) ==="
