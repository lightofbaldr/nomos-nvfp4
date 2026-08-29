# NVFP4 TC GEMM — ported to OUR nightly (1.0.0b2.dev2026050406, b1-era syntax).
# Same inline-PTX mma.sync.mxf4nvf4 as fp4_tc_gemm_nvfp4.mojo (the fork-API spike);
# only the host launch changes: comptime->alias, and the fused enqueue_function[fn]
# -> the kernel's two-step compile_function -> enqueue_function(handle,...) idiom
# (see lib/q4_weights.mojo gpu_dequant_q4). Confirms the MMA assembles + runs
# bit-exact on the SINGLE nightly we standardize everything on. Build:
#   pixi run mojo run --target-accelerator sm_121a tools/fp4_nvfp4_stock_test.mojo

from max.gpu.host import DeviceContext
from std.gpu.primitives import thread_idx, block_idx
from std.sys._assembly import inlined_assembly
from std.sys import _RegisterPackType
from std.memory import UnsafePointer
from std.random import random_ui64

alias M = 64
alias N = 64
alias K = 256
alias KB = K // 2        # FP4 bytes per row
alias NB = K // 16       # scale blocks per row (group size 16)


def fp4_tc_nvfp4(
    c_out: UnsafePointer[Float32, MutAnyOrigin],
    a_bytes: UnsafePointer[UInt8, MutAnyOrigin],
    b_bytes: UnsafePointer[UInt8, MutAnyOrigin],
    a_sc: UnsafePointer[UInt8, MutAnyOrigin],
    b_sc: UnsafePointer[UInt8, MutAnyOrigin],
):
    var wm = Int(block_idx.y)
    var wn = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    var gid = lane >> 2
    var tid = lane & 3
    var mrow = wm * 16
    var ncol = wn * 8

    @parameter
    def lda(p: UnsafePointer[UInt8, MutAnyOrigin], row: Int, k: Int) -> UInt32:
        return (p + (row * KB + (k >> 1))).bitcast[UInt32]()[0]

    @parameter
    def lds(p: UnsafePointer[UInt8, MutAnyOrigin], row: Int, blk: Int) -> UInt32:
        return (p + (row * NB + blk)).bitcast[UInt32]()[0]

    var c0 = Float32(0)
    var c1 = Float32(0)
    var c2 = Float32(0)
    var c3 = Float32(0)
    for k0 in range(0, K, 64):
        var a0 = lda(a_bytes, mrow + gid, k0 + tid * 8)
        var a1 = lda(a_bytes, mrow + gid + 8, k0 + tid * 8)
        var a2 = lda(a_bytes, mrow + gid, k0 + tid * 8 + 32)
        var a3 = lda(a_bytes, mrow + gid + 8, k0 + tid * 8 + 32)
        var b0 = lda(b_bytes, ncol + gid, k0 + tid * 8)
        var b1 = lda(b_bytes, ncol + gid, k0 + tid * 8 + 32)

        var blk = k0 // 16
        var sca = UInt32(0)
        if tid == 0:
            sca = lds(a_sc, mrow + gid, blk)
        elif tid == 1:
            sca = lds(a_sc, mrow + gid + 8, blk)
        var scb = UInt32(0)
        if tid == 0:
            scb = lds(b_sc, ncol + gid, blk)

        var r = inlined_assembly[
            (
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
                ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {$0,$1,$2,$3},"
                " {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, {$14}, {0, 0},"
                " {$15}, {0, 0};"
            ),
            _RegisterPackType[Float32, Float32, Float32, Float32],
            constraints="=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
        ](a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sca, scb)
        c0 = r[0]
        c1 = r[1]
        c2 = r[2]
        c3 = r[3]

    c_out[(mrow + gid) * N + ncol + 2 * tid] = c0
    c_out[(mrow + gid) * N + ncol + 2 * tid + 1] = c1
    c_out[(mrow + gid + 8) * N + ncol + 2 * tid] = c2
    c_out[(mrow + gid + 8) * N + ncol + 2 * tid + 1] = c3


def e2m1_decode(nib: Int) -> Float32:
    var mag = SIMD[DType.float32, 8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)[nib & 7]
    return -mag if (nib & 8) != 0 else mag


def main() raises:
    with DeviceContext() as ctx:
        var a_h = ctx.enqueue_create_host_buffer[DType.uint8](M * KB)
        var b_h = ctx.enqueue_create_host_buffer[DType.uint8](N * KB)
        var asc_h = ctx.enqueue_create_host_buffer[DType.uint8](M * NB)
        var bsc_h = ctx.enqueue_create_host_buffer[DType.uint8](N * NB)
        for i in range(M * KB):
            a_h[i] = UInt8(Int(random_ui64(0, 255)))
        for i in range(N * KB):
            b_h[i] = UInt8(Int(random_ui64(0, 255)))
        var scbytes = SIMD[DType.uint8, 4](0x30, 0x38, 0x40, 0x48)
        var scvals = SIMD[DType.float32, 4](0.5, 1.0, 2.0, 4.0)
        var av = ctx.enqueue_create_host_buffer[DType.float32](M * NB)
        var bv = ctx.enqueue_create_host_buffer[DType.float32](N * NB)
        for i in range(M * NB):
            var s = Int(random_ui64(0, 3))
            asc_h[i] = scbytes[s]
            av[i] = scvals[s]
        for i in range(N * NB):
            var s = Int(random_ui64(0, 3))
            bsc_h[i] = scbytes[s]
            bv[i] = scvals[s]

        var cref = ctx.enqueue_create_host_buffer[DType.float32](M * N)
        for m in range(M):
            for n in range(N):
                var acc = Float32(0)
                for kb in range(KB):
                    var k = kb * 2
                    var blk = k // 16
                    var sa = av[m * NB + blk]
                    var sb = bv[n * NB + blk]
                    var ab = Int(a_h[m * KB + kb])
                    var bb = Int(b_h[n * KB + kb])
                    acc += sa * sb * e2m1_decode(ab & 0xF) * e2m1_decode(bb & 0xF)
                    acc += sa * sb * e2m1_decode(ab >> 4) * e2m1_decode(bb >> 4)
                cref[m * N + n] = acc

        var a_d = ctx.enqueue_create_buffer[DType.uint8](M * KB)
        var b_d = ctx.enqueue_create_buffer[DType.uint8](N * KB)
        var asc_d = ctx.enqueue_create_buffer[DType.uint8](M * NB)
        var bsc_d = ctx.enqueue_create_buffer[DType.uint8](N * NB)
        var c_d = ctx.enqueue_create_buffer[DType.float32](M * N)
        var c_h = ctx.enqueue_create_host_buffer[DType.float32](M * N)
        ctx.enqueue_copy(a_d, a_h)
        ctx.enqueue_copy(b_d, b_h)
        ctx.enqueue_copy(asc_d, asc_h)
        ctx.enqueue_copy(bsc_d, bsc_h)

        # OUR nightly's two-step launch (vs the fork spike's fused enqueue_function[fn]).
        var kern = ctx.compile_function[fp4_tc_nvfp4, fp4_tc_nvfp4]()
        ctx.enqueue_function(
            kern,
            c_d.unsafe_ptr(), a_d.unsafe_ptr(), b_d.unsafe_ptr(),
            asc_d.unsafe_ptr(), bsc_d.unsafe_ptr(),
            grid_dim=(N // 8, M // 16), block_dim=32,
        )
        ctx.synchronize()
        ctx.enqueue_copy(c_h, c_d)
        ctx.synchronize()

        var bad = 0
        var maxrel = Float32(0)
        for i in range(M * N):
            var d = abs(c_h[i] - cref[i])
            var denom = abs(cref[i]) + 1.0
            if d / denom > maxrel:
                maxrel = d / denom
            if d > 1.0:
                bad += 1
        print("NVFP4 TC GEMM", M, "x", N, "x", K, " per-block scales: bad=", bad, "/", M * N, " maxrel=", maxrel)
        print("kernel C[0..3]=", c_h[0], c_h[1], c_h[2], c_h[3])
        print("ref    C[0..3]=", cref[0], cref[1], cref[2], cref[3])
        if bad == 0:
            print("COMPLETE NVFP4 TENSOR-CORE GEMM CORRECT (per-block scales) ON STOCK NIGHTLY")
        _ = a_d^
        _ = b_d^
        _ = asc_d^
        _ = bsc_d^
        _ = c_d^
