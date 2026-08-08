"""D2 G3 debug — W4A4 GEMM correctness probe across the EAGLE-3 head shapes.

For each (K, N): random act[K] + weight[N,K]; weight host-encoded to the NVFP4
blob; activation quantized ON GPU by the real decode path
(gpu_quant_act_nvfp4_dec); then BOTH operands are dequantized on the host and
the first CHK output rows are compared against the exact host dot product —
so quantization error cancels and any relative error is a GEMM-path bug.
Routes ps-fused vs fallback via NOMOS_FP4_QAFUSE, exactly like the live path.

Run: pixi run mojo run -I . tools/fp4_shape_probe.mojo
"""
from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.random import random_ui64
from lib.fp4_act import gpu_quant_act_nvfp4_dec, gpu_matmul_nvfp4_w4a4_prequant_dev
from lib.fp4_act import _f32_to_e4m3_byte, _e2m1_nibble
from lib.fp4_weights import e4m3_decode, e2m1_mag
from lib.cuda import cuda_malloc, cuda_free, cuda_memcpy, cuda_sync, cuda_last_error
from std.ffi import external_call, c_int

comptime CHK = 48   # output rows checked against the host reference


def checked_memcpy(dst: UInt64, src: UInt64, bytes: Int, kind: Int, tag: StaticString):
    var rc = external_call["cudaMemcpy", c_int, UInt64, UInt64, UInt64, c_int](
        dst, src, UInt64(bytes), c_int(kind))
    if rc != c_int(0):
        print("  MEMCPY FAIL", tag, " rc=", Int(rc), " bytes=", bytes, " kind=", kind)


def probe(ctx: DeviceContext, K: Int, N: Int) raises:
    var KB = K // 2
    var NB = K // 16
    var nb_w = (N * K) // 16

    # host act + weight
    var act = List[Float32](capacity=K)
    for _ in range(K):
        act.append(Float32(Int(random_ui64(0, 2000))) / 200.0 - 5.0)
    var wf = List[Float32](capacity=N * K)
    for _ in range(N * K):
        wf.append(Float32(Int(random_ui64(0, 2000))) / 1000.0 - 1.0)

    # host-encode weight blob [e4m3 : nb_w][packed : N*K/2] + host dequant mirror
    var wamax = Float32(0.0)
    for i in range(N * K):
        var a = abs(wf[i])
        if a > wamax:
            wamax = a
    var wglobal = wamax / Float32(6.0 * 448.0)
    var wblob = List[UInt8](capacity=nb_w + (N * K) // 2)
    for _ in range(nb_w + (N * K) // 2):
        wblob.append(0)
    var wdeq = List[Float32](capacity=N * K)
    for _ in range(N * K):
        wdeq.append(0.0)
    for b in range(nb_w):
        var bamax = Float32(0.0)
        for e in range(16):
            var a = abs(wf[b * 16 + e])
            if a > bamax:
                bamax = a
        var bsb = _f32_to_e4m3_byte(bamax / (Float32(6.0) * wglobal))
        wblob[b] = UInt8(bsb)
        var denom = wglobal * e4m3_decode(bsb)
        if denom == Float32(0.0):
            denom = Float32(1.0)
        for j in range(8):
            var n0 = _e2m1_nibble(wf[b * 16 + 2 * j] / denom)
            var n1 = _e2m1_nibble(wf[b * 16 + 2 * j + 1] / denom)
            wblob[nb_w + b * 8 + j] = UInt8((n1 << 4) | n0)
            var s0 = Float32(-1.0) if (n0 & 8) != 0 else Float32(1.0)
            var s1 = Float32(-1.0) if (n1 & 8) != 0 else Float32(1.0)
            wdeq[b * 16 + 2 * j] = denom * s0 * e2m1_mag(n0 & 7)
            wdeq[b * 16 + 2 * j + 1] = denom * s1 * e2m1_mag(n1 & 7)

    # device buffers
    var d_act = cuda_malloc(K * 4)
    var d_w = cuda_malloc(nb_w + (N * K) // 2)
    var d_pk = cuda_malloc(16 * KB)
    var d_bs = cuda_malloc(16 * NB)
    var d_gl = cuda_malloc(16 * 4)
    var d_cpad = cuda_malloc(16 * N * 4)
    var d_out = cuda_malloc(N * 4)
    checked_memcpy(d_act, UInt64(Int(act.unsafe_ptr())), K * 4, 1, "act_h2d")
    checked_memcpy(d_w, UInt64(Int(wblob.unsafe_ptr())), nb_w + (N * K) // 2, 1, "w_h2d")
    cuda_sync()

    # real decode path: act quant once + prequant GEMM
    gpu_quant_act_nvfp4_dec(ctx, d_act, d_pk, d_bs, d_gl, K)
    ctx.synchronize()
    var err_q = cuda_last_error()
    gpu_matmul_nvfp4_w4a4_prequant_dev(ctx, d_out, d_w, wglobal, d_pk, d_bs, d_gl, d_cpad, 1, K, N)
    ctx.synchronize()
    var err_g = cuda_last_error()
    if err_q != 0 or err_g != 0:
        print("  CUDA ERR: quant=", err_q, " gemm=", err_g)

    # download quantized act + dequant on host
    var pk = List[UInt8](capacity=KB)
    for _ in range(KB):
        pk.append(0)
    var bs = List[UInt8](capacity=NB)
    for _ in range(NB):
        bs.append(0)
    var gl = List[Float32](capacity=1)
    gl.append(0.0)
    checked_memcpy(UInt64(Int(pk.unsafe_ptr())), d_pk, KB, 2, "pk_d2h")
    checked_memcpy(UInt64(Int(bs.unsafe_ptr())), d_bs, NB, 2, "bs_d2h")
    checked_memcpy(UInt64(Int(gl.unsafe_ptr())), d_gl, 4, 2, "gl_d2h")
    var outv = List[Float32](capacity=N)
    for _ in range(N):
        outv.append(0.0)
    checked_memcpy(UInt64(Int(outv.unsafe_ptr())), d_out, N * 4, 2, "out_d2h")
    cuda_sync()

    var adeq = List[Float32](capacity=K)
    for _ in range(K):
        adeq.append(0.0)
    for b in range(NB):
        var denom = gl[0] * e4m3_decode(Int(bs[b]))
        for j in range(8):
            var byte = Int(pk[b * 8 + j])
            var n0 = byte & 0xF
            var n1 = (byte >> 4) & 0xF
            var s0 = Float32(-1.0) if (n0 & 8) != 0 else Float32(1.0)
            var s1 = Float32(-1.0) if (n1 & 8) != 0 else Float32(1.0)
            adeq[b * 16 + 2 * j] = denom * s0 * e2m1_mag(n0 & 7)
            adeq[b * 16 + 2 * j + 1] = denom * s1 * e2m1_mag(n1 & 7)

    # host reference over the first CHK rows (dequant x dequant — exact)
    var max_rel = Float32(0.0)
    var worst_row = -1
    for n in range(CHK):
        var acc = Float32(0.0)
        for k in range(K):
            acc += adeq[k] * wdeq[n * K + k]
        var d = abs(outv[n] - acc)
        var denom2 = abs(acc)
        if denom2 < Float32(1.0):
            denom2 = Float32(1.0)
        var rel = d / denom2
        if rel > max_rel:
            max_rel = rel
            worst_row = n
    var nz = 0
    var first_nz = -1
    for n in range(N):
        if outv[n] != Float32(0.0):
            nz += 1
            if first_nz < 0:
                first_nz = n
    print("K=", K, " N=", N, "  max_rel(first", CHK, "rows)=", max_rel,
          "  worst_row=", worst_row, "  out[0]=", outv[0],
          "  nonzero=", nz, "/", N, " first_nz=", first_nz)
    if nz == 0:
        # read-side check: is the device weight blob holding the uploaded bytes?
        var wchk = List[UInt8](capacity=4096)
        for _ in range(4096):
            wchk.append(0)
        checked_memcpy(UInt64(Int(wchk.unsafe_ptr())), d_w, 4096, 2, "wchk_d2h")
        cuda_sync()
        var wmis = 0
        for i in range(4096):
            if wchk[i] != wblob[i]:
                wmis += 1
        # and the packed region head
        var pchk = List[UInt8](capacity=1024)
        for _ in range(1024):
            pchk.append(0)
        checked_memcpy(UInt64(Int(pchk.unsafe_ptr())), d_w + UInt64(nb_w), 1024, 2, "pchk_d2h")
        cuda_sync()
        var pmis = 0
        for i in range(1024):
            if pchk[i] != wblob[nb_w + i]:
                pmis += 1
        print("   READBACK: w-scale head mismatches=", wmis, "/4096  packed head mismatches=", pmis, "/1024")
        # relaunch in place: does the SAME launch succeed the 2nd time?
        gpu_matmul_nvfp4_w4a4_prequant_dev(ctx, d_out, d_w, wglobal, d_pk, d_bs, d_gl, d_cpad, 1, K, N)
        ctx.synchronize()
        checked_memcpy(UInt64(Int(outv.unsafe_ptr())), d_out, N * 4, 2, "out2_d2h")
        cuda_sync()
        var nz2 = 0
        for n in range(N):
            if outv[n] != Float32(0.0):
                nz2 += 1
        print("   RELAUNCH: nonzero=", nz2, "/", N, " out[0]=", outv[0])
        # sentinel write: fill d_out via H2D, relaunch, see if kernel overwrites
        for n in range(N):
            outv[n] = Float32(777.0)
        checked_memcpy(d_out, UInt64(Int(outv.unsafe_ptr())), N * 4, 1, "sent_h2d")
        cuda_sync()
        gpu_matmul_nvfp4_w4a4_prequant_dev(ctx, d_out, d_w, wglobal, d_pk, d_bs, d_gl, d_cpad, 1, K, N)
        ctx.synchronize()
        checked_memcpy(UInt64(Int(outv.unsafe_ptr())), d_out, N * 4, 2, "out3_d2h")
        cuda_sync()
        var n777 = 0
        for n in range(N):
            if outv[n] == Float32(777.0):
                n777 += 1
        print("   SENTINEL: still-777=", n777, "/", N, " out[0]=", outv[0])

    cuda_free(d_act); cuda_free(d_w); cuda_free(d_pk); cuda_free(d_bs)
    cuda_free(d_gl); cuda_free(d_cpad); cuda_free(d_out)


def main() raises:
    with DeviceContext() as ctx:
        # engine shapes (known-good in the live decode) then the drafter's;
        # each twice to split shape-dependence from call-order nondeterminism
        for rep in range(2):
            print("── rep", rep, "──")
            probe(ctx, 5376, 8192)     # engine q-proj
            probe(ctx, 5376, 21504)    # engine gate/up
            probe(ctx, 21504, 5376)    # engine down / drafter down
            probe(ctx, 8192, 5376)     # engine o / drafter o
            probe(ctx, 10752, 8192)    # drafter q
            probe(ctx, 10752, 4096)    # drafter k/v
            probe(ctx, 16128, 5376)    # drafter fc
            probe(ctx, 5376, 32000)    # drafter lm_head
