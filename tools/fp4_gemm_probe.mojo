"""Standalone perf probe for the NVFP4 W4A4 decode GEMM (lib/fp4_gemm.fp4_gemm_kernel).

The 3-second iteration harness for the sm_120 kernel sprint on (RTX PRO 4000
Blackwell, ~672 GB/s). Drives the EXACT kernel the engine calls in decode
(`gpu_fp4_gemm` -> `fp4_gemm_kernel`, the `lib_fp4_gemm_fp4_gemm_kernel...` instance
in nsys/ncu traces: 73.6% of decode GPU time, avg 216 us/instance, ~387 inst/token)
at the real gemma-4-31b M=1 decode shapes (Mpad=16), with REAL random inputs:

  - weights: random E2M1 packed bytes + random e4m3 block scales in the engine's
    blob layout ([e4m3 scales : N*K/16][packed : N*K/2], lib/fp4_act convention);
  - activation: random fp32 [1,K] pushed through the REAL decode act-quant kernel
    (lib/fp4_act.gpu_quant_act_nvfp4, M=16/Mreal=1) -> NVFP4 packed + scales.

DCE guard: constant-input microbenchmarks get dead-code-eliminated. Here
inputs are random, the output buffer is downloaded and checksummed after timing
(printed per shape), and the weighted per-token average is compared against the
engine's known 216 us/instance. If a shape reports wildly faster than the engine
at the same shape, suspect DCE / a dead launch, NOT a speedup.

Per shape it reports: avg us/call, effective GB/s (bytes = NVFP4 weight bytes +
weight scales + act packed/scales + fp32 out), % of 672 GB/s peak, and the
weight-bandwidth-bound time (ratio = how far off the roofline we are; engine
profile says ~3x).

L2 honesty: in the engine every instance reads a DIFFERENT layer's weight (L2-cold);
a naive loop re-reads one buffer and gets L2 hits (qslide showed 111% of DRAM peak).
Default mode rotates across enough weight copies to exceed L2 (total >= 256MB, <=16
copies) so per-call traffic is DRAM-honest like the engine.

Env knobs (the fast iteration loop):
  FP4_PROBE_MS=3000       measured window per shape (ms)
  FP4_PROBE_SHAPE=gateup  substring filter on shape name -> one shape = ~3s round
  FP4_PROBE_FULLOP=1      also time the full engine op (act-quant + GEMM + postscale,
                          lib/fp4_act.gpu_matmul_nvfp4_w4a4_dev — what _mm_dev calls)
  FP4_PROBE_WCOPIES=N     weight copies (0 = auto to beat L2 [default]; 1 = L2-warm)

Build (discrete Blackwell, sm_120a — do NOT copy binaries across boxes):
  pixi run mojo build tools/fp4_gemm_probe.mojo -I . \
      --target-accelerator sm_120a -o build/fp4_gemm_probe \
      -Xlinker -L/usr/local/cuda/lib64 -Xlinker -lcudart -Xlinker -lm

Run (guards GPU0-free first):
  bash tools/run_fp4_probe.sh
"""

from max.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.ffi import external_call

from lib.fp4_gemm import gpu_fp4_gemm, gpu_fp4_gemm_ps_geo, gpu_fp4_gemm_qa
from lib.fp4_act import gpu_quant_act_nvfp4, gpu_matmul_nvfp4_w4a4_dev
from lib.fp4_act import gpu_matmul_nvfp4_w4a4_prequant_dev
from lib.fp4_act import gpu_act_amax_gscale, gpu_quant_act_nvfp4_fast
from lib.fp4_act import postscale_w4a4_kernel

comptime MPAD = 16                  # decode M=1 padded to the MMA m-dim (W4A4_MPAD)
comptime PEAK_GBS = 672.0           # RTX PRO 4000 Blackwell HBM peak
comptime ENGINE_REF_US = 216.0      # engine trace avg us/instance (all shapes mixed)
comptime ENGINE_REF_INST = 387     # engine trace instances/token (probe table = 411:
                                   # trace window likely clips partial tokens)
comptime WARM = 10


# ─────────────────────────────────────────────────────────────────────────
# Decode shape table — gemma-4-31b, M=1 (Mpad=16). K,N per lib/gemma4_layer._mm_dev
# call sites; counts = fp4_gemm instances per decode token (60 layers: 50 sliding +
# 10 full [layer%6==5], lib/gemma4_engine layer table).
# ─────────────────────────────────────────────────────────────────────────
@fieldwise_init
struct ProbeShape(Copyable, Movable):
    var name: String
    var k: Int
    var n: Int
    var count: Int      # instances per decode token


def _xs(state: UInt64) -> UInt64:
    """xorshift64 — fast bulk random fill (values don't affect timing, only DCE/checksum)."""
    var x = state
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x


def _env_bytes(name: String) -> List[UInt8]:
    """getenv via libc (same pattern as lib/engine_init._read_env_bytes)."""
    var nb = List[UInt8](capacity=name.byte_length() + 1)
    for i in range(name.byte_length()):
        nb.append(name.as_bytes()[i])
    nb.append(0)
    var out = List[UInt8]()
    var raw = external_call["getenv", Int64, UnsafePointer[UInt8, origin_of(nb)]](nb.unsafe_ptr())
    if raw == 0:
        return out^
    var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(raw))
    var n = 0
    while p[n] != 0:
        n += 1
    for i in range(n):
        out.append(p[i])
    return out^


def _env_int(name: String, default: Int) -> Int:
    var b = _env_bytes(name)
    var v = 0
    var got = False
    for i in range(len(b)):
        var c = Int(b[i])
        if c < 48 or c > 57:
            break
        v = v * 10 + (c - 48)
        got = True
    return v if got else default


def _name_matches(name: String, filt: List[UInt8]) -> Bool:
    """True when filt is empty or a byte-substring of name."""
    var m = len(filt)
    if m == 0:
        return True
    var n = name.byte_length()
    if m > n:
        return False
    var nb = name.as_bytes()
    for i in range(n - m + 1):
        var ok = True
        for j in range(m):
            if nb[i + j] != filt[j]:
                ok = False
                break
        if ok:
            return True
    return False


def _r2(x: Float64) -> Float64:
    return Float64(Int(x * 100.0 + 0.5)) / 100.0


def _shape_bytes(K: Int, N: Int) -> Int:
    """Effective bytes/call: NVFP4 weight packed + weight e4m3 scales + act packed
    + act scales + fp32 output ([Mpad,N])."""
    return N * (K // 2) + N * (K // 16) + MPAD * (K // 2) + MPAD * (K // 16) + MPAD * N * 4


comptime L2_BEAT_BYTES = 268435456   # 256MB total weight footprint >> Blackwell L2


def run_shape(
    ctx: DeviceContext, sh: ProbeShape, window_ms: Int, fullop: Bool, wcopies_env: Int
) raises -> List[Float64]:
    """Benchmark one decode shape. Returns [bare, ps@NS3, ps@NS4] avg us/call."""
    var K = sh.k
    var N = sh.n
    var KB = K // 2
    var NBK = K // 16
    var nb_w = N * NBK              # weight e4m3 scale bytes (blob prefix)
    var wpk = N * KB                # weight E2M1 packed bytes
    var blob_bytes = nb_w + wpk

    # L2-honest mode: rotate across enough distinct weight buffers that the working
    # set exceeds L2 (the engine is L2-cold per instance — 60 different layers).
    var copies = wcopies_env
    if copies <= 0:
        copies = (L2_BEAT_BYTES + blob_bytes - 1) // blob_bytes
        if copies > 16:
            copies = 16
    if copies < 1:
        copies = 1

    # ── weight blobs: copies x [e4m3 scales : nb_w][E2M1 packed], random (engine layout) ──
    var blob_h = ctx.enqueue_create_host_buffer[DType.uint8](copies * blob_bytes)
    ctx.synchronize()
    var wp = blob_h.unsafe_ptr().bitcast[UInt64]()
    var s = UInt64(0x9E3779B97F4A7C15) ^ (UInt64(K) << 32) ^ UInt64(N)
    var blob_words = blob_bytes // 8
    var scale_words = nb_w // 8     # all our shapes: nb_w % 8 == 0, blob_bytes % 8 == 0
    for i in range(blob_words):
        s = _xs(s)
        if i < scale_words:
            # e4m3 scale bytes constrained to [0x28,0x47] (~0.08..3.75) — realistic
            # magnitude, keeps fp32 accumulators finite for the checksum.
            wp[i] = (s & 0x1F1F1F1F1F1F1F1F) + 0x2828282828282828
        else:
            wp[i] = s               # E2M1 packed: any byte is a valid nibble pair
    for c in range(1, copies):      # replicate: same bytes, distinct addresses (DRAM traffic)
        for i in range(blob_words):
            wp[c * blob_words + i] = wp[i]

    # ── activation: random fp32 row 0 (Mreal=1), pad rows zeroed by the quant kernel ──
    var act_h = ctx.enqueue_create_host_buffer[DType.float32](MPAD * K)
    for i in range(MPAD * K):
        act_h[i] = Float32(0)
    for i in range(K):
        s = _xs(s)
        act_h[i] = Float32(Int((s >> 40) & 0xFFF)) / Float32(1024.0) - Float32(2.0)

    var blob_d = ctx.enqueue_create_buffer[DType.uint8](copies * blob_bytes)
    var act_d = ctx.enqueue_create_buffer[DType.float32](MPAD * K)
    var apk_d = ctx.enqueue_create_buffer[DType.uint8](MPAD * KB)   # act E2M1
    var abs_d = ctx.enqueue_create_buffer[DType.uint8](MPAD * NBK)  # act e4m3 scales
    var agl_d = ctx.enqueue_create_buffer[DType.float32](MPAD)      # act per-token global
    var c_d = ctx.enqueue_create_buffer[DType.float32](MPAD * N)    # raw MMA out (cpad)
    var out_d = ctx.enqueue_create_buffer[DType.float32](MPAD * N)  # full-op post-scaled out
    ctx.enqueue_copy(blob_d, blob_h)
    ctx.enqueue_copy(act_d, act_h)
    c_d.enqueue_fill(Float32(0.0))
    out_d.enqueue_fill(Float32(0.0))
    ctx.synchronize()

    var blob_a = UInt64(Int(blob_d.unsafe_ptr()))   # copy c at blob_a + c*blob_bytes;
    # within a copy: weight scales at offset 0, packed weights at offset nb_w
    var act_a = UInt64(Int(act_d.unsafe_ptr()))
    var apk_a = UInt64(Int(apk_d.unsafe_ptr()))
    var abs_a = UInt64(Int(abs_d.unsafe_ptr()))
    var agl_a = UInt64(Int(agl_d.unsafe_ptr()))
    var c_a = UInt64(Int(c_d.unsafe_ptr()))
    var out_a = UInt64(Int(out_d.unsafe_ptr()))

    # Real decode act-quant (one warp/row, Mreal=1, pad rows -> exact zeros).
    gpu_quant_act_nvfp4(ctx, act_a, apk_a, abs_a, agl_a, MPAD, 1, K)
    ctx.synchronize()

    var window_ns = window_ms * 1_000_000

    # ── timed loop A: bare fp4_gemm (== the ncu `lib_fp4_gemm_fp4_gemm_kernel` inst) ──
    # Rotates the weight copy per call (ci) so L2 can't serve repeat reads.
    var ci = 0
    var warm_n = WARM if WARM > copies else copies
    for _ in range(warm_n):
        var ba = blob_a + UInt64(ci * blob_bytes)
        gpu_fp4_gemm(ctx, c_a, apk_a, ba + UInt64(nb_w), abs_a, ba, MPAD, N, K)
        ci = (ci + 1) % copies
    ctx.synchronize()
    var calls = 0
    var batch = 16
    var t0 = Int(perf_counter_ns())
    while True:
        for _ in range(batch):
            var ba = blob_a + UInt64(ci * blob_bytes)
            gpu_fp4_gemm(ctx, c_a, apk_a, ba + UInt64(nb_w), abs_a, ba, MPAD, N, K)
            ci = (ci + 1) % copies
        ctx.synchronize()
        calls += batch
        var el = Int(perf_counter_ns()) - t0
        if el >= window_ns:
            break
        var per = el // calls
        if per > 0:
            batch = window_ns // (per * 40)     # aim ~40 sync batches per window
            if batch < 8:
                batch = 8
            if batch > 4096:
                batch = 4096
    var avg_us = Float64(Int(perf_counter_ns()) - t0) / Float64(calls) / 1000.0

    var bytes_per = _shape_bytes(K, N)
    var gbps = Float64(bytes_per) / (avg_us * 1000.0)
    var pct = gbps / PEAK_GBS * 100.0
    var bound_us = Float64(bytes_per) / (PEAK_GBS * 1000.0)

    print(
        sh.name, ": K=", K, " N=", N, " x", sh.count, "/tok | ",
        _r2(avg_us), " us/call (", calls, " calls, ", copies, " wcopies) | ",
        _r2(gbps), " GB/s = ", _r2(pct), "% peak | bw-bound ",
        _r2(bound_us), " us -> ", _r2(avg_us / bound_us), "x off roofline | ",
        _r2(avg_us * Float64(sh.count) / 1000.0), " ms/tok",
    )

    # ── DCE guard: zero the output, run once more (copy 0), download + checksum row 0 ──
    c_d.enqueue_fill(Float32(0.0))
    gpu_fp4_gemm(ctx, c_a, apk_a, blob_a + UInt64(nb_w), abs_a, blob_a, MPAD, N, K)
    ctx.synchronize()
    var c_h = ctx.enqueue_create_host_buffer[DType.float32](MPAD * N)
    ctx.enqueue_copy(c_h, c_d)
    ctx.synchronize()
    var csum = Float64(0)
    var nz = 0
    for j in range(N):                          # row 0 = the live decode row
        var v = c_h[j]
        csum += Float64(abs(v))
        if v != Float32(0.0):
            nz += 1
    print("    DCE guard: out row0 |sum|=", _r2(csum), " nonzero=", nz, "/", N,
          ("  OK" if nz > N // 2 else "  *** SUSPECT: output mostly zero ***"))

    # ── R2 parity lanes. Reference = the explicit pre-R2 chain (one-warp
    # quant_act_nvfp4 + bare fp4_gemm + the REAL postscale kernel) into out_d.
    # Candidates (both must match BITWISE, incl the gscale):
    #   A) SHIPPED decode path: fast quant (amax + parallel encode) + fused-postscale
    #      GEMM (gpu_fp4_gemm_ps)                      -> out2_d
    #   B) EXPERIMENTAL in-GEMM act-quant (gpu_fp4_gemm_qa; probe-only, R3) -> out3_d
    var out2_d = ctx.enqueue_create_buffer[DType.float32](MPAD * N)
    var out3_d = ctx.enqueue_create_buffer[DType.float32](MPAD * N)
    var agl2_d = ctx.enqueue_create_buffer[DType.float32](MPAD)
    var apk2_d = ctx.enqueue_create_buffer[DType.uint8](MPAD * KB)
    var abs2_d = ctx.enqueue_create_buffer[DType.uint8](MPAD * NBK)
    out2_d.enqueue_fill(Float32(0.0))
    out3_d.enqueue_fill(Float32(0.0))
    agl2_d.enqueue_fill(Float32(-1.0))
    apk2_d.enqueue_fill(UInt8(0xAA))     # junk: parallel encode must overwrite fully
    abs2_d.enqueue_fill(UInt8(0xAA))
    var out2_a = UInt64(Int(out2_d.unsafe_ptr()))
    var out3_a = UInt64(Int(out3_d.unsafe_ptr()))
    var agl2_a = UInt64(Int(agl2_d.unsafe_ptr()))
    var apk2_a = UInt64(Int(apk2_d.unsafe_ptr()))
    var abs2_a = UInt64(Int(abs2_d.unsafe_ptr()))
    var wg_par = Float32(0.01)
    # reference chain (exactly the pre-R2 kernels)
    gpu_quant_act_nvfp4(ctx, act_a, apk_a, abs_a, agl_a, MPAD, 1, K)
    gpu_fp4_gemm(ctx, c_a, apk_a, blob_a + UInt64(nb_w), abs_a, blob_a, MPAD, N, K)
    var pk = ctx.compile_function[postscale_w4a4_kernel]()
    ctx.enqueue_function(
        pk,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(out_a)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(c_a)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(agl_a)),
        wg_par, 1, N,
        grid_dim=(N + 255) // 256, block_dim=256,
    )
    # candidate A (shipped): parallel fast quant into SEPARATE scratch + ps GEMM
    # R3: ALL stage geometries must be bit-equal (NS/TK change stage buffering
    # only; k-ascending accumulation order is geometry-invariant).
    var tk6_ok = (K % 384) == 0     # TK=6 stage = 384 k-values
    gpu_quant_act_nvfp4_fast(ctx, act_a, apk2_a, abs2_a, agl2_a, K)
    gpu_fp4_gemm_ps_geo(ctx, out2_a, apk2_a, blob_a + UInt64(nb_w), abs2_a, blob_a,
                        agl2_a, wg_par, N, K, 3, 4)
    var out4_d = ctx.enqueue_create_buffer[DType.float32](MPAD * N)
    out4_d.enqueue_fill(Float32(0.0))
    var out4_a = UInt64(Int(out4_d.unsafe_ptr()))
    gpu_fp4_gemm_ps_geo(ctx, out4_a, apk2_a, blob_a + UInt64(nb_w), abs2_a, blob_a,
                        agl2_a, wg_par, N, K, 4, 4)
    var out6_d = ctx.enqueue_create_buffer[DType.float32](MPAD * N)
    out6_d.enqueue_fill(Float32(0.0))
    var out6_a = UInt64(Int(out6_d.unsafe_ptr()))
    if tk6_ok:
        gpu_fp4_gemm_ps_geo(ctx, out6_a, apk2_a, blob_a + UInt64(nb_w), abs2_a, blob_a,
                            agl2_a, wg_par, N, K, 3, 6)
    # candidate B (experimental): in-GEMM act-quant (reads raw act + gscale only)
    gpu_fp4_gemm_qa(ctx, out3_a, act_a, blob_a + UInt64(nb_w), blob_a, agl2_a, wg_par, N, K)
    ctx.synchronize()
    var pref_h = ctx.enqueue_create_host_buffer[DType.float32](MPAD * N)
    var pcan_h = ctx.enqueue_create_host_buffer[DType.float32](MPAD * N)
    var pns4_h = ctx.enqueue_create_host_buffer[DType.float32](MPAD * N)
    var ptk6_h = ctx.enqueue_create_host_buffer[DType.float32](MPAD * N)
    var pfus_h = ctx.enqueue_create_host_buffer[DType.float32](MPAD * N)
    var agl_h = ctx.enqueue_create_host_buffer[DType.float32](MPAD)
    var agl2_h = ctx.enqueue_create_host_buffer[DType.float32](MPAD)
    var apk_h = ctx.enqueue_create_host_buffer[DType.uint8](MPAD * KB)
    var apk2_h = ctx.enqueue_create_host_buffer[DType.uint8](MPAD * KB)
    var abs_h = ctx.enqueue_create_host_buffer[DType.uint8](MPAD * NBK)
    var abs2_h = ctx.enqueue_create_host_buffer[DType.uint8](MPAD * NBK)
    ctx.enqueue_copy(pref_h, out_d)
    ctx.enqueue_copy(pcan_h, out2_d)
    ctx.enqueue_copy(pns4_h, out4_d)
    ctx.enqueue_copy(ptk6_h, out6_d)
    ctx.enqueue_copy(pfus_h, out3_d)
    ctx.enqueue_copy(agl_h, agl_d)
    ctx.enqueue_copy(agl2_h, agl2_d)
    ctx.enqueue_copy(apk_h, apk_d)
    ctx.enqueue_copy(apk2_h, apk2_d)
    ctx.enqueue_copy(abs_h, abs_d)
    ctx.enqueue_copy(abs2_h, abs2_d)
    ctx.synchronize()
    if agl_h.unsafe_ptr().bitcast[UInt32]()[0] != agl2_h.unsafe_ptr().bitcast[UInt32]()[0]:
        raise Error("GSCALE FAIL: amax micro-kernel gscale != one-warp quant gscale")
    var qb_mm = 0                        # fast-quant bytes vs one-warp quant bytes
    for j in range(MPAD * KB):
        if apk_h[j] != apk2_h[j]:
            qb_mm += 1
    for j in range(MPAD * NBK):
        if abs_h[j] != abs2_h[j]:
            qb_mm += 1
    if qb_mm != 0:
        raise Error(String("FAST-QUANT BYTES FAIL: ") + String(qb_mm)
                    + " scratch bytes differ (packed+scales, incl pad rows)")
    var pmm = 0
    var nmm = 0
    var tmm = 0
    var fmm = 0
    for j in range(N):      # every path writes exactly M*N = N floats
        var rw = pref_h.unsafe_ptr().bitcast[UInt32]()[j]
        if rw != pcan_h.unsafe_ptr().bitcast[UInt32]()[j]:
            pmm += 1
        if rw != pns4_h.unsafe_ptr().bitcast[UInt32]()[j]:
            nmm += 1
        if tk6_ok and rw != ptk6_h.unsafe_ptr().bitcast[UInt32]()[j]:
            tmm += 1
        if rw != pfus_h.unsafe_ptr().bitcast[UInt32]()[j]:
            fmm += 1
    if pmm != 0:
        raise Error(String("FASTQUANT+PS(NS3) PARITY FAIL: ") + String(pmm) + " of "
                    + String(N) + " outputs differ bitwise")
    if nmm != 0:
        raise Error(String("PS NS4 PARITY FAIL: ") + String(nmm) + " of "
                    + String(N) + " outputs differ bitwise")
    if tmm != 0:
        raise Error(String("PS TK6 PARITY FAIL: ") + String(tmm) + " of "
                    + String(N) + " outputs differ bitwise")
    if fmm != 0:
        raise Error(String("QA-FUSED PARITY FAIL: ") + String(fmm) + " of "
                    + String(N) + " outputs differ bitwise")
    print("    parity: fastquant bytes OK + ps-NS3 OK + ps-NS4 OK + ps-TK6 ",
          ("OK" if tk6_ok else "n/a(K%384)"), " + QA(exp) OK (",
          N, " outputs bit-equal, gscale bit-equal)")

    # ── R3 geometry timing: bare vs ps@NS3/TK4 (R2 ship) vs ps@NS4 vs ps@TK6,
    # INTERLEAVED round-robin batches in ONE window so all lanes see identical
    # clock/thermal state (a sequential A-then-B measurement charges B for the
    # boost-clock decay — R1/R2 lesson). All rotate the same weight-copy ring
    # (L2-cold). Gates: ps3 <=1.05x bare (lmhead <=1.02x, R2 continuity); the
    # R3 decision numbers are ps4/ps3 and tk6/ps3 per shape (ship a variant
    # only where it wins BOTH runs; regress-gate 3%). ──
    var qci = 0
    for _ in range(warm_n):
        var ba = blob_a + UInt64(qci * blob_bytes)
        gpu_fp4_gemm_ps_geo(ctx, out2_a, apk2_a, ba + UInt64(nb_w), abs2_a, ba, agl2_a, wg_par, N, K, 3, 4)
        gpu_fp4_gemm_ps_geo(ctx, out4_a, apk2_a, ba + UInt64(nb_w), abs2_a, ba, agl2_a, wg_par, N, K, 4, 4)
        if tk6_ok:
            gpu_fp4_gemm_ps_geo(ctx, out6_a, apk2_a, ba + UInt64(nb_w), abs2_a, ba, agl2_a, wg_par, N, K, 3, 6)
        qci = (qci + 1) % copies
    ctx.synchronize()
    var nlanes = 4 if tk6_ok else 3
    var ib_ns = 0
    var ip_ns = 0
    var i4_ns = 0
    var i6_ns = 0
    var icalls = 0
    var ibatch = 16
    while ib_ns + ip_ns + i4_ns + i6_ns < nlanes * window_ns:
        var t1 = Int(perf_counter_ns())
        for _ in range(ibatch):
            var ba = blob_a + UInt64(qci * blob_bytes)
            gpu_fp4_gemm(ctx, c_a, apk2_a, ba + UInt64(nb_w), abs2_a, ba, MPAD, N, K)
            qci = (qci + 1) % copies
        ctx.synchronize()
        var t2 = Int(perf_counter_ns())
        for _ in range(ibatch):
            var ba = blob_a + UInt64(qci * blob_bytes)
            gpu_fp4_gemm_ps_geo(ctx, out2_a, apk2_a, ba + UInt64(nb_w), abs2_a, ba, agl2_a, wg_par, N, K, 3, 4)
            qci = (qci + 1) % copies
        ctx.synchronize()
        var t3 = Int(perf_counter_ns())
        for _ in range(ibatch):
            var ba = blob_a + UInt64(qci * blob_bytes)
            gpu_fp4_gemm_ps_geo(ctx, out4_a, apk2_a, ba + UInt64(nb_w), abs2_a, ba, agl2_a, wg_par, N, K, 4, 4)
            qci = (qci + 1) % copies
        ctx.synchronize()
        var t4 = Int(perf_counter_ns())
        if tk6_ok:
            for _ in range(ibatch):
                var ba = blob_a + UInt64(qci * blob_bytes)
                gpu_fp4_gemm_ps_geo(ctx, out6_a, apk2_a, ba + UInt64(nb_w), abs2_a, ba, agl2_a, wg_par, N, K, 3, 6)
                qci = (qci + 1) % copies
            ctx.synchronize()
        var t5 = Int(perf_counter_ns())
        ib_ns += t2 - t1
        ip_ns += t3 - t2
        i4_ns += t4 - t3
        i6_ns += t5 - t4
        icalls += ibatch
        var iper = (ib_ns + ip_ns + i4_ns + i6_ns) // (nlanes * icalls)
        if iper > 0:
            ibatch = window_ns // (iper * 40)
            if ibatch < 8:
                ibatch = 8
            if ibatch > 4096:
                ibatch = 4096
    var ibare = Float64(ib_ns) / Float64(icalls) / 1000.0
    var ips = Float64(ip_ns) / Float64(icalls) / 1000.0
    var ips4 = Float64(i4_ns) / Float64(icalls) / 1000.0
    var itk6 = Float64(i6_ns) / Float64(icalls) / 1000.0 if tk6_ok else ips
    var ps_gate_x = 1.02 if N >= 262144 else 1.05
    print("    ps-GEMM (interleaved): bare ", _r2(ibare),
          " us | ps-NS3 ", _r2(ips), " us (", _r2(ips / ibare), "x bare)",
          ("  GATE-OK" if ips <= ibare * ps_gate_x else "  *** GATE-FAIL ***"))
    print("    R3 geometry: ps-NS4 ", _r2(ips4), " us = ", _r2(ips4 / ips),
          "x ps-NS3 | ps-TK6 ",
          (_r2(itk6) if tk6_ok else Float64(-1)), " us = ",
          (_r2(itk6 / ips) if tk6_ok else Float64(-1)),
          "x ps-NS3 (", icalls, " calls each; TK6 -1 = n/a)")
    _ = out4_d^
    _ = out6_d^

    # ── quant timing: one-warp (pre-R2) vs parallel fast (amax + encode) ──
    var oqcalls = 0
    var oqt0 = Int(perf_counter_ns())
    while True:
        for _ in range(64):
            gpu_quant_act_nvfp4(ctx, act_a, apk_a, abs_a, agl_a, MPAD, 1, K)
        ctx.synchronize()
        oqcalls += 64
        if Int(perf_counter_ns()) - oqt0 >= window_ns // 6:
            break
    var oqavg = Float64(Int(perf_counter_ns()) - oqt0) / Float64(oqcalls) / 1000.0
    var fqcalls = 0
    var fqt0 = Int(perf_counter_ns())
    while True:
        for _ in range(64):
            gpu_quant_act_nvfp4_fast(ctx, act_a, apk2_a, abs2_a, agl2_a, K)
        ctx.synchronize()
        fqcalls += 64
        if Int(perf_counter_ns()) - fqt0 >= window_ns // 6:
            break
    var fqavg = Float64(Int(perf_counter_ns()) - fqt0) / Float64(fqcalls) / 1000.0
    print("    act-quant: one-warp ", _r2(oqavg), " us -> fast ", _r2(fqavg),
          " us (amax+encode, ", _r2(oqavg / fqavg), "x)")

    # ── experimental in-GEMM QA timing (NOT dispatched; R3 reference) ──
    var xcalls = 0
    var xt0 = Int(perf_counter_ns())
    while True:
        for _ in range(16):
            var ba = blob_a + UInt64(qci * blob_bytes)
            gpu_fp4_gemm_qa(ctx, out3_a, act_a, ba + UInt64(nb_w), ba, agl2_a, wg_par, N, K)
            qci = (qci + 1) % copies
        ctx.synchronize()
        xcalls += 16
        if Int(perf_counter_ns()) - xt0 >= window_ns // 6:
            break
    var xavg = Float64(Int(perf_counter_ns()) - xt0) / Float64(xcalls) / 1000.0
    print("    QA in-GEMM (exp, not dispatched): ", _r2(xavg), " us = ",
          _r2(xavg / avg_us), "x bare GEMM")
    _ = out2_d^
    _ = out3_d^
    _ = agl2_d^
    _ = apk2_d^
    _ = abs2_d^

    # ── optional timed loop B: full engine op (act-quant + GEMM + post-scale) ──
    if fullop:
        for _ in range(warm_n):
            gpu_matmul_nvfp4_w4a4_dev(
                ctx, UInt64(0), out_a, act_a, blob_a + UInt64(ci * blob_bytes),
                Float32(0.01), Float32(0.0), apk_a, abs_a, agl_a, UInt64(0), UInt64(0),
                c_a, 1, MPAD, K, N,
            )
            ci = (ci + 1) % copies
        ctx.synchronize()
        var fcalls = 0
        var fbatch = 16
        var ft0 = Int(perf_counter_ns())
        while True:
            for _ in range(fbatch):
                gpu_matmul_nvfp4_w4a4_dev(
                    ctx, UInt64(0), out_a, act_a, blob_a + UInt64(ci * blob_bytes),
                    Float32(0.01), Float32(0.0), apk_a, abs_a, agl_a, UInt64(0), UInt64(0),
                    c_a, 1, MPAD, K, N,
                )
                ci = (ci + 1) % copies
            ctx.synchronize()
            fcalls += fbatch
            var fel = Int(perf_counter_ns()) - ft0
            if fel >= window_ns:
                break
            var fper = fel // fcalls
            if fper > 0:
                fbatch = window_ns // (fper * 40)
                if fbatch < 8:
                    fbatch = 8
                if fbatch > 4096:
                    fbatch = 4096
        var favg = Float64(Int(perf_counter_ns()) - ft0) / Float64(fcalls) / 1000.0
        print("    full op (quant+gemm+postscale): ", _r2(favg), " us/call (",
              fcalls, " calls) | wrapper overhead ", _r2(favg - avg_us), " us")

    _ = blob_d^
    _ = act_d^
    _ = apk_d^
    _ = abs_d^
    _ = agl_d^
    _ = c_d^
    _ = out_d^
    var res: List[Float64] = [avg_us, ibare, ips, ips4, itk6]
    return res^


def main() raises:
    var ctx = DeviceContext()
    var window_ms = _env_int("FP4_PROBE_MS", 3000)
    var fullop = _env_int("FP4_PROBE_FULLOP", 0) != 0
    var wcopies = _env_int("FP4_PROBE_WCOPIES", 0)   # 0 = auto (L2-honest)
    var filt = _env_bytes("FP4_PROBE_SHAPE")

    print("== fp4_gemm decode perf probe (gemma-4-31b, M=1/Mpad=16, sm_120) ==")
    print("device compute=", Float64(ctx.default_device_info.compute),
          " | peak ", PEAK_GBS, " GB/s | window ", window_ms, " ms/shape | warm ", WARM)
    print("engine ref: ", ENGINE_REF_US, " us/inst avg, ~", ENGINE_REF_INST,
          " inst/tok (73.6% of decode GPU time)")

    var shapes = List[ProbeShape]()
    # q-proj, 50 sliding layers (NH=32 x HD=256)
    shapes.append(ProbeShape("qslide", 5376, 8192, 50))
    # k-proj + v-proj, sliding (NKV=16 x HD=256) — 2 GEMMs x 50 layers
    shapes.append(ProbeShape("kvslide", 5376, 4096, 100))
    # q-proj, 10 full-attn layers (NH=32 x FULL_HD=512)
    shapes.append(ProbeShape("qfull", 5376, 16384, 10))
    # k-proj, full (4 kv-heads x 512; V=K on full layers -> no v-proj)
    shapes.append(ProbeShape("kfull", 5376, 2048, 10))
    # o-proj, sliding (l_qd=8192 -> D)
    shapes.append(ProbeShape("oslide", 8192, 5376, 50))
    # o-proj, full (l_qd=16384 -> D)
    shapes.append(ProbeShape("ofull", 16384, 5376, 10))
    # gate + up (D -> FF) — 2 GEMMs x 60 layers, the bandwidth heavyweight
    shapes.append(ProbeShape("gateup", 5376, 21504, 120))
    # tail-wave discriminator (H1, R3 pre-arbitration): N=17920 -> grid.x=560 =
    # exactly 2.0 waves at 4 blocks/SM x 70 SM. NOT an engine shape (count=1,
    # negligible in the weighted summary). If GB/s here jumps to ~down's level,
    # gateup's deficit is tail-wave quantization, not K-depth.
    shapes.append(ProbeShape("gateup2w", 5376, 17920, 1))
    # down (FF -> D) x 60 layers
    shapes.append(ProbeShape("down", 21504, 5376, 60))
    # lm-head (D -> VOCAB=262144), once per token (engine_decode Feature A path)
    shapes.append(ProbeShape("lmhead", 5376, 262144, 1))

    var wsum_us = Float64(0)
    var wsum_ps3 = Float64(0)
    var wsum_ps4 = Float64(0)
    var wsum_tk6 = Float64(0)
    var wbytes = 0
    var wcount = 0
    var ran = 0
    for i in range(len(shapes)):
        if not _name_matches(shapes[i].name, filt):
            continue
        var avgs = run_shape(ctx, shapes[i], window_ms, fullop, wcopies)
        wsum_us += avgs[0] * Float64(shapes[i].count)
        wsum_ps3 += avgs[2] * Float64(shapes[i].count)
        wsum_ps4 += avgs[3] * Float64(shapes[i].count)
        wsum_tk6 += avgs[4] * Float64(shapes[i].count)   # = ps3 where TK6 n/a
        wbytes += _shape_bytes(shapes[i].k, shapes[i].n) * shapes[i].count
        wcount += shapes[i].count
        ran += 1

    if ran > 1:
        var wavg = wsum_us / Float64(wcount)
        print("== per-token summary (", wcount, " fp4_gemm inst) ==")
        print("weighted avg ", _r2(wavg), " us/inst (engine ref ", ENGINE_REF_US,
              ") | GEMM total ", _r2(wsum_us / 1000.0), " ms/tok | bw-bound ",
              _r2(Float64(wbytes) / (PEAK_GBS * 1.0e6)), " ms/tok")
        print("R3 decode-path totals (interleaved lanes): ps-NS3 ",
              _r2(wsum_ps3 / 1000.0), " ms/tok | ps-NS4 ",
              _r2(wsum_ps4 / 1000.0), " ms/tok (", _r2(wsum_ps4 / wsum_ps3),
              "x) | tk6-dispatch ", _r2(wsum_tk6 / 1000.0), " ms/tok (",
              _r2(wsum_tk6 / wsum_ps3), "x)")
        if wavg < ENGINE_REF_US * 0.33:
            print("*** SANITY: probe >3x faster than engine at same shapes — "
                  "suspect DCE/dead launches before believing a speedup ***")
