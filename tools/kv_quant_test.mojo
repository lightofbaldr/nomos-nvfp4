"""Round-trip test for lib/kv_cache_quant INT8 KV codec (FP4 Step 5).
Quantize a [n_rows, dim] bf16 buffer (per-row absmax INT8) then dequant, report
RMSE/||X|| — INT8 should be near-lossless (well under 1%). Run:
  pixi run mojo run -I . --target-accelerator sm_121a tools/kv_quant_test.mojo
"""
from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.random import random_ui64
from std.math import sqrt
from lib.kv_cache_quant import gpu_quant_kv_i8, gpu_dequant_kv_i8
from lib.kv_cache_quant import gpu_append_quant_kv_i8, gpu_dequant_kv_i8_layer


def _u64bf(p: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]) -> UInt64:
    return UInt64(Int(p))


def _u64i8(p: UnsafePointer[Int8, MutAnyOrigin]) -> UInt64:
    return UInt64(Int(p))


def _u64f(p: UnsafePointer[Float32, MutAnyOrigin]) -> UInt64:
    return UInt64(Int(p))


def main() raises:
    with DeviceContext() as ctx:
        var n_rows = 256          # head x position rows
        var dim = 512             # FULL_HD
        var N = n_rows * dim

        var in_h = ctx.enqueue_create_host_buffer[DType.bfloat16](N)
        for i in range(N):
            var v = Float32(Int(random_ui64(0, 2000))) / 333.0 - 3.0   # ~U[-3,3)
            in_h[i] = v.cast[DType.bfloat16]()

        var in_d = ctx.enqueue_create_buffer[DType.bfloat16](N)
        var i8_d = ctx.enqueue_create_buffer[DType.int8](N)
        var sc_d = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var out_d = ctx.enqueue_create_buffer[DType.bfloat16](N)
        var out_h = ctx.enqueue_create_host_buffer[DType.bfloat16](N)
        ctx.enqueue_copy(in_d, in_h)

        gpu_quant_kv_i8(ctx, _u64bf(in_d.unsafe_ptr()), _u64i8(i8_d.unsafe_ptr()),
                        _u64f(sc_d.unsafe_ptr()), n_rows, dim)
        gpu_dequant_kv_i8(ctx, _u64i8(i8_d.unsafe_ptr()), _u64f(sc_d.unsafe_ptr()),
                          _u64bf(out_d.unsafe_ptr()), n_rows, dim)
        ctx.synchronize()
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()

        var sq = Float64(0.0)
        var sn = Float64(0.0)
        for i in range(N):
            var a = Float32(in_h[i])
            var d = Float64(Float32(out_h[i]) - a)
            sq += d * d
            sn += Float64(a) * Float64(a)
        var rel = sqrt(sq / sn)
        print("KV INT8 round-trip: RMSE/||X|| =", rel)
        if rel < 0.01:
            print("KV INT8 CODEC OK — near-lossless (<1%)")
        else:
            print("FAIL: INT8 KV reconstruction error too large")

        # ── Live-cache layout round-trip: append-quant -> dequant-layer ──
        # src [n_s, nkv, l_hd] (stride l_kvd) -> int8 cache [nkv, max_seq, cache_hd]
        # -> dequant positions [0,n_s) into the max_seq-strided bf16 scratch.
        var n_s = 4
        var nkv = 2
        var l_hd = 256
        var cache_hd = 512
        var max_seq = 16
        var l_kvd = nkv * l_hd
        var sN = n_s * nkv * l_hd
        var cN = nkv * max_seq * cache_hd
        var sk_h = ctx.enqueue_create_host_buffer[DType.float32](sN)
        var sv_h = ctx.enqueue_create_host_buffer[DType.float32](sN)
        for i in range(sN):
            sk_h[i] = Float32(Int(random_ui64(0, 2000))) / 333.0 - 3.0
            sv_h[i] = Float32(Int(random_ui64(0, 2000))) / 333.0 - 3.0
        var sk_d = ctx.enqueue_create_buffer[DType.float32](sN)
        var sv_d = ctx.enqueue_create_buffer[DType.float32](sN)
        var ck_d = ctx.enqueue_create_buffer[DType.int8](cN)
        var cv_d = ctx.enqueue_create_buffer[DType.int8](cN)
        var ks_d = ctx.enqueue_create_buffer[DType.float32](nkv * max_seq)
        var vs_d = ctx.enqueue_create_buffer[DType.float32](nkv * max_seq)
        var dk_d = ctx.enqueue_create_buffer[DType.bfloat16](cN)
        var dk_h = ctx.enqueue_create_host_buffer[DType.bfloat16](cN)
        ctx.enqueue_copy(sk_d, sk_h)
        ctx.enqueue_copy(sv_d, sv_h)
        gpu_append_quant_kv_i8(ctx, _u64f(sk_d.unsafe_ptr()), _u64f(sv_d.unsafe_ptr()),
            _u64i8(ck_d.unsafe_ptr()), _u64i8(cv_d.unsafe_ptr()),
            _u64f(ks_d.unsafe_ptr()), _u64f(vs_d.unsafe_ptr()),
            n_s, nkv, l_hd, l_kvd, max_seq, cache_hd, 0)
        gpu_dequant_kv_i8_layer(ctx, _u64i8(ck_d.unsafe_ptr()), _u64f(ks_d.unsafe_ptr()),
            _u64bf(dk_d.unsafe_ptr()), nkv, n_s, cache_hd, max_seq)
        ctx.synchronize()
        ctx.enqueue_copy(dk_h, dk_d)
        ctx.synchronize()
        var sq2 = Float64(0.0)
        var sn2 = Float64(0.0)
        for s in range(n_s):
            for h in range(nkv):
                for d in range(l_hd):
                    var orig = sk_h[s * l_kvd + h * l_hd + d]
                    var deq = Float32(dk_h[(h * max_seq + s) * cache_hd + d])
                    var df = Float64(deq - orig)
                    sq2 += df * df
                    sn2 += Float64(orig) * Float64(orig)
        var rel2 = sqrt(sq2 / sn2)
        print("KV append->dequant (cache layout): RMSE/||K|| =", rel2)
        if rel2 < 0.01:
            print("KV LIVE-CACHE APPEND/DEQUANT OK — layout + INT8 correct")
        else:
            print("FAIL: append/dequant layout or precision wrong")
