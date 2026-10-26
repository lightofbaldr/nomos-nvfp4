"""Standalone FFI probe for the production Qwen interleaved M-RoPE kernel.

Build as a small shared library for tensor-level parity without adding a
debug symbol to libnomos_kernel's common ABI.
"""

from std.memory import UnsafePointer
from max.gpu.host import DeviceContext
from lib.cuda import cuda_malloc, cuda_free, cuda_memcpy
from lib.ops_gpu_mojo import gpu_mrope_interleaved_batched_mojo


@export
def nomos_probe_qwen_mrope(
    q_host: Int64,
    position_ids_host: Int64,
    S_arg: Int32,
    heads_arg: Int32,
    hd_arg: Int32,
    rope_dim_arg: Int32,
    theta: Float32,
    out_host: Int64,
) -> Int32:
    if q_host == 0 or position_ids_host == 0 or out_host == 0:
        return Int32(-1)
    var S = Int(S_arg)
    var heads = Int(heads_arg)
    var hd = Int(hd_arg)
    var rope_dim = Int(rope_dim_arg)
    if S <= 0 or heads <= 0 or hd <= 0 or rope_dim <= 0:
        return Int32(-1)
    try:
        var ctx = DeviceContext()
        var elems = S * heads * hd
        var d_q = cuda_malloc(elems * 4)
        var d_pos = cuda_malloc(3 * S * 4)
        cuda_memcpy(d_q, UInt64(q_host), elems * 4, 1)
        cuda_memcpy(d_pos, UInt64(position_ids_host), 3 * S * 4, 1)
        gpu_mrope_interleaved_batched_mojo(
            ctx, d_q, d_pos, S, 0, S, heads, hd, theta, rope_dim,
        )
        ctx.synchronize()
        cuda_memcpy(UInt64(out_host), d_q, elems * 4, 2)
        cuda_free(d_pos)
        cuda_free(d_q)
        return Int32(0)
    except e:
        print("[nomos_probe_qwen_mrope EXC]", e)
        return Int32(-99)
