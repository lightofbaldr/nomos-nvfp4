"""Pure-Mojo FP32 → BF16 device cast.

Replaces fp32_to_bf16 from cast_kernel.cu (libcast.so). Used by lib/cublas.mojo
right before BF16 cuBLAS GEMMs to shrink the FP32 activation to BF16 in a
scratch buffer. Hardware round-to-nearest-even via Mojo's native bf16 cast.

Drop-in for the libcast.so external_call, wrapper takes UInt64 device pointers
so call sites don't need to change.
"""

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer, bitcast


def fp32_to_bf16_kernel(
    inp: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[UInt16, MutAnyOrigin],
    n_arg: Int32,
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var n = Int(n_arg)
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx < n:
        var f = SIMD[DType.float32, 1](inp[idx])
        var bf = f.cast[DType.bfloat16]()
        dst[idx] = bitcast[DType.uint16, 1](bf)[0]


def gpu_fp32_to_bf16_mojo(
    ctx: DeviceContext, d_in_fp32: UInt64, d_out_bf16: UInt64, n: Int
) raises:
    """In-place cast FP32 → BF16. d_in_fp32 stays unchanged; d_out_bf16
    receives the truncated BF16 (packed as UInt16, n*2 bytes).
    """
    var threads = 256
    var blocks = (n + threads - 1) // threads
    var k = ctx.compile_function[fp32_to_bf16_kernel]()
    var inp_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_in_fp32))
    var dst_ptr = UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=Int(d_out_bf16))
    ctx.enqueue_function(
        k, inp_ptr, dst_ptr, Int32(n),
        grid_dim=blocks, block_dim=threads,
    )
