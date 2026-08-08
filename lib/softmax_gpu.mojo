"""GPU softmax for the attention path.

Replaces the CPU softmax sandwich in attention_gpu.mojo. Single FFI call into
libsoftmax.so (custom CUDA kernel) — eliminates the 3 host↔device round-trips
per layer (download scores, CPU softmax, upload back).

One thread block per head, warp+shared-memory reduction. Operates in-place on
the [nh, seq_len] scores tensor on device.
"""
from std.ffi import external_call, c_int


def softmax_gpu(d_scores: UInt64, nh: Int, seq_len: Int) -> Int:
    """In-place softmax over the seq_len axis for each head.

    Args:
        d_scores: device pointer to [nh, seq_len] float32 tensor (tight, row-major)
        nh: number of heads (= grid_x dim)
        seq_len: sequence length (softmax axis)

    Returns:
        0 on success, CUDA error code otherwise.
    """
    var rc = external_call["softmax_over_heads", c_int, UInt64, c_int, c_int](
        d_scores, c_int(nh), c_int(seq_len)
    )
    return Int(rc)
