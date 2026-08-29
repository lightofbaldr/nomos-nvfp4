"""Deterministic check for Qwen's per-head [query|gate] projection split."""

from max.gpu.host import DeviceContext
from lib.cuda import cuda_malloc, cuda_free, cuda_upload, cuda_download
from lib.ops_gpu_mojo import gpu_split_q_gate_interleaved_mojo


def main() raises:
    var ctx = DeviceContext()
    # Two rows, two heads, head_dim=2. Each row is
    # [h0.q0,q1,h0.g0,g1,h1.q0,q1,h1.g0,g1].
    var raw = List[Float32](capacity=16)
    for v in [1, 2, 11, 12, 3, 4, 13, 14, 5, 6, 15, 16, 7, 8, 17, 18]:
        raw.append(Float32(v))
    var q = List[Float32](capacity=8)
    var gate = List[Float32](capacity=8)
    for _ in range(8): q.append(0.0); gate.append(0.0)
    var d_raw = cuda_malloc(16 * 4)
    var d_q = cuda_malloc(8 * 4)
    var d_gate = cuda_malloc(8 * 4)
    cuda_upload(d_raw, raw)
    gpu_split_q_gate_interleaved_mojo(ctx, d_q, d_gate, d_raw, 2, 4, 2)
    ctx.synchronize()
    cuda_download(q, d_q, 8); cuda_download(gate, d_gate, 8)
    var q_expected = List[Float32](capacity=8)
    var g_expected = List[Float32](capacity=8)
    for v in [1, 2, 3, 4, 5, 6, 7, 8]: q_expected.append(Float32(v))
    for v in [11, 12, 13, 14, 15, 16, 17, 18]: g_expected.append(Float32(v))
    for i in range(8):
        if q[i] != q_expected[i] or gate[i] != g_expected[i]:
            raise Error("q/gate interleaved split mismatch")
    print("Qwen q/gate split check PASS")
    cuda_free(d_raw); cuda_free(d_q); cuda_free(d_gate)
