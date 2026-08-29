"""mtp_discrete_probe.mojo — SCOPING PROBE (Nano).

Question: does the MTP assistant's Q8 weight GEMV (`gpu_matmul_q8_fused_dev`) MMU-fault
on the discrete box's DISCRETE VRAM (RTX PRO 4000 / sm_120), the way the 31B's Q4_0-dp4a weight GEMV
does? Base decode on discrete (w4a4) never exercises the Q8 path, so its discrete-safety is
unproven — and it decides whether the MTP re-wire is "just the embed" (small) or "requant the
whole 49-tensor assistant to NVFP4" (big). This probe answers it with a fault check + a
correctness check, standalone on GPU0.

Run (repo root): pixi run mojo run tools/mtp_discrete_probe.mojo
"""
from max.gpu.host import DeviceContext
from lib.cuda import cuda_malloc, cuda_upload_u8, cuda_upload, cuda_download, cuda_sync, cuda_last_error
from lib.q8_weights import gpu_matmul_q8_fused_dev

comptime GROUP = 32


# Build a header-stripped .q8 blob in host bytes: [nb*fp16 scales][N*K int8 values].
# scales all = 1.0 (fp16 0x3C00); int8 value at flat index i = (i % 5) (0..4, nonneg).
def _build_q8_blob(K: Int, N: Int) -> List[UInt8]:
    var nb = (K * N + GROUP - 1) // GROUP
    var total = nb * 2 + K * N
    var blob = List[UInt8](capacity=total)
    for _ in range(total):
        blob.append(0)
    # fp16(1.0) = 0x3C00 -> little-endian [0x00, 0x3C]
    for b in range(nb):
        blob[2 * b] = 0x00
        blob[2 * b + 1] = 0x3C
    for i in range(K * N):
        blob[nb * 2 + i] = UInt8(i % 5)
    return blob^


def _run_case(ctx: DeviceContext, K: Int, N: Int, full_check: Bool) raises -> Bool:
    print("  case K=", K, " N=", N, " (blob ~", (((K * N + GROUP - 1)//GROUP)*2 + K*N)//(1024*1024), "MB)")
    var blob = _build_q8_blob(K, N)
    var d_w = cuda_malloc(len(blob))
    cuda_upload_u8(d_w, blob)
    # input: all ones (K elems)
    var inp = List[Float32](capacity=K)
    for _ in range(K):
        inp.append(1.0)
    var d_in = cuda_malloc(K * 4)
    cuda_upload(d_in, inp)
    var d_out = cuda_malloc(N * 4)
    _ = cuda_last_error()          # clear any prior state
    gpu_matmul_q8_fused_dev(ctx, d_out, d_in, d_w, K, N)
    cuda_sync()
    var err = cuda_last_error()
    if err != 0:
        print("    >>> CUDA ERROR after Q8 GEMV: code", err, "(ILLEGAL_ADDRESS=700) — FAULTS ON DISCRETE")
        return False
    print("    no CUDA fault (err=0)")
    # correctness: out[n] = sum_k ((n*K+k) % 5) * 1.0
    var outv = List[Float32](capacity=N)
    for _ in range(N):
        outv.append(0.0)
    cuda_download(outv, d_out, N)
    var ok = True
    var check_n = N if full_check else 3
    var idxs = List[Int]()
    if full_check:
        for n in range(N): idxs.append(n)
    else:
        idxs.append(0); idxs.append(N // 2); idxs.append(N - 1)
    for ii in range(len(idxs)):
        var n = idxs[ii]
        var refv = Float32(0.0)
        for k in range(K):
            refv += Float32((n * K + k) % 5)
        var got = outv[n]
        var d = got - refv
        if d < 0: d = -d
        if d > 0.05:
            print("    MISMATCH n=", n, " got=", got, " ref=", refv)
            ok = False
    if ok:
        print("    correctness OK (checked", len(idxs), "rows)")
    return ok


def main() raises:
    print("=== MTP Q8-GEMV discrete-VRAM probe (GPU0 / sm_120) ===")
    with DeviceContext() as ctx:
        var a = _run_case(ctx, 128, 256, True)       # small: full correctness
        var b = _run_case(ctx, 1024, 262144, False)  # real tied-logits shape: fault + spot check
        var c = _run_case(ctx, 10752, 1024, False)   # real pre_proj shape
        print("=== VERDICT ===")
        if a and b and c:
            print("Q8 GEMV is DISCRETE-SAFE (no fault, correct) — assistant forward needs NO requant.")
        else:
            print("Q8 GEMV FAULTS/WRONG on discrete — assistant must be re-quantized to NVFP4 (BIG scope).")
