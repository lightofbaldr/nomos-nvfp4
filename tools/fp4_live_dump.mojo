"""Live NVFP4 dequant dump — run dequant_nvfp4_kernel on a REAL .nvfp4 weight (real n_w/nb
args, real blob) vs the host formula, at start/mid/end. Splits 'dequant correct on synthetic'
from 'garbage in the live engine path': all windows match => live dequant correct (scratch RIGHT,
bug is GEMM/activation); any mismatch => the live invocation differs (real args/scale)."""
from max.gpu.host import DeviceContext
from lib.io import read_nvfp4_bytes
from lib.fp4_weights import dequant_nvfp4_kernel, e4m3_decode, E2M1_MAG, FP4_GROUP


def main() raises:
    with DeviceContext() as ctx:
        var path = String("~/nomos_data/gemma-4-31b/embed_tokens_weight.nvfp4")
        var hdr = List[Int]()
        var gscale = List[Float32]()
        var body = read_nvfp4_bytes(path, hdr, gscale)
        if len(body) < 24 or len(hdr) < 2:
            print("READ FAILED len=", len(body))
            return
        var n = hdr[0]
        var nb = hdr[1]
        var gs = gscale[0]
        print("n=", n, " nb=", nb, " gscale=", gs, " body_bytes=", len(body))

        var blob_h = ctx.enqueue_create_host_buffer[DType.uint8](len(body))
        for i in range(len(body)):
            blob_h[i] = body[i]
        var blob_d = ctx.enqueue_create_buffer[DType.uint8](len(body))
        var out_d = ctx.enqueue_create_buffer[DType.bfloat16](n)
        var out_h = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
        ctx.enqueue_copy(blob_d, blob_h)
        var k = ctx.compile_function[dequant_nvfp4_kernel]()
        var threads = 256
        var blocks = (n + threads - 1) // threads
        ctx.enqueue_function(k, blob_d.unsafe_ptr(), out_d.unsafe_ptr(), gs, n, nb,
                             grid_dim=blocks, block_dim=threads)
        ctx.synchronize()
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()

        var starts = [0, (n // 2) // 16 * 16, n - 8]
        var bad = 0
        for s in range(len(starts)):
            var i0 = starts[s]
            print("WINDOW start i=", i0)
            for i in range(i0, i0 + 8):
                var block = i // FP4_GROUP
                var bs_byte = Int(blob_h[block])
                var byte = Int(blob_h[nb + (i // 2)])
                var nib = (byte & 0x0F) if (i & 1) == 0 else ((byte >> 4) & 0x0F)
                var sign = Float32(-1.0) if (nib & 8) != 0 else Float32(1.0)
                var mag = E2M1_MAG[nib & 7]
                var want = (Float32(gs) * e4m3_decode(bs_byte) * sign * mag).cast[DType.bfloat16]()
                var got = out_h[i]
                var tag = String("ok") if got == want else String("**MISMATCH**")
                if got != want:
                    bad += 1
                print("  i=", i, " scale_byte=", bs_byte, " gpu=", Float32(got), " host=", Float32(want), " ", tag)
        print("TOTAL bad windows-sample=", bad, "/24")
        if bad == 0:
            print("LIVE DEQUANT CORRECT on real weight -> scratch RIGHT, bug is GEMM/activation")
        else:
            print("LIVE DEQUANT WRONG -> the live invocation differs (real args/scale)")
        _ = blob_d^
        _ = out_d^
