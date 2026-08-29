"""D2 G3 debug — load the 9 eagle NVFP4 blobs exactly like Eagle3Drafter
(load_to_gpu_nvfp4) and read every byte back, comparing against the file.
Catches upload elision/reordering (FFI gotcha: opaque-pointer external_calls)
at the load seam. Run with the eagle3-flat dir as cwd-relative default."""
from max.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.ffi import external_call, c_int
from lib.fp4_weights import load_to_gpu_nvfp4
from lib.io import read_nvfp4_bytes
from lib.cuda import cuda_sync, cuda_free

comptime DIR = "~/nomos-refs/drafter-study/eagle3-flat/"


def check(name: String) raises:
    var hdr = List[Int]()
    var gs = List[Float32]()
    var file_bytes = read_nvfp4_bytes(DIR + name, hdr, gs)
    var gs2 = List[Float32]()
    var ags2 = List[Float32]()
    var d_w = load_to_gpu_nvfp4(DIR + name, gs2, ags2)
    cuda_sync()
    var nb = len(file_bytes)
    var back = List[UInt8](capacity=nb)
    for _ in range(nb):
        back.append(0)
    var rc = external_call["cudaMemcpy", c_int, UInt64, UInt64, UInt64, c_int](
        UInt64(Int(back.unsafe_ptr())), d_w, UInt64(nb), c_int(2))
    cuda_sync()
    var mis = 0
    var first = -1
    for i in range(nb):
        if back[i] != file_bytes[i]:
            mis += 1
            if first < 0:
                first = i
    print(name, " bytes=", nb, " gs=", gs2[0], " mismatches=", mis,
          " first=", first, " rc=", Int(rc))
    cuda_free(d_w)


def main() raises:
    with DeviceContext() as ctx:
        # touch the ctx so the device is initialized the same way as the engine
        _ = ctx.default_device_info.compute
        check("eagle_fc_weight.nvfp4")
        check("eagle_layer_self_attn_q_proj_weight.nvfp4")
        check("eagle_layer_self_attn_k_proj_weight.nvfp4")
        check("eagle_layer_self_attn_v_proj_weight.nvfp4")
        check("eagle_layer_self_attn_o_proj_weight.nvfp4")
        check("eagle_layer_mlp_gate_proj_weight.nvfp4")
        check("eagle_layer_mlp_up_proj_weight.nvfp4")
        check("eagle_layer_mlp_down_proj_weight.nvfp4")
        check("eagle_lm_head_weight.nvfp4")
