"""Deterministic smoke test for the full-K conv and recurrent GDN kernels."""

from max.gpu.host import DeviceContext
from lib.cuda import cuda_malloc, cuda_free, cuda_upload, cuda_upload_u16, cuda_download, cuda_sync, cuda_memset_2d
from lib.gdn_scan import (
    gpu_gdn_conv_scan, gpu_gdn_recurrent_scan, gpu_gdn_bf16_to_f32,
    gpu_gdn_prepare_decay_beta, gpu_gdn_gated_rmsnorm,
)
from lib.model_profiles.qwen3_5 import GDN_CONV_DIM, GDN_CONV_KERNEL, GDN_VALUE_DIM, GDN_NUM_V_HEADS


def zeros_f32(n: Int) -> List[Float32]:
    var out_list = List[Float32](capacity=n)
    for _ in range(n):
        out_list.append(0.0)
    return out_list^


def main() raises:
    var ctx = DeviceContext()
    # Allocate the Qwen-sized state directly.  This checker must remain valid
    # even when the generated common model_config currently selects Gemma or
    # Muse (whose GDN pool strides are deliberately zero).
    var d_conv_state = cuda_malloc(GDN_CONV_DIM * GDN_CONV_KERNEL * 2)
    var d_rec_state = cuda_malloc(48 * 128 * 128 * 2)
    cuda_memset_2d(d_rec_state, 48 * 128 * 128 * 2, 0, 48 * 128 * 128 * 2, 1)

    # Conv: prior channel-0 state [10,20,30,40], new rows [1,2], all-one
    # channel-0 filter. Expected outputs 91,73 and final state [30,40,1,2].
    var proj = zeros_f32(2 * GDN_CONV_DIM)
    proj[0] = 1.0
    proj[1] = 1.0
    proj[GDN_CONV_DIM] = 2.0
    var weights = List[UInt16](capacity=GDN_CONV_DIM * GDN_CONV_KERNEL)
    for _ in range(GDN_CONV_DIM * GDN_CONV_KERNEL): weights.append(0)
    for k in range(GDN_CONV_KERNEL): weights[k] = UInt16(0x3f80)  # bf16 1.0
    for k in range(GDN_CONV_KERNEL): weights[GDN_CONV_KERNEL + k] = UInt16(0x3f80)
    var state = List[UInt16](capacity=GDN_CONV_DIM * GDN_CONV_KERNEL)
    for _ in range(GDN_CONV_DIM * GDN_CONV_KERNEL): state.append(0)
    state[0] = UInt16(0x4120)  # 10
    state[1] = UInt16(0x41a0)  # 20
    state[2] = UInt16(0x41f0)  # 30
    state[3] = UInt16(0x4220)  # 40
    cuda_upload_u16(d_conv_state, state)
    var d_proj = cuda_malloc(len(proj) * 4)
    var d_weights = cuda_malloc(len(weights) * 2)
    var d_conv_out = cuda_malloc(len(proj) * 4)
    cuda_upload(d_proj, proj)
    cuda_upload_u16(d_weights, weights)
    gpu_gdn_conv_scan(ctx, d_conv_out, d_proj, d_weights, d_conv_state, 2)
    ctx.synchronize()
    var conv_out = zeros_f32(len(proj))
    cuda_download(conv_out, d_conv_out, len(conv_out))
    var d_conv_state_f32 = cuda_malloc(GDN_CONV_DIM * GDN_CONV_KERNEL * 4)
    gpu_gdn_bf16_to_f32(ctx, d_conv_state_f32, d_conv_state, GDN_CONV_DIM * GDN_CONV_KERNEL)
    ctx.synchronize()
    var conv_state = zeros_f32(GDN_CONV_DIM * GDN_CONV_KERNEL)
    cuda_download(conv_state, d_conv_state_f32, len(conv_state))
    if conv_out[0] != 91.0 or conv_out[GDN_CONV_DIM] != 73.0:
        raise Error("conv output mismatch")
    if conv_out[1] < 0.730 or conv_out[1] > 0.732:
        raise Error("conv SiLU mismatch")
    if conv_state[0] != 30.0 or conv_state[1] != 40.0 or conv_state[2] != 1.0 or conv_state[3] != 2.0:
        raise Error("conv full-K commit mismatch")

    # Recurrence: q=k=e0, v(head0,col0)=2, g=0, beta=1, zero state.
    # Expected state[0,0,0] ~=2 and out[0,0] ~=2/sqrt(128).
    var qkv = zeros_f32(GDN_CONV_DIM)
    qkv[0] = 1.0
    qkv[2048] = 1.0
    qkv[4096] = 2.0
    var g = zeros_f32(GDN_NUM_V_HEADS)
    var beta = zeros_f32(GDN_NUM_V_HEADS)
    beta[0] = 1.0
    var d_qkv = cuda_malloc(len(qkv) * 4)
    var d_g = cuda_malloc(len(g) * 4)
    var d_beta = cuda_malloc(len(beta) * 4)
    var d_rec_out = cuda_malloc(GDN_VALUE_DIM * 4)
    cuda_upload(d_qkv, qkv); cuda_upload(d_g, g); cuda_upload(d_beta, beta)
    gpu_gdn_recurrent_scan(ctx, d_rec_out, d_qkv, d_g, d_beta, d_rec_state, 1)
    ctx.synchronize()
    var rec_out = zeros_f32(GDN_VALUE_DIM)
    cuda_download(rec_out, d_rec_out, len(rec_out))
    var d_rec_state_f32 = cuda_malloc(48 * 128 * 128 * 4)
    gpu_gdn_bf16_to_f32(ctx, d_rec_state_f32, d_rec_state, 48 * 128 * 128)
    ctx.synchronize()
    var first_state = zeros_f32(1)
    cuda_download(first_state, d_rec_state_f32, 1)
    if first_state[0] < 1.99 or first_state[0] > 2.01:
        raise Error("recurrent state orientation/update mismatch")
    if rec_out[0] < 0.176 or rec_out[0] > 0.178:
        raise Error("recurrent readout mismatch")

    # Gate transforms: a=b=A_log=dt_bias=0 => g=-log(2), beta=0.5.
    var gate_rows = zeros_f32(GDN_NUM_V_HEADS)
    var gate_params = List[UInt16](capacity=GDN_NUM_V_HEADS)
    for _ in range(GDN_NUM_V_HEADS): gate_params.append(0)
    var d_a = cuda_malloc(GDN_NUM_V_HEADS * 4)
    var d_b = cuda_malloc(GDN_NUM_V_HEADS * 4)
    var d_gate_param = cuda_malloc(GDN_NUM_V_HEADS * 2)
    var d_g_out = cuda_malloc(GDN_NUM_V_HEADS * 4)
    var d_beta_out = cuda_malloc(GDN_NUM_V_HEADS * 4)
    cuda_upload(d_a, gate_rows); cuda_upload(d_b, gate_rows)
    cuda_upload_u16(d_gate_param, gate_params)
    gpu_gdn_prepare_decay_beta(
        ctx, d_g_out, d_beta_out, d_a, d_b, d_gate_param,
        d_gate_param, 1,
    )
    ctx.synchronize()
    var g_out = zeros_f32(GDN_NUM_V_HEADS)
    var beta_out = zeros_f32(GDN_NUM_V_HEADS)
    cuda_download(g_out, d_g_out, GDN_NUM_V_HEADS)
    cuda_download(beta_out, d_beta_out, GDN_NUM_V_HEADS)
    if g_out[0] > -0.692 or g_out[0] < -0.694 or beta_out[0] != 0.5:
        raise Error("g/beta transform mismatch")

    # Gated RMSNorm: core=1, z=1, raw norm weight=2 -> ~2*silu(1).
    var core = zeros_f32(GDN_VALUE_DIM)
    var z = zeros_f32(GDN_VALUE_DIM)
    for i in range(GDN_VALUE_DIM):
        core[i] = 1.0; z[i] = 1.0
    var norm_w = List[UInt16](capacity=128)
    for _ in range(128): norm_w.append(UInt16(0x4000))  # bf16 2.0
    var d_core = cuda_malloc(GDN_VALUE_DIM * 4)
    var d_z = cuda_malloc(GDN_VALUE_DIM * 4)
    var d_norm_w = cuda_malloc(128 * 2)
    var d_norm_out = cuda_malloc(GDN_VALUE_DIM * 4)
    cuda_upload(d_core, core); cuda_upload(d_z, z)
    cuda_upload_u16(d_norm_w, norm_w)
    gpu_gdn_gated_rmsnorm(
        ctx, d_norm_out, d_core, d_z, d_norm_w, 1, 0.000001
    )
    ctx.synchronize()
    var norm_out = zeros_f32(GDN_VALUE_DIM)
    cuda_download(norm_out, d_norm_out, GDN_VALUE_DIM)
    if norm_out[0] < 1.461 or norm_out[0] > 1.463:
        raise Error("gated RMSNorm mismatch")

    print("GDN scan check PASS", conv_out[0], conv_out[GDN_CONV_DIM], first_state[0], rec_out[0], g_out[0], beta_out[0], norm_out[0])
    cuda_free(d_proj); cuda_free(d_weights); cuda_free(d_conv_out); cuda_free(d_conv_state_f32)
    cuda_free(d_qkv); cuda_free(d_g); cuda_free(d_beta); cuda_free(d_rec_out); cuda_free(d_rec_state_f32)
    cuda_free(d_a); cuda_free(d_b); cuda_free(d_gate_param); cuda_free(d_g_out); cuda_free(d_beta_out)
    cuda_free(d_core); cuda_free(d_z); cuda_free(d_norm_w); cuda_free(d_norm_out)
    cuda_free(d_conv_state); cuda_free(d_rec_state)
