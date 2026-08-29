"""GDN (Gated-DeltaNet) per-layer weights for the qwen3_5 second engine.

OWNERSHIP: Kvasir owns this loader + tools/convert_qwen3_5_gdn.py (the converter that produces the blobs).
Kvasir owns adding the GdnWeights field to the engine, the prefill/decode routing that GEMMs these
weights, and the g/beta + gated-RMSNorm transforms (computed FP32 in-engine from A_log/dt_bias/norm).

The shipped/default contract keeps all 9 tensors as BF16 device blobs:
  in_proj_qkv [10240,5120], in_proj_z [6144,5120], out_proj [5120,6144]  — dequant FP8 E4M3 x weight_scale
      [out,1] -> BF16 at CONVERT time (correctness-first; FP8-native GEMM + #518 FP8 roofline deferred).
  in_proj_a [48,5120], in_proj_b [48,5120], conv1d [10240,4] (checkpoint [10240,1,4] squeezed),
      A_log [48], dt_bias [48], norm [128]  — native BF16, preserved raw (engine computes g/beta/gated-norm).
Phase B optionally loads the three large projections as NVFP4.  The six recurrence-critical small
tensors stay BF16 in both modes.  Projection outputs remain FP32, so the scan contract is unchanged.

Blob layout the converter writes and this loads: <gdn_dir>/<gdn_slot>.<name>.{bf16,nvfp4},
name in {in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,conv1d,A_log,dt_bias,norm}. gdn_slot is the
ordered GDN index (same order as GdnStatePools / the engine's full-layer complement), NOT a transformer id.

Non-GDN profiles never construct this (engine gates on HAS_LINEAR_ATTENTION); the struct still compiles
(it references no GDN_* aliases — sizes come from the blob files).
"""
from lib.io import load_bf16_file_to_gpu
from lib.fp4_weights import load_to_gpu_nvfp4
from lib.cuda import cuda_free


struct GdnWeights(Movable):
    """Per-GDN-slot weights. Only qkv/z/out may use the Phase-B NVFP4 route."""

    var in_proj_qkv: List[UInt64]   # [n_gdn] -> [10240,5120] bf16
    var in_proj_z: List[UInt64]     # -> [6144,5120]
    var in_proj_a: List[UInt64]     # -> [48,5120]
    var in_proj_b: List[UInt64]     # -> [48,5120]
    var out_proj: List[UInt64]      # -> [5120,6144]
    var conv1d: List[UInt64]        # -> [10240,4]  (singleton squeezed)
    var a_log: List[UInt64]         # -> [48]
    var dt_bias: List[UInt64]       # -> [48]
    var norm: List[UInt64]          # -> [128]
    var qkv_gs: List[Float32]
    var z_gs: List[Float32]
    var out_gs: List[Float32]
    var qkv_ags: List[Float32]
    var z_ags: List[Float32]
    var out_ags: List[Float32]
    var proj_nvfp4: Bool
    var n_gdn: Int

    def __init__(out self, gdn_dir: String, n_gdn: Int, proj_nvfp4: Bool) raises:
        self.n_gdn = n_gdn
        self.proj_nvfp4 = proj_nvfp4
        self.in_proj_qkv = List[UInt64]()
        self.in_proj_z = List[UInt64]()
        self.in_proj_a = List[UInt64]()
        self.in_proj_b = List[UInt64]()
        self.out_proj = List[UInt64]()
        self.conv1d = List[UInt64]()
        self.a_log = List[UInt64]()
        self.dt_bias = List[UInt64]()
        self.norm = List[UInt64]()
        self.qkv_gs = List[Float32]()
        self.z_gs = List[Float32]()
        self.out_gs = List[Float32]()
        self.qkv_ags = List[Float32]()
        self.z_ags = List[Float32]()
        self.out_ags = List[Float32]()
        var base = gdn_dir if gdn_dir.endswith("/") else gdn_dir + "/"
        for s in range(n_gdn):
            var p = base + String(s) + "."
            if proj_nvfp4:
                self.in_proj_qkv.append(load_to_gpu_nvfp4(
                    p + "in_proj_qkv.nvfp4", self.qkv_gs, self.qkv_ags,
                ))
                self.in_proj_z.append(load_to_gpu_nvfp4(
                    p + "in_proj_z.nvfp4", self.z_gs, self.z_ags,
                ))
                self.out_proj.append(load_to_gpu_nvfp4(
                    p + "out_proj.nvfp4", self.out_gs, self.out_ags,
                ))
                if (self.in_proj_qkv[s] == 0 or self.in_proj_z[s] == 0
                        or self.out_proj[s] == 0 or self.qkv_gs[s] == 0.0
                        or self.z_gs[s] == 0.0 or self.out_gs[s] == 0.0):
                    raise Error(
                        "NOMOS_GDN_NVFP4=1 requires live qkv/z/out NVFP4 blobs "
                        "and scales for GDN slot " + String(s)
                    )
            else:
                self.in_proj_qkv.append(load_bf16_file_to_gpu(p + "in_proj_qkv.bf16"))
                self.in_proj_z.append(load_bf16_file_to_gpu(p + "in_proj_z.bf16"))
                self.out_proj.append(load_bf16_file_to_gpu(p + "out_proj.bf16"))
                self.qkv_gs.append(0.0); self.qkv_ags.append(0.0)
                self.z_gs.append(0.0); self.z_ags.append(0.0)
                self.out_gs.append(0.0); self.out_ags.append(0.0)
            self.in_proj_a.append(load_bf16_file_to_gpu(p + "in_proj_a.bf16"))
            self.in_proj_b.append(load_bf16_file_to_gpu(p + "in_proj_b.bf16"))
            self.conv1d.append(load_bf16_file_to_gpu(p + "conv1d.bf16"))
            self.a_log.append(load_bf16_file_to_gpu(p + "A_log.bf16"))
            self.dt_bias.append(load_bf16_file_to_gpu(p + "dt_bias.bf16"))
            self.norm.append(load_bf16_file_to_gpu(p + "norm.bf16"))

    def free(mut self):
        """Release every GDN weight blob (idempotent)."""
        for s in range(self.n_gdn):
            cuda_free(self.in_proj_qkv[s])
            cuda_free(self.in_proj_z[s])
            cuda_free(self.in_proj_a[s])
            cuda_free(self.in_proj_b[s])
            cuda_free(self.out_proj[s])
            cuda_free(self.conv1d[s])
            cuda_free(self.a_log[s])
            cuda_free(self.dt_bias[s])
            cuda_free(self.norm[s])
        self.n_gdn = 0
