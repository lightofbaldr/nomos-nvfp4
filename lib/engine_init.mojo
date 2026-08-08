"""engine_init — model-load + construction helpers extracted from GemmaEngine
(file-size rule: keep modules in the 500-800 band). Holds the env-byte reader,
the zero-init List helpers used by __init__.
"""
from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from std.collections import List


def zeros_f32(n: Int) -> List[Float32]:
    var v = List[Float32](capacity=n)
    for _ in range(n):
        v.append(Float32(0))
    return v^


def zeros_f64(n: Int) -> List[Float64]:
    var v = List[Float64](capacity=n)
    for _ in range(n):
        v.append(Float64(0))
    return v^


def zeros_i32(n: Int) -> List[Int32]:
    var v = List[Int32](capacity=n)
    for _ in range(n):
        v.append(Int32(0))
    return v^


def _read_env_bytes(name: String) -> List[UInt8]:
    """Read an environment variable through libc getenv. Returns an empty
    List if unset, otherwise the value bytes (NOT null-terminated).

    Returning raw bytes instead of String sidesteps Mojo 1.0's lack of a
    public `String(bytes=...)` constructor — every byte-level operation
    we need (split on `:`, slice into per-path NUL-terminated buffers)
    can stay in `List[UInt8]` land."""
    var nb = List[UInt8](capacity=name.byte_length() + 1)
    for i in range(name.byte_length()): nb.append(name.as_bytes()[i])
    nb.append(0)
    var out = List[UInt8]()
    var raw = external_call["getenv", Int64, UnsafePointer[UInt8, origin_of(nb)]](nb.unsafe_ptr())
    if raw == 0:
        return out^
    var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(raw))
    var n = 0
    while p[n] != 0:
        n += 1
    for i in range(n):
        out.append(p[i])
    return out^


# Flag IDs for _env_flag_cached — MUST match NOMOS_FLAG_NAMES[] order in
# nomos_libc_shim.c (the C side holds the actual env-var name strings).
alias ENV_NVFP4_V3 = 0
alias ENV_NVFP4_V3_TREE = 1
alias ENV_NVFP4_FUSED = 2
alias ENV_Q4_FUSED = 3
alias ENV_NVFP4_GEMV_V2 = 4
alias ENV_Q4_DP4A = 5      # Q4 decode acts: =1 -> q8 dp4a (prod), else fp32 occ4
alias ENV_STRICT_Q4 = 6    # precision-law guard: =1 -> record any non-Q4/Q8 path reached

# Precision-law violation codes (mirror nomos_flag_violation bits in nomos_libc_shim.c).
# Each forbidden precision DISPATCH flags one of these under NOMOS_STRICT_Q4=1, so a single
# prefill+decode+verify run enumerates every reachable violation at once. THE LAW: weights Q4,
# acts Q8, KV INT4; fp32 only as a transient accumulator register; bf16/fp16 never; no fp32 matmul.
alias VIOL_BF16_PROJ = 0           # bf16 weight matmul (decode _mm_dev fallback)
alias VIOL_BF16_PROJ_BATCHED = 1   # Q4->bf16 dequant + bf16 cuBLAS (prefill projections)
alias VIOL_BF16_ATTN_DECODE = 2    # bf16 attention (per-token decode)
alias VIOL_BF16_ATTN_BATCHED = 3   # bf16 attention (batched/verify prefill)
alias VIOL_FP32_ATTN_DECODE = 4    # fp32-matmul attention (per-token decode)
alias VIOL_FP32_ATTN_BATCHED = 5   # fp32-matmul attention (batched/verify prefill)
alias VIOL_FP32_PROJ = 6           # fp32 weight matmul (occ4 / any fp32 GEMM)


def _strict_q4() -> Bool:
    """True when the precision-law guard is armed (NOMOS_STRICT_Q4=1). Callers gate
    _flag_violation on this so production pays nothing."""
    return _env_flag_cached(ENV_STRICT_Q4) == 1


def _flag_violation(code: Int):
    """Record a precision-law violation under the guard (see VIOL_* + nomos_libc_shim.c)."""
    _ = external_call["nomos_flag_violation", Int32, Int32](Int32(code))


def _env_flag_cached(id: Int) -> Int:
    """Process-cached env tri-state via the C shim (read ONCE per process, not
    per-GEMV). The decode dispatch (_mm_dev) consults immutable NOMOS_* toggles
    on every weight GEMV; routing them through here collapses ~1000 getenv +
    2-alloc calls/token down to one cheap cached int read. See nomos_libc_shim.c.
    Returns: -1 unset, 0 == "0", 1 == "1", 2 == other set value."""
    return Int(external_call["nomos_env_flag_cached", Int32, Int32](Int32(id)))
