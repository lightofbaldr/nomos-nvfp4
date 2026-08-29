"""GDN (Gated-DeltaNet) recurrent + conv state pools — the qwen3_5 second engine's linear-attention state.

OWNERSHIP: Kvasir owns this module (the pools + accessors + ordered-slot mapping). Kvasir owns
lib/gdn_scan.mojo (the scan kernels that read/write these pools) and lib/gemma4_engine.mojo (single
writer: the GdnStatePools field + init/free + layer routing + the nomos_debug_gdn_state bf16->fp32 cast).

NO MODULO HERE (Codex's guard): the ordered GDN transformer-layer list is PASSED IN by the engine, whose
`layer_is_full` is the single source of truth. GDN slots are the complement of the engine's own full-layer
set, so GDN and softmax layers can never disagree; this module never re-derives layer types.

Pools (batch-1, layer-major, native BF16 — the debug export casts to fp32, and that cast lives in
nomos_debug_gdn_state, Kvasir's file):
  rec_base[gdn_slot, v_head, key_dim, value_dim] -> [n_gdn, GDN_NUM_V_HEADS, GDN_KEY_HEAD_DIM, GDN_VALUE_HEAD_DIM]
                                                    = [48,48,128,128] = 72 MiB at n_gdn=48
  conv_base[gdn_slot, channel, age]              -> [n_gdn, GDN_CONV_DIM, GDN_CONV_KERNEL]
                                                    = [48,10240,4]  FULL K=4 (oldest->newest), 3.75 MiB
`gdn_slot` indexes the ordered GDN list, NEVER the transformer layer id, NEVER layer%4.

Non-GDN profiles: HAS_LINEAR_ATTENTION=0 -> GDN_* aliases are 0 and the engine never constructs this;
the struct still compiles (the byte-stride aliases are 0, no allocation ever runs).
"""
from lib.cuda import cuda_malloc, cuda_free, cuda_memset_2d, cuda_memcpy
from lib.model_config import (
    GDN_NUM_V_HEADS, GDN_KEY_HEAD_DIM, GDN_VALUE_HEAD_DIM, GDN_CONV_DIM, GDN_CONV_KERNEL,
)

alias BF16_BYTES = 2
# Per-slot byte strides (compile-time). Zero on non-GDN profiles (never allocated there).
alias REC_BYTES_PER_SLOT = GDN_NUM_V_HEADS * GDN_KEY_HEAD_DIM * GDN_VALUE_HEAD_DIM * BF16_BYTES
alias CONV_BYTES_PER_SLOT = GDN_CONV_DIM * GDN_CONV_KERNEL * BF16_BYTES


struct GdnStatePools(Movable):
    """Two contiguous device BF16 pools + the engine-provided ordered GDN layer mapping."""

    var rec_base: UInt64        # device ptr [n_gdn, 48,128,128] bf16 — recurrent SSM state
    var conv_base: UInt64       # device ptr [n_gdn, 10240,4] bf16 — causal-conv state (full K)
    var rec_snapshot: UInt64    # lazy DFlash rollback snapshot (same layout as rec_base)
    var conv_snapshot: UInt64   # lazy DFlash rollback snapshot (same layout as conv_base)
    var snapshot_valid: Bool
    var n_gdn: Int
    var gdn_layers: List[Int]   # ordered GDN transformer-layer ids (engine's full-layer complement)

    def __init__(out self, gdn_layers: List[Int]):
        self.gdn_layers = gdn_layers.copy()   # Mojo 1.0: List is not ImplicitlyCopyable; copy (caller keeps theirs)
        self.n_gdn = len(gdn_layers)
        if self.n_gdn == 0:
            self.rec_base = 0
            self.conv_base = 0
            self.rec_snapshot = 0
            self.conv_snapshot = 0
            self.snapshot_valid = False
            return
        self.rec_base = cuda_malloc(self.n_gdn * REC_BYTES_PER_SLOT)
        self.conv_base = cuda_malloc(self.n_gdn * CONV_BYTES_PER_SLOT)
        self.rec_snapshot = 0
        self.conv_snapshot = 0
        self.snapshot_valid = False
        self.reset()

    def reset(mut self):
        """Zero both pools — call before each fresh prefill (the GDN analogue of reset_kv_cache)."""
        if self.n_gdn == 0:
            return
        var rb = self.n_gdn * REC_BYTES_PER_SLOT
        var cb = self.n_gdn * CONV_BYTES_PER_SLOT
        cuda_memset_2d(self.rec_base, rb, 0, rb, 1)
        cuda_memset_2d(self.conv_base, cb, 0, cb, 1)
        self.snapshot_valid = False

    def snapshot_for_verify(mut self) raises:
        """Snapshot both contiguous pools before speculative target verify mutates them."""
        if self.n_gdn == 0:
            return
        if self.snapshot_valid:
            raise Error("GDN verify snapshot already pending")
        var rb = self.n_gdn * REC_BYTES_PER_SLOT
        var cb = self.n_gdn * CONV_BYTES_PER_SLOT
        if self.rec_snapshot == 0:
            self.rec_snapshot = cuda_malloc(rb)
            self.conv_snapshot = cuda_malloc(cb)
        cuda_memcpy(self.rec_snapshot, self.rec_base, rb, 3)
        cuda_memcpy(self.conv_snapshot, self.conv_base, cb, 3)
        self.snapshot_valid = True

    def restore_verify_snapshot(mut self) raises:
        """Restore the exact pre-verify recurrent+conv state; keep it valid until commit."""
        if self.n_gdn == 0:
            return
        if not self.snapshot_valid:
            raise Error("GDN verify restore requested without a snapshot")
        var rb = self.n_gdn * REC_BYTES_PER_SLOT
        var cb = self.n_gdn * CONV_BYTES_PER_SLOT
        cuda_memcpy(self.rec_base, self.rec_snapshot, rb, 3)
        cuda_memcpy(self.conv_base, self.conv_snapshot, cb, 3)

    def finish_verify_transaction(mut self):
        self.snapshot_valid = False

    def rec_ptr(self, gdn_slot: Int) -> UInt64:
        """Device pointer to the recurrent state for an ordered gdn_slot (NOT a transformer layer id)."""
        return self.rec_base + UInt64(gdn_slot * REC_BYTES_PER_SLOT)

    def conv_ptr(self, gdn_slot: Int) -> UInt64:
        """Device pointer to the conv state for an ordered gdn_slot."""
        return self.conv_base + UInt64(gdn_slot * CONV_BYTES_PER_SLOT)

    def gdn_slot_for_layer(self, layer: Int) -> Int:
        """Ordered slot index for a transformer layer id, or -1 for full/softmax layers.
        Linear scan over the (<=48-entry) engine-provided list; no modulo, no fallback."""
        for i in range(self.n_gdn):
            if self.gdn_layers[i] == layer:
                return i
        return -1

    def free(mut self):
        """Release both device pools (idempotent)."""
        if self.rec_base != 0:
            cuda_free(self.rec_base)
            self.rec_base = 0
        if self.conv_base != 0:
            cuda_free(self.conv_base)
            self.conv_base = 0
        if self.rec_snapshot != 0:
            cuda_free(self.rec_snapshot)
            self.rec_snapshot = 0
        if self.conv_snapshot != 0:
            cuda_free(self.conv_snapshot)
            self.conv_snapshot = 0
        self.snapshot_valid = False
