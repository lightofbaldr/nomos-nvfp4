"""What the KV session store supports (docs/KV_SESSION_STORE_DESIGN.md §1): Muse-Glimmer with quantized KV."""

comptime MUSE_MODEL_ID = 2

comptime KV_MODE_UNSUPPORTED = 0
comptime KV_MODE_INT8 = 1
comptime KV_MODE_INT4_B1 = 2
comptime KV_MODE_INT4_B32 = 3


def store_supported(model_id: Int) -> Bool:
    """v1 supports Muse-Glimmer only: other profiles carry state the store does not save (Qwen's GDN pools)."""
    return model_id == MUSE_MODEL_ID
