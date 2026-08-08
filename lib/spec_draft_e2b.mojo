"""Gemma-4-E2B LM-draft contracts for Unit F.

This module intentionally contains no model weights.  It records the draft
model shape and the token-id seam that the executor will bind to the real
``gemma-4-E2B`` checkpoint.

The E2B draft is a standalone Gemma-4 text tower, not the old 4-layer
Gemma-4 assistant scaffold.  It should be loaded by a dedicated draft engine
instead of widening the production 31B ``GemmaEngine`` shape.
"""

comptime E2B_HIDDEN = 1536
comptime E2B_NUM_LAYERS = 35
comptime E2B_NUM_HEADS = 8
comptime E2B_NUM_KV_HEADS = 1
comptime E2B_HEAD_DIM = 256
comptime E2B_GLOBAL_HEAD_DIM = 512
comptime E2B_MAX_HEAD_DIM = E2B_GLOBAL_HEAD_DIM
comptime E2B_INTERMEDIATE_EARLY = 6144
comptime E2B_INTERMEDIATE_LATE = 12288
# Allocation ceiling.  The first 15 layers use 6144-wide FFN weights; the
# final 20 KV-shared layers use 12288-wide FFN weights.
comptime E2B_INTERMEDIATE = E2B_INTERMEDIATE_LATE
comptime E2B_VOCAB = 262144
comptime E2B_SLIDING_WINDOW = 512
comptime E2B_NUM_KV_SHARED_LAYERS = 20
comptime E2B_VOCAB_SIZE_PER_LAYER_INPUT = 262144


def e2b_layer_uses_shared_kv(layer: Int) -> Bool:
    """The final 20 text layers share KV per the Gemma-4-E2B config."""
    return layer >= E2B_NUM_LAYERS - E2B_NUM_KV_SHARED_LAYERS


def e2b_layer_intermediate_size(layer: Int) -> Int:
    """Per-layer MLP width from the converted manifest."""
    if e2b_layer_uses_shared_kv(layer):
        return E2B_INTERMEDIATE_LATE
    return E2B_INTERMEDIATE_EARLY


def e2b_draft_contract_ok(K: Int, vocab: Int) -> Bool:
    """Cheap CPU sanity gate for the draft seam."""
    if K <= 0:
        return False
    if vocab != E2B_VOCAB:
        return False
    return True
