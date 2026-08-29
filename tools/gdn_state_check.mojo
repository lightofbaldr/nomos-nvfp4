# Compile-check harness for lib/gdn_state.mojo (Kvasir's GDN state pools).
# Not shipped; `pixi run mojo build tools/gdn_state_check.mojo` typechecks the module against the
# current model_config. On a non-GDN profile (GDN_*=0) it compiles but must NOT be RUN (a 0-byte
# cuda_malloc aborts); the point is the compile. On qwen3_5 it also exercises the mapping.
from lib.gdn_state import GdnStatePools


def main():
    # The engine will pass the ordered GDN list (complement of layer_is_full); here we build the
    # qwen3_5 set explicitly for the mapping check only.
    var layers = List[Int]()
    for i in range(64):
        if (i + 1) % 4 != 0:
            layers.append(i)
    var pools = GdnStatePools(layers)
    print("n_gdn =", pools.n_gdn)
    print("slot(layer 0) =", pools.gdn_slot_for_layer(0))   # expect 0
    print("slot(layer 3) =", pools.gdn_slot_for_layer(3))   # full -> expect -1
    print("slot(layer 62) =", pools.gdn_slot_for_layer(62)) # expect 47
    print("rec_ptr(0) =", pools.rec_ptr(0))
    print("conv_ptr(1) =", pools.conv_ptr(1))
    pools.free()
