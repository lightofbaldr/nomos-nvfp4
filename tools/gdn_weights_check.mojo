# Compile-check harness for lib/gdn_weights.mojo. `pixi run mojo build -I . tools/gdn_weights_check.mojo`.
from lib.gdn_weights import GdnWeights
def main():
    var w = GdnWeights("/tmp/nonexistent", 0)  # n_gdn=0 -> no file loads; compile-only
    print("n_gdn =", w.n_gdn)
    w.free()
