/* No-op stub for the cgo callback symbol. In production the Go bridge
 * exports this; standalone test binaries link this stub to satisfy the
 * linker. encode_topk64 only calls run_inference with cb_id=0, which
 * short-circuits before invoking this function. */
#include <stdint.h>

int32_t nomos_token_cb(int64_t cb_id, int32_t token_id) {
    (void)cb_id;
    (void)token_id;
    return 0;
}

/* Optional token-override callback stub: -1 = no override (kernel samples
 * normally). The Go bridge exports the real implementation. */
int32_t nomos_blend_cb(int64_t cb_id, int64_t a, int64_t b,
                       int64_t c, int64_t d) {
    (void)cb_id; (void)a; (void)b; (void)c; (void)d;
    return -1;
}
