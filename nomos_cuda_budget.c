#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_EVENTS = 1024, CATS = 14 };

static cudaEvent_t events[MAX_EVENTS];
static int event_cats[MAX_EVENTS];
static int event_count;
static int active;
static int initialized;
static double totals_ms[CATS];
static uint64_t traced_tokens;

static void ensure_events(void) {
    if (initialized) return;
    for (int i = 0; i < MAX_EVENTS; ++i)
        cudaEventCreateWithFlags(&events[i], cudaEventDefault);
    initialized = 1;
}

__attribute__((constructor)) static void nomos_budget_init(void) {
    /* Keep one-time event creation out of the first scored token. */
    ensure_events();
}

int nomos_budget_reset(void) {
    ensure_events();
    memset(totals_ms, 0, sizeof(totals_ms));
    traced_tokens = 0;
    event_count = 0;
    active = 1;
    return 0;
}

int nomos_budget_active(void) {
    if (!active) {
        const char *enabled = getenv("NOMOS_CUDA_BUDGET");
        if (enabled && enabled[0] == '1') nomos_budget_reset();
    }
    return active;
}

int nomos_budget_token_begin(void *stream) {
    if (!active) return 0;
    event_count = 0;
    cudaEventRecord(events[event_count++], (cudaStream_t)stream);
    return 0;
}

int nomos_budget_mark(void *stream, int category) {
    if (!active || category < 0 || category >= CATS || event_count >= MAX_EVENTS)
        return 0;
    cudaEventRecord(events[event_count], (cudaStream_t)stream);
    /* Category describes work between the previous event and this one. */
    event_cats[event_count] = category;
    ++event_count;
    return 0;
}

int nomos_budget_token_end(void *stream) {
    (void)stream;
    if (!active || event_count < 2) return 0;
    cudaEventSynchronize(events[event_count - 1]);
    for (int idx = 1; idx < event_count; ++idx) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, events[idx - 1], events[idx]);
        totals_ms[event_cats[idx]] += ms;
    }
    ++traced_tokens;
    return 0;
}

int nomos_budget_report(void) {
    static const char *names[CATS] = {
        "qkv", "kv_append", "kv_deq_attn", "o_proj", "layer_small_1",
        "gate", "up", "activation", "down", "layer_small_2",
        "final_norm_transfer", "lmhead_argmax", "attention_gate",
        "attention_sigmoid"
    };
    active = 0;
    printf("[cuda-budget] tokens=%llu\n", (unsigned long long)traced_tokens);
    double event_total = 0.0;
    for (int i = 0; i < CATS; ++i)
        event_total += totals_ms[i];
    printf("[cuda-budget] event_total          total_ms=%.3f ms_per_tok=%.3f\n",
           event_total, traced_tokens ? event_total / traced_tokens : 0.0);
    for (int i = 0; i < CATS; ++i)
        printf("[cuda-budget] %-20s total_ms=%.3f ms_per_tok=%.3f\n",
               names[i], totals_ms[i], traced_tokens ? totals_ms[i] / traced_tokens : 0.0);
    fflush(stdout);
    return 0;
}

__attribute__((destructor)) static void nomos_budget_report_at_exit(void) {
    if (traced_tokens) nomos_budget_report();
}
