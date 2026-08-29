#!/usr/bin/env python3
"""#71 — batched-prefill vs per-token-decode SELF-parity (Prime, 2026-08-10).

Kvasir's reframe: prefill tracks HF (~0.988) while decode drifts below it (~0.975),
so our TWO paths disagree with each other. HF cannot arbitrate that -- it has one
code path and no such split. Measuring each of ours against HF separately estimates
the disagreement through a noisy third party.

This measures it DIRECTLY, kernel against itself:

    same prefix, same weights, same engine, two routes to the SAME position
      A (k = N): prefill(ids[:N])                      -> logits predicting N
      B (k < N): prefill(ids[:k]) then decode_step each of ids[k:N]  -> logits predicting N

A self-consistent kernel returns the SAME logits for every k. Any spread IS the
parity break, with no reference-model error in the measurement. Sweeping k also
separates the two candidate shapes:

    spread grows with (N-k)  -> per-decode-step error accumulating in the KV cache
    spread flat in (N-k)     -> a fixed prefill/decode entry difference

CONTROL FIRST: path A is run twice. If the kernel is not bit-deterministic for an
identical call, every number below is noise and the run says so instead of
reporting a divergence that is really nondeterminism. CUBLAS_WORKSPACE_CONFIG must
be set for that control to be meaningful.

Prompts are the two #79 goldens EXTENDED by their own golden continuations, so the
prose/code axis (which both arches show) is preserved at a length where accumulation
has room to show: en 6+32=38, code 14+32=46.
"""
import ctypes as C
import os
import sys

import numpy as np

VOCAB = 202048


def bind(so_path):
    lib = C.CDLL(so_path)
    sigs = {
        "nomos_init":        (C.c_int64, [C.c_int64]),
        "nomos_prefill":     (C.c_int32, [C.c_int64, C.c_int64, C.c_int32, C.c_int64]),
        "nomos_decode_step": (C.c_int32, [C.c_int64, C.c_int32, C.c_int64]),
        "nomos_reset_kv":    (C.c_int32, [C.c_int64]),
    }
    for name, (r, a) in sigs.items():
        f = getattr(lib, name)
        f.restype, f.argtypes = r, a
    return lib


def cos(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else 0.0


def run(lib, handle, ids, k, N):
    """Prefill the first k tokens, decode the rest. Returns logits predicting N."""
    out = np.zeros(VOCAB, np.float32)
    lib.nomos_reset_kv(handle)
    pre = np.ascontiguousarray(ids[:k], dtype=np.int32)
    rc = lib.nomos_prefill(handle, pre.ctypes.data, k, out.ctypes.data)
    if rc != 0:
        raise RuntimeError(f"prefill rc={rc} (k={k})")
    for j in range(k, N):
        rc = lib.nomos_decode_step(handle, C.c_int32(int(ids[j])), out.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"decode rc={rc} at j={j}")
    return out.copy()


def main():
    so, wdir = sys.argv[1], sys.argv[2]
    goldens_dir = (
        sys.argv[3] if len(sys.argv) > 3
        else "~/Shares/efs/nomos-artifacts/muse-goldens-2026-08-10"
    )
    if not wdir.endswith("/"):
        wdir += "/"
    lib = bind(so)
    wbuf = C.create_string_buffer(wdir.encode())
    handle = lib.nomos_init(C.cast(wbuf, C.c_void_p).value)
    if not handle:
        sys.exit("FATAL: nomos_init returned 0")
    print(f"engine up (handle={handle:#x})\n", flush=True)

    for name in ("en_short", "code_short"):
        z = np.load(os.path.join(goldens_dir, f"golden_{name}.npz"))
        ids = np.concatenate([z["prompt_ids"].astype(np.int32),
                              z["step_tokens"].astype(np.int32)])
        N = len(ids)
        prompt_n = len(z["prompt_ids"])
        kind = "PROSE" if name == "en_short" else "CODE"
        print(f"=== {name} ({kind})  N={N} (prompt {len(z['prompt_ids'])} + "
              f"{len(z['step_tokens'])} golden continuation) ===", flush=True)

        # Minimal boundary repro for the first decode output: compare prefill(prompt +
        # first golden token) with prefill(prompt) then one decode_step(first token).
        # Both predict golden step 1; only the route used for the prompt's final token
        # differs in run_inference_impl's split prefill (batched prefix + single last row).
        boundary_n = prompt_n + 1
        boundary_prefill = run(lib, handle, ids, boundary_n, boundary_n)
        boundary_decode = run(lib, handle, ids, prompt_n, boundary_n)
        boundary_decode_repeat = run(lib, handle, ids, prompt_n, boundary_n)
        boundary_gold = int(z["step_tokens"][1])
        print(
            f"  boundary P={prompt_n}: prefill(P+1) vs prefill(P)+decode1 "
            f"cos={cos(boundary_prefill, boundary_decode):.6f} "
            f"max|d|={float(np.abs(boundary_prefill-boundary_decode).max()):.4f} "
            f"top1={int(np.argmax(boundary_prefill))}/{int(np.argmax(boundary_decode))} "
            f"gold={boundary_gold} repeat={'identical' if np.array_equal(boundary_decode, boundary_decode_repeat) else 'NONDETERMINISTIC'}",
            flush=True,
        )

        ref = run(lib, handle, ids, N, N)          # all-prefill
        ref2 = run(lib, handle, ids, N, N)         # CONTROL: identical call again
        d = float(np.abs(ref - ref2).max())
        if d > 0:
            print(f"  !! NONDETERMINISM: identical prefill differs by max|d|={d:.3e}."
                  f" Numbers below are not interpretable.", flush=True)
        else:
            print("  control: all-prefill is bit-identical across repeat calls  OK",
                  flush=True)

        top_ref = int(np.argmax(ref))
        ks = [k for k in [1, 2, 4, 8, 16, 24, 32, N - 1] if 1 <= k < N]

        # ORDER CONTROL. The first sweep grouped into blocks that were contiguous in
        # CALL ORDER, and k=N-1 (one decode step) diverged MORE than k=1 (N-1 steps) --
        # backwards for accumulation. That is equally consistent with state surviving
        # nomos_reset_kv between differently-shaped calls. So run the sweep ascending
        # and again descending: if a k's result depends on k it is identical in both
        # passes; if it depends on what ran before it, the two passes disagree.
        res = {}
        for tag, order in (("asc", ks), ("desc", list(reversed(ks)))):
            for k in order:
                res[(tag, k)] = run(lib, handle, ids, k, N)

        print(f"\n  {'k':>6} {'decoded':>8} {'cos vs prefill':>15} {'max|d|':>10} "
              f"{'top1':>10} | {'asc==desc?':>12}")
        print("  " + "-" * 74)
        order_dep = False
        for k in ks:
            a, d_ = res[("asc", k)], res[("desc", k)]
            same = np.array_equal(a, d_)
            order_dep |= not same
            c = cos(a, ref)
            md = float(np.abs(a - ref).max())
            t = int(np.argmax(a))
            mark = "same" if t == top_ref else f"DIFF({t})"
            print(f"  {k:>6} {N - k:>8} {c:>15.6f} {md:>10.4f} {mark:>10} | "
                  f"{'identical' if same else 'ORDER-DEPENDENT':>12}", flush=True)
        if order_dep:
            print("\n  >> RESULTS DEPEND ON CALL ORDER: state survives nomos_reset_kv.")
            print("     The k-sweep measures history, not prefill-vs-decode. STOP.")
        else:
            print("\n  >> order-independent: the k-dependence is real.")
        print(flush=True)


if __name__ == "__main__":
    main()
