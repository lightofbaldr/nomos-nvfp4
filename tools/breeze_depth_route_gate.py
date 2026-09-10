#!/usr/bin/env python3
"""Depth recipe gate through unchanged FFI; no sampling or performance claims.

Checks combined-step (call-local caches), persistent S=1/2 and paired S=4/2
against all 35 HF frames. Also exercises full-prefix S=16 at frame0 from the
actual post-final-norm backbone hidden. Banked arrays permit byte gates.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np

from breeze_cfg_pair_gate import bind, checked, metrics, replay


def extra_bind(lib):
    i, p = C.c_int64, C.c_void_p
    lib.nomos_breeze_model_step.argtypes = [i, C.c_int32, p, p, p]
    lib.nomos_breeze_model_step.restype = C.c_int32
    lib.nomos_breeze_model_depth.argtypes = [i, p, p, p]
    lib.nomos_breeze_model_depth.restype = C.c_int32


def combined(lib, h, prefixes, codes):
    lm = np.empty((len(codes)+1, 2, 2052), 'f4')
    dp = np.empty((len(codes), 15, 2, 2051), 'f4')
    full = np.empty((1, 15, 2, 2051), 'f4')
    for lane, pre in enumerate(prefixes):
        final = np.empty((len(pre), 2048), 'f4')
        checked(lib.nomos_breeze_model_prefill_lane(
            h, lane, pre.ctypes.data, len(pre), None, final.ctypes.data,
            lm[0, lane].ctypes.data))
        # Matches _run's last_hidden write: n (post-final-norm), not x.
        backbone_hidden = np.ascontiguousarray(final[-1])
        out = np.empty((15, 2051), 'f4')
        checked(lib.nomos_breeze_model_depth(
            h, codes[0].ctypes.data, backbone_hidden.ctypes.data, out.ctypes.data))
        full[0, :, lane] = out
        for f, frame in enumerate(codes):
            checked(lib.nomos_breeze_model_step(
                h, lane, frame.ctypes.data, lm[f+1, lane].ctypes.data,
                out.ctypes.data))
            dp[f, :, lane] = out
    assert np.isfinite(lm).all() and np.isfinite(dp).all() and np.isfinite(full).all()
    return lm, dp, full


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--so', type=Path, required=True)
    ap.add_argument('--blobs', type=Path, required=True)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--equal-to', type=Path, help='Prior output directory; assert exact arrays per cadence')
    ap.add_argument('--backbone-equal-to', type=Path, help='BF16 output directory; forced-code backbone must stay exact')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    lib = bind(a.so)
    extra_bind(lib)
    with np.load(a.bench/'breeze_model_goldens.npz') as g:
        prefixes = [np.ascontiguousarray(g[k].reshape(-1, 2048), 'f4') for k in
                    ['merge__out_inputs_embeds', 'merge__out_inputs_embeds_call1']]
        lg = g['headtrace__lm_head']
        dg = g['headtrace__depth_decoder__codebooks_head'].reshape(-1, 15, 2, 2051)
    with np.load(a.bench/'breeze_codec_codes_golden.npz') as g:
        codes = np.ascontiguousarray(g['codes_full'][0].T, 'i8')
    report = dict(sha256=hashlib.sha256(a.so.read_bytes()).hexdigest(), gates={})
    h = lib.nomos_breeze_model_init(str(a.blobs).encode())
    assert h, 'init failed'
    arrays = {}
    try:
        lm, dp, full = combined(lib, h, prefixes, codes)
        arrays['combined'] = dict(lm=lm, depth=dp)
        arrays['full_prefix_frame0'] = dict(lm=lm[:1], depth=full)
        for name, paired in [('persistent', False), ('paired', True)]:
            lm, dp, _ = replay(lib, h, prefixes, codes, paired)
            arrays[name] = dict(lm=lm, depth=dp)
    finally:
        checked(lib.nomos_breeze_model_free(h))
    for name, out in arrays.items():
        lm, dp = out['lm'], out['depth']
        report['gates'][name] = metrics(lm, dp, lg[:len(lm)], dg[:len(dp)])
        np.savez(a.output/f'{name}.npz', **out)
        if a.equal_to:
            with np.load(a.equal_to/f'{name}.npz') as old:
                for key in out:
                    assert np.array_equal(out[key], old[key]), (name, key, 'regression')
        if a.backbone_equal_to:
            with np.load(a.backbone_equal_to/f'{name}.npz') as old:
                assert np.array_equal(lm, old['lm']), (name, 'backbone changed')
        print(name, json.dumps(report['gates'][name]), flush=True)
    report['pass'] = all(g['pass'] for group in report['gates'].values() for g in group)
    report['exact_regression_checked'] = bool(a.equal_to)
    report['backbone_invariance_checked'] = bool(a.backbone_equal_to)
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
