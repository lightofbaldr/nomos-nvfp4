#!/usr/bin/env python3
"""Compare Breeze BF16, NVFP4 scratch, and opt-in small-S fused routes.

Uses the shipped FFI and the CFG replay contract. Every route is independently
scored against HF; scratch/fused differences are reported, NOT assumed zero.
All logits are banked for independent analysis. Timings exclude init/prefill,
the first frame, and codec. Nonzero exit means an HF gate failed, not a crash.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from breeze_cfg_pair_gate import bind, checked, metrics, replay


def delta(a, b):
    d = a.astype('f8') - b.astype('f8')
    return dict(array_equal=bool(np.array_equal(a, b)),
                n_diff=int(np.count_nonzero(d)), max_abs=float(np.abs(d).max()),
                relL2=float(np.linalg.norm(d) / max(np.linalg.norm(b), 1e-30)),
                top1_equal=int(np.sum(a.argmax(-1) == b.argmax(-1))),
                top1_total=int(np.prod(a.shape[:-1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--so', required=True, type=Path)
    ap.add_argument('--baseline-so', type=Path)
    ap.add_argument('--bf16-blobs', required=True, type=Path)
    ap.add_argument('--nvfp4-blobs', required=True, type=Path)
    ap.add_argument('--bench', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()
    assert a.repeats > 0
    a.output.mkdir(parents=True, exist_ok=True)
    with np.load(a.bench / 'breeze_model_goldens.npz') as g:
        prefixes = [np.ascontiguousarray(g[k].reshape(-1, 2048), 'f4') for k in
                    ('merge__out_inputs_embeds', 'merge__out_inputs_embeds_call1')]
        lg = g['headtrace__lm_head']
        dg = g['headtrace__depth_decoder__codebooks_head'].reshape(-1, 15, 2, 2051)
    with np.load(a.bench / 'breeze_codec_codes_golden.npz') as g:
        codes = np.ascontiguousarray(g['codes_full'][0].T, 'i8')
    lib = bind(a.so)
    # Per-handle env snapshot permits same-process A/B with fully fresh caches.
    arms = [('bf16', lib, a.bf16_blobs, '0'),
            ('scratch', lib, a.nvfp4_blobs, '0'),
            ('fused', lib, a.nvfp4_blobs, '1'),
            ('bf16_flag1', lib, a.bf16_blobs, '1')]
    if a.baseline_so:
        arms.insert(0, ('baseline_bf16', bind(a.baseline_so), a.bf16_blobs, '0'))
    report = dict(sha256=hashlib.sha256(a.so.read_bytes()).hexdigest(),
                  frames=len(codes), runs=[], comparisons={}, timing={})
    if a.baseline_so:
        report['baseline_sha256'] = hashlib.sha256(a.baseline_so.read_bytes()).hexdigest()
    for rep in range(a.repeats):
        for name, dll, blobs, flag in (arms if rep % 2 == 0 else arms[::-1]):
            os.environ['NOMOS_BREEZE_NVFP4_FUSED'] = flag
            h = dll.nomos_breeze_model_init(str(blobs).encode())
            assert h, name
            try:
                for paired in ((False, True) if rep % 2 == 0 else (True, False)):
                    lm, dp, timing = replay(dll, h, prefixes, codes, paired)
                    # Metrics threshold is unchanged. Also report fat BB misses
                    # so a recipe's pre-existing failure isn't called routing noise.
                    gate = metrics(lm, dp, lg, dg)
                    for lane in range(2):
                        s = np.sort(lg[:, lane].astype('f8'), axis=-1)
                        miss = lm[:, lane].argmax(-1) != lg[:, lane].argmax(-1)
                        gate[lane]['backbone_fat_misses'] = int(np.sum(miss & ((s[:, -1]-s[:, -2]) >= .1)))
                    row = dict(rep=rep, arm=name, paired=paired, timing=timing, gate=gate)
                    report['runs'].append(row)
                    np.savez(a.output / f'{name}-{int(paired)}-{rep}.npz', lm=lm, depth=dp)
                    print(json.dumps(row), flush=True)
            finally:
                checked(dll.nomos_breeze_model_free(h))
    for paired in (False, True):
        for left, right in [('fused', 'scratch'), ('bf16_flag1', 'bf16')] + (
                [('bf16', 'baseline_bf16')] if a.baseline_so else []):
            with np.load(a.output / f'{left}-{int(paired)}-0.npz') as x, np.load(
                    a.output / f'{right}-{int(paired)}-0.npz') as y:
                report['comparisons'][f'{left}-vs-{right}-paired{int(paired)}'] = {
                    key: delta(x[key], y[key]) for key in ('lm', 'depth')}
                # S=23/10 prefill must never enter the S<=2 route. BF16 logits
                # must stay byte-exact at ALL frames regardless of the flag.
                assert np.array_equal(x['lm'][0], y['lm'][0]), 'prefill changed'
                if left.startswith('bf16'):
                    assert all(np.array_equal(x[key], y[key]) for key in ('lm', 'depth'))
        for name, *_ in arms:
            runs = [r['timing'] for r in report['runs'] if r['arm'] == name and r['paired'] == paired]
            med = {k: float(np.median([r[k] for r in runs])) for k in runs[0]}
            med['model_ms'] = med['backbone_ms'] + med['depth_ms']
            report['timing'][f'{name}-paired{int(paired)}'] = med
    report['pass'] = all(g['pass'] for r in report['runs'] for g in r['gate'])
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}), flush=True)
    raise SystemExit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
