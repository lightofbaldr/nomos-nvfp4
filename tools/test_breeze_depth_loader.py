#!/usr/bin/env python3
"""Run the standalone loader probe on generated positive/negative fixtures.

No model weights or shipping FFI exports are needed. Each process must fail
for the expected loader reason, not merely crash or fail to find CUDA.
"""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('probe', type=Path)
    a = ap.parse_args()
    probe = str(a.probe.resolve())
    k, n = 32, 16
    elems = k*n
    body = b'\x38'*(elems//16) + b'\x22'*(elems//2)
    def packed(count=elems, blocks=elems//16, scale=1.0, payload=body):
        return struct.pack('<qqff', count, blocks, scale, 0.0) + payload
    cases = [
        ('packed-selected', packed(), True, 1, 1, None),
        ('disabled-bf16', packed(), True, 0, 0, None),
        ('enabled-absent-fallback', None, True, 1, 0, None),
        ('disabled-absent-fallback', None, True, 0, 0, None),
        ('packed-only', packed(), False, 1, 1, None),
        ('disabled-packed-only', packed(), False, 0, 0, 'BF16 projection'),
        ('both-missing', None, False, 1, 0, 'BF16 projection'),
        ('empty-packed-no-fallback', b'', True, 1, 0, 'NVFP4 header'),
        ('short-header', b'\0'*23, True, 1, 0, 'NVFP4 header'),
        ('wrong-elements', packed(count=elems+16), True, 1, 0, 'dimensions/body'),
        ('wrong-blocks', packed(blocks=1), True, 1, 0, 'dimensions/body'),
        ('short-body', packed(payload=body[:-1]), True, 1, 0, 'dimensions/body'),
        ('extra-body', packed(payload=body+b'\0'), True, 1, 0, 'dimensions/body'),
        ('zero-scale', packed(scale=0), True, 1, 0, 'global scale'),
        ('negative-scale', packed(scale=-1), True, 1, 0, 'global scale'),
        ('nan-scale', packed(scale=float('nan')), True, 1, 0, 'global scale'),
        ('inf-scale', packed(scale=float('inf')), True, 1, 0, 'global scale'),
        ('disabled-ignores-bad-packed', b'', True, 0, 0, None),
    ]
    with tempfile.TemporaryDirectory(prefix='breeze-depth-loader-') as td:
        for name, nv, bf, enabled, expected, error in cases:
            stem = Path(td)/name
            if nv is not None:
                stem.with_suffix('.nvfp4').write_bytes(nv)
            if bf:
                stem.with_suffix('.bf16').write_bytes(bytes(elems*2))
            r = subprocess.run([probe, str(stem), str(k), str(n), str(enabled), str(expected)],
                               capture_output=True, text=True, timeout=30)
            output = r.stdout + r.stderr
            if error:
                assert r.returncode > 0 and error in output, (name, r.returncode, output)
            else:
                assert r.returncode == 0 and 'PASS depth loader' in output, (name, output)
            print('PASS', name)
    print(f'{len(cases)}/{len(cases)} loader checks passed')


if __name__ == '__main__':
    main()
