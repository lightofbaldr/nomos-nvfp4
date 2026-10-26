#!/usr/bin/env python3
"""KV session store benchmarks on Glimmer (docs/KV_SESSION_STORE_DESIGN.md §1, criterion 4):
restore latency of a 4K-token session from RAM and from NVMe (page cache dropped with posix_fadvise), and the
store's startup scan, which the first request after a restart pays. The scan is timed with the page cache dropped too,
so its figure is a cold-cache one.
GPU; run alone: pixi run --no-install python3 tools/kv_session_bench.py [--dflash]
--dflash loads the DFlash drafter, so the sessions also carry its context. The label on the result line comes from the
engine's bytes per token, so it follows KVSESSION_KV (int8 or int4) and --dflash. The session files stay under
build/kvstore_bench/ until the next run."""
import os
import shutil
import sys
import time

from kvsession_testlib import BOS, ROOT, Kernel, doc_tokens

DIR = ROOT / "build/kvstore_bench"
DRAFTER_BPT = 40960      # the drafter's fp32 context per token: 5 layers x K and V x 1024 (design §2)
KV_BPT = {27456: "INT8", 15808: "INT4 block-32"}       # Glimmer's KV bytes per token by mode (design §2)


def kv_label(bytes_per_token, dflash):
    """The mode the engine's bytes per token say it runs in, so the line never labels INT4 data as INT8."""
    kv = bytes_per_token - (DRAFTER_BPT if dflash else 0)
    name = KV_BPT.get(kv, f"{kv} B/token KV (unrecognised mode)")
    return f"{name} + drafter" if dflash else name


def drop_page_cache(root):
    """Evict every session file from the page cache, so the next read comes from the NVMe itself."""
    for d in root.iterdir():
        for f in d.glob("*.nkvs"):
            fd = os.open(f, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)


def main():
    args = sys.argv[1:]
    if any(a != "--dflash" for a in args):
        sys.exit(f"usage: kv_session_bench.py [--dflash]   (got: {' '.join(args)})")
    dflash = "--dflash" in args
    shutil.rmtree(DIR, ignore_errors=True)
    k = Kernel({"NOMOS_KVSTORE": "1", "NOMOS_KVSTORE_DIR": str(DIR), "NOMOS_KVSTORE_RAM_GB": "8"})
    if dflash:
        k.dflash_load()
    a = doc_tokens(4096)
    k.turn(a[:2048], 0, spec=dflash)
    k.prefill_cont(a[2048:])                     # 4096 tokens of KV, committed below
    k.commit()
    other = doc_tokens(8000)[4200:4800]
    k.turn(other, 0, spec=dflash)                # park the 4K session
    t0 = time.perf_counter()
    c = k.prepare(a + [a[-1]], spec=dflash)
    ram_ms = (time.perf_counter() - t0) * 1000
    k.turn(other + [other[-1]], 0, spec=dflash)  # park it again
    k.shutdown()                                 # the new Kernel below starts with an empty RAM tier: only the files remain
    k2 = Kernel({"NOMOS_KVSTORE": "1", "NOMOS_KVSTORE_DIR": str(DIR)})
    if dflash:
        k2.dflash_load()
    drop_page_cache(DIR)                         # and out of the page cache: the scan and the restore read the NVMe
    t0 = time.perf_counter()
    k2.prepare([BOS, 1])                         # creates the store: directory scan + last-record CRC per file
    startup_ms = (time.perf_counter() - t0) * 1000
    drop_page_cache(DIR)                         # the scan read the last records: make the restore cold again
    t0 = time.perf_counter()
    c2 = k2.prepare(a + [a[-1]], spec=dflash)
    nvme_ms = (time.perf_counter() - t0) * 1000
    bpt = k2.bytes_per_token()
    print(f"{kv_label(bpt, dflash)}: restore of {c} tokens: RAM {ram_ms:.1f} ms; "
          f"NVMe (cold) {nvme_ms:.1f} ms for {c2} tokens; store startup {startup_ms:.1f} ms; "
          f"bytes/token {bpt}")
    k2.shutdown()


if __name__ == "__main__":
    main()
