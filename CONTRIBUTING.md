# Contributing to Nomos Kernel

Thanks for your interest. Nomos is a correctness-first inference kernel: its
central promise is that speculative decoding is **bit-identical to greedy
decoding by construction**. Contributions are held to that bar.

## The gates (every change to the compute path must pass)

1. **It builds, on-target.** `bash refresh_build.sh` succeeds and
   `nm -D libnomos_kernel.so | grep -c ' T nomos_'` still reports the full ABI
   (currently 85). A change to the exported surface must be intentional and
   documented.
2. **Lossless greedy parity.** `./nomos bench --spec` must report every prompt
   `token-exact vs greedy`. Speculative output that diverges from the target
   model's own greedy stream is a bug, not a speedup.
3. **Refactors preserve behavior byte-for-byte.** A pure refactor (moving code,
   deleting dead code, splitting a file) must leave the greedy token stream
   byte-identical to before. Dump ids before and after and diff them.

## Benchmark hygiene

Performance numbers are only meaningful under controlled conditions:

- Measure on an **idle GPU** (no other CUDA workload sharing the card).
- Compare **same-session, paired A/B** — absolute tok/s carries double-digit
  box-state variance across reboots, so never compare a number today against
  one from another day. Re-measure the baseline in the same session.
- Watch the base-decode canary: a depressed base number means contamination,
  not a real regression in your change.

## Mojo version

The kernel pins a specific Mojo nightly in `pixi.toml` (Mojo evolves quickly and
its GPU/stdlib APIs change). Build with the pinned toolchain; if you bump it,
expect to fix breaking changes and re-run all three gates.

## Style

Match the surrounding code. Keep modules in the ~500–800 line band; extract
free `def`s taking `mut self: GemmaEngine` rather than growing the engine struct.

## Sign-off

By contributing you agree your contribution is licensed under the repository's
MIT License (see `LICENSE`).
