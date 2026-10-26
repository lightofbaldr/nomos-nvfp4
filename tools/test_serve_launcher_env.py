#!/usr/bin/env python3
"""The launcher's environment rules, through `serve_nomos.sh [--model X] env`, which resolves the environment exactly as
`up` does and starts nothing. CPU only, a few seconds.

What is pinned (docs/RUNBOOK.md §3d and §7):
- the KV session store is forced off for every model but Muse, and for a launch without --model (the generic
  libnomos_kernel.so is the production OLMo build), with the message saying so;
- NOMOS_HOST_ENV names the host env to source instead of ~/.nomos_host_env, and a path that does not exist is an error;
- a host env written in the override-friendly form (export VAR="${VAR:-value}") lets the caller's value win;
- the string each tmux child is started with forwards NOMOS_MAX_SEQ, NOMOS_PREFILL_CHUNK and NOMOS_HOST_ENV.

It never reads or writes the real ~/.nomos_host_env: every run gets a temporary HOME and a scrubbed environment.
Run: python3 tools/test_serve_launcher_env.py -v"""
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "pyhost/serve_nomos.sh"
FORCED = "is not Muse: NOMOS_KVSTORE=1 forced to 0 (the KV session store supports Muse only)"
# What `up` starts and stops things with. `env` must call none of them: each is replaced by a recorder.
RECORDED = ("tmux", "pkill", "pgrep", "curl")


class LauncherEnvTest(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="launcher_env_", dir=ROOT / "build"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.calls = self.tmp / "calls.log"
        for name in RECORDED:
            p = self.bin / name
            p.write_text(f'#!/bin/sh\necho "{name} $*" >> "{self.calls}"\nexit 0\n')
            p.chmod(0o755)

    def host_env(self, name, *lines):
        p = self.tmp / name
        p.write_text("\n".join(lines) + "\n")
        return p

    def run_env(self, *args, host_env=None, home_env=None, **caller):
        """`serve_nomos.sh <args> env` with nothing from this process's environment but PATH: HOME is temporary (its
        .nomos_host_env is `home_env`, if given), NOMOS_HOST_ENV is `host_env`, and `caller` is the caller's own env."""
        if home_env is not None:
            (self.home / ".nomos_host_env").write_text("\n".join(home_env) + "\n")
        env = {"PATH": f"{self.bin}:{os.environ['PATH']}", "HOME": str(self.home), **caller}
        if host_env is not None:
            env["NOMOS_HOST_ENV"] = str(host_env)
        r = subprocess.run(["bash", str(LAUNCHER), *args, "env"], cwd=self.tmp, env=env,
                           capture_output=True, text=True, timeout=120)
        out = {}
        for line in r.stdout.splitlines():
            name, sep, value = line.partition("=")
            if sep and name.isidentifier() and name.isupper():
                out[name] = value
        spec = dict(t.split("=", 1) for t in shlex.split(out.get("SPEC_ENV", "")))
        return r, out, spec

    def test_env_starts_nothing(self):
        cand = self.host_env("cand.env", 'export NOMOS_KVSTORE="${NOMOS_KVSTORE:-1}"')
        for args in ((), ("--model", "muse"), ("--model", "olmo3")):
            with self.subTest(args=args):
                r, out, _ = self.run_env(*args, host_env=cand)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn("SPEC_ENV", out)
        self.assertFalse(self.calls.exists(), "env ran " + (self.calls.read_text() if self.calls.exists() else ""))

    def test_the_store_is_forced_off_for_another_model_and_for_a_launch_without_one(self):
        on = (("override-friendly", 'export NOMOS_KVSTORE="${NOMOS_KVSTORE:-1}"'), ("plain", "NOMOS_KVSTORE=1"))
        for form, line in on:
            cand = self.host_env(f"{form}.env", line)
            for args, who in (((), "model=(none: the generic libnomos_kernel.so)"), (("--model", "olmo3"), "model=olmo3")):
                with self.subTest(form=form, args=args):
                    r, out, spec = self.run_env(*args, host_env=cand)
                    self.assertEqual(r.returncode, 0, r.stderr)
                    self.assertEqual(out["NOMOS_KVSTORE"], "0")
                    self.assertEqual(spec["NOMOS_KVSTORE"], "0", "the daemon and the serve must get it off too")
                    self.assertIn(f"[serve] {who} {FORCED}", r.stdout)
        # Turned on by the caller rather than by the host env: the same rule.
        r, out, spec = self.run_env(NOMOS_KVSTORE="1")
        self.assertEqual((out["NOMOS_KVSTORE"], spec["NOMOS_KVSTORE"]), ("0", "0"))
        self.assertIn(FORCED, r.stdout)

    def test_a_launch_without_the_store_prints_no_forced_off_line(self):
        r, out, _ = self.run_env()
        self.assertEqual(out["NOMOS_KVSTORE"], "0")
        self.assertNotIn("forced to 0", r.stdout)
        self.assertEqual(out["SO_PATH"], str(ROOT / "libnomos_kernel.so"))

    def test_the_store_stays_on_for_muse(self):
        for line in ('export NOMOS_KVSTORE="${NOMOS_KVSTORE:-1}"', "NOMOS_KVSTORE=1"):
            with self.subTest(line=line):
                r, out, spec = self.run_env("--model", "muse", host_env=self.host_env("muse.env", line))
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual((out["NOMOS_KVSTORE"], spec["NOMOS_KVSTORE"]), ("1", "1"))
                self.assertEqual(out["SO_PATH"], str(ROOT / "libnomos_kernel-muse.so"))
                self.assertNotIn("forced to 0", r.stdout)

    def test_NOMOS_HOST_ENV_is_sourced_instead_of_the_default_file(self):
        home = ['NOMOS_KVSTORE_DIR=/from/home', "NOMOS_PORT=18001"]
        cand = self.host_env("cand.env", "NOMOS_KVSTORE_DIR=/from/candidate", "NOMOS_PORT=18002")
        _, out, spec = self.run_env("--model", "muse", home_env=home)
        self.assertEqual((out["NOMOS_KVSTORE_DIR"], out["NOMOS_PORT"]), ("/from/home", "18001"))
        _, out, spec = self.run_env("--model", "muse", host_env=cand, home_env=home)
        self.assertEqual((out["NOMOS_KVSTORE_DIR"], out["NOMOS_PORT"]), ("/from/candidate", "18002"))
        self.assertEqual((spec["NOMOS_KVSTORE_DIR"], spec["NOMOS_PORT"]), ("/from/candidate", "18002"))
        self.assertEqual(out["NOMOS_HOST_ENV"], str(cand))

    def test_a_NOMOS_HOST_ENV_that_does_not_exist_is_an_error(self):
        missing = self.tmp / "no-such.env"
        r, out, _ = self.run_env("--model", "muse", host_env=missing, home_env=["NOMOS_PORT=18001"])
        self.assertEqual(r.returncode, 1)
        self.assertIn(f"NOMOS_HOST_ENV={missing}: no such file", r.stderr)
        self.assertNotIn("SPEC_ENV", out, "it must stop, not fall back to ~/.nomos_host_env")

    def test_the_override_friendly_form_lets_a_caller_override(self):
        friendly = self.host_env("friendly.env", 'export NOMOS_MAX_SEQ="${NOMOS_MAX_SEQ:-32768}"',
                                 'export NOMOS_PREFILL_CHUNK="${NOMOS_PREFILL_CHUNK:-256}"')
        _, out, _ = self.run_env("--model", "muse", host_env=friendly)
        self.assertEqual((out["NOMOS_MAX_SEQ"], out["NOMOS_PREFILL_CHUNK"]), ("32768", "256"))
        _, out, _ = self.run_env("--model", "muse", host_env=friendly, NOMOS_MAX_SEQ="8192", NOMOS_PREFILL_CHUNK="512")
        self.assertEqual((out["NOMOS_MAX_SEQ"], out["NOMOS_PREFILL_CHUNK"]), ("8192", "512"))
        # The plain form is why the friendly one is asked for (RUNBOOK §3d): it beats the caller.
        plain = self.host_env("plain.env", "NOMOS_MAX_SEQ=32768")
        _, out, _ = self.run_env("--model", "muse", host_env=plain, NOMOS_MAX_SEQ="8192")
        self.assertEqual(out["NOMOS_MAX_SEQ"], "32768")

    def test_the_children_get_max_seq_prefill_chunk_and_the_host_env(self):
        cand = self.host_env("cand.env", 'export NOMOS_MAX_SEQ="${NOMOS_MAX_SEQ:-32768}"',
                             'export NOMOS_PREFILL_CHUNK="${NOMOS_PREFILL_CHUNK:-256}"',
                             'export NOMOS_KVSTORE="${NOMOS_KVSTORE:-1}"')
        _, out, spec = self.run_env("--model", "muse", host_env=cand)
        self.assertEqual(spec["NOMOS_MAX_SEQ"], "32768")
        self.assertEqual(spec["NOMOS_PREFILL_CHUNK"], "256")
        self.assertEqual(spec["NOMOS_HOST_ENV"], str(cand))
        self.assertEqual(spec["NOMOS_KVSTORE"], "1")
        # Without a candidate the children source ~/.nomos_host_env themselves, so the variable goes empty.
        _, out, spec = self.run_env("--model", "muse")
        self.assertEqual(spec["NOMOS_HOST_ENV"], "")
        self.assertEqual((spec["NOMOS_MAX_SEQ"], spec["NOMOS_PREFILL_CHUNK"]), ("4096", "0"))   # the launcher's defaults

    def test_up_says_the_store_is_forced_off_once_and_never_for_muse(self):
        # `up` first runs this script's `down`, without --model. The rule and its message are about what a launch starts,
        # and `down` starts nothing: when it reported them, a Muse launch with the store on said "forced to 0" while the
        # store was on, and every other launch said it twice (final re-review m1).
        fast = self.tmp / "bin_up"            # recorders under which `up` gives up at once: no daemon ever appears
        fast.mkdir()
        for name, rc in (("tmux", 1), ("pkill", 0), ("pgrep", 1), ("curl", 1)):
            p = fast / name
            p.write_text(f'#!/bin/sh\necho "{name} $*" >> "{self.calls}"\nexit {rc}\n')
            p.chmod(0o755)
        cand = self.host_env("cand.env", 'export NOMOS_KVSTORE="${NOMOS_KVSTORE:-1}"')
        env = {"PATH": f"{fast}:{os.environ['PATH']}", "HOME": str(self.home), "NOMOS_HOST_ENV": str(cand),
               "NOMOS_ENGINE_SOCKET": str(self.tmp / "engine.sock")}
        for args, times in ((("--model", "muse"), 0), (("--model", "olmo3"), 1), ((), 1)):
            with self.subTest(args=args):
                r = subprocess.run(["bash", str(LAUNCHER), *args, "up"], cwd=self.tmp, env=env,
                                   capture_output=True, text=True, timeout=60)
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)       # NOT ready: nothing was started
                said = r.stdout.split("NOT ready")[0]       # what follows is the tail of the repo's real daemon.log
                self.assertEqual(said.count(FORCED), times, said)


if __name__ == "__main__":
    unittest.main()
