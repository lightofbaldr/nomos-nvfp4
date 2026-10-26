#!/usr/bin/env python3
"""CPU-only fixtures plus optional read-only local OLMo artifact controls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from check_profile_tokens import InvalidTokens, validate

ROOT = Path(__file__).resolve().parent.parent
CHECKER = ROOT / "tools/check_profile_tokens.py"


def profile(bos=100257, eos=100257, extra=-1, vocab=100278):
    return (f"alias VOCAB = {vocab}\nalias TARGET_BOS_ID = {bos}\n"
            f"alias TARGET_EOS_ID_0 = {eos}\nalias TARGET_EOS_ID_1 = {extra}\n")


def write_json(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


class TokenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nomos-token-test-")
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.tok = self.dir / "tokenizer with spaces"
        self.tok.mkdir()
        self.tokenizer = {
            "model": {"type": "BPE", "vocab": {'"': 1, " word": 2}},
            "added_tokens": [
                {"id": 100257, "content": "<|endoftext|>", "special": True},
                {"id": 100265, "content": "<|im_end|>", "special": True},
            ],
        }
        self.config = {"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>"}
        self.save()
        self.prof = self.dir / "profile.mojo"
        self.prof.write_text(profile())

    def save(self):
        write_json(self.tok / "tokenizer.json", self.tokenizer)
        write_json(self.tok / "tokenizer_config.json", self.config)

    def rejects(self, message, source=None):
        with self.assertRaisesRegex(InvalidTokens, message):
            validate(source or profile(), self.tok)

    def cli(self, *args):
        return subprocess.run([sys.executable, str(CHECKER), "--profile", str(self.prof),
                               "--tokenizer-dir", str(self.tok), *map(str, args)],
                              capture_output=True, text=True, timeout=20)

    def test_good_bos_equals_eos(self):
        self.assertTrue(validate(profile(), self.tok))

    def test_known_bad_olmo_quote_bos(self):
        self.rejects("TARGET_BOS_ID.*mismatches.*bos_token", profile(bos=1))
        self.prof.write_text(profile(bos=1))
        result = self.cli()
        self.assertEqual(result.returncode, 1)
        self.assertIn("100257", result.stderr)
        self.assertIn('"', result.stderr)
        self.assertNotIn("PROFILE SPECIAL TOKENS OK", result.stdout)

    def test_token_object_and_decoder(self):
        self.config["bos_token"] = {"content": "<|endoftext|>"}
        self.config["added_tokens_decoder"] = {
            "100257": {"content": "<|endoftext|>", "special": True}}
        self.save()
        self.assertTrue(validate(profile(), self.tok))

    def test_decoder_cannot_invent_or_reassign_id(self):
        self.config["added_tokens_decoder"] = {"1": {"content": "<|endoftext|>", "special": True}}
        self.save()
        self.rejects("decoder disagrees")

    def test_missing_files_fail(self):
        for filename in ("tokenizer.json", "tokenizer_config.json"):
            with self.subTest(filename=filename):
                (self.tok / filename).unlink()
                self.rejects("cannot read")
                self.save()

    def test_malformed_json_fails(self):
        (self.tok / "tokenizer.json").write_text("{")
        self.rejects("cannot read")

    def test_duplicate_json_field_fails(self):
        (self.tok / "tokenizer_config.json").write_text(
            '{"bos_token":"wrong", "bos_token":"<|endoftext|>"}')
        self.rejects("duplicate JSON key")

    def test_absent_bos_not_assumed_equal(self):
        del self.config["bos_token"]
        self.save()
        self.rejects("no authoritative BOS")

    def test_numeric_bos_from_text_config(self):
        del self.config["bos_token"]
        self.save()
        write_json(self.tok / "config.json", {"text_config": {"bos_token_id": 100257}})
        self.assertTrue(validate(profile(), self.tok))

    def test_conflicting_numeric_bos_fails(self):
        write_json(self.tok / "config.json", {"bos_token_id": 1})
        self.rejects("mismatches config.json:bos_token_id")

    def test_conflicting_special_tokens_map_fails(self):
        write_json(self.tok / "special_tokens_map.json", {"bos_token": '"'})
        self.rejects("mismatches special_tokens_map.json")

    def test_unknown_token_name_fails(self):
        self.config["bos_token"] = "missing"
        self.save()
        self.rejects("not in tokenizer.json")

    def test_wrong_eos_fails(self):
        self.rejects("TARGET_EOS_IDs.*mismatches", profile(eos=100265))

    def test_extra_profile_eos_must_be_special(self):
        self.rejects("not declared special", profile(extra=2))
        self.assertTrue(validate(profile(extra=100265), self.tok))

    def test_generation_eos_superset_allowed(self):
        write_json(self.tok / "generation_config.json", {"eos_token_id": [100257, 100265]})
        self.assertTrue(validate(profile(), self.tok))

    def test_conflicting_generation_eos_fails(self):
        write_json(self.tok / "generation_config.json", {"eos_token_id": 100265})
        self.rejects("mismatches generation_config.json:eos_token_id")

    def test_profile_missing_duplicate_expression_out_of_range(self):
        for source in (profile().replace("alias TARGET_BOS_ID = 100257\n", ""),
                       profile() + "alias TARGET_BOS_ID = 1\n",
                       profile().replace("= 100257", "= 100000 + 257", 1),
                       profile(bos=-1), profile(extra=-2), profile(vocab=2)):
            with self.subTest(source=source):
                with self.assertRaises(InvalidTokens):
                    validate(source, self.tok)

    def test_invalid_ids_are_not_coerced(self):
        for value in (True, "100257", -1, [], [100257, 100265]):
            with self.subTest(value=value):
                write_json(self.tok / "config.json", {"bos_token_id": value})
                with self.assertRaises(InvalidTokens):
                    validate(profile(), self.tok)

    def test_conflicting_vocabulary_entries_fail(self):
        self.tokenizer["added_tokens"].append({"id": 1, "content": "wrong", "special": True})
        self.save()
        self.rejects("conflicting content")

    def test_unigram_supported(self):
        self.tokenizer["model"] = {"type": "Unigram", "vocab": [["x", 0.0], ['"', -1.0]]}
        self.save()
        self.assertTrue(validate(profile(), self.tok))

    def test_atomic_config_preserves_old_on_failure(self):
        output = self.dir / "model_config.mojo"
        output.write_text("existing config\n")
        self.prof.write_text(profile(bos=1))
        result = self.cli("--write-config", output)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(output.read_text(), "existing config\n")
        self.prof.write_text(profile())
        result = self.cli("--write-config", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual("\n".join(output.read_text().splitlines()[2:]) + "\n", profile())
        self.assertEqual(list(self.dir.glob(".model_config-*")), [])

    def build_fixture(self, *args, tokenizer_env=None, expect_compiler_boundary=False):
        build = self.dir / "build"
        (build / "tools").mkdir(parents=True, exist_ok=True)
        (build / "lib/model_profiles").mkdir(parents=True, exist_ok=True)
        # Keep the real preflight verbatim; replace EVERYTHING downstream of it
        # with a tripwire so even a regressed gate cannot launch CUDA/pixi/gcc.
        build_source = (ROOT / "refresh_build.sh").read_text()
        boundary = 'case "$(uname -m)" in'
        self.assertEqual(build_source.count(boundary), 1)
        (build / "refresh_build.sh").write_text(build_source.split(boundary)[0] +
            'echo "UNEXPECTED COMPILER BOUNDARY" >&2\nexit 91\n')
        shutil.copy2(CHECKER, build / "tools/check_profile_tokens.py")
        shutil.copy2(self.prof, build / "lib/model_profiles/olmo3.mojo")
        (build / "lib/model_config.mojo").write_text("keep existing config")
        (build / "libnomos_kernel.so").write_bytes(b"keep existing binary")
        env = os.environ.copy()
        env.pop("NOMOS_TOKENIZER_DIR", None)
        env["NOMOS_BUILD_PYTHON"] = sys.executable
        if tokenizer_env is not None:
            env["NOMOS_TOKENIZER_DIR"] = tokenizer_env
        result = subprocess.run(["bash", str(build / "refresh_build.sh"), "--model", "olmo3", *args],
                                capture_output=True, text=True, timeout=20, env=env, cwd=self.dir)
        generated = (build / "lib/model_config.mojo").read_text()
        if expect_compiler_boundary:
            self.assertEqual("\n".join(generated.splitlines()[2:]) + "\n", self.prof.read_text())
            self.assertEqual(result.returncode, 91)
            self.assertIn("UNEXPECTED COMPILER BOUNDARY", result.stderr)
        else:
            self.assertEqual(generated, "keep existing config")
            self.assertNotIn("UNEXPECTED COMPILER BOUNDARY", result.stderr)
        self.assertEqual((build / "libnomos_kernel.so").read_bytes(), b"keep existing binary")
        self.assertNotIn("CUDA_ROOT=", result.stdout)
        return result

    def test_build_rejects_missing_path_before_any_build_work(self):
        result = self.build_fixture()
        self.assertEqual(result.returncode, 1)
        self.assertIn("is required", result.stderr)
        self.assertIn('pyhost/models.py: MODELS["olmo3"]["tok_dir"]', result.stderr)
        self.assertIn("not selected automatically", result.stderr)

    def test_build_rejects_known_bad_profile_without_dry_run(self):
        self.prof.write_text(profile(bos=1))
        result = self.build_fixture("--tokenizer-dir", str(self.tok))
        self.assertEqual(result.returncode, 1)
        self.assertIn("TARGET_BOS_ID", result.stderr)

    def test_build_positive_publishes_checked_source_before_compiler(self):
        result = self.build_fixture("--tokenizer-dir", str(self.tok), expect_compiler_boundary=True)
        self.assertIn("PROFILE SPECIAL TOKENS OK", result.stdout)

    def test_build_positive_check_only_and_environment(self):
        for args, tok_env in ((("--tokenizer-dir", str(self.tok)), None), ((), str(self.tok))):
            with self.subTest(args=args):
                result = self.build_fixture("--check-profile", *args, tokenizer_env=tok_env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("PROFILE SPECIAL TOKENS OK", result.stdout)

    def test_cli_path_overrides_environment(self):
        result = self.build_fixture("--check-profile", "--tokenizer-dir=" + str(self.tok),
                                    tokenizer_env="/nonexistent-tokenizer")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_build_missing_argument(self):
        result = self.build_fixture("--tokenizer-dir")
        self.assertEqual(result.returncode, 1)
        self.assertIn("requires a value", result.stderr)


def local_controls(tokenizer_dir):
    # Never mutate the live profile/tokenizer. Exercise BOTH states irrespective of
    # whether the concurrently owned live OLMo profile has already been corrected.
    source = (ROOT / "lib/model_profiles/olmo3.mojo").read_text()
    import re
    with tempfile.TemporaryDirectory(prefix="nomos-local-token-control-") as td:
        for bos, expected in ((1, 1), (100257, 0)):
            path = Path(td) / f"olmo-bos-{bos}.mojo"
            path.write_text(re.sub(r"(?m)^alias TARGET_BOS_ID = .*", f"alias TARGET_BOS_ID = {bos}", source))
            result = subprocess.run([sys.executable, str(CHECKER), "--profile", str(path),
                                     "--tokenizer-dir", str(tokenizer_dir)],
                                    capture_output=True, text=True, timeout=30)
            print(result.stdout + result.stderr, end="")
            if result.returncode != expected:
                raise AssertionError((bos, expected, result.returncode))
            if bos == 1 and "TARGET_BOS_ID" not in result.stderr:
                raise AssertionError("negative control failed for the wrong reason")
            if bos == 100257 and "PROFILE SPECIAL TOKENS OK" not in result.stdout:
                raise AssertionError("missing positive result")
    print("LOCAL OLMO NEGATIVE AND POSITIVE CONTROLS PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-tokenizer", type=Path)
    args = parser.parse_args()
    if args.local_tokenizer:
        local_controls(args.local_tokenizer.expanduser())
    else:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TokenTests))
        if not result.wasSuccessful():
            sys.exit(1)
        print("PROFILE TOKEN TESTS PASSED")
