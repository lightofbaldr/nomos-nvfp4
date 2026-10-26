#!/usr/bin/env python3
"""CPU-only, offline build gate for profile special tokens against local artifacts.

No transformers import, remote code, network, shared-library load, or GPU access.
This checks source inputs to a build, NOT the identity of an already-built binary.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile


class InvalidTokens(ValueError):
    pass


def read_object(path: Path) -> dict:
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InvalidTokens(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    except (OSError, ValueError) as exc:
        raise InvalidTokens(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InvalidTokens(f"{path}: expected a JSON object")
    return value


def integer(value, source: str) -> int:
    if type(value) is not int or value < 0:
        raise InvalidTokens(f"{source}: expected a nonnegative integer ID, got {value!r}")
    return value


def profile_values(source: str) -> dict[str, int]:
    values = {}
    for name in ("VOCAB", "TARGET_BOS_ID", "TARGET_EOS_ID_0", "TARGET_EOS_ID_1"):
        matches = re.findall(rf"^\s*alias\s+{name}\s*=\s*([^#\n]+)", source, re.M)
        if len(matches) != 1 or not re.fullmatch(r"-?\d+", matches[0].strip()):
            raise InvalidTokens(f"profile must declare exactly one literal integer {name}")
        values[name] = int(matches[0].strip())
    vocab = values["VOCAB"]
    if vocab <= 0:
        raise InvalidTokens("profile VOCAB must be positive")
    for name, value in values.items():
        if name == "VOCAB" or (name == "TARGET_EOS_ID_1" and value == -1):
            continue
        if not 0 <= value < vocab:
            raise InvalidTokens(f"profile {name}={value} is outside [0, VOCAB={vocab})")
    return values


def validate(profile_source: str, tokenizer_dir: Path) -> list[str]:
    values = profile_values(profile_source)
    tokenizer = read_object(tokenizer_dir / "tokenizer.json")
    config = read_object(tokenizer_dir / "tokenizer_config.json")
    by_text: dict[str, int] = {}
    by_id: dict[int, str] = {}
    special: set[int] = set()

    def add(text, token_id, source, is_special=False):
        token_id = integer(token_id, source)
        if not isinstance(text, str) or not text:
            raise InvalidTokens(f"{source}: missing token content")
        if text in by_text and by_text[text] != token_id:
            raise InvalidTokens(f"{source}: conflicting IDs for token {text!r}")
        if token_id in by_id and by_id[token_id] != text:
            raise InvalidTokens(f"{source}: conflicting content for token ID {token_id}")
        by_text[text] = token_id
        by_id[token_id] = text
        if is_special:
            special.add(token_id)

    model = tokenizer.get("model")
    if not isinstance(model, dict):
        raise InvalidTokens("tokenizer.json: missing model vocabulary")
    vocab = model.get("vocab")
    if isinstance(vocab, dict):
        for text, token_id in vocab.items():
            add(text, token_id, "tokenizer.json:model.vocab")
    elif isinstance(vocab, list):
        for token_id, entry in enumerate(vocab):
            if not isinstance(entry, list) or len(entry) != 2:
                raise InvalidTokens("tokenizer.json: invalid Unigram vocabulary entry")
            add(entry[0], token_id, "tokenizer.json:model.vocab")
    else:
        raise InvalidTokens("tokenizer.json: unsupported or absent vocabulary")
    added = tokenizer.get("added_tokens", [])
    if not isinstance(added, list):
        raise InvalidTokens("tokenizer.json: added_tokens must be a list")
    for entry in added:
        if not isinstance(entry, dict):
            raise InvalidTokens("tokenizer.json: invalid added token")
        add(entry.get("content"), entry.get("id"), "tokenizer.json:added_tokens",
            entry.get("special") is True)
    if not by_id:
        raise InvalidTokens("tokenizer.json: empty token vocabulary")
    # The decoder is corroboration, not authority to invent IDs absent from tokenizer.json.
    decoder = config.get("added_tokens_decoder", {})
    if not isinstance(decoder, dict):
        raise InvalidTokens("tokenizer_config.json: added_tokens_decoder must be an object")
    for key, entry in decoder.items():
        if not key.isdecimal() or not isinstance(entry, dict):
            raise InvalidTokens("tokenizer_config.json: invalid added_tokens_decoder entry")
        token_id = int(key)
        if by_id.get(token_id) != entry.get("content"):
            raise InvalidTokens(f"tokenizer_config.json: decoder disagrees with tokenizer.json at ID {key}")
        if entry.get("special") is True:
            special.add(token_id)
    if max(by_id) >= values["VOCAB"]:
        raise InvalidTokens(f"tokenizer ID {max(by_id)} exceeds profile VOCAB={values['VOCAB']}")

    documents = [("tokenizer_config.json", config)]
    for name in ("special_tokens_map.json", "config.json", "generation_config.json"):
        path = tokenizer_dir / name
        if path.exists():
            obj = read_object(path)
            documents.append((name, obj))
            if "text_config" in obj:
                if not isinstance(obj["text_config"], dict):
                    raise InvalidTokens(f"{name}: text_config must be an object")
                documents.append((name + ":text_config", obj["text_config"]))

    evidence: dict[str, list[tuple[str, set[int]]]] = {"bos": [], "eos": []}
    for source, obj in documents:
        for kind in evidence:
            text = obj.get(kind + "_token")
            if text is not None:
                if isinstance(text, dict):
                    text = text.get("content")
                if not isinstance(text, str) or text not in by_text:
                    raise InvalidTokens(f"{source}:{kind}_token {text!r} not in tokenizer.json")
                evidence[kind].append((source + ":" + kind + "_token", {by_text[text]}))
                special.add(by_text[text])
            ids = obj.get(kind + "_token_id")
            if ids is not None:
                ids = ids if isinstance(ids, list) else [ids]
                if not ids or (kind == "bos" and len(ids) != 1):
                    raise InvalidTokens(f"{source}:{kind}_token_id: invalid ID list")
                resolved = {integer(v, source + ":" + kind + "_token_id") for v in ids}
                if not resolved <= by_id.keys():
                    raise InvalidTokens(f"{source}:{kind}_token_id absent from tokenizer.json: {sorted(resolved)}")
                evidence[kind].append((source + ":" + kind + "_token_id", resolved))
    messages = []
    bos = values["TARGET_BOS_ID"]
    eos = {values["TARGET_EOS_ID_0"], values["TARGET_EOS_ID_1"]} - {-1}
    for kind, sources in evidence.items():
        if not sources:
            raise InvalidTokens(f"no authoritative {kind.upper()} metadata in {tokenizer_dir}")
        expected = {bos} if kind == "bos" else eos
        for source, actual in sources:
            # EOS can have additional generation-only turn terminators; require overlap,
            # not equality with that whole policy list (e.g. Muse eot vs canonical eos).
            if (kind == "bos" and actual != expected) or not actual & expected:
                raise InvalidTokens(
                    f"profile TARGET_{kind.upper()}_ID{'s' if kind == 'eos' else ''}={sorted(expected)} "
                    f"mismatches {source}={sorted(actual)}; "
                    f"profile IDs decode as {[by_id.get(i, '<absent>') for i in sorted(expected)]!r}"
                )
            messages.append(f"{source}={sorted(actual)} matches profile {kind.upper()}={sorted(expected)}")
    for token_id in {bos} | eos:
        if token_id not in special:
            raise InvalidTokens(f"profile special-token ID {token_id} ({by_id.get(token_id)!r}) is not declared special")
    return messages


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--write-config", type=Path,
                        help="atomically write the exact validated profile snapshot as model_config")
    args = parser.parse_args(argv)
    profile = args.profile.expanduser().resolve()
    tok_dir = args.tokenizer_dir.expanduser().resolve()
    try:
        source = profile.read_text(encoding="utf-8")
        messages = validate(source, tok_dir)
        if args.write_config:
            output = args.write_config.expanduser()
            generated = ("# GENERATED by refresh_build.sh -- DO NOT EDIT, DO NOT COMMIT.\n"
                         f"# Source of truth: {args.profile}\n" + source)
            # Only publish after validation; preserve previous config on every rejection.
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                             prefix=".model_config-", delete=False) as tmp:
                temporary = Path(tmp.name)
                try:
                    tmp.write(generated)
                    tmp.flush()
                    os.replace(temporary, output)
                finally:
                    temporary.unlink(missing_ok=True)
        print(f"profile: {profile}\ntokenizer: {tok_dir}")
        for message in messages:
            print("  " + message)
        print("PROFILE SPECIAL TOKENS OK")
        return 0
    except (OSError, InvalidTokens) as exc:
        print(f"FATAL special-token check: profile={profile}, tokenizer={tok_dir}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
