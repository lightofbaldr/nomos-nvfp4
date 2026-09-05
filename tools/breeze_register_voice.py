"""Register a voice for the Breeze TTS serve: reference WAV -> voices/<name>/{codes.npy, ref.txt}.

Runs in the reference venv (needs qwen-tts + torch): encoding reference audio is the one
step that stays on the HF stack BY DESIGN — it happens once per voice at registration time,
never at serve time. The serve reads only the cached codes.

Usage:
  .venv/bin/python tools/breeze_register_voice.py <name> <ref.wav> "<exact transcript>"
      [--voices-dir ~/nomos_data/breeze-tts-2/voices]
      [--audio-tokenizer ~/nomos_data/breeze-tts-2/hf/audio_tokenizer]

The transcript must be the EXACT words spoken in the reference clip (the clone prompt
conditions on both). Clean, low-noise references clone best. Voice names are registry
slugs: [A-Za-z0-9_-] only (the serve enforces the same rule).
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("ref_wav")
    ap.add_argument("transcript")
    ap.add_argument("--voices-dir", default=os.path.expanduser("~/nomos_data/breeze-tts-2/voices"))
    ap.add_argument("--audio-tokenizer",
                    default=os.path.expanduser("~/nomos_data/breeze-tts-2/hf/audio_tokenizer"))
    a = ap.parse_args()

    if not a.name or not all(c.isalnum() or c in "_-" for c in a.name):
        sys.exit(f"invalid voice name {a.name!r}: use [A-Za-z0-9_-] only")
    if not a.transcript.strip():
        sys.exit("transcript must be the exact words spoken in the reference clip")

    from qwen_tts import Qwen3TTSTokenizer
    tok = Qwen3TTSTokenizer.from_pretrained(a.audio_tokenizer, device_map="cuda")
    encoded = tok.encode(a.ref_wav)
    codes = encoded["audio_codes"] if isinstance(encoded, dict) else encoded.audio_codes
    if isinstance(codes, (list, tuple)):                  # batch-of-1 list from encode()
        codes = codes[0]
    if hasattr(codes, "detach"):
        codes = codes.detach().cpu()
    codes = np.asarray(codes).squeeze()
    if codes.ndim != 2:
        sys.exit(f"unexpected codes shape {codes.shape}")
    if codes.shape[0] == 16 and codes.shape[1] != 16:
        codes = codes.T                                   # -> [T,16], the registry layout
    assert codes.shape[1] == 16, f"expected 16 codebooks, got {codes.shape}"

    d = os.path.join(a.voices_dir, a.name)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "codes.npy"), codes.astype(np.int64))
    with open(os.path.join(d, "ref.txt"), "w") as f:
        f.write(a.transcript.strip() + "\n")
    print(f"registered voice '{a.name}': {codes.shape[0]} frames "
          f"({codes.shape[0] / 12.5:.2f}s) -> {d}")


if __name__ == "__main__":
    main()
