"""Breeze TTS runtime for the LAN serve — 100%-kernel generation path.

Loads the three sha-pinned deploy artifacts (encoder / model / codec .so) plus the raw
bf16 projection + tied audio embedding blobs, and implements:
  - prompt rendering (plain / instruction=design / clone), raw `tokenizers` + manual BOS
    (transformers-5.x cannot load this tokenizer config; parity vs the reference pipeline's
    ids is verified in tools and at module import here against a golden vector)
  - prefix assembly per the measured merge laws (text rows = encoder->proj; audio rows =
    sum of 16 offset tied-embedding rows; audio-EOS row = sum of the 16 codebook-EOS rows)
  - the committed Tier-0 sampled generation loop (c1ba605 semantics: CFG mix
    uncond + s*(cond-uncond); reserved 2048..2050 suppressed; temp .9 / top_k 50; backbone
    repetition penalty 1.1 over unique history after frame 0; EOS class 2051; depth_lane
    re-read per codebook)
  - codec decode to 24 kHz PCM.

Voices live in VOICES_DIR/<name>/{codes.npy [T,16] int64, ref.txt} — codes are produced
offline at registration time (tools/breeze_register_voice.py, HF encode by design).
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

DEPLOY = Path(os.environ.get("BREEZE_DEPLOY", os.path.expanduser("~/nomos_data/breeze-tts-2/deploy")))
BLOBS_MODEL = os.environ.get("BREEZE_MODEL_BLOBS", os.path.expanduser("~/nomos_data/breeze-tts-2/model-blobs"))
BLOBS_ENC = os.environ.get("BREEZE_ENC_BLOBS", os.path.expanduser("~/nomos_data/breeze-tts-2/encoder-blobs"))
BLOBS_CODEC = os.environ.get("BREEZE_CODEC_BLOBS", os.path.expanduser("~/nomos_data/breeze-tts-2/codec-blobs"))
TOKENIZER_JSON = os.environ.get("BREEZE_TOKENIZER", os.path.expanduser("~/nomos_data/breeze-tts-2/hf/tokenizer.json"))
VOICES_DIR = Path(os.environ.get("BREEZE_VOICES", os.path.expanduser("~/nomos_data/breeze-tts-2/voices")))

V = 2051
EOS = 2051          # backbone extra EOS class
CODEBOOK = 2048     # reserved ids 2048..2050 suppressed at sampling
CBS = 16
BOS_ID = 2
SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920

_GOLD_IDS = [2, 262146, 262156, 236776, 16680, 236764, 3582, 8937, 8300, 607, 496, 8434,
             8341, 236761, 262157, 48496, 12203, 13315, 573, 506, 161544, 13818, 236761]


def _bf16(path: str, shape) -> np.ndarray:
    a = np.fromfile(path, dtype=np.uint16).reshape(shape)
    return (a.astype(np.uint32) << 16).view(np.float32)


class BreezeTTS:
    def __init__(self) -> None:
        P = ctypes.c_void_p
        self.enc = ctypes.CDLL(str(DEPLOY / "libnomos_encoder-breeze.so"))
        self.enc.nomos_breeze_encoder_init.argtypes = [ctypes.c_char_p]
        self.enc.nomos_breeze_encoder_init.restype = ctypes.c_int64
        self.enc.nomos_breeze_encoder_run.argtypes = [ctypes.c_int64, P, ctypes.c_int32, P, P]
        self.eh = self.enc.nomos_breeze_encoder_init(BLOBS_ENC.encode() + b"/")
        assert self.eh, "encoder init failed"

        self.mod = ctypes.CDLL(str(DEPLOY / "libnomos_model-breeze.so"))
        self.mod.nomos_breeze_model_init.argtypes = [ctypes.c_char_p]
        self.mod.nomos_breeze_model_init.restype = ctypes.c_int64
        self.mod.nomos_breeze_model_prefill_lane.argtypes = \
            [ctypes.c_int64, ctypes.c_int32, P, ctypes.c_int32, P, P, P]
        self.mod.nomos_breeze_model_depth_lane.argtypes = [ctypes.c_int64, ctypes.c_int32, P, P]
        self.mod.nomos_breeze_model_step_backbone.argtypes = [ctypes.c_int64, ctypes.c_int32, P, P]
        # Paired CFG route (both lanes through one weight pass; 1.86x model). Optional so an
        # older single-lane .so still loads — design mode uses it when present, else falls back.
        self._paired = all(hasattr(self.mod, n) for n in
                           ("nomos_breeze_model_depth_begin2", "nomos_breeze_model_depth_advance2",
                            "nomos_breeze_model_step_backbone2"))
        if self._paired:
            self.mod.nomos_breeze_model_depth_begin2.argtypes = [ctypes.c_int64, P, P]
            self.mod.nomos_breeze_model_depth_advance2.argtypes = [ctypes.c_int64, ctypes.c_int32, P, P]
            self.mod.nomos_breeze_model_step_backbone2.argtypes = [ctypes.c_int64, P, P]
        self.mh = self.mod.nomos_breeze_model_init(BLOBS_MODEL.encode() + b"/")
        assert self.mh, "model init failed"

        self.codec = ctypes.CDLL(str(DEPLOY / "libnomos_codec-qwen3_tts.so"))
        self.codec.nomos_breeze_codec_init.argtypes = [ctypes.c_char_p]
        self.codec.nomos_breeze_codec_init.restype = ctypes.c_int64
        self.codec.nomos_breeze_codec_decode.argtypes = [ctypes.c_int64, P, ctypes.c_int32, P]
        self.ch = self.codec.nomos_breeze_codec_init(BLOBS_CODEC.encode() + b"/")
        assert self.ch, "codec init failed"

        self.tok = Tokenizer.from_file(TOKENIZER_JSON)
        probe = self._ids("[S0]<ins_bos>A calm, clear male voice with a measured delivery."
                          "<ins_eos>Golden capture sentence for the codec gate.")
        assert probe == _GOLD_IDS, "tokenizer parity broken — refuse to serve"

        self.proj = _bf16(f"{BLOBS_MODEL}/text_encoder_proj_weight.bf16", (2048, 1152))
        self.aemb = _bf16(f"{BLOBS_MODEL}/depth_decoder_model_embed_tokens_weight.bf16", (32816, 2048))
        self._audio_eos_row = self.aemb[np.arange(CBS) * V + 0].sum(0)

    # ── prompt/prefix assembly ────────────────────────────────────────────────
    def _ids(self, text: str) -> list[int]:
        return [BOS_ID] + self.tok.encode(text, add_special_tokens=False).ids

    def _encode_text(self, ids: list[int]) -> np.ndarray:
        a = np.ascontiguousarray([ids], dtype=np.int64)
        S = a.shape[1]
        layers = np.zeros((26, S, 1152), np.float32)
        final = np.zeros((S, 1152), np.float32)
        rc = self.enc.nomos_breeze_encoder_run(self.eh, a.ctypes.data, S,
                                               layers.ctypes.data, final.ctypes.data)
        assert rc == 0, "encoder run failed"
        return final @ self.proj.T                     # [S,2048]

    def _audio_rows(self, codes: np.ndarray) -> np.ndarray:
        # codes [T,16] -> [T,2048]: sum of 16 offset tied-embedding rows (measured law)
        off = np.arange(CBS) * V
        return np.stack([self.aemb[off + codes[i]].sum(0) for i in range(codes.shape[0])])

    def build_prefixes(self, text: str, speaker: str = "S0", instruction: str | None = None,
                       voice_codes: np.ndarray | None = None, ref_text: str | None = None):
        """Returns (cond_prefix [S,2048], uncond_prefix or None)."""
        sp = f"[{speaker}]" if not speaker.startswith("[") else speaker
        if voice_codes is not None:
            assert ref_text, "clone mode needs the reference transcript"
            seg_ref = self._encode_text(self._ids(f"{sp}{ref_text}"))
            seg_tgt = self._encode_text(self._ids(text))
            cond = np.concatenate([seg_ref, self._audio_rows(voice_codes),
                                   self._audio_eos_row[None], seg_tgt])
            return cond.astype(np.float32), None       # clone: single lane, cfg 1.0
        if instruction:
            cond = self._encode_text(self._ids(f"{sp}<ins_bos>{instruction}<ins_eos>{text}"))
            unc = self._encode_text(self._ids(f"{sp}{text}"))
            return cond.astype(np.float32), unc.astype(np.float32)
        return self._encode_text(self._ids(f"{sp}{text}")).astype(np.float32), None

    # ── committed Tier-0 sampled loop (c1ba605 semantics) ────────────────────
    @staticmethod
    def _pick(logits, rng, temp=.9, top_k=50, top_p=1.0):
        x = np.asarray(logits, dtype=np.float64).copy()
        x[CODEBOOK:V] = -np.inf
        x /= temp
        if top_k > 0:
            keep = np.argpartition(x, -min(top_k, len(x)))[-min(top_k, len(x)):]
            mask = np.ones(len(x), bool)
            mask[keep] = False
            x[mask] = -np.inf
        order = np.argsort(x)[::-1]
        p = np.exp(x[order] - np.max(x))
        p /= p.sum()
        if top_p < 1:
            cut = np.searchsorted(np.cumsum(p), top_p) + 1
            order = order[:cut]
            p = p[:cut] / p[:cut].sum()
        return int(rng.choice(order, p=p))

    def _iter_frames(self, cond, uncond, cfg_scale, max_frames, seed):
        """Yield each completed frame's 16 codes. Uses the paired CFG route (both lanes in one
        weight pass, 1.86x) when a second lane is present and the .so exports it; otherwise the
        single-lane looped route. Both produce equivalent samples (gated: paired depth 1026/1050
        top1 zero fat-margin vs the single-lane goldens)."""
        def mix(a, b):
            return a if b is None else b + cfg_scale * (a - b)
        prefixes = [cond] + ([uncond] if uncond is not None else [])
        paired = self._paired and len(prefixes) == 2
        lm = []
        for lane, x in enumerate(prefixes):
            x = np.ascontiguousarray(x[None] if x.ndim == 2 else x, np.float32)
            o = np.empty(EOS + 1, np.float32)
            rc = self.mod.nomos_breeze_model_prefill_lane(
                self.mh, lane, x.ctypes.data, x.shape[1], None, None, o.ctypes.data)
            assert rc == 0, f"prefill lane{lane} rc={rc}"
            lm.append(o)
        rng = np.random.default_rng(seed)
        history = []
        for _ in range(max_frames):
            mixed = mix(lm[0], lm[1] if len(lm) > 1 else None).copy()
            for tok in set(history):
                mixed[tok] = mixed[tok] / 1.1 if mixed[tok] > 0 else mixed[tok] * 1.1
            cb0 = self._pick(mixed, rng)
            if cb0 == EOS:
                break
            if paired:
                codes = np.full(CBS, -1, dtype=np.int64)   # progressive fill per the paired ABI
                codes[0] = cb0
                out2 = np.empty((2, V), np.float32)
                for cb in range(1, CBS):
                    if cb == 1:
                        self.mod.nomos_breeze_model_depth_begin2(
                            self.mh, codes.ctypes.data, out2.ctypes.data)
                    else:
                        self.mod.nomos_breeze_model_depth_advance2(
                            self.mh, cb - 1, codes.ctypes.data, out2.ctypes.data)
                    codes[cb] = self._pick(mix(out2[0], out2[1]), rng)
                bb = np.empty((2, EOS + 1), np.float32)
                self.mod.nomos_breeze_model_step_backbone2(self.mh, codes.ctypes.data, bb.ctypes.data)
                lm = [bb[0].copy(), bb[1].copy()]
            else:
                codes = np.zeros(CBS, dtype=np.int64)
                codes[0] = cb0
                for cb in range(1, CBS):
                    ds = []
                    for lane in range(len(prefixes)):
                        d = np.empty((15, V), np.float32)
                        rc = self.mod.nomos_breeze_model_depth_lane(
                            self.mh, lane, codes.ctypes.data, d.ctypes.data)
                        assert rc == 0
                        ds.append(d[cb - 1])
                    codes[cb] = self._pick(mix(ds[0], ds[1] if len(ds) > 1 else None), rng)
                lm = []
                for lane in range(len(prefixes)):
                    o = np.empty(EOS + 1, np.float32)
                    rc = self.mod.nomos_breeze_model_step_backbone(
                        self.mh, lane, codes.ctypes.data, o.ctypes.data)
                    assert rc == 0
                    lm.append(o)
            history.append(cb0)
            yield codes.copy()

    def generate(self, cond: np.ndarray, uncond: np.ndarray | None, cfg_scale: float = 1.0,
                 max_frames: int = 250, seed: int | None = None) -> np.ndarray:
        frames = list(self._iter_frames(cond, uncond, cfg_scale, max_frames, seed))
        if not frames:
            return np.zeros(0, np.float32)
        return self._decode_frames(frames)

    def _decode_frames(self, frames) -> np.ndarray:
        codes = np.ascontiguousarray(np.stack(frames, axis=1)[None], np.int64)  # [1,16,T]
        T = codes.shape[2]
        pcm = np.empty(FRAME_SAMPLES * T, np.float32)
        rc = self.codec.nomos_breeze_codec_decode(self.ch, codes.ctypes.data, T, pcm.ctypes.data)
        assert rc == 0, "codec decode failed"
        return pcm

    def generate_stream(self, cond, uncond, cfg_scale: float = 1.0, max_frames: int = 250,
                        seed: int | None = None, chunk_frames: int = 4):
        """Yields float32 PCM chunks as frames are generated, for interactive latency.

        Semantics: each yield re-decodes the growing code sequence and emits only the newly
        completed tail — so every emitted sample had full LEFT context when it was produced,
        which is correct streaming. It is NOT bit-identical to the non-streaming decode: that
        path sees all frames at once, and the codec is not sample-stable across decode lengths
        (measured 2026-09-04: the reference's own streaming deviates from one-shot by up to
        ~0.11, an edge/receptive-field effect). Streaming output sits within that same envelope.
        Use non-streaming wav mode for the bit-exact waveform; true low-latency bit-exact
        streaming would need a stateful codec ABI (persistent conv state across calls) — a
        future codec-side extension. Re-decode cost is O(blocks²) but trivial beside generation."""
        frames, emitted = [], 0
        for codes in self._iter_frames(cond, uncond, cfg_scale, max_frames, seed):
            frames.append(codes)
            if len(frames) % chunk_frames == 0:
                pcm = self._decode_frames(frames)
                yield pcm[emitted:]
                emitted = pcm.size
            lm = []
            for lane in range(len(prefixes)):
                o = np.empty(EOS + 1, np.float32)
                rc = self.mod.nomos_breeze_model_step_backbone(
                    self.mh, lane, codes.ctypes.data, o.ctypes.data)
                assert rc == 0
                lm.append(o)
        if frames:
            pcm = self._decode_frames(frames)
            if pcm.size > emitted:
                yield pcm[emitted:]

    # ── voices ────────────────────────────────────────────────────────────────
    @staticmethod
    def list_voices() -> dict:
        out = {}
        if VOICES_DIR.is_dir():
            for d in sorted(VOICES_DIR.iterdir()):
                if (d / "codes.npy").exists() and (d / "ref.txt").exists():
                    out[d.name] = {"frames": int(np.load(d / "codes.npy").shape[0])}
        return out

    @staticmethod
    def load_voice(name: str):
        # `name` arrives from the HTTP request — constrain to a bare registry slug so it can
        # never traverse out of VOICES_DIR (security review finding, 2026-09-04).
        if not name or not all(c.isalnum() or c in "_-" for c in name):
            raise FileNotFoundError(f"invalid voice name {name!r}")
        d = (VOICES_DIR / name).resolve()
        if d.parent != VOICES_DIR.resolve():
            raise FileNotFoundError(f"invalid voice name {name!r}")
        codes = np.load(d / "codes.npy").astype(np.int64)   # [T,16]
        ref_text = (d / "ref.txt").read_text().strip()
        return codes, ref_text
