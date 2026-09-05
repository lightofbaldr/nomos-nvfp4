"""Breeze TTS LAN serve — OpenAI-compatible /v1/audio/speech over the kernel path.

Standalone FastAPI app (sibling to serve.py; its own port, its own artifacts — the text
serve's engine daemon is not involved). Serial by design: one kernel handle set, one lock.

  GET  /health
  GET  /v1/voices                          registered voice names
  POST /v1/audio/speech
       {input, voice?, instruction?, speaker?, cfg_scale?, seed?, max_frames?,
        response_format? (wav)}
       voice = registered name -> clone mode (single lane)
       instruction without voice -> design mode (CFG, default scale 4)
       neither -> plain TTS

Run: BREEZE_PORT=8095 pixi run uvicorn pyhost.breeze_serve:app --host 0.0.0.0 --port 8095
"""
from __future__ import annotations

import io
import struct
import threading
import wave
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from breeze_tts import BreezeTTS, SAMPLE_RATE  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tts = BreezeTTS()
    app.state.lock = threading.Lock()
    print("[breeze-serve] kernel TTS ready", flush=True)
    yield


app = FastAPI(title="nomos-breeze-tts", version="0.1.0", lifespan=lifespan)


class SpeechReq(BaseModel):
    input: str
    model: str | None = None
    voice: str | None = None
    instruction: str | None = None
    speaker: str = "S0"
    cfg_scale: float | None = Field(default=None, gt=0)
    seed: int | None = None
    max_frames: int = Field(default=250, ge=1, le=1500)
    response_format: str = "wav"
    stream: bool = False


def _wav_bytes(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "ready": True, "engine": "breeze-kernel",
            "voices": list(BreezeTTS.list_voices())}


@app.get("/v1/voices")
def voices() -> dict[str, Any]:
    return BreezeTTS.list_voices()


@app.post("/v1/audio/speech")
def speech(req: SpeechReq) -> Response:
    tts: BreezeTTS = app.state.tts
    if req.stream and req.response_format not in ("pcm",):
        raise HTTPException(400, "stream=true requires response_format 'pcm' (s16le mono 24000 Hz)")
    if not req.stream and req.response_format not in ("wav",):
        raise HTTPException(400, "response_format must be 'wav' (or 'pcm' with stream=true)")
    voice_codes = ref_text = None
    if req.voice:
        try:
            voice_codes, ref_text = BreezeTTS.load_voice(req.voice)
        except FileNotFoundError:
            raise HTTPException(404, f"unknown voice '{req.voice}'; see /v1/voices")
    cfg = req.cfg_scale if req.cfg_scale is not None else (4.0 if (req.instruction and not req.voice) else 1.0)
    if req.stream:
        def chunks():
            # The lock spans the whole generation: the kernel handles are serial state.
            with app.state.lock:
                cond, uncond = tts.build_prefixes(req.input, speaker=req.speaker,
                                                  instruction=req.instruction,
                                                  voice_codes=voice_codes, ref_text=ref_text)
                for pcm in tts.generate_stream(cond, uncond, cfg_scale=cfg,
                                               max_frames=req.max_frames, seed=req.seed):
                    yield (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
        return StreamingResponse(chunks(), media_type="audio/pcm",
                                 headers={"X-Sample-Rate": "24000", "X-Channels": "1",
                                          "X-Format": "s16le"})
    with app.state.lock:
        cond, uncond = tts.build_prefixes(req.input, speaker=req.speaker,
                                          instruction=req.instruction,
                                          voice_codes=voice_codes, ref_text=ref_text)
        pcm = tts.generate(cond, uncond, cfg_scale=cfg, max_frames=req.max_frames, seed=req.seed)
    if pcm.size == 0:
        raise HTTPException(500, "generation produced no frames")
    return Response(content=_wav_bytes(pcm), media_type="audio/wav",
                    headers={"X-Frames": str(pcm.size // 1920),
                             "X-Duration-Seconds": f"{pcm.size / SAMPLE_RATE:.2f}"})
