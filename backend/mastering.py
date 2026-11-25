"""Broadcast-grade FFmpeg Studio Mastering DSP presets (EQ, de-ess, compression, LUFS leveling)."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

from backend.audio_utils import _run_ffmpeg, load_wav_float32

MASTERING_PRESETS: List[Dict[str, str]] = [
    {
        "id": "raw",
        "label": "原始干声（无滤波）",
        "target_lufs": "Raw",
        "description": "保留神经声码器原始输出，不做额外 EQ 或动态处理",
        "af": "",
    },
    {
        "id": "podcast",
        "label": "播客温暖人声（-16 LUFS）",
        "target_lufs": "-16 LUFS",
        "description": "低频温暖胸腔补偿 + 齿音柔化 + 播客级压缩与响度标准化",
        "af": (
            "highpass=f=75,"
            "equalizer=f=180:t=q:w=1.2:g=2.2,"
            "equalizer=f=3200:t=q:w=1.5:g=1.6,"
            "equalizer=f=7500:t=q:w=2.0:g=-1.8,"
            "acompressor=threshold=-20dB:ratio=3:attack=12:release=120:makeup=1.5,"
            "loudnorm=I=-16:TP=-1.5:LRA=11"
        ),
    },
    {
        "id": "shorts",
        "label": "短视频高穿透（-14 LUFS）",
        "target_lufs": "-14 LUFS",
        "description": "适配 YouTube / B站 / 抖音响度标准，中高频清晰穿透手机外放",
        "af": (
            "highpass=f=95,"
            "equalizer=f=2800:t=q:w=1.2:g=2.8,"
            "equalizer=f=6000:t=q:w=1.8:g=1.4,"
            "acompressor=threshold=-18dB:ratio=3.8:attack=8:release=90:makeup=2.0,"
            "loudnorm=I=-14:TP=-1.0:LRA=8"
        ),
    },
    {
        "id": "audiobook",
        "label": "有声书柔和耐听（-18 LUFS）",
        "target_lufs": "-18 LUFS",
        "description": "柔和高频防听觉疲劳，动态平滑，适合长时间佩戴耳机收听",
        "af": (
            "highpass=f=70,"
            "equalizer=f=240:t=q:w=1.2:g=1.2,"
            "equalizer=f=6800:t=q:w=1.8:g=-2.4,"
            "acompressor=threshold=-22dB:ratio=2.2:attack=20:release=180:makeup=1.0,"
            "loudnorm=I=-18:TP=-2.0:LRA=12"
        ),
    },
]

PRESET_BY_ID: Dict[str, Dict[str, str]] = {p["id"]: p for p in MASTERING_PRESETS}


def apply_mastering_dsp(audio: np.ndarray, sample_rate: int, preset_id: str = "raw") -> np.ndarray:
    """Apply selected broadcast mastering DSP chain via FFmpeg; falls back cleanly to raw on error."""
    preset = PRESET_BY_ID.get((preset_id or "raw").strip().lower())
    if not preset or not preset["af"] or audio.size < int(sample_rate * 0.25):
        return audio

    with tempfile.TemporaryDirectory(prefix="voxpoly_dsp_") as tmp:
        in_p = Path(tmp) / "in.wav"
        out_p = Path(tmp) / "out.wav"
        sf.write(str(in_p), np.asarray(audio, dtype=np.float32), int(sample_rate), subtype="PCM_16")
        try:
            _run_ffmpeg(
                [
                    "-i",
                    str(in_p),
                    "-af",
                    preset["af"],
                    "-ac",
                    "1",
                    "-ar",
                    str(sample_rate),
                    "-c:a",
                    "pcm_s16le",
                    str(out_p),
                ]
            )
            processed, _ = load_wav_float32(out_p)
            return processed if processed.size > 0 else audio
        except Exception:
            return audio
