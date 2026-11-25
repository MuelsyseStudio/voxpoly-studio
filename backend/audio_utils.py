from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from backend.config import MAX_REF_AUDIO_SECONDS, MIN_REF_AUDIO_SECONDS, QWEN_SAMPLE_RATE


def get_ffmpeg_path() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未检测到 FFmpeg，请在 Colab 中执行 apt-get install -y ffmpeg。")
    return ffmpeg


def _run_ffmpeg(args: List[str]) -> None:
    cmd = [get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-protocol_whitelist", "file,pipe", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg 处理失败: {proc.stderr.strip() or proc.stdout.strip()}")


def compute_waveform_peaks(audio: np.ndarray, num_bars: int = 120) -> List[float]:
    if audio.size == 0:
        return [0.05] * num_bars
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    mono = np.nan_to_num(mono.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n = mono.size
    if n < num_bars:
        vals = np.abs(mono)
        padded = np.pad(vals, (0, num_bars - n), mode="constant")
        peak_max = float(np.max(padded)) or 1.0
        return [round(max(0.05, min(1.0, float(v / peak_max))), 4) for v in padded]

    edges = np.linspace(0, n, num_bars + 1, dtype=np.int64)
    bars: List[float] = []
    for i in range(num_bars):
        seg = mono[edges[i] : edges[i + 1]]
        if seg.size == 0:
            bars.append(0.05)
        else:
            rms = float(np.sqrt(np.mean(seg * seg)))
            peak = float(np.max(np.abs(seg)))
            bars.append(0.65 * peak + 0.35 * rms)

    max_bar = max(bars) if bars else 1.0
    if max_bar < 1e-6:
        return [0.05] * num_bars
    return [round(max(0.05, min(1.0, v / max_bar)), 4) for v in bars]


def load_wav_float32(wav_path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio, int(sr)


def analyze_audio(wav_path: Path, original_duration_sec: Optional[float] = None) -> Dict[str, object]:
    audio, sr = load_wav_float32(wav_path)
    duration_sec = float(audio.size / sr) if sr > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms_db = 20.0 * math.log10(max(rms, 1e-7))
    peak_db = 20.0 * math.log10(max(peak, 1e-7))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.985)) if audio.size else 0.0

    warnings: List[str] = []
    score = 100

    if duration_sec < 3.0:
        warnings.append("有效人声不足 3 秒，建议录制 5–15 秒自然语句以提升声纹还原度")
        score -= 28
    elif duration_sec < 5.0:
        warnings.append("录音偏短（3–5 秒），5–15 秒通常能获得更稳定的跨语种韵律")
        score -= 10

    if original_duration_sec is not None and original_duration_sec > MAX_REF_AUDIO_SECONDS + 0.5:
        warnings.append(
            f"原音频长 {original_duration_sec:.1f}s，已自动截取前 {duration_sec:.1f}s 清晰人声用于声纹提取"
        )

    if rms_db < -34.0:
        warnings.append("录音音量偏低，建议靠近麦克风并在安静环境录制")
        score -= 22
    elif rms_db < -28.0:
        warnings.append("音量略轻，克隆引擎已自动补偿电平")
        score -= 8

    if clipping_ratio > 0.008:
        warnings.append("检测到轻微爆音/削波，建议稍微远离麦克风")
        score -= 18

    if score >= 88:
        grade, label = "excellent", "声纹素材优秀"
    elif score >= 70:
        grade, label = "good", "适合克隆"
    elif score >= 50:
        grade, label = "fair", "可用（建议优化录音）"
    else:
        grade, label = "poor", "质量偏低"

    return {
        "duration_sec": round(duration_sec, 2),
        "original_duration_sec": round(original_duration_sec if original_duration_sec is not None else duration_sec, 2),
        "sample_rate": sr,
        "rms_db": round(rms_db, 1),
        "peak_db": round(peak_db, 1),
        "clipping_ratio": round(clipping_ratio, 4),
        "quality_score": max(0, min(100, score)),
        "quality_grade": grade,
        "quality_label": label,
        "warnings": warnings,
        "waveform_peaks": compute_waveform_peaks(audio, num_bars=120),
    }


def normalize_reference_audio(src_path: Path, full_wav_path: Path, clone_wav_path: Path) -> Dict[str, object]:
    full_wav_path.parent.mkdir(parents=True, exist_ok=True)
    clone_wav_path.parent.mkdir(parents=True, exist_ok=True)

    _run_ffmpeg(
        [
            "-i",
            str(src_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(QWEN_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(full_wav_path),
        ]
    )
    full_audio, sr = load_wav_float32(full_wav_path)
    original_duration = float(full_audio.size / sr) if sr > 0 else 0.0
    if original_duration < MIN_REF_AUDIO_SECONDS:
        raise ValueError(
            f"上传音频过短（仅 {original_duration:.2f} 秒），请提供至少 {MIN_REF_AUDIO_SECONDS:.1f} 秒的人声录音。"
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        try:
            _run_ffmpeg(
                [
                    "-i",
                    str(full_wav_path),
                    "-af",
                    "highpass=f=65,silenceremove=start_periods=1:start_silence=0.12:start_threshold=-42dB",
                    "-ac",
                    "1",
                    "-ar",
                    str(QWEN_SAMPLE_RATE),
                    "-c:a",
                    "pcm_s16le",
                    str(tmp_path),
                ]
            )
            trimmed, _ = load_wav_float32(tmp_path)
            if trimmed.size < int(MIN_REF_AUDIO_SECONDS * QWEN_SAMPLE_RATE):
                trimmed = full_audio
        except Exception:
            trimmed = full_audio
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    abs_t = np.abs(trimmed)
    peak = float(np.max(abs_t)) if trimmed.size else 0.0
    if peak > 1e-4:
        thresh = max(peak * 0.02, 0.004)
        active_idx = np.where(abs_t > thresh)[0]
        if active_idx.size > 0:
            end_pad = min(trimmed.size, int(active_idx[-1]) + int(0.18 * QWEN_SAMPLE_RATE))
            start_pad = max(0, int(active_idx[0]) - int(0.08 * QWEN_SAMPLE_RATE))
            candidate = trimmed[start_pad:end_pad]
            if candidate.size >= int(MIN_REF_AUDIO_SECONDS * QWEN_SAMPLE_RATE):
                trimmed = candidate

    max_samples = int(MAX_REF_AUDIO_SECONDS * QWEN_SAMPLE_RATE)
    was_trimmed = trimmed.size > max_samples
    if was_trimmed:
        trimmed = trimmed[:max_samples]

    rms = float(np.sqrt(np.mean(trimmed * trimmed))) if trimmed.size else 0.0
    if 1e-5 < rms < 0.06:
        target_rms = 0.10
        gain = min(target_rms / rms, 0.87 / max(float(np.max(np.abs(trimmed))), 1e-5))
        trimmed = np.clip(trimmed * gain, -0.95, 0.95).astype(np.float32)

    sf.write(str(clone_wav_path), trimmed, QWEN_SAMPLE_RATE, subtype="PCM_16")
    metrics = analyze_audio(clone_wav_path, original_duration_sec=original_duration)
    metrics["was_trimmed"] = was_trimmed
    return metrics


def save_generated_audio(
    wav_np: np.ndarray,
    sample_rate: int,
    wav_path: Path,
    mp3_path: Optional[Path] = None,
) -> Dict[str, object]:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(wav_np, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.95:
        audio = audio * (0.92 / peak)

    sf.write(str(wav_path), audio, int(sample_rate), subtype="PCM_16")

    mp3_size = 0
    if mp3_path is not None:
        mp3_path.parent.mkdir(parents=True, exist_ok=True)
        _run_ffmpeg(
            [
                "-i",
                str(wav_path),
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(mp3_path),
            ]
        )
        mp3_size = mp3_path.stat().st_size if mp3_path.exists() else 0

    duration_sec = float(audio.size / sample_rate) if sample_rate > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
    rms_db = 20.0 * math.log10(max(rms, 1e-7))
    peak_db = 20.0 * math.log10(max(float(np.max(np.abs(audio))) if audio.size else 0.0, 1e-7))

    return {
        "duration_sec": round(duration_sec, 2),
        "sample_rate": int(sample_rate),
        "rms_db": round(rms_db, 1),
        "peak_db": round(peak_db, 1),
        "wav_size_bytes": wav_path.stat().st_size,
        "mp3_size_bytes": mp3_size,
        "waveform_peaks": compute_waveform_peaks(audio, num_bars=120),
    }


def merge_multilingual_reel(
    wav_paths: List[Path],
    out_mp3_path: Path,
    gap_sec: float = 0.55,
    target_sr: int = QWEN_SAMPLE_RATE,
) -> Dict[str, object]:
    if not wav_paths:
        raise ValueError("没有可合成的音轨文件。")

    segments: List[np.ndarray] = []
    gap = np.zeros(int(gap_sec * target_sr), dtype=np.float32)
    fade_len = int(0.015 * target_sr)

    for idx, p in enumerate(wav_paths):
        audio, sr = load_wav_float32(p)
        if sr != target_sr and audio.size > 0:
            new_len = int(round(audio.size * target_sr / sr))
            audio = np.interp(
                np.linspace(0, audio.size - 1, new_len),
                np.arange(audio.size),
                audio,
            ).astype(np.float32)
        if audio.size == 0:
            continue

        rms = float(np.sqrt(np.mean(audio * audio)))
        peak = float(np.max(np.abs(audio)))
        if rms > 1e-5:
            desired_gain = 0.125 / rms
            max_safe_gain = 0.89 / max(peak, 1e-5)
            gain = min(desired_gain, max_safe_gain, 3.0)
            audio = (audio * gain).astype(np.float32)

        if audio.size > fade_len * 2:
            ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            audio[:fade_len] *= ramp
            audio[-fade_len:] *= ramp[::-1]

        if idx > 0:
            segments.append(gap)
        segments.append(audio)

    if not segments:
        raise ValueError("所选音轨均为空，无法合成串烧。")

    merged = np.concatenate(segments)
    out_mp3_path.parent.mkdir(parents=True, exist_ok=True)
    wav_tmp = out_mp3_path.with_suffix(".wav")
    metrics = save_generated_audio(merged, target_sr, wav_tmp, out_mp3_path)
    wav_tmp.unlink(missing_ok=True)
    return metrics
