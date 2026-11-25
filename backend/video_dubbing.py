"""End-to-end multilingual video dubbing pipeline (BGM separation + time-aligned dubbing + SRT + MP4 mux)."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from backend.audio_utils import _run_ffmpeg, get_ffmpeg_path, load_wav_float32, save_generated_audio
from backend.config import QWEN_SAMPLE_RATE


def get_ffprobe_path() -> str:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe
    ffmpeg = Path(get_ffmpeg_path())
    cand = ffmpeg.with_name("ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe")
    if cand.exists():
        return str(cand)
    raise RuntimeError("未检测到 ffprobe，请确认 FFmpeg 完整安装。")


def probe_video(video_path: Path) -> Dict[str, Any]:
    cmd = [
        get_ffprobe_path(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"视频信息读取失败: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    fmt = data.get("format", {})
    v_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not v_stream:
        raise ValueError("上传文件中未检测到有效视频轨。")
    if not a_stream:
        raise ValueError("上传视频没有音轨，无法提取语音进行译配。")

    duration = float(fmt.get("duration") or v_stream.get("duration") or 0.0)
    if duration < 1.5:
        raise ValueError("视频时长不足 1.5 秒。")

    fps_raw = str(v_stream.get("r_frame_rate") or "30/1")
    try:
        num, den = fps_raw.split("/")
        fps = round(float(num) / max(float(den), 1.0), 2)
    except Exception:
        fps = 30.0

    return {
        "duration_sec": round(duration, 2),
        "width": int(v_stream.get("width") or 0),
        "height": int(v_stream.get("height") or 0),
        "fps": fps,
        "video_codec": str(v_stream.get("codec_name") or "unknown"),
        "audio_codec": str(a_stream.get("codec_name") or "unknown"),
        "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
    }


def extract_video_audio(video_path: Path, work_dir: Path) -> Tuple[Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    stereo_wav = work_dir / "original_stereo_44k.wav"
    mono_24k = work_dir / "original_mono_24k.wav"
    _run_ffmpeg(["-i", str(video_path), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(stereo_wav)])
    _run_ffmpeg(
        ["-i", str(video_path), "-vn", "-ac", "1", "-ar", str(QWEN_SAMPLE_RATE), "-c:a", "pcm_s16le", str(mono_24k)]
    )
    return stereo_wav, mono_24k


def has_demucs() -> bool:
    return shutil.which("demucs") is not None or importlib.util.find_spec("demucs") is not None


def separate_vocals_and_bgm(stereo_wav: Path, mono_24k: Path, work_dir: Path) -> Tuple[Path, Path, str]:
    """Separate vocals and background music using Demucs htdemucs (with clean FFmpeg fallback)."""
    vocals_24k = work_dir / "vocals_24k.wav"
    bgm_24k = work_dir / "bgm_24k.wav"

    if has_demucs():
        demucs_out = work_dir / "demucs_sep"
        demucs_out.mkdir(parents=True, exist_ok=True)
        try:
            import sys
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            cmd = [
                sys.executable,
                "-m",
                "demucs",
                "--two-stems=vocals",
                "-n",
                "htdemucs",
                "-d",
                device,
                "-o",
                str(demucs_out),
                str(stereo_wav),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
            stem_dir = demucs_out / "htdemucs" / stereo_wav.stem
            v_file = stem_dir / "vocals.wav"
            nv_file = stem_dir / "no_vocals.wav"
            if proc.returncode == 0 and v_file.exists() and nv_file.exists():
                _run_ffmpeg(["-i", str(v_file), "-ac", "1", "-ar", str(QWEN_SAMPLE_RATE), "-c:a", "pcm_s16le", str(vocals_24k)])
                _run_ffmpeg(["-i", str(nv_file), "-ac", "2", "-ar", str(QWEN_SAMPLE_RATE), "-c:a", "pcm_s16le", str(bgm_24k)])
                shutil.rmtree(demucs_out, ignore_errors=True)
                return vocals_24k, bgm_24k, "demucs-htdemucs"
        except Exception:
            shutil.rmtree(demucs_out, ignore_errors=True)

    # Clean fallback when Demucs isn't installed: speech-filtered vocals + original stereo bed for ducking
    _run_ffmpeg(
        [
            "-i",
            str(mono_24k),
            "-af",
            "highpass=f=85,lowpass=f=7800",
            "-ac",
            "1",
            "-ar",
            str(QWEN_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(vocals_24k),
        ]
    )
    _run_ffmpeg(["-i", str(stereo_wav), "-ac", "2", "-ar", str(QWEN_SAMPLE_RATE), "-c:a", "pcm_s16le", str(bgm_24k)])
    return vocals_24k, bgm_24k, "ffmpeg-ducking"


def extract_best_voice_reference(
    vocals_wav: Path,
    segments: List[Dict[str, Any]],
    out_ref_wav: Path,
) -> Tuple[Path, str]:
    """Automatically slice the cleanest 6-14s continuous speaker reference from the video's vocal stem."""
    audio, sr = load_wav_float32(vocals_wav)
    total_dur = audio.size / sr if sr > 0 else 0.0
    if not segments:
        end_s = min(audio.size, int(12.0 * sr))
        sf.write(str(out_ref_wav), audio[:end_s], sr, subtype="PCM_16")
        return out_ref_wav, ""

    best_span = None
    best_score = -1.0
    for i in range(len(segments)):
        acc_text = []
        s0 = float(segments[i]["start"])
        for j in range(i, len(segments)):
            e1 = float(segments[j]["end"])
            dur = e1 - s0
            acc_text.append(str(segments[j]["text"]).strip())
            if dur > 15.0:
                break
            if dur >= 4.0:
                # Prefer 8-12s continuous speech windows with solid RMS
                s_idx = max(0, int(s0 * sr))
                e_idx = min(audio.size, int(e1 * sr))
                seg_a = audio[s_idx:e_idx]
                rms = float(np.sqrt(np.mean(seg_a * seg_a))) if seg_a.size else 0.0
                ideal_bonus = 1.0 - abs(dur - 10.0) / 10.0
                score = rms * 10.0 + ideal_bonus
                if score > best_score:
                    best_score = score
                    best_span = (s0, e1, " ".join(acc_text))

    if best_span is None:
        s0 = max(0.0, float(segments[0]["start"]))
        e1 = min(total_dur, max(float(segments[0]["end"]), s0 + 8.0))
        ref_text = str(segments[0]["text"]).strip()
    else:
        s0, e1, ref_text = best_span

    s_idx = max(0, int(s0 * sr))
    e_idx = min(audio.size, int(e1 * sr))
    clip = audio[s_idx:e_idx]
    if clip.size < int(1.8 * sr):
        clip = audio[: min(audio.size, int(12.0 * sr))]
    sf.write(str(out_ref_wav), clip, sr, subtype="PCM_16")
    return out_ref_wav, ref_text


def _srt_ts(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    h = ms_total // 3600000
    m = (ms_total % 3600000) // 60000
    s = (ms_total % 60000) // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(
    segments: List[Dict[str, Any]],
    target_texts: List[str],
    out_path: Path,
    bilingual: bool = False,
) -> Path:
    lines: List[str] = []
    for idx, seg in enumerate(segments):
        t_txt = target_texts[idx].strip() if idx < len(target_texts) else str(seg["text"]).strip()
        src_txt = str(seg["text"]).strip()
        lines.append(str(idx + 1))
        lines.append(f"{_srt_ts(float(seg['start']))} --> {_srt_ts(float(seg['end']))}")
        if bilingual and src_txt and src_txt != t_txt:
            lines.append(t_txt)
            lines.append(src_txt)
        else:
            lines.append(t_txt)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def fit_segment_tempo(seg_wav: np.ndarray, sr: int, max_slot_sec: float, work_dir: Path, tag: str) -> np.ndarray:
    """Pitch-preserving gentle speed-up (up to 1.28x) if dubbed speech overruns its timeline slot."""
    cur_dur = seg_wav.size / sr if sr > 0 else 0.0
    if max_slot_sec <= 0.3 or cur_dur <= max_slot_sec * 1.04:
        return seg_wav

    ratio = min(1.28, cur_dur / max_slot_sec)
    if ratio <= 1.03:
        return seg_wav

    in_p = work_dir / f"tmp_in_{tag}.wav"
    out_p = work_dir / f"tmp_out_{tag}.wav"
    try:
        sf.write(str(in_p), seg_wav.astype(np.float32), sr, subtype="PCM_16")
        _run_ffmpeg(
            [
                "-i",
                str(in_p),
                "-af",
                f"atempo={ratio:.3f}",
                "-ac",
                "1",
                "-ar",
                str(sr),
                "-c:a",
                "pcm_s16le",
                str(out_p),
            ]
        )
        fitted, _ = load_wav_float32(out_p)
        return fitted
    except Exception:
        return seg_wav
    finally:
        in_p.unlink(missing_ok=True)
        out_p.unlink(missing_ok=True)


def _escape_ffmpeg_sub_path(path: Path) -> str:
    # FFmpeg subtitles filter on Windows/Linux requires forward slashes and escaped colons
    p = str(path.resolve()).replace("\\", "/")
    p = p.replace(":", "\\:")
    p = p.replace("'", "\\'")
    return p


def mix_and_mux_dubbed_video(
    video_path: Path,
    video_duration_sec: float,
    bgm_wav_path: Path,
    bgm_method: str,
    placed_segments: List[Tuple[float, np.ndarray]],
    out_mp4_path: Path,
    out_mp3_path: Path,
    subtitle_srt_path: Optional[Path] = None,
    burn_subtitles: bool = False,
    bgm_volume: float = 0.85,
    sr: int = QWEN_SAMPLE_RATE,
) -> Dict[str, Any]:
    """Place dubbed voice segments onto timeline, duck & mix BGM stereo bed, mux MP4."""
    total_samples = max(int(round(video_duration_sec * sr)), sr)
    voice_canvas = np.zeros(total_samples, dtype=np.float32)
    speech_mask = np.zeros(total_samples, dtype=np.float32)
    fade = int(0.012 * sr)

    cursor_end = 0
    for start_sec, wav in placed_segments:
        arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if arr.size == 0:
            continue
        # Level each dubbed line smoothly around -18 dBFS
        rms = float(np.sqrt(np.mean(arr * arr)))
        if rms > 1e-5:
            peak = float(np.max(np.abs(arr)))
            gain = min(0.13 / rms, 0.90 / max(peak, 1e-5), 2.5)
            arr = (arr * gain).astype(np.float32)
        if arr.size > fade * 2:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            arr = arr.copy()
            arr[:fade] *= ramp
            arr[-fade:] *= ramp[::-1]

        s_idx = max(int(round(start_sec * sr)), cursor_end)
        if s_idx >= total_samples:
            continue
        e_idx = min(total_samples, s_idx + arr.size)
        n = e_idx - s_idx
        voice_canvas[s_idx:e_idx] += arr[:n]
        speech_mask[s_idx:e_idx] = 1.0
        cursor_end = e_idx

    # Load stereo BGM bed and apply smooth ducking envelope during speech
    bgm_raw, bgm_sr = sf.read(str(bgm_wav_path), dtype="float32", always_2d=True)
    if bgm_raw.shape[1] == 1:
        bgm_raw = np.repeat(bgm_raw, 2, axis=1)
    if bgm_sr != sr:
        new_n = int(round(bgm_raw.shape[0] * sr / bgm_sr))
        t_old = np.arange(bgm_raw.shape[0])
        t_new = np.linspace(0, bgm_raw.shape[0] - 1, new_n)
        bgm_raw = np.stack([np.interp(t_new, t_old, bgm_raw[:, ch]) for ch in range(2)], axis=1).astype(np.float32)

    bgm_canvas = np.zeros((total_samples, 2), dtype=np.float32)
    copy_n = min(total_samples, bgm_raw.shape[0])
    bgm_canvas[:copy_n] = bgm_raw[:copy_n]

    # Smooth ducking envelope: Demucs keeps BGM rich (0.72 during speech); fallback ducks original voice deeply (0.14)
    duck_floor = 0.70 if bgm_method == "demucs-htdemucs" else 0.14
    # Simple box smooth over 120ms for natural ducking attack/release
    win = max(1, int(0.12 * sr))
    smooth_mask = np.convolve(speech_mask, np.ones(win, dtype=np.float32) / win, mode="same")
    smooth_mask = np.clip(smooth_mask, 0.0, 1.0)
    bgm_gain = (1.0 - smooth_mask * (1.0 - duck_floor)) * float(bgm_volume)

    mixed_stereo = bgm_canvas * bgm_gain[:, None] + voice_canvas[:, None]
    peak_mix = float(np.max(np.abs(mixed_stereo))) if mixed_stereo.size else 0.0
    if peak_mix > 0.94:
        mixed_stereo = mixed_stereo * (0.91 / peak_mix)

    mixed_wav = out_mp4_path.with_suffix(".mix.wav")
    sf.write(str(mixed_wav), mixed_stereo.astype(np.float32), sr, subtype="PCM_16")

    # Save standalone MP3 dubbing track + waveform metrics
    audio_metrics = save_generated_audio(mixed_stereo.mean(axis=1), sr, out_mp4_path.with_suffix(".mono.wav"), out_mp3_path)
    out_mp4_path.with_suffix(".mono.wav").unlink(missing_ok=True)

    # Mux dubbed audio back with original video
    try:
        if burn_subtitles and subtitle_srt_path and subtitle_srt_path.exists():
            sub_esc = _escape_ffmpeg_sub_path(subtitle_srt_path)
            _run_ffmpeg(
                [
                    "-i",
                    str(video_path),
                    "-i",
                    str(mixed_wav),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-vf",
                    f"subtitles='{sub_esc}':force_style='FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=1.5,MarginV=24'",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "21",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    "-shortest",
                    str(out_mp4_path),
                ]
            )
        else:
            _run_ffmpeg(
                [
                    "-i",
                    str(video_path),
                    "-i",
                    str(mixed_wav),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    "-shortest",
                    str(out_mp4_path),
                ]
            )
    finally:
        mixed_wav.unlink(missing_ok=True)

    return {
        "duration_sec": audio_metrics["duration_sec"],
        "rms_db": audio_metrics["rms_db"],
        "waveform_peaks": audio_metrics["waveform_peaks"],
        "mp4_size_mb": round(out_mp4_path.stat().st_size / (1024 * 1024), 2) if out_mp4_path.exists() else 0.0,
    }
