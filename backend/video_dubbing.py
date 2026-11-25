"""End-to-end multilingual video dubbing V3 (yt-dlp URL + Demucs + Speaker Diarization + Styled ASS + Lip-Sync)."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from backend.audio_utils import _run_ffmpeg, get_ffmpeg_path, load_wav_float32, save_generated_audio
from backend.config import BASE_DIR, LIPSYNC_BACKEND, LIPSYNC_PYTHON, LIPSYNC_REPO_DIR, MAX_VIDEO_MB, QWEN_SAMPLE_RATE
from backend.net_safety import validate_public_http_url

SUBTITLE_STYLES: List[Dict[str, str]] = [
    {
        "id": "cinema",
        "label": "横屏电影双语风",
        "description": "白色主译文清晰黑描边 + 下方浅灰原文，适合 YouTube / B站横屏",
    },
    {
        "id": "shorts_yellow",
        "label": "竖屏短视频醒目黄字",
        "description": "高对比亮黄粗体 + 厚黑描边，抬高安全区避开抖音/Reels 底部遮挡",
    },
    {
        "id": "minimal_bar",
        "label": "极简黑底字幕栏",
        "description": "半透明深色底条托底，教程/录屏/复杂画面下清晰可读",
    },
]


def get_ffprobe_path() -> str:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe
    ffmpeg = Path(get_ffmpeg_path())
    cand = ffmpeg.with_name("ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe")
    if cand.exists():
        return str(cand)
    raise RuntimeError("未检测到 ffprobe，请确认 FFmpeg 完整安装。")


def has_demucs() -> bool:
    return shutil.which("demucs") is not None or importlib.util.find_spec("demucs") is not None


def has_yt_dlp() -> bool:
    return shutil.which("yt-dlp") is not None or importlib.util.find_spec("yt_dlp") is not None


def has_lipsync() -> bool:
    return bool(LIPSYNC_PYTHON and Path(LIPSYNC_PYTHON).exists() and LIPSYNC_REPO_DIR and Path(LIPSYNC_REPO_DIR).exists())


def download_video_from_url(url: str, out_dir: Path) -> Tuple[Path, str]:
    """Download video from YouTube / Bilibili / X / TikTok via isolated socket-guarded yt-dlp runner."""
    clean_url = validate_public_http_url(url)
    if importlib.util.find_spec("yt_dlp") is None:
        raise RuntimeError("未安装 yt-dlp Python 包，请运行 pip install yt-dlp 后重试链接导入。")

    out_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = MAX_VIDEO_MB * 1024 * 1024
    env = os.environ.copy()
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        env.pop(k, None)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(BASE_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        [sys.executable, "-m", "backend.safe_ytdlp", clean_url, str(out_dir), str(MAX_VIDEO_MB)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(BASE_DIR),
        timeout=240,
    )
    payload = None
    for line in reversed((proc.stdout or "").splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                payload = json.loads(s)
                break
            except Exception:
                pass
    if proc.returncode != 0 or not payload or not payload.get("ok"):
        err = (payload.get("error") if isinstance(payload, dict) else None) or (proc.stderr or proc.stdout)[-400:]
        raise RuntimeError(f"视频下载失败: {err}")
    title = str(payload.get("title") or "在线视频")[:48]

    cand = next((p for p in out_dir.glob("source.*") if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")), None)
    if not cand:
        raise RuntimeError("视频下载完成但未找到有效视频文件。")
    if cand.stat().st_size > max_bytes:
        cand.unlink(missing_ok=True)
        raise ValueError(f"下载视频体积超过 {MAX_VIDEO_MB} MB 上限，已中止处理。")
    if cand.suffix.lower() != ".mp4":
        mp4_path = out_dir / "source.mp4"
        _run_ffmpeg(["-i", str(cand), "-c:v", "copy", "-c:a", "aac", str(mp4_path)])
        cand.unlink(missing_ok=True)
        cand = mp4_path
    return cand, title


def probe_video(video_path: Path) -> Dict[str, Any]:
    cmd = [
        get_ffprobe_path(),
        "-protocol_whitelist",
        "file,pipe",
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
        "width": int(v_stream.get("width") or 1280),
        "height": int(v_stream.get("height") or 720),
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
    _run_ffmpeg(["-i", str(video_path), "-vn", "-ac", "1", "-ar", str(QWEN_SAMPLE_RATE), "-c:a", "pcm_s16le", str(mono_24k)])
    return stereo_wav, mono_24k


def separate_vocals_and_bgm(stereo_wav: Path, mono_24k: Path, work_dir: Path) -> Tuple[Path, Path, str]:
    vocals_24k = work_dir / "vocals_24k.wav"
    bgm_24k = work_dir / "bgm_24k.wav"

    if has_demucs():
        demucs_out = work_dir / "demucs_sep"
        demucs_out.mkdir(parents=True, exist_ok=True)
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-d", device, "-o", str(demucs_out), str(stereo_wav)]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
            stem_dir = demucs_out / "htdemucs" / stereo_wav.stem
            v_file, nv_file = stem_dir / "vocals.wav", stem_dir / "no_vocals.wav"
            if proc.returncode == 0 and v_file.exists() and nv_file.exists():
                _run_ffmpeg(["-i", str(v_file), "-ac", "1", "-ar", str(QWEN_SAMPLE_RATE), "-c:a", "pcm_s16le", str(vocals_24k)])
                _run_ffmpeg(["-i", str(nv_file), "-ac", "2", "-ar", str(QWEN_SAMPLE_RATE), "-c:a", "pcm_s16le", str(bgm_24k)])
                shutil.rmtree(demucs_out, ignore_errors=True)
                return vocals_24k, bgm_24k, "demucs-htdemucs"
        except Exception:
            shutil.rmtree(demucs_out, ignore_errors=True)

    _run_ffmpeg(
        ["-i", str(mono_24k), "-af", "highpass=f=85,lowpass=f=7800", "-ac", "1", "-ar", str(QWEN_SAMPLE_RATE), "-c:a", "pcm_s16le", str(vocals_24k)]
    )
    _run_ffmpeg(["-i", str(stereo_wav), "-ac", "2", "-ar", str(QWEN_SAMPLE_RATE), "-c:a", "pcm_s16le", str(bgm_24k)])
    return vocals_24k, bgm_24k, "ffmpeg-ducking"


def diarize_video_segments(
    vocals_wav: Path,
    segments: List[Dict[str, Any]],
    work_dir: Path,
    max_speakers: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Zero-token lightweight acoustic speaker diarization + per-speaker clean reference extraction."""
    audio, sr = load_wav_float32(vocals_wav)
    if not segments:
        return segments, []

    # Default single speaker if too few segments
    assigned = ["SPEAKER_1"] * len(segments)
    if len(segments) >= 3:
        try:
            import librosa
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import StandardScaler

            feats = []
            valid_idx = []
            for idx, seg in enumerate(segments):
                s = max(0, int(float(seg["start"]) * sr))
                e = min(audio.size, int(float(seg["end"]) * sr))
                clip = audio[s:e]
                if clip.size < int(0.45 * sr):
                    continue
                mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=20)
                centroid = librosa.feature.spectral_centroid(y=clip, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=clip, sr=sr)
                vec = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), centroid.mean(axis=1), rolloff.mean(axis=1)])
                if np.isfinite(vec).all():
                    feats.append(vec)
                    valid_idx.append(idx)

            if len(feats) >= 3:
                X = StandardScaler().fit_transform(np.vstack(feats))
                best_k = 1
                best_sil = 0.25  # Only split into multiple speakers when acoustically distinct
                best_labels = np.zeros(len(feats), dtype=int)
                for k in range(2, min(max_speakers, len(feats) - 1) + 1):
                    labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)
                    if len(set(labels)) < 2:
                        continue
                    sil = float(silhouette_score(X, labels, metric="cosine"))
                    if sil > best_sil:
                        best_sil = sil
                        best_k = k
                        best_labels = labels

                if best_k >= 2:
                    # Re-index speakers in order of first appearance
                    label_map: Dict[int, str] = {}
                    for v_i, lbl in zip(valid_idx, best_labels):
                        if int(lbl) not in label_map:
                            label_map[int(lbl)] = f"SPEAKER_{len(label_map) + 1}"
                        assigned[v_i] = label_map[int(lbl)]
                    # Fill any skipped ultra-short segment with nearest neighbor speaker
                    for idx in range(len(segments)):
                        if idx not in valid_idx and idx > 0:
                            assigned[idx] = assigned[idx - 1]
        except Exception:
            pass

    for seg, spk in zip(segments, assigned):
        seg["speaker"] = spk

    # Extract clean 6-14s reference clip per speaker
    unique_speakers = list(dict.fromkeys(assigned))
    speaker_profiles: List[Dict[str, Any]] = []
    for spk in unique_speakers:
        spk_segs = [s for s in segments if s["speaker"] == spk]
        ref_wav_path = work_dir / f"auto_ref_{spk}.wav"
        _, ref_text = extract_best_voice_reference(vocals_wav, spk_segs, ref_wav_path)
        speaker_profiles.append(
            {
                "speaker": spk,
                "segment_count": len(spk_segs),
                "ref_filename": ref_wav_path.name,
                "ref_text": ref_text,
            }
        )

    # Also maintain default auto_ref.wav pointing to primary speaker for backward compatibility
    if speaker_profiles:
        primary_ref = work_dir / speaker_profiles[0]["ref_filename"]
        if primary_ref.exists():
            shutil.copy2(primary_ref, work_dir / "auto_ref.wav")

    return segments, speaker_profiles


def extract_best_voice_reference(
    vocals_wav: Path,
    segments: List[Dict[str, Any]],
    out_ref_wav: Path,
) -> Tuple[Path, str]:
    audio, sr = load_wav_float32(vocals_wav)
    total_dur = audio.size / sr if sr > 0 else 0.0
    if not segments:
        sf.write(str(out_ref_wav), audio[: min(audio.size, int(12.0 * sr))], sr, subtype="PCM_16")
        return out_ref_wav, ""

    # Gather up to 14s of this speaker's cleanest segments
    collected = []
    texts = []
    acc_dur = 0.0
    for seg in segments:
        s_idx = max(0, int(float(seg["start"]) * sr))
        e_idx = min(audio.size, int(float(seg["end"]) * sr))
        piece = audio[s_idx:e_idx]
        if piece.size < int(0.35 * sr):
            continue
        dur = piece.size / sr
        if acc_dur + dur > 14.5 and acc_dur >= 4.5:
            break
        collected.append(piece)
        texts.append(str(seg["text"]).strip())
        acc_dur += dur
        if acc_dur >= 10.5:
            break

    clip = np.concatenate(collected) if collected else audio[: min(audio.size, int(12.0 * sr))]
    if clip.size < int(1.6 * sr):
        s0 = max(0.0, float(segments[0]["start"]))
        e1 = min(total_dur, s0 + 8.0)
        clip = audio[int(s0 * sr) : int(e1 * sr)]
    sf.write(str(out_ref_wav), clip.astype(np.float32), sr, subtype="PCM_16")
    return out_ref_wav, " ".join(texts).strip()


def _srt_ts(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    h = ms_total // 3600000
    m = (ms_total % 3600000) // 60000
    s = (ms_total % 60000) // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_ts(seconds: float) -> str:
    cs_total = max(0, int(round(seconds * 100)))
    h = cs_total // 360000
    m = (cs_total % 360000) // 6000
    s = (cs_total % 6000) // 100
    cs = cs_total % 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def write_srt(segments: List[Dict[str, Any]], target_texts: List[str], out_path: Path, bilingual: bool = False) -> Path:
    lines: List[str] = []
    for idx, seg in enumerate(segments):
        t_txt = target_texts[idx].strip() if idx < len(target_texts) else str(seg["text"]).strip()
        src_txt = str(seg["text"]).strip()
        lines.append(str(idx + 1))
        lines.append(f"{_srt_ts(float(seg['start']))} --> {_srt_ts(float(seg['end']))}")
        lines.append(t_txt)
        if bilingual and src_txt and src_txt != t_txt:
            lines.append(src_txt)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_styled_ass(
    segments: List[Dict[str, Any]],
    target_texts: List[str],
    out_path: Path,
    style_id: str = "cinema",
    width: int = 1280,
    height: int = 720,
    bilingual: bool = True,
) -> Path:
    """Write broadcast-ready ASS subtitle file adapting automatically to 16:9 or 9:16 video dimensions."""
    play_w = max(width, 480)
    play_h = max(height, 360)
    is_vertical = play_h > play_w
    base_fs = max(24, int(play_h * (0.038 if is_vertical else 0.048)))
    sub_fs = max(18, int(base_fs * 0.72))

    # ASS color format is &HAABBGGRR
    if style_id == "shorts_yellow":
        main_style = f"Style: Main,Noto Sans CJK SC,{int(base_fs * 1.12)},&H0000E5FF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,3.2,1.5,2,36,36,{int(play_h * (0.14 if is_vertical else 0.08))},1"
        sub_style = f"Style: Orig,Noto Sans CJK SC,{sub_fs},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2.0,1.0,2,36,36,{int(play_h * (0.09 if is_vertical else 0.04))},1"
    elif style_id == "minimal_bar":
        main_style = f"Style: Main,Noto Sans CJK SC,{base_fs},&H00FFFFFF,&H000000FF,&H00111111,&HB8000000,-1,0,0,0,100,100,0,0,3,8,0,2,40,40,{int(play_h * 0.075)},1"
        sub_style = f"Style: Orig,Noto Sans CJK SC,{sub_fs},&H00D8E2EC,&H000000FF,&H00111111,&HA8000000,0,0,0,0,100,100,0,0,3,6,0,2,40,40,{int(play_h * 0.03)},1"
    else:  # cinema
        main_style = f"Style: Main,Noto Sans CJK SC,{base_fs},&H00FFFFFF,&H000000FF,&H00101418,&H80000000,-1,0,0,0,100,100,0,0,1,2.4,1.2,2,44,44,{int(play_h * 0.075)},1"
        sub_style = f"Style: Orig,Noto Sans CJK SC,{sub_fs},&H00D0D8E0,&H000000FF,&H00101418,&H70000000,0,0,0,0,100,100,0,0,1,1.6,0.8,2,44,44,{int(play_h * 0.032)},1"

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {play_w}",
        f"PlayResY: {play_h}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        main_style,
        sub_style,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events = []
    for idx, seg in enumerate(segments):
        t_txt = (target_texts[idx].strip() if idx < len(target_texts) else str(seg["text"]).strip()).replace("\n", "\\N")
        src_txt = str(seg["text"]).strip().replace("\n", "\\N")
        st, et = _ass_ts(float(seg["start"])), _ass_ts(float(seg["end"]))
        events.append(f"Dialogue: 1,{st},{et},Main,,0,0,0,,{t_txt}")
        if bilingual and src_txt and src_txt != t_txt:
            events.append(f"Dialogue: 0,{st},{et},Orig,,0,0,0,,{src_txt}")

    out_path.write_text("\n".join(header + events) + "\n", encoding="utf-8")
    return out_path


def fit_segment_tempo(seg_wav: np.ndarray, sr: int, max_slot_sec: float, work_dir: Path, tag: str) -> np.ndarray:
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
        _run_ffmpeg(["-i", str(in_p), "-af", f"atempo={ratio:.3f}", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(out_p)])
        fitted, _ = load_wav_float32(out_p)
        return fitted
    except Exception:
        return seg_wav
    finally:
        in_p.unlink(missing_ok=True)
        out_p.unlink(missing_ok=True)


def run_optional_lipsync(video_path: Path, mixed_audio_wav: Path, out_mp4: Path) -> bool:
    """Invoke optional isolated Lip-Sync GPU worker (MuseTalk / Wav2Lip) when installed on Colab L4/A100."""
    if not has_lipsync():
        return False
    env = os.environ.copy()
    env["VOXPOLY_LIPSYNC_REPO"] = LIPSYNC_REPO_DIR
    env["VOXPOLY_LIPSYNC_BACKEND"] = LIPSYNC_BACKEND
    env["PYTHONPATH"] = str(BASE_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.Popen(
        [LIPSYNC_PYTHON, "-u", "-m", "backend.lipsync_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(BASE_DIR),
    )
    try:
        assert proc.stdin and proc.stdout
        proc.stdin.write(
            json.dumps({"cmd": "sync", "video_path": str(video_path), "audio_path": str(mixed_audio_wav), "out_mp4_path": str(out_mp4)}) + "\n"
        )
        proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
        proc.stdin.flush()
        for line in proc.stdout:
            s = line.strip()
            if s.startswith("{") and s.endswith("}"):
                resp = json.loads(s)
                if resp.get("ok") and out_mp4.exists():
                    return True
                if not resp.get("ok"):
                    raise RuntimeError(resp.get("error") or "Lip-Sync worker error")
        return out_mp4.exists()
    finally:
        proc.terminate()


def _escape_ffmpeg_sub_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def mix_and_mux_dubbed_video(
    video_path: Path,
    video_duration_sec: float,
    bgm_wav_path: Path,
    bgm_method: str,
    placed_segments: List[Tuple[float, np.ndarray]],
    out_mp4_path: Path,
    out_mp3_path: Path,
    subtitle_ass_path: Optional[Path] = None,
    burn_subtitles: bool = False,
    enable_lipsync: bool = False,
    bgm_volume: float = 0.85,
    sr: int = QWEN_SAMPLE_RATE,
) -> Dict[str, Any]:
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

    duck_floor = 0.70 if bgm_method == "demucs-htdemucs" else 0.14
    win = max(1, int(0.12 * sr))
    smooth_mask = np.clip(np.convolve(speech_mask, np.ones(win, dtype=np.float32) / win, mode="same"), 0.0, 1.0)
    bgm_gain = (1.0 - smooth_mask * (1.0 - duck_floor)) * float(bgm_volume)

    mixed_stereo = bgm_canvas * bgm_gain[:, None] + voice_canvas[:, None]
    peak_mix = float(np.max(np.abs(mixed_stereo))) if mixed_stereo.size else 0.0
    if peak_mix > 0.94:
        mixed_stereo = mixed_stereo * (0.91 / peak_mix)

    mixed_wav = out_mp4_path.with_suffix(".mix.wav")
    sf.write(str(mixed_wav), mixed_stereo.astype(np.float32), sr, subtype="PCM_16")
    audio_metrics = save_generated_audio(mixed_stereo.mean(axis=1), sr, out_mp4_path.with_suffix(".mono.wav"), out_mp3_path)
    out_mp4_path.with_suffix(".mono.wav").unlink(missing_ok=True)

    video_source = video_path
    lipsync_applied = False
    synced_tmp = out_mp4_path.with_suffix(".lipsync.mp4")

    try:
        if enable_lipsync and has_lipsync():
            try:
                if run_optional_lipsync(video_path, mixed_wav, synced_tmp):
                    video_source = synced_tmp
                    lipsync_applied = True
            except Exception:
                lipsync_applied = False

        if burn_subtitles and subtitle_ass_path and subtitle_ass_path.exists():
            sub_esc = _escape_ffmpeg_sub_path(subtitle_ass_path)
            _run_ffmpeg(
                [
                    "-i",
                    str(video_source),
                    "-i",
                    str(mixed_wav),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-vf",
                    f"ass='{sub_esc}'",
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
                    str(video_source),
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
        synced_tmp.unlink(missing_ok=True)

    return {
        "duration_sec": audio_metrics["duration_sec"],
        "rms_db": audio_metrics["rms_db"],
        "waveform_peaks": audio_metrics["waveform_peaks"],
        "lipsync_applied": lipsync_applied,
        "mp4_size_mb": round(out_mp4_path.stat().st_size / (1024 * 1024), 2) if out_mp4_path.exists() else 0.0,
    }
