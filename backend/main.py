from __future__ import annotations

import shutil
import tempfile
import threading
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.asr_service import asr_service
from backend.audio_utils import (
    analyze_audio,
    load_wav_float32,
    merge_multilingual_reel,
    normalize_reference_audio,
    save_generated_audio,
)
from backend.config import (
    ACCESS_TOKEN,
    COSYVOICE_INSTRUCT_PRESETS,
    DEFAULT_SELECTED_LANGS,
    FRONTEND_DIR,
    LANGUAGE_BY_CODE,
    LANGUAGES,
    MAX_UPLOAD_MB,
    MAX_VIDEO_MB,
    OUTPUTS_DIR,
    PRESET_SCENARIOS,
    QWEN_BATCH_SIZE,
    QWEN_SAMPLE_RATE,
    QWEN_VARIANTS,
    VIDEOS_DIR,
    VOICES_DIR,
)
from backend.store import store, utc_now_iso
from backend.text_chunker import split_text_for_tts, stitch_chunk_audios
from backend.translator import translate_scripts, translate_segments_for_dubbing
from backend.tts_engine import engine_manager
from backend.video_dubbing import (
    extract_best_voice_reference,
    extract_video_audio,
    fit_segment_tempo,
    has_demucs,
    mix_and_mux_dubbed_video,
    probe_video,
    separate_vocals_and_bgm,
    write_srt,
)
from backend.watermark import detect_watermark, embed_watermark

ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".aac", ".mp4", ".opus"}
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
DEMO_VOICE_ID = "demo_official"
DEMO_WAV_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
DEMO_REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."


def _decorate_voice(v: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(v)
    out["audio_url"] = f"/media/voices/{v['id']}/clone_ref.wav"
    return out


def _decorate_job(job: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(job)
    tracks = []
    for t in job.get("tracks", []):
        td = dict(t)
        code = td["lang_code"]
        if td.get("status") == "completed":
            td["wav_url"] = f"/media/outputs/{job['id']}/{code}.wav"
            td["mp3_url"] = f"/media/outputs/{job['id']}/{code}.mp3"
        if td.get("ready_chunks"):
            td["chunk_urls"] = [f"/media/outputs/{job['id']}/chunks/{code}/chunk_{i:03d}.wav" for i in td["ready_chunks"]]
        tracks.append(td)
    out["tracks"] = tracks
    if out.get("reel"):
        reel = dict(out["reel"])
        reel["mp3_url"] = f"/media/outputs/{job['id']}/multilingual_reel.mp3"
        out["reel"] = reel
    out["zip_url"] = f"/api/jobs/{job['id']}/zip"
    return out


def _decorate_video(video: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(video)
    vid = video["id"]
    if video.get("source_filename"):
        out["source_video_url"] = f"/media/videos/{vid}/{video['source_filename']}"
    if (VIDEOS_DIR / vid / "auto_ref.wav").exists():
        out["auto_ref_url"] = f"/media/videos/{vid}/auto_ref.wav"
    outputs = []
    for item in video.get("outputs", []):
        od = dict(item)
        code = od["lang_code"]
        if od.get("status") == "completed":
            od["mp4_url"] = f"/media/videos/{vid}/dub_{code}.mp4"
            od["mp3_url"] = f"/media/videos/{vid}/dub_{code}.mp3"
            od["srt_url"] = f"/media/videos/{vid}/sub_{code}.srt"
            od["bilingual_srt_url"] = f"/media/videos/{vid}/sub_{code}_bilingual.srt"
        outputs.append(od)
    out["outputs"] = outputs
    out["zip_url"] = f"/api/videos/{vid}/zip"
    return out


def ensure_demo_voice_registered() -> None:
    demo_dir = VOICES_DIR / DEMO_VOICE_ID
    clone_wav = demo_dir / "clone_ref.wav"
    raw_wav = demo_dir / "raw_clone.wav"
    demo_dir.mkdir(parents=True, exist_ok=True)
    if not clone_wav.exists() and not raw_wav.exists():
        try:
            urllib.request.urlretrieve(DEMO_WAV_URL, str(raw_wav))
        except Exception:
            return
    if not clone_wav.exists() and raw_wav.exists():
        try:
            normalize_reference_audio(raw_wav, demo_dir / "source_24k.wav", clone_wav)
        except Exception:
            return
    if clone_wav.exists() and not store.get_voice(DEMO_VOICE_ID):
        metrics = analyze_audio(clone_wav)
        store.upsert_voice(
            {
                "id": DEMO_VOICE_ID,
                "name": "官方示范音色（体验用）",
                "is_demo": True,
                "source_filename": "clone.wav",
                "ref_text": DEMO_REF_TEXT,
                "detected_language": "en",
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
                **metrics,
            }
        )


def _idle_watchdog_loop(stop_event: threading.Event) -> None:
    while not stop_event.wait(30.0):
        try:
            idle_mins = int(store.get_settings(masked=False).get("idle_unload_minutes") or 0)
            if idle_mins <= 0:
                continue
            if engine_manager.has_any_loaded() or asr_service.status["loaded"]:
                if (time.time() - engine_manager.last_activity_ts) >= idle_mins * 60:
                    engine_manager.unload_all()
                    asr_service.unload()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_demo_voice_registered()
    saved_variant = str(store.get_settings(masked=False).get("qwen_variant") or "1.7b")
    engine_manager.qwen3.set_variant(saved_variant)
    stop_ev = threading.Event()
    wd = threading.Thread(target=_idle_watchdog_loop, args=(stop_ev,), daemon=True)
    wd.start()
    yield
    stop_ev.set()


app = FastAPI(title="VoxPoly Studio V2 Pro", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def optional_access_token_guard(request: Request, call_next):
    if ACCESS_TOKEN and request.url.path.startswith("/api/") and request.url.path != "/api/health":
        supplied = (
            request.headers.get("x-access-token")
            or request.query_params.get("token")
            or request.cookies.get("voxpoly_token")
            or ""
        ).strip()
        if supplied != ACCESS_TOKEN:
            return JSONResponse(status_code=401, content={"detail": "访问受口令保护，请输入正确的访问口令。"})
    return await call_next(request)


FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media/voices", StaticFiles(directory=str(VOICES_DIR)), name="media_voices")
app.mount("/media/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="media_outputs")
app.mount("/media/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="media_videos")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static_frontend")


# ---------- Models ----------
class SettingsPatch(BaseModel):
    translation_provider: Optional[str] = None
    llm_api_base: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    clear_api_key: Optional[bool] = False
    qwen_variant: Optional[str] = Field(default=None, pattern=r"^(1\.7b|0\.6b)$")
    enable_watermark: Optional[bool] = None
    enable_batch_accel: Optional[bool] = None
    idle_unload_minutes: Optional[int] = Field(default=None, ge=0, le=120)


class PreloadRequest(BaseModel):
    engine_id: str = Field(default="qwen3", pattern="^(qwen3|chatterbox|cosyvoice3)$")


class VoicePatch(BaseModel):
    name: Optional[str] = None
    ref_text: Optional[str] = None


class TranslateRequest(BaseModel):
    source_text: str = Field(..., min_length=1, max_length=8000)
    target_langs: List[str] = Field(..., min_length=1)
    source_lang: Optional[str] = None


class GenerateJobRequest(BaseModel):
    voice_id: str
    scripts: Dict[str, str]
    engine_id: str = Field(default="qwen3", pattern="^(qwen3|chatterbox|cosyvoice3)$")
    mode: str = Field(default="icl", pattern="^(icl|xvector)$")
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.5)
    cfg_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    instruct: str = Field(default="", max_length=300)
    source_text: Optional[str] = ""
    preset_id: Optional[str] = None
    auto_reel: bool = True


class RegenerateTrackRequest(BaseModel):
    text: Optional[str] = None
    mode: Optional[str] = Field(default=None, pattern="^(icl|xvector)$")
    exaggeration: Optional[float] = Field(default=None, ge=0.0, le=1.5)
    cfg_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    instruct: Optional[str] = None


class ReelRequest(BaseModel):
    lang_codes: Optional[List[str]] = None
    gap_sec: float = Field(default=0.55, ge=0.1, le=2.0)


class VideoTranslateRequest(BaseModel):
    target_langs: List[str] = Field(..., min_length=1)
    segments: Optional[List[Dict[str, Any]]] = None


class VideoDubRequest(BaseModel):
    target_langs: List[str] = Field(..., min_length=1)
    translations: Dict[str, List[str]]
    engine_id: str = Field(default="qwen3", pattern="^(qwen3|chatterbox|cosyvoice3)$")
    voice_id: str = Field(default="auto_video")  # "auto_video" clones speaker from video itself!
    mode: str = Field(default="icl", pattern="^(icl|xvector)$")
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.5)
    cfg_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    instruct: str = Field(default="", max_length=300)
    burn_subtitles: bool = False
    bgm_volume: float = Field(default=0.85, ge=0.0, le=1.2)


# ---------- Core Synthesis Helper (Long-form chunking + Qwen3 batching + chunk streaming + watermark) ----------
def _synthesize_text_smart(
    eng: Any,
    engine_id: str,
    voice_id: str,
    ref_wav_path: Path,
    ref_text: str,
    mode: str,
    text: str,
    lang_meta: Dict[str, Any],
    exaggeration: float,
    cfg_weight: float,
    instruct: str,
    chunk_dir: Optional[Path] = None,
    on_chunk_ready: Optional[Any] = None,
    enable_batch: bool = True,
    apply_watermark: bool = True,
) -> Tuple[np.ndarray, int, float, float, int]:
    lang_code = str(lang_meta["code"])
    chunks = split_text_for_tts(text, lang_code)
    if not chunks:
        raise ValueError(f"{lang_meta['label_zh']} 合成文本不能为空。")

    if chunk_dir is not None:
        chunk_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    chunk_audios: List[Tuple[np.ndarray, int, float]] = []

    if engine_id == "qwen3" and len(chunks) > 1 and enable_batch:
        def _save_chunk(c_idx: int, arr: np.ndarray, c_sr: int) -> None:
            if chunk_dir is not None:
                sf.write(str(chunk_dir / f"chunk_{c_idx:03d}.wav"), arr.astype(np.float32), c_sr, subtype="PCM_16")
            if on_chunk_ready:
                on_chunk_ready(c_idx, len(chunks))

        wavs, sr, _, _ = eng.synthesize_batch(
            voice_id=voice_id,
            ref_wav_path=ref_wav_path,
            ref_text=ref_text,
            mode=mode,
            texts=[str(c["text"]) for c in chunks],
            lang_meta=lang_meta,
            batch_size=QWEN_BATCH_SIZE,
            on_chunk_done=_save_chunk,
        )
        for c, w in zip(chunks, wavs):
            chunk_audios.append((w, sr, float(c["pause_after_sec"])))
        final_wav = stitch_chunk_audios(chunk_audios, target_sr=sr)
    elif len(chunks) == 1:
        w, sr, _, _ = eng.synthesize(
            voice_id=voice_id,
            ref_wav_path=ref_wav_path,
            ref_text=ref_text,
            mode=mode,
            text=str(chunks[0]["text"]),
            lang_meta=lang_meta,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            instruct=instruct,
        )
        if chunk_dir is not None:
            sf.write(str(chunk_dir / "chunk_000.wav"), w.astype(np.float32), sr, subtype="PCM_16")
        if on_chunk_ready:
            on_chunk_ready(0, 1)
        final_wav = w
    else:
        sr = QWEN_SAMPLE_RATE
        for c_idx, c in enumerate(chunks):
            w, sr, _, _ = eng.synthesize(
                voice_id=voice_id,
                ref_wav_path=ref_wav_path,
                ref_text=ref_text,
                mode=mode,
                text=str(c["text"]),
                lang_meta=lang_meta,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                instruct=instruct,
            )
            if chunk_dir is not None:
                sf.write(str(chunk_dir / f"chunk_{c_idx:03d}.wav"), w.astype(np.float32), sr, subtype="PCM_16")
            if on_chunk_ready:
                on_chunk_ready(c_idx, len(chunks))
            chunk_audios.append((w, sr, float(c["pause_after_sec"])))
        final_wav = stitch_chunk_audios(chunk_audios, target_sr=sr)

    if apply_watermark:
        final_wav = embed_watermark(final_wav, sample_rate=sr)

    elapsed = round(time.perf_counter() - t0, 2)
    dur = float(final_wav.size / sr) if sr > 0 else 0.0
    rtf = round(elapsed / dur, 2) if dur > 0.05 else 0.0
    engine_manager.touch()
    return final_wav, sr, elapsed, rtf, len(chunks)


# ---------- Audio Generation Background Worker ----------
def _run_generation_job(job_id: str) -> None:
    job = store.get_job(job_id)
    if not job:
        return
    voice = store.get_voice(job["voice_id"])
    if not voice:
        job["status"] = "failed"
        job["error"] = "所选声纹不存在或已被删除。"
        job["updated_at"] = utc_now_iso()
        store.save_job(job)
        return

    settings = store.get_settings(masked=False)
    apply_wm = bool(settings.get("enable_watermark", True))
    enable_batch = bool(settings.get("enable_batch_accel", True))
    job["watermarked"] = apply_wm

    engine_id = job.get("engine_id", "qwen3")
    ref_wav_path = VOICES_DIR / voice["id"] / "clone_ref.wav"
    ref_text = str(voice.get("ref_text") or "")
    mode = job.get("mode", "icl")
    effective_mode = (
        engine_id
        if engine_id in ("chatterbox", "cosyvoice3")
        else ("xvector" if (mode == "xvector" or not ref_text.strip()) else "icl")
    )
    job["effective_mode"] = effective_mode

    if engine_manager.get_engine(engine_id).get_status()["state"] != "ready":
        job["status"] = "loading_model"
        job["updated_at"] = utc_now_iso()
        store.save_job(job)

    try:
        eng = engine_manager.prepare_engine(engine_id)
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = f"语音引擎加载失败: {exc}"
        job["updated_at"] = utc_now_iso()
        store.save_job(job)
        return

    job["status"] = "running"
    job["updated_at"] = utc_now_iso()
    store.save_job(job)

    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0
    for track in job["tracks"]:
        lang_code = track["lang_code"]
        lang_meta = LANGUAGE_BY_CODE[lang_code]
        track["status"] = "generating"
        track["error"] = None
        track["ready_chunks"] = []
        track["total_chunks"] = len(split_text_for_tts(track["text"], lang_code))
        job["current_lang"] = lang_code
        job["updated_at"] = utc_now_iso()
        store.save_job(job)

        def _on_chunk(c_idx: int, total_c: int, _track=track) -> None:
            if c_idx not in _track["ready_chunks"]:
                _track["ready_chunks"].append(c_idx)
            _track["total_chunks"] = total_c
            job["updated_at"] = utc_now_iso()
            store.save_job(job)

        try:
            wav_np, sr, elapsed, rtf, n_chunks = _synthesize_text_smart(
                eng=eng,
                engine_id=engine_id,
                voice_id=voice["id"],
                ref_wav_path=ref_wav_path,
                ref_text=ref_text,
                mode=effective_mode,
                text=track["text"],
                lang_meta=lang_meta,
                exaggeration=float(job.get("exaggeration", 0.5)),
                cfg_weight=float(job.get("cfg_weight", 0.35)),
                instruct=str(job.get("instruct") or ""),
                chunk_dir=job_dir / "chunks" / lang_code,
                on_chunk_ready=_on_chunk,
                enable_batch=enable_batch,
                apply_watermark=apply_wm,
            )
            metrics = save_generated_audio(wav_np, sr, job_dir / f"{lang_code}.wav", job_dir / f"{lang_code}.mp3")
            track.update(
                {
                    "status": "completed",
                    "elapsed_sec": elapsed,
                    "rtf": rtf,
                    "total_chunks": n_chunks,
                    "watermarked": apply_wm,
                    **metrics,
                }
            )
            succeeded += 1
        except Exception as exc:
            track["status"] = "failed"
            track["error"] = str(exc)
            failed += 1

        job["completed_count"] = succeeded
        job["failed_count"] = failed
        job["updated_at"] = utc_now_iso()
        store.save_job(job)

    job["current_lang"] = None
    if succeeded >= 2 and job.get("auto_reel", True):
        try:
            wav_paths = [
                job_dir / f"{t['lang_code']}.wav"
                for t in job["tracks"]
                if t.get("status") == "completed" and (job_dir / f"{t['lang_code']}.wav").exists()
            ]
            if len(wav_paths) >= 2:
                reel_metrics = merge_multilingual_reel(wav_paths, job_dir / "multilingual_reel.mp3")
                job["reel"] = {
                    "lang_codes": [t["lang_code"] for t in job["tracks"] if t.get("status") == "completed"],
                    "created_at": utc_now_iso(),
                    **reel_metrics,
                }
        except Exception as exc:
            job["reel_error"] = str(exc)

    if succeeded == len(job["tracks"]):
        job["status"] = "completed"
    elif succeeded > 0:
        job["status"] = "partial"
    else:
        job["status"] = "failed"
        job["error"] = "所有语种均生成失败，请检查引擎状态与文本内容。"

    job["finished_at"] = utc_now_iso()
    job["updated_at"] = utc_now_iso()
    store.save_job(job)


def _run_single_track_regen(
    job_id: str,
    lang_code: str,
    new_text: str,
    mode: str,
    exaggeration: float,
    cfg_weight: float,
    instruct: str,
) -> None:
    job = store.get_job(job_id)
    if not job:
        return
    voice = store.get_voice(job["voice_id"])
    if not voice:
        return
    settings = store.get_settings(masked=False)
    apply_wm = bool(settings.get("enable_watermark", True))
    enable_batch = bool(settings.get("enable_batch_accel", True))
    engine_id = job.get("engine_id", "qwen3")
    ref_wav_path = VOICES_DIR / voice["id"] / "clone_ref.wav"
    ref_text = str(voice.get("ref_text") or "")
    effective_mode = (
        engine_id
        if engine_id in ("chatterbox", "cosyvoice3")
        else ("xvector" if (mode == "xvector" or not ref_text.strip()) else "icl")
    )
    lang_meta = LANGUAGE_BY_CODE[lang_code]
    job_dir = OUTPUTS_DIR / job_id

    try:
        eng = engine_manager.prepare_engine(engine_id)
        wav_np, sr, elapsed, rtf, n_chunks = _synthesize_text_smart(
            eng=eng,
            engine_id=engine_id,
            voice_id=voice["id"],
            ref_wav_path=ref_wav_path,
            ref_text=ref_text,
            mode=effective_mode,
            text=new_text,
            lang_meta=lang_meta,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            instruct=instruct,
            chunk_dir=job_dir / "chunks" / lang_code,
            enable_batch=enable_batch,
            apply_watermark=apply_wm,
        )
        metrics = save_generated_audio(wav_np, sr, job_dir / f"{lang_code}.wav", job_dir / f"{lang_code}.mp3")
        for t in job["tracks"]:
            if t["lang_code"] == lang_code:
                t.update(
                    {
                        "text": new_text,
                        "status": "completed",
                        "error": None,
                        "elapsed_sec": elapsed,
                        "rtf": rtf,
                        "total_chunks": n_chunks,
                        "watermarked": apply_wm,
                        **metrics,
                    }
                )
                break
        if job.get("reel"):
            wav_paths = [
                job_dir / f"{t['lang_code']}.wav"
                for t in job["tracks"]
                if t.get("status") == "completed" and (job_dir / f"{t['lang_code']}.wav").exists()
            ]
            if len(wav_paths) >= 2:
                reel_metrics = merge_multilingual_reel(wav_paths, job_dir / "multilingual_reel.mp3")
                job["reel"] = {
                    "lang_codes": [t["lang_code"] for t in job["tracks"] if t.get("status") == "completed"],
                    "created_at": utc_now_iso(),
                    **reel_metrics,
                }
        job["completed_count"] = sum(1 for t in job["tracks"] if t.get("status") == "completed")
        job["failed_count"] = sum(1 for t in job["tracks"] if t.get("status") == "failed")
        job["status"] = "completed" if job["failed_count"] == 0 else "partial"
    except Exception as exc:
        for t in job["tracks"]:
            if t["lang_code"] == lang_code:
                t["status"] = "failed"
                t["error"] = str(exc)
                break
    job["updated_at"] = utc_now_iso()
    store.save_job(job)


# ---------- Video Dubbing Background Worker ----------
def _run_video_dubbing_job(video_id: str, req_data: Dict[str, Any]) -> None:
    video = store.get_video(video_id)
    if not video:
        return
    vdir = VIDEOS_DIR / video_id
    src_video = vdir / video["source_filename"]
    bgm_wav = vdir / "bgm_24k.wav"
    segments = video.get("segments", [])
    settings = store.get_settings(masked=False)
    apply_wm = bool(settings.get("enable_watermark", True))

    voice_id = str(req_data.get("voice_id") or "auto_video")
    if voice_id == "auto_video":
        ref_wav = vdir / "auto_ref.wav"
        ref_text = str(video.get("auto_ref_text") or "")
        effective_voice_id = f"vid_{video_id}"
    else:
        v_obj = store.get_voice(voice_id)
        if not v_obj:
            video["dub_status"] = "failed"
            video["dub_error"] = "所选声纹不存在。"
            store.save_video(video)
            return
        ref_wav = VOICES_DIR / voice_id / "clone_ref.wav"
        ref_text = str(v_obj.get("ref_text") or "")
        effective_voice_id = voice_id

    engine_id = str(req_data.get("engine_id") or "qwen3")
    mode = str(req_data.get("mode") or "icl")
    effective_mode = (
        engine_id
        if engine_id in ("chatterbox", "cosyvoice3")
        else ("xvector" if (mode == "xvector" or not ref_text.strip()) else "icl")
    )

    video["dub_status"] = "loading_model"
    video["updated_at"] = utc_now_iso()
    store.save_video(video)

    try:
        eng = engine_manager.prepare_engine(engine_id)
    except Exception as exc:
        video["dub_status"] = "failed"
        video["dub_error"] = f"语音引擎加载失败: {exc}"
        video["updated_at"] = utc_now_iso()
        store.save_video(video)
        return

    video["dub_status"] = "running"
    video["updated_at"] = utc_now_iso()
    store.save_video(video)

    succeeded = 0
    for out_item in video.get("outputs", []):
        code = out_item["lang_code"]
        lang_meta = LANGUAGE_BY_CODE[code]
        target_lines = req_data["translations"].get(code, [])
        out_item["status"] = "generating"
        out_item["done_segments"] = 0
        out_item["total_segments"] = len(segments)
        out_item["error"] = None
        video["updated_at"] = utc_now_iso()
        store.save_video(video)

        try:
            t0 = time.perf_counter()
            placed: List[tuple] = []
            for s_idx, seg in enumerate(segments):
                line_txt = (target_lines[s_idx] if s_idx < len(target_lines) else seg["text"]).strip()
                if not line_txt:
                    continue
                wav_np, sr, _, _, _ = _synthesize_text_smart(
                    eng=eng,
                    engine_id=engine_id,
                    voice_id=effective_voice_id,
                    ref_wav_path=ref_wav,
                    ref_text=ref_text,
                    mode=effective_mode,
                    text=line_txt,
                    lang_meta=lang_meta,
                    exaggeration=float(req_data.get("exaggeration", 0.5)),
                    cfg_weight=float(req_data.get("cfg_weight", 0.35)),
                    instruct=str(req_data.get("instruct") or ""),
                    enable_batch=False,
                    apply_watermark=apply_wm,
                )
                next_start = (
                    float(segments[s_idx + 1]["start"])
                    if s_idx + 1 < len(segments)
                    else float(video["duration_sec"])
                )
                slot_budget = max(float(seg["end"]) - float(seg["start"]), next_start - float(seg["start"]) - 0.08)
                fitted = fit_segment_tempo(wav_np, sr, slot_budget, vdir, f"{code}_{s_idx}")
                placed.append((float(seg["start"]), fitted))
                out_item["done_segments"] = s_idx + 1
                video["updated_at"] = utc_now_iso()
                store.save_video(video)

            srt_target = write_srt(segments, target_lines, vdir / f"sub_{code}.srt", bilingual=False)
            srt_bi = write_srt(segments, target_lines, vdir / f"sub_{code}_bilingual.srt", bilingual=True)

            mux_metrics = mix_and_mux_dubbed_video(
                video_path=src_video,
                video_duration_sec=float(video["duration_sec"]),
                bgm_wav_path=bgm_wav,
                bgm_method=str(video.get("bgm_method") or "ffmpeg-ducking"),
                placed_segments=placed,
                out_mp4_path=vdir / f"dub_{code}.mp4",
                out_mp3_path=vdir / f"dub_{code}.mp3",
                subtitle_srt_path=srt_bi,
                burn_subtitles=bool(req_data.get("burn_subtitles", False)),
                bgm_volume=float(req_data.get("bgm_volume", 0.85)),
            )
            out_item.update(
                {
                    "status": "completed",
                    "elapsed_sec": round(time.perf_counter() - t0, 2),
                    "burned_subtitles": bool(req_data.get("burn_subtitles", False)),
                    **mux_metrics,
                }
            )
            succeeded += 1
        except Exception as exc:
            out_item["status"] = "failed"
            out_item["error"] = str(exc)

        video["updated_at"] = utc_now_iso()
        store.save_video(video)

    video["dub_status"] = "completed" if succeeded == len(video["outputs"]) else ("partial" if succeeded > 0 else "failed")
    video["updated_at"] = utc_now_iso()
    store.save_video(video)


# ---------- Routes ----------
@app.get("/")
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="前端页面尚未就绪")
    return FileResponse(str(index_path))


@app.get("/api/health")
def health():
    return {"ok": True, "service": "voxpoly-studio-v2", "requires_token": bool(ACCESS_TOKEN)}


@app.get("/api/system/status")
def system_status():
    return {
        "engine": engine_manager.get_status(),
        "asr": asr_service.status,
        "features": {
            "demucs_available": has_demucs(),
            "access_token_enabled": bool(ACCESS_TOKEN),
            "qwen_batch_size": QWEN_BATCH_SIZE,
            "watermark_supported": True,
        },
        "languages": LANGUAGES,
        "default_selected_langs": DEFAULT_SELECTED_LANGS,
        "presets": PRESET_SCENARIOS,
        "cosyvoice_presets": COSYVOICE_INSTRUCT_PRESETS,
        "qwen_variants": [
            {"id": k, "model_id": v["model_id"], "label": v["label"], "vram_gb": v["vram_gb"]}
            for k, v in QWEN_VARIANTS.items()
        ],
        "settings": store.get_settings(masked=True),
    }


@app.post("/api/system/preload")
def preload_model(background_tasks: BackgroundTasks, payload: Optional[PreloadRequest] = None):
    engine_id = payload.engine_id if payload else "qwen3"
    st = engine_manager.get_engine(engine_id).get_status()
    if st["state"] in ("ready", "loading", "downloading"):
        return {"started": False, "engine": engine_manager.get_status()}
    background_tasks.add_task(engine_manager.prepare_engine, engine_id)
    return {"started": True, "engine": engine_manager.get_status()}


@app.post("/api/system/unload")
def unload_models():
    engine_manager.unload_all()
    asr_service.unload()
    return {"unloaded": True, "engine": engine_manager.get_status(), "asr": asr_service.status}


@app.get("/api/settings")
def get_settings():
    return store.get_settings(masked=True)


@app.post("/api/settings")
def update_settings(payload: SettingsPatch):
    updated = store.update_settings(payload.model_dump(exclude_none=True))
    if payload.qwen_variant:
        engine_manager.qwen3.set_variant(payload.qwen_variant)
    return updated


@app.post("/api/watermark/verify")
async def verify_audio_watermark(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    ext = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    with tempfile.TemporaryDirectory(prefix="voxpoly_wm_") as tmp:
        raw_p = Path(tmp) / f"in{ext}"
        wav_p = Path(tmp) / "in_24k.wav"
        raw_p.write_bytes(raw)
        try:
            from backend.audio_utils import _run_ffmpeg

            _run_ffmpeg(["-i", str(raw_p), "-vn", "-ac", "1", "-ar", str(QWEN_SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav_p)])
            audio, sr = load_wav_float32(wav_p)
            return {"filename": file.filename, **detect_watermark(audio, sr)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"音频解析失败: {exc}") from exc


# ---------- Voices ----------
@app.get("/api/voices")
def list_voices():
    return {"voices": [_decorate_voice(v) for v in store.list_voices()]}


@app.post("/api/voices")
async def create_voice(
    file: UploadFile = File(...),
    name: str = Form(default=""),
    ref_text: str = Form(default=""),
    auto_transcribe: bool = Form(default=True),
):
    filename = file.filename or "recording.webm"
    ext = Path(filename).suffix.lower() or ".wav"
    if ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(status_code=400, detail=f"暂不支持的音频格式 {ext}。")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"文件体积超过 {MAX_UPLOAD_MB}MB 上限。")

    voice_id = f"v_{uuid.uuid4().hex[:10]}"
    voice_dir = VOICES_DIR / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    raw_path = voice_dir / f"raw{ext}"
    raw_path.write_bytes(raw_bytes)

    try:
        metrics = normalize_reference_audio(raw_path, voice_dir / "source_24k.wav", voice_dir / "clone_ref.wav")
    except Exception as exc:
        shutil.rmtree(voice_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    clean_ref = ref_text.strip()
    detected_lang = "unknown"
    asr_note = None
    if not clean_ref and auto_transcribe:
        try:
            asr_res = asr_service.transcribe(voice_dir / "clone_ref.wav")
            clean_ref = str(asr_res.get("text") or "").strip()
            detected_lang = str(asr_res.get("language") or "unknown")
        except Exception as exc:
            asr_note = f"自动转写暂不可用（可手动填写参考文本）: {exc}"

    display_name = name.strip() or Path(filename).stem[:32] or f"我的声音 {voice_id[-4:]}"
    record: Dict[str, Any] = {
        "id": voice_id,
        "name": display_name,
        "is_demo": False,
        "source_filename": filename,
        "ref_text": clean_ref,
        "detected_language": detected_lang,
        "asr_note": asr_note,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        **metrics,
    }
    store.upsert_voice(record)
    return {"voice": _decorate_voice(record)}


@app.post("/api/voices/{voice_id}/transcribe")
def transcribe_voice(voice_id: str):
    voice = store.get_voice(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="未找到该声纹档案。")
    clone_wav = VOICES_DIR / voice_id / "clone_ref.wav"
    if not clone_wav.exists():
        raise HTTPException(status_code=404, detail="参考音频文件缺失。")
    try:
        asr_res = asr_service.transcribe(clone_wav)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"语音转写失败: {exc}") from exc
    updated = store.update_voice_fields(
        voice_id,
        {"ref_text": asr_res["text"], "detected_language": asr_res["language"], "asr_note": None},
    )
    engine_manager.invalidate_voice_cache(voice_id)
    return {"voice": _decorate_voice(updated), "asr": asr_res}


@app.patch("/api/voices/{voice_id}")
def patch_voice(voice_id: str, payload: VoicePatch):
    fields = {k: v.strip() for k, v in payload.model_dump(exclude_none=True).items()}
    if not fields:
        raise HTTPException(status_code=400, detail="没有需要更新的字段。")
    updated = store.update_voice_fields(voice_id, fields)
    if not updated:
        raise HTTPException(status_code=404, detail="未找到该声纹档案。")
    engine_manager.invalidate_voice_cache(voice_id)
    return {"voice": _decorate_voice(updated)}


@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str):
    if not store.delete_voice(voice_id):
        raise HTTPException(status_code=404, detail="未找到该声纹档案。")
    engine_manager.invalidate_voice_cache(voice_id)
    return {"deleted": True, "voice_id": voice_id}


# ---------- Translation & Audio Jobs ----------
@app.post("/api/translate")
def api_translate(payload: TranslateRequest):
    settings = store.get_settings(masked=False)
    try:
        translations, provider, warnings = translate_scripts(
            source_text=payload.source_text,
            target_langs=payload.target_langs,
            settings=settings,
            source_lang=payload.source_lang,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"translations": translations, "provider": provider, "warnings": warnings}


@app.post("/api/jobs/generate")
def create_generation_job(payload: GenerateJobRequest):
    voice = store.get_voice(payload.voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="所选声纹不存在，请先上传或选择一段录音。")

    eng_st = engine_manager.get_engine(payload.engine_id).get_status()
    if not eng_st["installed"]:
        raise HTTPException(status_code=400, detail=f"{eng_st['display_name']} 尚未安装，请在 Colab Cell 2 勾选启用。")

    ordered_tracks = []
    for lang in LANGUAGES:
        code = str(lang["code"])
        if code in payload.scripts and str(payload.scripts[code]).strip():
            if payload.engine_id not in lang["engines"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"当前引擎 {eng_st['display_name']} 不支持 {lang['label_zh']}。",
                )
            txt = str(payload.scripts[code]).strip()
            ordered_tracks.append(
                {
                    "lang_code": code,
                    "lang_label": lang["label_zh"],
                    "native_name": lang["native_name"],
                    "region_tag": lang["region_tag"],
                    "text": txt,
                    "status": "pending",
                    "ready_chunks": [],
                    "total_chunks": len(split_text_for_tts(txt, code)),
                    "error": None,
                }
            )

    if not ordered_tracks:
        raise HTTPException(status_code=400, detail="请至少填写一种语言的待合成文本。")

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    now = utc_now_iso()
    job: Dict[str, Any] = {
        "id": job_id,
        "engine_id": payload.engine_id,
        "engine_name": eng_st["display_name"],
        "voice_id": voice["id"],
        "voice_name": voice["name"],
        "mode": payload.mode,
        "effective_mode": payload.mode,
        "exaggeration": payload.exaggeration,
        "cfg_weight": payload.cfg_weight,
        "instruct": payload.instruct.strip(),
        "source_text": (payload.source_text or "").strip(),
        "preset_id": payload.preset_id,
        "auto_reel": payload.auto_reel,
        "status": "queued",
        "error": None,
        "current_lang": None,
        "total_count": len(ordered_tracks),
        "completed_count": 0,
        "failed_count": 0,
        "tracks": ordered_tracks,
        "reel": None,
        "created_at": now,
        "updated_at": now,
    }
    store.save_job(job)
    threading.Thread(target=_run_generation_job, args=(job_id,), daemon=True).start()
    return {"job": _decorate_job(job)}


@app.get("/api/jobs")
def list_jobs(limit: int = 20):
    return {"jobs": [_decorate_job(j) for j in store.list_jobs(limit=limit)]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="未找到该生成任务。")
    return {"job": _decorate_job(job)}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    if not store.delete_job(job_id):
        raise HTTPException(status_code=404, detail="未找到该生成任务。")
    return {"deleted": True, "job_id": job_id}


@app.post("/api/jobs/{job_id}/regenerate/{lang_code}")
def regenerate_track(job_id: str, lang_code: str, payload: RegenerateTrackRequest):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="未找到该生成任务。")
    if lang_code not in LANGUAGE_BY_CODE:
        raise HTTPException(status_code=400, detail=f"不支持的语种代码: {lang_code}")
    target_track = next((t for t in job["tracks"] if t["lang_code"] == lang_code), None)
    if not target_track:
        raise HTTPException(status_code=404, detail=f"任务中不包含语种 {lang_code}")

    new_text = (payload.text if payload.text is not None else target_track["text"]).strip()
    if not new_text:
        raise HTTPException(status_code=400, detail="待合成文本不能为空。")
    mode = payload.mode or job.get("mode", "icl")
    exaggeration = float(payload.exaggeration if payload.exaggeration is not None else job.get("exaggeration", 0.5))
    cfg_weight = float(payload.cfg_weight if payload.cfg_weight is not None else job.get("cfg_weight", 0.35))
    instruct = payload.instruct if payload.instruct is not None else str(job.get("instruct") or "")

    target_track["text"] = new_text
    target_track["status"] = "generating"
    target_track["error"] = None
    job["status"] = "running"
    job["updated_at"] = utc_now_iso()
    store.save_job(job)

    threading.Thread(
        target=_run_single_track_regen,
        args=(job_id, lang_code, new_text, mode, exaggeration, cfg_weight, instruct),
        daemon=True,
    ).start()
    return {"job": _decorate_job(job)}


@app.post("/api/jobs/{job_id}/reel")
def build_reel(job_id: str, payload: ReelRequest):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="未找到该生成任务。")
    job_dir = OUTPUTS_DIR / job_id
    chosen_codes = payload.lang_codes or [t["lang_code"] for t in job["tracks"] if t.get("status") == "completed"]
    wav_paths = [job_dir / f"{code}.wav" for code in chosen_codes if (job_dir / f"{code}.wav").exists()]
    if not wav_paths:
        raise HTTPException(status_code=400, detail="没有已完成的音轨可用于合成串烧。")
    try:
        reel_metrics = merge_multilingual_reel(wav_paths, job_dir / "multilingual_reel.mp3", gap_sec=payload.gap_sec)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"合成串烧失败: {exc}") from exc
    job["reel"] = {"lang_codes": [p.stem for p in wav_paths], "created_at": utc_now_iso(), **reel_metrics}
    job["updated_at"] = utc_now_iso()
    store.save_job(job)
    return {"job": _decorate_job(job)}


@app.get("/api/jobs/{job_id}/zip")
def download_job_zip(job_id: str):
    try:
        zip_path = store.build_job_zip(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path=str(zip_path), media_type="application/zip", filename=f"VoxPoly_{job_id}.zip")


# ---------- Video Multilingual Dubbing Routes ----------
@app.get("/api/videos")
def list_videos(limit: int = 15):
    return {"videos": [_decorate_video(v) for v in store.list_videos(limit=limit)]}


@app.get("/api/videos/{video_id}")
def get_video(video_id: str):
    video = store.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="未找到该视频译配任务。")
    return {"video": _decorate_video(video)}


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str):
    if not store.delete_video(video_id):
        raise HTTPException(status_code=404, detail="未找到该视频译配任务。")
    return {"deleted": True, "video_id": video_id}


@app.post("/api/videos/analyze")
async def analyze_uploaded_video(file: UploadFile = File(...)):
    filename = file.filename or "input.mp4"
    ext = Path(filename).suffix.lower() or ".mp4"
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"暂不支持的视频格式 {ext}，支持 mp4/mov/mkv/webm。")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="上传视频为空。")
    if len(raw_bytes) > MAX_VIDEO_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"视频超过 {MAX_VIDEO_MB}MB 上限。")

    video_id = f"vid_{uuid.uuid4().hex[:10]}"
    vdir = VIDEOS_DIR / video_id
    vdir.mkdir(parents=True, exist_ok=True)
    src_name = f"source{ext}"
    src_path = vdir / src_name
    src_path.write_bytes(raw_bytes)

    try:
        meta = probe_video(src_path)
        stereo_wav, mono_24k = extract_video_audio(src_path, vdir)
        vocals_wav, bgm_wav, bgm_method = separate_vocals_and_bgm(stereo_wav, mono_24k, vdir)
        asr_res = asr_service.transcribe_segments(vocals_wav)
        segments = asr_res.get("segments") or []
        if not segments:
            raise ValueError("未在视频中检测到清晰人声台词，请换一段包含人声讲解的视频。")
        _, auto_ref_text = extract_best_voice_reference(vocals_wav, segments, vdir / "auto_ref.wav")
    except Exception as exc:
        shutil.rmtree(vdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    now = utc_now_iso()
    record: Dict[str, Any] = {
        "id": video_id,
        "title": Path(filename).stem[:40],
        "original_filename": filename,
        "source_filename": src_name,
        "bgm_method": bgm_method,
        "detected_language": asr_res.get("language", "unknown"),
        "full_transcript": asr_res.get("text", ""),
        "segments": segments,
        "auto_ref_text": auto_ref_text,
        "translations": {},
        "dub_status": "analyzed",
        "dub_error": None,
        "outputs": [],
        "created_at": now,
        "updated_at": now,
        **meta,
    }
    store.save_video(record)
    return {"video": _decorate_video(record)}


@app.post("/api/videos/{video_id}/translate")
def translate_video_segments(video_id: str, payload: VideoTranslateRequest):
    video = store.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="未找到该视频记录。")
    if payload.segments:
        video["segments"] = payload.segments
    segments = video.get("segments", [])
    settings = store.get_settings(masked=False)
    try:
        translations, provider, warnings = translate_segments_for_dubbing(
            segments=segments,
            target_langs=payload.target_langs,
            settings=settings,
            source_lang=video.get("detected_language"),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    video["translations"] = {**(video.get("translations") or {}), **translations}
    video["updated_at"] = utc_now_iso()
    store.save_video(video)
    return {"video": _decorate_video(video), "translations": translations, "provider": provider, "warnings": warnings}


@app.post("/api/videos/{video_id}/dub")
def start_video_dubbing(video_id: str, payload: VideoDubRequest):
    video = store.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="未找到该视频记录。")
    eng_st = engine_manager.get_engine(payload.engine_id).get_status()
    if not eng_st["installed"]:
        raise HTTPException(status_code=400, detail=f"{eng_st['display_name']} 尚未安装。")

    outputs = []
    for code in payload.target_langs:
        lang = LANGUAGE_BY_CODE.get(code)
        if not lang or payload.engine_id not in lang["engines"]:
            raise HTTPException(status_code=400, detail=f"引擎 {eng_st['display_name']} 不支持语种 {code}。")
        outputs.append(
            {
                "lang_code": code,
                "lang_label": lang["label_zh"],
                "native_name": lang["native_name"],
                "region_tag": lang["region_tag"],
                "status": "pending",
                "done_segments": 0,
                "total_segments": len(video.get("segments", [])),
                "error": None,
            }
        )

    video["translations"] = {**(video.get("translations") or {}), **payload.translations}
    video["dub_engine_id"] = payload.engine_id
    video["dub_engine_name"] = eng_st["display_name"]
    video["dub_status"] = "queued"
    video["dub_error"] = None
    video["outputs"] = outputs
    video["updated_at"] = utc_now_iso()
    store.save_video(video)

    threading.Thread(target=_run_video_dubbing_job, args=(video_id, payload.model_dump()), daemon=True).start()
    return {"video": _decorate_video(video)}


@app.get("/api/videos/{video_id}/zip")
def download_video_zip(video_id: str):
    try:
        zip_path = store.build_video_zip(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path=str(zip_path), media_type="application/zip", filename=f"VoxPoly_Video_{video_id}.zip")
