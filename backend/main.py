from __future__ import annotations

import hashlib
import hmac
import os
import posixpath
import re
import secrets
import shutil
import tempfile
import threading
import time
import urllib.request
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette._utils import get_route_path

from backend.asr_service import asr_service
from backend.audio_utils import (
    _run_ffmpeg,
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
    PODCASTS_DIR,
    PRESET_SCENARIOS,
    QWEN_BATCH_SIZE,
    QWEN_SAMPLE_RATE,
    QWEN_VARIANTS,
    VIDEOS_DIR,
    VOICES_DIR,
)
from backend.document_parser import parse_document_bytes, parse_url_article
from backend.glossary import apply_tts_pronunciation, glossary_store
from backend.mastering import MASTERING_PRESETS, apply_mastering_dsp
from backend.net_safety import validate_public_http_url
from backend.podcast_engine import (
    assemble_podcast_audio,
    generate_two_host_podcast_script,
    parse_dialogue_script,
    translate_dialogue_turns,
)
from backend.store import store, utc_now_iso
from backend.text_chunker import split_text_for_tts, stitch_chunk_audios
from backend.translator import detect_script_language, translate_scripts, translate_segments_for_dubbing
from backend.tts_engine import engine_manager
from backend.video_dubbing import (
    SUBTITLE_STYLES,
    diarize_video_segments,
    download_video_from_url,
    extract_video_audio,
    fit_segment_tempo,
    has_demucs,
    has_lipsync,
    has_yt_dlp,
    mix_and_mux_dubbed_video,
    probe_video,
    separate_vocals_and_bgm,
    write_srt,
    write_styled_ass,
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


def _decorate_podcast(pod: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(pod)
    tracks = []
    for t in pod.get("tracks", []):
        td = dict(t)
        code = td["lang_code"]
        if td.get("status") == "completed":
            td["mp3_url"] = f"/media/podcasts/{pod['id']}/podcast_{code}.mp3"
            td["wav_url"] = f"/media/podcasts/{pod['id']}/podcast_{code}.wav"
        tracks.append(td)
    out["tracks"] = tracks
    out["zip_url"] = f"/api/podcasts/{pod['id']}/zip"
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
            od["ass_url"] = f"/media/videos/{vid}/sub_{code}.ass"
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
    threading.Thread(target=_idle_watchdog_loop, args=(stop_ev,), daemon=True).start()
    yield
    stop_ev.set()


app = FastAPI(title="VoxPoly Studio V3 Pro", version="3.0.1", lifespan=lifespan)

_PUBLIC_READ_EXACT = {"/", "/favicon.ico", "/api/health"}
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_SESSION_COOKIE_NAME = "voxpoly_session"
_SESSION_TTL_SEC = 24 * 3600
_MAX_SESSIONS = 256
_AUTH_FAIL_WINDOW_SEC = 60.0
_AUTH_FAIL_MAX = 15
_MAX_RATE_LIMIT_KEYS = 2048

_MEDIA_BUCKETS: Dict[str, Path] = {
    "voices": VOICES_DIR,
    "outputs": OUTPUTS_DIR,
    "podcasts": PODCASTS_DIR,
    "videos": VIDEOS_DIR,
}
_ALLOWED_MEDIA_SERVE_EXTS = {
    ".wav",
    ".mp3",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".m4a",
    ".ogg",
    ".flac",
    ".aac",
    ".opus",
    ".srt",
    ".ass",
}


class _AuthState:
    """In-memory revocable session store + sliding-window failed-auth rate limiter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, Tuple[str, float]] = {}  # sid -> (token_sha256, exp_mono)
        self._fail_hits: Dict[str, Deque[float]] = {}

    @staticmethod
    def _token_digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_session(self, expected_token: str) -> str:
        sid = secrets.token_urlsafe(32)
        now = time.monotonic()
        digest = self._token_digest(expected_token)
        with self._lock:
            expired = [k for k, (_, exp) in self._sessions.items() if exp <= now]
            for k in expired:
                self._sessions.pop(k, None)
            while len(self._sessions) >= _MAX_SESSIONS:
                oldest = next(iter(self._sessions))
                self._sessions.pop(oldest, None)
            self._sessions[sid] = (digest, now + _SESSION_TTL_SEC)
        return sid

    def validate_session(self, sid: str, expected_token: str) -> bool:
        if not sid or not expected_token:
            return False
        now = time.monotonic()
        expected_digest = self._token_digest(expected_token)
        with self._lock:
            entry = self._sessions.get(sid)
            if not entry:
                return False
            stored_digest, exp = entry
            if exp <= now or not hmac.compare_digest(stored_digest, expected_digest):
                self._sessions.pop(sid, None)
                return False
            return True

    def revoke_session(self, sid: str) -> None:
        if not sid:
            return
        with self._lock:
            self._sessions.pop(sid, None)

    def is_rate_limited(self, client_key: str) -> bool:
        now = time.monotonic()
        cutoff = now - _AUTH_FAIL_WINDOW_SEC
        with self._lock:
            dq = self._fail_hits.get(client_key)
            if not dq:
                return False
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if not dq:
                self._fail_hits.pop(client_key, None)
                return False
            return len(dq) >= _AUTH_FAIL_MAX

    def record_failure(self, client_key: str) -> None:
        now = time.monotonic()
        cutoff = now - _AUTH_FAIL_WINDOW_SEC
        with self._lock:
            if client_key not in self._fail_hits and len(self._fail_hits) >= _MAX_RATE_LIMIT_KEYS:
                # Prune stale buckets first, then evict oldest if still at cap
                stale = [k for k, q in self._fail_hits.items() if not q or q[-1] <= cutoff]
                for k in stale:
                    self._fail_hits.pop(k, None)
                while len(self._fail_hits) >= _MAX_RATE_LIMIT_KEYS:
                    self._fail_hits.pop(next(iter(self._fail_hits)), None)
            dq = self._fail_hits.setdefault(client_key, deque())
            while dq and dq[0] <= cutoff:
                dq.popleft()
            dq.append(now)

    def clear_failures(self, client_key: str) -> None:
        with self._lock:
            self._fail_hits.pop(client_key, None)

    def reset_all(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._fail_hits.clear()


_auth_state = _AuthState()


def _active_access_token() -> str:
    return os.getenv("VOXPOLY_ACCESS_TOKEN", ACCESS_TOKEN).strip()


def _normalize_route_path(scope: Dict[str, Any]) -> str:
    """Use the exact same ASGI route path (scope['path'] minus root_path) that Starlette routes on."""
    raw = get_route_path(scope) or "/"
    collapsed = re.sub(r"/+", "/", raw)
    cleaned = posixpath.normpath(collapsed)
    return "/" + cleaned.lstrip("/")


def _peer_host(request: Request) -> str:
    """Return raw TCP peer IP (preserved intact when Uvicorn runs with --no-proxy-headers)."""
    return request.client.host if request.client and request.client.host else "unknown"


def _client_key(request: Request) -> str:
    peer = _peer_host(request)
    # When explicitly deployed behind local Cloudflare Tunnel proxy, separate rate-limit buckets by real client IP
    if peer in ("127.0.0.1", "::1", "localhost") and os.getenv("VOXPOLY_TRUST_CF_IP", "").strip() == "1":
        cf_ip = (request.headers.get("cf-connecting-ip") or "").strip()
        if cf_ip:
            return cf_ip[:64]
    return peer


def _is_https_request(request: Request) -> bool:
    if request.scope.get("scheme") == "https":
        return True
    # Trust X-Forwarded-Proto strictly when the raw TCP peer is local loopback (e.g. cloudflared on 127.0.0.1)
    if _peer_host(request) in ("127.0.0.1", "::1", "localhost"):
        proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
        return proto == "https"
    return False


def _extract_header_token(request: Request) -> str:
    auth_h = request.headers.get("authorization", "")
    bearer = auth_h[7:].strip() if auth_h.lower().startswith("bearer ") else ""
    return (request.headers.get("x-access-token") or bearer or "").strip()


def _tokens_match(supplied: str, expected: str) -> bool:
    if not supplied or not expected:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def _passes_same_origin_check(request: Request, require_custom_header: bool) -> bool:
    sec_fetch = (request.headers.get("sec-fetch-site") or "").strip().lower()
    if sec_fetch and sec_fetch not in ("same-origin", "none"):
        return False
    origin = (request.headers.get("origin") or "").strip()
    if origin:
        host = (request.headers.get("host") or "").strip().lower()
        origin_netloc = urlparse(origin).netloc.lower()
        if not host or origin_netloc != host:
            return False
    if require_custom_header and request.headers.get("x-voxpoly-request") != "1":
        return False
    return True


@app.middleware("http")
async def security_and_access_token_guard(request: Request, call_next):
    norm_path = _normalize_route_path(request.scope)
    method = request.method.upper()
    expected = _active_access_token()
    is_public = False

    if expected:
        is_public = (
            (norm_path in _PUBLIC_READ_EXACT and method in ("GET", "HEAD"))
            or (norm_path.startswith("/static/") and method in ("GET", "HEAD"))
            or (norm_path == "/api/auth/session" and method in ("POST", "DELETE"))
        )

        if not is_public:
            cookie_sid = (request.cookies.get(_SESSION_COOKIE_NAME) or "").strip()
            authed_via_cookie = _auth_state.validate_session(cookie_sid, expected)
            authed_via_header = False

            if not authed_via_cookie:
                ckey = f"hdr:{_client_key(request)}"
                if _auth_state.is_rate_limited(ckey):
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "认证失败次数过多，请 60 秒后重试。"},
                        headers={"Cache-Control": "private, no-store", "Retry-After": "60"},
                    )
                header_tok = _extract_header_token(request)
                authed_via_header = _tokens_match(header_tok, expected)
                if authed_via_header:
                    _auth_state.clear_failures(ckey)
                else:
                    _auth_state.record_failure(ckey)
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "访问受口令保护，请输入正确的访问口令。"},
                        headers={"Cache-Control": "private, no-store"},
                    )

            # Enforce same-origin + custom header CSRF protection on cookie-authenticated mutations
            if authed_via_cookie and method not in _SAFE_METHODS:
                if not _passes_same_origin_check(request, require_custom_header=True):
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "跨源写请求已被 CSRF 防护策略拒绝。"},
                        headers={"Cache-Control": "private, no-store"},
                    )

    response = await call_next(request)
    if norm_path == "/api/auth/session" or (expected and not is_public):
        response.headers["Cache-Control"] = "private, no-store"
    return response


_EXTRA_CORS_ORIGINS = [
    o.strip() for o in os.getenv("VOXPOLY_CORS_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_EXTRA_CORS_ORIGINS,
    allow_origin_regex=r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "HEAD", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Access-Token", "X-VoxPoly-Request"],
)


FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static_frontend")


@app.api_route("/media/{bucket}/{file_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
def serve_media_allowlisted(bucket: str, file_path: str):
    """Serve strictly allowlisted audio/video/subtitle files after canonical filesystem resolution."""
    base_dir = _MEDIA_BUCKETS.get(bucket)
    if base_dir is None or not file_path or "\\" in file_path or "\x00" in file_path:
        raise HTTPException(status_code=404, detail="Not found")

    segments = file_path.split("/")
    for seg in segments:
        if not seg or seg in (".", "..") or seg.startswith(".") or seg.endswith((".", " ", "\t")):
            raise HTTPException(status_code=404, detail="Not found")

    root = base_dir.resolve()
    try:
        target = (root / Path(*segments)).resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        raise HTTPException(status_code=404, detail="Not found")

    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    # Enforce extension allowlist on the canonical resolved filename (closes Windows trailing-dot/space & 8.3 aliases)
    if target.name.startswith(".") or target.suffix.lower() not in _ALLOWED_MEDIA_SERVE_EXTS:
        raise HTTPException(status_code=404, detail="Not found")

    return FileResponse(str(target))


# ---------- Schemas ----------
class SettingsPatch(BaseModel):
    translation_provider: Optional[str] = None
    llm_api_base: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    clear_api_key: Optional[bool] = False
    qwen_variant: Optional[str] = Field(default=None, pattern=r"^(1\.7b|0\.6b)$")
    mastering_preset: Optional[str] = Field(default=None, pattern="^(raw|podcast|shorts|audiobook)$")
    enable_watermark: Optional[bool] = None
    enable_batch_accel: Optional[bool] = None
    idle_unload_minutes: Optional[int] = Field(default=None, ge=0, le=120)


class PreloadRequest(BaseModel):
    engine_id: str = Field(default="qwen3", pattern="^(qwen3|chatterbox|cosyvoice3)$")


class VoicePatch(BaseModel):
    name: Optional[str] = None
    ref_text: Optional[str] = None


class GlossaryUpsert(BaseModel):
    id: Optional[str] = None
    term: str = Field(..., min_length=1, max_length=120)
    keep_original: bool = True
    lang_overrides: Dict[str, str] = Field(default_factory=dict)
    tts_pronunciation: Dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class TranslateRequest(BaseModel):
    source_text: str = Field(..., min_length=1, max_length=12000)
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
    mastering_preset: Optional[str] = Field(default=None, pattern="^(raw|podcast|shorts|audiobook)$")
    source_text: Optional[str] = ""
    preset_id: Optional[str] = None
    auto_reel: bool = True


class RegenerateTrackRequest(BaseModel):
    text: Optional[str] = None
    mode: Optional[str] = Field(default=None, pattern="^(icl|xvector)$")
    exaggeration: Optional[float] = Field(default=None, ge=0.0, le=1.5)
    cfg_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    instruct: Optional[str] = None
    mastering_preset: Optional[str] = Field(default=None, pattern="^(raw|podcast|shorts|audiobook)$")


class ReelRequest(BaseModel):
    lang_codes: Optional[List[str]] = None
    gap_sec: float = Field(default=0.55, ge=0.1, le=2.0)


class DocumentUrlRequest(BaseModel):
    url: str = Field(..., min_length=6)


class PodcastScriptRequest(BaseModel):
    source_text: str = Field(..., min_length=20, max_length=30000)
    host_a: str = "主持人"
    host_b: str = "嘉宾"
    target_lang: str = "zh"
    style: str = "deep_dive"


class PodcastGenerateRequest(BaseModel):
    title: str = "AI 多角色播客"
    script_text: str = Field(..., min_length=4, max_length=20000)
    speaker_voices: Dict[str, str]
    target_langs: List[str] = Field(default_factory=lambda: ["zh"], min_length=1)
    engine_id: str = Field(default="qwen3", pattern="^(qwen3|chatterbox|cosyvoice3)$")
    mode: str = Field(default="icl", pattern="^(icl|xvector)$")
    mastering_preset: str = Field(default="podcast", pattern="^(raw|podcast|shorts|audiobook)$")
    add_ambient_bgm: bool = True


class VideoUrlImportRequest(BaseModel):
    url: str = Field(..., min_length=6)


class VideoTranslateRequest(BaseModel):
    target_langs: List[str] = Field(..., min_length=1)
    segments: Optional[List[Dict[str, Any]]] = None


class VideoDubRequest(BaseModel):
    target_langs: List[str] = Field(..., min_length=1)
    translations: Dict[str, List[str]]
    engine_id: str = Field(default="qwen3", pattern="^(qwen3|chatterbox|cosyvoice3)$")
    voice_id: str = Field(default="auto_video")
    speaker_voices: Dict[str, str] = Field(default_factory=dict)  # e.g. {"SPEAKER_1": "auto_video", "SPEAKER_2": "v_xxx"}
    mode: str = Field(default="icl", pattern="^(icl|xvector)$")
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.5)
    cfg_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    instruct: str = Field(default="", max_length=300)
    mastering_preset: str = Field(default="shorts", pattern="^(raw|podcast|shorts|audiobook)$")
    burn_subtitles: bool = False
    subtitle_style: str = Field(default="cinema", pattern="^(cinema|shorts_yellow|minimal_bar)$")
    enable_lipsync: bool = False
    bgm_volume: float = Field(default=0.85, ge=0.0, le=1.2)


class OpenAISpeechRequest(BaseModel):
    model: str = "voxpoly-qwen3-1.7b"
    input: str = Field(..., min_length=1, max_length=8000)
    voice: str = "alloy"
    response_format: str = Field(default="mp3", pattern="^(mp3|wav|flac|opus)$")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    language: Optional[str] = None
    mastering_preset: Optional[str] = None


# ---------- Core Synthesis Pipeline (Glossary + Long-Form Batch + Mastering DSP + Watermark) ----------
def _synthesize_text_smart(
    eng: Any,
    engine_id: str,
    voice_id: str,
    ref_wav_path: Path,
    ref_text: str,
    mode: str,
    text: str,
    lang_meta: Dict[str, Any],
    exaggeration: float = 0.5,
    cfg_weight: float = 0.35,
    instruct: str = "",
    chunk_dir: Optional[Path] = None,
    on_chunk_ready: Optional[Any] = None,
    enable_batch: bool = True,
    mastering_preset: str = "raw",
    apply_watermark: bool = True,
) -> Tuple[np.ndarray, int, float, float, int]:
    lang_code = str(lang_meta["code"])
    spoken_text = apply_tts_pronunciation(text, lang_code)
    chunks = split_text_for_tts(spoken_text, lang_code)
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

    # Apply broadcast mastering DSP BEFORE embedding watermark so EQ/compression never attenuates watermark
    if mastering_preset and mastering_preset != "raw":
        final_wav = apply_mastering_dsp(final_wav, sr, mastering_preset)

    if apply_watermark:
        final_wav = embed_watermark(final_wav, sample_rate=sr)

    elapsed = round(time.perf_counter() - t0, 2)
    dur = float(final_wav.size / sr) if sr > 0 else 0.0
    rtf = round(elapsed / dur, 2) if dur > 0.05 else 0.0
    engine_manager.touch()
    return final_wav, sr, elapsed, rtf, len(chunks)


# ---------- Background Workers ----------
def _run_generation_job(job_id: str) -> None:
    job = store.get_job(job_id)
    if not job:
        return
    voice = store.get_voice(job["voice_id"])
    if not voice:
        job["status"] = "failed"
        job["error"] = "所选声纹不存在或已被删除。"
        store.save_job(job)
        return

    settings = store.get_settings(masked=False)
    apply_wm = bool(settings.get("enable_watermark", True))
    enable_batch = bool(settings.get("enable_batch_accel", True))
    mastering_preset = str(job.get("mastering_preset") or settings.get("mastering_preset") or "raw")
    job["watermarked"] = apply_wm
    job["mastering_preset"] = mastering_preset

    engine_id = job.get("engine_id", "qwen3")
    ref_wav_path = VOICES_DIR / voice["id"] / "clone_ref.wav"
    ref_text = str(voice.get("ref_text") or "")
    mode = job.get("mode", "icl")
    effective_mode = (
        engine_id if engine_id in ("chatterbox", "cosyvoice3") else ("xvector" if (mode == "xvector" or not ref_text.strip()) else "icl")
    )
    job["effective_mode"] = effective_mode

    if engine_manager.get_engine(engine_id).get_status()["state"] != "ready":
        job["status"] = "loading_model"
        store.save_job(job)

    try:
        eng = engine_manager.prepare_engine(engine_id)
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = f"语音引擎加载失败: {exc}"
        store.save_job(job)
        return

    job["status"] = "running"
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
                mastering_preset=mastering_preset,
                apply_watermark=apply_wm,
            )
            metrics = save_generated_audio(wav_np, sr, job_dir / f"{lang_code}.wav", job_dir / f"{lang_code}.mp3")
            track.update(
                {
                    "status": "completed",
                    "elapsed_sec": elapsed,
                    "rtf": rtf,
                    "total_chunks": n_chunks,
                    "mastering_preset": mastering_preset,
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
            wav_paths = [job_dir / f"{t['lang_code']}.wav" for t in job["tracks"] if t.get("status") == "completed" and (job_dir / f"{t['lang_code']}.wav").exists()]
            if len(wav_paths) >= 2:
                reel_metrics = merge_multilingual_reel(wav_paths, job_dir / "multilingual_reel.mp3")
                job["reel"] = {"lang_codes": [t["lang_code"] for t in job["tracks"] if t.get("status") == "completed"], "created_at": utc_now_iso(), **reel_metrics}
        except Exception as exc:
            job["reel_error"] = str(exc)

    job["status"] = "completed" if succeeded == len(job["tracks"]) else ("partial" if succeeded > 0 else "failed")
    job["finished_at"] = utc_now_iso()
    job["updated_at"] = utc_now_iso()
    store.save_job(job)


def _run_single_track_regen(job_id: str, lang_code: str, new_text: str, mode: str, exaggeration: float, cfg_weight: float, instruct: str, mastering_preset: str) -> None:
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
    effective_mode = engine_id if engine_id in ("chatterbox", "cosyvoice3") else ("xvector" if (mode == "xvector" or not ref_text.strip()) else "icl")
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
            lang_meta=LANGUAGE_BY_CODE[lang_code],
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            instruct=instruct,
            chunk_dir=job_dir / "chunks" / lang_code,
            enable_batch=enable_batch,
            mastering_preset=mastering_preset,
            apply_watermark=apply_wm,
        )
        metrics = save_generated_audio(wav_np, sr, job_dir / f"{lang_code}.wav", job_dir / f"{lang_code}.mp3")
        for t in job["tracks"]:
            if t["lang_code"] == lang_code:
                t.update({"text": new_text, "status": "completed", "error": None, "elapsed_sec": elapsed, "rtf": rtf, "total_chunks": n_chunks, "mastering_preset": mastering_preset, "watermarked": apply_wm, **metrics})
                break
        if job.get("reel"):
            wav_paths = [job_dir / f"{t['lang_code']}.wav" for t in job["tracks"] if t.get("status") == "completed" and (job_dir / f"{t['lang_code']}.wav").exists()]
            if len(wav_paths) >= 2:
                reel_metrics = merge_multilingual_reel(wav_paths, job_dir / "multilingual_reel.mp3")
                job["reel"] = {"lang_codes": [t["lang_code"] for t in job["tracks"] if t.get("status") == "completed"], "created_at": utc_now_iso(), **reel_metrics}
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


def _run_podcast_job(podcast_id: str) -> None:
    pod = store.get_podcast(podcast_id)
    if not pod:
        return
    settings = store.get_settings(masked=False)
    apply_wm = bool(settings.get("enable_watermark", True))
    engine_id = pod.get("engine_id", "qwen3")
    mastering_preset = pod.get("mastering_preset", "podcast")
    pdir = PODCASTS_DIR / podcast_id
    pdir.mkdir(parents=True, exist_ok=True)

    pod["status"] = "loading_model"
    store.save_podcast(pod)
    try:
        eng = engine_manager.prepare_engine(engine_id)
    except Exception as exc:
        pod["status"] = "failed"
        pod["error"] = str(exc)
        store.save_podcast(pod)
        return

    pod["status"] = "running"
    store.save_podcast(pod)

    turns_by_lang: Dict[str, List[Dict[str, Any]]] = pod.get("turns_by_lang", {})
    speaker_voices: Dict[str, str] = pod.get("speaker_voices", {})
    fallback_voice = store.list_voices()[0] if store.list_voices() else None
    succeeded = 0

    for track in pod["tracks"]:
        code = track["lang_code"]
        lang_meta = LANGUAGE_BY_CODE[code]
        lang_turns = turns_by_lang.get(code, [])
        track["status"] = "generating"
        track["done_turns"] = 0
        track["total_turns"] = len(lang_turns)
        store.save_podcast(pod)

        try:
            t0 = time.perf_counter()
            turn_audios = []
            for idx, turn in enumerate(lang_turns):
                spk = str(turn["speaker"])
                vid = speaker_voices.get(spk) or (fallback_voice["id"] if fallback_voice else "")
                v_obj = store.get_voice(vid) or fallback_voice
                if not v_obj:
                    raise RuntimeError(f"角色 {spk} 未绑定有效声纹。")
                ref_wav = VOICES_DIR / v_obj["id"] / "clone_ref.wav"
                ref_text = str(v_obj.get("ref_text") or "")
                mode = pod.get("mode", "icl")
                eff_mode = engine_id if engine_id in ("chatterbox", "cosyvoice3") else ("xvector" if (mode == "xvector" or not ref_text.strip()) else "icl")

                wav_np, sr, _, _, _ = _synthesize_text_smart(
                    eng=eng,
                    engine_id=engine_id,
                    voice_id=v_obj["id"],
                    ref_wav_path=ref_wav,
                    ref_text=ref_text,
                    mode=eff_mode,
                    text=str(turn["text"]),
                    lang_meta=lang_meta,
                    enable_batch=False,
                    mastering_preset="raw",
                    apply_watermark=False,
                )
                turn_audios.append((spk, wav_np, sr))
                track["done_turns"] = idx + 1
                pod["updated_at"] = utc_now_iso()
                store.save_podcast(pod)

            final_wav = assemble_podcast_audio(turn_audios, add_ambient_bgm=bool(pod.get("add_ambient_bgm", True)), target_sr=QWEN_SAMPLE_RATE)
            if mastering_preset != "raw":
                final_wav = apply_mastering_dsp(final_wav, QWEN_SAMPLE_RATE, mastering_preset)
            if apply_wm:
                final_wav = embed_watermark(final_wav, QWEN_SAMPLE_RATE)

            metrics = save_generated_audio(final_wav, QWEN_SAMPLE_RATE, pdir / f"podcast_{code}.wav", pdir / f"podcast_{code}.mp3")
            track.update({"status": "completed", "elapsed_sec": round(time.perf_counter() - t0, 2), "watermarked": apply_wm, **metrics})
            succeeded += 1
        except Exception as exc:
            track["status"] = "failed"
            track["error"] = str(exc)

        pod["updated_at"] = utc_now_iso()
        store.save_podcast(pod)

    pod["status"] = "completed" if succeeded == len(pod["tracks"]) else ("partial" if succeeded > 0 else "failed")
    pod["updated_at"] = utc_now_iso()
    store.save_podcast(pod)


def _run_video_dubbing_job(video_id: str, req_data: Dict[str, Any]) -> None:
    video = store.get_video(video_id)
    if not video:
        return
    vdir = VIDEOS_DIR / video_id
    src_video = vdir / video["source_filename"]
    bgm_wav = vdir / "bgm_24k.wav"
    segments = video.get("segments", [])
    speaker_profiles = {p["speaker"]: p for p in (video.get("speakers") or [])}
    speaker_voice_map: Dict[str, str] = req_data.get("speaker_voices") or {}
    default_voice_id = str(req_data.get("voice_id") or "auto_video")

    settings = store.get_settings(masked=False)
    apply_wm = bool(settings.get("enable_watermark", True))
    mastering_preset = str(req_data.get("mastering_preset") or "shorts")
    subtitle_style = str(req_data.get("subtitle_style") or "cinema")
    engine_id = str(req_data.get("engine_id") or "qwen3")
    mode = str(req_data.get("mode") or "icl")

    video["dub_status"] = "loading_model"
    store.save_video(video)
    try:
        eng = engine_manager.prepare_engine(engine_id)
    except Exception as exc:
        video["dub_status"] = "failed"
        video["dub_error"] = f"语音引擎加载失败: {exc}"
        store.save_video(video)
        return

    video["dub_status"] = "running"
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
        store.save_video(video)

        try:
            t0 = time.perf_counter()
            placed: List[tuple] = []
            for s_idx, seg in enumerate(segments):
                line_txt = (target_lines[s_idx] if s_idx < len(target_lines) else seg["text"]).strip()
                if not line_txt:
                    continue
                spk = str(seg.get("speaker") or "SPEAKER_1")
                chosen_vid = speaker_voice_map.get(spk) or default_voice_id

                if chosen_vid == "auto_video":
                    prof = speaker_profiles.get(spk)
                    ref_wav = (vdir / prof["ref_filename"]) if (prof and (vdir / prof["ref_filename"]).exists()) else (vdir / "auto_ref.wav")
                    ref_text = str(prof["ref_text"]) if prof else str(video.get("auto_ref_text") or "")
                    eff_vid = f"vid_{video_id}_{spk}"
                else:
                    v_obj = store.get_voice(chosen_vid)
                    if not v_obj:
                        raise RuntimeError(f"声纹 {chosen_vid} 不存在。")
                    ref_wav = VOICES_DIR / chosen_vid / "clone_ref.wav"
                    ref_text = str(v_obj.get("ref_text") or "")
                    eff_vid = chosen_vid

                eff_mode = engine_id if engine_id in ("chatterbox", "cosyvoice3") else ("xvector" if (mode == "xvector" or not ref_text.strip()) else "icl")
                wav_np, sr, _, _, _ = _synthesize_text_smart(
                    eng=eng,
                    engine_id=engine_id,
                    voice_id=eff_vid,
                    ref_wav_path=ref_wav,
                    ref_text=ref_text,
                    mode=eff_mode,
                    text=line_txt,
                    lang_meta=lang_meta,
                    exaggeration=float(req_data.get("exaggeration", 0.5)),
                    cfg_weight=float(req_data.get("cfg_weight", 0.35)),
                    instruct=str(req_data.get("instruct") or ""),
                    enable_batch=False,
                    mastering_preset=mastering_preset,
                    apply_watermark=apply_wm,
                )
                next_start = float(segments[s_idx + 1]["start"]) if s_idx + 1 < len(segments) else float(video["duration_sec"])
                slot_budget = max(float(seg["end"]) - float(seg["start"]), next_start - float(seg["start"]) - 0.08)
                fitted = fit_segment_tempo(wav_np, sr, slot_budget, vdir, f"{code}_{s_idx}")
                placed.append((float(seg["start"]), fitted))
                out_item["done_segments"] = s_idx + 1
                store.save_video(video)

            write_srt(segments, target_lines, vdir / f"sub_{code}.srt", bilingual=False)
            write_srt(segments, target_lines, vdir / f"sub_{code}_bilingual.srt", bilingual=True)
            ass_path = write_styled_ass(
                segments,
                target_lines,
                vdir / f"sub_{code}.ass",
                style_id=subtitle_style,
                width=int(video.get("width") or 1280),
                height=int(video.get("height") or 720),
                bilingual=True,
            )

            mux_metrics = mix_and_mux_dubbed_video(
                video_path=src_video,
                video_duration_sec=float(video["duration_sec"]),
                bgm_wav_path=bgm_wav,
                bgm_method=str(video.get("bgm_method") or "ffmpeg-ducking"),
                placed_segments=placed,
                out_mp4_path=vdir / f"dub_{code}.mp4",
                out_mp3_path=vdir / f"dub_{code}.mp3",
                subtitle_ass_path=ass_path,
                burn_subtitles=bool(req_data.get("burn_subtitles", False)),
                enable_lipsync=bool(req_data.get("enable_lipsync", False)),
                bgm_volume=float(req_data.get("bgm_volume", 0.85)),
            )
            out_item.update(
                {
                    "status": "completed",
                    "elapsed_sec": round(time.perf_counter() - t0, 2),
                    "burned_subtitles": bool(req_data.get("burn_subtitles", False)),
                    "subtitle_style": subtitle_style,
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


def _analyze_video_file(src_path: Path, video_id: str, title: str, orig_filename: str, source_url: str = "") -> Dict[str, Any]:
    vdir = src_path.parent
    meta = probe_video(src_path)
    stereo_wav, mono_24k = extract_video_audio(src_path, vdir)
    vocals_wav, _, bgm_method = separate_vocals_and_bgm(stereo_wav, mono_24k, vdir)
    asr_res = asr_service.transcribe_segments(vocals_wav)
    segments = asr_res.get("segments") or []
    if not segments:
        raise ValueError("未在视频中检测到清晰人声台词。")
    segments, speakers = diarize_video_segments(vocals_wav, segments, vdir)
    auto_ref_text = speakers[0]["ref_text"] if speakers else ""

    now = utc_now_iso()
    record: Dict[str, Any] = {
        "id": video_id,
        "title": title[:48],
        "original_filename": orig_filename,
        "source_filename": src_path.name,
        "source_url": source_url,
        "bgm_method": bgm_method,
        "detected_language": asr_res.get("language", "unknown"),
        "full_transcript": asr_res.get("text", ""),
        "segments": segments,
        "speakers": speakers,
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
    return record


class AuthSessionRequest(BaseModel):
    token: str = Field(default="", max_length=4096)


# ---------- Core System Routes ----------
@app.get("/")
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="前端页面尚未就绪")
    return FileResponse(str(index_path))


@app.get("/api/health")
def health():
    expected = _active_access_token()
    return {
        "ok": True,
        "service": "voxpoly-studio-v3",
        "version": "3.0.1",
        "requires_token": bool(expected),
        "token_strong": len(expected) >= 12,
        "media_protected": bool(expected),
    }


@app.post("/api/auth/session")
def create_auth_session(request: Request, payload: AuthSessionRequest):
    if not _passes_same_origin_check(request, require_custom_header=False):
        return JSONResponse(
            status_code=403,
            content={"detail": "跨源登录请求已被拒绝。"},
            headers={"Cache-Control": "private, no-store"},
        )
    expected = _active_access_token()
    if not expected:
        return JSONResponse(
            {"authenticated": True, "requires_token": False},
            headers={"Cache-Control": "private, no-store"},
        )
    ckey = f"sess:{_client_key(request)}"
    if _auth_state.is_rate_limited(ckey):
        return JSONResponse(
            status_code=429,
            content={"detail": "认证失败次数过多，请 60 秒后重试。"},
            headers={"Cache-Control": "private, no-store", "Retry-After": "60"},
        )
    supplied = (payload.token or _extract_header_token(request)).strip()
    if not _tokens_match(supplied, expected):
        _auth_state.record_failure(ckey)
        return JSONResponse(
            status_code=401,
            content={"detail": "访问口令不正确，请重试。"},
            headers={"Cache-Control": "private, no-store"},
        )
    _auth_state.clear_failures(ckey)
    sid = _auth_state.create_session(expected)
    resp = JSONResponse({"authenticated": True, "requires_token": True}, headers={"Cache-Control": "private, no-store"})
    resp.set_cookie(
        key=_SESSION_COOKIE_NAME,
        value=sid,
        httponly=True,
        samesite="lax",
        secure=_is_https_request(request),
        max_age=_SESSION_TTL_SEC,
        path="/",
    )
    return resp


@app.delete("/api/auth/session")
def clear_auth_session(request: Request):
    if _active_access_token() and not _passes_same_origin_check(request, require_custom_header=True):
        return JSONResponse(
            status_code=403,
            content={"detail": "跨源登出请求已被拒绝。"},
            headers={"Cache-Control": "private, no-store"},
        )
    sid = (request.cookies.get(_SESSION_COOKIE_NAME) or "").strip()
    _auth_state.revoke_session(sid)
    resp = JSONResponse({"authenticated": False}, headers={"Cache-Control": "private, no-store"})
    resp.delete_cookie(key=_SESSION_COOKIE_NAME, path="/")
    return resp


@app.get("/api/system/status")
def system_status():
    return {
        "engine": engine_manager.get_status(),
        "asr": asr_service.status,
        "features": {
            "demucs_available": has_demucs(),
            "yt_dlp_available": has_yt_dlp(),
            "lipsync_available": has_lipsync(),
            "access_token_enabled": bool(_active_access_token()),
            "qwen_batch_size": QWEN_BATCH_SIZE,
            "watermark_supported": True,
            "openai_compat_url": "/v1/audio/speech",
        },
        "languages": LANGUAGES,
        "default_selected_langs": DEFAULT_SELECTED_LANGS,
        "presets": PRESET_SCENARIOS,
        "cosyvoice_presets": COSYVOICE_INSTRUCT_PRESETS,
        "mastering_presets": [{k: v for k, v in p.items() if k != "af"} for p in MASTERING_PRESETS],
        "subtitle_styles": SUBTITLE_STYLES,
        "qwen_variants": [{"id": k, "model_id": v["model_id"], "label": v["label"], "vram_gb": v["vram_gb"]} for k, v in QWEN_VARIANTS.items()],
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
    if payload.llm_api_base is not None:
        base_clean = payload.llm_api_base.strip()
        if base_clean:
            parsed = urlparse(base_clean)
            if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
                raise HTTPException(status_code=400, detail="llm_api_base 必须是有效的 http:// 或 https:// 地址。")
            # In token-protected (remote/public) deployments, block private/metadata LLM endpoints unless admin opts in
            if _active_access_token() and os.getenv("VOXPOLY_ALLOW_LOCAL_LLM", "").strip() != "1":
                try:
                    validate_public_http_url(base_clean)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=f"公网模式下拒绝内网 LLM 地址: {exc}") from exc
    updated = store.update_settings(payload.model_dump(exclude_none=True))
    if payload.qwen_variant:
        engine_manager.qwen3.set_variant(payload.qwen_variant)
    return updated


# ---------- OpenAI Compatible TTS API (/v1/models & /v1/audio/speech) ----------
@app.get("/v1/models")
def openai_list_models():
    now = int(time.time())
    models = [
        {"id": "voxpoly-qwen3-1.7b", "object": "model", "created": now, "owned_by": "voxpoly-studio"},
        {"id": "voxpoly-qwen3-0.6b", "object": "model", "created": now, "owned_by": "voxpoly-studio"},
        {"id": "voxpoly-chatterbox-v3", "object": "model", "created": now, "owned_by": "voxpoly-studio"},
        {"id": "voxpoly-cosyvoice3", "object": "model", "created": now, "owned_by": "voxpoly-studio"},
        {"id": "tts-1", "object": "model", "created": now, "owned_by": "voxpoly-studio"},
        {"id": "tts-1-hd", "object": "model", "created": now, "owned_by": "voxpoly-studio"},
    ]
    return {"object": "list", "data": models}


@app.get("/v1/audio/voices")
def openai_list_voices():
    return {"voices": [{"id": v["id"], "name": v["name"], "duration_sec": v["duration_sec"]} for v in store.list_voices()]}


@app.post("/v1/audio/speech")
def openai_create_speech(payload: OpenAISpeechRequest):
    voices = store.list_voices()
    if not voices:
        raise HTTPException(status_code=400, detail="声纹库为空，请先上传至少一段参考声纹。")

    # Match by exact ID, display name, or fall back gracefully for OpenAI aliases (alloy/echo/nova/...)
    chosen = next(
        (v for v in voices if v["id"] == payload.voice or v["name"] == payload.voice),
        voices[0],
    )
    model_l = payload.model.lower()
    if "chatterbox" in model_l:
        engine_id = "chatterbox"
    elif "cosyvoice" in model_l:
        engine_id = "cosyvoice3"
    else:
        engine_id = "qwen3"
        if "0.6b" in model_l:
            engine_manager.qwen3.set_variant("0.6b")

    lang_code = payload.language if payload.language in LANGUAGE_BY_CODE else (detect_script_language(payload.input) or "en")
    if engine_id not in LANGUAGE_BY_CODE[lang_code]["engines"]:
        lang_code = "en"

    settings = store.get_settings(masked=False)
    mastering = payload.mastering_preset or str(settings.get("mastering_preset") or "raw")
    apply_wm = bool(settings.get("enable_watermark", True))

    eng = engine_manager.prepare_engine(engine_id)
    ref_wav = VOICES_DIR / chosen["id"] / "clone_ref.wav"
    ref_text = str(chosen.get("ref_text") or "")
    eff_mode = engine_id if engine_id in ("chatterbox", "cosyvoice3") else ("icl" if ref_text.strip() else "xvector")

    wav_np, sr, _, _, _ = _synthesize_text_smart(
        eng=eng,
        engine_id=engine_id,
        voice_id=chosen["id"],
        ref_wav_path=ref_wav,
        ref_text=ref_text,
        mode=eff_mode,
        text=payload.input,
        lang_meta=LANGUAGE_BY_CODE[lang_code],
        enable_batch=bool(settings.get("enable_batch_accel", True)),
        mastering_preset=mastering,
        apply_watermark=apply_wm,
    )

    with tempfile.TemporaryDirectory(prefix="voxpoly_oai_") as tmp:
        wav_p = Path(tmp) / "out.wav"
        sf.write(str(wav_p), wav_np.astype(np.float32), sr, subtype="PCM_16")
        if abs(payload.speed - 1.0) > 0.03:
            sp_p = Path(tmp) / "speed.wav"
            _run_ffmpeg(["-i", str(wav_p), "-af", f"atempo={payload.speed:.3f}", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(sp_p)])
            wav_p = sp_p

        fmt = payload.response_format
        if fmt == "wav":
            return Response(content=wav_p.read_bytes(), media_type="audio/wav")
        out_p = Path(tmp) / f"out.{fmt}"
        codec_map = {
            "mp3": (["-c:a", "libmp3lame", "-b:a", "192k"], "audio/mpeg"),
            "flac": (["-c:a", "flac"], "audio/flac"),
            "opus": (["-c:a", "libopus", "-b:a", "128k"], "audio/opus"),
        }
        c_args, mime = codec_map[fmt]
        _run_ffmpeg(["-i", str(wav_p), *c_args, str(out_p)])
        return Response(content=out_p.read_bytes(), media_type=mime)


# ---------- Glossary Routes ----------
@app.get("/api/glossary")
def list_glossary():
    return {"terms": glossary_store.list_terms()}


@app.post("/api/glossary")
def upsert_glossary(payload: GlossaryUpsert):
    try:
        rec = glossary_store.upsert_term(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"term": rec, "terms": glossary_store.list_terms()}


@app.delete("/api/glossary/{term_id}")
def delete_glossary(term_id: str):
    if not glossary_store.delete_term(term_id):
        raise HTTPException(status_code=404, detail="未找到该词典条目。")
    return {"deleted": True, "terms": glossary_store.list_terms()}


# ---------- Watermark Verify ----------
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
            _run_ffmpeg(["-i", str(raw_p), "-vn", "-ac", "1", "-ar", str(QWEN_SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav_p)])
            audio, sr = load_wav_float32(wav_p)
            return {"filename": file.filename, **detect_watermark(audio, sr)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"音频解析失败: {exc}") from exc


# ---------- Document Importer & Multi-Speaker Podcast Routes ----------
@app.post("/api/documents/import-file")
async def import_document_file(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传文档为空。")
    try:
        return {"document": parse_document_bytes(raw, file.filename or "doc.txt")}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/documents/import-url")
def import_document_url(payload: DocumentUrlRequest):
    try:
        return {"document": parse_url_article(payload.url)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/podcasts/script-from-text")
def build_podcast_script(payload: PodcastScriptRequest):
    try:
        script, provider = generate_two_host_podcast_script(
            source_text=payload.source_text,
            settings=store.get_settings(masked=False),
            host_a=payload.host_a,
            host_b=payload.host_b,
            target_lang=payload.target_lang,
            style=payload.style,
        )
        turns, speakers = parse_dialogue_script(script)
        return {"script_text": script, "turns": turns, "speakers": speakers, "provider": provider}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/podcasts/generate")
def create_podcast_job(payload: PodcastGenerateRequest):
    turns, speakers = parse_dialogue_script(payload.script_text)
    if not turns:
        raise HTTPException(status_code=400, detail="未能解析出有效对话台词，请使用 [主持人]: 台词 格式。")
    eng_st = engine_manager.get_engine(payload.engine_id).get_status()
    if not eng_st["installed"]:
        raise HTTPException(status_code=400, detail=f"{eng_st['display_name']} 尚未安装。")

    valid_langs = [c for c in payload.target_langs if c in LANGUAGE_BY_CODE and payload.engine_id in LANGUAGE_BY_CODE[c]["engines"]]
    if not valid_langs:
        raise HTTPException(status_code=400, detail="请至少选择一个当前引擎支持的语种。")

    turns_by_lang, provider, _ = translate_dialogue_turns(turns, valid_langs, store.get_settings(masked=False))
    pod_id = f"pod_{uuid.uuid4().hex[:10]}"
    now = utc_now_iso()
    record: Dict[str, Any] = {
        "id": pod_id,
        "title": payload.title.strip() or "多角色播客",
        "script_text": payload.script_text,
        "speakers": speakers,
        "speaker_voices": payload.speaker_voices,
        "turns_by_lang": turns_by_lang,
        "translation_provider": provider,
        "engine_id": payload.engine_id,
        "engine_name": eng_st["display_name"],
        "mode": payload.mode,
        "mastering_preset": payload.mastering_preset,
        "add_ambient_bgm": payload.add_ambient_bgm,
        "status": "queued",
        "error": None,
        "tracks": [
            {
                "lang_code": c,
                "lang_label": LANGUAGE_BY_CODE[c]["label_zh"],
                "region_tag": LANGUAGE_BY_CODE[c]["region_tag"],
                "status": "pending",
                "done_turns": 0,
                "total_turns": len(turns),
            }
            for c in valid_langs
        ],
        "created_at": now,
        "updated_at": now,
    }
    store.save_podcast(record)
    threading.Thread(target=_run_podcast_job, args=(pod_id,), daemon=True).start()
    return {"podcast": _decorate_podcast(record)}


@app.get("/api/podcasts")
def list_podcasts(limit: int = 20):
    return {"podcasts": [_decorate_podcast(p) for p in store.list_podcasts(limit=limit)]}


@app.get("/api/podcasts/{podcast_id}")
def get_podcast(podcast_id: str):
    pod = store.get_podcast(podcast_id)
    if not pod:
        raise HTTPException(status_code=404, detail="未找到该播客任务。")
    return {"podcast": _decorate_podcast(pod)}


@app.delete("/api/podcasts/{podcast_id}")
def delete_podcast(podcast_id: str):
    if not store.delete_podcast(podcast_id):
        raise HTTPException(status_code=404, detail="未找到该播客任务。")
    return {"deleted": True}


@app.get("/api/podcasts/{podcast_id}/zip")
def download_podcast_zip(podcast_id: str):
    try:
        zp = store.build_podcast_zip(podcast_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path=str(zp), media_type="application/zip", filename=f"VoxPoly_Podcast_{podcast_id}.zip")


# ---------- Voices ----------
@app.get("/api/voices")
def list_voices():
    return {"voices": [_decorate_voice(v) for v in store.list_voices()]}


@app.post("/api/voices")
async def create_voice(file: UploadFile = File(...), name: str = Form(default=""), ref_text: str = Form(default=""), auto_transcribe: bool = Form(default=True)):
    filename = file.filename or "recording.webm"
    ext = Path(filename).suffix.lower() or ".wav"
    if ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(status_code=400, detail=f"暂不支持的音频格式 {ext}。")
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"文件超过 {MAX_UPLOAD_MB}MB 上限。")

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
            asr_note = f"自动转写暂不可用: {exc}"

    record: Dict[str, Any] = {
        "id": voice_id,
        "name": name.strip() or Path(filename).stem[:32] or f"我的声音 {voice_id[-4:]}",
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
    if not store.get_voice(voice_id):
        raise HTTPException(status_code=404, detail="未找到该声纹档案。")
    clone_wav = VOICES_DIR / voice_id / "clone_ref.wav"
    asr_res = asr_service.transcribe(clone_wav)
    updated = store.update_voice_fields(voice_id, {"ref_text": asr_res["text"], "detected_language": asr_res["language"], "asr_note": None})
    engine_manager.invalidate_voice_cache(voice_id)
    return {"voice": _decorate_voice(updated), "asr": asr_res}


@app.patch("/api/voices/{voice_id}")
def patch_voice(voice_id: str, payload: VoicePatch):
    fields = {k: v.strip() for k, v in payload.model_dump(exclude_none=True).items()}
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
    try:
        translations, provider, warnings = translate_scripts(
            source_text=payload.source_text,
            target_langs=payload.target_langs,
            settings=store.get_settings(masked=False),
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
        raise HTTPException(status_code=404, detail="所选声纹不存在。")
    eng_st = engine_manager.get_engine(payload.engine_id).get_status()
    if not eng_st["installed"]:
        raise HTTPException(status_code=400, detail=f"{eng_st['display_name']} 尚未安装。")

    ordered_tracks = []
    for lang in LANGUAGES:
        code = str(lang["code"])
        if code in payload.scripts and str(payload.scripts[code]).strip():
            if payload.engine_id not in lang["engines"]:
                raise HTTPException(status_code=400, detail=f"{eng_st['display_name']} 不支持 {lang['label_zh']}。")
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
        "mastering_preset": payload.mastering_preset or str(store.get_settings(masked=False).get("mastering_preset") or "raw"),
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
    return {"deleted": True}


@app.post("/api/jobs/{job_id}/regenerate/{lang_code}")
def regenerate_track(job_id: str, lang_code: str, payload: RegenerateTrackRequest):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="未找到该生成任务。")
    target_track = next((t for t in job["tracks"] if t["lang_code"] == lang_code), None)
    if not target_track:
        raise HTTPException(status_code=404, detail=f"任务中不包含语种 {lang_code}")
    new_text = (payload.text if payload.text is not None else target_track["text"]).strip()
    target_track["text"] = new_text
    target_track["status"] = "generating"
    target_track["error"] = None
    job["status"] = "running"
    store.save_job(job)
    threading.Thread(
        target=_run_single_track_regen,
        args=(
            job_id,
            lang_code,
            new_text,
            payload.mode or job.get("mode", "icl"),
            float(payload.exaggeration if payload.exaggeration is not None else job.get("exaggeration", 0.5)),
            float(payload.cfg_weight if payload.cfg_weight is not None else job.get("cfg_weight", 0.35)),
            payload.instruct if payload.instruct is not None else str(job.get("instruct") or ""),
            payload.mastering_preset or str(job.get("mastering_preset") or "raw"),
        ),
        daemon=True,
    ).start()
    return {"job": _decorate_job(job)}


@app.post("/api/jobs/{job_id}/reel")
def build_reel(job_id: str, payload: ReelRequest):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="未找到该生成任务。")
    job_dir = OUTPUTS_DIR / job_id
    chosen = payload.lang_codes or [t["lang_code"] for t in job["tracks"] if t.get("status") == "completed"]
    wav_paths = [job_dir / f"{c}.wav" for c in chosen if (job_dir / f"{c}.wav").exists()]
    if not wav_paths:
        raise HTTPException(status_code=400, detail="没有已完成音轨。")
    reel_metrics = merge_multilingual_reel(wav_paths, job_dir / "multilingual_reel.mp3", gap_sec=payload.gap_sec)
    job["reel"] = {"lang_codes": [p.stem for p in wav_paths], "created_at": utc_now_iso(), **reel_metrics}
    store.save_job(job)
    return {"job": _decorate_job(job)}


@app.get("/api/jobs/{job_id}/zip")
def download_job_zip(job_id: str):
    try:
        zp = store.build_job_zip(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path=str(zp), media_type="application/zip", filename=f"VoxPoly_{job_id}.zip")


# ---------- Video Dubbing V3 Routes ----------
@app.get("/api/videos")
def list_videos(limit: int = 15):
    return {"videos": [_decorate_video(v) for v in store.list_videos(limit=limit)]}


@app.get("/api/videos/{video_id}")
def get_video(video_id: str):
    video = store.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="未找到该视频记录。")
    return {"video": _decorate_video(video)}


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str):
    if not store.delete_video(video_id):
        raise HTTPException(status_code=404, detail="未找到该视频记录。")
    return {"deleted": True}


@app.post("/api/videos/analyze")
async def analyze_uploaded_video(file: UploadFile = File(...)):
    filename = file.filename or "input.mp4"
    ext = Path(filename).suffix.lower() or ".mp4"
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"暂不支持的视频格式 {ext}。")
    raw_bytes = await file.read()
    if not raw_bytes or len(raw_bytes) > MAX_VIDEO_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"视频为空或超过 {MAX_VIDEO_MB}MB 上限。")

    video_id = f"vid_{uuid.uuid4().hex[:10]}"
    vdir = VIDEOS_DIR / video_id
    vdir.mkdir(parents=True, exist_ok=True)
    src_path = vdir / f"source{ext}"
    src_path.write_bytes(raw_bytes)
    try:
        rec = _analyze_video_file(src_path, video_id, Path(filename).stem, filename)
    except Exception as exc:
        shutil.rmtree(vdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"video": _decorate_video(rec)}


@app.post("/api/videos/import-url")
def import_video_url(payload: VideoUrlImportRequest):
    video_id = f"vid_{uuid.uuid4().hex[:10]}"
    vdir = VIDEOS_DIR / video_id
    vdir.mkdir(parents=True, exist_ok=True)
    try:
        src_path, title = download_video_from_url(payload.url, vdir)
        rec = _analyze_video_file(src_path, video_id, title, src_path.name, source_url=payload.url)
    except Exception as exc:
        shutil.rmtree(vdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"video": _decorate_video(rec)}


@app.post("/api/videos/{video_id}/translate")
def translate_video_segments(video_id: str, payload: VideoTranslateRequest):
    video = store.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="未找到该视频记录。")
    if payload.segments:
        video["segments"] = payload.segments
    translations, provider, warnings = translate_segments_for_dubbing(
        segments=video.get("segments", []),
        target_langs=payload.target_langs,
        settings=store.get_settings(masked=False),
        source_lang=video.get("detected_language"),
    )
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
            raise HTTPException(status_code=400, detail=f"{eng_st['display_name']} 不支持语种 {code}。")
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
        zp = store.build_video_zip(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path=str(zp), media_type="application/zip", filename=f"VoxPoly_Video_{video_id}.zip")
