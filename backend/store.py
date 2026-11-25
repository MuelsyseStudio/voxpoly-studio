from __future__ import annotations

import json
import shutil
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import DEFAULT_QWEN_VARIANT, OUTPUTS_DIR, PODCASTS_DIR, SETTINGS_FILE, VIDEOS_DIR, VOICES_DIR

VOICES_INDEX = VOICES_DIR / "index.json"
JOBS_INDEX = OUTPUTS_DIR / "index.json"
PODCASTS_INDEX = PODCASTS_DIR / "index.json"
VIDEOS_INDEX = VIDEOS_DIR / "index.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "translation_provider": "auto",
    "llm_api_base": "",
    "llm_api_key": "",
    "llm_model": "deepseek-chat",
    "qwen_variant": DEFAULT_QWEN_VARIANT,
    "mastering_preset": "raw",
    "enable_watermark": True,
    "enable_batch_accel": True,
    "idle_unload_minutes": 0,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class StudioStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ensure_files()

    def _ensure_files(self) -> None:
        for d in (VOICES_DIR, OUTPUTS_DIR, PODCASTS_DIR, VIDEOS_DIR):
            d.mkdir(parents=True, exist_ok=True)
        for p, default in (
            (VOICES_INDEX, []),
            (JOBS_INDEX, []),
            (PODCASTS_INDEX, []),
            (VIDEOS_INDEX, []),
            (SETTINGS_FILE, DEFAULT_SETTINGS),
        ):
            if not p.exists():
                self._write_json(p, default)

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        try:
            if not path.exists():
                return default
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    # ---------- Settings ----------
    def get_settings(self, masked: bool = False) -> Dict[str, Any]:
        with self._lock:
            raw = self._read_json(SETTINGS_FILE, DEFAULT_SETTINGS)
            data = {**DEFAULT_SETTINGS, **(raw if isinstance(raw, dict) else {})}
            if not masked:
                return dict(data)
            out = dict(data)
            key = str(out.get("llm_api_key") or "")
            out["has_llm_api_key"] = bool(key)
            out["llm_api_key_masked"] = f"{key[:4]}••••{key[-4:]}" if len(key) > 10 else ("••••" if key else "")
            out.pop("llm_api_key", None)
            return out

    def update_settings(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            cur = self.get_settings(masked=False)
            if "translation_provider" in patch and patch["translation_provider"] in ("auto", "free", "llm"):
                cur["translation_provider"] = patch["translation_provider"]
            old_api_base = str(cur.get("llm_api_base") or "").strip()
            base_changed = False
            if "llm_api_base" in patch:
                new_base = str(patch["llm_api_base"] or "").strip()
                base_changed = bool(new_base != old_api_base)
                cur["llm_api_base"] = new_base
            if "llm_model" in patch:
                cur["llm_model"] = str(patch["llm_model"] or "").strip() or "deepseek-chat"
            if "enable_watermark" in patch:
                cur["enable_watermark"] = bool(patch["enable_watermark"])
            if "enable_batch_accel" in patch:
                cur["enable_batch_accel"] = bool(patch["enable_batch_accel"])
            if "qwen_variant" in patch and patch["qwen_variant"] in ("1.7b", "0.6b"):
                cur["qwen_variant"] = patch["qwen_variant"]
            if "mastering_preset" in patch and patch["mastering_preset"] in ("raw", "podcast", "shorts", "audiobook"):
                cur["mastering_preset"] = patch["mastering_preset"]
            if "idle_unload_minutes" in patch:
                cur["idle_unload_minutes"] = max(0, min(120, int(patch["idle_unload_minutes"])))

            supplied_key = str(patch.get("llm_api_key") or "").strip() if "llm_api_key" in patch else ""
            if patch.get("clear_api_key"):
                cur["llm_api_key"] = ""
            elif supplied_key and "••••" not in supplied_key:
                cur["llm_api_key"] = supplied_key
            elif base_changed:
                # Never forward an old saved API key to a newly changed llm_api_base host
                cur["llm_api_key"] = ""
            self._write_json(SETTINGS_FILE, cur)
            return self.get_settings(masked=True)

    # ---------- Voices ----------
    def list_voices(self) -> List[Dict[str, Any]]:
        with self._lock:
            voices = self._read_json(VOICES_INDEX, [])
            return [v for v in voices if (VOICES_DIR / v["id"] / "clone_ref.wav").exists()]

    def get_voice(self, voice_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return next((v for v in self.list_voices() if v["id"] == voice_id), None)

    def upsert_voice(self, voice: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            voices = [v for v in self._read_json(VOICES_INDEX, []) if v["id"] != voice["id"]]
            voices.insert(0, voice)
            self._write_json(VOICES_INDEX, voices)
            return voice

    def update_voice_fields(self, voice_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            voices = self._read_json(VOICES_INDEX, [])
            updated = None
            for v in voices:
                if v["id"] == voice_id:
                    v.update(fields)
                    v["updated_at"] = utc_now_iso()
                    updated = v
                    break
            if updated is not None:
                self._write_json(VOICES_INDEX, voices)
            return updated

    def delete_voice(self, voice_id: str) -> bool:
        with self._lock:
            voices = self._read_json(VOICES_INDEX, [])
            new_voices = [v for v in voices if v["id"] != voice_id]
            if len(new_voices) == len(voices):
                return False
            self._write_json(VOICES_INDEX, new_voices)
            shutil.rmtree(VOICES_DIR / voice_id, ignore_errors=True)
            return True

    # ---------- Audio Jobs ----------
    def list_jobs(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self._lock:
            return self._read_json(JOBS_INDEX, [])[:limit]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return next((j for j in self._read_json(JOBS_INDEX, []) if j["id"] == job_id), None)

    def save_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            jobs = self._read_json(JOBS_INDEX, [])
            for idx, existing in enumerate(jobs):
                if existing["id"] == job["id"]:
                    jobs[idx] = job
                    break
            else:
                jobs.insert(0, job)
            self._write_json(JOBS_INDEX, jobs[:100])
            return job

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            jobs = self._read_json(JOBS_INDEX, [])
            new_jobs = [j for j in jobs if j["id"] != job_id]
            if len(new_jobs) == len(jobs):
                return False
            self._write_json(JOBS_INDEX, new_jobs)
            shutil.rmtree(OUTPUTS_DIR / job_id, ignore_errors=True)
            return True

    def build_job_zip(self, job_id: str) -> Path:
        job = self.get_job(job_id)
        if not job:
            raise FileNotFoundError(f"未找到任务: {job_id}")
        job_dir = OUTPUTS_DIR / job_id
        zip_path = job_dir / f"voxpoly_{job_id}.zip"

        lines = [
            "VoxPoly Studio V2 多语种语音导出清单",
            f"任务 ID: {job['id']}",
            f"推理引擎: {job.get('engine_name', job.get('engine_id', 'qwen3'))}",
            f"声纹名称: {job.get('voice_name', '')}",
            f"水印保护: {'已嵌入 VoxPoly 隐形水印' if job.get('watermarked') else '未启用'}",
            f"创建时间: {job.get('created_at', '')}",
            "-" * 50,
        ]
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for track in job.get("tracks", []):
                if track.get("status") != "completed":
                    continue
                code = track["lang_code"]
                lines.append(f"[{code.upper()} · {track.get('lang_label', code)}] ({track.get('duration_sec', 0)}s)")
                lines.append(track.get("text", ""))
                lines.append("")
                for ext in ("mp3", "wav"):
                    fpath = job_dir / f"{code}.{ext}"
                    if fpath.exists():
                        zf.write(fpath, arcname=f"tracks/{code}_{track.get('lang_label', code)}.{ext}")
            reel_path = job_dir / "multilingual_reel.mp3"
            if job.get("reel") and reel_path.exists():
                zf.write(reel_path, arcname="multilingual_reel.mp3")
            zf.writestr("scripts.txt", "\n".join(lines).encode("utf-8"))
            zf.writestr("manifest.json", json.dumps(job, ensure_ascii=False, indent=2).encode("utf-8"))
        return zip_path

    # ---------- Video Dubbing Jobs ----------
    def list_videos(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return self._read_json(VIDEOS_INDEX, [])[:limit]

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return next((v for v in self._read_json(VIDEOS_INDEX, []) if v["id"] == video_id), None)

    def save_video(self, video: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            vids = self._read_json(VIDEOS_INDEX, [])
            for idx, existing in enumerate(vids):
                if existing["id"] == video["id"]:
                    vids[idx] = video
                    break
            else:
                vids.insert(0, video)
            self._write_json(VIDEOS_INDEX, vids[:50])
            return video

    def delete_video(self, video_id: str) -> bool:
        with self._lock:
            vids = self._read_json(VIDEOS_INDEX, [])
            new_vids = [v for v in vids if v["id"] != video_id]
            if len(new_vids) == len(vids):
                return False
            self._write_json(VIDEOS_INDEX, new_vids)
            shutil.rmtree(VIDEOS_DIR / video_id, ignore_errors=True)
            return True

    def build_video_zip(self, video_id: str) -> Path:
        video = self.get_video(video_id)
        if not video:
            raise FileNotFoundError(f"未找到视频译配任务: {video_id}")
        vdir = VIDEOS_DIR / video_id
        zip_path = vdir / f"voxpoly_video_{video_id}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in video.get("outputs", []):
                if item.get("status") != "completed":
                    continue
                code = item["lang_code"]
                for fname, arc in (
                    (f"dub_{code}.mp4", f"{code}/dubbed_{code}.mp4"),
                    (f"dub_{code}.mp3", f"{code}/dubbed_{code}.mp3"),
                    (f"sub_{code}.srt", f"{code}/subtitle_{code}.srt"),
                    (f"sub_{code}_bilingual.srt", f"{code}/subtitle_{code}_bilingual.srt"),
                    (f"sub_{code}.ass", f"{code}/subtitle_{code}.ass"),
                ):
                    p = vdir / fname
                    if p.exists():
                        zf.write(p, arcname=arc)
            zf.writestr("video_manifest.json", json.dumps(video, ensure_ascii=False, indent=2).encode("utf-8"))
        return zip_path

    # ---------- Multi-Speaker Podcasts ----------
    def list_podcasts(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return self._read_json(PODCASTS_INDEX, [])[:limit]

    def get_podcast(self, podcast_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return next((p for p in self._read_json(PODCASTS_INDEX, []) if p["id"] == podcast_id), None)

    def save_podcast(self, podcast: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            items = self._read_json(PODCASTS_INDEX, [])
            for idx, existing in enumerate(items):
                if existing["id"] == podcast["id"]:
                    items[idx] = podcast
                    break
            else:
                items.insert(0, podcast)
            self._write_json(PODCASTS_INDEX, items[:50])
            return podcast

    def delete_podcast(self, podcast_id: str) -> bool:
        with self._lock:
            items = self._read_json(PODCASTS_INDEX, [])
            new_items = [p for p in items if p["id"] != podcast_id]
            if len(new_items) == len(items):
                return False
            self._write_json(PODCASTS_INDEX, new_items)
            shutil.rmtree(PODCASTS_DIR / podcast_id, ignore_errors=True)
            return True

    def build_podcast_zip(self, podcast_id: str) -> Path:
        pod = self.get_podcast(podcast_id)
        if not pod:
            raise FileNotFoundError(f"未找到播客任务: {podcast_id}")
        pdir = PODCASTS_DIR / podcast_id
        zip_path = pdir / f"voxpoly_podcast_{podcast_id}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for tr in pod.get("tracks", []):
                if tr.get("status") != "completed":
                    continue
                code = tr["lang_code"]
                for ext in ("mp3", "wav"):
                    f = pdir / f"podcast_{code}.{ext}"
                    if f.exists():
                        zf.write(f, arcname=f"podcast_{code}.{ext}")
            zf.writestr("dialogue_script.txt", str(pod.get("script_text") or "").encode("utf-8"))
            zf.writestr("podcast_manifest.json", json.dumps(pod, ensure_ascii=False, indent=2).encode("utf-8"))
        return zip_path


store = StudioStore()
