from __future__ import annotations

import gc
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from backend.config import WHISPER_DEVICE, WHISPER_DIR, WHISPER_MODEL_SIZE


class ASRService:
    """Colab-optimized Whisper ASR: supports full transcript AND timestamped video dubbing segments."""

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE) -> None:
        self.model_size = model_size
        self._lock = threading.Lock()
        self._model = None
        self._device: str = "unloaded"
        self._active_model_size: str = model_size
        self._load_error: Optional[str] = None

    @property
    def status(self) -> Dict[str, object]:
        return {
            "loaded": self._model is not None,
            "model_size": self._active_model_size,
            "configured_model_size": self.model_size,
            "device": self._device,
            "error": self._load_error,
        }

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel

            WHISPER_DIR.mkdir(parents=True, exist_ok=True)
            use_cuda = False
            if WHISPER_DEVICE in ("auto", "cuda"):
                try:
                    import torch

                    use_cuda = torch.cuda.is_available()
                except Exception:
                    use_cuda = False

            if use_cuda:
                try:
                    self._model = WhisperModel(
                        self.model_size,
                        device="cuda",
                        compute_type="float16",
                        download_root=str(WHISPER_DIR),
                    )
                    self._device = "cuda (float16)"
                    self._active_model_size = self.model_size
                    self._load_error = None
                    return
                except Exception:
                    pass

            fallback_size = "small" if "large" in self.model_size and WHISPER_DEVICE != "cpu" else self.model_size
            try:
                self._model = WhisperModel(
                    fallback_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(WHISPER_DIR),
                    cpu_threads=4,
                )
                self._device = "cpu (int8)"
                self._active_model_size = fallback_size
                self._load_error = None
            except Exception as exc:
                self._load_error = str(exc)
                raise

    def transcribe_segments(self, wav_path: Path) -> Dict[str, object]:
        """Transcribe audio and return both merged text and clean timestamped segments."""
        t0 = time.perf_counter()
        self._ensure_loaded()
        import librosa
        from backend.audio_utils import load_wav_float32

        audio, sr = load_wav_float32(wav_path)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype("float32")

        with self._lock:
            raw_segs, info = self._model.transcribe(
                audio,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 380},
            )
            seg_list: List[Dict[str, object]] = []
            for s in raw_segs:
                txt = (s.text or "").strip()
                if not txt:
                    continue
                start = round(float(s.start), 2)
                end = round(float(s.end), 2)
                if end <= start:
                    end = round(start + 0.4, 2)
                # Merge ultra-short fragments (< 0.8s) into previous segment if gap is tiny (< 0.35s)
                if (
                    seg_list
                    and (end - start) < 0.8
                    and (start - float(seg_list[-1]["end"])) < 0.35
                    and (end - float(seg_list[-1]["start"])) < 10.0
                ):
                    joiner = "" if getattr(info, "language", "") in ("zh", "ja") else " "
                    seg_list[-1]["text"] = f"{seg_list[-1]['text']}{joiner}{txt}".strip()
                    seg_list[-1]["end"] = end
                    seg_list[-1]["duration_sec"] = round(end - float(seg_list[-1]["start"]), 2)
                else:
                    seg_list.append(
                        {
                            "index": len(seg_list),
                            "start": start,
                            "end": end,
                            "duration_sec": round(end - start, 2),
                            "text": txt,
                        }
                    )

        lang = getattr(info, "language", "unknown") or "unknown"
        joiner = "" if lang in ("zh", "ja") else " "
        full_text = joiner.join(str(s["text"]) for s in seg_list).strip()
        elapsed = round(time.perf_counter() - t0, 2)

        return {
            "text": full_text,
            "segments": seg_list,
            "language": lang,
            "language_prob": round(float(getattr(info, "language_probability", 0.0) or 0.0), 3),
            "duration_sec": round(float(getattr(info, "duration", 0.0) or 0.0), 2),
            "elapsed_sec": elapsed,
            "device": self._device,
            "model_size": self._active_model_size,
        }

    def transcribe(self, wav_path: Path) -> Dict[str, object]:
        return self.transcribe_segments(wav_path)

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._device = "unloaded"
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


asr_service = ASRService()
