from __future__ import annotations

import gc
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from backend.audio_utils import load_wav_float32
from backend.config import (
    BASE_DIR,
    CHATTERBOX_PYTHON,
    CHATTERBOX_T3_MODEL,
    COSYVOICE_LOCAL_DIR,
    COSYVOICE_MODEL_ID,
    COSYVOICE_PYTHON,
    COSYVOICE_REPO_DIR,
    DEFAULT_QWEN_VARIANT,
    DOWNLOAD_SOURCE,
    LANGUAGES,
    MODELS_DIR,
    QWEN_BATCH_SIZE,
    QWEN_LOCAL_DIR,
    QWEN_MODEL_ID,
    QWEN_VARIANTS,
    TORCH_COMPILE_ENABLED,
)


def _detect_gpu_info() -> Dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info(0)
            alloc_b = torch.cuda.memory_allocated(0)
            cc_major, cc_minor = torch.cuda.get_device_capability(0)
            is_rocm = bool(getattr(torch.version, "hip", None))
            supports_bf16 = (cc_major >= 8) and (not is_rocm)
            return {
                "available": True,
                "backend": "rocm" if is_rocm else "cuda",
                "name": torch.cuda.get_device_name(0),
                "compute_capability": f"{cc_major}.{cc_minor}",
                "recommended_precision": "fp16 (ROCm)" if is_rocm else ("bf16" if supports_bf16 else "fp16"),
                "supports_bf16": supports_bf16,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.hip if is_rocm else torch.version.cuda,
                "vram_total_gb": round(total_b / (1024**3), 2),
                "vram_free_gb": round(free_b / (1024**3), 2),
                "vram_used_gb": round((total_b - free_b) / (1024**3), 2),
                "vram_model_gb": round(alloc_b / (1024**3), 2),
            }
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            alloc_b = 0
            try:
                alloc_b = int(torch.mps.current_allocated_memory())
            except Exception:
                alloc_b = 0
            return {
                "available": True,
                "backend": "mps",
                "name": "Apple Silicon (MPS)",
                "compute_capability": "mps",
                "recommended_precision": "fp16 (MPS)",
                "supports_bf16": False,
                "torch_version": torch.__version__,
                "cuda_version": "mps",
                "vram_total_gb": 0,
                "vram_free_gb": 0,
                "vram_used_gb": round(alloc_b / (1024**3), 2),
                "vram_model_gb": round(alloc_b / (1024**3), 2),
            }
        return {
            "available": False,
            "backend": "cpu",
            "name": "CPU Only",
            "compute_capability": None,
            "recommended_precision": "fp32",
            "supports_bf16": False,
            "torch_version": torch.__version__,
            "cuda_version": None,
            "vram_total_gb": 0,
            "vram_free_gb": 0,
            "vram_used_gb": 0,
            "vram_model_gb": 0,
        }
    except Exception as exc:
        return {
            "available": False,
            "backend": "cpu",
            "name": f"检测异常: {exc}",
            "compute_capability": None,
            "recommended_precision": "fp32",
            "supports_bf16": False,
            "torch_version": "unknown",
            "cuda_version": None,
            "vram_total_gb": 0,
            "vram_free_gb": 0,
            "vram_used_gb": 0,
            "vram_model_gb": 0,
        }


class BaseVoiceCloneEngine(ABC):
    engine_id: str
    display_name: str
    license_name: str

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def load_model(self) -> None:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...


class Qwen3TTSEngine(BaseVoiceCloneEngine):
    engine_id = "qwen3"
    display_name = "Qwen3-TTS 12Hz Base"
    license_name = "Apache-2.0"

    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._model = None
        self._variant: str = DEFAULT_QWEN_VARIANT
        self._state: str = "unloaded"
        self._state_detail: str = "模型未加载（首次生成时自动装载）"
        self._last_error: Optional[str] = None
        self._load_time_sec: Optional[float] = None
        self._precision_label: str = "auto"
        self._attn_label: str = "sdpa"
        self._compiled: bool = False
        self._prompt_cache: "OrderedDict[str, Any]" = OrderedDict()
        self._max_cached_prompts = 16

    @property
    def active_model_id(self) -> str:
        if self._variant == DEFAULT_QWEN_VARIANT:
            return QWEN_MODEL_ID
        return str(QWEN_VARIANTS[self._variant]["model_id"])

    @property
    def active_local_dir(self) -> Path:
        if self._variant == DEFAULT_QWEN_VARIANT:
            return QWEN_LOCAL_DIR
        return Path(QWEN_VARIANTS[self._variant]["local_dir"]).resolve()

    def set_variant(self, variant: str) -> None:
        v = variant.strip().lower()
        if v not in QWEN_VARIANTS or v == self._variant:
            return
        self.unload()
        self._variant = v
        self.display_name = f"Qwen3-TTS 12Hz {v.upper()} Base"

    def is_available(self) -> bool:
        return importlib.util.find_spec("qwen_tts") is not None

    def has_local_weights(self) -> bool:
        d = self.active_local_dir
        if not d.exists():
            return False
        return (d / "config.json").exists() and (any(d.glob("*.safetensors")) or any(d.glob("*.bin")))

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "display_name": f"Qwen3-TTS {self._variant.upper()} Base",
            "variant": self._variant,
            "model_id": self.active_model_id,
            "license": self.license_name,
            "installed": self.is_available(),
            "state": self._state,
            "state_detail": self._state_detail,
            "last_error": self._last_error,
            "precision": self._precision_label,
            "attn": self._attn_label,
            "compiled": self._compiled,
            "batch_size": QWEN_BATCH_SIZE,
            "weights_downloaded": self.has_local_weights(),
            "load_time_sec": self._load_time_sec,
            "cached_prompts": len(self._prompt_cache),
            "supported_languages_count": sum(1 for l in LANGUAGES if "qwen3" in l["engines"]),
        }

    def invalidate_voice_cache(self, voice_id: str) -> None:
        with self._infer_lock:
            for k in [k for k in self._prompt_cache if k.startswith(f"{voice_id}|")]:
                self._prompt_cache.pop(k, None)

    def ensure_weights_downloaded(self) -> Path:
        target_dir = self.active_local_dir
        target_id = self.active_model_id
        if self.has_local_weights():
            return target_dir
        self._state = "downloading"
        target_dir.mkdir(parents=True, exist_ok=True)
        order = ("hf", "modelscope") if DOWNLOAD_SOURCE == "hf" else ("modelscope", "hf")
        last_exc: Optional[Exception] = None
        for src in order:
            try:
                if src == "hf":
                    self._state_detail = f"正在从 HuggingFace 下载 {target_id}..."
                    from huggingface_hub import snapshot_download as hf_download

                    hf_download(repo_id=target_id, local_dir=str(target_dir))
                else:
                    self._state_detail = f"正在从 ModelScope 下载 {target_id}..."
                    from modelscope.hub.snapshot_download import snapshot_download as ms_download

                    ms_download(model_id=target_id, local_dir=str(target_dir))
                if self.has_local_weights():
                    return target_dir
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"未能完整下载 {target_id} 权重至 {target_dir}: {last_exc}")

    def load_model(self) -> None:
        if self._model is not None and self._state == "ready":
            return
        with self._load_lock:
            if self._model is not None and self._state == "ready":
                return
            t0 = time.perf_counter()
            self._last_error = None
            try:
                model_path = self.ensure_weights_downloaded()
                self._state = "loading"
                self._state_detail = f"正在装载 Qwen3-TTS {self._variant.upper()} Base..."

                import torch
                from qwen_tts import Qwen3TTSModel

                use_cuda = torch.cuda.is_available()
                use_mps = (not use_cuda) and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

                if use_cuda:
                    cc_major, _ = torch.cuda.get_device_capability(0)
                    is_rocm = bool(getattr(torch.version, "hip", None))
                    if cc_major >= 8 and not is_rocm:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                        torch.set_float32_matmul_precision("high")
                        dtype = torch.bfloat16
                        self._precision_label = "bf16+TF32"
                    else:
                        dtype = torch.float16
                        self._precision_label = "fp16 (ROCm)" if is_rocm else "fp16"
                    device_map = "cuda:0"
                    has_fa2 = (not is_rocm) and importlib.util.find_spec("flash_attn") is not None and cc_major >= 8
                    self._attn_label = "flash_attention_2" if has_fa2 else "sdpa"
                elif use_mps:
                    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
                    dtype = torch.float16
                    self._precision_label = "fp16 (Apple MPS)"
                    device_map = "mps"
                    self._attn_label = "sdpa"
                else:
                    dtype = torch.float32
                    self._precision_label = "fp32 (CPU)"
                    device_map = "cpu"
                    self._attn_label = "sdpa"

                sig = inspect.signature(Qwen3TTSModel.from_pretrained)
                kwargs: Dict[str, Any] = {"device_map": device_map}
                if use_cuda or use_mps:
                    kwargs["attn_implementation"] = self._attn_label

                if "dtype" in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                ):
                    kwargs["dtype"] = dtype
                else:
                    kwargs["torch_dtype"] = dtype

                try:
                    self._model = Qwen3TTSModel.from_pretrained(str(model_path), **kwargs)
                except TypeError:
                    kwargs.pop("dtype", None)
                    kwargs["torch_dtype"] = dtype
                    self._model = Qwen3TTSModel.from_pretrained(str(model_path), **kwargs)
                except RuntimeError:
                    if use_mps and dtype == torch.float16:
                        # Clean fallback to float32 on older macOS MPS builds
                        kwargs["dtype"] = torch.float32
                        kwargs.pop("torch_dtype", None)
                        self._precision_label = "fp32 (Apple MPS)"
                        self._model = Qwen3TTSModel.from_pretrained(str(model_path), **kwargs)
                    else:
                        raise

                self._compiled = False
                if TORCH_COMPILE_ENABLED and use_cuda and sys.platform != "win32":
                    try:
                        if hasattr(self._model, "model") and hasattr(self._model.model, "talker"):
                            self._model.model.talker = torch.compile(self._model.model.talker, mode="reduce-overhead")
                            self._compiled = True
                    except Exception:
                        self._compiled = False

                self._load_time_sec = round(time.perf_counter() - t0, 2)
                self._state = "ready"
                self._state_detail = f"就绪 · {self._precision_label} · {self._attn_label} ({self._load_time_sec}s)"
            except Exception as exc:
                self._state = "error"
                self._last_error = str(exc)
                self._state_detail = f"模型加载失败: {exc}"
                raise

    def _get_or_create_prompt(self, voice_id: str, ref_wav_path: Path, ref_text: str, mode: str) -> Tuple[Any, bool]:
        clean_ref = (ref_text or "").strip()
        x_vector_only = (mode == "xvector") or (not clean_ref)
        text_hash = hashlib.sha1(clean_ref.encode("utf-8")).hexdigest()[:12] if not x_vector_only else "xvec"
        mtime = int(ref_wav_path.stat().st_mtime) if ref_wav_path.exists() else 0
        cache_key = f"{voice_id}|{x_vector_only}|{text_hash}|{mtime}"

        if cache_key in self._prompt_cache:
            self._prompt_cache.move_to_end(cache_key)
            return self._prompt_cache[cache_key], x_vector_only

        audio_np, sr = load_wav_float32(ref_wav_path)
        prompt_kwargs: Dict[str, Any] = {
            "ref_audio": (audio_np, sr),
            "x_vector_only_mode": x_vector_only,
        }
        if not x_vector_only:
            prompt_kwargs["ref_text"] = clean_ref

        prompt = self._model.create_voice_clone_prompt(**prompt_kwargs)
        self._prompt_cache[cache_key] = prompt
        while len(self._prompt_cache) > self._max_cached_prompts:
            self._prompt_cache.popitem(last=False)
        return prompt, x_vector_only

    def synthesize(
        self,
        voice_id: str,
        ref_wav_path: Path,
        ref_text: str,
        mode: str,
        text: str,
        lang_meta: Dict[str, Any],
        **kwargs,
    ) -> Tuple[np.ndarray, int, float, float]:
        wavs, sr, elapsed, rtf = self.synthesize_batch(
            voice_id=voice_id,
            ref_wav_path=ref_wav_path,
            ref_text=ref_text,
            mode=mode,
            texts=[text],
            lang_meta=lang_meta,
            batch_size=1,
        )
        return wavs[0], sr, elapsed, rtf

    def synthesize_batch(
        self,
        voice_id: str,
        ref_wav_path: Path,
        ref_text: str,
        mode: str,
        texts: List[str],
        lang_meta: Dict[str, Any],
        batch_size: int = QWEN_BATCH_SIZE,
        on_chunk_done: Optional[Callable[[int, np.ndarray, int], None]] = None,
    ) -> Tuple[List[np.ndarray], int, float, float]:
        """True batched inference for long-form chunks — ~1.8-2.4x faster than sequential calls."""
        self.load_model()
        qwen_language = lang_meta.get("qwen_name")
        if not qwen_language:
            raise ValueError(f"Qwen3-TTS 暂不支持 {lang_meta['label_zh']}，请切换到 Chatterbox V3（23 语种）。")

        clean_texts = [t.strip() for t in texts if t and t.strip()]
        if not clean_texts:
            raise ValueError(f"{lang_meta['label_zh']} 合成文本不能为空。")

        with self._infer_lock:
            import torch

            t0 = time.perf_counter()
            prompt, _ = self._get_or_create_prompt(voice_id, ref_wav_path, ref_text, mode)
            all_wavs: List[np.ndarray] = []
            out_sr = 24000
            bs = max(1, min(batch_size, 4))

            idx = 0
            while idx < len(clean_texts):
                sub = clean_texts[idx : idx + bs]
                try:
                    with torch.inference_mode():
                        if len(sub) == 1:
                            wavs, sr = self._model.generate_voice_clone(
                                text=sub[0],
                                language=str(qwen_language),
                                voice_clone_prompt=prompt,
                            )
                        else:
                            wavs, sr = self._model.generate_voice_clone(
                                text=sub,
                                language=[str(qwen_language)] * len(sub),
                                voice_clone_prompt=prompt,
                            )
                    out_sr = int(sr)
                    wav_list = list(wavs) if isinstance(wavs, (list, tuple)) else [wavs]
                    for offset, w in enumerate(wav_list):
                        arr = np.asarray(w, dtype=np.float32).squeeze()
                        all_wavs.append(arr)
                        if on_chunk_done:
                            on_chunk_done(idx + offset, arr, out_sr)
                except RuntimeError as oom_exc:
                    if bs > 1 and "out of memory" in str(oom_exc).lower():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        bs = 1
                        continue
                    raise
                idx += len(sub)

            elapsed = round(time.perf_counter() - t0, 2)
            total_dur = sum(float(w.size / out_sr) for w in all_wavs) if out_sr > 0 else 0.0
            rtf = round(elapsed / total_dur, 2) if total_dur > 0.05 else 0.0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return all_wavs, out_sr, elapsed, rtf

    def unload(self) -> None:
        with self._load_lock:
            with self._infer_lock:
                self._prompt_cache.clear()
                if self._model is not None:
                    self._model = None
                    gc.collect()
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                    except Exception:
                        pass
                self._state = "unloaded"
                self._state_detail = "显存已释放（按需自动加载）"


class _SubprocessWorkerEngine(BaseVoiceCloneEngine):
    """Base class for isolated subprocess GPU workers (Chatterbox V3 & CosyVoice 3)."""

    worker_module: str = ""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._state: str = "unloaded"
        self._state_detail: str = "隔离环境扩展引擎（按需装载）"
        self._last_error: Optional[str] = None
        self._load_time_sec: Optional[float] = None
        self._resolved_python: Optional[str] = None

    @abstractmethod
    def _find_python(self) -> Optional[str]:
        ...

    @abstractmethod
    def _extra_env(self) -> Dict[str, str]:
        ...

    def is_available(self) -> bool:
        return self._find_python() is not None

    def _ensure_worker(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        py = self._find_python()
        if not py:
            raise RuntimeError(f"尚未安装 {self.display_name}，请在 Colab Cell 2 勾选启用。")
        env = os.environ.copy()
        env.update(self._extra_env())
        env["PYTHONPATH"] = str(BASE_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self._proc = subprocess.Popen(
            [py, "-u", "-m", self.worker_module],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(BASE_DIR),
        )
        return self._proc

    def _rpc(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        proc = self._ensure_worker()
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        while True:
            line = proc.stdout.readline()
            if not line:
                stderr_tail = ""
                if proc.stderr:
                    try:
                        stderr_tail = proc.stderr.read()[-1200:]
                    except Exception:
                        pass
                self._proc = None
                raise RuntimeError(f"{self.display_name} 推理进程异常退出: {stderr_tail or 'no output'}")
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    continue

    def load_model(self) -> None:
        with self._lock:
            if self._state == "ready" and self._proc is not None and self._proc.poll() is None:
                return
            self._state = "loading"
            self._state_detail = f"正在启动 {self.display_name} 隔离进程并加载权重..."
            self._last_error = None
            try:
                resp = self._rpc({"cmd": "load"})
                if not resp.get("ok"):
                    raise RuntimeError(resp.get("error") or "未知加载错误")
                self._load_time_sec = resp.get("load_time_sec", self._load_time_sec)
                self._state = "ready"
                self._state_detail = f"就绪 · 隔离加速 ({self._load_time_sec or 0}s)"
            except Exception as exc:
                self._state = "error"
                self._last_error = str(exc)
                self._state_detail = f"加载失败: {exc}"
                self._kill_proc()
                raise

    def _kill_proc(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.poll() is None and self._proc.stdin:
                    self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=3)
            except Exception:
                pass
            try:
                if self._proc.poll() is None:
                    self._proc.terminate()
                    self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def unload(self) -> None:
        with self._lock:
            self._kill_proc()
            self._state = "unloaded"
            self._state_detail = "显存已释放（按需自动加载）"


class ChatterboxEngine(_SubprocessWorkerEngine):
    engine_id = "chatterbox"
    display_name = "Chatterbox Multilingual V3"
    license_name = "MIT"
    worker_module = "backend.chatterbox_worker"

    def _find_python(self) -> Optional[str]:
        if self._resolved_python and Path(self._resolved_python).exists():
            return self._resolved_python
        if CHATTERBOX_PYTHON and Path(CHATTERBOX_PYTHON).exists():
            self._resolved_python = CHATTERBOX_PYTHON
            return self._resolved_python
        if importlib.util.find_spec("chatterbox") is not None:
            self._resolved_python = sys.executable
            return self._resolved_python
        return None

    def _extra_env(self) -> Dict[str, str]:
        hf_cache = MODELS_DIR / "chatterbox_hf"
        hf_cache.mkdir(parents=True, exist_ok=True)
        return {"HF_HOME": str(hf_cache), "VOXPOLY_CHATTERBOX_T3": CHATTERBOX_T3_MODEL}

    def get_status(self) -> Dict[str, Any]:
        installed = self.is_available()
        return {
            "engine_id": self.engine_id,
            "display_name": self.display_name,
            "model_id": f"ResembleAI/chatterbox ({CHATTERBOX_T3_MODEL})",
            "license": self.license_name,
            "installed": installed,
            "state": self._state if installed else "unavailable",
            "state_detail": (
                self._state_detail if installed else "未启用（在 Colab Cell 2 勾选 ENABLE_CHATTERBOX_23LANG 即可启用）"
            ),
            "last_error": self._last_error,
            "precision": "mixed",
            "attn": "isolated-worker",
            "weights_downloaded": (MODELS_DIR / "chatterbox_hf").exists(),
            "load_time_sec": self._load_time_sec,
            "supported_languages_count": sum(1 for l in LANGUAGES if "chatterbox" in l["engines"]),
        }

    def synthesize(
        self,
        voice_id: str,
        ref_wav_path: Path,
        ref_text: str,
        mode: str,
        text: str,
        lang_meta: Dict[str, Any],
        exaggeration: float = 0.5,
        cfg_weight: float = 0.35,
        **kwargs,
    ) -> Tuple[np.ndarray, int, float, float]:
        self.load_model()
        lang_id = lang_meta.get("chatterbox_id")
        if not lang_id:
            raise ValueError(f"Chatterbox 暂不支持 {lang_meta['label_zh']}。")
        with self._lock:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_out = Path(tmp.name)
            try:
                resp = self._rpc(
                    {
                        "cmd": "synthesize",
                        "ref_wav_path": str(ref_wav_path),
                        "text": text.strip(),
                        "language_id": str(lang_id),
                        "out_wav_path": str(tmp_out),
                        "exaggeration": float(exaggeration),
                        "cfg_weight": float(cfg_weight),
                    }
                )
                if not resp.get("ok"):
                    raise RuntimeError(resp.get("error") or "Chatterbox 推理失败")
                wav_np, sr = load_wav_float32(tmp_out)
                return wav_np, sr, float(resp["elapsed_sec"]), float(resp["rtf"])
            finally:
                tmp_out.unlink(missing_ok=True)


class CosyVoice3Engine(_SubprocessWorkerEngine):
    engine_id = "cosyvoice3"
    display_name = "Fun-CosyVoice 3.0"
    license_name = "Apache-2.0"
    worker_module = "backend.cosyvoice_worker"

    def _find_python(self) -> Optional[str]:
        if self._resolved_python and Path(self._resolved_python).exists():
            return self._resolved_python
        if COSYVOICE_PYTHON and Path(COSYVOICE_PYTHON).exists() and Path(COSYVOICE_REPO_DIR).exists():
            self._resolved_python = COSYVOICE_PYTHON
            return self._resolved_python
        if importlib.util.find_spec("cosyvoice") is not None:
            self._resolved_python = sys.executable
            return self._resolved_python
        return None

    def _extra_env(self) -> Dict[str, str]:
        return {
            "VOXPOLY_COSYVOICE_REPO": COSYVOICE_REPO_DIR,
            "VOXPOLY_COSYVOICE_MODEL_ID": COSYVOICE_MODEL_ID,
            "VOXPOLY_COSYVOICE_LOCAL_DIR": str(COSYVOICE_LOCAL_DIR),
            "VOXPOLY_DOWNLOAD_SOURCE": DOWNLOAD_SOURCE,
        }

    def get_status(self) -> Dict[str, Any]:
        installed = self.is_available()
        return {
            "engine_id": self.engine_id,
            "display_name": self.display_name,
            "model_id": COSYVOICE_MODEL_ID,
            "license": self.license_name,
            "installed": installed,
            "state": self._state if installed else "unavailable",
            "state_detail": (
                self._state_detail if installed else "未启用（在 Colab Cell 2 勾选 ENABLE_COSYVOICE3 即可启用方言与情绪指令）"
            ),
            "last_error": self._last_error,
            "precision": "fp16/bf16",
            "attn": "isolated-worker",
            "weights_downloaded": COSYVOICE_LOCAL_DIR.exists(),
            "load_time_sec": self._load_time_sec,
            "supported_languages_count": sum(1 for l in LANGUAGES if "cosyvoice3" in l["engines"]),
        }

    def synthesize(
        self,
        voice_id: str,
        ref_wav_path: Path,
        ref_text: str,
        mode: str,
        text: str,
        lang_meta: Dict[str, Any],
        instruct: str = "",
        **kwargs,
    ) -> Tuple[np.ndarray, int, float, float]:
        self.load_model()
        if "cosyvoice3" not in lang_meta.get("engines", []):
            raise ValueError(f"CosyVoice 3 暂不支持 {lang_meta['label_zh']}。")
        with self._lock:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_out = Path(tmp.name)
            try:
                resp = self._rpc(
                    {
                        "cmd": "synthesize",
                        "ref_wav_path": str(ref_wav_path),
                        "ref_text": ref_text,
                        "text": text.strip(),
                        "instruct": (instruct or "").strip(),
                        "out_wav_path": str(tmp_out),
                    }
                )
                if not resp.get("ok"):
                    raise RuntimeError(resp.get("error") or "CosyVoice 3 推理失败")
                wav_np, sr = load_wav_float32(tmp_out)
                return wav_np, sr, float(resp["elapsed_sec"]), float(resp["rtf"])
            finally:
                tmp_out.unlink(missing_ok=True)


class EngineManager:
    """Manages all 3 engines (Qwen3-TTS, Chatterbox V3, CosyVoice 3) + idle VRAM watchdog."""

    def __init__(self) -> None:
        self.qwen3 = Qwen3TTSEngine()
        self.chatterbox = ChatterboxEngine()
        self.cosyvoice3 = CosyVoice3Engine()
        self._active_engine_id: str = "qwen3"
        self._switch_lock = threading.Lock()
        self.last_activity_ts: float = time.time()

    def touch(self) -> None:
        self.last_activity_ts = time.time()

    def get_engine(self, engine_id: str) -> BaseVoiceCloneEngine:
        if engine_id == "chatterbox":
            return self.chatterbox
        if engine_id == "cosyvoice3":
            return self.cosyvoice3
        return self.qwen3

    def prepare_engine(self, engine_id: str) -> BaseVoiceCloneEngine:
        self.touch()
        with self._switch_lock:
            target = engine_id if engine_id in ("qwen3", "chatterbox", "cosyvoice3") else "qwen3"
            for eid, eng in (("qwen3", self.qwen3), ("chatterbox", self.chatterbox), ("cosyvoice3", self.cosyvoice3)):
                if eid != target and eng.get_status()["state"] == "ready":
                    eng.unload()
            self._active_engine_id = target
            engine = self.get_engine(target)
            engine.load_model()
            self.touch()
            return engine

    def invalidate_voice_cache(self, voice_id: str) -> None:
        self.qwen3.invalidate_voice_cache(voice_id)

    def unload_all(self) -> None:
        with self._switch_lock:
            self.qwen3.unload()
            self.chatterbox.unload()
            self.cosyvoice3.unload()

    def has_any_loaded(self) -> bool:
        return any(
            e.get_status()["state"] == "ready" for e in (self.qwen3, self.chatterbox, self.cosyvoice3)
        )

    def get_status(self) -> Dict[str, Any]:
        statuses = {
            "qwen3": self.qwen3.get_status(),
            "chatterbox": self.chatterbox.get_status(),
            "cosyvoice3": self.cosyvoice3.get_status(),
        }
        active = statuses.get(self._active_engine_id, statuses["qwen3"])
        return {
            **active,
            "active_engine_id": self._active_engine_id,
            "engines": statuses,
            "gpu": _detect_gpu_info(),
        }


engine_manager = EngineManager()
