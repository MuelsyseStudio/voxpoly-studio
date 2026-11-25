"""Isolated GPU worker process for Chatterbox Multilingual V3 (23 languages).

Runs inside a dedicated Python environment (e.g. /content/chatterbox_env) so its
transformers/diffusers pins never conflict with Qwen3-TTS, and terminating this
process guarantees 100% OS-level GPU VRAM reclamation.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    model = None
    last_cond_key = None
    t3_model = os.getenv("VOXPOLY_CHATTERBOX_T3", "v3").strip() or "v3"
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        Path(hf_home).mkdir(parents=True, exist_ok=True)

    def ensure_loaded() -> dict:
        nonlocal model
        if model is not None:
            return {"ok": True, "already_loaded": True, "sr": int(getattr(model, "sr", 24000))}
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        t0 = time.perf_counter()
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            device = "mps"
        else:
            device = "cpu"
        try:
            model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model=t3_model)
        except TypeError:
            # Backward compatibility if an older wheel without t3_model arg is installed
            model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        elapsed = round(time.perf_counter() - t0, 2)
        return {
            "ok": True,
            "loaded": True,
            "device": device,
            "t3_model": t3_model,
            "sr": int(getattr(model, "sr", 24000)),
            "load_time_sec": elapsed,
        }

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            emit({"ok": False, "error": f"invalid_json: {exc}"})
            continue

        cmd = req.get("cmd")
        try:
            if cmd == "ping":
                emit({"ok": True, "loaded": model is not None})
            elif cmd == "load":
                emit(ensure_loaded())
            elif cmd == "synthesize":
                ensure_loaded()
                import torch

                ref_wav = str(req["ref_wav_path"])
                text = str(req["text"]).strip()
                lang_id = str(req["language_id"]).strip().lower()
                out_wav = str(req["out_wav_path"])
                exaggeration = float(req.get("exaggeration", 0.5))
                cfg_weight = float(req.get("cfg_weight", 0.35))

                t0 = time.perf_counter()
                mtime = os.path.getmtime(ref_wav) if os.path.exists(ref_wav) else 0
                cond_key = f"{ref_wav}|{mtime}"
                if cond_key != last_cond_key:
                    model.prepare_conditionals(ref_wav, exaggeration=exaggeration)
                    last_cond_key = cond_key

                wav_tensor = model.generate(
                    text=text,
                    language_id=lang_id,
                    audio_prompt_path=None,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )
                wav_np = wav_tensor.squeeze().detach().cpu().float().numpy()
                wav_np = np.nan_to_num(wav_np, nan=0.0, posinf=0.0, neginf=0.0)
                sr = int(getattr(model, "sr", 24000))
                Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
                sf.write(out_wav, wav_np, sr, subtype="PCM_16")

                elapsed = round(time.perf_counter() - t0, 2)
                dur = float(wav_np.size / sr) if sr > 0 else 0.0
                rtf = round(elapsed / dur, 2) if dur > 0.05 else 0.0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                emit(
                    {
                        "ok": True,
                        "sr": sr,
                        "duration_sec": round(dur, 2),
                        "elapsed_sec": elapsed,
                        "rtf": rtf,
                    }
                )
            elif cmd == "quit":
                model = None
                gc.collect()
                emit({"ok": True, "bye": True})
                break
            else:
                emit({"ok": False, "error": f"unknown_cmd: {cmd}"})
        except Exception as exc:
            emit({"ok": False, "error": str(exc), "traceback": traceback.format_exc()[-1200:]})


if __name__ == "__main__":
    main()
