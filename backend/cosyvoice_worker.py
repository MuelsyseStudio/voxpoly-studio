"""Isolated GPU worker process for Fun-CosyVoice 3.0 (Fun-CosyVoice3-0.5B-2512).

Supports 9+ languages, 18+ Chinese dialects (Cantonese, Sichuanese, Dongbei, Shanghainese, etc.),
and natural-language emotion/speed/whisper instructions + [laughter]/[breath] paralinguistic tags.
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
    repo_dir = os.getenv("VOXPOLY_COSYVOICE_REPO", "/content/CosyVoice").strip()
    model_id = os.getenv("VOXPOLY_COSYVOICE_MODEL_ID", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512").strip()
    local_dir = Path(os.getenv("VOXPOLY_COSYVOICE_LOCAL_DIR", "/content/models/Fun-CosyVoice3-0.5B"))
    download_source = os.getenv("VOXPOLY_DOWNLOAD_SOURCE", "hf").strip().lower()

    if repo_dir and Path(repo_dir).exists():
        sys.path.insert(0, repo_dir)
        matcha = Path(repo_dir) / "third_party" / "Matcha-TTS"
        if matcha.exists():
            sys.path.insert(0, str(matcha))

    model = None

    def ensure_weights() -> Path:
        if local_dir.exists() and (any(local_dir.glob("*.yaml")) and (any(local_dir.glob("*.pt")) or any(local_dir.glob("*.safetensors")))):
            return local_dir
        local_dir.mkdir(parents=True, exist_ok=True)
        order = ("hf", "modelscope") if download_source == "hf" else ("modelscope", "hf")
        last_err = None
        for src in order:
            try:
                if src == "hf":
                    from huggingface_hub import snapshot_download as hf_dl

                    hf_dl(repo_id=model_id, local_dir=str(local_dir))
                else:
                    from modelscope import snapshot_download as ms_dl

                    ms_dl(model_id, local_dir=str(local_dir))
                if any(local_dir.glob("*.yaml")):
                    return local_dir
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"CosyVoice 3 权重下载失败: {last_err}")

    def ensure_loaded() -> dict:
        nonlocal model
        if model is not None:
            return {"ok": True, "already_loaded": True, "sr": int(getattr(model, "sample_rate", 24000))}
        from cosyvoice.cli.cosyvoice import AutoModel

        t0 = time.perf_counter()
        weights_dir = ensure_weights()
        model = AutoModel(model_dir=str(weights_dir))
        elapsed = round(time.perf_counter() - t0, 2)
        return {
            "ok": True,
            "loaded": True,
            "sr": int(getattr(model, "sample_rate", 24000)),
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
                ref_text = str(req.get("ref_text") or "").strip()
                text = str(req["text"]).strip()
                instruct = str(req.get("instruct") or "").strip()
                out_wav = str(req["out_wav_path"])

                t0 = time.perf_counter()
                if instruct:
                    # Official CosyVoice 3 instruct format
                    prompt_instruct = (
                        instruct
                        if "<|endofprompt|>" in instruct
                        else f"You are a helpful assistant. {instruct}<|endofprompt|>"
                    )
                    gen = model.inference_instruct2(text, prompt_instruct, ref_wav, stream=False)
                elif ref_text:
                    prompt_text = (
                        ref_text
                        if "<|endofprompt|>" in ref_text
                        else f"You are a helpful assistant.<|endofprompt|>{ref_text}"
                    )
                    gen = model.inference_zero_shot(text, prompt_text, ref_wav, stream=False)
                else:
                    cl_text = (
                        text
                        if "<|endofprompt|>" in text
                        else f"You are a helpful assistant.<|endofprompt|>{text}"
                    )
                    gen = model.inference_cross_lingual(cl_text, ref_wav, stream=False)

                pieces = []
                for item in gen:
                    speech = item["tts_speech"]
                    arr = speech.squeeze().detach().cpu().float().numpy()
                    if arr.size > 0:
                        pieces.append(arr)

                if not pieces:
                    raise RuntimeError("CosyVoice 3 未输出有效音频数据")
                wav_np = np.concatenate(pieces).astype(np.float32)
                wav_np = np.nan_to_num(wav_np, nan=0.0, posinf=0.0, neginf=0.0)
                sr = int(getattr(model, "sample_rate", 24000))
                Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
                sf.write(out_wav, wav_np, sr, subtype="PCM_16")

                elapsed = round(time.perf_counter() - t0, 2)
                dur = float(wav_np.size / sr) if sr > 0 else 0.0
                rtf = round(elapsed / dur, 2) if dur > 0.05 else 0.0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                emit({"ok": True, "sr": sr, "duration_sec": round(dur, 2), "elapsed_sec": elapsed, "rtf": rtf})
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
