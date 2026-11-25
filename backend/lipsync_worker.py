"""Optional Isolated GPU Worker for AI Lip-Sync (MuseTalk / Wav2Lip) on Colab L4/A100.

Runs only when explicitly installed in an isolated environment (/content/lipsync_env)
so local lightweight deployments remain completely free of heavy vision dependencies.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def run_musetalk(repo_dir: Path, video_path: Path, audio_path: Path, out_mp4: Path) -> None:
    """Invoke MuseTalk inference script via temporary YAML config."""
    with tempfile.TemporaryDirectory(prefix="voxpoly_mt_") as tmp:
        cfg_path = Path(tmp) / "task.yaml"
        out_dir = Path(tmp) / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(
            f"task_0:\n  video_path: \"{video_path}\"\n  audio_path: \"{audio_path}\"\n",
            encoding="utf-8",
        )
        cmd = [
            sys.executable,
            "-m",
            "scripts.inference",
            "--inference_config",
            str(cfg_path),
            "--result_dir",
            str(out_dir),
            "--use_float16",
        ]
        proc = subprocess.run(cmd, cwd=str(repo_dir), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"MuseTalk 推理失败: {(proc.stderr or proc.stdout)[-800:]}")
        produced = list(out_dir.rglob("*.mp4"))
        if not produced:
            raise RuntimeError("MuseTalk 未生成输出 MP4 文件。")
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(produced[0].read_bytes())


def run_wav2lip(repo_dir: Path, checkpoint: Path, video_path: Path, audio_path: Path, out_mp4: Path) -> None:
    cmd = [
        sys.executable,
        str(repo_dir / "inference.py"),
        "--checkpoint_path",
        str(checkpoint),
        "--face",
        str(video_path),
        "--audio",
        str(audio_path),
        "--outfile",
        str(out_mp4),
        "--nosmooth",
    ]
    proc = subprocess.run(cmd, cwd=str(repo_dir), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not out_mp4.exists():
        raise RuntimeError(f"Wav2Lip 推理失败: {(proc.stderr or proc.stdout)[-800:]}")


def main() -> None:
    repo_dir = Path(os.getenv("VOXPOLY_LIPSYNC_REPO", "/content/MuseTalk"))
    backend_kind = os.getenv("VOXPOLY_LIPSYNC_BACKEND", "musetalk").strip().lower()
    wav2lip_ckpt = Path(os.getenv("VOXPOLY_WAV2LIP_CKPT", str(repo_dir / "checkpoints" / "wav2lip_gan.pth")))

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "ping":
                emit({"ok": True, "backend": backend_kind, "repo_exists": repo_dir.exists()})
            elif cmd == "sync":
                t0 = time.perf_counter()
                v_in = Path(req["video_path"])
                a_in = Path(req["audio_path"])
                v_out = Path(req["out_mp4_path"])
                if backend_kind == "wav2lip":
                    run_wav2lip(repo_dir, wav2lip_ckpt, v_in, a_in, v_out)
                else:
                    run_musetalk(repo_dir, v_in, a_in, v_out)
                emit({"ok": True, "elapsed_sec": round(time.perf_counter() - t0, 2)})
            elif cmd == "quit":
                emit({"ok": True, "bye": True})
                break
            else:
                emit({"ok": False, "error": f"unknown_cmd: {cmd}"})
        except Exception as exc:
            emit({"ok": False, "error": str(exc), "traceback": traceback.format_exc()[-1000:]})


if __name__ == "__main__":
    main()
