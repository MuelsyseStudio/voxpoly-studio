"""VoxPoly Studio V3 CLI — Batch multilingual voice cloning, podcasts, audiobooks & video dubbing.

Examples:
  python -m backend.cli voices
  python -m backend.cli speak --voice demo_official --text "Hello world" --langs zh,en,ja --out ./out
  python -m backend.cli batch ./articles --voice demo_official --langs zh,en --mastering audiobook --out ./books
  python -m backend.cli dub --input ./demo.mp4 --langs zh,ja --subtitle-style shorts_yellow --out ./dubbed
  python -m backend.cli verify ./out/zh.mp3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.audio_utils import load_wav_float32, save_generated_audio
from backend.config import LANGUAGE_BY_CODE, VOICES_DIR
from backend.document_parser import parse_document_bytes
from backend.glossary import apply_tts_pronunciation
from backend.mastering import apply_mastering_dsp
from backend.store import store
from backend.text_chunker import split_text_for_tts, stitch_chunk_audios
from backend.translator import translate_scripts
from backend.tts_engine import engine_manager
from backend.watermark import detect_watermark, embed_watermark


def _cmd_voices(_args: argparse.Namespace) -> int:
    voices = store.list_voices()
    if not voices:
        print("暂无已保存声纹。请先在 Web 工作台上传一段参考录音。")
        return 0
    for v in voices:
        print(f"{v['id']:<16} | {v['name']:<24} | {v['duration_sec']}s | {v.get('quality_label', '')}")
    return 0


def _synthesize_file(
    voice: dict,
    text: str,
    langs: list[str],
    out_dir: Path,
    stem: str,
    engine_id: str,
    mode: str,
    mastering: str,
    watermark: bool,
) -> None:
    settings = store.get_settings(masked=False)
    translations, provider, _ = translate_scripts(text, langs, settings)
    print(f"  [翻译完成: {provider}] -> {', '.join(translations.keys())}")
    eng = engine_manager.prepare_engine(engine_id)
    ref_wav = VOICES_DIR / voice["id"] / "clone_ref.wav"
    ref_text = str(voice.get("ref_text") or "")

    out_dir.mkdir(parents=True, exist_ok=True)
    for code, script in translations.items():
        lang_meta = LANGUAGE_BY_CODE[code]
        spoken = apply_tts_pronunciation(script, code)
        chunks = split_text_for_tts(spoken, code)
        if engine_id == "qwen3" and len(chunks) > 1:
            wavs, sr, elapsed, _ = eng.synthesize_batch(
                voice_id=voice["id"],
                ref_wav_path=ref_wav,
                ref_text=ref_text,
                mode=mode,
                texts=[str(c["text"]) for c in chunks],
                lang_meta=lang_meta,
            )
            wav = stitch_chunk_audios([(w, sr, float(c["pause_after_sec"])) for c, w in zip(chunks, wavs)], sr)
        else:
            pieces = []
            sr = 24000
            elapsed = 0.0
            for c in chunks:
                w, sr, e, _ = eng.synthesize(
                    voice_id=voice["id"],
                    ref_wav_path=ref_wav,
                    ref_text=ref_text,
                    mode=mode,
                    text=str(c["text"]),
                    lang_meta=lang_meta,
                )
                elapsed += e
                pieces.append((w, sr, float(c["pause_after_sec"])))
            wav = pieces[0][0] if len(pieces) == 1 else stitch_chunk_audios(pieces, sr)

        wav = apply_mastering_dsp(wav, sr, mastering)
        if watermark:
            wav = embed_watermark(wav, sr)
        m = save_generated_audio(wav, sr, out_dir / f"{stem}_{code}.wav", out_dir / f"{stem}_{code}.mp3")
        print(f"  ✅ [{code.upper()}] {out_dir / f'{stem}_{code}.mp3'} ({m['duration_sec']}s, {elapsed:.1f}s)")


def _cmd_speak(args: argparse.Namespace) -> int:
    voice = store.get_voice(args.voice)
    if not voice:
        print(f"错误: 未找到声纹 ID '{args.voice}'", file=sys.stderr)
        return 1
    langs = [x.strip() for x in args.langs.split(",") if x.strip() in LANGUAGE_BY_CODE]
    _synthesize_file(voice, args.text, langs, Path(args.out), args.name, args.engine, args.mode, args.mastering, not args.no_watermark)
    engine_manager.unload_all()
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    voice = store.get_voice(args.voice)
    if not voice:
        print(f"错误: 未找到声纹 ID '{args.voice}'", file=sys.stderr)
        return 1
    src_dir = Path(args.folder)
    files = sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in (".txt", ".md", ".pdf", ".epub", ".html"))
    if not files:
        print(f"目录 {src_dir} 中未找到支持的文档文件 (.txt/.md/.pdf/.epub/.html)")
        return 1
    langs = [x.strip() for x in args.langs.split(",") if x.strip() in LANGUAGE_BY_CODE]
    for f in files:
        print(f"\n📄 正在处理: {f.name}")
        doc = parse_document_bytes(f.read_bytes(), f.name)
        _synthesize_file(voice, doc["full_text"], langs, Path(args.out) / f.stem, f.stem, args.engine, args.mode, args.mastering, not args.no_watermark)
    engine_manager.unload_all()
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    p = Path(args.audio)
    if not p.exists():
        print(f"文件不存在: {p}", file=sys.stderr)
        return 1
    import tempfile
    from backend.audio_utils import _run_ffmpeg

    with tempfile.TemporaryDirectory() as tmp:
        wav_p = Path(tmp) / "check.wav"
        _run_ffmpeg(["-i", str(p), "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(wav_p)])
        audio, sr = load_wav_float32(wav_p)
        res = detect_watermark(audio, sr)
        print(json.dumps({"file": str(p), **res}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voxpoly-cli", description="VoxPoly Studio V3 批量命令行工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("voices", help="列出全部声纹档案").set_defaults(func=_cmd_voices)

    sp = sub.add_parser("speak", help="合成单段或多语种语音")
    sp.add_argument("--voice", required=True, help="声纹 ID")
    sp.add_argument("--text", required=True, help="待合成文本")
    sp.add_argument("--langs", default="zh,en", help="逗号分隔语种列表，如 zh,en,ja")
    sp.add_argument("--engine", default="qwen3", choices=["qwen3", "chatterbox", "cosyvoice3"])
    sp.add_argument("--mode", default="icl", choices=["icl", "xvector"])
    sp.add_argument("--mastering", default="podcast", choices=["raw", "podcast", "shorts", "audiobook"])
    sp.add_argument("--no-watermark", action="store_true")
    sp.add_argument("--name", default="output")
    sp.add_argument("--out", default="./cli_out")
    sp.set_defaults(func=_cmd_speak)

    bp = sub.add_parser("batch", help="批量把文件夹内文档转为多语种有声书/语音")
    bp.add_argument("folder", help="包含 .txt/.md/.pdf/.epub 的目录")
    bp.add_argument("--voice", required=True)
    bp.add_argument("--langs", default="zh,en")
    bp.add_argument("--engine", default="qwen3", choices=["qwen3", "chatterbox", "cosyvoice3"])
    bp.add_argument("--mode", default="icl", choices=["icl", "xvector"])
    bp.add_argument("--mastering", default="audiobook", choices=["raw", "podcast", "shorts", "audiobook"])
    bp.add_argument("--no-watermark", action="store_true")
    bp.add_argument("--out", default="./cli_batch_out")
    bp.set_defaults(func=_cmd_batch)

    vp = sub.add_parser("verify", help="检测音视频文件是否含 VoxPoly AI 隐形水印")
    vp.add_argument("audio", help="音视频文件路径")
    vp.set_defaults(func=_cmd_verify)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
