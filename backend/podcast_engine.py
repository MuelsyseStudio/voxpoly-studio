"""Multi-Speaker Dialogue & AI Two-Host Podcast Engine (NotebookLM-style script & audio synthesis)."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np

from backend.config import LANGUAGE_BY_CODE, QWEN_SAMPLE_RATE
from backend.translator import _call_llm_json, _clean_translated_text, translate_scripts


_SPEAKER_LINE_RE = re.compile(
    r"^\s*(?:\[([^\]]{1,24})\]|【([^】]{1,24})】|([A-Za-z\u4e00-\u9fff0-9_\-]{1,18}))\s*[:：]\s*(.+)$"
)


def parse_dialogue_script(script_text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse '[Host]: ...' or 'A: ...' dialogue script into ordered speaker turns + unique speaker list."""
    turns: List[Dict[str, Any]] = []
    speakers: List[str] = []
    last_speaker = "主持人"

    for raw_line in script_text.replace("\r\n", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _SPEAKER_LINE_RE.match(line)
        if m:
            spk = (m.group(1) or m.group(2) or m.group(3) or "").strip()
            body = m.group(4).strip()
            last_speaker = spk
        else:
            spk = last_speaker
            body = line

        if not body:
            continue
        if spk not in speakers:
            speakers.append(spk)
        turns.append({"index": len(turns), "speaker": spk, "text": body})

    return turns, speakers


def generate_two_host_podcast_script(
    source_text: str,
    settings: Dict[str, Any],
    host_a: str = "主持人",
    host_b: str = "嘉宾",
    target_lang: str = "zh",
    style: str = "deep_dive",
) -> Tuple[str, str]:
    """Convert any article/document into a lively 2-host podcast script (LLM when configured, smart local fallback otherwise)."""
    clean = source_text.strip()
    if len(clean) < 20:
        raise ValueError("请输入至少 20 字的文章内容后再生成双人播客剧本。")

    api_base = str(settings.get("llm_api_base") or "").strip()
    api_key = str(settings.get("llm_api_key") or "").strip()
    llm_model = str(settings.get("llm_model") or "deepseek-chat").strip()
    lang_label = str(LANGUAGE_BY_CODE.get(target_lang, {}).get("native_name") or "中文")

    if api_base and api_key:
        try:
            sys_p = (
                "You are an award-winning podcast producer (like NotebookLM Audio Overview). "
                f"Turn the source article into a natural, engaging, conversational two-host podcast dialogue in {lang_label}. "
                f"Use exactly two speaker labels: [{host_a}] and [{host_b}]. "
                "Create 8 to 14 punchy spoken turns (1-3 sentences each) covering the key insights, surprises, and practical takeaways. "
                "Return ONLY JSON: {\"turns\": [{\"speaker\": \"...\", \"text\": \"...\"}]}."
            )
            usr_p = f"Style: {style}\n\nSource material (truncated to 6000 chars):\n{clean[:6000]}"
            parsed = _call_llm_json(api_base, api_key, llm_model, sys_p, usr_p)
            turns = parsed.get("turns") if isinstance(parsed, dict) else None
            if isinstance(turns, list) and len(turns) >= 2:
                lines = [
                    f"[{str(t.get('speaker') or host_a).strip()}]: {_clean_translated_text(str(t.get('text') or ''))}"
                    for t in turns
                    if str(t.get("text") or "").strip()
                ]
                if lines:
                    return "\n".join(lines), f"LLM 播客编剧 ({llm_model})"
        except Exception:
            pass

    # Offline rule-based structured dialogue adapter (works 100% without API key!)
    paras = [p.strip() for p in re.split(r"\n\s*\n+", clean) if len(p.strip()) >= 12]
    if len(paras) == 1:
        # Split sentences
        sents = [s.strip() for s in re.split(r"(?<=[。！？.!?])\s*", paras[0]) if len(s.strip()) >= 6]
        paras = []
        buf = ""
        for s in sents:
            if len(buf) + len(s) > 110 and buf:
                paras.append(buf)
                buf = s
            else:
                buf = f"{buf}{s}" if buf else s
        if buf:
            paras.append(buf)

    paras = paras[:10]
    lines: List[str] = []
    if target_lang == "en":
        lines.append(f"[{host_a}]: Welcome back! Today we're diving into a really interesting piece.")
        lines.append(f"[{host_b}]: Let's jump right into the main points.")
    else:
        lines.append(f"[{host_a}]: 欢迎回到今天的声音实验室！我们刚刚读到一篇很有意思的内容，一起拆解一下核心观点。")
        lines.append(f"[{host_b}]: 没问题，咱们直接从最关键的部分聊起。")

    for idx, p in enumerate(paras):
        spk = host_a if idx % 2 == 0 else host_b
        lead = ""
        if idx > 0 and idx % 2 == 1:
            lead = "顺着这个点补充一下，" if target_lang == "zh" else "Building on that, "
        elif idx > 1 and idx % 2 == 0:
            lead = "还有一个很值得注意的地方：" if target_lang == "zh" else "Another really interesting point is: "
        lines.append(f"[{spk}]: {lead}{p[:260]}")

    if target_lang == "en":
        lines.append(f"[{host_a}]: That wraps up today's deep dive. Thanks for listening!")
    else:
        lines.append(f"[{host_a}]: 以上就是今天这期节目的核心拆解，感谢收听，我们下期见！")

    return "\n".join(lines), "本地智能对话改编（填入 LLM API Key 可获得 NotebookLM 级自由创作）"


def translate_dialogue_turns(
    turns: List[Dict[str, Any]],
    target_langs: List[str],
    settings: Dict[str, Any],
) -> Tuple[Dict[str, List[Dict[str, Any]]], str, List[str]]:
    """Translate all dialogue turns into each target language while preserving speaker labels."""
    by_lang: Dict[str, List[Dict[str, Any]]] = {code: [] for code in target_langs}
    provider_used = "原文直通"
    all_warnings: List[str] = []

    for turn in turns:
        tr_map, prov, warns = translate_scripts(
            source_text=str(turn["text"]),
            target_langs=target_langs,
            settings=settings,
        )
        provider_used = prov
        all_warnings.extend(warns)
        for code in target_langs:
            by_lang[code].append(
                {
                    "index": turn["index"],
                    "speaker": turn["speaker"],
                    "text": tr_map.get(code, turn["text"]),
                }
            )
    return by_lang, provider_used, list(dict.fromkeys(all_warnings))


def synthesize_warm_ambient_bed(duration_sec: float, sr: int = QWEN_SAMPLE_RATE) -> np.ndarray:
    """Generate a subtle, copyright-free warm ambient lo-fi synth pad for optional podcast background."""
    n = max(int(round(duration_sec * sr)), sr)
    t = np.arange(n, dtype=np.float32) / sr
    # Soft Cmaj9 chord pad with slow breathing LFO envelope
    lfo = 0.75 + 0.25 * np.sin(2.0 * np.pi * 0.11 * t)
    pad = (
        0.35 * np.sin(2.0 * np.pi * 130.81 * t)
        + 0.28 * np.sin(2.0 * np.pi * 196.00 * t)
        + 0.22 * np.sin(2.0 * np.pi * 246.94 * t)
        + 0.15 * np.sin(2.0 * np.pi * 293.66 * t)
    ) * lfo
    # Gentle intro/outro fade
    fade = min(int(1.8 * sr), n // 4)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        pad[:fade] *= ramp
        pad[-fade:] *= ramp[::-1]
    return (pad * 0.022).astype(np.float32)


def assemble_podcast_audio(
    turn_audios: List[Tuple[str, np.ndarray, int]],
    add_ambient_bgm: bool = False,
    target_sr: int = QWEN_SAMPLE_RATE,
) -> np.ndarray:
    """Stitch multi-speaker turns with natural conversational cadence and optional subtle ambient BGM."""
    if not turn_audios:
        return np.zeros(0, dtype=np.float32)

    fade = int(0.012 * target_sr)
    parts: List[np.ndarray] = []
    prev_speaker = None

    for idx, (spk, wav, sr) in enumerate(turn_audios):
        arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if sr != target_sr and arr.size > 0:
            new_n = int(round(arr.size * target_sr / sr))
            arr = np.interp(np.linspace(0, arr.size - 1, new_n), np.arange(arr.size), arr).astype(np.float32)
        if arr.size == 0:
            continue
        rms = float(np.sqrt(np.mean(arr * arr)))
        if rms > 1e-5:
            peak = float(np.max(np.abs(arr)))
            gain = min(0.125 / rms, 0.90 / max(peak, 1e-5), 2.5)
            arr = (arr * gain).astype(np.float32)
        if arr.size > fade * 2:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            arr = arr.copy()
            arr[:fade] *= ramp
            arr[-fade:] *= ramp[::-1]

        if idx > 0:
            gap_sec = 0.36 if spk != prev_speaker else 0.20
            parts.append(np.zeros(int(gap_sec * target_sr), dtype=np.float32))
        parts.append(arr)
        prev_speaker = spk

    merged = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    if add_ambient_bgm and merged.size > target_sr:
        bed = synthesize_warm_ambient_bed(merged.size / target_sr, target_sr)
        n = min(merged.size, bed.size)
        merged = merged.copy()
        merged[:n] = np.clip(merged[:n] + bed[:n], -0.96, 0.96)
    return merged
