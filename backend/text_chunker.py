"""Smart multilingual long-form text chunker & smooth prosody audio stitcher."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np


CJK_LANGS = {"zh", "ja", "ko"}

# Sentence boundary regex preserving delimiters and skipping decimal numbers (e.g. 3.14)
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；…!\?;\.।؟])(?:\s+|(?=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]))")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[，、,;：:])(?:\s+|(?=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]))")


def default_max_chars(lang_code: str) -> int:
    return 95 if lang_code in CJK_LANGS else 175


def _hard_wrap(text: str, max_chars: int, is_cjk: bool) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    if is_cjk:
        return [text[i : i + max_chars].strip() for i in range(0, len(text), max_chars) if text[i : i + max_chars].strip()]
    words = text.split()
    out: List[str] = []
    cur = ""
    for w in words:
        cand = f"{cur} {w}".strip() if cur else w
        if len(cand) <= max_chars or not cur:
            cur = cand
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def split_text_for_tts(
    text: str,
    lang_code: str,
    max_chars: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Split arbitrary long text (up to 5,000+ chars) into TTS-optimal sentence chunks.

    Returns list of dicts: {index, text, pause_after_sec, is_paragraph_end}.
    """
    clean = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not clean:
        return []

    limit = max_chars or default_max_chars(lang_code)
    is_cjk = lang_code in CJK_LANGS
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", clean) if p.strip()]
    if not paragraphs:
        paragraphs = [clean]

    chunks: List[Dict[str, object]] = []
    for p_idx, para in enumerate(paragraphs):
        # Single newlines inside a paragraph act like sentence breaks
        para_norm = re.sub(r"\s*\n\s*", "。" if is_cjk else ". ", para)
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(para_norm) if s.strip()]
        if not sentences:
            sentences = [para_norm]

        # Break any overly long sentence on clause punctuation, then hard-wrap if still huge
        pieces: List[str] = []
        for sent in sentences:
            if len(sent) <= limit:
                pieces.append(sent)
            else:
                clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(sent) if c.strip()]
                for cl in clauses:
                    pieces.extend(_hard_wrap(cl, limit, is_cjk))

        # Pack adjacent short pieces together up to `limit`
        packed: List[str] = []
        buf = ""
        joiner = "" if is_cjk else " "
        for piece in pieces:
            if not buf:
                buf = piece
            elif len(buf) + len(joiner) + len(piece) <= limit:
                buf = f"{buf}{joiner}{piece}"
            else:
                packed.append(buf)
                buf = piece
        if buf:
            packed.append(buf)

        for i, chunk_text in enumerate(packed):
            is_para_end = (i == len(packed) - 1) and (p_idx < len(paragraphs) - 1)
            is_last_overall = (i == len(packed) - 1) and (p_idx == len(paragraphs) - 1)
            pause = 0.0 if is_last_overall else (0.52 if is_para_end else 0.22)
            chunks.append(
                {
                    "index": len(chunks),
                    "text": chunk_text,
                    "pause_after_sec": pause,
                    "is_paragraph_end": is_para_end,
                }
            )

    return chunks


def _trim_edges(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.size < int(0.2 * sr):
        return audio
    abs_a = np.abs(audio)
    peak = float(np.max(abs_a))
    if peak < 1e-4:
        return audio
    thresh = max(peak * 0.015, 0.003)
    idx = np.where(abs_a > thresh)[0]
    if idx.size == 0:
        return audio
    start = max(0, int(idx[0]) - int(0.03 * sr))
    end = min(audio.size, int(idx[-1]) + int(0.06 * sr))
    return audio[start:end]


def stitch_chunk_audios(
    chunk_items: List[Tuple[np.ndarray, int, float]],
    target_sr: int = 24000,
) -> np.ndarray:
    """Level loudness across chunks, apply micro cosine fades, and concatenate with natural pauses."""
    if not chunk_items:
        return np.zeros(0, dtype=np.float32)

    fade_len = int(0.012 * target_sr)
    segments: List[np.ndarray] = []

    # Compute median RMS across non-silent chunks so prosody volume stays uniform across paragraphs
    rms_vals = []
    processed = []
    for wav, sr, pause_sec in chunk_items:
        arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if sr != target_sr and arr.size > 0:
            new_len = int(round(arr.size * target_sr / sr))
            arr = np.interp(np.linspace(0, arr.size - 1, new_len), np.arange(arr.size), arr).astype(np.float32)
        arr = _trim_edges(arr, target_sr)
        if arr.size > 0:
            rms = float(np.sqrt(np.mean(arr * arr)))
            if rms > 1e-4:
                rms_vals.append(rms)
            processed.append((arr, rms, pause_sec))

    target_rms = float(np.median(rms_vals)) if rms_vals else 0.09
    target_rms = max(0.06, min(0.16, target_rms))

    for idx, (arr, rms, pause_sec) in enumerate(processed):
        if rms > 1e-5:
            peak = float(np.max(np.abs(arr)))
            gain = min(target_rms / rms, 0.90 / max(peak, 1e-5), 2.2)
            arr = (arr * gain).astype(np.float32)

        if arr.size > fade_len * 2:
            ramp = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, fade_len, dtype=np.float32)))
            arr = arr.copy()
            arr[:fade_len] *= ramp
            arr[-fade_len:] *= ramp[::-1]

        segments.append(arr)
        if pause_sec > 0 and idx < len(processed) - 1:
            segments.append(np.zeros(int(pause_sec * target_sr), dtype=np.float32))

    return np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
