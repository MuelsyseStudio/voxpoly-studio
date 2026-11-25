"""Pronunciation & Brand Terminology Glossary for VoxPoly Studio V3.

Ensures brand names, people, and technical acronyms (e.g. Qwen, VRAM, K8s) are never
mistranslated and are pronounced accurately across languages.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backend.config import DATA_DIR

GLOSSARY_FILE = DATA_DIR / "glossary.json"

DEFAULT_GLOSSARY: List[Dict[str, Any]] = [
    {
        "id": "term_voxpoly",
        "term": "VoxPoly",
        "keep_original": True,
        "lang_overrides": {},
        "tts_pronunciation": {"zh": "VoxPoly"},
        "notes": "品牌名称保持原样不翻译",
    },
]


class GlossaryStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        if not GLOSSARY_FILE.exists():
            self._save(DEFAULT_GLOSSARY)

    @staticmethod
    def _load() -> List[Dict[str, Any]]:
        try:
            if not GLOSSARY_FILE.exists():
                return list(DEFAULT_GLOSSARY)
            data = json.loads(GLOSSARY_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else list(DEFAULT_GLOSSARY)
        except Exception:
            return list(DEFAULT_GLOSSARY)

    @staticmethod
    def _save(items: List[Dict[str, Any]]) -> None:
        GLOSSARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = GLOSSARY_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(GLOSSARY_FILE)

    def list_terms(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._load()

    def upsert_term(self, item: Dict[str, Any]) -> Dict[str, Any]:
        term = str(item.get("term") or "").strip()
        if not term:
            raise ValueError("术语原文不能为空。")
        with self._lock:
            items = self._load()
            tid = str(item.get("id") or f"term_{uuid.uuid4().hex[:8]}")
            rec = {
                "id": tid,
                "term": term,
                "keep_original": bool(item.get("keep_original", True)),
                "lang_overrides": {
                    str(k).strip().lower(): str(v).strip()
                    for k, v in (item.get("lang_overrides") or {}).items()
                    if str(v).strip()
                },
                "tts_pronunciation": {
                    str(k).strip().lower(): str(v).strip()
                    for k, v in (item.get("tts_pronunciation") or {}).items()
                    if str(v).strip()
                },
                "notes": str(item.get("notes") or "").strip(),
            }
            replaced = False
            for idx, existing in enumerate(items):
                if existing.get("id") == tid or existing.get("term", "").lower() == term.lower():
                    rec["id"] = existing.get("id", tid)
                    items[idx] = rec
                    replaced = True
                    break
            if not replaced:
                items.insert(0, rec)
            self._save(items)
            return rec

    def delete_term(self, term_id: str) -> bool:
        with self._lock:
            items = self._load()
            new_items = [x for x in items if x.get("id") != term_id]
            if len(new_items) == len(items):
                return False
            self._save(new_items)
            return True


glossary_store = GlossaryStore()


def _term_pattern(term: str) -> re.Pattern:
    esc = re.escape(term)
    # Use word boundary for pure ASCII alphanumeric terms so "AI" doesn't match inside "wait"
    if re.fullmatch(r"[A-Za-z0-9_\-\.\+]+", term):
        return re.compile(rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])", re.IGNORECASE)
    return re.compile(esc)


def protect_terms_for_translation(
    text: str,
    target_lang: str,
    entries: List[Dict[str, Any]] | None = None,
) -> Tuple[str, Dict[str, str]]:
    """Replace glossary terms with safe placeholders before translation; return (masked_text, restore_map)."""
    items = entries if entries is not None else glossary_store.list_terms()
    if not items or not text:
        return text, {}

    # Sort longest terms first so longer compound names match before substrings
    sorted_items = sorted(items, key=lambda x: len(str(x.get("term", ""))), reverse=True)
    masked = text
    restore_map: Dict[str, str] = {}
    counter = 0

    for entry in sorted_items:
        term = str(entry.get("term") or "").strip()
        if not term:
            continue
        overrides = entry.get("lang_overrides") or {}
        if target_lang in overrides:
            replacement = str(overrides[target_lang])
        elif entry.get("keep_original", True):
            replacement = term
        else:
            continue

        pat = _term_pattern(term)
        if pat.search(masked):
            token = f"VPGLOSS{counter}X"
            counter += 1
            masked = pat.sub(token, masked)
            restore_map[token] = replacement

    return masked, restore_map


def restore_terms_after_translation(text: str, restore_map: Dict[str, str]) -> str:
    if not restore_map or not text:
        return text
    out = text
    for token, replacement in restore_map.items():
        # Tolerate translator lowercasing or spacing around placeholder token
        out = re.sub(re.escape(token), lambda _m, r=replacement: r, out, flags=re.IGNORECASE)
    return out


def apply_tts_pronunciation(
    text: str,
    lang_code: str,
    entries: List[Dict[str, Any]] | None = None,
) -> str:
    """Apply TTS phonetic/reading overrides right before neural speech synthesis."""
    items = entries if entries is not None else glossary_store.list_terms()
    if not items or not text:
        return text

    out = text
    sorted_items = sorted(items, key=lambda x: len(str(x.get("term", ""))), reverse=True)
    for entry in sorted_items:
        pron_map = entry.get("tts_pronunciation") or {}
        target_pron = pron_map.get(lang_code) or pron_map.get("*")
        if not target_pron:
            continue
        term = str(entry.get("term") or "").strip()
        overrides = entry.get("lang_overrides") or {}
        # Match either original term or language override form
        forms = [f for f in {term, str(overrides.get(lang_code) or "").strip()} if f]
        for form in sorted(forms, key=len, reverse=True):
            out = _term_pattern(form).sub(lambda _m, r=target_pron: r, out)
    return out
