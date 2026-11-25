from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import httpx

from backend.config import LANGUAGE_BY_CODE


def detect_script_language(text: str) -> Optional[str]:
    clean = text.strip()
    if not clean:
        return None
    total = len(re.sub(r"\s+", "", clean))
    if total == 0:
        return None

    hiragana_katakana = len(re.findall(r"[\u3040-\u30ff]", clean))
    hangul = len(re.findall(r"[\uac00-\ud7af]", clean))
    cyrillic = len(re.findall(r"[\u0400-\u04ff]", clean))
    arabic = len(re.findall(r"[\u0600-\u06ff]", clean))
    hebrew = len(re.findall(r"[\u0590-\u05ff]", clean))
    devanagari = len(re.findall(r"[\u0900-\u097f]", clean))
    greek = len(re.findall(r"[\u0370-\u03ff]", clean))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", clean))
    latin = len(re.findall(r"[A-Za-z]", clean))

    if hiragana_katakana / total >= 0.12:
        return "ja"
    if hangul / total >= 0.15:
        return "ko"
    if arabic / total >= 0.25:
        return "ar"
    if hebrew / total >= 0.25:
        return "he"
    if devanagari / total >= 0.25:
        return "hi"
    if greek / total >= 0.25:
        return "el"
    if cyrillic / total >= 0.25:
        return "ru"
    if cjk / total >= 0.25:
        return "zh"
    if latin / total >= 0.65:
        if re.search(r"[äöüßÄÖÜ]", clean):
            return "de"
        if re.search(r"[¿¡ñÑ]", clean):
            return "es"
        if re.search(r"[ãõçÃÕ]", clean):
            return "pt"
    return None


def _clean_translated_text(text: str) -> str:
    cleaned = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _call_llm_json(api_base: str, api_key: str, model: str, system_prompt: str, user_prompt: str) -> Any:
    endpoint = api_base.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"].strip()
    fence_match = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", content, re.S)
    raw_json = fence_match.group(1) if fence_match else content
    return json.loads(raw_json)


def _translate_via_llm(
    source_text: str,
    target_langs: List[str],
    api_base: str,
    api_key: str,
    model: str,
) -> Dict[str, str]:
    lang_spec = [
        f'- "{code}": {LANGUAGE_BY_CODE[code]["native_name"]} ({LANGUAGE_BY_CODE[code]["label_zh"]})'
        for code in target_langs
        if code in LANGUAGE_BY_CODE
    ]
    system_prompt = (
        "You are an expert voice-over & dubbing localization writer. "
        "Translate the user's input script into natural, idiomatic, spoken-style speech "
        "suitable for neural voice cloning in each target language. "
        "Preserve paragraph breaks (\\n\\n) for long scripts, keep spoken duration comparable to the original, "
        "avoid parenthetical notes, and return ONLY a valid JSON object mapping each language code to its localized text."
    )
    user_prompt = (
        f"Source script:\n{source_text}\n\n"
        f"Target language codes:\n" + "\n".join(lang_spec) + "\n\n"
        "Respond strictly with JSON."
    )
    parsed = _call_llm_json(api_base, api_key, model, system_prompt, user_prompt)
    return {
        code: _clean_translated_text(str(parsed[code]))
        for code in target_langs
        if code in parsed and str(parsed[code]).strip()
    }


def _translate_single_free(source_text: str, lang_code: str, detected_lang: Optional[str] = None) -> str:
    meta = LANGUAGE_BY_CODE[lang_code]
    errors: List[str] = []

    # Preserve paragraph breaks for long-form texts cleanly
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", source_text) if p.strip()]
    if len(paragraphs) > 1:
        translated_paras = [_translate_single_free(p, lang_code, detected_lang) for p in paragraphs]
        return "\n\n".join(translated_paras)

    # Tier 1: Google Translate gtx JSON endpoint (handles all 23 languages cleanly)
    try:
        with httpx.Client(timeout=14.0, follow_redirects=True) as client:
            resp = client.get(
                "https://translate.googleapis.com/translate_a/single",
                params={
                    "client": "gtx",
                    "sl": str(LANGUAGE_BY_CODE[detected_lang]["translator_code"])
                    if detected_lang in LANGUAGE_BY_CODE
                    else "auto",
                    "tl": str(meta["translator_code"]),
                    "dt": "t",
                    "q": source_text,
                },
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, list) and payload and isinstance(payload[0], list):
                joined = "".join(str(seg[0]) for seg in payload[0] if isinstance(seg, list) and seg and seg[0])
                joined = _clean_translated_text(joined)
                if joined:
                    return joined
    except Exception as exc:
        errors.append(f"gtx:{exc}")

    # Tier 2: MyMemory API
    try:
        from deep_translator import MyMemoryTranslator

        src_mm = (
            str(LANGUAGE_BY_CODE[detected_lang]["mymemory_code"])
            if detected_lang in LANGUAGE_BY_CODE
            else ("zh-CN" if re.search(r"[\u4e00-\u9fff]", source_text) else "en-US")
        )
        res = MyMemoryTranslator(source=src_mm, target=str(meta["mymemory_code"])).translate(source_text)
        res = _clean_translated_text(res or "")
        if res:
            return res
    except Exception as exc:
        errors.append(f"mymemory:{exc}")

    # Tier 3: deep-translator Google fallback
    try:
        from deep_translator import GoogleTranslator

        res = GoogleTranslator(source="auto", target=str(meta["translator_code"])).translate(source_text)
        res = _clean_translated_text(res or "")
        if res:
            return res
    except Exception as exc:
        errors.append(f"google:{exc}")

    raise RuntimeError(f"免费翻译引擎暂不可用（{meta['label_zh']}）: {'; '.join(errors)}")


def translate_scripts(
    source_text: str,
    target_langs: List[str],
    settings: Dict[str, object],
    source_lang: Optional[str] = None,
) -> Tuple[Dict[str, str], str, List[str]]:
    clean_text = source_text.strip()
    if not clean_text:
        raise ValueError("请输入需要翻译的文案内容。")

    valid_langs = [code for code in target_langs if code in LANGUAGE_BY_CODE]
    if not valid_langs:
        raise ValueError("请至少选择一种目标语言。")

    detected_lang = source_lang if source_lang in LANGUAGE_BY_CODE else detect_script_language(clean_text)
    results: Dict[str, str] = {}
    warnings: List[str] = []

    langs_to_translate: List[str] = []
    for code in valid_langs:
        if detected_lang and code == detected_lang:
            results[code] = clean_text
        else:
            langs_to_translate.append(code)

    if not langs_to_translate:
        return results, "原文直通", warnings

    provider = str(settings.get("translation_provider") or "auto")
    api_base = str(settings.get("llm_api_base") or "").strip()
    api_key = str(settings.get("llm_api_key") or "").strip()
    llm_model = str(settings.get("llm_model") or "deepseek-chat").strip()

    if provider in ("auto", "llm") and api_base and api_key:
        try:
            llm_out = _translate_via_llm(clean_text, langs_to_translate, api_base, api_key, llm_model)
            results.update(llm_out)
            missing = [c for c in langs_to_translate if c not in results]
            if not missing:
                return results, f"LLM ({llm_model})", warnings
            langs_to_translate = missing
            warnings.append("部分语种由免费翻译引擎补全")
        except Exception as exc:
            if provider == "llm":
                raise RuntimeError(f"自定义大模型翻译失败: {exc}") from exc
            warnings.append(f"大模型翻译失败，已自动回退到免费翻译引擎（{exc}）")

    with ThreadPoolExecutor(max_workers=min(8, len(langs_to_translate))) as pool:
        fut_map = {
            pool.submit(_translate_single_free, clean_text, code, detected_lang): code
            for code in langs_to_translate
        }
        for fut in as_completed(fut_map):
            code = fut_map[fut]
            try:
                results[code] = fut.result()
            except Exception as exc:
                warnings.append(str(exc))
                results[code] = clean_text

    ordered = {code: results[code] for code in valid_langs if code in results}
    return ordered, "Google 免费翻译引擎", warnings


def translate_segments_for_dubbing(
    segments: List[Dict[str, Any]],
    target_langs: List[str],
    settings: Dict[str, object],
    source_lang: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], str, List[str]]:
    """Duration-aware translation of timestamped video segments for natural lip/scene-synced dubbing."""
    valid_langs = [c for c in target_langs if c in LANGUAGE_BY_CODE]
    if not segments or not valid_langs:
        raise ValueError("缺少视频语音分段或目标语种。")

    sample_text = " ".join(str(s.get("text", "")) for s in segments[:5])
    detected_lang = source_lang if source_lang in LANGUAGE_BY_CODE else detect_script_language(sample_text)

    provider = str(settings.get("translation_provider") or "auto")
    api_base = str(settings.get("llm_api_base") or "").strip()
    api_key = str(settings.get("llm_api_key") or "").strip()
    llm_model = str(settings.get("llm_model") or "deepseek-chat").strip()

    out: Dict[str, List[str]] = {}
    warnings: List[str] = []
    langs_todo: List[str] = []

    for code in valid_langs:
        if detected_lang and code == detected_lang:
            out[code] = [str(s["text"]).strip() for s in segments]
        else:
            langs_todo.append(code)

    if not langs_todo:
        return out, "原文直通", warnings

    # Duration-matched dubbing localization via LLM if configured
    if provider in ("auto", "llm") and api_base and api_key:
        try:
            compact_segs = [
                {"i": idx, "dur": round(float(s["end"]) - float(s["start"]), 1), "text": str(s["text"]).strip()}
                for idx, s in enumerate(segments)
            ]
            lang_spec = [f'"{c}": {LANGUAGE_BY_CODE[c]["native_name"]}' for c in langs_todo]
            sys_p = (
                "You are an expert audiovisual dubbing translator. "
                "Translate each timed video dialogue segment into natural spoken language whose spoken duration fits closely within each segment's `dur` seconds. "
                "Keep phrasing concise and natural for voice acting. "
                "Return ONLY JSON mapping each language code to an array of translated strings of exact length matching the input segments."
            )
            usr_p = (
                f"Target languages: {', '.join(lang_spec)}\n"
                f"Segments:\n{json.dumps(compact_segs, ensure_ascii=False)}\n"
                "Return strictly JSON."
            )
            parsed = _call_llm_json(api_base, api_key, llm_model, sys_p, usr_p)
            remaining = []
            for code in langs_todo:
                arr = parsed.get(code)
                if isinstance(arr, list) and len(arr) == len(segments):
                    out[code] = [_clean_translated_text(str(x)) or str(segments[i]["text"]) for i, x in enumerate(arr)]
                else:
                    remaining.append(code)
            if not remaining:
                return {c: out[c] for c in valid_langs}, f"LLM 等时长译配 ({llm_model})", warnings
            langs_todo = remaining
            warnings.append("部分视频语种由免费翻译补全")
        except Exception as exc:
            if provider == "llm":
                raise RuntimeError(f"大模型视频译配失败: {exc}") from exc
            warnings.append(f"大模型视频译配失败，已回退免费翻译 ({exc})")

    # Free concurrent segment translation
    def _tr_lang(code: str) -> List[str]:
        res_list: List[str] = []
        for s in segments:
            txt = str(s["text"]).strip()
            try:
                res_list.append(_translate_single_free(txt, code, detected_lang))
            except Exception:
                res_list.append(txt)
        return res_list

    with ThreadPoolExecutor(max_workers=min(6, len(langs_todo))) as pool:
        futs = {pool.submit(_tr_lang, code): code for code in langs_todo}
        for fut in as_completed(futs):
            code = futs[fut]
            out[code] = fut.result()

    return {c: out[c] for c in valid_langs}, "Google 免费译配引擎", warnings
