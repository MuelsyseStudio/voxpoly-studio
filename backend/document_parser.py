"""Document & Web Article Importer (URL / PDF / EPUB / Markdown / TXT) for Audiobooks & AI Podcasts."""
from __future__ import annotations

import html
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from backend.net_safety import fetch_public_url_bytes

MAX_URL_HTML_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5


def _strip_html(raw_html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form"]):
            tag.decompose()
        root = soup.find("article") or soup.find("main") or soup.body or soup
        blocks = []
        for el in root.find_all(["h1", "h2", "h3", "p", "li", "blockquote"]):
            t = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
            if len(t) >= 2:
                blocks.append(t)
        text = "\n\n".join(blocks) if blocks else root.get_text("\n", strip=True)
        return _clean_doc_text(text)
    except Exception:
        no_script = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_html, flags=re.I | re.S)
        no_tags = re.sub(r"<[^>]+>", "\n", no_script)
        return _clean_doc_text(html.unescape(no_tags))


def _clean_doc_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    # Collapse excessive blank lines while keeping paragraph boundaries
    out: List[str] = []
    blank = False
    for ln in lines:
        if not ln:
            if not blank and out:
                out.append("")
            blank = True
        else:
            out.append(ln)
            blank = False
    return "\n".join(out).strip()


def _split_into_chapters(full_text: str, default_title: str) -> List[Dict[str, Any]]:
    clean = _clean_doc_text(full_text)
    if not clean:
        return []

    # Detect Markdown headings or Chinese/English chapter markers
    heading_re = re.compile(
        r"^(?:#{1,3}\s+.+|第[一二三四五六七八九十百千0-9]+[章节回篇幕].*|Chapter\s+[0-9IVX]+.*)$",
        re.MULTILINE | re.IGNORECASE,
    )
    matches = list(heading_re.finditer(clean))
    chapters: List[Dict[str, Any]] = []

    if len(matches) >= 2:
        for idx, m in enumerate(matches):
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(clean)
            ch_title = re.sub(r"^#{1,3}\s+", "", m.group(0)).strip()[:60]
            body = clean[start:end].strip()
            if len(body) >= 10:
                chapters.append({"index": len(chapters), "title": ch_title, "text": body, "char_count": len(body)})
    if not chapters:
        # Chunk very long flat text into ~1500-char readable chapters
        paras = [p.strip() for p in re.split(r"\n\s*\n+", clean) if p.strip()]
        buf = ""
        for p in paras:
            if buf and len(buf) + len(p) > 1600:
                chapters.append(
                    {
                        "index": len(chapters),
                        "title": f"{default_title} · 第 {len(chapters) + 1} 节",
                        "text": buf.strip(),
                        "char_count": len(buf.strip()),
                    }
                )
                buf = p
            else:
                buf = f"{buf}\n\n{p}".strip() if buf else p
        if buf:
            title = default_title if not chapters else f"{default_title} · 第 {len(chapters) + 1} 节"
            chapters.append({"index": len(chapters), "title": title, "text": buf.strip(), "char_count": len(buf.strip())})

    return chapters


def _parse_epub_bytes(raw: bytes, fallback_title: str) -> Dict[str, Any]:
    chapters: List[Dict[str, Any]] = []
    title = fallback_title
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        names = zf.namelist()
        # Try reading OPF spine order
        opf_path = next((n for n in names if n.lower().endswith(".opf")), None)
        ordered_html: List[str] = []
        if opf_path:
            try:
                opf_dir = str(Path(opf_path).parent).replace("\\", "/")
                opf_dir = "" if opf_dir == "." else opf_dir.rstrip("/") + "/"
                root = ET.fromstring(zf.read(opf_path))
                ns = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}
                t_el = root.find(".//dc:title", ns)
                if t_el is not None and t_el.text and t_el.text.strip():
                    title = t_el.text.strip()[:60]
                manifest = {}
                for item in root.findall(".//{*}manifest/{*}item"):
                    manifest[item.attrib.get("id", "")] = item.attrib.get("href", "")
                for itemref in root.findall(".//{*}spine/{*}itemref"):
                    href = manifest.get(itemref.attrib.get("idref", ""), "")
                    if href:
                        cand = f"{opf_dir}{href}".lstrip("/")
                        if cand in names:
                            ordered_html.append(cand)
            except Exception:
                ordered_html = []

        if not ordered_html:
            ordered_html = sorted(n for n in names if n.lower().endswith((".xhtml", ".html", ".htm")))

        for item_name in ordered_html:
            try:
                raw_html = zf.read(item_name).decode("utf-8", errors="ignore")
                txt = _strip_html(raw_html)
                if len(txt) < 35:
                    continue
                first_line = txt.split("\n", 1)[0].strip()[:50]
                chapters.append(
                    {
                        "index": len(chapters),
                        "title": first_line or f"第 {len(chapters) + 1} 章",
                        "text": txt,
                        "char_count": len(txt),
                    }
                )
            except Exception:
                continue

    if not chapters:
        raise ValueError("未能从该 EPUB 文件中提取到有效正文内容。")
    full_text = "\n\n".join(c["text"] for c in chapters)
    return {"title": title, "source_type": "epub", "chapters": chapters, "full_text": full_text, "total_chars": len(full_text)}


def _parse_pdf_bytes(raw: bytes, fallback_title: str) -> Dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("解析 PDF 需要安装 pypdf：pip install pypdf") from exc

    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for p in reader.pages:
        t = (p.extract_text() or "").strip()
        if t:
            pages.append(t)
    full_text = _clean_doc_text("\n\n".join(pages))
    if not full_text:
        raise ValueError("该 PDF 未提取到可选中文本（可能是纯图片扫描件）。")
    chapters = _split_into_chapters(full_text, fallback_title)
    return {"title": fallback_title, "source_type": "pdf", "chapters": chapters, "full_text": full_text, "total_chars": len(full_text)}


def parse_document_bytes(raw: bytes, filename: str) -> Dict[str, Any]:
    ext = Path(filename or "doc.txt").suffix.lower()
    stem = Path(filename or "文档").stem[:50]
    if ext == ".epub":
        return _parse_epub_bytes(raw, stem)
    if ext == ".pdf":
        return _parse_pdf_bytes(raw, stem)
    if ext in (".html", ".htm"):
        txt = _strip_html(raw.decode("utf-8", errors="ignore"))
        chapters = _split_into_chapters(txt, stem)
        return {"title": stem, "source_type": "html", "chapters": chapters, "full_text": txt, "total_chars": len(txt)}
    if ext in (".txt", ".md", ".markdown", ".srt"):
        raw_txt = raw.decode("utf-8", errors="ignore")
        if ext == ".md" or ext == ".markdown":
            # Strip markdown links/images/code fences gently for natural narration
            raw_txt = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", raw_txt)
            raw_txt = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw_txt)
            raw_txt = re.sub(r"```[\s\S]*?```", "", raw_txt)
            raw_txt = re.sub(r"[*_`>]+", "", raw_txt)
        clean = _clean_doc_text(raw_txt)
        if not clean:
            raise ValueError("文档内容为空。")
        chapters = _split_into_chapters(clean, stem)
        return {"title": stem, "source_type": ext.lstrip("."), "chapters": chapters, "full_text": clean, "total_chars": len(clean)}
    raise ValueError(f"暂不支持的文档格式 {ext}，支持 PDF / EPUB / Markdown / TXT / HTML。")


def parse_url_article(url: str) -> Dict[str, Any]:
    clean_url, raw_bytes, encoding = fetch_public_url_bytes(
        url,
        max_bytes=MAX_URL_HTML_BYTES,
        max_redirects=MAX_REDIRECTS,
        total_timeout_sec=20.0,
    )
    raw_html = raw_bytes.decode(encoding or "utf-8", errors="ignore")

    title = "网页文章"
    m_title = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.I | re.S)
    if m_title:
        title = html.unescape(re.sub(r"\s+", " ", m_title.group(1))).strip()[:60]

    txt = _strip_html(raw_html)
    if len(txt) < 20:
        raise ValueError("未能从该网页提取到有效正文（页面可能依赖动态渲染或需要登录）。")
    chapters = _split_into_chapters(txt, title)
    return {
        "title": title,
        "source_url": clean_url,
        "source_type": "url",
        "chapters": chapters,
        "full_text": txt,
        "total_chars": len(txt),
    }
