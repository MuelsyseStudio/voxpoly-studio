from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.document_parser import parse_document_bytes
from backend.glossary import apply_tts_pronunciation, protect_terms_for_translation, restore_terms_after_translation
from backend.main import app
from backend.mastering import MASTERING_PRESETS, apply_mastering_dsp
from backend.podcast_engine import assemble_podcast_audio, generate_two_host_podcast_script, parse_dialogue_script
from backend.text_chunker import split_text_for_tts
from backend.video_dubbing import SUBTITLE_STYLES, write_styled_ass
from backend.watermark import detect_watermark, embed_watermark


def test_long_form_text_chunker():
    text = (
        "欢迎来到声音实验室。今天我们要测试长篇文章的智能分句与跨段拼接能力。\n\n"
        "无论是播客脚本、课程讲解还是短视频口播，系统都会优先在段落和句号处自然切分，"
        "避免在句子中间生硬截断，确保跨语种韵律连贯稳定。"
    )
    chunks = split_text_for_tts(text, "zh", max_chars=45)
    assert len(chunks) >= 3
    assert any(c["is_paragraph_end"] for c in chunks)
    assert chunks[-1]["pause_after_sec"] == 0.0


def test_glossary_protection_and_tts_pronunciation():
    entries = [
        {
            "id": "t1",
            "term": "VoxPoly",
            "keep_original": True,
            "lang_overrides": {},
            "tts_pronunciation": {"zh": "VoxPoly Studio"},
        },
        {
            "id": "t2",
            "term": "K8s",
            "keep_original": True,
            "lang_overrides": {"ja": "クバネティス"},
            "tts_pronunciation": {"en": "Kubernetes"},
        },
    ]
    masked, rmap = protect_terms_for_translation("Welcome to VoxPoly on K8s!", "ja", entries)
    assert "VoxPoly" not in masked and "K8s" not in masked
    restored = restore_terms_after_translation(masked.lower(), rmap)
    assert "VoxPoly" in restored and "クバネティス" in restored

    spoken_en = apply_tts_pronunciation("Deploying K8s cluster with VoxPoly", "en", entries)
    assert "Kubernetes" in spoken_en


def test_document_parser_markdown_and_epub():
    md = b"# Chapter 1 Intro\n\nHello world from VoxPoly Studio V3.\n\n# Chapter 2 Details\n\nDeep dive into voice cloning."
    doc_md = parse_document_bytes(md, "guide.md")
    assert doc_md["source_type"] == "md"
    assert len(doc_md["chapters"]) == 2

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "content.opf",
            '<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/">'
            "<metadata><dc:title>Test Book</dc:title></metadata>"
            '<manifest><item id="c1" href="ch1.xhtml"/></manifest>'
            '<spine><itemref idref="c1"/></spine></package>',
        )
        zf.writestr(
            "ch1.xhtml",
            "<html><body><h1>Chapter One</h1><p>This is a synthetic EPUB chapter with enough content to parse cleanly.</p></body></html>",
        )
    doc_epub = parse_document_bytes(buf.getvalue(), "sample.epub")
    assert doc_epub["title"] == "Test Book"
    assert len(doc_epub["chapters"]) == 1


def test_podcast_dialogue_parser_and_assembly():
    script, prov = generate_two_host_podcast_script(
        "人工智能语音克隆正在走向开源本地化。过去需要昂贵的云端按字计费，现在普通消费级显卡即可运行。",
        settings={},
        host_a="主持人",
        host_b="嘉宾",
        target_lang="zh",
    )
    assert prov
    turns, speakers = parse_dialogue_script(script)
    assert len(turns) >= 4
    assert speakers == ["主持人", "嘉宾"]

    sr = 24000
    t = np.arange(sr, dtype=np.float32) / sr
    tone = (0.12 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    merged = assemble_podcast_audio([("主持人", tone, sr), ("嘉宾", tone, sr)], add_ambient_bgm=True, target_sr=sr)
    assert merged.size > int(2.2 * sr)


def test_mastering_then_watermark_roundtrip():
    rng = np.random.default_rng(42)
    sr = 24000
    t = np.arange(sr * 5, dtype=np.float32) / sr
    clean = (
        0.15 * np.sin(2 * np.pi * 210 * t)
        + 0.09 * np.sin(2 * np.pi * 420 * t)
        + 0.008 * rng.standard_normal(t.size).astype(np.float32)
    ).astype(np.float32)

    assert len(MASTERING_PRESETS) == 4
    assert detect_watermark(clean, sr)["detected"] is False
    mastered = apply_mastering_dsp(clean, sr, "podcast")
    marked = embed_watermark(mastered, sr)
    res = detect_watermark(marked, sr)
    assert res["detected"] is True
    assert res["confidence"] >= 80.0


def test_styled_ass_generator(tmp_path: Path):
    assert len(SUBTITLE_STYLES) == 3
    segs = [{"start": 0.0, "end": 2.5, "text": "原文字幕"}]
    for style in SUBTITLE_STYLES:
        p = write_styled_ass(segs, ["Translated line"], tmp_path / f"{style['id']}.ass", style_id=style["id"], width=1080, height=1920)
        txt = p.read_text(encoding="utf-8")
        assert "PlayResX: 1080" in txt and "PlayResY: 1920" in txt
        assert "Translated line" in txt


def test_fastapi_v3_routes_and_glossary_crud():
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        st = client.get("/api/system/status")
        assert st.status_code == 200
        data = st.json()
        assert len(data["languages"]) == 23
        assert set(data["engine"]["engines"].keys()) == {"qwen3", "chatterbox", "cosyvoice3"}
        assert len(data["mastering_presets"]) == 4
        assert len(data["subtitle_styles"]) == 3
        assert data["features"]["openai_compat_url"] == "/v1/audio/speech"

        # OpenAI compatible model list
        m = client.get("/v1/models")
        assert m.status_code == 200
        assert any(x["id"] == "voxpoly-qwen3-1.7b" for x in m.json()["data"])

        # Glossary CRUD
        g_add = client.post("/api/glossary", json={"term": "TestTermX", "keep_original": True, "tts_pronunciation": {"zh": "测试词"}})
        assert g_add.status_code == 200
        tid = g_add.json()["term"]["id"]
        assert any(x["id"] == tid for x in client.get("/api/glossary").json()["terms"])
        assert client.delete(f"/api/glossary/{tid}").status_code == 200

        # Variant & mastering settings switch
        r1 = client.post("/api/settings", json={"qwen_variant": "0.6b", "mastering_preset": "podcast"})
        assert r1.status_code == 200 and r1.json()["qwen_variant"] == "0.6b" and r1.json()["mastering_preset"] == "podcast"
        r2 = client.post("/api/settings", json={"qwen_variant": "1.7b", "mastering_preset": "raw"})
        assert r2.status_code == 200 and r2.json()["qwen_variant"] == "1.7b"


def test_mcp_server_stdio_and_cli_help():
    # Verify MCP server JSON-RPC handshake & tool registration
    req_lines = (
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        + "\n"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "backend.mcp_server"],
        input=req_lines,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert proc.returncode == 0
    replies = [json.loads(ln) for ln in proc.stdout.strip().splitlines() if ln.strip()]
    assert len(replies) == 2
    assert replies[0]["result"]["serverInfo"]["name"] == "voxpoly-studio"
    tool_names = {t["name"] for t in replies[1]["result"]["tools"]}
    assert {"voxpoly_list_voices", "voxpoly_speak", "voxpoly_clone_multilingual", "voxpoly_generate_podcast", "voxpoly_verify_watermark"} <= tool_names

    # Verify CLI --help
    cli = subprocess.run([sys.executable, "-m", "backend.cli", "--help"], capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert cli.returncode == 0 and "speak" in cli.stdout and "batch" in cli.stdout
