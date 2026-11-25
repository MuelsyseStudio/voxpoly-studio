from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.text_chunker import split_text_for_tts
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


def test_imperceptible_watermark_roundtrip():
    rng = np.random.default_rng(42)
    sr = 24000
    t = np.arange(sr * 5, dtype=np.float32) / sr
    # Synthetic voice-like harmonic signal with realistic microphone floor
    clean = (
        0.15 * np.sin(2 * np.pi * 210 * t)
        + 0.09 * np.sin(2 * np.pi * 420 * t)
        + 0.008 * rng.standard_normal(t.size).astype(np.float32)
    ).astype(np.float32)

    assert detect_watermark(clean, sr)["detected"] is False
    marked = embed_watermark(clean, sr)
    res = detect_watermark(marked, sr)
    assert res["detected"] is True
    assert res["confidence"] >= 80.0


def test_fastapi_system_and_variant_switch():
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        st = client.get("/api/system/status")
        assert st.status_code == 200
        data = st.json()
        assert len(data["languages"]) == 23
        assert set(data["engine"]["engines"].keys()) == {"qwen3", "chatterbox", "cosyvoice3"}
        assert len(data["qwen_variants"]) == 2

        # Switch to 0.6b low-VRAM variant and back
        r1 = client.post("/api/settings", json={"qwen_variant": "0.6b"})
        assert r1.status_code == 200 and r1.json()["qwen_variant"] == "0.6b"
        st2 = client.get("/api/system/status").json()
        assert st2["engine"]["engines"]["qwen3"]["variant"] == "0.6b"

        r2 = client.post("/api/settings", json={"qwen_variant": "1.7b"})
        assert r2.status_code == 200 and r2.json()["qwen_variant"] == "1.7b"
