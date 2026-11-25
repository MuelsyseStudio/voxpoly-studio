from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config import VOICES_DIR
from backend.document_parser import parse_document_bytes
from backend.glossary import apply_tts_pronunciation, protect_terms_for_translation, restore_terms_after_translation
from backend.main import app
from backend.mastering import MASTERING_PRESETS, apply_mastering_dsp
from backend.net_safety import validate_public_http_url
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


def test_access_token_protects_api_v1_and_media_and_blocks_index_json(monkeypatch):
    from backend.main import _auth_state

    _auth_state.reset_all()
    sample_dir = VOICES_DIR / "sec_test_voice"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_wav = sample_dir / "clone_ref.wav"
    sample_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")

    # Test with non-ASCII UTF-8 passcode to verify cookie session stays 100% ASCII-safe
    secret = "口令_StrongTestPasscode_2026!"
    monkeypatch.setenv("VOXPOLY_ACCESS_TOKEN", secret)
    try:
        with TestClient(app) as client:
            # Public shell & health remain reachable and expose security flags
            assert client.get("/").status_code == 200
            h = client.get("/api/health")
            assert h.status_code == 200
            h_json = h.json()
            assert h_json["requires_token"] is True
            assert h_json["token_strong"] is True
            assert h_json["media_protected"] is True

            # Unauthenticated API, OpenAI v1, docs, and /media/* must ALL be 401
            for path in (
                "/api/voices",
                "/api/system/status",
                "/v1/models",
                "/media/voices/sec_test_voice/clone_ref.wav",
                "/docs",
                "/openapi.json",
            ):
                assert client.get(path).status_code == 401, f"Expected 401 for {path}"

            # Malformed Host header (GHSA-86qp-5c8j-p5mr style) must NOT bypass ASGI route-path auth
            assert client.get("/api/voices", headers={"Host": "evil.example/static/app.js"}).status_code == 401
            assert client.get("/media/voices/sec_test_voice/clone_ref.wav", headers={"Host": "evil.example/api/health"}).status_code == 401

            # Query-string token must NOT be accepted
            assert client.get(f"/api/voices?token={secret}").status_code == 401
            assert client.get(f"/media/voices/sec_test_voice/clone_ref.wav?token={secret}").status_code == 401

            # Wrong passcode rejected by session endpoint
            assert client.post("/api/auth/session", json={"token": "wrong"}).status_code == 401

            # Correct UTF-8 passcode sets revocable random ASCII HttpOnly cookie
            login = client.post("/api/auth/session", json={"token": secret}, headers={"X-VoxPoly-Request": "1"})
            assert login.status_code == 200
            set_cookie = login.headers.get("set-cookie", "")
            assert "voxpoly_session=" in set_cookie and "httponly" in set_cookie.lower()
            assert secret not in set_cookie  # raw token is NEVER placed in cookie

            # Capture a copy of the session cookie to prove server-side revocation works on logout
            stolen_sid = client.cookies.get("voxpoly_session")
            assert stolen_sid

            # Cookie unlocks GET /media/*, GET /api/*, and sets Cache-Control: private, no-store everywhere (incl. /docs)
            media_ok = client.get("/media/voices/sec_test_voice/clone_ref.wav")
            assert media_ok.status_code == 200
            assert "no-store" in media_ok.headers.get("cache-control", "").lower()
            docs_ok = client.get("/docs")
            assert docs_ok.status_code == 200
            assert "no-store" in docs_ok.headers.get("cache-control", "").lower()
            assert client.get("/api/voices").status_code == 200

            # CSRF protection on cookie-authenticated mutations: missing custom header or cross-site Origin rejected with 403
            assert client.post("/api/glossary", json={"term": "CsrfTest"}).status_code == 403
            assert client.post(
                "/api/glossary",
                json={"term": "CsrfTest"},
                headers={"X-VoxPoly-Request": "1", "Sec-Fetch-Site": "same-site"},
            ).status_code == 403
            assert client.post(
                "/api/glossary",
                json={"term": "CsrfTest"},
                headers={"X-VoxPoly-Request": "1", "Origin": "https://attacker.example"},
            ).status_code == 403

            # Same-origin cookie mutation with custom header succeeds
            g_ok = client.post(
                "/api/glossary",
                json={"term": "CsrfAllowedTerm"},
                headers={"X-VoxPoly-Request": "1"},
            )
            assert g_ok.status_code == 200
            tid = g_ok.json()["term"]["id"]
            assert client.delete(f"/api/glossary/{tid}", headers={"X-VoxPoly-Request": "1"}).status_code == 200

            # Internal index.json AND Windows trailing-dot/space aliases must ALWAYS be 404 even when authenticated
            for meta_path in (
                "/media/voices/index.json",
                "/media/voices/index.json.",
                "/media/voices/index.json%20",
                "/media/voices/.hidden",
                "/media/unknown/clone_ref.wav",
            ):
                assert client.get(meta_path).status_code == 404, f"Expected 404 for {meta_path}"

            # Header auth works for headless OpenAI / MCP clients (use ASCII secret for HTTP headers)
            ascii_secret = "AsciiHeaderPasscode_2026!"
            monkeypatch.setenv("VOXPOLY_ACCESS_TOKEN", ascii_secret)
            # Rotating VOXPOLY_ACCESS_TOKEN immediately invalidates prior sessions
            assert client.get("/media/voices/sec_test_voice/clone_ref.wav").status_code == 401
            with TestClient(app) as api_client:
                assert api_client.get("/v1/models", headers={"Authorization": f"Bearer {ascii_secret}"}).status_code == 200
                assert api_client.get("/api/voices", headers={"X-Access-Token": ascii_secret}).status_code == 200

            # Re-login and verify DELETE /api/auth/session revokes stolen cookie copy server-side
            login2 = client.post("/api/auth/session", json={"token": ascii_secret}, headers={"X-VoxPoly-Request": "1"})
            assert login2.status_code == 200
            sid2 = client.cookies.get("voxpoly_session")
            assert client.delete("/api/auth/session", headers={"X-VoxPoly-Request": "1"}).status_code == 200
            with TestClient(app) as replay_client:
                replay_client.cookies.set("voxpoly_session", sid2)
                assert replay_client.get("/media/voices/sec_test_voice/clone_ref.wav").status_code == 401

            # Verify llm_api_base blocks private URLs in token-protected mode & clears old API key when changed
            assert client.post(
                "/api/settings",
                json={"llm_api_base": "http://127.0.0.1:11434/v1"},
                headers={"X-Access-Token": ascii_secret},
            ).status_code == 400
    finally:
        _auth_state.reset_all()
        sample_wav.unlink(missing_ok=True)
        try:
            sample_dir.rmdir()
        except OSError:
            pass


def test_auth_rate_limit_and_root_path_guard(monkeypatch):
    from backend.main import _auth_state, _normalize_route_path

    # Verify root_path normalization matches Starlette router
    assert _normalize_route_path({"path": "/sub/api/voices", "root_path": "/sub"}) == "/api/voices"
    assert _normalize_route_path({"path": "//media/voices/index.json", "root_path": ""}) == "/media/voices/index.json"

    _auth_state.reset_all()
    monkeypatch.setenv("VOXPOLY_ACCESS_TOKEN", "RateLimitPasscode_2026!")
    try:
        with TestClient(app) as client:
            for _ in range(15):
                assert client.post("/api/auth/session", json={"token": "wrong"}).status_code == 401
            r429 = client.post("/api/auth/session", json={"token": "wrong"})
            assert r429.status_code == 429
    finally:
        _auth_state.reset_all()


def test_ssrf_guard_settings_key_isolation_and_colab_lifecycle():
    for bad_url in (
        "http://127.0.0.1:8765/api/voices",
        "http://localhost/secret",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/internal",
        "http://100.64.0.1/cgnat",
        "http://[::1]:8765/",
        "http://[::ffff:127.0.0.1]/",
        "http://[fec0::1]/site-local",
        "file:///etc/passwd",
        "http://user:pass@example.com/",
    ):
        with pytest.raises(ValueError):
            validate_public_http_url(bad_url)

    with TestClient(app) as client:
        assert client.post("/api/documents/import-url", json={"url": "http://127.0.0.1:8765/"}).status_code == 400
        assert client.post("/api/videos/import-url", json={"url": "http://169.254.169.254/latest/meta-data/"}).status_code == 400

        # Changing llm_api_base without supplying a new key must clear the previous saved key
        s1 = client.post("/api/settings", json={"llm_api_base": "https://api.deepseek.com/v1", "llm_api_key": "sk-test-secret-123456"})
        assert s1.status_code == 200 and s1.json()["has_llm_api_key"] is True
        s2 = client.post("/api/settings", json={"llm_api_base": "https://other.example/v1"})
        assert s2.status_code == 200 and s2.json()["has_llm_api_key"] is False
        client.post("/api/settings", json={"llm_api_base": "", "clear_api_key": True})

    # Verify Colab notebook defaults public tunnel OFF, stops cloudflared cleanly in Step 3 & 4, supports UTF-8 session login, and enforces strong token + media_protected
    nb_path = Path(__file__).resolve().parent.parent / "colab" / "VoxPoly_Colab.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    step3_code, step4_code = code_cells[2], code_cells[3]
    assert "def stop_cloudflared" in step3_code and "stop_cloudflared()" in step3_code
    assert "/api/auth/session" in step3_code
    assert "ENABLE_PUBLIC_CLOUDFLARE_URL = False" in step4_code
    assert step4_code.index("stop_cloudflared()") < step4_code.index("if ENABLE_PUBLIC_CLOUDFLARE_URL:")
    assert 'health.get("requires_token")' in step4_code
    assert 'health.get("token_strong")' in step4_code
    assert 'health.get("media_protected")' in step4_code

    # Verify Docker Compose defaults strictly to host loopback
    compose_txt = (Path(__file__).resolve().parent.parent / "docker-compose.yml").read_text(encoding="utf-8")
    assert '"127.0.0.1:8765:8765"' in compose_txt


def test_pinned_http_fetch_llm_connection_guard_and_cookie_not_locked_by_bruteforce(monkeypatch):
    import httpx

    from backend.main import _auth_state
    from backend.net_safety import fetch_public_url_bytes
    from backend.safe_ytdlp import install_public_only_socket_guard
    from backend.translator import _call_llm_json

    # 1. Mocked streaming HTTP 200 through fetch_public_url_bytes works cleanly (verifies Response lifecycle & byte cap)
    def _mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ok":
            return httpx.Response(
                200,
                stream=httpx.ByteStream(b"<html><title>Hi</title><body><p>Hello world article content here.</p></body></html>"),
                headers={"content-type": "text/html; charset=utf-8"},
            )
        if request.url.path == "/too-big":
            return httpx.Response(200, stream=httpx.ByteStream(b"A" * 2048))
        if request.url.path == "/redir-local":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:8765/api/voices"})
        return httpx.Response(404)

    transport = httpx.MockTransport(_mock_handler)
    final_url, body, enc = fetch_public_url_bytes("https://93.184.216.34/ok", transport=transport)
    assert final_url == "https://93.184.216.34/ok" and b"Hello world" in body and enc.lower() == "utf-8"
    with pytest.raises(ValueError):
        fetch_public_url_bytes("https://93.184.216.34/too-big", max_bytes=1024, transport=transport)

    # 2. Redirect to loopback must be rejected on second hop
    with pytest.raises(ValueError):
        fetch_public_url_bytes("https://93.184.216.34/redir-local", transport=transport)

    # 3. _call_llm_json enforces public-IP policy at connection time even for pre-saved private llm_api_base
    monkeypatch.setenv("VOXPOLY_ACCESS_TOKEN", "ConnGuardPasscode_2026!")
    monkeypatch.delenv("VOXPOLY_ALLOW_LOCAL_LLM", raising=False)
    with pytest.raises(ValueError):
        _call_llm_json("http://127.0.0.1:11434/v1", "sk-dummy", "model", "sys", "usr")

    # 4. Already-authenticated session cookie is NOT locked out by failed unauthenticated brute-force attempts; 401/429 carry no-store
    _auth_state.reset_all()
    try:
        with TestClient(app) as client:
            login = client.post("/api/auth/session", json={"token": "ConnGuardPasscode_2026!"}, headers={"X-VoxPoly-Request": "1"})
            assert login.status_code == 200
            with TestClient(app) as attacker:
                for _ in range(16):
                    r = attacker.get("/api/voices", headers={"X-Access-Token": "bad"})
                    assert "no-store" in r.headers.get("cache-control", "").lower()
                r_sess_bad = attacker.post("/api/auth/session", json={"token": "bad"})
                assert r_sess_bad.status_code == 401 and "no-store" in r_sess_bad.headers.get("cache-control", "").lower()
            # Logged-in user with valid session cookie still succeeds cleanly
            assert client.get("/api/voices").status_code == 200
    finally:
        _auth_state.reset_all()

    # 5. safe_ytdlp subclassable Popen guard, FFmpeg file,pipe injection, and non-double-counted cumulative byte cap
    import socket
    import subprocess as sp
    import time

    from backend.safe_ytdlp import _inject_ffmpeg_file_whitelist, install_subprocess_network_guard, make_cumulative_byte_cap_hook

    orig_gai = socket.getaddrinfo
    orig_popen = sp.Popen
    try:
        install_public_only_socket_guard()
        with pytest.raises(PermissionError):
            socket.getaddrinfo("127.0.0.1", 80)

        install_subprocess_network_guard()
        # Verify yt-dlp's `class Popen(subprocess.Popen)` subclassing works without TypeError
        class _SubPopen(sp.Popen):
            pass

        with pytest.raises(PermissionError):
            _SubPopen(["ffmpeg", "-i", "http://169.254.169.254/stream.m3u8"])

        rewritten = _inject_ffmpeg_file_whitelist(["ffmpeg", "-protocol_whitelist", "file,http,https", "-i", "v.mp4", "-i", "a.m4a", "out.mp4"])
        assert rewritten == ["ffmpeg", "-protocol_whitelist", "file,pipe", "-i", "v.mp4", "-protocol_whitelist", "file,pipe", "-i", "a.m4a", "out.mp4"]
        rewritten_probe = _inject_ffmpeg_file_whitelist(["ffprobe", "-protocol_whitelist", "http,https", "v.mp4"])
        assert rewritten_probe == ["ffprobe", "-protocol_whitelist", "file,pipe", "v.mp4"]
    finally:
        socket.getaddrinfo = orig_gai
        sp.Popen = orig_popen

    # Single track downloading (.part) + finished (no .part) must NOT double-count
    hook = make_cumulative_byte_cap_hook(max_bytes=1000, max_mb=1)
    hook({"tmpfilename": "video.mp4.part", "filename": "video.mp4", "downloaded_bytes": 600})
    hook({"filename": "video.mp4", "downloaded_bytes": 600})
    # Distinct second track pushing cumulative total over 1000 MUST raise
    with pytest.raises(ValueError):
        hook({"tmpfilename": "audio.m4a.part", "filename": "audio.m4a", "downloaded_bytes": 500})

    # Verify hard wall-clock deadline aborts slow streaming fetch promptly
    def _slow_handler(_request: httpx.Request) -> httpx.Response:
        time.sleep(0.8)
        return httpx.Response(200, stream=httpx.ByteStream(b"slow"))

    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        fetch_public_url_bytes("https://93.184.216.34/slow", total_timeout_sec=0.2, transport=httpx.MockTransport(_slow_handler))
    assert (time.monotonic() - t0) < 0.55

    # 6. /api/auth/session sets Secure cookie behind loopback Cloudflare HTTPS proxy even when VOXPOLY_TRUST_CF_IP=1, and always sends no-store
    monkeypatch.setenv("VOXPOLY_TRUST_CF_IP", "1")
    monkeypatch.setenv("VOXPOLY_ACCESS_TOKEN", "CfHttpsPasscode_2026!")
    _auth_state.reset_all()
    try:
        with TestClient(app, client=("127.0.0.1", 50000)) as cf_client:
            r_cf = cf_client.post(
                "/api/auth/session",
                json={"token": "CfHttpsPasscode_2026!"},
                headers={"X-VoxPoly-Request": "1", "CF-Connecting-IP": "203.0.113.10", "X-Forwarded-Proto": "https"},
            )
            assert r_cf.status_code == 200 and "secure" in r_cf.headers.get("set-cookie", "").lower()
    finally:
        _auth_state.reset_all()

    monkeypatch.delenv("VOXPOLY_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("VOXPOLY_TRUST_CF_IP", raising=False)
    with TestClient(app) as c_no_tok:
        r422 = c_no_tok.post("/api/auth/session", json={"token": 12345})
        assert r422.status_code == 422 and "no-store" in r422.headers.get("cache-control", "").lower()
