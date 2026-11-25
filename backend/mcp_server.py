"""Built-in Model Context Protocol (MCP) stdio server for VoxPoly Studio V3.

Lets Antigravity, Claude Desktop, Cursor, or Codex directly invoke your local
voiceprints to speak, generate multilingual reels, produce podcasts, or verify watermarks.

Usage in mcp_config.json:
{
  "mcpServers": {
    "voxpoly": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "backend.mcp_server"],
      "cwd": "/path/to/voxpoly-studio",
      "env": { "VOXPOLY_API_URL": "http://127.0.0.1:8765" }
    }
  }
}
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

API_URL = os.getenv("VOXPOLY_API_URL", "http://127.0.0.1:8765").rstrip("/")
ACCESS_TOKEN = os.getenv("VOXPOLY_ACCESS_TOKEN", "").strip()

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "voxpoly_list_voices",
        "description": "List all saved speaker voiceprints in VoxPoly Studio along with quality score and duration.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "voxpoly_speak",
        "description": "Synthesize speech using a saved voiceprint (via OpenAI-compatible /v1/audio/speech) and save to a local audio file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Spoken text to synthesize."},
                "voice": {"type": "string", "description": "Voice ID or display name (defaults to primary voice)."},
                "model": {
                    "type": "string",
                    "description": "Engine model alias: voxpoly-qwen3-1.7b, voxpoly-chatterbox-v3, or voxpoly-cosyvoice3.",
                    "default": "voxpoly-qwen3-1.7b",
                },
                "output_path": {"type": "string", "description": "Absolute output file path (.mp3 or .wav)."},
                "response_format": {"type": "string", "enum": ["mp3", "wav"], "default": "mp3"},
            },
            "required": ["text", "output_path"],
        },
    },
    {
        "name": "voxpoly_clone_multilingual",
        "description": "Translate one sentence into multiple languages and batch-clone them using a saved voiceprint, producing individual tracks plus a combined multilingual MP3 reel.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "voice_id": {"type": "string", "description": "Saved voice ID (call voxpoly_list_voices first)."},
                "source_text": {"type": "string", "description": "Source sentence or long-form script."},
                "target_langs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Language codes, e.g. ['zh', 'en', 'ja', 'fr'].",
                },
                "engine_id": {"type": "string", "enum": ["qwen3", "chatterbox", "cosyvoice3"], "default": "qwen3"},
                "mastering_preset": {"type": "string", "enum": ["raw", "podcast", "shorts", "audiobook"], "default": "podcast"},
            },
            "required": ["voice_id", "source_text", "target_langs"],
        },
    },
    {
        "name": "voxpoly_generate_podcast",
        "description": "Synthesize a multi-speaker dialogue / two-host podcast script ('[Host]: ...\\n[Guest]: ...') into one or more languages.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "script_text": {"type": "string", "description": "Dialogue lines formatted as [Speaker]: text."},
                "speaker_voices": {
                    "type": "object",
                    "description": "Map of speaker name -> voice_id, e.g. {'Host': 'demo_official'}.",
                },
                "target_langs": {"type": "array", "items": {"type": "string"}, "default": ["zh"]},
                "add_ambient_bgm": {"type": "boolean", "default": True},
            },
            "required": ["script_text", "speaker_voices"],
        },
    },
    {
        "name": "voxpoly_verify_watermark",
        "description": "Check whether a local audio or video file carries the imperceptible VoxPoly Studio AI voice watermark.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to local audio/video file."},
            },
            "required": ["file_path"],
        },
    },
]


def _headers() -> Dict[str, str]:
    return {"X-Access-Token": ACCESS_TOKEN} if ACCESS_TOKEN else {}


def _call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    with httpx.Client(timeout=300.0) as client:
        if name == "voxpoly_list_voices":
            r = client.get(f"{API_URL}/api/voices", headers=_headers())
            r.raise_for_status()
            voices = [
                {"id": v["id"], "name": v["name"], "duration_sec": v["duration_sec"], "quality_score": v.get("quality_score")}
                for v in r.json().get("voices", [])
            ]
            return {"voices": voices}

        if name == "voxpoly_speak":
            fmt = str(args.get("response_format") or "mp3")
            out_p = Path(str(args["output_path"]))
            out_p.parent.mkdir(parents=True, exist_ok=True)
            r = client.post(
                f"{API_URL}/v1/audio/speech",
                headers={**_headers(), "Content-Type": "application/json"},
                json={
                    "model": args.get("model") or "voxpoly-qwen3-1.7b",
                    "input": args["text"],
                    "voice": args.get("voice") or "alloy",
                    "response_format": fmt,
                },
            )
            r.raise_for_status()
            out_p.write_bytes(r.content)
            return {"saved_path": str(out_p), "bytes": len(r.content), "format": fmt}

        if name == "voxpoly_clone_multilingual":
            langs = list(args["target_langs"])
            tr = client.post(
                f"{API_URL}/api/translate",
                headers=_headers(),
                json={"source_text": args["source_text"], "target_langs": langs},
            )
            tr.raise_for_status()
            scripts = tr.json()["translations"]
            gen = client.post(
                f"{API_URL}/api/jobs/generate",
                headers=_headers(),
                json={
                    "voice_id": args["voice_id"],
                    "scripts": scripts,
                    "engine_id": args.get("engine_id") or "qwen3",
                    "mastering_preset": args.get("mastering_preset") or "podcast",
                    "source_text": args["source_text"],
                    "auto_reel": True,
                },
            )
            gen.raise_for_status()
            jid = gen.json()["job"]["id"]
            for _ in range(240):
                st = client.get(f"{API_URL}/api/jobs/{jid}", headers=_headers()).json()["job"]
                if st["status"] in ("completed", "partial", "failed"):
                    return {
                        "job_id": jid,
                        "status": st["status"],
                        "tracks": [
                            {"lang": t["lang_code"], "status": t["status"], "duration_sec": t.get("duration_sec"), "mp3_url": f"{API_URL}{t.get('mp3_url', '')}"}
                            for t in st["tracks"]
                        ],
                        "reel_url": f"{API_URL}{st['reel']['mp3_url']}" if st.get("reel") else None,
                        "zip_url": f"{API_URL}{st['zip_url']}",
                    }
                time.sleep(1.5)
            return {"job_id": jid, "status": "timeout"}

        if name == "voxpoly_generate_podcast":
            gen = client.post(
                f"{API_URL}/api/podcasts/generate",
                headers=_headers(),
                json={
                    "title": "MCP Podcast",
                    "script_text": args["script_text"],
                    "speaker_voices": args["speaker_voices"],
                    "target_langs": args.get("target_langs") or ["zh"],
                    "add_ambient_bgm": bool(args.get("add_ambient_bgm", True)),
                    "mastering_preset": "podcast",
                },
            )
            gen.raise_for_status()
            pid = gen.json()["podcast"]["id"]
            for _ in range(240):
                st = client.get(f"{API_URL}/api/podcasts/{pid}", headers=_headers()).json()["podcast"]
                if st["status"] in ("completed", "partial", "failed"):
                    return {"podcast_id": pid, "status": st["status"], "tracks": st["tracks"], "zip_url": f"{API_URL}{st['zip_url']}"}
                time.sleep(1.5)
            return {"podcast_id": pid, "status": "timeout"}

        if name == "voxpoly_verify_watermark":
            fp = Path(str(args["file_path"]))
            if not fp.exists():
                raise FileNotFoundError(f"File not found: {fp}")
            r = client.post(
                f"{API_URL}/api/watermark/verify",
                headers=_headers(),
                files={"file": (fp.name, fp.read_bytes(), "application/octet-stream")},
            )
            r.raise_for_status()
            return r.json()

    raise ValueError(f"Unknown tool: {name}")


def _send(msg: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}

        if method == "notifications/initialized":
            continue
        if method == "initialize":
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "voxpoly-studio", "version": "3.0.0"},
                    },
                }
            )
        elif method == "ping":
            _send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            tname = params.get("name", "")
            targs = params.get("arguments") or {}
            try:
                out = _call_tool(tname, targs)
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False, indent=2)}]},
                    }
                )
            except Exception as exc:
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"isError": True, "content": [{"type": "text", "text": f"Error calling {tname}: {exc}"}]},
                    }
                )
        elif req_id is not None:
            _send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}})


if __name__ == "__main__":
    main()
