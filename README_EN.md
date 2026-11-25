<div align="center">

# 🎙️ VoxPoly Studio V3.0 Pro

**Open-Source Multilingual Voice Cloning, Two-Host AI Podcast & Audiobook Studio, Multi-Speaker Video Dubbing, OpenAI TTS API & MCP Server**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-0D6E6E.svg)](./LICENSE)
[![Python 3.10-3.12](https://img.shields.io/badge/Python-3.10%20--%203.12-blue.svg)](#quick-start)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zianxu78-ux/voxpoly-studio/blob/main/colab/VoxPoly_Colab.ipynb)

[简体中文](./README.md) · [English](./README_EN.md) · [Security & Responsible AI](./SECURITY.md)

<br/>

![VoxPoly Studio Preview](./docs/assets/studio-preview.png)

</div>

## What's New in V3.0 Pro

- **3 Conflict-Free Neural Engines (Isolated Workers + Auto VRAM Reclamation)**
  - **Qwen3-TTS 12Hz Base** (`1.7B` & low-VRAM `0.6B` switcher, Apache-2.0): 10 languages, ICL high-fidelity alignment, x-vector cross-lingual cloning, and **true batched long-form synthesis**.
  - **Chatterbox Multilingual V3** (MIT): **23 languages** (Arabic, Hindi, Turkish, Dutch, Polish, Nordic languages, Hebrew, Greek, Malay, Swahili, etc.) with `exaggeration` & `cfg_weight` controls.
  - **Fun-CosyVoice 3.0** (Apache-2.0): 9 languages, 18+ Chinese dialects, natural-language style instructions, and `[laughter]` / `[breath]` tags.
- **Two-Host AI Podcast & Document-to-Audiobook Studio**
  - Import **Web URLs, PDF, EPUB, Markdown, TXT, or HTML** with automatic chapter detection.
  - Send chapters straight to long-form audiobook synthesis (`-18 LUFS`) or adapt into a **NotebookLM-style two-host podcast script** (`[Host]: ...` / `[Guest]: ...`) with per-role voiceprints, multilingual rendering, and optional copyright-free warm Lo-Fi ambient bed.
- **Video Dubbing Pro**
  - Paste **YouTube / Bilibili / X / TikTok URLs** (`yt-dlp`) or drag-and-drop local MP4/MOV/WebM.
  - **Zero-token acoustic multi-speaker diarization** automatically detects multiple speakers, extracts per-speaker reference clips, and lets you map voices per speaker.
  - **3 Bilingual Styled ASS Subtitle Presets** (`cinema`, `shorts_yellow` for 9:16 Reels/TikTok/Shorts, `minimal_bar`) + optional isolated **MuseTalk Lip-Sync GPU worker** on Colab L4/A100.
- **Pronunciation & Brand Glossary + Broadcast Mastering DSP**
  - Lock brand names and acronyms from mistranslation and set per-language phonetic TTS overrides globally.
  - 4 broadcast FFmpeg mastering chains (`raw`, `podcast -16 LUFS`, `shorts -14 LUFS`, `audiobook -18 LUFS`), applied *before* spread-spectrum watermarking so watermarks stay 100% detectable.
- **OpenAI-Compatible TTS API, Built-in MCP Server, CLI & Docker**
  - Plug `http://127.0.0.1:8765/v1` directly into **Cherry Studio, OpenWebUI, LobeChat, or Obsidian** (`/v1/audio/speech`).
  - Zero-dependency stdio **MCP Server** (`python -m backend.mcp_server`) exposing 5 tools for Antigravity, Claude Desktop, and Cursor.
  - **CLI** (`python -m backend.cli speak | batch | voices | verify`) and **NVIDIA GPU `docker-compose.yml`**.

## Quick Start

> Requirements: Python 3.10–3.12 and `ffmpeg` in PATH.

**Windows:**
```bat
git clone https://github.com/zianxu78-ux/voxpoly-studio.git
cd voxpoly-studio
install_windows.bat
start_windows.bat
```

**Linux / macOS (Apple Silicon supported):**
```bash
git clone https://github.com/zianxu78-ux/voxpoly-studio.git
cd voxpoly-studio
chmod +x install_linux_mac.sh start_linux_mac.sh
./install_linux_mac.sh            # Optional: --with-demucs --with-chatterbox --with-cosyvoice3 --all
./start_linux_mac.sh
```

**Docker (NVIDIA GPU):**
```bash
docker compose up -d
```

Open `http://127.0.0.1:8765/` in your browser.

## OpenAI API & MCP Integration

```bash
curl -X POST http://127.0.0.1:8765/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"voxpoly-qwen3-1.7b","voice":"demo_official","input":"Hello from VoxPoly V3!","response_format":"mp3"}' \
  --output hello.mp3
```

```json
{
  "mcpServers": {
    "voxpoly": {
      "command": "python",
      "args": ["-m", "backend.mcp_server"],
      "env": { "VOXPOLY_API_URL": "http://127.0.0.1:8765" }
    }
  }
}
```

## Hardware Compatibility

| Accelerator | Typical Hardware | Support | Notes |
| :--- | :--- | :---: | :--- |
| NVIDIA Blackwell (`sm_120`) | RTX 5070 / 5080 / 5090 | ✅ Full | Auto `bf16 + TF32`, PyTorch `cu128` |
| NVIDIA Ampere / Ada / Hopper | RTX 30/40 series, L4, A100, H100 | ✅ Full | Auto `bf16 + TF32`, PyTorch `cu124`+ |
| NVIDIA Turing / Volta | RTX 20 series, GTX 1660, T4, V100 | ✅ Auto | Auto `fp16` fallback |
| Apple Silicon | M1 / M2 / M3 / M4 (16 GB+ RAM) | ✅ MPS | Auto `MPS` (`0.6B` recommended on 16 GB) |
| AMD Radeon (Linux) | RX 6000 / 7000 series | ✅ ROCm | Linux ROCm 6.2 + `sdpa` |
| CPU Only | Any modern x64/ARM64 CPU | ⚠️ Slow | Switch to `Qwen3 0.6B` in Settings |

## License & Responsible Use

Licensed under [Apache-2.0](./LICENSE). Please review [SECURITY.md](./SECURITY.md): only clone your own voice or voices for which you hold explicit, documented consent.
