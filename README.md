<div align="center">

# 🎙️ VoxPoly Studio V3.0 Pro

**开源一站式多语种声纹克隆、AI 双人播客/有声书、视频一键译配与 OpenAI TTS / MCP 生态工作台**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-0D6E6E.svg)](./LICENSE)
[![Python 3.10-3.12](https://img.shields.io/badge/Python-3.10%20--%203.12-blue.svg)](#快速本地部署)
[![Engines](https://img.shields.io/badge/Engines-Qwen3--TTS%20%7C%20Chatterbox%20V3%20%7C%20CosyVoice%203-D92D20.svg)](#核心特性v30-pro)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zianxu78-ux/voxpoly-studio/blob/main/colab/VoxPoly_Colab.ipynb)

[简体中文](./README.md) · [English](./README_EN.md) · [开源选型与架构报告](./docs/GITHUB_RESEARCH_AND_ARCHITECTURE.md) · [安全与合规](./SECURITY.md)

<br/>

![VoxPoly Studio Preview](./docs/assets/studio-preview.png)

</div>

---

## ✨ 为什么做 VoxPoly Studio？

GitHub 上优秀的开源 TTS 模型很多，但大多数只停留在单句 Demo。**VoxPoly Studio V3.0 Pro** 把完整的个人与创作者有声内容生产链整合进一个干净的广播级浅色工作台，同时开放 **OpenAI 兼容 API、MCP Server、CLI 批处理与 Docker 部署**：

> **5–15 秒本人干声** → 降噪质检 + Whisper 识别 → **专有名词发音词典锁词** → **三引擎自由切换** → 一键输出 **10/23 国语言克隆语音、双人 AI 播客、PDF/EPUB 有声书、多说话人译配 MP4（带样式双语 ASS 字幕）与广播级 LUFS 母带**。

## 🚀 核心特性（V3.0 Pro）

- **🎛️ 三引擎零冲突架构（独立进程 + 切换自动清空显存）**
  - **Qwen3-TTS 12Hz Base**（`1.7B 旗舰` / `0.6B 低显存轻量` 热切换，Apache-2.0）：原生支持 10 国语言、ICL 高保真对齐与长文真批处理加速；
  - **Chatterbox Multilingual V3**（MIT）：原生覆盖 **23 国语言**（阿语/印地/土语/北欧/希伯来/马来语等），支持表现力与 CFG 口音抑制；
  - **Fun-CosyVoice 3.0**（Apache-2.0）：支持 **18+ 中文方言**（粤语、四川、东北、上海等）、自然语言情绪指令及 `[laughter]` / `[breath]` 副语言标签。
- **🎙️ 双人 AI 播客 & PDF / EPUB / 网页有声书工坊（V3 新增）**
  - 支持直接导入 **网页文章 URL、PDF、EPUB、Markdown、TXT、HTML**，自动按标题/章节拆解正文；
  - 一键送入 **单人长文有声书**（自动应用 `-18 LUFS` 柔和母带），或一键改编为 **NotebookLM 同款双人播客剧本**（支持配置大模型自由创作，无 Key 时也内置离线对话改编器）；
  - `[主持人]: ...` / `[嘉宾]: ...` 多角色剧本自动分配不同声纹档案，支持多语种双人播客合成 + 无版权温暖 Lo-Fi 氛围背景音。
- **🎬 视频一键多语种译配 Pro（V3 全面升级）**
  - **在线视频链接直抓**：粘贴 YouTube / B站 / X / TikTok 链接（via `yt-dlp`）直接下载并解析，也支持本地拖入 MP4 / MOV / WebM；
  - **零 Token 多说话人分轨（Diarization）**：无需申请 HuggingFace 受限令牌，自动基于声学特征聚类识别 `SPEAKER_1 / SPEAKER_2 / SPEAKER_3`，分别为每位说话人提取清晰原声参考并支持按角色指派声纹；
  - **3 套双语 ASS 字幕排版预设**：`横屏电影双语风`、`竖屏短视频醒目黄字`（自动适配 9:16 安全区）、`极简黑底字幕栏`，支持外挂 SRT/ASS 或直接硬压进 MP4；
  - **可选 AI 唇形同步（Lip-Sync）**：内置隔离式 GPU Worker（支持 Colab L4/A100 勾选 MuseTalk）。
- **📖 专有名词与读音纠错词典 + 🎚️ 广播级 LUFS 母带链路（V3 新增）**
  - 全局术语词典锁住品牌名/人名/缩写（如 `VoxPoly`、`K8s`、`DeepSeek`），翻译不翻车、读音不念错；
  - 内置 4 档广播级 FFmpeg DSP 母带预设（`原始干声`、`播客温暖 -16 LUFS`、`短视频高穿透 -14 LUFS`、`有声书柔和 -18 LUFS`），自动在 **隐形水印嵌入前** 执行，兼顾专业听感与 100% 水印可检出。
- **🔌 开放开发者生态：OpenAI 兼容 API + MCP Server + CLI + Docker（V3 新增）**
  - **OpenAI `/v1/audio/speech`**：零改造接入 **Cherry Studio / OpenWebUI / LobeChat / Obsidian TTS**，直接用你的真实声纹朗读 AI 回复；
  - **内置零依赖 MCP Server**：让 Antigravity、Claude Desktop、Cursor 等 AI Agent 直接调用你的声纹发声、做播客或核验水印；
  - **CLI 批处理与 Docker Compose**：一条命令夜间批转整个文件夹的电子书，或 `docker compose up -d` 生产部署。
- **🛡️ 合规防伪水印 & 显存守护**
  - `7.2–9.6 kHz` 人耳不可听扩频水印（抗母带处理与 192kbps MP3 压缩），内置查验台；支持一键「释放显存」、空闲自动回收与 `VOXPOLY_ACCESS_TOKEN` 公网访问口令。

---

## 🔌 开发者接入示例

### 1. OpenAI 兼容 TTS 接口（Cherry Studio / OpenWebUI / LobeChat）

在客户端 TTS 设置中将 **Base URL** 填为 `http://127.0.0.1:8765/v1`（云端 Colab 填 Cloudflare 地址 + `/v1`），`model` 填 `voxpoly-qwen3-1.7b`（或 `voxpoly-chatterbox-v3` / `voxpoly-cosyvoice3`），`voice` 填任意声纹 ID（如 `demo_official`）：

```bash
curl -X POST http://127.0.0.1:8765/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxpoly-qwen3-1.7b",
    "voice": "demo_official",
    "input": "你好，这是通过 OpenAI 兼容接口合成的 VoxPoly 真实声纹。",
    "response_format": "mp3"
  }' --output speech.mp3
```

### 2. MCP Server（Antigravity / Claude Desktop / Cursor）

在你的 `mcp_config.json` 中加入：

```json
{
  "mcpServers": {
    "voxpoly": {
      "command": "/path/to/voxpoly-studio/.venv/bin/python",
      "args": ["-m", "backend.mcp_server"],
      "cwd": "/path/to/voxpoly-studio",
      "env": {
        "VOXPOLY_API_URL": "http://127.0.0.1:8765"
      }
    }
  }
}
```

提供 5 个即用工具：`voxpoly_list_voices`、`voxpoly_speak`、`voxpoly_clone_multilingual`、`voxpoly_generate_podcast`、`voxpoly_verify_watermark`。

### 3. CLI 命令行批处理 & Docker GPU 部署

```bash
# 查看声纹列表
python -m backend.cli voices

# 单句多语种克隆 + 播客母带
python -m backend.cli speak --voice demo_official --text "Welcome to VoxPoly V3" --langs zh,en,ja --mastering podcast --out ./out

# 批量把整个目录的 PDF / EPUB / Markdown 转为多语种有声书
python -m backend.cli batch ./my_books --voice demo_official --langs zh,en --mastering audiobook --out ./audiobooks

# 查验音频防伪水印
python -m backend.cli verify ./out/output_zh.mp3

# NVIDIA GPU Docker 一键部署
docker compose up -d
```

---

## 💻 硬件与显卡适配矩阵

| 显卡 / 平台 | 典型型号 | 支持状态 | 自动精度 / 建议 |
| :--- | :--- | :---: | :--- |
| **NVIDIA RTX 50 系**（Blackwell `sm_120`） | 5090 / 5090 D / 5080 / 5070 Ti / 5070 | ✅ 完美支持 | `bf16 + TF32`，安装脚本自动选 PyTorch **`cu128`** |
| **NVIDIA RTX 30 / 40 系 & 计算卡**（`sm_80–90`） | 3060 / 3070 / 3080 / 3090 / 4060–4090 / L4 / A100 / H100 | ✅ 黄金主力 | `bf16 + TF32`，PyTorch `cu124` / `cu126` / `cu128` 均可 |
| **NVIDIA RTX 20 / 16 系 & T4 / V100**（`sm_70–75`） | RTX 2060–2080 Ti / GTX 1660 / Tesla T4 | ✅ 自动兼容 | 自动降级 **`fp16`**，PyTorch `cu124` |
| **Apple Silicon（macOS）** | M1 / M2 / M3 / M4（16 GB+ 统一内存） | ✅ 原生 MPS | 自动启用 **`Apple MPS`**，建议在设置页选 `0.6B` 或 `1.7B` |
| **AMD 显卡（Linux ROCm）** | RX 6700 XT / 7800 XT / 7900 XTX | ✅ Linux 支持 | 自动走 PyTorch ROCm + `sdpa`（Windows 暂不推荐） |
| **纯 CPU / 核显** | 无独立显卡设备 | ⚠️ 可运行（较慢） | 自动退回 CPU `fp32`，强烈建议在设置页切 **`Qwen3 0.6B`** |

### 显存容量参考（单引擎按需加载，切换自动清显存）

| 显存容量 | 推荐配置 |
| :--- | :--- |
| **4 – 6 GB** | 设置页将 Qwen3 切为 **`0.6B 轻量版`**（仅占 ~1.8 GB），`VOXPOLY_QWEN_BATCH_SIZE=1`，Whisper 用 `small (CPU)` |
| **8 GB** | 流畅运行 **`1.7B 旗舰版`**（占 ~4.2 GB）、Chatterbox V3 或 CosyVoice 3 单开切换，Whisper 用 `small (CPU)` |
| **12 – 16 GB** | 舒适开启长文批加速（`batch_size=3`）+ Whisper `large-v3-turbo (GPU)` + Demucs 视频人声分离 |
| **24 GB+** | 全功能满血运行（RTX 3090 / 4090 / 5090 / Colab L4 / A100） |

---

## 🚀 快速本地部署

> 前置要求：**Python 3.10 – 3.12** 与 **FFmpeg**（Windows：`winget install Gyan.FFmpeg`；macOS：`brew install ffmpeg`；Ubuntu：`sudo apt install ffmpeg`）。

### Windows 一键安装与启动

```bat
git clone https://github.com/zianxu78-ux/voxpoly-studio.git
cd voxpoly-studio
install_windows.bat
start_windows.bat
```

浏览器打开 **`http://127.0.0.1:8765/`** 即可使用。
> **中国大陆用户加速下载模型**：启动前设置 `set VOXPOLY_DOWNLOAD_SOURCE=modelscope`（PowerShell：`$env:VOXPOLY_DOWNLOAD_SOURCE="modelscope"`）。

### Linux / macOS（含 Apple Silicon M1–M4）

```bash
git clone https://github.com/zianxu78-ux/voxpoly-studio.git
cd voxpoly-studio
chmod +x install_linux_mac.sh start_linux_mac.sh
./install_linux_mac.sh            # 可按需加 --with-demucs --with-chatterbox --with-cosyvoice3 --all
./start_linux_mac.sh
```

### ☁️ Google Colab 一键云端部署（使用你自己的免费或 Pro 算力积分）

1. 点击 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zianxu78-ux/voxpoly-studio/blob/main/colab/VoxPoly_Colab.ipynb) 在你的 Google 账号中打开 [colab/VoxPoly_Colab.ipynb](./colab/VoxPoly_Colab.ipynb)（完全运行在你自己的 Colab 实例与 Google Drive 中）；
2. 选择 **`L4` / `A100`**（或 `T4`）GPU 运行时，依次运行 4 个 Cell；
3. 自动生成 Colab 官方免密 HTTPS 预览链接与 Cloudflare 公网链接（附带 `/v1` OpenAI TTS 入口）。

---

## ⚙️ 常用环境变量

| 变量名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `VOXPOLY_DOWNLOAD_SOURCE` | `hf` | 模型下载源：`hf`（HuggingFace）或 `modelscope`（国内推荐） |
| `VOXPOLY_QWEN_VARIANT` | `1.7b` | 默认 Qwen3 规格：`1.7b` 或 `0.6b`（也可在网页设置中热切换） |
| `VOXPOLY_WHISPER_MODEL` | `small` | 参考音与视频字幕识别模型：`small` / `medium` / `large-v3-turbo` |
| `VOXPOLY_QWEN_BATCH_SIZE` | `3` | Qwen3 长文批处理并发段数（低显存可设为 `1`） |
| `VOXPOLY_ACCESS_TOKEN` | *(空)* | 设值后开启网页、API 及 `/v1/audio/speech` 口令保护 |
| `VOXPOLY_DATA_DIR` | `./data` | 自定义模型、声纹库、词典与作品输出目录 |

---

## ⚖️ 开源协议与合规声明

- 本项目代码采用 **[Apache-2.0](./LICENSE)** 开源协议，底层核心引擎均为商业友好协议（`Qwen3-TTS` Apache-2.0、`Fun-CosyVoice 3.0` Apache-2.0、`Chatterbox Multilingual` MIT、`faster-whisper` MIT）。
- 请严格遵守 [SECURITY.md](./SECURITY.md) 中的负责任 AI 准则：**严禁未经本人明确授权克隆他人声音，严禁用于欺诈、冒充或违法用途**。本项目默认启用隐形防伪水印以支持内容溯源。
