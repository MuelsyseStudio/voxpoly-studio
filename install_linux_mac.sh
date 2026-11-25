#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

WITH_DEMUCS=0
WITH_CHATTERBOX=0
WITH_COSYVOICE3=0

for arg in "$@"; do
  case "$arg" in
    --with-demucs) WITH_DEMUCS=1 ;;
    --with-chatterbox) WITH_CHATTERBOX=1 ;;
    --with-cosyvoice3) WITH_COSYVOICE3=1 ;;
    --all) WITH_DEMUCS=1; WITH_CHATTERBOX=1; WITH_COSYVOICE3=1 ;;
    *) echo "Unknown option: $arg (supported: --with-demucs --with-chatterbox --with-cosyvoice3 --all)"; exit 1 ;;
  esac
done

echo "============================================================"
echo "  VoxPoly Studio V3.0 Pro - Linux / macOS Installer"
echo "============================================================"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[WARN] ffmpeg not found. Install via 'brew install ffmpeg' (macOS) or 'sudo apt install ffmpeg' (Ubuntu/Debian)."
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ ! -x ".venv/bin/python" ]; then
  echo "[1/3] Creating virtual environment .venv ..."
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip wheel setuptools

echo "[2/3] Detecting hardware accelerator ..."
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

if [ "$OS_NAME" = "Darwin" ]; then
  echo "Detected macOS ($ARCH_NAME) -> installing PyTorch with Apple Silicon MPS support"
  .venv/bin/pip install torch torchvision torchaudio
elif command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  echo "Detected NVIDIA GPU: $GPU_NAME"
  if echo "$GPU_NAME" | grep -Eqi "50[5-9]0|Blackwell|B100|B200"; then
    echo "Selected PyTorch CUDA 12.8"
    .venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  else
    echo "Selected PyTorch CUDA 12.4"
    .venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  fi
elif command -v rocminfo >/dev/null 2>&1 || [ -d "/opt/rocm" ]; then
  echo "Detected AMD ROCm -> installing PyTorch ROCm 6.2"
  .venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
else
  echo "No GPU detected -> installing CPU PyTorch (recommended: switch Qwen3 to 0.6B in Settings)"
  .venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "[3/3] Installing VoxPoly Studio requirements ..."
.venv/bin/pip install -r requirements.txt

if [ "$WITH_DEMUCS" -eq 1 ]; then
  echo "[Optional] Installing Demucs for neural video vocal/BGM separation ..."
  .venv/bin/pip install demucs
fi

if [ "$WITH_CHATTERBOX" -eq 1 ]; then
  echo "[Optional] Installing Chatterbox Multilingual V3 (23 languages) in isolated .venv_chatterbox ..."
  .venv/bin/python -m venv --system-site-packages .venv_chatterbox
  .venv_chatterbox/bin/pip install --ignore-installed \
    transformers==5.2.0 diffusers==0.29.0 conformer==0.3.2 \
    s3tokenizer spacy-pkuseg pykakasi==2.3.0 pyloudnorm omegaconf safetensors soundfile
  .venv_chatterbox/bin/pip install "git+https://github.com/resemble-ai/Perth.git@master"
  .venv_chatterbox/bin/pip install --no-deps chatterbox-tts
fi

if [ "$WITH_COSYVOICE3" -eq 1 ]; then
  echo "[Optional] Installing Fun-CosyVoice 3.0 in isolated .venv_cosyvoice ..."
  if [ ! -d "third_party/CosyVoice" ]; then
    mkdir -p third_party
    git clone --depth 1 --recursive https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice
  fi
  .venv/bin/python -m venv --system-site-packages .venv_cosyvoice
  .venv_cosyvoice/bin/pip install \
    conformer diffusers gdown hydra-core HyperPyYAML inflect librosa lightning \
    modelscope huggingface_hub omegaconf onnxruntime openai-whisper pyworld \
    rich soundfile wetext x-transformers
fi

echo ""
echo "Installation complete! Run ./start_linux_mac.sh to launch http://127.0.0.1:8765"
