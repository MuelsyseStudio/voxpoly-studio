#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "[ERROR] .venv not found. Run ./install_linux_mac.sh first."
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
[ -x ".venv_chatterbox/bin/python" ] && export VOXPOLY_CHATTERBOX_PYTHON="$(pwd)/.venv_chatterbox/bin/python"
[ -x ".venv_cosyvoice/bin/python" ] && export VOXPOLY_COSYVOICE_PYTHON="$(pwd)/.venv_cosyvoice/bin/python"
[ -d "third_party/CosyVoice" ] && export VOXPOLY_COSYVOICE_REPO="$(pwd)/third_party/CosyVoice"

PORT="${PORT:-8765}"
echo "Starting VoxPoly Studio V2.0 Pro at http://127.0.0.1:${PORT} ..."
exec .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port "$PORT"
