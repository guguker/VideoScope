#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

require_command() {
  local command_name="$1"
  local display_name="$2"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "$display_name is required but was not found in PATH." >&2
    exit 1
  fi
}

require_command python3.12 "Python 3.12"
require_command pnpm "pnpm"
require_command ffmpeg "FFmpeg"
require_command ffprobe "FFprobe"

PYTHON="$(command -v python3.12)"
UV_VERSION="0.12.3"

if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON" -m venv .venv
fi

.venv/bin/python -m pip install "uv==$UV_VERSION"
UV_PROJECT_ENVIRONMENT="$ROOT/.venv" .venv/bin/uv sync \
  --project backend \
  --locked \
  --extra dev
pnpm --dir frontend install --frozen-lockfile

echo "Base environment is ready. Install isolated providers with 'make install-vision', 'make install-whisper', 'make install-video', 'make install-ocr', or 'make install-lighthouse'."
