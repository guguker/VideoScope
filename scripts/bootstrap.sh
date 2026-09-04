#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

BACKEND_ONLY=0
if [[ "$#" -gt 1 ]]; then
  echo "Usage: $0 [--backend-only]" >&2
  exit 2
fi
case "${1:-}" in
  "") ;;
  --backend-only) BACKEND_ONLY=1 ;;
  *)
    echo "Usage: $0 [--backend-only]" >&2
    exit 2
    ;;
esac

require_command() {
  local command_name="$1"
  local display_name="$2"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "$display_name is required but was not found in PATH." >&2
    exit 1
  fi
}

UV_VERSION="0.12.3"

if [[ -L .venv ]]; then
  echo ".venv must be a real checkout-owned directory, not a symbolic link." >&2
  exit 1
fi
if [[ -e .venv && ( ! -d .venv || ! -O .venv ) ]]; then
  echo ".venv exists but is not a checkout-owned directory; refusing to reuse it." >&2
  exit 1
fi

require_command uv "uv $UV_VERSION"
if [[ "$BACKEND_ONLY" -eq 0 ]]; then
  require_command pnpm "pnpm"
fi
require_command ffmpeg "FFmpeg"
require_command ffprobe "FFprobe"

UV_OUTPUT="$(uv --version)"
if [[ "$UV_OUTPUT" != "uv $UV_VERSION" && "$UV_OUTPUT" != "uv $UV_VERSION "* ]]; then
  echo "uv $UV_VERSION is required; found: $UV_OUTPUT" >&2
  exit 1
fi

if [[ -e .venv ]]; then
  if [[ ! -x .venv/bin/python ]]; then
    echo ".venv exists but is not a usable environment; refusing to replace it." >&2
    exit 1
  fi
else
  uv venv --python 3.12.13 --managed-python .venv
fi

.venv/bin/python -I scripts/check-worker-platform.py
UV_PROJECT_ENVIRONMENT="$ROOT/.venv" uv sync \
  --project backend \
  --locked \
  --extra dev \
  --python .venv/bin/python \
  --no-python-downloads
if [[ "$BACKEND_ONLY" -eq 0 ]]; then
  pnpm --dir frontend install --frozen-lockfile
fi

if [[ "$BACKEND_ONLY" -eq 1 ]]; then
  echo "Backend environment is ready. The frontend was intentionally not installed."
else
  echo "Base environment is ready. Install isolated providers with 'make install-vision', 'make install-whisper', 'make install-video', 'make install-ocr', or 'make install-lighthouse'."
fi
