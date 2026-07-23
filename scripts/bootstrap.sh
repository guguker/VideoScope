#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v python3.12 >/dev/null 2>&1; then
  PYTHON="$(command -v python3.12)"
elif [[ -x "$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3" ]]; then
  PYTHON="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
else
  echo "Python 3.12 is required." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e 'backend[dev]'
pnpm --dir frontend install

echo "Base environment is ready. Run 'make install-ml' for the ML providers."

