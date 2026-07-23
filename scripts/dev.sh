#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]] || [[ ! -d frontend/node_modules ]]; then
  echo "Run 'make install' first." >&2
  exit 1
fi

cleanup() {
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

.venv/bin/python -m videoscope &
BACKEND_PID=$!
pnpm --dir frontend dev &
FRONTEND_PID=$!

wait "$BACKEND_PID" "$FRONTEND_PID"

