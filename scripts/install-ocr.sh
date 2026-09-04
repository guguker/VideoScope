#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python || ! -x .venv/bin/uv ]]; then
  echo "The base environment is required; run 'make install' first." >&2
  exit 1
fi

# Reject an unsupported host before creating or downloading a managed runtime.
.venv/bin/python -I scripts/check-worker-platform.py --host-only

OCR_VENV="$ROOT/.venv-ocr"
if [[ -e "$OCR_VENV" || -L "$OCR_VENV" ]]; then
  echo "Refusing to reuse an existing OCR environment; move it aside and retry." >&2
  exit 1
fi

.venv/bin/uv venv --python 3.12.13 --managed-python "$OCR_VENV"
"$OCR_VENV/bin/python" -I scripts/check-worker-platform.py
.venv/bin/uv pip sync --python "$OCR_VENV/bin/python" --require-hashes \
  --only-binary=:all: \
  --no-python-downloads \
  workers/ocr/requirements.lock
.venv/bin/uv pip check --python "$OCR_VENV/bin/python"

OCR_MODEL_ROOT="${VIDEOSCOPE_OCR_MODEL_ROOT:-$HOME/.paddlex/official_models}"
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
PYTHONNOUSERSITE=1 \
VIDEOSCOPE_OCR_DEPENDENCY_LOCK="$ROOT/workers/ocr/requirements.lock" \
VIDEOSCOPE_OCR_MODEL_MANIFEST="$ROOT/workers/ocr/model-artifacts.lock.json" \
VIDEOSCOPE_OCR_MODEL_ROOT="$OCR_MODEL_ROOT" \
  "$OCR_VENV/bin/python" -I "$ROOT/scripts/paddle-ocr-worker.py" --attest-only >/dev/null

echo "Attested isolated PaddleOCR worker is ready in .venv-ocr."
