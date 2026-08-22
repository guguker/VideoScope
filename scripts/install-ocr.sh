#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12.13 is required for the isolated OCR worker." >&2
  exit 1
fi

if [[ "$(python3.12 -I -c 'import platform; print(platform.python_version())')" != "3.12.13" ]]; then
  echo "The reviewed OCR runtime requires exactly Python 3.12.13." >&2
  exit 1
fi

# This must run before venv creation or package installation so an unsupported
# host cannot fail only after the expensive, externally visible sync step.
python3.12 -I scripts/check-worker-platform.py

OCR_VENV="$ROOT/.venv-ocr"
if [[ -e "$OCR_VENV" || -L "$OCR_VENV" ]]; then
  echo "Refusing to reuse an existing OCR environment; move it aside and retry." >&2
  exit 1
fi

python3.12 -I -m venv "$OCR_VENV"
"$OCR_VENV/bin/python" -I -m pip --isolated --disable-pip-version-check install \
  --require-hashes \
  --only-binary=:all: \
  --no-deps \
  --requirement workers/ocr/requirements.lock
"$OCR_VENV/bin/python" -I -m pip --isolated --disable-pip-version-check check

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
