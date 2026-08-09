#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12 is required for the isolated OCR worker." >&2
  exit 1
fi

python3.12 -m venv .venv-ocr
.venv-ocr/bin/python -m pip install --upgrade pip
.venv-ocr/bin/python -m pip install \
  'paddleocr==3.7.0' \
  'paddlex[ocr-core]==3.7.2' \
  'transformers>=5.8,<6' \
  'torch>=2.6,<3' \
  'torchvision>=0.21,<1'
.venv-ocr/bin/python -m pip check

echo "Isolated PaddleOCR worker is ready in .venv-ocr."
