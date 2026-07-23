#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Run 'make install' before installing Lighthouse." >&2
  exit 1
fi

LIGHTHOUSE_COMMIT="d095eaa552cecef240897a8b750306b3b2a08740"
CLIP_COMMIT="d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
MODEL_DIR="data/models/lighthouse"
CHECKPOINT="$MODEL_DIR/clip_qd_detr_qvhighlight.ckpt"
CHECKPOINT_URL="https://zenodo.org/records/13960580/files/clip_qd_detr_qvhighlight.ckpt?download=1"
CHECKPOINT_MD5="909ab9d5002e2812298b71b099b1068d"

has_commit() {
  .venv/bin/python - "$1" "$2" <<'PY'
import json
import sys
from importlib import metadata

distribution = metadata.distribution(sys.argv[1])
for entry in distribution.files or []:
    if entry.name != "direct_url.json":
        continue
    payload = json.loads(distribution.locate_file(entry).read_text())
    if payload.get("vcs_info", {}).get("commit_id") == sys.argv[2]:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

if ! has_commit lighthouse "$LIGHTHOUSE_COMMIT"; then
  .venv/bin/python -m pip install --no-deps \
    "lighthouse @ git+https://github.com/line/lighthouse.git@$LIGHTHOUSE_COMMIT"
fi

if ! has_commit clip "$CLIP_COMMIT"; then
  .venv/bin/python -m pip install --no-deps \
    "clip @ git+https://github.com/openai/CLIP.git@$CLIP_COMMIT"
fi

.venv/bin/python -m pip install "easydict>=1.13,<2"

mkdir -p "$MODEL_DIR"
if [[ ! -f "$CHECKPOINT" ]] || [[ "$(md5 -q "$CHECKPOINT")" != "$CHECKPOINT_MD5" ]]; then
  curl --fail --location --retry 3 --output "$CHECKPOINT.part" "$CHECKPOINT_URL"
  if [[ "$(md5 -q "$CHECKPOINT.part")" != "$CHECKPOINT_MD5" ]]; then
    echo "Lighthouse checkpoint checksum mismatch." >&2
    exit 1
  fi
  mv "$CHECKPOINT.part" "$CHECKPOINT"
fi

.venv/bin/python - <<'PY'
import clip

clip.load("ViT-B/32", device="cpu", jit=False)
PY

echo "Lighthouse QD-DETR is ready: $CHECKPOINT"
