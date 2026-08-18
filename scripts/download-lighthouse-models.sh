#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="data/models/lighthouse"
QDDETR_PATH="$MODEL_DIR/clip_qd_detr_qvhighlight.ckpt"
QDDETR_URL="https://zenodo.org/records/13960580/files/clip_qd_detr_qvhighlight.ckpt?download=1"
QDDETR_SHA256="42798d352dde089a835cb4995eb9c8084a2e97337166abbbb09c12856aec2c55"
CLIP_PATH="$MODEL_DIR/ViT-B-32.pt"
CLIP_URL="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
CLIP_SHA256="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"

download_verified() {
  local destination="$1"
  local url="$2"
  local expected="$3"
  if [[ -f "$destination" ]] && [[ "$(shasum -a 256 "$destination" | awk '{print $1}')" == "$expected" ]]; then
    return
  fi
  mkdir -p "$(dirname "$destination")"
  local partial
  partial="$(mktemp "${destination}.part.XXXXXX")"
  if ! curl --fail --location --retry 3 --output "$partial" "$url"; then
    rm -f "$partial"
    return 1
  fi
  if [[ "$(shasum -a 256 "$partial" | awk '{print $1}')" != "$expected" ]]; then
    echo "Downloaded Lighthouse artifact checksum mismatch: $destination" >&2
    rm -f "$partial"
    return 1
  fi
  chmod 0600 "$partial"
  mv "$partial" "$destination"
}

download_verified "$QDDETR_PATH" "$QDDETR_URL" "$QDDETR_SHA256"
download_verified "$CLIP_PATH" "$CLIP_URL" "$CLIP_SHA256"
echo "Pinned Lighthouse model artifacts are ready in $MODEL_DIR"
