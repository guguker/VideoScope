#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="$ROOT/data/demo/videoscope-demo.mp4"
mkdir -p "$(dirname "$OUTPUT")"

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "testsrc2=size=1280x720:rate=30:duration=18" \
  -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=18" \
  -map 0:v:0 -map 1:a:0 \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest -movflags +faststart \
  "$OUTPUT"

echo "$OUTPUT"

