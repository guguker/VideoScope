#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE="$(mktemp -d "${TMPDIR:-/tmp}/videoscope-hygiene.XXXXXX")"
trap 'rm -rf "$FIXTURE"' EXIT

git -C "$FIXTURE" init --quiet
mkdir -p "$FIXTURE/scripts"
cp "$ROOT/scripts/check-repository-hygiene.sh" "$FIXTURE/scripts/"
printf '%s\n' '# fixture' > "$FIXTURE/README.md"
git -C "$FIXTURE" add README.md scripts/check-repository-hygiene.sh
(cd "$FIXTURE" && bash scripts/check-repository-hygiene.sh >/dev/null)

for path in \
  data/video.mp4 \
  .venv/pyvenv.cfg \
  .venv-ocr/pyvenv.cfg \
  frontend/dist/index.html \
  test-results/result.json \
  playwright-report/index.html \
  .pnpm-store/index.json \
  .pytest_cache/root-entry \
  .mypy_cache/root-entry \
  .ruff_cache/root-entry \
  backend/src/videoscope/__pycache__/api.pyc \
  .env.production
do
  mkdir -p "$FIXTURE/$(dirname "$path")"
  printf '%s\n' 'generated' > "$FIXTURE/$path"
  git -C "$FIXTURE" add --force "$path"
  if (cd "$FIXTURE" && bash scripts/check-repository-hygiene.sh >/dev/null 2>&1); then
    echo "Hygiene check accepted forbidden fixture: $path" >&2
    exit 1
  fi
  git -C "$FIXTURE" rm --cached --quiet -- "$path"
  rm -f "$FIXTURE/$path"
done

echo "Repository hygiene fixture checks passed."
