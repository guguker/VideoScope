#!/usr/bin/env bash
set -euo pipefail

failed=0

while IFS= read -r -d '' path; do
  # A cleanup worktree may contain tracked paths already deleted but not staged yet.
  if [[ ! -e "$path" && ! -L "$path" ]]; then
    continue
  fi

  case "$path" in
    .env.example)
      ;;
    .artifact-work/*|.codex-tmp/*|.playwright-cli/*|output/*|outputs/*|\
    data/*|.venv/*|.venv-ocr/*|*/.venv/*|node_modules|*/node_modules|*/node_modules/*|\
    dist/*|*/dist/*|frontend/.vite/*|.pnpm-store/*|*/.pnpm-store/*|\
    playwright-report/*|*/playwright-report/*|test-results/*|*/test-results/*|\
    __pycache__/*|*/__pycache__/*|*.pyc|*.pyo|*.egg-info/*|\
    .pytest_cache/*|*/.pytest_cache/*|.mypy_cache/*|*/.mypy_cache/*|\
    .ruff_cache/*|*/.ruff_cache/*|*.log|*.tsbuildinfo|coverage.xml|htmlcov/*|\
    .DS_Store|*/.DS_Store|.coverage|.coverage.*|.env|.env.*)
      echo "Tracked generated or local-only path: $path" >&2
      failed=1
      ;;
  esac

  if [[ -L "$path" ]]; then
    target="$(readlink "$path")"
    if [[ "$target" = /* ]]; then
      echo "Tracked absolute symlink: $path -> $target" >&2
      failed=1
    fi
  fi
done < <(git ls-files -z)

if (( failed )); then
  exit 1
fi

echo "Repository hygiene checks passed."
