.PHONY: install install-ml install-vision install-whisper install-ocr install-video vision-worker whisper-worker qwen-worker worker-platform-check lock-vision lock-whisper lock-ocr lock-ocr-check lock-lighthouse install-lighthouse lighthouse-worker models models-base models-ml models-vision models-whisper models-video models-lighthouse index-visual index-visual-quality index-lighthouse dev test test-backend test-frontend build demo

install:
	./scripts/bootstrap.sh

install-ml: install-vision install-whisper

worker-platform-check:
	.venv/bin/python scripts/check-worker-platform.py

lock-vision: worker-platform-check
	.venv/bin/uv pip compile workers/vision/pyproject.toml --python .venv/bin/python --generate-hashes --output-file workers/vision/requirements.lock --custom-compile-command 'make lock-vision'

install-vision:
	.venv/bin/uv venv --python 3.12.13 --managed-python .venv-vision
	.venv-vision/bin/python scripts/check-worker-platform.py
	.venv/bin/uv pip sync --python .venv-vision/bin/python --require-hashes workers/vision/requirements.lock

vision-worker:
	PYTHONPATH="$(CURDIR)/backend/src" .venv-vision/bin/python -m videoscope.providers.vision_worker

lock-whisper: worker-platform-check
	.venv/bin/uv pip compile workers/whisper/pyproject.toml --python .venv/bin/python --generate-hashes --output-file workers/whisper/requirements.lock --custom-compile-command 'make lock-whisper'

lock-ocr: worker-platform-check
	.venv/bin/uv pip compile workers/ocr/pyproject.toml --python .venv/bin/python --generate-hashes --exclude-newer 2026-08-18T00:00:00Z --output-file workers/ocr/requirements.lock --custom-compile-command 'make lock-ocr' --no-python-downloads

lock-ocr-check: worker-platform-check
	@ocr_lock_tmp="$$(mktemp -t videoscope-ocr-lock.XXXXXX)"; \
	trap 'rm -f "$$ocr_lock_tmp"' EXIT; \
	.venv/bin/uv pip compile workers/ocr/pyproject.toml --python .venv/bin/python --generate-hashes --exclude-newer 2026-08-18T00:00:00Z --output-file "$$ocr_lock_tmp" --custom-compile-command 'make lock-ocr' --no-python-downloads; \
	cmp -s workers/ocr/requirements.lock "$$ocr_lock_tmp" || { echo "workers/ocr/requirements.lock is stale; run make lock-ocr" >&2; exit 1; }

install-whisper:
	.venv/bin/uv venv --python 3.12.13 --managed-python .venv-whisper
	.venv-whisper/bin/python scripts/check-worker-platform.py
	.venv/bin/uv pip sync --python .venv-whisper/bin/python --require-hashes workers/whisper/requirements.lock

whisper-worker:
	PYTHONPATH="$(CURDIR)/backend/src" .venv-whisper/bin/python -m videoscope.providers.whisper_worker

install-ocr:
	./scripts/install-ocr.sh

install-video: worker-platform-check
	UV_PROJECT_ENVIRONMENT="$(CURDIR)/.venv-qwen" .venv/bin/uv sync --project backend --locked --extra video --python .venv/bin/python --no-python-downloads

qwen-worker:
	.venv-qwen/bin/python -m videoscope.providers.qwen_worker

models: models-base models-vision models-whisper models-video

models-base:
	.venv/bin/python scripts/download-models.py --profile base

models-ml: models-vision models-whisper

models-vision:
	PYTHONPATH="$(CURDIR)/backend/src" .venv-vision/bin/python scripts/download-models.py --profile vision

models-whisper:
	PYTHONPATH="$(CURDIR)/backend/src" .venv-whisper/bin/python scripts/download-models.py --profile whisper

models-video:
	.venv-qwen/bin/python scripts/download-models.py --profile video

lock-lighthouse:
	.venv/bin/uv pip compile workers/lighthouse/pyproject.toml --overrides workers/lighthouse/overrides.txt --python-version 3.11.14 --python-platform aarch64-apple-darwin --generate-hashes --output-file workers/lighthouse/requirements.lock --custom-compile-command 'make lock-lighthouse' --no-python-downloads

install-lighthouse:
	.venv/bin/uv venv --python 3.11.14 --managed-python .venv-lighthouse
	.venv/bin/uv pip sync --python .venv-lighthouse/bin/python --require-hashes workers/lighthouse/requirements.lock

lighthouse-worker:
	PYTHONPATH="$(CURDIR)/backend/src" .venv-lighthouse/bin/python -m videoscope.providers.lighthouse_worker

models-lighthouse:
	./scripts/download-lighthouse-models.sh

index-visual:
	.venv/bin/python scripts/index-visual.py

index-visual-quality:
	VIDEOSCOPE_SIGLIP_MODEL=google/siglip2-base-patch16-384 .venv/bin/python scripts/index-visual.py

index-lighthouse:
	.venv/bin/python scripts/index-lighthouse.py

dev:
	./scripts/dev.sh

test: test-backend test-frontend

test-backend:
	.venv/bin/python -m pytest backend/tests --cov=videoscope --cov-report=term-missing --cov-fail-under=80

test-frontend:
	pnpm --dir frontend test

build:
	pnpm --dir frontend build

demo:
	./scripts/create-demo-video.sh
