.PHONY: install install-backend install-ml install-vision install-whisper install-ocr install-video vision-worker whisper-worker qwen-worker worker-platform-check ml-attest-offline full-ml-smoke lock-vision lock-whisper lock-ocr lock-ocr-check lock-lighthouse install-lighthouse lighthouse-worker models models-base models-ml models-vision models-whisper models-ocr models-video models-lighthouse index-visual index-visual-quality index-lighthouse dev test test-backend test-frontend build demo

install:
	./scripts/bootstrap.sh

install-backend:
	./scripts/bootstrap.sh --backend-only

install-ml: install-vision install-whisper

worker-platform-check:
	.venv/bin/python scripts/check-worker-platform.py

ml-attest-offline:
	@/usr/bin/env -i \
		HF_DATASETS_OFFLINE=1 \
		HF_HUB_OFFLINE=1 \
		PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
		PYTHONDONTWRITEBYTECODE=1 \
		PYTHONNOUSERSITE=1 \
		TRANSFORMERS_OFFLINE=1 \
		UV_OFFLINE=1 \
		"$(CURDIR)/.venv/bin/python" -I -m videoscope.ml_environment_attestation

full-ml-smoke:
	@: "$${HF_HOME:?HF_HOME must select the reviewed Hugging Face cache}"
	@: "$${VIDEOSCOPE_OCR_MODEL_ROOT:?VIDEOSCOPE_OCR_MODEL_ROOT must select the reviewed OCR model root}"
	@: "$${VIDEOSCOPE_FULL_ML_SMOKE_ROOT:?VIDEOSCOPE_FULL_ML_SMOKE_ROOT must select an existing mode-0700 disposable root}"
	@: "$${VIDEOSCOPE_FULL_ML_SMOKE_MODELS_ROOT:?VIDEOSCOPE_FULL_ML_SMOKE_MODELS_ROOT must select the external reviewed model root}"
	@: "$${VIDEOSCOPE_FULL_ML_SMOKE_VISION_PYTHON:?VIDEOSCOPE_FULL_ML_SMOKE_VISION_PYTHON must be explicit}"
	@: "$${VIDEOSCOPE_FULL_ML_SMOKE_WHISPER_PYTHON:?VIDEOSCOPE_FULL_ML_SMOKE_WHISPER_PYTHON must be explicit}"
	@: "$${VIDEOSCOPE_FULL_ML_SMOKE_OCR_PYTHON:?VIDEOSCOPE_FULL_ML_SMOKE_OCR_PYTHON must be explicit}"
	@: "$${VIDEOSCOPE_FULL_ML_SMOKE_LIGHTHOUSE_PYTHON:?VIDEOSCOPE_FULL_ML_SMOKE_LIGHTHOUSE_PYTHON must be explicit}"
	@: "$${VIDEOSCOPE_FULL_ML_SMOKE_QWEN_PYTHON:?VIDEOSCOPE_FULL_ML_SMOKE_QWEN_PYTHON must be explicit}"
	@: "$${VIDEOSCOPE_FULL_ML_SMOKE_FFMPEG_BINARY:?VIDEOSCOPE_FULL_ML_SMOKE_FFMPEG_BINARY must be explicit}"
	@: "$${VIDEOSCOPE_FULL_ML_SMOKE_FFPROBE_BINARY:?VIDEOSCOPE_FULL_ML_SMOKE_FFPROBE_BINARY must be explicit}"
	@HF_DATASETS_OFFLINE=1 \
		HF_HOME="$$HF_HOME" \
		HF_HUB_OFFLINE=1 \
		PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
		PYTHONDONTWRITEBYTECODE=1 \
		PYTHONNOUSERSITE=1 \
		TRANSFORMERS_OFFLINE=1 \
		UV_OFFLINE=1 \
		VIDEOSCOPE_DATA_DIR="$$VIDEOSCOPE_FULL_ML_SMOKE_ROOT/product" \
		VIDEOSCOPE_FULL_ML_SMOKE_MODELS_ROOT="$$VIDEOSCOPE_FULL_ML_SMOKE_MODELS_ROOT" \
		VIDEOSCOPE_FULL_ML_SMOKE_VISION_PYTHON="$$VIDEOSCOPE_FULL_ML_SMOKE_VISION_PYTHON" \
		VIDEOSCOPE_FULL_ML_SMOKE_WHISPER_PYTHON="$$VIDEOSCOPE_FULL_ML_SMOKE_WHISPER_PYTHON" \
		VIDEOSCOPE_FULL_ML_SMOKE_OCR_PYTHON="$$VIDEOSCOPE_FULL_ML_SMOKE_OCR_PYTHON" \
		VIDEOSCOPE_FULL_ML_SMOKE_LIGHTHOUSE_PYTHON="$$VIDEOSCOPE_FULL_ML_SMOKE_LIGHTHOUSE_PYTHON" \
		VIDEOSCOPE_FULL_ML_SMOKE_QWEN_PYTHON="$$VIDEOSCOPE_FULL_ML_SMOKE_QWEN_PYTHON" \
		VIDEOSCOPE_FULL_ML_SMOKE_FFMPEG_BINARY="$$VIDEOSCOPE_FULL_ML_SMOKE_FFMPEG_BINARY" \
		VIDEOSCOPE_FULL_ML_SMOKE_FFPROBE_BINARY="$$VIDEOSCOPE_FULL_ML_SMOKE_FFPROBE_BINARY" \
		VIDEOSCOPE_OCR_MODEL_ROOT="$$VIDEOSCOPE_OCR_MODEL_ROOT" \
		VIDEOSCOPE_VISION_WORKER_INPUT_ROOT="$$VIDEOSCOPE_FULL_ML_SMOKE_ROOT" \
		VIDEOSCOPE_WHISPER_WORKER_INPUT_ROOT="$$VIDEOSCOPE_FULL_ML_SMOKE_ROOT/product/media" \
		VIDEOSCOPE_WHISPER_WORKER_WORK_ROOT="$$VIDEOSCOPE_FULL_ML_SMOKE_ROOT/tmp/whisper-worker" \
		"$(CURDIR)/.venv/bin/python" -I "$(CURDIR)/scripts/full-ml-smoke.py" \
			--root "$$VIDEOSCOPE_FULL_ML_SMOKE_ROOT" \
			--models-root "$$VIDEOSCOPE_FULL_ML_SMOKE_MODELS_ROOT"

lock-vision: worker-platform-check
	.venv/bin/uv pip compile workers/vision/pyproject.toml --python .venv/bin/python --generate-hashes --output-file workers/vision/requirements.lock --custom-compile-command 'make lock-vision'

install-vision:
	@if [ -e ".venv-vision" ] || [ -L ".venv-vision" ]; then echo ".venv-vision already exists; refusing to replace it" >&2; exit 1; fi
	.venv/bin/uv venv --python 3.12.13 --managed-python .venv-vision
	.venv-vision/bin/python -I scripts/check-worker-platform.py
	.venv/bin/uv pip sync --python .venv-vision/bin/python --require-hashes --only-binary=:all: --no-python-downloads workers/vision/requirements.lock
	.venv/bin/uv pip check --python .venv-vision/bin/python

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
	@if [ -e ".venv-whisper" ] || [ -L ".venv-whisper" ]; then echo ".venv-whisper already exists; refusing to replace it" >&2; exit 1; fi
	.venv/bin/uv venv --python 3.12.13 --managed-python .venv-whisper
	.venv-whisper/bin/python -I scripts/check-worker-platform.py
	.venv/bin/uv pip sync --python .venv-whisper/bin/python --require-hashes --only-binary=:all: --no-python-downloads workers/whisper/requirements.lock
	.venv/bin/uv pip check --python .venv-whisper/bin/python

whisper-worker:
	PYTHONPATH="$(CURDIR)/backend/src" .venv-whisper/bin/python -m videoscope.providers.whisper_worker

install-ocr:
	./scripts/install-ocr.sh

install-video: worker-platform-check
	@if [ -e ".venv-qwen" ] || [ -L ".venv-qwen" ]; then echo ".venv-qwen already exists; refusing to replace it" >&2; exit 1; fi
	.venv/bin/uv venv --python 3.12.13 --managed-python .venv-qwen
	.venv-qwen/bin/python -I scripts/check-worker-platform.py
	UV_PROJECT_ENVIRONMENT="$(CURDIR)/.venv-qwen" .venv/bin/uv sync --project backend --locked --extra video --python .venv-qwen/bin/python --no-python-downloads
	.venv/bin/uv pip check --python .venv-qwen/bin/python

qwen-worker:
	.venv-qwen/bin/python -m videoscope.providers.qwen_worker

models: models-base models-vision models-whisper models-ocr models-video

models-base:
	.venv/bin/python scripts/download-models.py --profile base

models-ml: models-vision models-whisper

models-vision:
	PYTHONPATH="$(CURDIR)/backend/src" .venv-vision/bin/python scripts/download-models.py --profile vision

models-whisper:
	PYTHONPATH="$(CURDIR)/backend/src" .venv-whisper/bin/python scripts/download-models.py --profile whisper

models-ocr:
	.venv/bin/python -I scripts/download-ocr-models.py

models-video:
	.venv-qwen/bin/python scripts/download-models.py --profile video

lock-lighthouse:
	.venv/bin/uv pip compile workers/lighthouse/pyproject.toml --overrides workers/lighthouse/overrides.txt --python-version 3.11.14 --python-platform aarch64-apple-darwin --generate-hashes --output-file workers/lighthouse/requirements.lock --custom-compile-command 'make lock-lighthouse' --no-python-downloads

install-lighthouse:
	@if [ -e ".venv-lighthouse" ] || [ -L ".venv-lighthouse" ]; then echo ".venv-lighthouse already exists; refusing to replace it" >&2; exit 1; fi
	.venv/bin/uv venv --python 3.11.14 --managed-python .venv-lighthouse
	.venv-lighthouse/bin/python -I scripts/check-worker-platform.py --python-version 3.11.14
	.venv/bin/uv pip sync --python .venv-lighthouse/bin/python --require-hashes --no-python-downloads workers/lighthouse/requirements.lock
	.venv/bin/uv pip check --python .venv-lighthouse/bin/python

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
