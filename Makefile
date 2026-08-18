.PHONY: install install-ml install-ocr install-video qwen-worker models models-ml models-video install-lighthouse index-objects index-visual index-visual-quality index-speech index-lighthouse dev test test-backend test-frontend build demo

install:
	./scripts/bootstrap.sh

install-ml:
	UV_PROJECT_ENVIRONMENT="$(CURDIR)/.venv" .venv/bin/uv sync --project backend --locked --inexact --extra dev --extra apple --extra roboflow --extra vision

install-ocr:
	./scripts/install-ocr.sh

install-video:
	UV_PROJECT_ENVIRONMENT="$(CURDIR)/.venv-qwen" .venv/bin/uv sync --project backend --locked --extra video

qwen-worker:
	.venv-qwen/bin/python -m videoscope.providers.qwen_worker

models: models-ml models-video

models-ml:
	.venv/bin/python scripts/download-models.py --profile ml

models-video:
	.venv-qwen/bin/python scripts/download-models.py --profile video

install-lighthouse:
	./scripts/install-lighthouse.sh

index-objects:
	.venv/bin/python scripts/index-objects.py

index-visual:
	.venv/bin/python scripts/index-visual.py

index-visual-quality:
	VIDEOSCOPE_SIGLIP_MODEL=google/siglip2-base-patch16-384 .venv/bin/python scripts/index-visual.py

index-speech:
	.venv/bin/python scripts/index-speech.py

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
