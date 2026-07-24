.PHONY: install install-ml install-video models install-lighthouse index-objects index-visual index-visual-quality index-speech dev test test-backend test-frontend build demo

install:
	./scripts/bootstrap.sh

install-ml:
	.venv/bin/python -m pip install -e 'backend[apple,ocr,roboflow,vision]'

install-video:
	.venv/bin/python -m pip install -e 'backend[video]'

models:
	.venv/bin/python scripts/download-models.py

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
