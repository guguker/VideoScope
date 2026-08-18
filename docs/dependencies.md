# Зависимости VideoScope

VideoScope разделяет базовое приложение и ресурсоёмкие ML-провайдеры. Backend
использует `.venv`, SigLIP/RF-DETR — `.venv-vision`, Whisper —
`.venv-whisper`, OCR — `.venv-ocr`, Qwen — `.venv-qwen`, а Lighthouse —
`.venv-lighthouse`. Основному приложению нужен Python 3.12, а Vision/Whisper
workers воспроизводимо поддерживаются на Apple Silicon с macOS 14+ и ровно
Python 3.12.13. Lighthouse отдельно закреплён на Python 3.11.14. Также нужны
Node.js 22, pnpm 11, FFmpeg и FFprobe. Intel Mac и Linux не входят в заявленный
контракт локальных Apple-ML workers.

## Профили Python

| Профиль | Команда | Назначение |
| --- | --- | --- |
| base + dev | `make install` | FastAPI, SQLite/Qdrant, PySceneDetect, тесты (включая Torch-boundary regressions) и frontend-зависимости |
| Vision worker | `make install-vision` | Изолированный SigLIP 2 + RF-DETR runtime из хэшированного lock |
| Whisper worker | `make install-whisper` | Изолированный MLX Whisper runtime из хэшированного lock |
| OCR worker | `make install-ocr` | PaddleOCR в изолированном `.venv-ocr`, связь по локальному JSONL stdio |
| Qwen worker | `make install-video` | Изолированные `.venv-qwen`, MLX-VLM и loopback HTTP worker |
| Lighthouse worker | `make install-lighthouse` | Отдельный Python 3.11 lock с закреплёнными Lighthouse/OpenAI CLIP; модели ставятся только `make models-lighthouse` |

`pydantic` объявлен напрямую в базовом профиле и в отдельном `deploy/internvideo/requirements.txt`, потому что оба сервиса импортируют его публичный API. `huggingface-hub` также является прямой базовой зависимостью: провайдеры проверяют закреплённые локальные snapshots, а `scripts/download-models.py` загружает их через Hub.

FastEmbed закреплён на `0.8.0`: для MPNet это фиксирует mean-pooling семантику.
Версия runtime и pooling входят в identity Qdrant collection, поэтому их осознанное
обновление автоматически требует полной перестройки текстового индекса.

`make models-base`, `make models-vision`, `make models-whisper` и
`make models-video` запускаются только после установки соответствующего
окружения. Compatibility-команда `make install-ml` устанавливает два отдельных
Vision/Whisper worker, но ничего не добавляет в `.venv`. Общая команда
`make models` последовательно загружает эти четыре профиля. Hugging Face
snapshots закреплены проверенными commit SHA в
`scripts/download-models.py`, чтобы повторный bootstrap не переключал модель на
новую ревизию незаметно. Runtime открывает те же commit snapshots в offline-режиме,
а идентификаторы производных Qwen/SigLIP/Qdrant-артефактов включают revision,
поэтому смена manifest не переиспользует старые оценки или векторы.

### Почему ML-провайдеры вынесены в отдельные процессы

Основное vision-окружение закрепляет `opencv-python>=4.12,<5`. MLX-VLM также
приносит собственный video/OpenCV-стек, а PaddleOCR 3.7 через
`paddlex[ocr-core]` требует ровно
`opencv-contrib-python==4.10.0.84`. Эти колёса нельзя безопасно установить рядом:
они публикуют один namespace `cv2`, а сами сопровождающие OpenCV прямо требуют
выбрать только один пакет. Поэтому OCR, Qwen, Vision и Whisper имеют собственные
точные окружения. OCR использует ограниченный JSONL-stdio контракт, остальные
workers — аутентифицированные loopback-only HTTP-контракты. Paddle, MLX-VLM,
MLX Whisper, RF-DETR, TorchVision и Transformers не импортируются процессом
backend; их dependency graphs проверяются независимо. См. официальные metadata
[`mlx-vlm`](https://pypi.org/pypi/mlx-vlm/0.6.7/json),
[`paddleocr`](https://pypi.org/pypi/paddleocr/3.7.0/json) и правило выбора одного
OpenCV wheel в документации
[`opencv-contrib-python`](https://pypi.org/project/opencv-contrib-python/).

Облачный Roboflow вызывается напрямую через базовый `httpx` с ограничением
размера и строгой проверкой ответа; тяжёлый `inference-sdk` больше не является
скрытой зависимостью локального RF-DETR и не ограничивает версию OpenCV.

## Изолированные Vision и Whisper

`workers/vision/requirements.lock` и `workers/whisper/requirements.lock` содержат
полные hashes всех пакетов. `make install-vision` и `make install-whisper`
выполняют exact `uv pip sync --require-hashes`; они не изменяют `.venv`.
Установка кода и загрузка моделей разделены. Vision worker принимает только
кадры из разрешённых подкаталогов `data/`, Whisper — только media; оба копируют
вход в приватный bounded snapshot и повторно проверяют исходник после inference.

Vision worker возвращает отдельно закреплённые identity SigLIP и RF-DETR, поэтому
смена detector не инвалидирует dense-вектора. Переключение 224/384 меняет
спецификацию visual generation и требует явного `make index-visual` или
`make index-visual-quality` при остановленном API. Whisper stage использует один
immutable prompt snapshot: glossary, StageSpecification и HTTP request не могут
увидеть разные версии файла в одном прогоне.

## Изолированный Lighthouse

Upstream Lighthouse объявляет `numpy<=1.23.5` и
`transformers<=4.51.3`, несовместимые с основным Python 3.12 vision-стеком.
`workers/lighthouse/pyproject.toml` поэтому является самостоятельным Python 3.11
проектом. Он не синхронизирует `backend/pyproject.toml`, не использует `--inexact`
или `--no-deps`, не добавляет `clip`/`lighthouse` и не подменяет версии
Torch/Transformers основного vision-профиля.
Source-архивы Lighthouse и OpenAI CLIP закреплены полными Git SHA и отдельными
SHA-256; полный dependency graph фиксирует собственный hashed
`workers/lighthouse/requirements.lock`.

Установка кода и загрузка моделей разделены: `make install-lighthouse` не
скачивает модели, а `make models-lighthouse` загружает QD-DETR и ViT-B/32 во
временные файлы и активирует их только после SHA-256. Официальный QD-DETR
checkpoint использует pickle-совместимый legacy-формат; небезопасный decode
возможен только в worker после проверки обоих модельных файлов.

Новый кэш хранится как неизменяемые поколения безопасных NumPy-массивов.
Manifest включает source SHA-256, полный preprocessing/model/runtime spec и hash
каждого окна. Только полностью записанное и повторно прочитанное поколение может
атомарно заменить `active.json`; любая ошибка сохраняет прежнее поколение.
Старые `.pt` окна не удаляются автоматически, но новым runtime считаются stale.

Worker слушает только `127.0.0.1`, требует bearer token, игнорирует proxy-env,
не следует redirect, ограничивает request/response/input и допускает одну
операцию ML одновременно. Подробности: `docs/lighthouse-worker.md`.

## Воспроизводимость

Frontend закреплён файлом `frontend/pnpm-lock.yaml`, Python —
`backend/uv.lock` для Python 3.12 и отдельными hashed locks для Vision, Whisper
и Lighthouse workers.
Bootstrap устанавливает `uv==0.12.3` и всегда
использует точный `uv sync --locked`; CI использует тот же lock и отклоняет
рассинхронизацию с `pyproject.toml`. Поэтому повторный
`make install-video` синхронизирует `.venv-qwen` из этого же lock-файла без
`--inexact`; Qwen package graph поэтому не зависит от текущего состояния `.venv`.
Повторный `make install` также удаляет устаревшие пакеты, которых больше нет в
проекте (включая прежние in-process Whisper/Roboflow/SigLIP стеки). После такого
базового reset нужные workers устанавливаются отдельно; ни один ML target больше
не использует `--inexact` для изменения основного окружения.

Обновление Python-зависимостей выполняется явно:

```bash
.venv/bin/uv lock --project backend --upgrade
make test-backend
```

После изменения ML-зависимостей дополнительно проверяются соответствующие worker
locks и миграции производных индексов. Model revision, lock identity и checksum
обновляются вместе с тестами manifest/identity.
