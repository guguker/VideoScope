# Зависимости VideoScope

VideoScope разделяет базовое приложение и ресурсоёмкие ML-провайдеры. Базовый и
vision-профили используют `.venv`, OCR — `.venv-ocr`, Qwen — `.venv-qwen`, а
Lighthouse — `.venv-lighthouse`. Основному приложению нужен Python 3.12;
Lighthouse отдельно закреплён на Python 3.11. Также нужны Node.js 22, pnpm 11,
FFmpeg и FFprobe.

## Профили Python

| Профиль | Команда | Назначение |
| --- | --- | --- |
| base + dev | `make install` | FastAPI, SQLite/Qdrant, PySceneDetect, тесты (включая Torch-boundary regressions) и frontend-зависимости |
| apple + roboflow + vision | `make install-ml` | Whisper MLX, локальный RF-DETR, SigLIP и Torch-стек |
| OCR worker | `make install-ocr` | PaddleOCR в изолированном `.venv-ocr`, связь по локальному JSONL stdio |
| Qwen worker | `make install-video` | Изолированные `.venv-qwen`, MLX-VLM и loopback HTTP worker |
| Lighthouse worker | `make install-lighthouse` | Отдельный Python 3.11 lock с закреплёнными Lighthouse/OpenAI CLIP; модели ставятся только `make models-lighthouse` |

`pydantic` объявлен напрямую в базовом профиле и в отдельном `deploy/internvideo/requirements.txt`, потому что оба сервиса импортируют его публичный API. `huggingface-hub` также является прямой базовой зависимостью: провайдеры проверяют закреплённые локальные snapshots, а `scripts/download-models.py` загружает их через Hub.

FastEmbed закреплён на `0.8.0`: для MPNet это фиксирует mean-pooling семантику.
Версия runtime и pooling входят в identity Qdrant collection, поэтому их осознанное
обновление автоматически требует полной перестройки текстового индекса.

`make models-ml` запускается после `make install-ml`, а `make models-video` — после
`make install-video`. Общая команда `make models` последовательно загружает оба
профиля. Hugging Face snapshots закреплены проверенными commit SHA в
`scripts/download-models.py`, чтобы повторный bootstrap не переключал модель на
новую ревизию незаметно. Runtime открывает те же commit snapshots в offline-режиме,
а идентификаторы производных Qwen/SigLIP/Qdrant-артефактов включают revision,
поэтому смена manifest не переиспользует старые оценки или векторы.

### Почему OCR и Qwen вынесены в отдельные процессы

Основное vision-окружение закрепляет `opencv-python>=4.12,<5`. MLX-VLM также
приносит собственный video/OpenCV-стек, а PaddleOCR 3.7 через
`paddlex[ocr-core]` требует ровно
`opencv-contrib-python==4.10.0.84`. Эти колёса нельзя безопасно установить рядом:
они публикуют один namespace `cv2`, а сами сопровождающие OpenCV прямо требуют
выбрать только один пакет. Поэтому `make install-ocr` создаёт `.venv-ocr`, а
`make install-video` — отдельное `.venv-qwen`. OCR использует ограниченный
JSONL-stdio контракт, Qwen — аутентифицированный loopback-only HTTP-контракт.
Ни Paddle, ни MLX-VLM больше не должны импортироваться процессом backend; их
dependency graphs проверяются независимо. См. официальные metadata
[`mlx-vlm`](https://pypi.org/pypi/mlx-vlm/0.6.7/json),
[`paddleocr`](https://pypi.org/pypi/paddleocr/3.7.0/json) и правило выбора одного
OpenCV wheel в документации
[`opencv-contrib-python`](https://pypi.org/project/opencv-contrib-python/).

Облачный Roboflow вызывается напрямую через базовый `httpx` с ограничением
размера и строгой проверкой ответа; тяжёлый `inference-sdk` больше не является
скрытой зависимостью локального RF-DETR и не ограничивает версию OpenCV.

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
`backend/uv.lock` для Python 3.12 и hashed
`workers/lighthouse/requirements.lock` для Python 3.11.14 на Apple Silicon.
Bootstrap устанавливает `uv==0.12.3` и всегда
использует точный `uv sync --locked`; CI использует тот же lock и отклоняет
рассинхронизацию с `pyproject.toml`. Поэтому повторный
`make install-video` синхронизирует `.venv-qwen` из этого же lock-файла без
`--inexact`; Qwen package graph поэтому не зависит от текущего состояния `.venv`.
Повторный `make install` также удаляет устаревшие пакеты, которых больше нет в проекте
(например, прежний in-process OCR/Roboflow stack). После такого базового reset
нужные optional-профили устанавливаются заново. Флаг `--inexact` применяется
только в `make install-ml`; Lighthouse больше не меняет `.venv` и не зависит от
её текущего состояния.

Обновление Python-зависимостей выполняется явно:

```bash
.venv/bin/uv lock --project backend --upgrade
make test-backend
```

После изменения ML-зависимостей дополнительно проверяются соответствующие
профили и миграции производных индексов. OCR, Qwen и Lighthouse имеют отдельные
ограниченные окружения. Model revision, lock identity и checksum обновляются
вместе с тестами manifest/identity.
