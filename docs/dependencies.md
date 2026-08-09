# Зависимости VideoScope

VideoScope разделяет базовое приложение и ресурсоёмкие ML-провайдеры. Все Python-команды ниже устанавливают пакеты в локальное окружение `.venv`; системными зависимостями остаются Python 3.12, Node.js 22, pnpm 11, FFmpeg и FFprobe.

## Профили Python

| Профиль | Команда | Назначение |
| --- | --- | --- |
| base + dev | `make install` | FastAPI, SQLite/Qdrant, PySceneDetect, тесты (включая Torch-boundary regressions) и frontend-зависимости |
| apple + roboflow + vision | `make install-ml` | Whisper MLX, локальный RF-DETR, SigLIP и Torch-стек |
| OCR worker | `make install-ocr` | PaddleOCR в изолированном `.venv-ocr`, связь по локальному JSONL stdio |
| video | `make install-video` | Локальная Qwen через MLX-VLM |
| Lighthouse | `make install-lighthouse` | Отдельно закреплённые Lighthouse и OpenAI CLIP, checkpoint QD-DETR |

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

### Почему OCR вынесен в отдельный процесс

Основное окружение закрепляет `opencv-python>=4.12,<5`, который требуется
`mlx-vlm==0.6.7`. PaddleOCR 3.7 через `paddlex[ocr-core]` требует ровно
`opencv-contrib-python==4.10.0.84`. Эти колёса нельзя безопасно установить рядом:
они публикуют один namespace `cv2`, а сами сопровождающие OpenCV прямо требуют
выбрать только один пакет. Поэтому `make install-ocr` создаёт `.venv-ocr`, а
основной backend держит единственный долгоживущий worker через ограниченный
JSONL-stdio контракт. Это сохраняет OCR и одновременно делает основной
`pip check` совместимым с Qwen. См. официальные metadata
[`mlx-vlm`](https://pypi.org/pypi/mlx-vlm/0.6.7/json),
[`paddleocr`](https://pypi.org/pypi/paddleocr/3.7.0/json) и правило выбора одного
OpenCV wheel в документации
[`opencv-contrib-python`](https://pypi.org/project/opencv-contrib-python/).

Облачный Roboflow вызывается напрямую через базовый `httpx` с ограничением
размера и строгой проверкой ответа; тяжёлый `inference-sdk` больше не является
скрытой зависимостью локального RF-DETR и не ограничивает версию OpenCV.

## Особый случай Lighthouse

Upstream Lighthouse рассчитан на другой монолитный стек и объявляет несовместимые с основным окружением ограничения: `numpy<=1.23.5` и `transformers<=4.51.3`, а также аудио- и Gradio-зависимости, которые CLIP-only путь VideoScope не использует. Поэтому `scripts/install-lighthouse.sh` устанавливает закреплённые Git-коммиты Lighthouse и OpenAI CLIP с `--no-deps`, а приложение использует изолированный адаптер `lighthouse_qdetr.py`. Официальный QD-DETR checkpoint использует pickle-совместимый legacy-формат, поэтому до его небезопасного декодирования runtime обязательно сверяет закреплённый SHA-256; иной файл отклоняется.

Кэш каждого видео содержит manifest с версией схемы, SHA-256 checkpoint и
ревизиями Lighthouse/CLIP. Несовместимые или незавершённые кэши не участвуют в
поиске. После первой установки провайдера или осознанного обновления этих
компонентов перестройте кэш существующей библиотеки командой
`make index-lighthouse`.

Это сознательное временное исключение, а не полностью согласованный dependency graph. После установки Lighthouse команда `pip check` может сообщать о его отсутствующих upstream-зависимостях и несовместимых версиях NumPy/Transformers, хотя используемый VideoScope CLIP-only путь работает без них. `open-clip-torch` пока остаётся в профиле `vision`: само приложение его не импортирует, но пакет приносит общие runtime-зависимости, необходимые OpenAI CLIP, установленному с `--no-deps`.

Надёжный следующий шаг — вынести Lighthouse в отдельное окружение/сервис либо перенести минимальный QD-DETR runtime в поддерживаемый пакет с собственным тестируемым набором зависимостей. После этого можно удалить исключение `--no-deps` и неиспользуемый `open-clip-torch`.

## Воспроизводимость

Frontend закреплён файлом `frontend/pnpm-lock.yaml`, Python —
`backend/uv.lock` для Python 3.12. Bootstrap устанавливает `uv==0.12.3` и всегда
использует точный `uv sync --locked`; CI использует тот же lock и отклоняет
рассинхронизацию с `pyproject.toml`. Поэтому повторный
`make install` также удаляет устаревшие пакеты, которых больше нет в проекте
(например, прежний in-process OCR/Roboflow stack). После такого базового reset
нужные optional-профили устанавливаются заново. Флаг `--inexact` применяется
только в `make install-ml`/`make install-video`, чтобы эти профили и вручную
установленный Lighthouse могли сосуществовать; все объявленные зависимости при
этом всё равно берутся в версиях из lock-файла.

Обновление Python-зависимостей выполняется явно:

```bash
.venv/bin/uv lock --project backend --upgrade
make test-backend
```

После изменения ML-зависимостей дополнительно проверяются соответствующие
профили и миграции производных индексов. OCR остаётся отдельным ограниченным
окружением, а Lighthouse — документированным исключением `--no-deps`; для них
нужны независимые smoke-проверки. Model revision и checksum обновляются отдельно
от package lock и проходят тесты manifest/identity.
