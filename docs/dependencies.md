# Зависимости VideoScope

VideoScope разделяет базовое приложение и ресурсоёмкие ML-провайдеры. Backend
использует `.venv`, SigLIP/RF-DETR — `.venv-vision`, Whisper —
`.venv-whisper`, OCR — `.venv-ocr`, Qwen — `.venv-qwen`, а Lighthouse —
`.venv-lighthouse`. Base, Vision, Whisper, OCR и Qwen воспроизводимо
поддерживаются на Apple Silicon с macOS 14+ и ровно CPython 3.12.13. Lighthouse
отдельно закреплён на CPython 3.11.14. Для bootstrap нужен `uv==0.12.3`; полный
`make install` также требует Node.js 22 и pnpm 11, а backend-only Phase 0 — нет.
FFmpeg и FFprobe нужны обоим вариантам. Intel Mac и Linux не входят в
заявленный контракт локальных Apple-ML workers.

## Профили Python

| Профиль | Команда | Назначение |
| --- | --- | --- |
| base + dev | `make install` | FastAPI, SQLite/Qdrant, PySceneDetect, тесты (включая Torch-boundary regressions) и frontend-зависимости |
| Phase 0 backend | `make install-backend` | Тот же закреплённый base + dev Python runtime без установки frontend; используется только для воспроизводимой offline-аттестации |
| Vision worker | `make install-vision` | Изолированный SigLIP 2 + RF-DETR runtime из хэшированного lock |
| Whisper worker | `make install-whisper` | Изолированный MLX Whisper runtime из хэшированного lock |
| OCR worker | `make models-ocr && make install-ocr` | PaddleOCR в изолированном `.venv-ocr`, revision-pinned model acquisition и аттестованный локальный JSONL stdio |
| Qwen worker | `make install-video` | Изолированные `.venv-qwen`, MLX-VLM и loopback HTTP worker |
| Lighthouse worker | `make install-lighthouse` | Отдельный Python 3.11 lock с закреплёнными Lighthouse/OpenAI CLIP; модели ставятся только `make models-lighthouse` |

Для clean-checkout Phase 0 окружения создаются без переиспользования старых
worker-каталогов и в фиксированном порядке:

```sh
UV_OFFLINE=1 make install-backend
make install-vision
make install-whisper
make install-video
make install-lighthouse
.venv/bin/python -I scripts/download-ocr-models.py \
  --destination /absolute/path/to/new-reviewed-ocr-model-root
VIDEOSCOPE_OCR_MODEL_ROOT=/absolute/path/to/new-reviewed-ocr-model-root \
  make install-ocr
```

До offline-проверки модели получают отдельным явно сетевым шагом: `models-base`,
`models-vision`, `models-whisper`, `models-video` запускаются с одним выбранным
абсолютным `HF_HOME`, а затем выполняется `models-lighthouse`. Полная команда с
явными cache/root bindings и порядок `make ml-attest-offline` →
`make full-ml-smoke` приведены в
[`ml-environment-attestation.md`](ml-environment-attestation.md).

`pydantic` объявлен напрямую в базовом профиле и в отдельном `deploy/internvideo/requirements.txt`, потому что оба сервиса импортируют его публичный API. `huggingface-hub` также является прямой базовой зависимостью: провайдеры проверяют закреплённые локальные snapshots, а `scripts/download-models.py` загружает их через Hub.

FastEmbed закреплён на `0.8.0`: для MPNet это фиксирует mean-pooling семантику.
Версия runtime и pooling входят в identity Qdrant collection, поэтому их осознанное
обновление автоматически требует полной перестройки текстового индекса.

OCR worker не загружает модели самостоятельно. Отдельная явная команда
`make models-ocr` связывает `workers/ocr/model-sources.lock.json` с точными
размерами и SHA-256 из `model-artifacts.lock.json`, загружает только разрешённые
файлы в приватный staging и публикует их после полной проверки. Существующий
неверный каталог не заменяется. `make install-ocr` затем выполняет только
hash-locked установку и offline startup attestation.

`make models-base`, `make models-vision`, `make models-whisper` и
`make models-video` запускаются только после установки соответствующего
окружения. Compatibility-команда `make install-ml` устанавливает два отдельных
Vision/Whisper worker, но ничего не добавляет в `.venv`. Общая команда
`make models` последовательно загружает base, Vision, Whisper, OCR и Qwen. Hugging Face
snapshots закреплены проверенными commit SHA в
`scripts/download-models.py`, чтобы повторный bootstrap не переключал модель на
новую ревизию незаметно. Runtime открывает те же commit snapshots в offline-режиме,
а идентификаторы производных Qwen/SigLIP/Qdrant-артефактов включают revision,
поэтому смена manifest не переиспользует старые оценки или векторы.

`make full-ml-smoke` ничего не устанавливает и не скачивает. Он требует
существующий mode-0700 disposable root, отдельный абсолютный model root вне
checkout, явные `HF_HOME`/OCR model root, exact `ffmpeg`/`ffprobe` и пять Python
worker paths, совпадающих с venv из environment manifest. Worker-процессы
получают только каталог этой media-пары в `PATH`, а `HOME`/`TMPDIR` находятся в
disposable root. Прямые FastEmbed/RF-DETR/Lighthouse файлы читаются из внешнего
model root в раскладке `data/models`; Hugging Face snapshots и OCR остаются в
своих явно выбранных cache/root. InternVideo не входит в локальные зависимости
этого smoke и в receipt остаётся `not_configured/provider_not_configured`.

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
Bootstrap требует внешний `uv==0.12.3`, создаёт uv-managed CPython 3.12.13 и
всегда использует точный `uv sync --locked`; CI использует тот же lock и отклоняет
рассинхронизацию с `pyproject.toml`. Поэтому повторный
`make install-video` синхронизирует `.venv-qwen` из этого же lock-файла без
`--inexact`; Qwen package graph поэтому не зависит от текущего состояния `.venv`.
Повторный `make install` также удаляет устаревшие пакеты, которых больше нет в
проекте (включая прежние in-process Whisper/Roboflow/SigLIP стеки). После такого
базового reset нужные workers устанавливаются отдельно; ни один ML target больше
не использует `--inexact` для изменения основного окружения.

Durable indexing дополнительно аттестует исполняемый base toolchain перед
созданием плана и непосредственно перед публикацией job release. В identity
входят SHA-256 reviewed `backend/uv.lock`, точные версии обязательных
распределений, реализация/patch-версия/платформа Python, а также содержимое и
bounded `-version` output разрешённых FFmpeg/FFprobe. Абсолютные пути в identity
не попадают. Аттестованный FFmpeg запускается с фиксированным пустым окружением,
поэтому ambient `PATH`, `DYLD_*`, `LD_*` и proxy-переменные не меняют уже
зафиксированный executor. Если локальная `.venv` расходится с lock-файлом,
durable indexing fail closed; штатное восстановление — `make install`
(`uv sync --locked`), а не ослабление проверки.

Обновление Python-зависимостей выполняется явно:

```bash
.venv/bin/uv lock --project backend --upgrade
make test-backend
```

После изменения ML-зависимостей дополнительно проверяются соответствующие worker
locks и миграции производных индексов. Model revision, lock identity и checksum
обновляются вместе с тестами manifest/identity.
