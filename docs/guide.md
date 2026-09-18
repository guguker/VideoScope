# Руководство VideoScope

[← Обзор проекта](../README.md)

Команды в этом руководстве выполняются из корня репозитория.

Локальное приложение для мультимодального поиска моментов внутри видео и сборки MP4-нарезок. Базовый сценарий универсален; спортивное видео, включая баскетбол, используется как сильный прикладной профиль, а не как жёсткое ограничение.

## Что уже реализовано

- загрузка больших видео с ограничением размера, проверкой расширения и последующей
  проверкой контейнера через FFprobe в фоновой очереди;
- durable SQLite-очередь индексации с прогрессом, отменой, повтором и
  восстановлением незавершённых попыток после перезапуска;
- сцены PySceneDetect и кадры-превью через FFmpeg;
- локальная расшифровка Whisper Large v3 Turbo через изолированный Apple MLX worker;
- опциональная аттестованная граница PaddleOCR/Transformers с закреплёнными
  source revisions, model hashes и отдельным offline worker;
- мультиязычный поиск по кадрам через изолированный SigLIP 2 worker с быстрым 224 и quality-профилем 384;
- локальный RF-DETR Small в том же изолированном vision worker;
- legacy Roboflow Serverless adapter для пользовательских Universe-моделей;
  durable jobs намеренно не доверяют этому неаттестованному cloud boundary;
- Lighthouse QD-DETR как подтверждающий visual moment retriever с окнами до 150 секунд;
- воспроизводимый плотный SigLIP-индекс по всей временной шкале с атомарным
  переключением проверенных поколений;
- локальная Qwen3.5 9B через изолированный MLX worker как дополнительная проверка коротких видеособытий;
- InternVideo 2.5 как опциональный финальный GPU-reranker;
- локальный Qdrant с multilingual MPNet и неизменяемыми поколениями по каждому
  видео; при отказе энкодера остаётся точный лексический поиск;
- маршрутизация запросов, калиброванное объединение сигналов и объяснение каждого совпадения;
- словарь имён и терминов, русские словоформы, транслитерация и таймкоды отдельных слов;
- локальный evaluation harness с Recall@K, MRR, temporal IoU и задержкой;
- спортивный профиль для поиска трёхочковых, двухочковых и штрафных бросков;
- структурированная карточка спортивного события с проверяемыми фактами, оценками стадий и честной пометкой неподтверждённого номера игрока;
- цепочка фактически выполненных этапов поиска и предупреждение об устаревшем прогоне контрольной выборки;
- переход к точному таймкоду, очередь фрагментов и экспорт MP4.

## Запуск

Требования: macOS 14+ на Apple Silicon, `uv==0.12.3`, Node.js 22, pnpm 11,
FFmpeg и FFprobe. Bootstrap и изолированные Vision/Whisper/OCR/Qwen workers
закреплены на uv-managed CPython 3.12.13; Lighthouse — на CPython 3.11.14.
Intel Mac и Linux не входят в заявленный контракт локальных Apple-ML workers.

Минимальный запуск не требует ML-профилей:

```bash
make install
cp .env.example .env
make dev
```

После открытия интерфейса загрузите видео. Ресурсоёмкие провайдеры подключаются
отдельно и не нужны для базового запуска:

```bash
make models-base
make install-vision
make models-vision
make install-whisper
make models-whisper
make models-ocr     # явная загрузка только закреплённых reviewed model bytes
make install-ocr    # необязательно; установка и offline-проверка worker
make install-video
make models-video
make install-lighthouse   # необязательно; отдельный Python 3.11 worker
make models-lighthouse
make ml-attest-offline
```

`make ml-attest-offline` проверяет locks, Python и identities, но не запускает
инференс. Полный Phase 0 smoke требует отдельные абсолютные roots, cache и пути
всех worker-интерпретаторов; точная clean-install и offline-команда
`make full-ml-smoke` приведены в
[docs/ml-environment-attestation.md](ml-environment-attestation.md). Smoke
не скачивает модели, сам поднимает измеряемые loopback workers и не пишет в
проектный `data/`.

После установки новых провайдеров перезапустите `make dev`. Vision, Whisper,
Qwen и Lighthouse запускаются в отдельных терминалах командами
`make vision-worker`, `make whisper-worker`, `make qwen-worker` и
`make lighthouse-worker`; их ML-стеки не импортируются штатным backend.

`make index-visual`, `make index-visual-quality` и `make index-lighthouse` —
offline maintenance-команды для уже загруженной библиотеки. Перед их запуском
остановите `make dev`: команды используют тот же exclusive lock каталога `data/`
и завершаются ошибкой, пока основной API владеет хранилищем. Lighthouse worker
при этом можно оставить запущенным. После backfill снова запустите `make dev`.
Для речи и объектов используйте только действие «Переиндексировать» или
`POST /api/videos/{video_id}/reindex`; старые скрипты намеренно отключены, потому
что не публикуют проверенные поколения. На пустой библиотеке backfill не нужен.
Кэш Lighthouse привязан к checkpoint и закреплённым ревизиям реализации/CLIP,
поэтому несовместимые старые окна автоматически не читаются. Профили и известные
ограничения зависимостей описаны в
[docs/dependencies.md](dependencies.md).
`make models-ocr` получает только файлы из закреплённых Hugging Face revisions,
проверяет license evidence, размер и SHA-256 в приватном staging и не заменяет
существующий несовместимый каталог. `make install-ocr` после этого проверяет те
же bytes offline. Подробности — в [workers/ocr/README.md](../workers/ocr/README.md).

После обновления старой базы до схемы поколений ранее сохранённые сегменты не
считаются проверенными и не попадают в поиск автоматически. Для каждого нужного
видео явно нажмите «Переиндексировать»; готовая библиотека специально не
перестраивается при запуске приложения.

Загрузка и переиндексация возвращают `202 Accepted` и создают durable job.
Интерфейс продолжает опрашивать один список видео, показывает persisted stage и
progress, позволяет запросить кооперативную отмену и повторить только
`failed`/`cancelled` попытку. Источник, canonical plan и lineage повтора
сохраняются в SQLite; сырые execution tokens, локальные пути и тексты внутренних
ошибок через API не возвращаются. Для диагностики доступны
`GET /api/jobs/{job_id}`, `POST /api/jobs/{job_id}/cancel` и
`POST /api/jobs/{job_id}/retry`.

Интерфейс: `http://127.0.0.1:5173`<br>
API и OpenAPI: `http://127.0.0.1:8765/api/docs`

Данные сохраняются в `./data` и исключены из Git.

## Провайдеры

| Провайдер | Роль | Поведение без настройки |
| --- | --- | --- |
| FFmpeg | probe, кадры, клипы, монтаж | критический |
| PySceneDetect | временные границы сцен | один полный сегмент |
| Изолированный Whisper MLX worker | речь с таймкодами | поиск работает по другим сигналам |
| PaddleOCR worker | текст на экране в отдельном Python-окружении | durable stage фиксируется как `failed` с warning; job может завершиться без OCR evidence |
| Изолированный Vision worker | SigLIP 2 dense encoder + локальный RF-DETR; 384 доступен как quality-профиль | visual/object этапы отключаются |
| Roboflow Serverless | legacy cloud boundary без immutable runtime attestation | durable indexing fail-closed; используйте изолированный Vision worker |
| Qdrant + MPNet | локальный семантический индекс речи, OCR и объектов с атомарным переключением поколений | остаётся точный поиск по словам |
| Lighthouse worker | video moment retrieval в отдельном Python 3.11 процессе | нужен настроенный worker и закреплённые модели |
| Qwen3.5 9B + MLX worker | локальная проверка коротких видеособытий по последовательности кадров в отдельном процессе | этап пропускается или используется настроенный InternVideo |
| InternVideo 2.5 | внешний GPU-reranker лучших кандидатов | локальный поиск продолжает работать без него |

Локальный RF-DETR работает только в vision worker, поэтому кадры не покидают
компьютер. Hosted Roboflow — отдельная взаимоисключающая конфигурация: задайте
`ROBOFLOW_MODEL_ID` в форме `project/version` и `ROBOFLOW_API_KEY`. Автоматического
fallback из локального worker в облако нет. Статус каждого провайдера виден в
интерфейсе. Режимы `Всё`, `Речь`, `Кадр` и `Текст` позволяют явно выбрать сигнал,
а каждый результат показывает источник совпадения.

Qwen и InternVideo подключаются на этапе выполнения запроса, а не фоновой индексации. Если локальная Qwen готова, она проверяет лучшие объединённые кандидаты; в противном случае может использоваться настроенный внешний InternVideo. Ни одна из этих моделей не блокирует основной поиск.

## Точность и оценка

Автоматический маршрутизатор различает запросы об именах, речи, тексте на экране,
объектах и действиях. Для действий SigLIP ищет по воспроизводимой плотной сетке
кадров всей записи; профиль выборки, модель и preprocessing входят в identity
индекса, поэтому несовместимый или незавершённый индекс не читается. Lighthouse
влияет на выдачу только при временном совпадении с другим визуальным сигналом. В
окне «Качество» доступны локальные контрольные запросы и сравнение режимов по
Recall@1/3/5, MRR, temporal IoU и времени ответа. Для перехода на скачанный
quality-профиль остановите `make dev`, задайте модель 384, перезапустите vision
worker, выполните `make index-visual-quality`, затем снова запустите приложение.

Словарь хранится в `data/search-glossary.json`, используется лексическим поиском и добавляется в initial prompt Whisper при следующей индексации. Контрольная выборка хранится в `data/evaluation/cases.json`. Она содержит локальные `video_id`, не входит в Git и не является переносимым встроенным датасетом.
В новой установке кейсов нет; developer-набор загружается через
`PUT /api/evaluation/cases` в локальном OpenAPI UI после добавления соответствующих
видео. Кейсы принимаются только для полностью обработанных видео и не могут
выходить за их длительность; пустой набор нельзя запускать как benchmark.
Сохранённый отчёт помечается устаревшим после смены кейсов, моделей, доступности
провайдеров, словаря, поисковой конфигурации, схемы или методологии метрик.

## Evidence и метрики

Текущие тесты публикует CI; README намеренно не дублирует изменчивое число тестов
и процент покрытия. Датированные снимки локального приложения находятся в
[docs/evidence](evidence/README.md). Два исторических JSON спортивного
эксперимента сохранены в [docs/benchmarks/basketball](benchmarks/basketball/README.md)
с явным описанием того, почему их нельзя считать воспроизводимым benchmark.

Evaluation harness вычисляет метрики только на локальной выборке владельца.
Универсальные и спортивные демонстрационные прогоны были слишком малы для выводов
о качестве модели; Qwen используется как дополнительное подтверждение, а не как
самостоятельный классификатор типа броска или номера игрока.

Переносимый benchmark runner принимает все закреплённые профили
`lexical_qdrant`, `dense_siglip`, `temporal_refinement`, `lighthouse`,
`qwen_verification` и `internvideo` в режиме `warm`. Он работает только с
существующими read-only snapshot/generation и никогда не индексирует данные или
не включает fallback. Пример безопасного capability preflight:

```sh
.venv/bin/python -m videoscope.benchmark run \
  --dataset /absolute/path/to/dataset.json \
  --registry /absolute/path/to/existing-run-registry \
  --data-dir /absolute/path/to/product-data \
  --scratch-parent /absolute/path/to/existing-mode-0700-scratch \
  --run-id phase0-preflight \
  --profile dense_siglip \
  --execution-mode warm \
  --preflight
```

Preflight возвращает полную capability matrix. Код `8` означает ошибку
product execution/preflight, а `9` — недоступное или неуспешное измерение.
Предзапущенные внешние ML workers пока нельзя привязать к PID/start-token, поэтому
их benchmark-профили честно получают
`external_worker_process_binding_unavailable` по измерению; отдельный full-ML
smoke решает это self-spawn и учитывает descendants. Подробный контракт — в
[docs/benchmark-core.md](benchmark-core.md).

## Vision и Whisper workers

SigLIP/RF-DETR и MLX Whisper больше не устанавливаются в `.venv`. Создайте два
разных локальных токена, заполните endpoint-пары в `.env` и запустите workers до
индексации:

```dotenv
VIDEOSCOPE_VISION_WORKER_ENDPOINT=http://127.0.0.1:8783
VIDEOSCOPE_VISION_WORKER_API_KEY=<отдельный URL-safe токен длиной не менее 32 символов>
VIDEOSCOPE_WHISPER_WORKER_ENDPOINT=http://127.0.0.1:8784
VIDEOSCOPE_WHISPER_WORKER_API_KEY=<другой токен длиной не менее 32 символов>
```

```bash
make install-vision models-vision
make install-whisper models-whisper
make vision-worker   # терминал 1
make whisper-worker  # терминал 2
```

Workers слушают только literal `127.0.0.1`, требуют bearer token, не следуют
redirect и принимают только ограниченные относительные пути внутри своих
подкаталогов `data/`. Backend сохраняет настроенный provider даже при временной
недоступности worker: конкретная стадия честно получает `failed`, а прежнее
проверенное поколение остаётся активным. После первого подключения или смены
model/lock/preprocessing identity остановите `make dev` и явно переиндексируйте
нужные видео. Подробности: [docs/vision-worker.md](vision-worker.md) и
[docs/whisper-worker.md](whisper-worker.md).

## Qwen3.5 9B

Локальная модель `mlx-community/Qwen3.5-9B-MLX-4bit` проверяет короткие клипы и раскадровки лучших кандидатов через Metal. Для спортивного запроса она оценивает наблюдаемые факты — попытку броска, прохождение мяча через кольцо и возможный номер игрока. Окончательный результат формируется только после согласования с временным поиском и другими сигналами.

Перед выполнением модели нужен профиль `make install-video`; он создаёт отдельное
окружение `.venv-qwen` по тому же закреплённому lock-файлу. Файлы Qwen загружаются
командой `make models-video`. Сгенерируйте отдельный локальный bearer-токен, внесите
его в `.env`, затем запустите worker в отдельном терминале командой
`make qwen-worker`:

```dotenv
QWEN_VIDEO_MODEL=mlx-community/Qwen3.5-9B-MLX-4bit
QWEN_VIDEO_ENDPOINT=http://127.0.0.1:8781
QWEN_VIDEO_API_KEY=<64-символьный hex-токен>
VIDEOSCOPE_QWEN_VIDEO_TOP_CANDIDATES=12
VIDEOSCOPE_QWEN_VIDEO_FRAME_COUNT=12
```

Например, токен можно создать командой `openssl rand -hex 32`. Worker принимает
только аутентифицированные соединения с loopback, а backend передаёт ему только
относительные пути к одноразовым клипам и раскадровкам внутри `data/tmp`. Полный
контракт, ограничения и миграция описаны в
[docs/qwen-worker.md](qwen-worker.md).

## InternVideo 2.5

Официальная модель `OpenGVLab/InternVideo2_5_Chat_8B` рассчитана на CUDA и FlashAttention, поэтому локальный Mac-путь остаётся на SigLIP 2. InternVideo подключается как отдельный GPU endpoint: VideoScope отправляет ему до восьми кадров только из четырёх лучших кандидатов и получает оценку релевантности и краткое обоснование. Сервис должен находиться в приватной сети, использовать HTTPS и обязательный API key; публичный plaintext HTTP для кадров и bearer-токена недопустим.

Замороженный benchmark-профиль `internvideo` существует, но текущий локальный
benchmark и full-ML smoke намеренно фиксируют его как
`not_configured/provider_not_configured`: до source-bound runtime attestation он
не может участвовать в promotion evidence и не заменяется другой моделью.

## Lighthouse

Официальная библиотека тестировалась авторами на старом Python/CUDA-стеке и
ограничивает входное видео 150 секундами. VideoScope разрезает длинные файлы на
окна, сохраняет признаки и возвращает результаты в исходной временной шкале. Для
CPU используется `feature_name="clip"`.

Upstream требует NumPy/Transformers, несовместимые с основным Python 3.12
окружением. Поэтому `make install-lighthouse` создаёт только `.venv-lighthouse`
из отдельного Python 3.11 lock, а `make models-lighthouse` отдельно загружает два
закреплённых файла модели. Оба SHA-256 проверяются до legacy checkpoint decode;
worker принимает только аутентифицированные loopback-запросы. Без него поиск
продолжает работать по речи, OCR, объектам, сценам и multilingual-векторам.

Укажите пути в `.env`:

```dotenv
LIGHTHOUSE_CHECKPOINT=./data/models/lighthouse/clip_qd_detr_qvhighlight.ckpt
LIGHTHOUSE_CLIP_CHECKPOINT=./data/models/lighthouse/ViT-B-32.pt
LIGHTHOUSE_ENDPOINT=http://127.0.0.1:8782
LIGHTHOUSE_API_KEY=<отдельный URL-safe токен длиной не менее 32 символов>
```

Полный контракт, миграция старого кэша и откат описаны в
[docs/lighthouse-worker.md](lighthouse-worker.md).

## Проверка

```bash
make test
make build
make demo
make ml-attest-offline
```

Фаза 0 завершена на serving revision `7054f13`: сохранены полный smoke на M4 Pro,
benchmark и принудительный rollback. [Принятый отчёт](benchmarks/phase0/accepted/7054f13/README.md)
фиксирует техническую воспроизводимость; обнаруженные ограничения качества
остаются открытыми, независимый holdout и собственное обучение ещё не завершены.

Архитектура и схема ранжирования описаны в [docs/architecture.md](architecture.md).
Контракты HTTP и локального хранения — в [docs/data-contracts.md](data-contracts.md).
Правила Git, evidence и очистки истории — в [docs/repository-maintenance.md](repository-maintenance.md).
Эксперименты и ограничения спортивного профиля — в [docs/basketball-model-upgrade.md](basketball-model-upgrade.md).

Актуальный выбор локальных моделей, A/B на M4 Pro и условия fine-tune — в [docs/local-model-strategy.md](local-model-strategy.md).
Пошаговый контракт качественной ML-автономии, включая data loop, обучение,
promotion gates и rollback, — в [docs/ml-autonomy-plan.md](ml-autonomy-plan.md).
Утверждённая системная переработка артефактов, индексов, benchmark и jobs — в
[docs/system-rebuild.md](system-rebuild.md).

## Лицензия

VideoScope распространяется по лицензии [MIT](../LICENSE).

## Основные источники

- [SigLIP 2 Base 224](https://huggingface.co/google/siglip2-base-patch16-224)
- [SigLIP 2 Base 384](https://huggingface.co/google/siglip2-base-patch16-384)
- [Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Qwen3.5 9B MLX 4-bit](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
- [InternVideo 2.5](https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B)
- [Lighthouse](https://github.com/line/lighthouse)
- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- [Roboflow Inference](https://inference.roboflow.com/inference_helpers/inference_sdk/)
- [PaddleOCR](https://www.paddleocr.ai/main/en/quick_start.html)
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
- [Qdrant](https://qdrant.tech/documentation/quickstart/)
