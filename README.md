# VideoScope

Локальное приложение для мультимодального поиска моментов внутри видео и сборки MP4-нарезок. Базовый сценарий универсален; спортивное видео, включая баскетбол, используется как сильный прикладной профиль, а не как жёсткое ограничение.

## Что уже реализовано

- загрузка больших видео с ограничением размера, проверкой расширения и последующей
  проверкой контейнера через FFprobe в фоновой очереди;
- durable SQLite-очередь индексации с прогрессом, отменой, повтором и
  восстановлением незавершённых попыток после перезапуска;
- сцены PySceneDetect и кадры-превью через FFmpeg;
- локальная расшифровка Whisper Large v3 Turbo через изолированный Apple MLX worker;
- опциональная аттестованная граница PaddleOCR/Transformers; worker остаётся
  недоступным без заранее предоставленных exact model bytes;
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

Требования: macOS 14+ на Apple Silicon, Python 3.12, Node.js 22, pnpm 11 и
FFmpeg. Изолированные Vision/Whisper workers закреплены точнее на Python
3.12.13 и намеренно не объявляются совместимыми с Intel Mac или Linux.

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
make install-ocr    # необязательно; требует заранее provisioned reviewed model bytes
make install-video
make models-video
make install-lighthouse   # необязательно; отдельный Python 3.11 worker
make models-lighthouse
```

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
[docs/dependencies.md](docs/dependencies.md).
Точные OCR model bytes пока не имеют воспроизводимого downloader в репозитории:
`make install-ocr` только проверяет заранее предоставленный набор из
`workers/ocr/model-artifacts.lock.json`. Подробности — в
`workers/ocr/README.md`.

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

Интерфейс: `http://127.0.0.1:5173`  
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
[docs/evidence](docs/evidence/README.md). Два исторических JSON спортивного
эксперимента сохранены в [docs/benchmarks/basketball](docs/benchmarks/basketball/README.md)
с явным описанием того, почему их нельзя считать воспроизводимым benchmark.

Evaluation harness вычисляет метрики только на локальной выборке владельца.
Универсальные и спортивные демонстрационные прогоны были слишком малы для выводов
о качестве модели; Qwen используется как дополнительное подтверждение, а не как
самостоятельный классификатор типа броска или номера игрока.

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
нужные видео. Подробности: [docs/vision-worker.md](docs/vision-worker.md) и
[docs/whisper-worker.md](docs/whisper-worker.md).

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
[docs/qwen-worker.md](docs/qwen-worker.md).

## InternVideo 2.5

Официальная модель `OpenGVLab/InternVideo2_5_Chat_8B` рассчитана на CUDA и FlashAttention, поэтому локальный Mac-путь остаётся на SigLIP 2. InternVideo подключается как отдельный GPU endpoint: VideoScope отправляет ему до восьми кадров только из четырёх лучших кандидатов и получает оценку релевантности и краткое обоснование. Сервис должен находиться в приватной сети, использовать HTTPS и обязательный API key; публичный plaintext HTTP для кадров и bearer-токена недопустим.

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
[docs/lighthouse-worker.md](docs/lighthouse-worker.md).

## Проверка

```bash
make test
make build
make demo
```

Архитектура и схема ранжирования описаны в [docs/architecture.md](docs/architecture.md).
Контракты HTTP и локального хранения — в [docs/data-contracts.md](docs/data-contracts.md).
Правила Git, evidence и очистки истории — в [docs/repository-maintenance.md](docs/repository-maintenance.md).
Эксперименты и ограничения спортивного профиля — в [docs/basketball-model-upgrade.md](docs/basketball-model-upgrade.md).
Утверждённая системная переработка артефактов, индексов, benchmark и jobs — в
[docs/system-rebuild.md](docs/system-rebuild.md).

## Лицензия

VideoScope распространяется по лицензии [MIT](LICENSE).

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
