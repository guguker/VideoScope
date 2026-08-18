# Изолированный Qwen worker

В целевой конфигурации Qwen3.5 не загружается в процесс VideoScope backend.
`make install-video` создаёт отдельное `.venv-qwen` из `backend/uv.lock`, а
`make qwen-worker` запускает MLX-VLM только из этого окружения. Это отделяет
Qwen/OpenCV dependency graph от base, vision и OCR и сохраняет прежний
`CandidateReranker`: backend всё так же выбирает кандидатов, создаёт короткий MP4
либо JPEG-раскадровку, кэширует проверку и объединяет её с остальными сигналами.
Deprecated in-process opt-in оставлен только для rollback-диагностики и описан
ниже.

## Установка и запуск

```bash
make install
make install-video
make models-video
openssl rand -hex 32
```

Последний вывод сохраните как секрет и добавьте в `.env`:

```dotenv
QWEN_VIDEO_MODEL=mlx-community/Qwen3.5-9B-MLX-4bit
QWEN_VIDEO_ENDPOINT=http://127.0.0.1:8781
QWEN_VIDEO_API_KEY=<64-символьный hex-токен>
QWEN_WORKER_PORT=8781
QWEN_WORKER_MAX_CONCURRENCY=1
VIDEOSCOPE_QWEN_VIDEO_TIMEOUT=180
```

Сначала запустите `make qwen-worker` в отдельном терминале, затем `make dev`.
Worker читает тот же `.env`, но его адрес намеренно нельзя изменить: он всегда
слушает только `127.0.0.1`. Endpoint backend тоже принимает только plain HTTP
loopback IP origin; hostname (включая `localhost`), удалённый адрес, URL с
credentials/query/fragment и слабый ключ отклоняются при старте. IP literal не
позволяет DNS/hosts-подмене увести bearer-токен с компьютера.
Для worker-режима имя модели должно быть зарегистрировано в
`model_manifest.py` вместе с точной commit revision; произвольный
или подвижный model ID отклоняется до запуска HTTP-сервера. Pinned
Hub snapshot ищется только в локальном кэше и не может быть подменён
одноимённым относительным каталогом.
Общий input root не настраивается отдельно и всегда равен
`VIDEOSCOPE_DATA_DIR/tmp`, поэтому backend и worker не могут незаметно разойтись
по разным каталогам.

## Контракт и границы доверия

Версия контракта — `qwen-worker-v1`. Оба endpoint требуют bearer-токен:

- `GET /v1/health` сообщает точную model identity, состояние lazy-load,
  закреплённую runtime identity (`mlx-vlm==0.6.7`), допустимые типы входа и
  inference-лимиты для файла, запроса, токенов и concurrency; transport body caps
  закреплены контрактом, но не рекламируются health-ответом;
- `POST /v1/judge` принимает request ID, те же точные model/runtime identity, тип входа,
  относительный путь, фиксированный prompt kind, ограниченный query, FPS и
  `max_tokens`; неизвестные поля запрещены;
- ответ повторяет request ID и model identity и содержит типизированное Qwen
  judgement без сырого model output.

Worker дополнительно проверяет peer address и `Host`, сравнивает токен constant
time и ограничивает request JSON 64 КиБ до его разбора. Строгая response-схема
сама по себе значительно меньше 64 КиБ, а backend дополнительно отклоняет
как заявленный, так и фактический response body больше 64 КиБ до JSON-декодирования.
Входной файл ограничен 128 МиБ, query — 500 символами, generation — 1024 токенами.
Backend HTTP transport игнорирует proxy-переменные окружения и не
следует redirects, поэтому bearer-токен физически остаётся на `127.0.0.1`. По
умолчанию одновременно выполняется один inference; следующий запрос получает
`429`, а backend ограничивает ожидание `VIDEOSCOPE_QWEN_VIDEO_TIMEOUT`.

Timeout ограничивает ожидание backend, но не пытается небезопасно прервать уже
запущенный нативный MLX-вызов: занятый слот освобождается, когда inference
вернётся. Если backend к этому моменту удалил одноразовый файл, post-check worker
завершит запрос fail-closed без результата.

Медиа не передаётся в JSON. Backend создаёт одноразовый файл под `data/tmp` и
отправляет только его POSIX-relative path. Worker запрещает absolute paths,
`..`, symlinks, нештатные расширения и non-regular files, проверяет размер и
снимок `device/inode/size/mtime/ctime` до и после inference. Корень
считается локальной доверенной зоной, доступной только владельцу процесса;
изменение файла во время
inference приводит к fail-closed `409`. Внешние ответы не содержат exception,
model path или traceback.

Capability проверяется перед включением Qwen в runtime и кратко кэшируется, чтобы
опрос статуса интерфейсом не создавал HTTP-запрос на каждом обращении. Перед
каждым inference exact model и runtime identity всё равно входят в request и
проверяются worker до чтения файла. Runtime identity входит и в cache key,
поэтому обновление MLX-VLM не переиспользует оценки старой реализации.

## Миграция со старого процесса

Раньше `make install-video` добавлял `mlx-vlm` в общий `.venv`, а наличие
`QWEN_VIDEO_MODEL` автоматически включало in-process загрузку. Теперь безопасный
путь требует `QWEN_VIDEO_ENDPOINT` и `QWEN_VIDEO_API_KEY`; одно только имя модели
не включает Qwen и поиск продолжает работать без этого необязательного этапа.

Для краткой диагностики старого окружения оставлен устаревающий opt-in:

```dotenv
QWEN_VIDEO_ALLOW_IN_PROCESS=true
```

Он работает только если legacy `.venv` уже содержит совместимый `mlx-vlm`, не
может использоваться одновременно с endpoint и не устанавливается новой командой
`make install-video`. Этот режим нужен лишь как временный rollback; целевая
конфигурация — отдельный `.venv-qwen` и worker. Пользовательские видео и старые
cache-файлы при миграции не удаляются, но записи без новой runtime identity
намеренно не переиспользуются: нужные оценки безопасно вычислятся заново и будут
ключеваться точной model revision, prompt, runtime и inference configuration.
