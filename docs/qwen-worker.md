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
По умолчанию общий input root равен `VIDEOSCOPE_DATA_DIR/tmp`. Для изолированного
benchmark harness его можно явно закрепить через
`VIDEOSCOPE_QWEN_WORKER_INPUT_ROOT`; backend-клиент должен получить тот же root,
а обе стороны связывают его path-free identity и отклоняют расхождение.

## Контракт и границы доверия

Версия HTTP-контракта — `qwen-worker-v4`; составная runtime identity имеет
версию `videoscope-qwen-worker-v4`. Все три endpoint требуют bearer-токен:

- `GET /v1/health` сообщает точную model identity, состояние lazy-load,
  составную runtime identity: CPython/platform, полный exact distribution
  manifest (`mlx-vlm==0.6.7` и вся dependency closure) и полный model-artifact
  manifest; также допустимые типы входа и inference-лимиты для файла, запроса,
  токенов и concurrency; transport body caps закреплены контрактом, но не
  рекламируются health-ответом;
- `POST /v1/judge` принимает request ID, те же точные model/runtime identity, тип
  входа, относительный путь, обязательные SHA-256/byte size точного содержимого,
  фиксированный prompt kind, ограниченный query, FPS и `max_tokens`; неизвестные
  поля запрещены;
- `basketball_facts` принимает только native MP4, обязательный FPS и не допускает
  пользовательский query; `generic_query` принимает JPEG storyboard без FPS либо
  native MP4 с обязательными query и FPS. Любая другая комбинация отклоняется до
  чтения входного файла и inference;
- `POST /v1/probe` до benchmark-запуска доказывает, что backend и worker видят
  один и тот же неизменённый файл внутри одного source root, не запуская модель;
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

После каждого inference worker оставляет веса и processor загруженными, но под
тем же последовательным inference-lock синхронизирует MLX, удаляет сохранённые
Qwen request-level position/rope tensors, освобождает result/traceback-ссылки и
очищает только свободный allocator cache. Это не сбрасывает peak-memory telemetry
и не затрагивает постоянные оптимизированные веса. Если хотя бы один шаг этой
очистки не подтверждён, runtime становится unavailable до перезапуска worker:
следующий запрос не может продолжить работу на потенциально повреждённом или
неограниченно растущем Metal-состоянии.

Медиа не передаётся в JSON. Backend создаёт одноразовый файл под `data/tmp`,
читает его через descriptor-relative `O_NOFOLLOW` traversal и отправляет только
POSIX-relative path вместе с exact SHA-256/byte size. Worker повторяет такое же
descriptor-safe чтение, удерживает исходный descriptor на всё время inference и
копирует проверенные байты в случайный private-каталог с правами `0700`; MLX
получает только read-only `0400` materialization, а не mutable shared pathname.
После inference worker повторно проверяет содержимое materialization, исходный
inode и metadata всех каталогов traversal. Поэтому in-place mutation, symlink и
даже временный rename-swap с восстановлением имени приводят к fail-closed `409`;
private copy и descriptors удаляются/закрываются при любом исходе. Внешние ответы
не содержат exception, model path или traceback.

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
