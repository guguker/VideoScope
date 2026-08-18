# Lighthouse worker

Lighthouse не является частью процесса backend. Штатный backend содержит только
строгие схемы запроса/ответа, безопасный reader производного кэша и HTTP adapter.
Torch, OpenAI CLIP, upstream Lighthouse и legacy checkpoint decode выполняются
только отдельным процессом Python 3.11.

## Установка и запуск

```bash
make install-lighthouse
make models-lighthouse
openssl rand -hex 32
```

Один и тот же новый токен укажите в `.env` как `LIGHTHOUSE_API_KEY`, затем
настройте `LIGHTHOUSE_ENDPOINT=http://127.0.0.1:8782`. Worker запускается отдельно:

```bash
make lighthouse-worker
```

`make install-lighthouse` синхронизирует только проект
`workers/lighthouse/requirements.lock` в `.venv-lighthouse` с Python 3.11.14.
Lock создан для Apple Silicon, содержит hashes всех registry distributions,
полные Git SHA и SHA-256 двух source-архивов. Команда не устанавливает ничего в
`.venv`; если Python 3.11.14 отсутствует, `uv` отдельно подготовит именно этот
managed interpreter для `.venv-lighthouse`. `make models-lighthouse` —
единственная команда, которая обращается к хранилищам моделей; QD-DETR и
ViT-B/32 активируются только после проверки закреплённых SHA-256.
Worker не объявляет readiness, если фактический Python отличается от 3.11.14,
изменён checked-in lock либо версии ключевых runtime-пакетов расходятся с ним.

## Контракт и границы

- endpoint — только буквальный `http://127.0.0.1:<port>`; hostname, credentials,
  path, query, fragment и HTTPS отклоняются;
- сервер проверяет loopback peer, `Host` и bearer token; ошибки не раскрывают
  пути, checkpoint или traceback;
- JSON request ограничен 64 KiB до parsing, JSON response — 256 KiB; клиент
  игнорирует proxy variables и не следует redirects;
- worker принимает только относительный путь к обычному видео под `data/media`,
  отклоняет symlink/path traversal и проверяет размер, SHA-256, inode, mtime и
  ctime до и после подготовки;
- health публикует capability без загрузки модели: полный canonical
  specification, model/checkpoint identity, dependency-lock identity, limits и
  состояние artifacts;
- prepare/search передают request ID и точные specification/model/runtime
  identity; mismatch закрывается с ошибкой, а одновременно выполняется не более
  одной ML-операции.

## Поколения кэша

Для каждого видео worker создаёт новый каталог
`data/cache/lighthouse/<video>/generations/<generation>`. В нём лежат только
`manifest.json` и NumPy feature arrays с `allow_pickle=False`. Manifest включает
source SHA-256/size, полный specification, boundaries/shapes и SHA-256 каждого
окна. После записи worker повторно читает и валидирует поколение, затем одним
атомарным обновлением меняет `active.json`.

Reader допускает не более 576 окон (24 часа при шаге 150 секунд), ровно 514
признаков на кадр и не более 512 KiB на один несжатый NPZ. Manifest, active
pointer и массивы читаются ограниченными immutable snapshots; compressed ZIP,
неожиданные entries/dtype/shape и symlink-каталоги закрывают cache как stale до
вызова NumPy. Запись поколения и смена pointer сопровождаются `fsync` каталогов.

Ошибка extraction, encode, persist, validation, source mutation или activation
не меняет предыдущий `active.json`. Повреждённый либо несовместимый active cache
не превращается в «нет совпадений»: он считается stale, поэтому строгая
evaluation требует backfill. Старые `window-*.pt` не читаются новым worker и не
удаляются автоматически.

## Миграция и откат

1. Установите и запустите worker, затем убедитесь, что provider стал `ready`.
2. Выполните `make index-lighthouse`; каждое видео получит новое поколение.
3. Не удаляйте старый cache до проверки поиска и evaluation.

Для безопасного отката остановите worker и очистите `LIGHTHOUSE_ENDPOINT`:
остальной поиск продолжит работать, а существующие поколения останутся на диске.
Временный `LIGHTHOUSE_ALLOW_IN_PROCESS=true` существует только для диагностики
старого пути; его нельзя совмещать с endpoint, и он возвращает прежние риски
dependency/import boundary. После подтверждения миграции этот флаг следует
удалить отдельным изменением.
