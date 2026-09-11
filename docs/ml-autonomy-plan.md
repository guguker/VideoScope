# План-инструкция по качественной ML-автономии VideoScope

Статус: утверждаемый рабочий контракт для следующих ML-итераций.

Дата фиксации: 2026-09-04.
Связанные решения: [исходные пользовательские сценарии](user-journeys.md),
[архитектура](architecture.md), [системная переработка](system-rebuild.md),
[локальная модельная стратегия](local-model-strategy.md) и
[баскетбольный профиль](basketball-model-upgrade.md).

Этот документ адресован в первую очередь следующему разработчику или агенту,
который будет продолжать ML-часть проекта. Он задаёт не список интересных моделей,
а порядок действий, обязательные артефакты, проверяемые ворота качества и условия
остановки.

## CAPABILITY

### Исходное видение, которое нельзя потерять

VideoScope превращает длинные пользовательские видео в локальную поисковую
библиотеку: человек формулирует естественный запрос, получает ранжированные и
объяснимые моменты с точными временными границами, сразу просматривает их, при
необходимости правит границы и собирает готовую MP4-нарезку. Баскетбол — сложный
испытательный доменный профиль, но не новое ограничение продукта.

Поэтому цель ML — улучшать весь путь
`запрос → найденный момент → точные границы → понятное доказательство → нарезка`,
а не отдельно training loss, accuracy классификатора или размер модели.

### Что здесь означает «качественная ML-автономия»

Это способность VideoScope локально и воспроизводимо выполнять замкнутый цикл:

```text
пользовательские исправления и разрешённые внешние данные
    → проверенная разметка и неизменяемый dataset snapshot
    → воспроизводимое обучение кандидата
    → независимая end-to-end оценка на целых невиденных видео
    → registration и shadow/canary
    → явное promotion-решение
    → наблюдение и мгновенный rollback
    → отбор следующих примеров по реальным ошибкам
```

Автономными должны стать механические части цикла: построение снимков данных,
запуск экспериментов, вычисление метрик, формирование отчёта, регистрация
кандидата, shadow-проверка, мониторинг и подготовка отката. Источники данных,
политика качества, спорная gold-разметка и смена активной модели пока остаются под
явным контролем владельца.

### Целевой уровень

| Уровень | Состояние | Критерий |
| --- | --- | --- |
| L0 | Inference orchestration | Внешние готовые модели вызываются по правилам, но своего training lifecycle нет. Это примерно текущее состояние. |
| L1 | Reproducible learning | Из sealed dataset одной командой обучается и повторно оценивается первый собственный компонент. |
| L2 | Assisted data loop | Система ищет informative/hard-negative примеры, человек подтверждает метки, новый snapshot строится автоматически. |
| L3 | Quality autonomy | Retraining, evaluation, registry, shadow/canary, отчёт и rollback автоматизированы; promotion требует явного решения владельца. |
| L4 | Bounded auto-promotion | Автопродвижение разрешено только после длинной истории стабильных L3-релизов и отдельного решения. Сейчас это не цель. |

Цель текущей программы — **L3**, а не обещание «модель учится сама без человека».

На дату этого решения VideoScope находится на L0 с важной частью фундамента для
L1: уже есть durable jobs, immutable artifact generations, model/runtime identity,
rollback-oriented publication и переносимый benchmark core. Ещё нет независимого
promotion dataset, единого data/train contract, project-trained checkpoint,
registry обученных releases, shadow/canary и замкнутого feedback loop. То есть
инфраструктурная основа сильная, но собственное обучение пока не началось.

### Definition of Done для L3

Из чистого checkout одна документированная последовательность команд должна:

1. проверить provenance, права, schema и хеши dataset snapshot;
2. доказать отсутствие split leakage;
3. обучить небольшой VideoScope-компонент с закреплённой внешней основой;
4. выпустить immutable `safetensors`-совместимый bundle с model/data card;
5. прогнать frozen end-to-end benchmark на невиденных целых видео;
6. вернуть машинное решение `eligible` либо `reject` с причинами;
7. запустить ровно этот artifact offline на целевом M4 Pro;
8. включить его в shadow/canary без изменения активного fallback;
9. доказать улучшение предметного профиля без ухудшения универсального поиска;
10. одним атомарным переключением вернуться к предыдущему поколению.

## CONSTRAINTS

### Небыблемые инварианты

1. **Пользовательский результат важнее модельной метрики.** Классификация события
   без полезных `start/end` не считается успехом.
2. **Продукт остаётся универсальным.** Любой sports-компонент является отключаемым
   domain pack поверх общего retrieval, а не отдельным обязательным индексатором.
3. **Local-first и владение данными.** Исходники, индексы, разметка и обученные
   артефакты остаются на устройстве по умолчанию. Никакой файл пользователя не
   становится training data без явного согласия.
4. **Рабочий fallback обязателен.** Ошибка, отсутствие или отклонение новой модели
   не ломают загрузку, общий поиск, просмотр и экспорт.
5. **Одна модель не является истиной.** Retrieval, temporal signal,
   detector/tracker, OCR и verifier дают независимые наблюдения; при конфликте
   система должна честно abstain.
6. **Evidence сохраняется.** Learned score не заменяет временной интервал,
   contributing signals, версию модели и объяснение результата.
7. **Экзамен изолирован от обучения.** Split выполняется по целым источникам,
   матчам, событиям и replay-группам; перекрывающиеся окна и производные crops
   одного события не расходятся по split.
8. **Pseudo-label не равен gold.** Teacher-модель и текущие эвристики могут
   предлагать `silver`-кандидаты, но не создают promotion ground truth.
9. **Продвижение доказательное и обратимое.** Candidate строится рядом с active,
   проходит frozen benchmark и может быть включён только при наличии рабочего
   rollback без переиндексации исходных видео.
10. **Артефакты воспроизводимы и адресуются содержимым.** Dataset, код,
    base-model revision, preprocessing, lockfile, config, seed, hardware, метрики и
    выходные SHA входят в identity прогона.
11. **Пользовательское состояние не перезаписывается.** Исходники, labels,
    glossary и готовые exports не удаляются ML-job; новые поколения публикуются
    атомарно после валидации.
12. **Автономия ограничена безопасностью и реальным железом.** Никаких плавающих
    ревизий, тихих скачиваний в production, непроверенного remote code, неизвестных
    pickle-весов или конфигураций с Metal OOM/sustained swap.

Конкретные модели, размер окна и loss-функции менять можно. Эти инварианты — нельзя.

### Ограничения данных и прав

- Текущие изученные MBA/интервью-ролики остаются только `regression_seen` и не
  доказывают обобщение.
- BARD, E-BARD Detection и E-BARD JNR рассматриваются как кандидаты для
  исследования только после локального аудита состава, дублей, annotation schema,
  attribution и прав на исходные трансляции. Одна пометка `CC-BY-4.0` на dataset
  card не доказывает права на все вложенные видео.
- BARD-поднаборы из одних и тех же матчей считаются единым leakage domain: test
  матча нельзя использовать для ranker, detector, OCR/JNR или teacher tuning.
- Приватные пользовательские кадры, аудио, OCR, embeddings и crops нельзя
  отправлять в Hugging Face Jobs или публиковать без отдельного opt-in и
  подтверждённого права на материал.
- Отзыв согласия блокирует использование соответствующего source в новых
  snapshots; уже выпущенный artifact помечается для impact review.

### Ограничения runtime

- Production target: Apple M4 Pro, 24 ГБ unified memory, полностью offline inference.
- Начальный бюджет всего процесса: не более 16 ГиБ peak, без Metal OOM и
  устойчивого роста swap; точное значение можно изменить только по измерениям.
- Интерактивный ranker не должен добавлять более 2 с p95 и более 25% к p95
  текущего поиска. Медленный verifier допустим лишь в явно выбранном
  quality/background mode.
- CUDA-обученный artifact не считается готовым, пока экспортированный и
  квантизованный serving-вариант не прошёл тот же benchmark на Metal.
- Production worker не скачивает код или веса во время запроса. Model ID,
  revision, SHA-256 и формат фиксируются заранее; `safetensors` предпочтителен.

## IMPLEMENTATION CONTRACT

### Целевая архитектура

```text
                             ┌──────────────────────────────┐
запрос → универсальный       │ speech / OCR / objects /     │
         high-recall search ─┤ scenes / MPNet / SigLIP2     │
                             └──────────────┬───────────────┘
                                            │ top-50/100 + окна
                                            ▼
                             VideoScope Temporal Ranker
                             (наши небольшие веса)
                                            │ top-12
                                            ▼
                             optional domain evidence pack
                             detector/tracker/OCR/geometry
                                            │
                                            ▼
                             optional Qwen fact verifier
                                            │
                                            ▼
                             fusion + abstain + boundaries
                                            │
                                            ▼
                             объяснимый момент → MP4 clip
```

Универсальный retrieval остаётся генератором кандидатов и fallback. Первая «наша
модель» — query-conditioned temporal ranker поверх уже доступных признаков, а не
foundation VLM с нуля. Это даёт общий train/serve preprocessing, малую стоимость
итерации и прямую оптимизацию ранжирования моментов.

### Портфель моделей и порядок экспериментов

| Приоритет | Компонент | Внешняя основа | Что обучаем | Условие перехода |
| ---: | --- | --- | --- | --- |
| 0 | Plumbing control | frozen SigLIP2 + текущие признаки | logistic/GBDT или линейную голову | Доказывает корректность dataset/train/eval/serve, но не обязательно идёт в production. |
| 1 | `VideoScope Temporal Ranker` | frozen SigLIP2, MPNet и доступные evidence features | temporal/query-conditioned head: цель 2–10M trainable parameters, потолок v0 — 25M | Первый production-кандидат после independent holdout. |
| 2 | X-CLIP challenger | `microsoft/xclip-base-patch32@revision` | сначала projection/temporal head; последние encoder blocks — только при достаточных данных | Только если error analysis показывает недостаточную temporal capacity ranker baseline. |
| 3 | `VideoScope Court Detector` | RF-DETR Small | ball, rim, backboard, player, referee | Отдельный component benchmark и положительный end-to-end вклад. |
| 4 | Tracking/geometry pack | ByteTrack/OC-SORT + правила | преимущественно калибровку и fusion, не большую модель | Когда detector recall достаточен для траекторий. |
| 5 | Jersey/scoreboard pack | небольшой OCR/classifier, E-BARD JNR как исследовательский источник | multi-frame classifier/voting с `unreadable` | Номер никогда не публикуется без abstain и временной устойчивости. |
| 6 | Qwen adapter | закреплённый Qwen 9B MLX-compatible base | короткий QLoRA с frozen vision path сначала | Только если ablation доказал, что bottleneck — verifier, а не candidate recall/геометрия/метки. |

Большая модель или «более свежий checkpoint» не является следующим шагом сама по
себе. EBQwen или другая готовая спортивная VLM может быть challenger/teacher для
silver-разметки, но не заменяет detector, gold labels и независимый benchmark.

Первый dataset будет преимущественно баскетбольным, поэтому первые обученные веса
ranker включаются только внутри basketball domain pack. Архитектура, prediction
contract и training pipeline делаются универсальными; глобально заменять общий
ranker можно лишь после отдельного multi-domain holdout.

Исходные HF-кандидаты и их роль до аудита:

| Hub ID | Разрешённая роль в плане | Не разрешено считать доказанным заранее |
| --- | --- | --- |
| `google/siglip2-base-patch16-224` | Текущий frozen encoder и самый дешёвый control | Что покадровых признаков достаточно для движения |
| `microsoft/xclip-base-patch32` | Video-text challenger после Phase-2 baseline | Что англоязычный pretrained backbone улучшит русский/local-domain поиск |
| `Roboflow/rf-detr-small` | Основа отдельного court detector | Что общий checkpoint уже видит маленький мяч/кольцо с нужным recall |
| `GabrieleGiudici/BARD` | Возможный warm-start action corpus | Что строки Viewer чисты, уникальны и права на broadcast footage достаточны |
| `GabrieleGiudici/E-BARD-detection` | Возможный warm-start detector corpus | Что заявленная schema совпадает с Viewer и все bbox пригодны |
| `GabrieleGiudici/E-BARD-JerseyNumberRecognition` | Возможный warm-start JNR corpus | Что покрытие номеров и камер достаточно для наших видео |
| `GabrieleGiudici/EBQwen2.5-VL-3B` | Только challenger/teacher для `silver` | Что model-card метрики переносятся на наши видео или заменяют gold benchmark |

Перед скачиванием для каждого ID фиксируются точная revision, license, files,
SHA-256, required remote code и разрешённое использование. Плавающий `main` не
участвует ни в одном сравнимом эксперименте.

### Контракт предсказания первого ranker

Одна единица inference — кандидатное окно длиной ориентировочно 8–12 секунд с
16–24 хронологическими кадрами и query/domain context. Точные значения выбираются
на dev и фиксируются в preprocessing revision.

Входы:

- frozen visual embeddings по кадрам;
- query embedding, язык и нормализованный event intent;
- базовые scene/speech/OCR/object similarity и temporal features;
- опционально detector/tracker/scoreboard features с явной маской отсутствия;
- никаких filename, каталога матча, gold caption, event label или служебного ID.

Выходы:

- `relevance_score` для listwise/pairwise reranking;
- nullable probabilities классов `made_3`, `made_2`, `free_throw`, `miss`,
  `non_shot` только для sports profile;
- per-frame `release_score`, `outcome_score` и предложенные `start/end`;
- calibrated confidence, `abstain_reason` и contributing feature groups;
- model/preprocessing revisions для evidence и воспроизводимости.

Первая loss-композиция может содержать multi-positive contrastive/listwise ranking,
class-balanced classification и temporal boundary loss. Наивные in-batch negatives
запрещены: несколько разных попаданий являются корректными ответами на один запрос;
hard negatives берутся преимущественно из того же матча и похожего контекста.

### Контракт данных

Локальная структура:

```text
data/ml/
  sources/<source_sha>/source-manifest.json
  annotations/<annotation_set_id>/records.jsonl
  datasets/<dataset_id>/<version>/{manifest.json,examples.jsonl,splits.json}
  runs/<run_id>/{run.json,metrics.json,logs/,artifacts/}
  registry/<component>/<version>/release.json
  evaluations/<evaluation_id>/report.json
```

В Git хранятся schemas, training configs, tiny synthetic fixtures и
санитизированные отчёты; пользовательское видео, приватные labels, crops и веса по
умолчанию в Git не попадают.

Обязательные сущности:

- `SourceManifest`: `source_sha256`, duration, origin, owner/consent, license,
  rights scope, privacy class, allowed uses, ingest timestamp;
- `Example`: source SHA, exact interval, prepared-input SHA, `event_id`,
  `replay_group_id`, query/event type, nullable labels, label tier
  (`gold|silver|pseudo`), evidence, annotator policy и annotation revision;
- `SplitManifest`: group assignment, split rule/version, seed, duplicate-audit
  report и признак frozen holdout;
- `TrainingRun`: code SHA, dataset version, base-model ID/revision, dependency lock,
  preprocessing/config hashes, seed, hardware, command и output hashes;
- `ModelRelease`: component/version, artifact SHA, compatibility contract,
  evaluation IDs, required providers/resources, status, decision и rollback target.

Dataset проходит состояния `draft → validating → sealed → superseded`. После
`sealed` строки, splits и хеши не меняются; исправление создаёт новую версию.

Минимальный стартовый объём, уже согласованный стратегией:

- train: ≥100 проверенных кандидатов минимум из трёх новых целых видео;
- dev: ≥40 кандидатов из другого целого видео;
- final test: ≥40 кандидатов из ещё одного нетронутого видео;
- в каждом критическом stratum: ≥20 завершённых gold-кейсов и больше одного
  исходного источника.

Это только порог инженерного pilot, а не доказательство production-качества. Для
promotion отчёт обязан вычислить доверительные интервалы; если выборка не может
подтвердить gate, кандидат остаётся `evaluated`, даже если point estimate красивый.

Обязательные strata для basketball v1: `made_3`, `made_2`, `free_throw`, `miss`,
`replay`, `advertisement/non-game`, `scoreboard_hidden`, `low_resolution`,
`different_camera_or_league`, `OOD/non-basketball`. Равномерные окна полного видео
добавляются вместе с top-K ошибками, иначе dataset наследует selection bias старого
retriever.

Запрещённые утечки:

- один game/source/replay/event в разных split;
- frame в train и его crop/audio/OCR/embedding в holdout;
- один и тот же BARD-матч в ranker-train и detector/JNR-train, если эти компоненты
  участвуют в общей end-to-end оценке этого матча;
- настройка prompt, threshold, epoch или архитектуры по final holdout;
- horizontal flip для labels, зависящих от табло/номера/геометрии, и time reversal
  для причинной последовательности броска.

После просмотра ошибок final holdout становится dev; следующему promotion нужен
новый нетронутый holdout.

### Контракт оценки и promotion

Оценивать нужно одновременно три уровня:

| Уровень | Обязательные метрики |
| --- | --- |
| Proposal | full-timeline candidate Recall@50, покрытие gold events, duplicates/window density |
| Component | ranker nDCG/Recall/MRR; temporal IoU и boundary error; detector mAP/recall; jersey accuracy/coverage/abstention; calibration/Brier/ECE |
| Product | Precision@5, Recall@10/20, полезный точный момент в top-K, hard-negative FPR, evidence correctness, p50/p95, peak memory, OOM/system failures |

Для basketball v1 сохраняются текущие продуктовые guardrails: оба известных
трёхочковых входят в top-10 как regression check; free throw и реклама не становятся
подтверждённой трёшкой; `Precision@5 ≥ 0.60`, `Recall@20 ≥ 0.80`; средняя ошибка
границ ≤1 с; jersey публикуется только при устойчивом multi-frame evidence. Эти
seen-кейсы не являются promotion evidence.

До обучения ranker proposal stage должен показывать заранее замороженный порог,
начально `Recall@50 ≥ 0.95`; иначе сначала исправляется генератор кандидатов. Любая
новая модель должна пройти ablation против frozen-feature control. Рекомендуемый
минимально значимый end-to-end эффект: не менее `+0.03 nDCG@10` или
`+3 п.п. Recall@10`, причём 95%-й bootstrap interval по целым source groups не
пересекает ноль. Пороги замораживаются до прогона и не меняются после просмотра
результата.

Release lifecycle:

```text
run:      queued → running → failed | evaluated
model:    registered → rejected | eligible → shadow → canary → active
                                              ↘ rolled_back → retired
```

Одинаковые candidate и baseline оцениваются на одном dataset snapshot, одним
runner, на одном классе железа и с полной аттестацией runtime. Инфраструктурный сбой
не считается model miss, но блокирует решение. Автоматизация выдаёт `eligible` или
`reject`; до отдельного решения о L4 смена `active` требует подтверждения владельца.

### Контракт serving, наблюдения и обратной связи

- Active release выбирается атомарным указателем; старый bundle и feature schema
  остаются доступными для rollback.
- `shadow` получает копию тех же кандидатов, но не меняет ответ пользователю.
- `canary` меняет только явно выбранный локальный профиль/долю запросов и всегда
  имеет kill switch.
- Логи содержат model/dataset/preprocessing revision, latency, memory, confidence,
  abstention и фактический fallback, но не скрытое пользовательское содержимое.
- Click, preview duration, ручная перестановка и экспорт — weak signals с position
  bias; они поступают в annotation inbox, но не становятся gold автоматически.
- Ручные исправления `relevant/not relevant`, класса и границ сохраняются вместе с
  исходным prediction/evidence; спорные критические cases проходят adjudication.
- Мониторинг сравнивает coverage, confidence/calibration, abstention, slice quality,
  latency/OOM и drift входных признаков. Нарушение hard guardrail автоматически
  выключает candidate/canary и возвращает last-known-good.

### План реализации по фазам

Критический путь идёт по exit gates, а не по календарю. Phase 0 и read-only аудит
данных из Phase 1 можно вести параллельно; после data contract Phase 2 ranker и
Phase 4 perception допускают параллельную реализацию. Phase 3 (X-CLIP) и Phase 7
(Qwen QLoRA) являются условными и пропускаются, если error analysis не доказывает
их необходимость.

#### Фаза 0 — заморозить честную точку отсчёта

- **Текущее доказанное состояние:** Фаза 0 завершена 6 сентября 2026 года на
  чистом serving SHA `7054f137f92c18676461242a18c8fc2619898b68`. Неизменённый
  collector принял пять benchmark-профилей, десять прямых verifier cases,
  аттестацию шести заново установленных offline-окружений, полный smoke и
  принудительный rollback в один согласованный bundle из четырёх артефактов.
  Чистый checkout прошёл 2 527 backend-тестов. Точные receipts, переносимые raw
  measurements и команды воспроизведения сохранены в
  [принятом baseline](benchmarks/phase0/accepted/7054f13/README.md).
  После предоставленной владельцем свежей macOS-сессии обычный full smoke на
  целевом M4 Pro завершил восемь шагов, пять профилей, обе Qwen-проверки и MP4
  export. Все 1 099 raw host samples содержат нулевые счётчики swap-in, swap-out
  и Metal recovery; OOM не наблюдался. Cleanup завершён, disposable root пуст,
  SHA до и после одинаков. Peak RSS дерева процессов — 8 253 521 920 байт,
  отдельный system-wide Metal in-use peak — 11 172 855 808 байт; эти величины
  не складываются. Окно измерения — 305,154 с. Предыдущие
  [отрицательные контроли](benchmarks/phase0/negative-controls/7054f13/README.md)
  сохранены без изменений; нулевое окно не устанавливает причину старых событий
  и не гарантирует нулевой swap при любой будущей нагрузке. Порог не ослаблен,
  фоновые значения не вычитались.
  Baseline остаётся `regression_seen`: прямой verifier дал 3 совпадения,
  7 модельных промахов и 0 инфраструктурных ошибок; проваленные quality guardrails
  и недостаточное покрытие critical slices сохранены в отчёте. Это проверенная
  точка отсчёта, а не promotion модели. Обучение не начато. Отдельное разрешение
  владельца на первый срез Фазы 1 от 2026-09-11 зафиксировано ниже.
- **Вход:** текущий benchmark core, `regression_seen`, production providers и
  исторические отчёты.
- **Сделать:** включить в Git прямой end-to-end runner verifier/ranker; завершить
  measurement attestation; закрепить все реально используемые runtime/profile
  combinations (`dense_siglip`, temporal refinement, Lighthouse, Qwen; InternVideo
  только optional); устранить clean-install проблему OCR и расхождения Python
  environment; зафиксировать candidate Recall@50, P@5, Recall@10/20, nDCG,
  boundaries, latency, Metal memory и failures; выполнить full-ML smoke на M4 Pro.
- **Артефакты:** baseline snapshot, frozen metric policy, environment manifest,
  portable raw measurements, sanitized report и error ledger.
- **Exit gate:** чистый checkout воспроизводит отчёт; каждый профиль действительно
  выполняется либо честно имеет `not_configured`; `regression_seen` отделён от
  holdout; infrastructure failures не превращаются в model misses; rollback
  generation проверен принудительным сбоем.
- **Если gate не пройден:** не обучать модель — сначала чинить измерительную
  систему.

#### Фаза 1 — построить data/annotation contract

- **Вход:** schema выше, собственные разрешённые новые видео и кандидаты
  BARD/E-BARD; первый пул для аудита — матчи UBA.
- **Сделать:** реализовать source/rights ledger, annotation format,
  duplicate/replay grouping, split builder, dataset validation/sealing; провести
  аудит HF datasets; добавить минимальный UI/CLI для correction и adjudication;
  собрать train/dev/test из целых независимых источников.
- **Артефакты:** `dataset-v1` sealed, data card, rights report, duplicate audit,
  frozen split manifest и coverage report по strata.
- **Exit gate:** минимум 100/40/40, ≥20 gold на critical stratum, больше одного
  source на stratum, ноль межsplit-дублей; promotion holdout физически недоступен
  train job.
- **Если gate не пройден:** разрешены только `research_only` runs; ничего не
  регистрировать как production candidate.

Приоритет источников уточнён владельцем 2026-09-11: начинаем с восьми файлов UBA
в локальной папке `Данные матчи/`, NBA отложена и не входит в ближайший аудит.
Аудит метаданных и SHA подтвердил восемь стабильных файлов H.264/AAC 1080p60;
точных дублей нет. Их независимость по game/replay groups и права на обучение
пока не доказаны. Один вероятный повтор изученного матча исключён из новых
проверочных данных. До декодирования выбраны два источника `development_review`,
пять остальных оставлены `reserve_uninspected`; это не frozen holdout.
Аудит BARD/E-BARD остаётся отдельным срезом, а не условием начала аудита UBA.
Пороги и обязательные strata не меняются: нехватку других камер или
`OOD/non-basketball` нужно отразить как пробел покрытия и закрыть отдельно
разрешёнными источниками. Само наличие восьми файлов не доказывает data exit gate.
Дополнительные матчи UBA подбираются под выявленные пробелы покрытия и разбиения;
заранее заданного нового количества для скачивания нет.

#### Фаза 2 — доказать полный цикл на самом простом learner

- **Вход:** sealed `dataset-v1`, frozen embeddings и baseline report.
- **Сделать:** реализовать linear/GBDT control и temporal ranker на 2–10M
  trainable parameters; один feature/preprocessing package использовать в train и
  serve; фиксировать run manifest и immutable weights; провести slice analysis и
  calibration.
- **Артефакты:** reproducible training CLI, два model bundles, model cards,
  ablation и end-to-end comparison.
- **Exit gate:** повторный прогон из чистого checkout совпадает в tolerance;
  temporal ranker проходит quality/resource gates и лучше control значимым
  эффектом.
- **Если gate не пройден:** исправить данные, признаки или objective; не переходить
  к X-CLIP/QLoRA для маскировки плохого контракта.

#### Фаза 3 — испытать внешнюю video foundation как challenger

- **Вход:** рабочий Phase-2 ranker и error taxonomy, показывающая temporal capacity
  как ограничение.
- **Сделать:** локально зафиксировать X-CLIP revision; сначала обучать только
  projection/temporal head, затем при достаточных данных отдельно проверить
  unfreeze последних blocks; сохранить русский query path через существующий
  multilingual retrieval; сравнить с теми же splits и resource budget.
- **Артефакты:** X-CLIP challenger bundle, ablations frozen/head/unfrozen, Metal
  export и comparison report.
- **Exit gate:** независимый end-to-end выигрыш, отсутствие generic-language
  регрессии, runtime fit и положительный вклад против Phase 2.
- **Если gate не пройден:** оставить X-CLIP исследовательским challenger, active
  ranker не менять.

#### Фаза 4 — добавить проверяемое баскетбольное восприятие

- **Вход:** audited detection data и ошибки вида ball/rim/geometry.
- **Сделать:** fine-tune RF-DETR Small; оценить каждый класс, особенно recall ball
  и rim; подключить ByteTrack/OC-SORT; построить court/hoop geometry,
  ball-through-rim observation и scoreboard OCR voting; признаки передавать ranker
  через версионный optional contract.
- **Артефакты:** detector bundle, tracker/geometry fixtures, component benchmarks
  и end-to-end ablation каждого сигнала.
- **Exit gate:** detector/tracker проходит свои slice gates и даёт независимый
  продуктовый выигрыш; при отсутствии evidence система abstain, а не угадывает.
- **Если gate не пройден:** слабый компонент не участвует в final fusion; сохранить
  baseline ranker.

#### Фаза 5 — jersey и scoreboard как отдельные capabilities

- **Вход:** устойчивые player tracks/crops и audited JNR data.
- **Сделать:** обучить небольшой multi-frame jersey recognizer с классом
  `unreadable`; измерить accuracy вместе с coverage; для scoreboard сделать ROI
  OCR с временным голосованием и правилами смены счёта.
- **Артефакты:** JNR/scoreboard bundles, abstention calibration и UI evidence
  contract.
- **Exit gate:** нулевая подтверждённая hallucination на promotion hard negatives,
  приемлемая coverage и согласованность нескольких кадров.
- **Если gate не пройден:** номер остаётся suggestion/unknown и не попадает в
  подтверждённый пользовательский факт.

#### Фаза 6 — собрать Basketball Event Engine v1

- **Вход:** eligible temporal ranker и только прошедшие свои gates observations из
  Phases 4–5.
- **Сделать:** реализовать явную state machine
  `attempt → release → approach_rim → outcome → reaction/possession`; отдельно
  оценивать попадание, тип броска, границы и номер; обучить/калибровать маленький
  fusion head; вернуть `confirmed`, `candidate` либо `insufficient_evidence` с
  provenance каждого факта.
- **Артефакты:** versioned Event Engine, calibration report, structured evidence
  contract и ablation каждого входного сигнала.
- **Exit gate:** independent promotion holdout проходит все product/slice gates;
  free throw, miss, replay и реклама не получают ложный `confirmed_three`; generic
  holdout не регрессирует.
- **Если gate не пройден:** выключить слабый signal/fusion generation и сохранить
  Phase-2 ranker плюс текущий rule-based profile.

#### Фаза 7 — Qwen QLoRA только по доказанной необходимости

- **Вход:** error analysis, где candidate recall, geometry, fusion и labels
  достаточны, а ошибка локализована именно в fact verifier.
- **Сделать:** 20–50-step canary с frozen vision path, batch size 1 и gradient
  checkpointing; учить только строгие nullable observable facts и abstention;
  затем повторить held-out generation тем же production protocol и MLX artifact.
- **Артефакты:** adapter, base-revision link, prompt/protocol manifest,
  corruption/OOM tests и verifier comparison.
- **Exit gate:** улучшение fact precision/recall/calibration без hallucination,
  generic regression и resource breach.
- **Если gate не пройден:** adapter отклонить; frozen Qwen либо отсутствие verifier
  остаются допустимым production состоянием.

#### Фаза 8 — автоматизировать lifecycle до L3

- **Вход:** хотя бы один реально полезный, воспроизводимый learned component.
- **Сделать:** переиспользовать durable jobs и immutable artifact generations для
  `dataset_seal`, `train`, `evaluate`, `register`, `shadow`, `canary`, `promote`,
  `rollback`; добавить quotas, cancellation/resume, health, retention и audit
  trail; сделать отчёт и decision machine-readable.
- **Артефакты:** registry, job APIs/CLI, promotion policy, last-known-good pointer,
  monitoring report и disaster-recovery test.
- **Exit gate:** Definition of Done L3 выполняется end-to-end, включая forced
  failure и rollback; ни один job не портит source/user state.
- **Если gate не пройден:** lifecycle остаётся L1/L2 и не называется автономным.

#### Фаза 9 — доказать исходную универсальность

- **Вход:** стабильный basketball domain pack и общий интерфейс features/evidence.
- **Сделать:** выбрать один отличный от спорта профиль с реальными пользовательскими
  запросами, например лекции/интервью или бытовые действия; собрать отдельный
  holdout; обучить только профильный head/config через тот же pipeline.
- **Артефакты:** второй domain pack, cross-domain regression и отчёт о повторном
  использовании контрактов.
- **Exit gate:** новый профиль не требует fork базового indexer/API и не ухудшает
  остальные профили при отключении. Это финальное доказательство, что VideoScope
  не стал basketball-only системой.
- **Если gate не пройден:** оставить domain pack экспериментальным и не менять
  universal baseline.

### Команды, которые должен получить итоговый контур

Названия ниже задают interface contract; реализация может быть единым
`videoscope-ml` CLI или подкомандами основного CLI:

```bash
videoscope-ml source audit <source-or-dataset>
videoscope-ml dataset validate <dataset-id>@<version>
videoscope-ml dataset seal <dataset-id>@<version>
videoscope-ml train --config <config> --dataset <dataset-id>@<version>
videoscope-ml evaluate --run <run-id> --profile <frozen-profile>
videoscope-ml register --run <run-id>
videoscope-ml shadow --release <component>@<version>
videoscope-ml canary --release <component>@<version>
videoscope-ml promote --release <component>@<version> --decision <decision-id>
videoscope-ml rollback --component <component> --to last-known-good
videoscope-ml mine --release <component>@<version> --strategy uncertainty,hard-negative,coverage
videoscope-ml status
```

Каждая команда должна быть resumable/idempotent, писать durable job и возвращать
machine-readable manifest. Notebook допустим для exploration, но не является
источником promotion evidence.

### Рабочая инструкция для каждого следующего ML-шага

Перед любой реализацией действовать строго так:

1. Прочитать этот документ и связанные архитектурные контракты; одним предложением
   записать, какую часть пользовательского пути улучшает задача.
2. Проверить Git state, активные artifacts, baseline, dataset revision и целевое
   железо; не принимать прошлый notebook или устный результат за baseline.
3. Сформулировать одну фальсифицируемую гипотезу и один компонент, который меняется
   в итерации. Зафиксировать primary metric, guardrails и minimum effect заранее.
4. Определить prediction contract и slices до кода. Проверить, существует ли
   достаточный high-recall candidate pool; ranker не исправляет пропавший кандидат.
5. Проверить provenance/consent, duplicate groups и невозможность доступа train job
   к holdout. Любая утечка аннулирует эксперимент.
6. Сначала добавить schema/contract/golden fixtures и failing tests, затем
   реализацию. Train и serve обязаны импортировать один preprocessing package.
7. Запустить самый дешёвый осмысленный control. Новую большую модель добавлять
   только после измеренной причины, которую control не закрывает.
8. Сохранить run manifest, логи, hashes, weights, environment и sanitized report;
   затем повторить ключевой прогон из чистого checkout.
9. Оценить proposal, component и product уровни на frozen profiles; отдельно
   показать every critical slice, confidence intervals, latency, memory и failures.
10. Выполнить ablation и error analysis. Не объяснять разницу одним aggregate score.
11. Если любой hard gate нарушен — статус `rejected`, запись причины и следующий
    data/model hypothesis. Не подгонять threshold по final holdout.
12. Если candidate eligible — shadow, real-hardware canary и только затем явное
    promotion. Перед включением принудительно проверить rollback.
13. После релиза собирать weak feedback лишь в annotation inbox; gold создаётся
    человеком. Следующий snapshot всегда новая immutable version.
14. Завершать итерацию обновлением decision log: что узнали, что отвергли, какой
    риск остался и какой один эксперимент имеет наибольшую ожидаемую ценность.

### Жёсткие stop rules

- **STOP-VISION:** sports pack нельзя полностью отключить без изменения готовности
  и результатов универсального baseline.
- **STOP-LICENSE/PRIVACY:** нет rights ledger или opt-in — нет cloud training,
  публикации данных/crops/checkpoint и заявления о коммерческой готовности.
- **STOP-SPLIT:** обнаружен общий source/game/replay/event или perceptual/audio
  duplicate между train и holdout — аннулировать holdout и все его метрики.
- **STOP-HOLDOUT:** после просмотра ошибок или настройки по final test он становится
  dev; для следующего решения нужен новый test.
- **STOP-GOLD:** teacher labels не входят в promotion set; критические споры
  разрешаются независимой человеческой adjudication.
- **STOP-CANDIDATES:** proposal Recall@50 ниже frozen gate — сначала улучшать
  генерацию кандидатов, а не reranker/verifier.
- **STOP-COMPLEXITY:** нет frozen-feature temporal control или доказанного capacity
  bottleneck — не начинать X-CLIP unfreeze, QLoRA и более крупную модель.
- **STOP-ABLATION:** независимый вклад ниже minimum effect или interval пересекает
  ноль — компонент исключить из release plan.
- **STOP-SLICES:** aggregate gain не компенсирует падение более 5 п.п. на critical
  slice; universal Recall/nDCG не может падать более чем на 2 п.п.
- **STOP-EVIDENCE:** скрытый learned score заменяет evidence/abstain — release
  отклонить независимо от accuracy.
- **STOP-MEMORY/LATENCY:** Metal OOM, sustained swap, превышение resource budget
  либо p95 gate — artifact не идёт в production.
- **STOP-PARITY:** train/serve preprocessing расходятся на golden fixture или
  CUDA/Metal score выходит за frozen tolerance — artifact заблокировать.
- **STOP-REPRO:** experiment нельзя повторить одной CLI-командой с immutable
  manifests — он не влияет на решение.
- **STOP-AUTOPROMOTE:** retraining может автоматически создать candidate и отчёт,
  но не менять `active` без явного решения владельца до отдельного допуска L4.
- **STOP-ROLLBACK:** нет проверенного last-known-good и мгновенного switch без
  reindex — нельзя включать даже canary.

## NON-GOALS

- Обучать собственную foundation VLM с нуля.
- Называть скачанный checkpoint, prompt engineering или неподтверждённую LoRA
  «собственной моделью» или «полной автономией».
- Превращать VideoScope в basketball-only классификатор.
- Заменять общий multimodal retrieval одной end-to-end моделью без fallback.
- Оптимизировать только detector mAP, classifier F1 или verifier accuracy без
  end-to-end `query → interval` эффекта.
- Автоматически считать клики/экспорты gold labels.
- Публично выгружать пользовательские данные или checkpoint с неясными правами.
- Строить публичный облачный сервис, macOS package или новый UI до доказательства
  ML-цикла; это отдельные продуктовые программы.
- Поддерживать одновременно много экспериментальных основ. На каждом этапе есть
  один baseline, один candidate и при необходимости один challenger.

## OPEN QUESTIONS

Эти решения не блокируют Фазу 0, но должны быть закрыты до соответствующего шага:

1. Какие новые целые видео владелец вправе использовать для train/dev/test, и можно
   ли хотя бы часть артефактов публиковать?
2. Разрешено ли использовать HF Jobs только на открытых datasets, или обучение
   всегда должно оставаться локальным/на отдельно контролируемой машине?
3. Какой окончательный ресурсный бюджет отдать interactive ranker и background
   verifier внутри 24 ГБ?
4. Какие пользовательские действия предложить как явную разметку, не смешивая их с
   неявными clicks/position bias?
5. После скольких независимых стабильных L3-релизов вообще обсуждать L4? До
   отдельного решения ответ — «не обсуждать автопродвижение».

## HANDOFF

### Текущий ограниченный срез Фазы 1 — первая подборка для разметки

Фаза 0 завершена; её принятый baseline и неизменённые отрицательные контроли
сохраняются как исходная точка. Владелец 2026-09-11 отдельно разрешил аудит восьми
локальных файлов UBA и подготовку первой частной подборки для человеческой
проверки. В разрешение входят локальные производные превью и минимальный инструмент
разметки. Импорт в production, внешняя передача, обучение и promotion не разрешены.
NBA отложена.

Этот срез должен:

1. составить локальный inventory: хеши, длительность, разрешение, FPS, кодеки,
   наличие аудио и результаты проверки читаемости;
2. записать происхождение и разрешённое использование каждого источника, отделяя
   подтверждённые сведения от неизвестных; наличие контакта в UBA само по себе
   не подтверждает права;
3. проверить дубли и source/game/replay groups, определить, какие источники уже
   изучены и остаются `regression_seen`;
4. подготовить annotation/coverage report и список пробелов для queries, событий,
   границ и сложных отрицательных примеров; предложить разделение по целым матчам,
   сохранив содержимое будущего final holdout закрытым для настройки модели;
5. подготовить отдельную подборку коротких превью только из заранее назначенных
   `development_review` источников; совместить предложения frozen local SigLIP
   с окнами по временной линии и сохранить происхождение каждого предложения;
6. дать владельцу локальный просмотр, исправление границ и явных меток,
   сохранение истории ревизий и экспорт разметки. Никакая модельная оценка,
   просмотр или клик не создаёт gold автоматически.

Результат среза — inventory, source/rights ledger, coverage/blocker report и
подборка, готовая к человеческой разметке. Условия остановки: все превью читаются,
связаны с исходным SHA/интервалом, сохранение и восстановление ответов проверены,
исходники стабильны, reserved sources не декодированы, до ответа владельца меток
нет. Команды и артефакты описаны в [контракте pilot](benchmarks/phase1/README.md).
Это ещё не sealed dataset, не полный exit gate Фазы 1 и не доказательство качества.

Первая подборка готова 2026-09-11: 24 превью 1080p60 из двух источников,
16 слабых предложений SigLIP и 8 равномерных контрольных окон, всего 432 секунды.
Повтор из committed implementation воспроизвёл кадры, оценки и интервалы;
полное декодирование превью и неизменность SHA всех восьми исходников проверены.
Backend: 2 647 тестов; frontend: 60 тестов, build и браузерная проверка сохранения
с перезапуском на синтетической подборке пройдены. В реальной подборке на момент
передачи 0 человеческих ответов. Свидетельства и оставшиеся data gates сохранены
в [pilot receipt](benchmarks/phase1/pilot-receipt.json). Следующее действие —
проверка эпизодов владельцем; до неё состав и покрытие событий не утверждаются.

По просьбе владельца форма разметки обновлена до версии 2: физический исход
броска отделяется от решения о засчитанных очках и обстоятельств остановки игры.
Новый бросок после свистка и продолжение броска с фолом различаются явно; одна
метка относится к одному выделенному событию. Старые ревизии сохраняются без
перезаписи, новые поля требуют человеческого ответа. Это уточнение annotation
contract Фазы 1, без изменения моделей, baseline, данных для обучения или gates.

Следующее уточнение по запросу владельца — версия 3 формы: несколько независимых
событий внутри одного превью, каждое со своими границами, ответами и историей.
Добавление второго броска не заменяет первый. Старые v1/v2-ревизии сохраняются;
их primary-событие и новые event IDs остаются привязаны к одному исходному SHA
и source/replay group. Это расширение разметки, без повторного индексирования
исходников, обучения или объявления data exit gate пройденным.

После аудита отдельные срезы Фазы 1 реализуют data schemas, validator и
deterministic split/seal на tiny synthetic fixtures, затем разрешённый dataset
snapshot. Обучение остаётся закрыто до полного data exit gate и frozen
candidate-recall prerequisites. Первый learner после их прохождения —
frozen SigLIP2 feature control согласно Фазе 2.

Формулировка результата проекта после достижения L3:

> VideoScope — универсальная local-first система поиска моментов, которая умеет
> безопасно дообучать собственные небольшие ranker/domain-компоненты на разрешённых
> данных, независимо доказывать их качество на невиденных видео, объяснимо включать
> их в каскад и без потери пользовательского состояния откатываться к предыдущей
> версии.
