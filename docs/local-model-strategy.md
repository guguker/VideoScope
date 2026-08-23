# Локальная модельная стратегия VideoScope

Статус: решение по результатам локального exploratory-прогона 2026-08-23.

## Решение

Основной VLM-verifier остаётся `mlx-community/Qwen3.5-9B-MLX-4bit` с точной
ревизией `938d8919941c6e7efd3c7150eff7fe9d12afa631`. Модель вызывается только для
короткого списка кандидатов (сейчас максимум 12) и подтверждает независимо
наблюдаемые факты. Она не получает права единолично определять тип броска, номер
игрока или границы события.

`Qwen3.8-27B` не входит в текущий 24-ГБ runtime при проверенном input protocol:
сторонняя 4-bit сборка завершилась Metal OOM и на native-video, и на одном
storyboard нужного качества. `Qwen3-VL-2B-Instruct-4bit` остаётся быстрым
диагностическим контролем, но не production-verifier. 2-bit вариант 27B не
загружается и не проверяется: card самой найденной сборки помечает его как severely
degraded.

Обучение foundation model с нуля исключено. QLoRA 9B допускается только после
появления независимых train/dev/test-видео и прохождения короткого canary; первым
обучаемым компонентом должен быть небольшой temporal scorer либо специализированный
детектор/трекер, а не VLM.

## Iteration Compact

- **Objective:** повысить recall и точность локального поиска видеособытий без
  облачного inference и без превышения 24 ГБ unified memory.
- **Hypothesis:** каскад `high-recall candidates → small temporal/domain signals →
  Qwen 9B fact verifier` устойчивее одной более крупной VLM.
- **Baseline:** исторический basketball regression, повторно прогнанный на двух
  заново экспортированных известных событиях; это `regression_seen`, а не holdout.
- **Current change:** зафиксировать content-addressed verifier dataset и одинаковый
  inference protocol; затем улучшать только один компонент за итерацию.
- **Primary metrics:** recall/FPR/abstention для `ball_through_hoop`, temporal IoU и
  boundary error, contract success, jersey hallucination rate, p50/p95 latency,
  peak Metal memory и OOM.
- **Promotion gate:** в полностью новом `promotion_holdout` не менее 20 завершённых
  кейсов в каждом критическом stratum, нулевая инфраструктурная ошибка и отсутствие
  регрессии hard negatives. Старые три видео не могут доказать обобщение.
- **Rollback:** production worker остаётся на зафиксированном `mlx-vlm==0.6.7` до
  отдельного совместимого benchmark; действующие индексы и модельная конфигурация
  не переписываются exploratory-прогонами.

## Локальный exploratory baseline

Оборудование: Apple M4 Pro, 16-core GPU, 24 ГБ unified memory,
macOS 26.6.2 (build 25G83). Перед экспериментом на Data volume было около 37 ГиБ
свободно. Оба исходных клипа получены штатным `FFmpeg.export_clip` из приватного
`sports_game_a` с SHA-256
`fa63155b818f0c227d42189d8e4366a02f87a1843a47d78f416326fecdb84206`.

Протокол для успешных прогонов:

- Python 3.12.13;
- `mlx==0.32.1`, `mlx-vlm==0.6.15`, `transformers==5.15.1`, `jinja2==3.1.6`;
- native MP4, 2 fps, `max_tokens=128`, `temperature=0`, thinking disabled;
- prompt `made-basket-facts-v3`;
- один model process и последовательные запросы;
- два gold-интервала: синий №15 и белый №9.

Это диагностический baseline, а не promotion-grade run: временный runner ещё не был
частью чистого Git SHA, а два события уже использовались при разработке.

Подготовленные входы зафиксированы отдельно от исходного видео:

| Input alias | Source interval | Prepared SHA-256 | Bytes |
| --- | ---: | --- | ---: |
| `sports-made-three-blue-15` | 1857.438–1869.438 | `884d52b34becf01adf940c2b979a67651e436f6deda03bf001f3b0973c9aa31b` | 6,882,258 |
| `sports-made-three-white-9` | 6860.000–6870.625 | `aa203e4639b33bceb61c31fc96da56cb96a20df1d133ea777e2a7cfea6aeeb49` | 5,345,772 |

| Модель | Синий №15 | Белый №9 | Inference time после load | Peak Metal | Вывод |
| --- | --- | --- | ---: | ---: | --- |
| Qwen3.5 9B 4-bit | попадание: да; номер: `4` | попадание: да; номер: `17` | 61.0 / 36.3 с | 8.01 / 8.63 ГиБ | на этих seen positives: `ball_through_hoop` 2/2; jersey 0/2 |
| Qwen3-VL 2B 4-bit | попадание: нет; номер: `15` | попадание: да; номер: `17`, событие названо штрафным | 17.1 / 14.7 с | 3.88 / 3.88 ГиБ | быстрый, но temporal/event качество недостаточно |
| Qwen3.8 27B 4-bit, сторонняя uncensored | Metal OOM | не запускалось | failure | выше доступного бюджета | исключена из текущего 24-ГБ protocol/runtime |

Для 27B был дополнительно проверен один 1280×684 storyboard из 12 тех же кадров.
Он также завершился `kIOGPUCommandBufferCallbackErrorOutOfMemory`. После попыток
используемый swap вырос примерно с 0.54 до 2.6 ГБ. Принудительное повышение wired
memory limit не является допустимым production-решением.

2B запускалась с ревизией `9c4f5209e57b31f4b9dfba735de3fb983739c9cc`.
Локальная 27B использовала ревизию
`401c79c2232c841da0cf3755e9a0ce3228d9cfc9`; identity зафиксирована без публикации
весов: config
`f238d3a11e8d96b4f688fd09c627879b9b87a067835c83cb7c1c102df7aa4fb0`, weight
index `13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8`, shards
`014e69ab23abb0ec453346930260b2f78785c36e23b1922c6992ec6ec9732b37`,
`3f275e43480498e0d1d113a726c746066dfab73927a22873c45a238be55cee13` и
`ef97c143824615510df09938a188bbdf475f9e41a9475113b3fb07090c842137`. Полные
локальные пути в переносимый benchmark не входят.

Сохранённый результат production path не подтвердил первое попадание, а exploratory-
прогон подтвердил. Старый артефакт не аттестует полный protocol, включая точную
model/runtime identity и generation parameters, поэтому причиной нельзя считать
только переход `mlx-vlm==0.6.7 → 0.6.15`. Runtime и параметры генерации должны
входить в identity каждого следующего прогона.

## Почему не 27B и не 2-bit

Официальная `Qwen3.8-27B` — нативная image/video модель с 27B параметров и контекстом
262K, но MLX 4-bit conversion занимает 16.1 ГБ только на диске. Найденная локальная
сборка — не aligned checkpoint, а сторонняя abliterated/refusal-removed версия без
значимых guardrails и с пометкой research-only. Её собственная документация
оценивает 4-bit примерно в 15 ГБ и предупреждает, что 2-bit приводит к повторам,
бессвязному тексту и лишь частично работающему vision. Поэтому она исключена из
production также по safety/supply-chain причинам, независимо от OOM.

На машине ровно с 24 ГБ веса, vision tower, кадры, KV cache, активации, Metal runtime
и macOS конкурируют за одну память. Локальный OOM важнее vendor minimum-RAM оценки:
эта конфигурация не имеет эксплуатационного запаса даже для одного storyboard.

## Выбранная архитектурная последовательность

1. Зафиксировать строгий verifier dataset: source SHA, prepared-input SHA, полный
   sampling protocol, nullable fact labels и split только по целым видео.
2. Повторить 9B baseline на 10–12 вручную проверенных `regression_seen` кейсах:
   попадания, промахи, штрафные, реклама/OOD и действия в интервью.
3. Вернуть оба известных трёхочковых в первые 10 результатов через небольшой
   temporal proposal scorer. VLM не может подтвердить событие, которое retrieval не
   предложил; top-10 здесь является promotion-критерием, а не размером rerank-пула.
4. Для номера игрока использовать crop по track, multi-frame OCR/classifier и
   голосование. Номер VLM остаётся только мягким признаком с возможностью abstain.
5. Для типа броска добавить геометрию площадки и/или OCR изменения счёта; факт
   `ball_through_hoop` остаётся отдельным сигналом Qwen 9B.
6. Только после независимых данных провести 20–50-step QLoRA canary 9B с frozen
   vision path, batch size 1 и gradient checkpointing. Сначала повторить held-out
   generation; при corruption/OOM немедленно откатить adapter.

Минимальный контракт данных перед fine-tune:

- train: не менее 100 проверенных кандидатов минимум из трёх новых исходных видео;
- dev: не менее 40 кандидатов из отдельного исходного видео;
- final test: не менее 40 кандидатов из ещё одного полностью нетронутого видео и не
  менее 20 в каждом критическом stratum;
- никакие перекрывающиеся окна одного source SHA не распределяются между splits;
- текущие MBA и интервью остаются только regression-набором и не попадают в train.

Эти числа являются стартовым инженерным порогом, а не обещанием статистической
значимости. Решение о расширении данных принимается по доверительным интервалам и
структуре ошибок после первого размеченного набора.

Санитизированные численные наблюдения сохранены в
[`exploratory-ab-2026-08-23.json`](benchmarks/video-verifier/exploratory-ab-2026-08-23.json),
а строгие десять regression-кейсов — в
[`seed-v1.json`](benchmarks/video-verifier/seed-v1.json). Оба артефакта прямо
запрещают использовать текущий набор для promotion.

## Проверенные источники

- Qwen Team: [официальный Qwen3.8 repository](https://github.com/QwenLM/Qwen3.8) и
  [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B).
- MLX Community: [Qwen3.8-27B 4-bit, 16.1 GB](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit).
- OrcaRouter: [локально найденная uncensored MLX-сборка и предупреждение о 2-bit](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX).
- MLX-VLM: [LoRA/QLoRA documentation](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/LORA.MD),
  [release history](https://github.com/Blaizzy/mlx-vlm/releases) и
  [открытая ошибка multi-image SFT для Qwen](https://github.com/Blaizzy/mlx-vlm/issues/1726).
- Google: [Gemma 4 12B model card](https://huggingface.co/google/gemma-4-12B-it),
  остаётся возможным независимым 12B challenger, но не загружается до появления
  достаточного regression-набора.
