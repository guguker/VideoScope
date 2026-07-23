# VideoScope

Локальное приложение для мультимодального поиска моментов внутри видео и сборки MP4-нарезок. Базовый сценарий универсален; спортивное видео, включая баскетбол, используется как сильный прикладной профиль, а не как жёсткое ограничение.

## Что уже реализовано

- потоковая загрузка больших видео с проверкой размера и контейнера;
- фоновая очередь индексации и прогресс по стадиям;
- сцены PySceneDetect и кадры-превью через FFmpeg;
- локальная расшифровка Whisper Large v3 Turbo на Apple MLX;
- PaddleOCR через актуальный Transformers engine;
- мультиязычный поиск по кадрам через SigLIP 2 с быстрым 224 и quality-профилем 384;
- локальный Roboflow RF-DETR Small + Supervision без API-ключа;
- Roboflow Serverless для пользовательских Universe-моделей при наличии ключа;
- Lighthouse QD-DETR как подтверждающий visual moment retriever с окнами до 150 секунд;
- точное уточнение двух лучших сцен по плотной выборке кадров;
- InternVideo 2.5 как опциональный финальный GPU-reranker;
- локальный Qdrant с multilingual MPNet; при отказе энкодера остаётся точный лексический поиск;
- маршрутизация запросов, калиброванное объединение сигналов и объяснение каждого совпадения;
- словарь имён и терминов, русские словоформы, транслитерация и таймкоды отдельных слов;
- встроенный benchmark с Recall@K, MRR, temporal IoU и задержкой;
- переход к точному таймкоду, очередь фрагментов и экспорт MP4.

## Запуск

Требования: macOS на Apple Silicon, Python 3.12, Node.js, pnpm и FFmpeg.

```bash
make install
make install-ml
make models
make install-lighthouse
make index-objects
make index-visual
make index-speech
cp .env.example .env
make dev
```

Lighthouse не обязателен для запуска. Его можно добавить позднее командой `make install-lighthouse`.

Интерфейс: `http://127.0.0.1:5173`  
API и OpenAPI: `http://127.0.0.1:8765/api/docs`

Данные сохраняются в `./data` и исключены из Git.

## Провайдеры

| Провайдер | Роль | Поведение без настройки |
| --- | --- | --- |
| FFmpeg | probe, кадры, клипы, монтаж | критический |
| PySceneDetect | временные границы сцен | один полный сегмент |
| Whisper MLX | речь с таймкодами | поиск работает по другим сигналам |
| PaddleOCR | текст на экране | этап пропускается |
| SigLIP 2 Base | основной локальный visual encoder и точное уточнение; 384 доступен как quality-профиль | визуальный режим отключается |
| Roboflow + Supervision | локальный универсальный RF-DETR или облачная предметная модель | `rfdetr-small` работает без ключа |
| Qdrant + MPNet | локальный семантический индекс речи и объектов | остаётся точный поиск по словам |
| Lighthouse | video moment retrieval | нужен пакет и checkpoint |
| InternVideo 2.5 | тяжёлый reranking лучших кандидатов | нужен внешний CUDA endpoint |

Roboflow RF-DETR по умолчанию работает локально, поэтому кадры не покидают компьютер. Чтобы использовать собственную модель Roboflow Universe, задайте её `ROBOFLOW_MODEL_ID` и ключ `ROBOFLOW_API_KEY`. Статус каждого провайдера виден в интерфейсе. Режимы `Всё`, `Речь`, `Кадр` и `Текст` позволяют явно выбрать сигнал, а каждый результат показывает источник совпадения.

## Точность и оценка

Автоматический маршрутизатор различает запросы об именах, речи, тексте на экране, объектах и действиях. Для действий SigLIP сначала выбирает сцены, затем повторно оценивает кадры внутри двух лучших сцен; Lighthouse влияет на выдачу только при временном совпадении с другим визуальным сигналом. В окне «Качество» доступны контрольные запросы и сравнение режимов по Recall@1/3/5, MRR, temporal IoU и времени ответа. Для перехода на скачанный quality-профиль задайте модель 384 и выполните `make index-visual-quality`.

Словарь хранится в `data/search-glossary.json`, используется лексическим поиском и добавляется в initial prompt Whisper при следующей индексации. Контрольная выборка хранится в `data/evaluation/cases.json`.

## InternVideo 2.5

Официальная модель `OpenGVLab/InternVideo2_5_Chat_8B` рассчитана на CUDA и FlashAttention, поэтому локальный Mac-путь остаётся на SigLIP 2. InternVideo подключается как отдельный GPU endpoint: VideoScope отправляет ему до восьми кадров только из четырёх лучших кандидатов и получает оценку релевантности и краткое обоснование. Для включения задайте `INTERNVIDEO_ENDPOINT` и при необходимости `INTERNVIDEO_API_KEY`.

## Lighthouse

Официальная библиотека тестировалась авторами на Python 3.9/CUDA и ограничивает входное видео 150 секундами. VideoScope разрезает длинные файлы на окна, сохраняет признаки и возвращает результаты в исходной временной шкале. Для CPU используется `feature_name="clip"`.

Upstream импортирует аудио-, Gradio- и Transformers-зависимости даже для CLIP-only режима. Поэтому VideoScope содержит изолированный адаптер: модель QD-DETR и формат checkpoint остаются официальными, но загружаются только OpenAI CLIP и нужное ядро Lighthouse. `make install-lighthouse` скачивает проверенный checkpoint с Zenodo автоматически. Без него поиск продолжает работать по речи, OCR, объектам, сценам и multilingual-векторам.

Укажите пути в `.env`:

```dotenv
LIGHTHOUSE_CHECKPOINT=./data/models/lighthouse/clip_qd_detr_qvhighlight.ckpt
```

## Проверка

```bash
make test
make build
make demo
```

Архитектура и схема ранжирования описаны в [docs/architecture.md](docs/architecture.md).

## Основные источники

- [SigLIP 2 Base 224](https://huggingface.co/google/siglip2-base-patch16-224)
- [SigLIP 2 Base 384](https://huggingface.co/google/siglip2-base-patch16-384)
- [InternVideo 2.5](https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B)
- [Lighthouse](https://github.com/line/lighthouse)
- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- [Roboflow Inference](https://inference.roboflow.com/inference_helpers/inference_sdk/)
- [PaddleOCR](https://www.paddleocr.ai/main/en/quick_start.html)
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
- [Qdrant](https://qdrant.tech/documentation/quickstart/)
