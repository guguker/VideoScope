from __future__ import annotations

from dataclasses import dataclass
import re

from videoscope.search.text_matching import tokens


ALL_MODALITIES = frozenset({"speech", "ocr", "objects", "visual", "lighthouse"})
MODE_MODALITIES = {
    "speech": frozenset({"speech"}),
    "visual": frozenset({"visual", "objects", "lighthouse"}),
    "ocr": frozenset({"ocr"}),
}

_SPEECH_CUES = {
    "говорит", "говорят", "говорил", "говорила", "сказал", "сказала", "произносит",
    "обсуждает", "рассказывает", "речь", "слово", "звучит", "mentions", "says", "said",
    "speaks", "talks",
}
_OCR_CUES = {
    "написано", "надпись", "субтитры", "табло", "титр", "экране", "текст", "логотип",
    "scoreboard", "caption", "written", "text", "screen",
}
_OBJECT_CUES = {
    "человек", "мужчина", "женщина", "игрок", "тренер", "мяч", "чашка", "бутылка",
    "телефон", "ноутбук", "книга", "машина", "стол", "стул", "кольцо", "person", "man",
    "woman", "player", "coach", "ball", "cup", "bottle", "phone", "laptop", "book", "car",
}
_ACTION_CUES = {
    "поднимает", "опускает", "бросает", "забивает", "бежит", "прыгает", "падает", "держит",
    "передает", "ловит", "машет", "садится", "встает", "выходит", "входит", "поворачивается",
    "raises", "lowers", "throws", "scores", "runs", "jumps", "falls", "holds", "passes",
    "catches", "waves", "sits", "stands", "enters", "leaves",
}


@dataclass(frozen=True, slots=True)
class QueryPlan:
    query: str
    intent: str
    modalities: frozenset[str]
    modality_weights: dict[str, float]
    use_lighthouse: bool
    refine_temporally: bool
    explanation: str


class QueryRouter:
    def route(
        self,
        query: str,
        *,
        mode: str = "all",
        requested_lighthouse: bool = True,
    ) -> QueryPlan:
        normalized = query.strip()
        query_tokens = set(tokens(normalized))
        if mode in MODE_MODALITIES:
            modalities = MODE_MODALITIES[mode]
            use_lighthouse = mode == "visual" and requested_lighthouse
            if not use_lighthouse:
                modalities = modalities - {"lighthouse"}
            return QueryPlan(
                normalized,
                mode,
                modalities,
                self._weights(mode),
                use_lighthouse,
                mode == "visual",
                {
                    "speech": "Выбран поиск только по распознанной речи",
                    "visual": "Выбран визуальный поиск по кадрам, объектам и действиям",
                    "ocr": "Выбран поиск только по тексту в кадре",
                }[mode],
            )

        has_speech = bool(query_tokens & _SPEECH_CUES)
        has_ocr = bool(query_tokens & _OCR_CUES)
        has_action = bool(query_tokens & _ACTION_CUES)
        has_object = bool(query_tokens & _OBJECT_CUES)
        original_tokens = re.findall(r"[^\W\d_]+", normalized, flags=re.UNICODE)
        looks_like_entity = (
            1 <= len(original_tokens) <= 3
            and all(token[:1].isupper() for token in original_tokens)
            and not (has_speech or has_ocr or has_action or has_object)
        )

        if looks_like_entity:
            intent = "entity"
            modalities = frozenset({"speech", "ocr"})
            explanation = "Имя или название: приоритет точным словам в речи и тексте кадра"
        elif has_ocr and not (has_speech or has_action):
            intent = "ocr"
            modalities = frozenset({"ocr"})
            explanation = "Запрос о видимом тексте: используется OCR"
        elif has_speech and has_action:
            intent = "mixed"
            modalities = ALL_MODALITIES
            explanation = "Смешанный запрос: объединяются речь, кадр, объекты и действие"
        elif has_speech:
            intent = "speech"
            modalities = frozenset({"speech"})
            explanation = "Запрос о сказанном: приоритет распознанной речи"
        elif has_action:
            intent = "action"
            modalities = ALL_MODALITIES
            explanation = (
                "Запрос о действии: визуальное событие проверяется по кадрам, "
                "речи и тексту в кадре"
            )
        elif has_object:
            intent = "object"
            modalities = frozenset({"objects", "visual"})
            explanation = "Запрос об объекте: объединяются детектор и визуальная семантика"
        else:
            intent = "mixed"
            modalities = ALL_MODALITIES
            explanation = "Общий запрос: используются все доступные источники"

        use_lighthouse = requested_lighthouse and intent in {"action", "mixed"}
        if not use_lighthouse:
            modalities = modalities - {"lighthouse"}
        return QueryPlan(
            normalized,
            intent,
            modalities,
            self._weights(intent),
            use_lighthouse,
            intent in {"action", "mixed"},
            explanation,
        )

    @staticmethod
    def _weights(intent: str) -> dict[str, float]:
        weights = {
            "speech": 1.0,
            "ocr": 0.88,
            "objects": 0.92,
            "visual": 1.08,
            "lighthouse": 0.78,
        }
        if intent == "entity":
            weights.update(speech=1.65, ocr=1.05, visual=0.2, objects=0.2, lighthouse=0.1)
        elif intent == "speech":
            weights.update(speech=1.55, ocr=0.55, visual=0.25, objects=0.2, lighthouse=0.1)
        elif intent == "ocr":
            weights.update(ocr=1.65, speech=0.35, visual=0.35, objects=0.25, lighthouse=0.1)
        elif intent == "object":
            weights.update(objects=1.48, visual=1.18, speech=0.3, ocr=0.3, lighthouse=0.35)
        elif intent == "action":
            weights.update(visual=1.48, lighthouse=1.05, objects=0.78, speech=0.35, ocr=0.25)
        elif intent == "mixed":
            weights.update(visual=1.25, speech=1.1, objects=1.0, lighthouse=0.82, ocr=0.85)
        return weights
