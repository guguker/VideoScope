from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re

from videoscope.storage import atomic_write_json


_CYRILLIC_TO_LATIN = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "yo",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

_INFLECTION_SUFFIXES = (
    "иями",
    "ями",
    "ами",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ими",
    "иях",
    "ах",
    "ях",
    "ов",
    "ев",
    "ей",
    "ом",
    "ем",
    "ым",
    "им",
    "ую",
    "юю",
    "ая",
    "яя",
    "ое",
    "ее",
    "а",
    "я",
    "у",
    "ю",
    "е",
    "ы",
    "и",
)


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", value.casefold().replace("ё", "е"), flags=re.UNICODE))


def tokens(value: str) -> tuple[str, ...]:
    normalized = normalize_text(value)
    return tuple(normalized.split()) if normalized else ()


def transliterate(value: str) -> str:
    return "".join(_CYRILLIC_TO_LATIN.get(character, character) for character in value.casefold())


def _phonetic(value: str) -> str:
    latin = transliterate(value)
    replacements = (
        ("sch", "sh"),
        ("zh", "sh"),
        ("ch", "sh"),
        ("b", "p"),
        ("v", "f"),
        ("g", "k"),
        ("d", "t"),
        ("z", "s"),
    )
    for source, target in replacements:
        latin = latin.replace(source, target)
    return re.sub(r"(.)\1+", r"\1", latin)


def _stems(token: str) -> set[str]:
    output = {token}
    if len(token) < 5 or not re.search(r"[а-я]", token):
        return output

    # Русские фамилии часто сохраняют форму именительного падежа с добавлением одного окончания.
    if re.search(r"(?:ов|ев|ин)(?:а|у|ым|е|ы|и)$", token):
        output.add(re.sub(r"(?:а|у|ым|е|ы|и)$", "", token))

    for suffix in _INFLECTION_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            output.add(token[: -len(suffix)])
    return output


def _token_similarity(query_token: str, candidate: str) -> tuple[float, str]:
    if query_token == candidate:
        return 1.0, "exact"
    if _stems(query_token) & _stems(candidate):
        return 0.95, "stem"

    query_latin = transliterate(query_token)
    candidate_latin = transliterate(candidate)
    if query_latin == candidate_latin:
        return 0.92, "transliteration"

    if min(len(query_token), len(candidate)) < 4 or min(len(query_latin), len(candidate_latin)) < 4:
        return 0.0, "none"
    if _phonetic(query_latin) == _phonetic(candidate_latin):
        return 0.78, "phonetic"

    if query_latin[:4] == candidate_latin[:4]:
        return 0.74, "stem"

    ratio = SequenceMatcher(None, query_latin, candidate_latin).ratio()
    if abs(len(query_latin) - len(candidate_latin)) <= 2 and ratio >= 0.72:
        return ratio * 0.90, "fuzzy"
    return 0.0, "none"


@dataclass(frozen=True, slots=True)
class LexicalMatch:
    score: float
    matched_terms: tuple[str, ...]
    strategy: str


def lexical_match(query: str, text: str) -> LexicalMatch:
    query_tokens = tokens(query)
    text_tokens = tokens(text)
    if not query_tokens or not text_tokens:
        return LexicalMatch(0.0, (), "none")

    similarities: list[float] = []
    matched_terms: list[str] = []
    strategies: list[str] = []
    for query_token in query_tokens:
        score, strategy = max(
            (_token_similarity(query_token, candidate) for candidate in text_tokens),
            key=lambda item: item[0],
        )
        if score < 0.72:
            similarities.append(0.0)
            continue
        similarities.append(score)
        matched_terms.append(query_token)
        strategies.append(strategy)

    coverage = len(matched_terms) / len(query_tokens)
    if coverage == 0:
        return LexicalMatch(0.0, (), "none")
    mean_similarity = sum(similarities) / len(similarities)
    phrase_bonus = 0.06 if normalize_text(query) in normalize_text(text) else 0.0
    score = min(1.0, 0.72 * coverage + 0.28 * mean_similarity + phrase_bonus)
    strategy_order = ("exact", "stem", "transliteration", "phonetic", "fuzzy")
    strategy = next((item for item in strategy_order if item in strategies), strategies[0])
    return LexicalMatch(score, tuple(matched_terms), strategy)


class SearchLexicon:
    """Небольшой редактируемый пользователем словарь имён и терминов предметной области."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read(self) -> dict[str, list[str]]:
        if not self.path.is_file():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(term): [str(alias) for alias in aliases if str(alias).strip()]
            for term, aliases in payload.items()
            if str(term).strip() and isinstance(aliases, list)
        }

    def replace(self, entries: dict[str, list[str]]) -> None:
        normalized = {
            " ".join(term.split()): sorted({" ".join(alias.split()) for alias in aliases if alias.strip()})
            for term, aliases in entries.items()
            if term.strip()
        }
        atomic_write_json(self.path, normalized)

    def expand(self, query: str) -> list[str]:
        normalized_query = normalize_text(query)
        output = [query.strip()]
        for canonical, aliases in self.read().items():
            variants = [canonical, *aliases]
            normalized_variants = [normalize_text(value) for value in variants]
            if not any(value and value in normalized_query for value in normalized_variants):
                continue
            output.extend(variants)
        return list(dict.fromkeys(value for value in output if value))
