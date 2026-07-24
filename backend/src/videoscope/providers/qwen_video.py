from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import re
import tempfile
from threading import Lock
from typing import Protocol

from videoscope.media.ffmpeg import SampledFrame
from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.repository import Repository
from videoscope.search.fusion import EvidenceHit, FusedResult


logger = logging.getLogger(__name__)

MADE_BASKET_PROMPT_VERSION = "made-basket-facts-v3"
LEGACY_MADE_THREE_PROMPT_VERSION = "made-three-facts-v2"
GENERIC_PROMPT_VERSION = "generic-storyboard-v2"


class FrameExtractor(Protocol):
    def export_clip(
        self,
        source: Path,
        destination: Path,
        start: float,
        end: float,
    ) -> None: ...

    def extract_frames(
        self,
        source: Path,
        destination: Path,
        start: float,
        end: float,
        *,
        step: float,
        max_width: int = 640,
    ) -> list[SampledFrame]: ...


@dataclass(frozen=True, slots=True)
class QwenVideoJudgement:
    matches_query: bool | None = None
    confidence: float = 0.0
    event_start: float | None = None
    event_end: float | None = None
    shot_attempt: bool | None = None
    ball_through_hoop: bool | None = None
    shooter_outside_arc: bool | None = None
    three_point_signal: bool | None = None
    shooter_jersey: str | None = None
    evidence: str = ""
    made: bool | None = None
    three_point: bool | None = None


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "да", "confirmed"}:
            return True
        if normalized in {"false", "no", "нет", "rejected"}:
            return False
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_jersey(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized if re.fullmatch(r"\d{1,3}", normalized) else None


def _judgement_from_mapping(payload: dict[str, object]) -> QwenVideoJudgement:
    confidence = _optional_float(payload.get("confidence"))
    confidence = max(0.0, min(1.0, confidence if confidence is not None else 0.0))
    event_start = _optional_float(payload.get("event_start"))
    event_end = _optional_float(payload.get("event_end"))
    if event_start is not None:
        event_start = max(0.0, event_start)
    if event_end is not None:
        event_end = max(0.0, event_end)
    if (
        event_start is not None
        and event_end is not None
        and event_end <= event_start
    ):
        event_start = None
        event_end = None

    return QwenVideoJudgement(
        matches_query=_optional_bool(payload.get("matches_query")),
        confidence=confidence,
        event_start=event_start,
        event_end=event_end,
        shot_attempt=_optional_bool(payload.get("shot_attempt")),
        ball_through_hoop=_optional_bool(payload.get("ball_through_hoop")),
        shooter_outside_arc=_optional_bool(payload.get("shooter_outside_arc")),
        three_point_signal=_optional_bool(payload.get("three_point_signal")),
        shooter_jersey=_normalize_jersey(payload.get("shooter_jersey")),
        evidence=str(payload.get("evidence") or "").strip(),
        made=_optional_bool(payload.get("made")),
        three_point=_optional_bool(payload.get("three_point")),
    )


def parse_qwen_judgement(text: str) -> QwenVideoJudgement:
    """Извлекает проверяемый ответ даже при Markdown-обрамлении модели."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Qwen did not return a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Qwen response must be a JSON object")
    return _judgement_from_mapping(payload)


_THREE_POINT_QUERY = re.compile(
    r"(?:тр[её]хочк|тр[её]шк|three[\s-]?point|3[\s-]?point)",
    flags=re.IGNORECASE,
)
_TWO_POINT_QUERY = re.compile(
    r"(?:двухочк|два\s+очка|two[\s-]?point|2[\s-]?(?:point|очк))",
    flags=re.IGNORECASE,
)
_FREE_THROW_QUERY = re.compile(
    r"(?:штрафн(?:ый|ого|ом|ые|ых)?(?:\s+бросок)?|free[\s-]?throw)",
    flags=re.IGNORECASE,
)
_MADE_SHOT_QUERY = re.compile(
    r"(?:забива|забил|заброс|попада|попал|реализ|"
    r"makes?|made|scores?|hits?|sinks?|converts?)",
    flags=re.IGNORECASE,
)

_SPORTS_EVENT_TYPES = frozenset(
    {
        "made_three_point",
        "made_two_point",
        "made_free_throw",
    }
)


def _required_sports_event_type(query: str) -> str | None:
    if not _MADE_SHOT_QUERY.search(query):
        return None
    if _FREE_THROW_QUERY.search(query):
        return "made_free_throw"
    if _THREE_POINT_QUERY.search(query):
        return "made_three_point"
    if _TWO_POINT_QUERY.search(query):
        return "made_two_point"
    return None


def _candidate_sports_event_types(candidate: FusedResult) -> set[str]:
    return {
        event_type
        for evidence in candidate.evidence
        if isinstance(event_type := evidence.metadata.get("event_type"), str)
        and event_type in _SPORTS_EVENT_TYPES
    }


def enforce_query_requirements(
    query: str,
    judgement: QwenVideoJudgement,
) -> QwenVideoJudgement:
    """Сохраняет совместимость с проверкой общих раскадровок."""
    needs_three_point = bool(_THREE_POINT_QUERY.search(query))
    needs_made_shot = bool(_MADE_SHOT_QUERY.search(query))
    if needs_three_point and needs_made_shot:
        verified = judgement.made is True and judgement.three_point is True
        return replace(judgement, matches_query=verified)
    if needs_three_point and judgement.three_point is False:
        return replace(judgement, matches_query=False)
    if needs_made_shot and judgement.made is False:
        return replace(judgement, matches_query=False)
    return judgement


class QwenVideoReranker:
    """Локально проверяет временные события на коротких клипах через Qwen3.5."""

    id = "qwen-video"

    def __init__(
        self,
        *,
        model_name: str | None,
        repository: Repository,
        extractor: FrameExtractor,
        temp_dir: Path,
        cache_dir: Path | None = None,
        top_candidates: int = 12,
        context_seconds: float = 4.0,
        min_clip_seconds: float = 7.0,
        max_clip_seconds: float = 12.0,
        frame_count: int = 12,
        video_fps: float = 2.0,
        max_tokens: int = 320,
    ) -> None:
        self.model_name = model_name.strip() if model_name else None
        self.repository = repository
        self.extractor = extractor
        self.temp_dir = Path(temp_dir)
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else self.temp_dir.parent / "cache" / "qwen-video"
        )
        self.top_candidates = max(1, top_candidates)
        self.context_seconds = max(0.0, context_seconds)
        self.max_clip_seconds = max(2.0, max_clip_seconds)
        self.min_clip_seconds = min(
            self.max_clip_seconds,
            max(2.0, min_clip_seconds),
        )
        self.frame_count = max(4, frame_count)
        self.video_fps = max(0.5, video_fps)
        self.max_tokens = max(64, max_tokens)
        self._model = None
        self._processor = None
        self._lock = Lock()

    @property
    def display_model_name(self) -> str:
        return self.model_name or "mlx-community/Qwen3.5-9B-MLX-4bit"

    def status(self) -> ProviderStatus:
        if not self.model_name:
            return ProviderStatus(
                self.id,
                "Qwen Video",
                ProviderState.NEEDS_CONFIGURATION,
                "Необязательная проверка действий и номеров игроков; "
                "задайте QWEN_VIDEO_MODEL",
                optional=True,
            )
        if importlib.util.find_spec("mlx_vlm") is None:
            return ProviderStatus(
                self.id,
                "Qwen Video",
                ProviderState.UNAVAILABLE,
                "Установите зависимость mlx-vlm",
                optional=True,
            )
        model_path = Path(self.model_name).expanduser()
        if not model_path.exists():
            try:
                from huggingface_hub import snapshot_download

                snapshot = Path(
                    snapshot_download(self.model_name, local_files_only=True)
                )
                config = snapshot / "config.json"
                weights = next(snapshot.glob("*.safetensors"), None)
            except Exception:
                config = None
                weights = None
            if not config or not config.is_file() or weights is None:
                return ProviderStatus(
                    self.id,
                    "Qwen Video",
                    ProviderState.NEEDS_CONFIGURATION,
                    f"Модель ещё не загружена: {self.model_name}",
                    optional=True,
                )
        return ProviderStatus(
            self.id,
            "Qwen Video",
            ProviderState.READY,
            f"Проверка видеособытий и возможных номеров на Metal: {self.model_name}",
            optional=True,
        )

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        if not self.model_name:
            raise RuntimeError("QWEN_VIDEO_MODEL is not configured")
        from mlx_vlm import load

        self._model, self._processor = load(self.model_name)
        return self._model, self._processor

    @staticmethod
    def _fact_prompt() -> str:
        return (
            "Report only independently visible basketball facts. Return exactly one "
            "compact JSON object with keys shot_attempt, ball_through_hoop, "
            "shooter_outside_arc, three_point_signal, shooter_jersey, evidence. "
            "Each fact must be true, false, or null independently; do not decide the "
            "event class. Use null when the clip does not prove a fact. A shooting pose "
            "does not prove a make. shooter_jersey is digits only or null. evidence is "
            "one short factual sentence. No text outside JSON."
        )

    @staticmethod
    def _generic_prompt(query: str) -> str:
        return (
            "You are a strict video judge. Read the chronological storyboard from left "
            "to right and top to bottom. "
            f"User query: {query!r}. "
            "Return exactly one compact JSON object with keys matches_query, confidence, "
            "event_start, event_end, shot_attempt, made, three_point, shooter_jersey, "
            "evidence. Use null whenever the frames do not prove a fact. Never infer a "
            "made basket from a shooting posture. A jersey number must be visible; "
            "otherwise use null."
        )

    @staticmethod
    def _build_storyboard(
        frames: list[SampledFrame],
        destination: Path,
        clip_start: float,
    ) -> Path:
        from PIL import Image, ImageDraw, ImageOps

        if not frames:
            raise ValueError("storyboard requires at least one frame")
        columns = 4
        cell_width = 320
        cell_height = 204
        label_height = 24
        rows = (len(frames) + columns - 1) // columns
        canvas = Image.new(
            "RGB",
            (columns * cell_width, rows * (cell_height + label_height)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for index, frame in enumerate(frames):
            with Image.open(frame.path) as source:
                image = ImageOps.fit(
                    source.convert("RGB"),
                    (cell_width, cell_height),
                    method=Image.Resampling.LANCZOS,
                )
            x = (index % columns) * cell_width
            y = (index // columns) * (cell_height + label_height)
            canvas.paste(image, (x, y))
            draw.rectangle(
                (x, y + cell_height, x + cell_width, y + cell_height + label_height),
                fill="white",
            )
            draw.text(
                (x + 6, y + cell_height + 4),
                f"{index + 1:02d}   +{max(0.0, frame.timestamp - clip_start):.1f}s",
                fill="black",
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, quality=90)
        return destination

    def _judge_video(self, clip: Path) -> QwenVideoJudgement:
        from mlx_vlm import apply_chat_template, generate

        model, processor = self._load()
        prompt = apply_chat_template(
            processor,
            model.config,
            [self._fact_prompt()],
            video=str(clip),
            fps=self.video_fps,
            enable_thinking=False,
        )
        with self._lock:
            result = generate(
                model,
                processor,
                prompt,
                video=[str(clip)],
                fps=self.video_fps,
                max_tokens=self.max_tokens,
                temperature=0.0,
                enable_thinking=False,
                verbose=False,
            )
        return parse_qwen_judgement(result.text)

    def _judge_storyboard(
        self,
        storyboard: Path,
        query: str,
    ) -> QwenVideoJudgement:
        from mlx_vlm import apply_chat_template, generate

        model, processor = self._load()
        prompt = apply_chat_template(
            processor,
            model.config,
            [self._generic_prompt(query)],
            num_images=1,
            enable_thinking=False,
        )
        with self._lock:
            result = generate(
                model,
                processor,
                prompt,
                image=[str(storyboard)],
                max_tokens=self.max_tokens,
                temperature=0.0,
                enable_thinking=False,
                verbose=False,
            )
        return parse_qwen_judgement(result.text)

    def _judge(self, storyboard: Path, query: str) -> QwenVideoJudgement:
        """Поддерживает отдельный стенд старых проверок раскадровок."""
        return self._judge_storyboard(storyboard, query)

    def _clip_interval(self, candidate: FusedResult) -> tuple[float, float] | None:
        video = self.repository.get_video(candidate.video_id)
        if video is None:
            return None
        duration = max(0.0, float(video.duration or candidate.end))
        if duration <= 0:
            return None
        start = max(0.0, candidate.start - self.context_seconds)
        end = min(duration, candidate.end + self.context_seconds)
        if end <= start:
            return None

        target = min(
            duration,
            max(self.min_clip_seconds, min(self.max_clip_seconds, end - start)),
        )
        midpoint = max(0.0, min(duration, (candidate.start + candidate.end) / 2))
        start = max(0.0, min(duration - target, midpoint - target / 2))
        end = min(duration, start + target)
        return start, end

    @staticmethod
    def _candidate_id(candidate: FusedResult) -> str:
        return f"{candidate.video_id}:{candidate.start:.3f}:{candidate.end:.3f}"

    def _prompt_version(self, query: str, native_video: bool) -> str:
        fps_suffix = f"-fps{self.video_fps:g}"
        if native_video:
            return MADE_BASKET_PROMPT_VERSION + fps_suffix
        query_digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        return f"{GENERIC_PROMPT_VERSION}-{query_digest}"

    def _legacy_fact_prompt_version(
        self,
        prompt_version: str,
    ) -> str | None:
        fps_suffix = f"-fps{self.video_fps:g}"
        if prompt_version != MADE_BASKET_PROMPT_VERSION + fps_suffix:
            return None
        return LEGACY_MADE_THREE_PROMPT_VERSION + fps_suffix

    def _cache_key(
        self,
        *,
        video_id: str,
        interval: tuple[float, float],
        prompt_version: str,
    ) -> dict[str, object]:
        return {
            "model": self.display_model_name,
            "video_id": video_id,
            "interval": [round(interval[0], 3), round(interval[1], 3)],
            "prompt_version": prompt_version,
        }

    def _cache_path(self, key: dict[str, object]) -> Path:
        serialized = json.dumps(
            key,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _load_cached(
        self,
        key: dict[str, object],
    ) -> QwenVideoJudgement | None:
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("key") != key:
                return None
            judgement = payload.get("judgement")
            if not isinstance(judgement, dict):
                return None
            return _judgement_from_mapping(judgement)
        except (OSError, ValueError, TypeError):
            logger.warning("Повреждён кэш проверки Qwen: %s", path)
            return None

    def _save_cached(
        self,
        key: dict[str, object],
        judgement: QwenVideoJudgement,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(key)
        temporary = path.with_suffix(".tmp")
        payload = {
            "key": key,
            "judgement": asdict(judgement),
        }
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def resolve_match(
        query: str,
        candidate: FusedResult,
        judgement: QwenVideoJudgement,
    ) -> QwenVideoJudgement:
        """Соединяет тип броска от SigLIP и факт попадания от Qwen."""
        required_event_type = _required_sports_event_type(query)
        if required_event_type is not None:
            if required_event_type not in _candidate_sports_event_types(candidate):
                return replace(judgement, matches_query=False)
            if judgement.ball_through_hoop is True:
                return replace(judgement, matches_query=True)
            if judgement.ball_through_hoop is False:
                return replace(judgement, matches_query=False)
            return replace(judgement, matches_query=None)
        return judgement

    @staticmethod
    def _verification_score(judgement: QwenVideoJudgement) -> float:
        if judgement.confidence > 0:
            return judgement.confidence
        if judgement.matches_query is True:
            return 0.92
        if judgement.shot_attempt is False:
            return 0.88
        return 0.50

    @classmethod
    def _score(
        cls,
        candidate: FusedResult,
        judgement: QwenVideoJudgement,
    ) -> float:
        if judgement.matches_query is True:
            verified = (
                candidate.score * 0.35
                + cls._verification_score(judgement) * 0.65
            )
            return min(1.0, max(candidate.score, verified))
        if judgement.shot_attempt is False:
            return candidate.score * 0.75
        if judgement.matches_query is False:
            return candidate.score * 0.96
        return candidate.score

    @staticmethod
    def _selection_key(
        query: str,
        candidate: FusedResult,
    ) -> tuple[bool, float]:
        required_event_type = _required_sports_event_type(query)
        event_matches = (
            required_event_type is None
            or required_event_type in _candidate_sports_event_types(candidate)
        )
        return event_matches, candidate.score

    @staticmethod
    def _result_key(candidate: FusedResult) -> tuple[int, float]:
        qwen_evidence = [
            evidence
            for evidence in candidate.evidence
            if evidence.modality == "qwen_video"
        ]
        if any(
            evidence.metadata.get("matches_query") is True
            for evidence in qwen_evidence
        ):
            verification_rank = 2
        elif any(
            evidence.metadata.get("shot_attempt") is False
            for evidence in qwen_evidence
        ):
            verification_rank = 0
        else:
            verification_rank = 1
        return verification_rank, candidate.score

    def rerank(self, query: str, candidates: list[FusedResult]) -> list[FusedResult]:
        if not self.model_name or not candidates:
            return candidates
        selected_ids = {
            self._candidate_id(candidate)
            for candidate in sorted(
                candidates,
                key=lambda candidate: self._selection_key(query, candidate),
                reverse=True,
            )[: self.top_candidates]
        }
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        output: list[FusedResult] = []
        with tempfile.TemporaryDirectory(
            prefix="videoscope-qwen-", dir=self.temp_dir
        ) as directory:
            workspace = Path(directory)
            for index, candidate in enumerate(candidates):
                if self._candidate_id(candidate) not in selected_ids:
                    output.append(candidate)
                    continue
                video = self.repository.get_video(candidate.video_id)
                interval = self._clip_interval(candidate)
                if video is None or interval is None:
                    output.append(candidate)
                    continue
                clip_start, clip_end = interval
                required_event_type = _required_sports_event_type(query)
                native_video = (
                    required_event_type is not None
                    and bool(_candidate_sports_event_types(candidate))
                )
                prompt_version = self._prompt_version(query, native_video)
                cache_key = self._cache_key(
                    video_id=candidate.video_id,
                    interval=interval,
                    prompt_version=prompt_version,
                )
                judgement = self._load_cached(cache_key)
                if judgement is None and native_video:
                    legacy_prompt_version = self._legacy_fact_prompt_version(
                        prompt_version
                    )
                    if legacy_prompt_version is not None:
                        legacy_key = self._cache_key(
                            video_id=candidate.video_id,
                            interval=interval,
                            prompt_version=legacy_prompt_version,
                        )
                        judgement = self._load_cached(legacy_key)
                        if judgement is not None:
                            try:
                                self._save_cached(cache_key, judgement)
                            except OSError:
                                logger.warning(
                                    "Не удалось обновить legacy-кэш Qwen для %s",
                                    self._candidate_id(candidate),
                                )
                cached = judgement is not None
                try:
                    if judgement is None and native_video:
                        clip = workspace / f"candidate-{index:04d}.mp4"
                        self.extractor.export_clip(
                            Path(video.media_path),
                            clip,
                            clip_start,
                            clip_end,
                        )
                        judgement = self._judge_video(clip)
                    elif judgement is None:
                        step = max(
                            0.4,
                            (clip_end - clip_start) / self.frame_count,
                        )
                        frames = self.extractor.extract_frames(
                            Path(video.media_path),
                            workspace / f"frames-{index:04d}",
                            clip_start,
                            clip_end,
                            step=step,
                            max_width=640,
                        )[: self.frame_count]
                        storyboard = self._build_storyboard(
                            frames,
                            workspace / f"candidate-{index:04d}.jpg",
                            clip_start,
                        )
                        judgement = self._judge_storyboard(storyboard, query)
                    if judgement is None:
                        raise RuntimeError("Qwen returned no judgement")
                    if not cached:
                        self._save_cached(cache_key, judgement)
                    judgement = self.resolve_match(query, candidate, judgement)
                except Exception:
                    logger.exception(
                        "Qwen не смог проверить кандидата %s",
                        self._candidate_id(candidate),
                    )
                    output.append(candidate)
                    continue

                start = candidate.start
                end = candidate.end
                if (
                    not native_video
                    and judgement.matches_query is True
                    and judgement.event_start is not None
                    and judgement.event_end is not None
                ):
                    start = max(
                        clip_start,
                        min(clip_end, clip_start + judgement.event_start),
                    )
                    end = max(
                        start,
                        min(clip_end, clip_start + judgement.event_end),
                    )

                evidence_text = (
                    "Qwen: попадание подтверждено"
                    if judgement.matches_query is True and native_video
                    else "Qwen: событие подтверждено"
                    if judgement.matches_query is True
                    else "Qwen: явного броска нет"
                    if judgement.shot_attempt is False
                    else "Qwen: результат не доказан"
                )
                evidence = EvidenceHit(
                    video_id=candidate.video_id,
                    segment_id=f"qwen-video:{self._candidate_id(candidate)}",
                    start=start,
                    end=end,
                    modality="qwen_video",
                    score=self._verification_score(judgement),
                    text=evidence_text,
                    metadata={
                        "source": "qwen-video-verifier",
                        "model": self.display_model_name,
                        "prompt_version": prompt_version,
                        "cache_hit": cached,
                        "matches_query": judgement.matches_query,
                        "shot_attempt": judgement.shot_attempt,
                        "ball_through_hoop": judgement.ball_through_hoop,
                        "shooter_outside_arc": judgement.shooter_outside_arc,
                        "three_point_signal": judgement.three_point_signal,
                        "possible_shooter_jersey": judgement.shooter_jersey,
                        "shooter_jersey_confirmed": False,
                        "model_evidence": judgement.evidence,
                        "clip_start": clip_start,
                        "clip_end": clip_end,
                    },
                )
                output.append(
                    replace(
                        candidate,
                        start=start,
                        end=end,
                        score=self._score(candidate, judgement),
                        modalities=sorted(
                            {*candidate.modalities, "qwen_video"}
                        ),
                        evidence=[evidence, *candidate.evidence],
                    )
                )
        return sorted(
            output,
            key=self._result_key,
            reverse=True,
        )
