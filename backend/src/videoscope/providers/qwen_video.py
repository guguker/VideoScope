from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import re
import tempfile
from threading import Lock
from typing import Protocol

from videoscope.media.ffmpeg import SampledFrame
from videoscope.model_manifest import model_identity
from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.repository import Repository
from videoscope.search.fusion import EvidenceHit, FusedResult
from videoscope.storage import atomic_write_json


logger = logging.getLogger(__name__)

MADE_BASKET_PROMPT_VERSION = "made-basket-facts-v4"
LEGACY_MADE_THREE_PROMPT_VERSION = "made-three-facts-v2"
GENERIC_PROMPT_VERSION = "generic-visual-v4"
QWEN_INPUT_SCHEMA_VERSION = "qwen-video-input-v3"
QWEN_RESPONSE_MAX_BYTES = 4096
QWEN_EVIDENCE_MAX_CHARS = 240
_BASKETBALL_FACTS_PROMPT = (
    "Report only independently visible basketball facts. Return exactly one "
    "compact JSON object with keys shot_attempt, ball_through_hoop, "
    "shooter_outside_arc, three_point_signal, shooter_jersey, evidence. "
    "Each fact must be true, false, or null independently; do not decide the "
    "event class. Use null when the clip does not prove a fact. A shooting pose "
    "does not prove a make. shooter_jersey is 0, 00, or 1 through 99 without "
    "a leading zero, or null. evidence is "
    "one short factual sentence of at most 240 characters. No text outside JSON."
)
_GENERIC_QUERY_PROMPT_PREFIX = (
    "You are a strict video judge. Inspect the supplied visual evidence in "
    "chronological order. User query: "
)
_GENERIC_QUERY_PROMPT_SUFFIX = (
    ". Return exactly one compact JSON object with keys matches_query, confidence, "
    "event_start, event_end, shot_attempt, made, three_point, shooter_jersey, "
    "evidence. Use null whenever the frames do not prove a fact. Never infer a "
    "made basket from a shooting posture. A jersey number must be visible; "
    "otherwise use null. confidence is a number from 0 to 1. event_start and "
    "event_end are nonnegative seconds or null. evidence is one short factual "
    "sentence of at most 240 characters. No text outside JSON."
)


def _flat_qwen_schema(properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


# Canonical immutable JSON supplies a fresh grammar schema for every request.
# The same schemas drive strict parsing and the content-bound prompt protocol.
_QWEN_RESPONSE_SCHEMAS_JSON = json.dumps(
    {
        "basketball_facts": _flat_qwen_schema({
            "shot_attempt": {"type": ["boolean", "null"]},
            "ball_through_hoop": {"type": ["boolean", "null"]},
            "shooter_outside_arc": {"type": ["boolean", "null"]},
            "three_point_signal": {"type": ["boolean", "null"]},
            "shooter_jersey": {
                "type": ["string", "null"], "pattern": "^(?:0|00|[1-9][0-9]?)$",
            },
            "evidence": {"type": "string", "maxLength": QWEN_EVIDENCE_MAX_CHARS},
        }),
        "generic_query": _flat_qwen_schema({
            "matches_query": {"type": ["boolean", "null"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "event_start": {"type": ["number", "null"], "minimum": 0},
            "event_end": {"type": ["number", "null"], "minimum": 0},
            "shot_attempt": {"type": ["boolean", "null"]},
            "made": {"type": ["boolean", "null"]},
            "three_point": {"type": ["boolean", "null"]},
            "shooter_jersey": {
                "type": ["string", "null"], "pattern": "^[0-9]{1,3}$",
            },
            "evidence": {"type": "string", "maxLength": QWEN_EVIDENCE_MAX_CHARS},
        }),
    },
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
)


def qwen_response_schema(prompt_kind: str) -> dict[str, object]:
    if prompt_kind not in {"basketball_facts", "generic_query"}:
        raise ValueError("Qwen response prompt contract is unsupported")
    return json.loads(_QWEN_RESPONSE_SCHEMAS_JSON)[prompt_kind]


_QWEN_PROMPT_PROTOCOL = {
    "basketball_prompt": _BASKETBALL_FACTS_PROMPT,
    "basketball_prompt_version": MADE_BASKET_PROMPT_VERSION,
    "generic_prompt_prefix": _GENERIC_QUERY_PROMPT_PREFIX,
    "generic_prompt_suffix": _GENERIC_QUERY_PROMPT_SUFFIX,
    "generic_prompt_version": GENERIC_PROMPT_VERSION,
    "input_schema": QWEN_INPUT_SCHEMA_VERSION,
    "input_content_binding": {
        "materialization": "private-read-only-copy-v1",
        "namespace_drift": "descriptor-fingerprint-v1",
        "request_fields": ["expected_sha256", "expected_byte_size"],
        "source_open": "descriptor-relative-o-nofollow-v1",
    },
    "query_interpolation": "python-repr",
    "generation": {
        "enable_thinking": False,
        "temperature": 0.0,
    },
    "structured_output": {
        "version": "qwen-typed-output-v1",
        "decoder": "mlx-vlm==0.6.7:llguidance==1.7.6:json-schema-v1",
        "parser": "strict-flat-json-v1",
        "completion": "finish_reason=stop",
        "max_response_utf8_bytes": QWEN_RESPONSE_MAX_BYTES,
        "interval": "event_end>event_start-when-both-present",
        "schemas": json.loads(_QWEN_RESPONSE_SCHEMAS_JSON),
    },
    "supported_prompt_inputs": {
        "basketball_facts": ["video"],
        "generic_query": ["storyboard", "video"],
    },
    "schema_version": 4,
}
QWEN_PROMPT_PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        _QWEN_PROMPT_PROTOCOL,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
QWEN_INFERENCE_RUNTIME_IDENTITY = (
    "videoscope-qwen-worker-v4|python==3.12.13|"
    "platform==aarch64-apple-darwin-macos14plus|"
    "runtime-manifest-sha256:"
    "0481dda4b0aef9927d5f9adb1a5f7292f997cac39a6d59ecef0087913d80335f|"
    "model-manifest-sha256:"
    "abaeb6d14ccbdc1741cccea700edd3385eb7cb3d215796b465c4d5a1a0504fe0"
)
QWEN_IN_PROCESS_RUNTIME_IDENTITY = "unattested-in-process:mlx-vlm==0.6.7"


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
class QwenInferenceStatus:
    ready: bool
    detail: str


class QwenInferenceClient(Protocol):
    @property
    def identity(self) -> dict[str, object]: ...

    def status(self) -> QwenInferenceStatus: ...

    def judge_video(
        self,
        source: Path,
        *,
        fps: float,
        max_tokens: int,
        expected_sha256: str | None = None,
        expected_byte_size: int | None = None,
    ) -> QwenVideoJudgement: ...

    def judge_video_query(
        self,
        source: Path,
        query: str,
        *,
        fps: float,
        max_tokens: int,
        expected_sha256: str | None = None,
        expected_byte_size: int | None = None,
    ) -> QwenVideoJudgement: ...

    def judge_storyboard(
        self,
        source: Path,
        query: str,
        *,
        max_tokens: int,
        expected_sha256: str | None = None,
        expected_byte_size: int | None = None,
    ) -> QwenVideoJudgement: ...


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
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


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


def _unique_qwen_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Qwen response has duplicate keys")
        result[key] = value
    return result


def _reject_qwen_json_constant(_value: str) -> object:
    raise ValueError("Qwen response has a non-finite number")


def _validate_qwen_scalar(value: object, spec: dict[str, object]) -> None:
    # This intentionally validates only the flat, primitive response contract.
    kind = {
        bool: "boolean", int: "number", float: "number",
        str: "string", type(None): "null",
    }.get(type(value))
    allowed = spec["type"]
    if kind is None or kind not in (allowed if isinstance(allowed, list) else [allowed]):
        raise ValueError("Qwen response has an invalid field type")
    if kind == "number":
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("Qwen response has a non-finite number")
        if "minimum" in spec and value < spec["minimum"]:
            raise ValueError("Qwen response number is out of bounds")
        if "maximum" in spec and value > spec["maximum"]:
            raise ValueError("Qwen response number is out of bounds")
    elif kind == "string":
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("Qwen response text is not valid UTF-8") from None
        if "maxLength" in spec and len(value) > spec["maxLength"]:
            raise ValueError("Qwen response text is too long")
        if "pattern" in spec and re.fullmatch(spec["pattern"], value) is None:
            raise ValueError("Qwen response text has an invalid format")


def parse_qwen_worker_judgement(text: str, *, prompt_kind: str) -> QwenVideoJudgement:
    """Reject malformed worker output before it can become an abstention."""
    schema = qwen_response_schema(prompt_kind)
    if type(text) is not str or len(text.encode("utf-8")) > QWEN_RESPONSE_MAX_BYTES:
        raise ValueError("Qwen response text is invalid or too large")
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_qwen_object,
            parse_constant=_reject_qwen_json_constant,
        )
    except (json.JSONDecodeError, RecursionError):
        raise ValueError("Qwen response is not strict JSON") from None
    if type(payload) is not dict or set(payload) != set(schema["required"]):
        raise ValueError("Qwen response fields do not match its prompt contract")
    for key, spec in schema["properties"].items():
        _validate_qwen_scalar(payload[key], spec)
    event_start = payload.get("event_start")
    event_end = payload.get("event_end")
    if event_start is not None and event_end is not None and event_end <= event_start:
        raise ValueError("Qwen response interval is invalid")
    return QwenVideoJudgement(**payload)


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
        model_revision: str | None = None,
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
        inference_client: QwenInferenceClient | None = None,
        allow_in_process: bool = False,
    ) -> None:
        self.model_name = model_name.strip() if model_name else None
        self.model_revision = model_revision
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
        self.inference_client = inference_client
        self.allow_in_process = allow_in_process
        self._model = None
        self._processor = None
        self._load_lock = Lock()
        self._lock = Lock()

    @property
    def display_model_name(self) -> str:
        return self.model_name or "mlx-community/Qwen3.5-9B-MLX-4bit"

    @property
    def model_identity(self) -> str:
        return model_identity(self.display_model_name, self.model_revision)

    @property
    def identity(self) -> dict[str, object]:
        if self.inference_client is not None:
            boundary: dict[str, object] = self.inference_client.identity
        elif self.allow_in_process:
            boundary = {
                "mode": "deprecated-in-process",
                "runtime_identity": QWEN_IN_PROCESS_RUNTIME_IDENTITY,
            }
        else:
            boundary = {"mode": "disabled"}
        return {
            "provider": self.id,
            "model": self.model_identity,
            "input_schema": QWEN_INPUT_SCHEMA_VERSION,
            "boundary": boundary,
            "video_fps": self.video_fps,
            "max_tokens": self.max_tokens,
        }

    @property
    def benchmark_attestation(self) -> dict[str, object] | None:
        """Describe the exact strict worker boundary without host-local paths."""
        if self.model_name is None or self.inference_client is None:
            return None
        try:
            boundary = self.inference_client.identity
        except Exception:
            return None
        if not isinstance(boundary, dict):
            return None
        contract = boundary.get("contract")
        worker_model = boundary.get("model")
        runtime_identity = boundary.get("runtime_identity")
        source_bundle_sha256 = boundary.get("source_bundle_sha256")
        prompt_protocol_sha256 = boundary.get("prompt_protocol_sha256")
        input_root_sha256 = boundary.get("input_root_sha256")
        if (
            boundary.get("mode") != "isolated-worker"
            or type(contract) is not str
            or not contract
            or worker_model != self.model_identity
            or type(runtime_identity) is not str
            or not runtime_identity
            or type(source_bundle_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", source_bundle_sha256) is None
            or type(prompt_protocol_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", prompt_protocol_sha256) is None
            or type(input_root_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", input_root_sha256) is None
        ):
            return None
        protocol = {
            "contract": contract,
            "frame_count": self.frame_count,
            "generic_prompt": GENERIC_PROMPT_VERSION,
            "input_schema": QWEN_INPUT_SCHEMA_VERSION,
            "made_basket_prompt": MADE_BASKET_PROMPT_VERSION,
            "max_tokens": self.max_tokens,
            "prompt_protocol_sha256": prompt_protocol_sha256,
            "video_fps": self.video_fps,
        }
        try:
            canonical = json.dumps(
                protocol,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError):
            return None
        return {
            "candidate_limit": self.top_candidates,
            "model_identity": self.model_identity,
            "protocol_identity": "sha256:"
            + hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "provider": self.id,
            "prompt_protocol_sha256": prompt_protocol_sha256,
            "runtime_identity": runtime_identity,
            "source_bundle_sha256": source_bundle_sha256,
            "input_root_sha256": input_root_sha256,
            "source_bound": True,
            "strict_complete": True,
        }

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
        if self.inference_client is not None:
            worker = self.inference_client.status()
            return ProviderStatus(
                self.id,
                "Qwen Video",
                ProviderState.READY if worker.ready else ProviderState.UNAVAILABLE,
                worker.detail,
                optional=True,
            )
        if not self.allow_in_process:
            return ProviderStatus(
                self.id,
                "Qwen Video",
                ProviderState.NEEDS_CONFIGURATION,
                "Запустите изолированный Qwen worker и задайте QWEN_VIDEO_ENDPOINT; "
                "внутрипроцессный режим доступен только как устаревающий opt-in",
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
                    snapshot_download(
                        self.model_name,
                        revision=self.model_revision,
                        local_files_only=True,
                    )
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
        if not self.allow_in_process:
            raise RuntimeError("in-process Qwen inference is disabled")
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return self._model, self._processor
            if not self.model_name:
                raise RuntimeError("QWEN_VIDEO_MODEL is not configured")
            from mlx_vlm import load

            model_reference = self.model_name
            if (
                self.model_revision is not None
                and not Path(self.model_name).expanduser().exists()
            ):
                from huggingface_hub import snapshot_download

                model_reference = snapshot_download(
                    self.model_name,
                    revision=self.model_revision,
                    local_files_only=True,
                )
            model, processor = load(model_reference)
            self._model, self._processor = model, processor
            return model, processor

    @staticmethod
    def _fact_prompt() -> str:
        return _BASKETBALL_FACTS_PROMPT

    @staticmethod
    def _generic_prompt(query: str) -> str:
        return _GENERIC_QUERY_PROMPT_PREFIX + repr(query) + _GENERIC_QUERY_PROMPT_SUFFIX

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
        if self.inference_client is not None:
            return self.inference_client.judge_video(
                clip,
                fps=self.video_fps,
                max_tokens=self.max_tokens,
            )
        if not self.allow_in_process:
            raise RuntimeError("Qwen worker is not configured")
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
        if self.inference_client is not None:
            return self.inference_client.judge_storyboard(
                storyboard,
                query,
                max_tokens=self.max_tokens,
            )
        if not self.allow_in_process:
            raise RuntimeError("Qwen worker is not configured")
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

    def refinement_bounds(
        self, query: str, candidate: FusedResult,
    ) -> tuple[float, float] | None:
        """Expose the reviewed storyboard context before generating a judgement."""
        native_video = (
            _required_sports_event_type(query) is not None
            and bool(_candidate_sports_event_types(candidate))
        )
        return None if native_video else self._clip_interval(candidate)

    @staticmethod
    def _candidate_id(candidate: FusedResult) -> str:
        return (
            f"{candidate.video_id}:{float(candidate.start).hex()}:"
            f"{float(candidate.end).hex()}"
        )

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
        runtime_identity = QWEN_IN_PROCESS_RUNTIME_IDENTITY
        execution_identity: dict[str, object] = {
            "mode": "deprecated-in-process",
            "runtime_identity": runtime_identity,
        }
        if self.inference_client is not None:
            execution_identity = self.benchmark_attestation or {}
            if not execution_identity:
                raise RuntimeError("Qwen worker execution identity is invalid")
            configured_identity = execution_identity.get("runtime_identity")
            if isinstance(configured_identity, str) and configured_identity:
                runtime_identity = configured_identity
        return {
            "model": self.model_identity,
            "video_id": video_id,
            "interval": [round(interval[0], 3), round(interval[1], 3)],
            "prompt_version": prompt_version,
            "inference_config": {
                "execution_identity": execution_identity,
                "input_schema": QWEN_INPUT_SCHEMA_VERSION,
                "runtime_identity": runtime_identity,
                "frame_count": self.frame_count,
                "max_tokens": self.max_tokens,
                "video_fps": self.video_fps,
            },
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
        payload = {
            "key": key,
            "judgement": asdict(judgement),
        }
        atomic_write_json(path, payload, sort_keys=True)

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
        return self._rerank(query, candidates, raise_on_error=False)

    def rerank_strict(
        self,
        query: str,
        candidates: list[FusedResult],
    ) -> list[FusedResult]:
        return self._rerank(query, candidates, raise_on_error=True)

    def _rerank(
        self,
        query: str,
        candidates: list[FusedResult],
        *,
        raise_on_error: bool,
    ) -> list[FusedResult]:
        if (
            not self.model_name
            or not candidates
            or (self.inference_client is None and not self.allow_in_process)
        ):
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
                    if raise_on_error:
                        raise
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
                    and 0 <= judgement.event_start < judgement.event_end
                    and judgement.event_end <= clip_end - clip_start
                ):
                    start = clip_start + judgement.event_start
                    end = clip_start + judgement.event_end

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
                        "model": self.model_identity,
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
