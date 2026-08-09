from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
import math
from pathlib import Path
import re
from statistics import mean
from time import perf_counter
from typing import Callable, Literal, Protocol

from videoscope.storage import atomic_write_json


logger = logging.getLogger(__name__)


EVALUATION_REPORT_SCHEMA_VERSION = 3
EVALUATION_METHODOLOGY_VERSION = 3
EVALUATION_CASES_FILE_VERSION = 1
EVALUATION_MAX_CASES = 200
EVALUATION_VIDEO_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
EVALUATION_VIDEO_ID_RE = re.compile(EVALUATION_VIDEO_ID_PATTERN)
EVALUATION_MODES = frozenset({"all", "speech", "visual", "ocr"})
EVALUATION_LABEL_SOURCES = frozenset({"gold", "silver"})
DEFAULT_RUNTIME_REVISION = sha256(b"videoscope-unspecified-runtime-v1").hexdigest()
_REVISION_RE = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_COMPONENT_ATTRIBUTES = (
    "id",
    "available",
    "identity",
    "embedding_identity",
    "model_identity",
    "model_name",
    "model_revision",
    "model_id",
    "checkpoint_sha256",
    "collection_name",
    "dimensions",
    "batch_size",
    "threshold",
    "minimum_confidence",
    "max_scene_seconds",
    "max_window_seconds",
    "top_candidates",
    "context_seconds",
    "min_clip_seconds",
    "max_clip_seconds",
    "frame_count",
    "video_fps",
    "max_tokens",
    "sample_step",
    "min_score",
    "score_drop",
    "max_candidate_seconds",
    "timeout",
    "language",
    "initial_prompt",
    "semantic_text_min_score",
    "visual_min_score",
)
_RUNTIME_SETTINGS_ATTRIBUTES = (
    "scene_threshold",
    "max_scene_seconds",
    "whisper_model",
    "whisper_language",
    "whisper_initial_prompt",
    "text_embedding_model",
    "text_embedding_dimensions",
    "siglip_model",
    "siglip_batch_size",
    "semantic_text_min_score",
    "visual_min_score",
    "temporal_refinement_candidates",
    "temporal_refinement_step",
    "roboflow_model_id",
    "qwen_video_model",
    "qwen_video_top_candidates",
    "qwen_video_context_seconds",
    "qwen_video_min_clip_seconds",
    "qwen_video_max_clip_seconds",
    "qwen_video_frame_count",
    "qwen_video_fps",
    "internvideo_top_candidates",
    "internvideo_timeout",
)


class EvaluationDataError(ValueError):
    """The persisted evaluation data does not satisfy the domain contract."""


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    query: str
    video_id: str
    start: float
    end: float
    mode: str = "all"
    label_source: str = "gold"
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    query: str
    relevant_rank: int | None
    temporal_iou: float
    latency_ms: float
    result_count: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class VariantMetrics:
    name: str
    total_case_count: int
    successful_case_count: int
    error_count: int
    status: Literal["complete", "partial", "failed"]
    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    mrr: float | None
    mean_temporal_iou: float | None
    mean_latency_ms: float | None
    cases: list[CaseEvaluation]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    generated_at: str
    schema_version: int
    methodology_version: int
    evaluation_revision: str
    runtime_revision: str
    cases_revision: str
    temporal_iou_threshold: float
    variants: list[VariantMetrics]


class SearchRunner(Protocol):
    def search(
        self,
        query: str,
        *,
        video_ids: list[str] | None = None,
        limit: int = 20,
        use_lighthouse: bool = True,
        mode: str = "all",
    ) -> list[object]: ...


class EvaluationVideoRepository(Protocol):
    def get_video(self, video_id: str) -> object | None: ...


VARIANTS: dict[str, tuple[str, bool]] = {
    "auto": ("all", False),
    "auto_lighthouse": ("all", True),
    "speech": ("speech", False),
    "visual": ("visual", False),
    "visual_lighthouse": ("visual", True),
    "ocr": ("ocr", False),
}
RELEVANCE_IOU_THRESHOLD = 0.3


def temporal_iou(first_start: float, first_end: float, second_start: float, second_end: float) -> float:
    intersection = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    union = max(first_end, second_end) - min(first_start, second_start)
    return intersection / union if union > 0 else 0.0


def evaluation_cases_revision(cases: list[EvaluationCase]) -> str:
    """Return an order-independent SHA-256 over the complete evaluation dataset."""
    canonical_cases = [
        {
            "id": case.id,
            "query": case.query,
            "video_id": case.video_id,
            "start": float(case.start),
            "end": float(case.end),
            "mode": case.mode,
            "label_source": case.label_source,
            "notes": case.notes,
        }
        for case in cases
    ]
    canonical_cases.sort(
        key=lambda case: (
            case["id"],
            json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    )
    canonical_json = json.dumps(
        canonical_cases,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical_json.encode("utf-8")).hexdigest()


def _qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _json_runtime_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or type(value) in {str, int, float, bool}:
        return value
    if isinstance(value, Path):
        return value.name
    if isinstance(value, (list, tuple)):
        return [_json_runtime_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_runtime_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return str(value)


def _runtime_component_descriptor(component: object | None) -> dict[str, object] | None:
    if component is None:
        return None
    descriptor: dict[str, object] = {"type": _qualified_type(component)}
    for name in _RUNTIME_COMPONENT_ATTRIBUTES:
        try:
            value = getattr(component, name)
        except Exception:
            descriptor[name] = "unavailable"
            continue
        if callable(value):
            continue
        descriptor[name] = _json_runtime_value(value)
    for nested_name in ("embedding", "scorer"):
        nested = getattr(component, nested_name, None)
        if nested is not None and nested is not component:
            descriptor[nested_name] = _runtime_component_descriptor(nested)
    return descriptor


def evaluation_provider_snapshot(providers: object) -> list[dict[str, object]]:
    """Capture provider identity/availability once for a runtime instance."""
    statuses_method = getattr(providers, "statuses", None)
    get_provider = getattr(providers, "get", None)
    provider_descriptors: list[dict[str, object]] = []
    if not callable(statuses_method):
        return provider_descriptors
    try:
        statuses = statuses_method()
    except Exception:
        return [{"registry": "unavailable"}]
    for status in sorted(statuses, key=lambda item: str(getattr(item, "id", ""))):
        provider_id = str(getattr(status, "id", ""))
        provider = get_provider(provider_id) if callable(get_provider) else None
        provider_descriptors.append({
            "id": provider_id,
            "state": str(getattr(status, "state", "unknown")),
            "optional": bool(getattr(status, "optional", False)),
            "component": _runtime_component_descriptor(provider),
        })
    return provider_descriptors


def evaluation_runtime_revision(
    search: object,
    providers: object | None = None,
    settings: object | None = None,
) -> str:
    """Fingerprint the active retrieval stack without exposing its configuration."""
    descriptor: dict[str, object] = {
        "search": _runtime_component_descriptor(search),
        "components": {
            name: _runtime_component_descriptor(getattr(search, name, None))
            for name in (
                "vector_index",
                "visual_search",
                "moment_search",
                "query_router",
                "temporal_refiner",
                "candidate_reranker",
            )
        },
    }
    lexicon = getattr(search, "lexicon", None)
    read_lexicon = getattr(lexicon, "read", None)
    if callable(read_lexicon):
        try:
            descriptor["lexicon"] = _json_runtime_value(read_lexicon())
        except Exception:
            descriptor["lexicon"] = "unavailable"
    if providers is not None:
        descriptor["providers"] = (
            _json_runtime_value(providers)
            if isinstance(providers, list)
            else evaluation_provider_snapshot(providers)
        )
    if settings is not None:
        descriptor["settings"] = {
            name: _json_runtime_value(getattr(settings, name))
            for name in _RUNTIME_SETTINGS_ATTRIBUTES
            if hasattr(settings, name)
        }
    canonical_json = json.dumps(
        descriptor,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical_json.encode("utf-8")).hexdigest()


def evaluation_revision(
    cases: list[EvaluationCase],
    runtime_revision: str = DEFAULT_RUNTIME_REVISION,
) -> str:
    """Fingerprint dataset, runtime and metric/report methodology."""
    canonical_json = json.dumps(
        {
            "cases_revision": evaluation_cases_revision(cases),
            "runtime_revision": runtime_revision,
            "methodology_version": EVALUATION_METHODOLOGY_VERSION,
            "report_schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
            "relevance_iou_threshold": RELEVANCE_IOU_THRESHOLD,
            "variants": {
                name: {"mode": mode, "use_lighthouse": use_lighthouse}
                for name, (mode, use_lighthouse) in sorted(VARIANTS.items())
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical_json.encode("utf-8")).hexdigest()


def validate_evaluation_video_references(
    cases: list[EvaluationCase],
    repository: EvaluationVideoRepository,
) -> None:
    for case in cases:
        video = repository.get_video(case.video_id)
        if video is None:
            raise ValueError("Evaluation video does not exist")
        if getattr(video, "status", None) != "ready":
            raise ValueError("Evaluation video is not ready")
        duration_value = getattr(video, "duration", None)
        if (
            isinstance(duration_value, bool)
            or not isinstance(duration_value, (int, float))
            or not math.isfinite(float(duration_value))
            or float(duration_value) <= 0
        ):
            raise ValueError("Evaluation video duration is unavailable")
        if case.end > float(duration_value):
            raise ValueError("Evaluation interval exceeds video duration")


def _invalid_cases(reason: str | None = None) -> EvaluationDataError:
    message = "Evaluation cases are invalid"
    if reason:
        message = f"{message}: {reason}"
    return EvaluationDataError(message)


def _normalized_string(value: object, *, minimum: int, maximum: int) -> str:
    if type(value) is not str:
        raise _invalid_cases()
    normalized = value.strip()
    if not minimum <= len(normalized) <= maximum:
        raise _invalid_cases()
    return normalized


def _normalize_case(case: EvaluationCase) -> EvaluationCase:
    identifier = _normalized_string(case.id, minimum=1, maximum=100)
    query = _normalized_string(case.query, minimum=1, maximum=500)
    video_id = _normalized_string(case.video_id, minimum=1, maximum=64)
    mode = _normalized_string(case.mode, minimum=1, maximum=20)
    label_source = _normalized_string(case.label_source, minimum=1, maximum=20)
    notes = _normalized_string(case.notes, minimum=0, maximum=500)
    if EVALUATION_VIDEO_ID_RE.fullmatch(video_id) is None:
        raise _invalid_cases()
    if mode not in EVALUATION_MODES or label_source not in EVALUATION_LABEL_SOURCES:
        raise _invalid_cases()
    if (
        isinstance(case.start, bool)
        or isinstance(case.end, bool)
        or not isinstance(case.start, (int, float))
        or not isinstance(case.end, (int, float))
    ):
        raise _invalid_cases()
    start = float(case.start)
    end = float(case.end)
    if not math.isfinite(start) or not math.isfinite(end):
        raise _invalid_cases()
    if start < 0 or end <= 0 or end <= start:
        raise _invalid_cases("interval must be positive and ordered")
    return EvaluationCase(
        id=identifier,
        query=query,
        video_id=video_id,
        start=start,
        end=end,
        mode=mode,
        label_source=label_source,
        notes=notes,
    )


def _normalize_cases(cases: list[EvaluationCase]) -> list[EvaluationCase]:
    if len(cases) > EVALUATION_MAX_CASES:
        raise _invalid_cases()
    normalized = [_normalize_case(case) for case in cases]
    identifiers = [case.id for case in normalized]
    if len(set(identifiers)) != len(identifiers):
        raise _invalid_cases("case ids must be unique")
    return normalized


def _finite_number(
    value: object,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return (
        math.isfinite(number)
        and number >= minimum
        and (maximum is None or number <= maximum)
    )


def _report_payload_is_valid(payload: dict[str, object]) -> bool:
    allowed_report_fields = {
        "generated_at",
        "schema_version",
        "methodology_version",
        "evaluation_revision",
        "runtime_revision",
        "cases_revision",
        "temporal_iou_threshold",
        "variants",
    }
    if set(payload) - allowed_report_fields:
        return False
    for version_name in ("schema_version", "methodology_version"):
        version = payload.get(version_name)
        if version is not None and (type(version) is not int or version < 1):
            return False
    for revision_name in (
        "evaluation_revision",
        "runtime_revision",
        "cases_revision",
    ):
        revision = payload.get(revision_name)
        if revision is not None and (
            type(revision) is not str or _REVISION_RE.fullmatch(revision) is None
        ):
            return False
    if payload.get("schema_version") == EVALUATION_REPORT_SCHEMA_VERSION and any(
        payload.get(field) is None
        for field in (
            "methodology_version",
            "evaluation_revision",
            "runtime_revision",
            "cases_revision",
            "temporal_iou_threshold",
        )
    ):
        return False
    threshold = payload.get("temporal_iou_threshold")
    if threshold is not None and not _finite_number(threshold, maximum=1.0):
        return False
    variants = payload.get("variants")
    if not isinstance(variants, list):
        return False
    variant_names: list[str] = []
    allowed_variant_fields = {
        "name",
        "total_case_count",
        "successful_case_count",
        "error_count",
        "status",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "mrr",
        "mean_temporal_iou",
        "mean_latency_ms",
        "cases",
    }
    allowed_case_fields = {
        "case_id",
        "query",
        "relevant_rank",
        "temporal_iou",
        "latency_ms",
        "result_count",
        "error",
    }
    for variant in variants:
        if not isinstance(variant, dict) or set(variant) != allowed_variant_fields:
            return False
        name = variant.get("name")
        if type(name) is not str or name not in VARIANTS:
            return False
        variant_names.append(name)
        total = variant.get("total_case_count")
        successful = variant.get("successful_case_count")
        errors = variant.get("error_count")
        if (
            type(total) is not int
            or type(successful) is not int
            or type(errors) is not int
            or min(total, successful, errors) < 0
            or successful + errors != total
        ):
            return False
        expected_status = (
            "complete" if errors == 0 else "failed" if successful == 0 else "partial"
        )
        if variant.get("status") != expected_status:
            return False
        cases = variant.get("cases")
        if not isinstance(cases, list) or len(cases) != total:
            return False
        observed_successful = 0
        for case in cases:
            if not isinstance(case, dict) or set(case) != allowed_case_fields:
                return False
            rank = case.get("relevant_rank")
            if rank is not None and (type(rank) is not int or rank < 1):
                return False
            if (
                type(case.get("case_id")) is not str
                or type(case.get("query")) is not str
                or not _finite_number(case.get("temporal_iou"), maximum=1.0)
                or not _finite_number(case.get("latency_ms"))
                or type(case.get("result_count")) is not int
                or case["result_count"] < 0
                or (case.get("error") is not None and type(case.get("error")) is not str)
            ):
                return False
            if case.get("error") is None:
                observed_successful += 1
        if observed_successful != successful:
            return False
        metric_names = (
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "mrr",
            "mean_temporal_iou",
        )
        metrics = [variant.get(metric_name) for metric_name in metric_names]
        latency = variant.get("mean_latency_ms")
        if successful == 0:
            if any(metric is not None for metric in [*metrics, latency]):
                return False
        elif (
            any(not _finite_number(metric, maximum=1.0) for metric in metrics)
            or not _finite_number(latency)
        ):
            return False
    return len(set(variant_names)) == len(variant_names)


class EvaluationStore:
    def __init__(self, cases_path: Path, report_path: Path) -> None:
        self.cases_path = Path(cases_path)
        self.report_path = Path(report_path)

    @staticmethod
    def _write(path: Path, payload: object) -> None:
        atomic_write_json(path, payload)

    def read_cases(self) -> list[EvaluationCase]:
        if not self.cases_path.is_file():
            return []
        try:
            payload = json.loads(self.cases_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise _invalid_cases() from error
        if not isinstance(payload, dict):
            raise _invalid_cases()
        version = payload.get("version", EVALUATION_CASES_FILE_VERSION)
        raw_cases = payload.get("cases")
        if (
            type(version) is not int
            or version != EVALUATION_CASES_FILE_VERSION
            or not isinstance(raw_cases, list)
        ):
            raise _invalid_cases()
        output: list[EvaluationCase] = []
        for raw in raw_cases:
            if not isinstance(raw, dict):
                raise _invalid_cases()
            try:
                output.append(EvaluationCase(**raw))
            except (TypeError, ValueError) as error:
                raise _invalid_cases() from error
        return _normalize_cases(output)

    def replace_cases(self, cases: list[EvaluationCase]) -> None:
        normalized = _normalize_cases(cases)
        self._write(
            self.cases_path,
            {
                "version": EVALUATION_CASES_FILE_VERSION,
                "cases": [asdict(case) for case in normalized],
            },
        )

    def write_report(self, report: EvaluationReport) -> None:
        self._write(self.report_path, asdict(report))

    def read_report(self) -> dict[str, object] | None:
        if not self.report_path.is_file():
            return None
        try:
            payload = json.loads(self.report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        variants = payload.get("variants")
        if type(payload.get("generated_at")) is not str or not isinstance(variants, list):
            return None
        normalized_variants: list[dict[str, object]] = []
        for raw_variant in variants:
            if not isinstance(raw_variant, dict) or not isinstance(raw_variant.get("cases", []), list):
                return None
            variant = dict(raw_variant)
            legacy_count = variant.pop("case_count", None)
            is_legacy_variant = legacy_count is not None
            total_count = variant.get("total_case_count", legacy_count)
            cases = variant.get("cases", [])
            if type(total_count) is not int or total_count < 0:
                return None
            inferred_error_count = sum(
                isinstance(case, dict) and case.get("error") is not None
                for case in cases
            )
            error_count = variant.get(
                "error_count",
                inferred_error_count if is_legacy_variant else None,
            )
            if type(error_count) is not int or not 0 <= error_count <= total_count:
                return None
            successful_count = variant.get(
                "successful_case_count",
                total_count - error_count if is_legacy_variant else None,
            )
            if (
                type(successful_count) is not int
                or successful_count < 0
                or successful_count + error_count != total_count
            ):
                return None
            status_value: Literal["complete", "partial", "failed"]
            if error_count == 0:
                status_value = "complete"
            elif successful_count == 0:
                status_value = "failed"
            else:
                status_value = "partial"
            variant["total_case_count"] = total_count
            variant["successful_case_count"] = successful_count
            variant["error_count"] = error_count
            if is_legacy_variant:
                variant["status"] = status_value
            elif variant.get("status") != status_value:
                return None
            if is_legacy_variant:
                if len(cases) != total_count:
                    return None
                successful_cases = [
                    case
                    for case in cases
                    if isinstance(case, dict) and case.get("error") is None
                ]
                if len(successful_cases) != successful_count:
                    return None
                if successful_count == 0:
                    for metric_name in (
                        "recall_at_1",
                        "recall_at_3",
                        "recall_at_5",
                        "mrr",
                        "mean_temporal_iou",
                        "mean_latency_ms",
                    ):
                        variant[metric_name] = None
                else:
                    ranks: list[int | None] = []
                    overlaps: list[float] = []
                    latencies: list[float] = []
                    for case in successful_cases:
                        rank = case.get("relevant_rank")
                        overlap = case.get("temporal_iou")
                        latency = case.get("latency_ms")
                        if (
                            (rank is not None and (type(rank) is not int or rank < 1))
                            or isinstance(overlap, bool)
                            or not isinstance(overlap, (int, float))
                            or not math.isfinite(float(overlap))
                            or not 0 <= float(overlap) <= 1
                            or isinstance(latency, bool)
                            or not isinstance(latency, (int, float))
                            or not math.isfinite(float(latency))
                            or float(latency) < 0
                        ):
                            return None
                        ranks.append(rank)
                        overlaps.append(float(overlap))
                        latencies.append(float(latency))
                    variant["recall_at_1"] = sum(rank is not None and rank <= 1 for rank in ranks) / successful_count
                    variant["recall_at_3"] = sum(rank is not None and rank <= 3 for rank in ranks) / successful_count
                    variant["recall_at_5"] = sum(rank is not None and rank <= 5 for rank in ranks) / successful_count
                    variant["mrr"] = sum(1 / rank for rank in ranks if rank) / successful_count
                    variant["mean_temporal_iou"] = mean(overlaps)
                    variant["mean_latency_ms"] = mean(latencies)
            normalized_variants.append(variant)
        payload["variants"] = normalized_variants
        # Legacy reports remain readable, while null methodology metadata makes
        # their freshness fingerprint unambiguously stale.
        payload.setdefault("schema_version", None)
        payload.setdefault("methodology_version", None)
        payload.setdefault("evaluation_revision", None)
        payload.setdefault("runtime_revision", None)
        payload.setdefault("cases_revision", None)
        payload.setdefault("temporal_iou_threshold", None)
        return payload if _report_payload_is_valid(payload) else None


class EvaluationService:
    def __init__(
        self,
        search: SearchRunner,
        store: EvaluationStore,
        *,
        repository: EvaluationVideoRepository | None = None,
        runtime_revision: Callable[[], str] | None = None,
    ) -> None:
        self.search = search
        self.store = store
        self.repository = repository
        self._runtime_revision = runtime_revision or (
            lambda: evaluation_runtime_revision(search)
        )

    def current_runtime_revision(self) -> str:
        revision = str(self._runtime_revision())
        if _REVISION_RE.fullmatch(revision):
            return revision
        return sha256(revision.encode("utf-8")).hexdigest()

    def run(self, variants: list[str] | None = None) -> EvaluationReport:
        selected = variants or ["auto"]
        unknown = set(selected) - VARIANTS.keys()
        if unknown:
            raise ValueError(f"unsupported evaluation variants: {sorted(unknown)}")
        cases = self.store.read_cases()
        if not cases:
            raise ValueError("evaluation requires at least one case")
        if self.repository is not None:
            validate_evaluation_video_references(cases, self.repository)
        runtime_revision = self.current_runtime_revision()
        report = EvaluationReport(
            generated_at=datetime.now(UTC).isoformat(),
            schema_version=EVALUATION_REPORT_SCHEMA_VERSION,
            methodology_version=EVALUATION_METHODOLOGY_VERSION,
            evaluation_revision=evaluation_revision(cases, runtime_revision),
            runtime_revision=runtime_revision,
            cases_revision=evaluation_cases_revision(cases),
            temporal_iou_threshold=RELEVANCE_IOU_THRESHOLD,
            variants=[self._run_variant(name, cases) for name in selected],
        )
        self.store.write_report(report)
        return report

    def _run_variant(self, name: str, cases: list[EvaluationCase]) -> VariantMetrics:
        mode, use_lighthouse = VARIANTS[name]
        evaluations: list[CaseEvaluation] = []
        search_for_case = getattr(
            self.search,
            "search_for_evaluation",
            self.search.search,
        )
        for case in cases:
            started = perf_counter()
            error = None
            relevant_rank = None
            overlap = 0.0
            result_count = 0
            try:
                results = search_for_case(
                    case.query,
                    video_ids=[case.video_id],
                    limit=20,
                    use_lighthouse=use_lighthouse,
                    mode=mode,
                )
                result_count = len(results)
                for rank, result in enumerate(results, start=1):
                    if str(getattr(result, "video_id", "")) != case.video_id:
                        continue
                    result_start = float(getattr(result, "start", float("nan")))
                    result_end = float(getattr(result, "end", float("nan")))
                    if (
                        not math.isfinite(result_start)
                        or not math.isfinite(result_end)
                        or result_start < 0
                        or result_end <= result_start
                    ):
                        raise ValueError("invalid search result interval")
                    current_iou = temporal_iou(
                        case.start,
                        case.end,
                        result_start,
                        result_end,
                    )
                    overlap = max(overlap, current_iou)
                    if relevant_rank is None and current_iou >= RELEVANCE_IOU_THRESHOLD:
                        relevant_rank = rank
            except Exception:
                error = "Search failed"
                relevant_rank = None
                overlap = 0.0
                result_count = 0
                logger.exception(
                    "Evaluation search failed",
                    extra={"case_id": case.id, "variant": name},
                )
            latency_ms = (perf_counter() - started) * 1000
            evaluations.append(
                CaseEvaluation(
                    case_id=case.id,
                    query=case.query,
                    relevant_rank=relevant_rank,
                    temporal_iou=overlap,
                    latency_ms=latency_ms,
                    result_count=result_count,
                    error=error,
                )
            )

        successful = [item for item in evaluations if item.error is None]
        total_count = len(evaluations)
        successful_count = len(successful)
        error_count = total_count - successful_count
        if error_count == 0:
            status_value: Literal["complete", "partial", "failed"] = "complete"
        elif successful_count == 0:
            status_value = "failed"
        else:
            status_value = "partial"
        return VariantMetrics(
            name=name,
            total_case_count=total_count,
            successful_case_count=successful_count,
            error_count=error_count,
            status=status_value,
            recall_at_1=(
                sum(item.relevant_rank is not None and item.relevant_rank <= 1 for item in successful)
                / successful_count
                if successful_count
                else None
            ),
            recall_at_3=(
                sum(item.relevant_rank is not None and item.relevant_rank <= 3 for item in successful)
                / successful_count
                if successful_count
                else None
            ),
            recall_at_5=(
                sum(item.relevant_rank is not None and item.relevant_rank <= 5 for item in successful)
                / successful_count
                if successful_count
                else None
            ),
            mrr=(
                sum(1 / item.relevant_rank for item in successful if item.relevant_rank)
                / successful_count
                if successful_count
                else None
            ),
            mean_temporal_iou=(mean(item.temporal_iou for item in successful) if successful else None),
            mean_latency_ms=(mean(item.latency_ms for item in successful) if successful else None),
            cases=evaluations,
        )
