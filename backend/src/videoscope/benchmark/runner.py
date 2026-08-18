from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import islice
import math
from statistics import mean
from time import perf_counter
from typing import Callable, Literal, Protocol, Sequence

from videoscope.evaluation import RELEVANCE_IOU_THRESHOLD, temporal_iou

from .catalog import AssetResolutionError, LocalAssetResolver, ResolvedAsset
from .profiles import BenchmarkProfile, get_profile
from .schema import (
    RUN_SCHEMA_VERSION,
    BenchmarkCaseOutcome,
    BenchmarkDataError,
    BenchmarkDataset,
    BenchmarkRunManifest,
    BenchmarkResultEvidence,
    ComponentIdentity,
    HardwareProfile,
    MetricValue,
    QueryCase,
    _CODE_SHA_RE,
    _require_finite_float,
    _require_id,
    _require_tuple,
)
from .serialization import dataset_revision
from .storage import BenchmarkRunRegistry


BENCHMARK_METHODOLOGY_VERSION = 1
_COMPLETE_CAPABILITY_STATE = "complete"
_KNOWN_INCOMPLETE_STATES = frozenset(
    {"queued", "running", "cancelled", "failed", "stale", "not_configured"}
)
_SLICE_DIMENSIONS = ("domain", "modality", "label_quality", "split_group")


class BenchmarkExecutionError(RuntimeError):
    """The benchmark cannot produce a trustworthy immutable run."""


@dataclass(frozen=True, slots=True)
class BenchmarkSearchHit:
    asset_id: str
    start_seconds: float
    end_seconds: float
    score: float

    def __post_init__(self) -> None:
        _require_id(self.asset_id, "benchmark search hit asset_id")
        start = _require_finite_float(
            self.start_seconds,
            "benchmark search hit start_seconds",
            minimum=0,
        )
        end = _require_finite_float(
            self.end_seconds,
            "benchmark search hit end_seconds",
            minimum=0,
        )
        if end <= start:
            raise BenchmarkDataError(
                "benchmark search hit end_seconds must be greater than start_seconds"
            )
        score = _require_finite_float(self.score, "benchmark search hit score")
        if not 0 <= score <= 1:
            raise BenchmarkDataError(
                "benchmark search hit score must be between zero and one"
            )
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(self, "score", score)


@dataclass(frozen=True, slots=True)
class ExecutionIdentities:
    model_identities: tuple[ComponentIdentity, ...]
    index_identities: tuple[ComponentIdentity, ...]
    config_identities: tuple[ComponentIdentity, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "model_identities",
            "index_identities",
            "config_identities",
        ):
            values = _require_tuple(getattr(self, field_name), field_name)
            if not values:
                raise BenchmarkDataError(f"{field_name} must not be empty")
            if any(not isinstance(value, ComponentIdentity) for value in values):
                raise BenchmarkDataError(
                    f"{field_name} must contain only ComponentIdentity values"
                )
            component_ids = tuple(value.component_id for value in values)
            if len(component_ids) != len(set(component_ids)):
                raise BenchmarkDataError(f"{field_name} must contain unique identities")


class BenchmarkSearchAdapter(Protocol):
    """Boundary between the portable runner and the product search stack.

    A concrete adapter is responsible for mapping ``ResolvedAsset.video_id``
    to the current search service and translating results back to portable
    asset aliases.  It must use the fail-closed evaluation search path.
    """

    def identities(self, profile: BenchmarkProfile) -> ExecutionIdentities: ...

    def capability_state(
        self,
        profile: BenchmarkProfile,
        asset: ResolvedAsset,
        capability: str,
    ) -> str: ...

    def search(
        self,
        profile: BenchmarkProfile,
        query: str,
        assets: tuple[ResolvedAsset, ...],
        *,
        limit: int,
    ) -> Sequence[BenchmarkSearchHit]: ...


@dataclass(frozen=True, slots=True)
class _CaseScore:
    case: QueryCase
    latency_ms: float
    result_count: int
    relevant_rank: int | None
    best_temporal_iou: float
    false_positive_count: int
    negative_false_positive: int
    hard_negative_hit: int
    hits: tuple[BenchmarkSearchHit, ...]


class _CaseFailure(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class BenchmarkRunner:
    def __init__(
        self,
        *,
        registry: BenchmarkRunRegistry,
        asset_resolver: LocalAssetResolver,
        search: BenchmarkSearchAdapter,
        hardware: HardwareProfile,
        code_sha: str,
        clock: Callable[[], datetime] | None = None,
        timer: Callable[[], float] = perf_counter,
        overlap: Callable[[float, float, float, float], float] = temporal_iou,
    ) -> None:
        self.registry = registry
        self._asset_resolver = asset_resolver
        self._search = search
        self._hardware = hardware
        self._code_sha = code_sha
        self._clock = clock or (lambda: datetime.now(UTC))
        self._timer = timer
        self._overlap = overlap

    def run(
        self,
        dataset: BenchmarkDataset,
        *,
        profile_id: str,
        run_id: str,
        execution_mode: Literal["cold", "warm"],
    ) -> BenchmarkRunManifest:
        if not isinstance(dataset, BenchmarkDataset):
            raise BenchmarkExecutionError("benchmark dataset contract is invalid")
        if not dataset.cases:
            raise BenchmarkExecutionError(
                "benchmark run requires at least one query case"
            )
        if not isinstance(self._hardware, HardwareProfile):
            raise BenchmarkExecutionError("benchmark hardware identity is invalid")
        if not isinstance(self._code_sha, str) or not _CODE_SHA_RE.fullmatch(
            self._code_sha
        ):
            raise BenchmarkExecutionError("benchmark code identity is invalid")
        _require_id(run_id, "run_id")
        if execution_mode not in {"cold", "warm"}:
            raise BenchmarkDataError("execution_mode must be cold or warm")
        profile = get_profile(profile_id)
        if any(entry.run_id == run_id for entry in self.registry.list()):
            raise FileExistsError(f"benchmark run {run_id!r} already exists")
        created_at = self._created_at()
        identities = self._execution_identities(profile)
        resolved, resolution_failures = self._resolve_assets(dataset)

        outcomes: list[BenchmarkCaseOutcome] = []
        scores: list[_CaseScore] = []
        for case in sorted(dataset.cases, key=lambda item: item.case_id):
            started = self._read_timer()
            score: _CaseScore | None = None
            diagnostic_code: str | None = None
            try:
                case_assets = self._assets_for_case(
                    case,
                    resolved,
                    resolution_failures,
                )
                self._require_capabilities(profile, case_assets)
                hits = self._search_case(profile, case, case_assets)
                self._validate_hits(profile, case, case_assets, hits)
                score = self._score_case(case, hits, latency_ms=0.0)
            except _CaseFailure as exc:
                diagnostic_code = exc.code
            latency_ms = self._elapsed_ms(started)
            if score is None:
                outcomes.append(
                    BenchmarkCaseOutcome(
                        case_id=case.case_id,
                        status="failed",
                        latency_ms=latency_ms,
                        result_count=0,
                        diagnostic_code=diagnostic_code or "execution_failed",
                    )
                )
                continue
            score = _CaseScore(
                case=score.case,
                latency_ms=latency_ms,
                result_count=score.result_count,
                relevant_rank=score.relevant_rank,
                best_temporal_iou=score.best_temporal_iou,
                false_positive_count=score.false_positive_count,
                negative_false_positive=score.negative_false_positive,
                hard_negative_hit=score.hard_negative_hit,
                hits=score.hits,
            )
            scores.append(score)
            outcomes.append(self._complete_outcome(score))

        ordered_outcomes = tuple(sorted(outcomes, key=lambda item: item.case_id))
        metrics = self._run_metrics(dataset, ordered_outcomes, tuple(scores))
        run = BenchmarkRunManifest(
            schema_version=RUN_SCHEMA_VERSION,
            run_id=run_id,
            created_at=created_at,
            code_sha=self._code_sha,
            dataset_revision=dataset_revision(dataset),
            model_identities=identities.model_identities,
            index_identities=identities.index_identities,
            config_identities=identities.config_identities,
            hardware=self._hardware,
            execution_mode=execution_mode,
            metrics=metrics,
            case_outcomes=ordered_outcomes,
        )
        self.registry.add(run)
        return run

    def _execution_identities(self, profile: BenchmarkProfile) -> ExecutionIdentities:
        try:
            identities = self._search.identities(profile)
        except Exception:
            raise BenchmarkExecutionError(
                "stable benchmark execution identity is unavailable"
            ) from None
        if not isinstance(identities, ExecutionIdentities):
            raise BenchmarkExecutionError(
                "stable benchmark execution identity has an invalid contract"
            )
        reserved = {"benchmark_profile", "benchmark_methodology"}
        if reserved & {
            identity.component_id for identity in identities.config_identities
        }:
            raise BenchmarkExecutionError(
                "execution identity uses a reserved benchmark component id"
            )
        config_identities = (
            *identities.config_identities,
            ComponentIdentity("benchmark_profile", profile.identity),
            ComponentIdentity(
                "benchmark_methodology",
                _methodology_identity(),
            ),
        )
        return ExecutionIdentities(
            model_identities=_ordered_identities(identities.model_identities),
            index_identities=_ordered_identities(identities.index_identities),
            config_identities=_ordered_identities(config_identities),
        )

    def _resolve_assets(
        self,
        dataset: BenchmarkDataset,
    ) -> tuple[dict[str, ResolvedAsset], dict[str, str]]:
        resolved: dict[str, ResolvedAsset] = {}
        failures: dict[str, str] = {}
        for asset in sorted(dataset.assets, key=lambda item: item.asset_id):
            try:
                resolved[asset.asset_id] = self._asset_resolver.resolve(asset)
            except AssetResolutionError as exc:
                failures[asset.asset_id] = exc.code
            except Exception:
                failures[asset.asset_id] = "asset_resolution_failed"
        return resolved, failures

    @staticmethod
    def _assets_for_case(
        case: QueryCase,
        resolved: dict[str, ResolvedAsset],
        failures: dict[str, str],
    ) -> tuple[ResolvedAsset, ...]:
        for asset_id in case.asset_ids:
            diagnostic = failures.get(asset_id)
            if diagnostic is not None:
                raise _CaseFailure(diagnostic)
        try:
            return tuple(resolved[asset_id] for asset_id in case.asset_ids)
        except KeyError:
            raise _CaseFailure("asset_resolution_failed") from None

    def _require_capabilities(
        self,
        profile: BenchmarkProfile,
        assets: tuple[ResolvedAsset, ...],
    ) -> None:
        for asset in assets:
            for capability in profile.required_capabilities:
                try:
                    state = self._search.capability_state(
                        profile,
                        asset,
                        capability,
                    )
                except Exception:
                    raise _CaseFailure("capability_check_failed") from None
                if state == _COMPLETE_CAPABILITY_STATE:
                    continue
                if state == "stale":
                    raise _CaseFailure("capability_stale")
                if state == "missing":
                    raise _CaseFailure("capability_missing")
                if state == "failed":
                    raise _CaseFailure("capability_failed")
                if state == "not_configured":
                    raise _CaseFailure("capability_not_configured")
                if state in _KNOWN_INCOMPLETE_STATES:
                    raise _CaseFailure("capability_incomplete")
                raise _CaseFailure("capability_invalid_state")

    def _search_case(
        self,
        profile: BenchmarkProfile,
        case: QueryCase,
        assets: tuple[ResolvedAsset, ...],
    ) -> tuple[BenchmarkSearchHit, ...]:
        try:
            value = self._search.search(
                profile,
                case.query,
                assets,
                limit=profile.result_limit,
            )
            return tuple(islice(iter(value), profile.result_limit + 1))
        except Exception:
            raise _CaseFailure("provider_failed") from None

    @staticmethod
    def _validate_hits(
        profile: BenchmarkProfile,
        case: QueryCase,
        assets: tuple[ResolvedAsset, ...],
        hits: tuple[BenchmarkSearchHit, ...],
    ) -> None:
        _validate_ranked_hits(
            case,
            {asset.asset_id: asset.duration_seconds for asset in assets},
            hits,
            limit=profile.result_limit,
        )

    def _score_case(
        self,
        case: QueryCase,
        hits: tuple[BenchmarkSearchHit, ...],
        *,
        latency_ms: float,
    ) -> _CaseScore:
        return _score_ranked_hits(
            case,
            hits,
            latency_ms=latency_ms,
            overlap=self._overlap,
        )

    @staticmethod
    def _complete_outcome(score: _CaseScore) -> BenchmarkCaseOutcome:
        case_metrics = [
            MetricValue(
                "false_positive_rate",
                (
                    score.false_positive_count / score.result_count
                    if score.result_count
                    else 0.0
                ),
                "ratio",
            ),
        ]
        if score.case.relevant_intervals:
            case_metrics.extend(
                (
                    MetricValue(
                        "reciprocal_rank",
                        1 / score.relevant_rank if score.relevant_rank else 0.0,
                        "ratio",
                    ),
                    MetricValue(
                        "hit_at_1",
                        float(score.relevant_rank is not None and score.relevant_rank <= 1),
                        "ratio",
                    ),
                    MetricValue(
                        "hit_at_3",
                        float(score.relevant_rank is not None and score.relevant_rank <= 3),
                        "ratio",
                    ),
                    MetricValue(
                        "hit_at_5",
                        float(score.relevant_rank is not None and score.relevant_rank <= 5),
                        "ratio",
                    ),
                    MetricValue(
                        "temporal_iou",
                        score.best_temporal_iou,
                        "ratio",
                    ),
                )
            )
        else:
            case_metrics.append(
                MetricValue(
                    "negative_false_positive",
                    score.negative_false_positive,
                    "ratio",
                )
            )
        if score.case.hard_negatives:
            case_metrics.append(
                MetricValue(
                    "hard_negative_hit",
                    score.hard_negative_hit,
                    "ratio",
                )
            )
        evidence = tuple(
            BenchmarkResultEvidence(
                rank=rank,
                asset_id=hit.asset_id,
                start_seconds=hit.start_seconds,
                end_seconds=hit.end_seconds,
                score=hit.score,
            )
            for rank, hit in enumerate(score.hits, start=1)
        )
        return BenchmarkCaseOutcome(
            case_id=score.case.case_id,
            status="complete",
            latency_ms=score.latency_ms,
            result_count=score.result_count,
            metrics=tuple(sorted(case_metrics, key=lambda item: item.name)),
            result_evidence=evidence,
        )

    @staticmethod
    def _run_metrics(
        dataset: BenchmarkDataset,
        outcomes: tuple[BenchmarkCaseOutcome, ...],
        scores: tuple[_CaseScore, ...],
    ) -> tuple[MetricValue, ...]:
        total_count = len(outcomes)
        completed_count = len(scores)
        error_count = total_count - completed_count
        metrics = [
            MetricValue("total_case_count", total_count, "count"),
            MetricValue("completed_case_count", completed_count, "count"),
            MetricValue("error_count", error_count, "count"),
            MetricValue(
                "error_rate",
                error_count / total_count if total_count else 0.0,
                "ratio",
            ),
        ]
        metrics.extend(_quality_metrics(scores))
        metrics.extend(BenchmarkRunner._slice_metrics(dataset, outcomes, scores))
        return tuple(sorted(metrics, key=lambda item: item.name))

    @staticmethod
    def _slice_metrics(
        dataset: BenchmarkDataset,
        outcomes: tuple[BenchmarkCaseOutcome, ...],
        scores: tuple[_CaseScore, ...],
    ) -> tuple[MetricValue, ...]:
        outcome_by_id = {outcome.case_id: outcome for outcome in outcomes}
        score_by_id = {score.case.case_id: score for score in scores}
        grouped: dict[tuple[str, str], set[str]] = {}
        for case in dataset.cases:
            values = {
                "domain": (case.domain,),
                "modality": case.modalities,
                "label_quality": (case.label_quality,),
                "split_group": (case.split_group,),
            }
            for dimension in _SLICE_DIMENSIONS:
                for value in values[dimension]:
                    grouped.setdefault((dimension, value), set()).add(case.case_id)

        metrics: list[MetricValue] = []
        for (dimension, value), case_ids in sorted(grouped.items()):
            prefix = f"slice.{dimension}.{value}"
            selected_outcomes = tuple(
                outcome_by_id[case_id] for case_id in sorted(case_ids)
            )
            selected_scores = tuple(
                score_by_id[case_id]
                for case_id in sorted(case_ids)
                if case_id in score_by_id
            )
            completed_count = len(selected_scores)
            error_count = len(selected_outcomes) - completed_count
            metrics.extend(
                (
                    MetricValue(
                        _bounded_metric_name(prefix, "case_count"),
                        len(selected_outcomes),
                        "count",
                    ),
                    MetricValue(
                        _bounded_metric_name(prefix, "completed_case_count"),
                        completed_count,
                        "count",
                    ),
                    MetricValue(
                        _bounded_metric_name(prefix, "error_count"),
                        error_count,
                        "count",
                    ),
                )
            )
            metrics.extend(_quality_metrics(selected_scores, prefix=prefix))
        return tuple(metrics)

    def _created_at(self) -> str:
        try:
            value = self._clock()
        except Exception:
            raise BenchmarkExecutionError("benchmark clock failed") from None
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise BenchmarkExecutionError("benchmark clock must return an aware datetime")
        try:
            utc_value = value.astimezone(UTC)
        except Exception:
            raise BenchmarkExecutionError(
                "benchmark clock must return UTC-convertible time"
            ) from None
        return utc_value.isoformat().replace("+00:00", "Z")

    def _read_timer(self) -> float:
        try:
            value = float(self._timer())
        except Exception:
            raise BenchmarkExecutionError("benchmark monotonic timer failed") from None
        if not math.isfinite(value):
            raise BenchmarkExecutionError("benchmark monotonic timer is not finite")
        return value

    def _elapsed_ms(self, started: float) -> float:
        elapsed = (self._read_timer() - started) * 1_000
        if elapsed < 0:
            raise BenchmarkExecutionError("benchmark monotonic timer moved backwards")
        return elapsed


def audit_run_manifest(
    dataset: BenchmarkDataset,
    run: BenchmarkRunManifest,
    *,
    overlap: Callable[[float, float, float, float], float] = temporal_iou,
) -> None:
    """Recompute persisted metrics from portable ranked evidence.

    This is intentionally independent of local repository rows and provider
    state.  It verifies that an immutable run can be audited from its dataset
    revision and portable result evidence alone.
    """

    if not isinstance(dataset, BenchmarkDataset) or not isinstance(
        run, BenchmarkRunManifest
    ):
        raise BenchmarkExecutionError("benchmark run audit failed: invalid contract")
    if run.dataset_revision != dataset_revision(dataset):
        raise BenchmarkExecutionError(
            "benchmark run audit failed: dataset revision mismatch"
        )
    profile_identity = next(
        (
            identity.identity
            for identity in run.config_identities
            if identity.component_id == "benchmark_profile"
        ),
        None,
    )
    methodology_identity = next(
        (
            identity.identity
            for identity in run.config_identities
            if identity.component_id == "benchmark_methodology"
        ),
        None,
    )
    if profile_identity is None or methodology_identity != _methodology_identity():
        raise BenchmarkExecutionError(
            "benchmark run audit failed: methodology identity mismatch"
        )
    profile_id, separator, _version = profile_identity.rpartition("@")
    if not separator:
        raise BenchmarkExecutionError(
            "benchmark run audit failed: profile identity mismatch"
        )
    try:
        profile = get_profile(profile_id)
    except BenchmarkDataError:
        raise BenchmarkExecutionError(
            "benchmark run audit failed: profile identity mismatch"
        ) from None
    if profile.identity != profile_identity:
        raise BenchmarkExecutionError(
            "benchmark run audit failed: profile identity mismatch"
        )

    cases_by_id = {case.case_id: case for case in dataset.cases}
    expected_case_ids = tuple(sorted(cases_by_id))
    actual_case_ids = tuple(outcome.case_id for outcome in run.case_outcomes)
    if actual_case_ids != expected_case_ids:
        raise BenchmarkExecutionError(
            "benchmark run audit failed: case identity mismatch"
        )
    durations = {asset.asset_id: asset.duration_seconds for asset in dataset.assets}
    scores: list[_CaseScore] = []
    for outcome in run.case_outcomes:
        if outcome.status != "complete":
            continue
        case = cases_by_id[outcome.case_id]
        hits = tuple(
            BenchmarkSearchHit(
                asset_id=evidence.asset_id,
                start_seconds=evidence.start_seconds,
                end_seconds=evidence.end_seconds,
                score=evidence.score,
            )
            for evidence in outcome.result_evidence
        )
        try:
            _validate_ranked_hits(case, durations, hits, limit=profile.result_limit)
            score = _score_ranked_hits(
                case,
                hits,
                latency_ms=outcome.latency_ms,
                overlap=overlap,
            )
        except (BenchmarkDataError, _CaseFailure):
            raise BenchmarkExecutionError(
                "benchmark run audit failed: invalid portable evidence"
            ) from None
        if BenchmarkRunner._complete_outcome(score) != outcome:
            raise BenchmarkExecutionError(
                "benchmark run audit failed: per-case metrics mismatch"
            )
        scores.append(score)
    expected_metrics = BenchmarkRunner._run_metrics(
        dataset,
        run.case_outcomes,
        tuple(scores),
    )
    if expected_metrics != run.metrics:
        raise BenchmarkExecutionError(
            "benchmark run audit failed: aggregate metrics mismatch"
        )


def _validate_ranked_hits(
    case: QueryCase,
    durations: dict[str, float],
    hits: tuple[BenchmarkSearchHit, ...],
    *,
    limit: int,
) -> None:
    if len(hits) > limit:
        raise _CaseFailure("invalid_search_result")
    exact_hits: set[tuple[str, float, float]] = set()
    for hit in hits:
        if not isinstance(hit, BenchmarkSearchHit):
            raise _CaseFailure("invalid_search_result")
        duration = durations.get(hit.asset_id)
        if duration is None or hit.asset_id not in case.asset_ids:
            raise _CaseFailure("invalid_search_result")
        if hit.end_seconds > duration:
            raise _CaseFailure("invalid_search_result")
        identity = (hit.asset_id, hit.start_seconds, hit.end_seconds)
        if identity in exact_hits:
            raise _CaseFailure("invalid_search_result")
        exact_hits.add(identity)


def _score_ranked_hits(
    case: QueryCase,
    hits: tuple[BenchmarkSearchHit, ...],
    *,
    latency_ms: float,
    overlap: Callable[[float, float, float, float], float],
) -> _CaseScore:
    relevant_rank: int | None = None
    best_iou = 0.0
    false_positive_count = 0
    hard_negative_hit = 0
    for rank, hit in enumerate(hits, start=1):
        relevant_overlap = max(
            (
                _checked_overlap(
                    overlap,
                    interval.start_seconds,
                    interval.end_seconds,
                    hit.start_seconds,
                    hit.end_seconds,
                )
                for interval in case.relevant_intervals
                if interval.asset_id == hit.asset_id
            ),
            default=0.0,
        )
        best_iou = max(best_iou, relevant_overlap)
        is_relevant = (
            bool(case.relevant_intervals)
            and relevant_overlap >= RELEVANCE_IOU_THRESHOLD
        )
        if is_relevant and relevant_rank is None:
            relevant_rank = rank
        if not is_relevant:
            false_positive_count += 1
        if any(
            negative.asset_id == hit.asset_id
            and _checked_overlap(
                overlap,
                negative.start_seconds,
                negative.end_seconds,
                hit.start_seconds,
                hit.end_seconds,
            )
            >= RELEVANCE_IOU_THRESHOLD
            for negative in case.hard_negatives
        ):
            hard_negative_hit = 1
    return _CaseScore(
        case=case,
        latency_ms=latency_ms,
        result_count=len(hits),
        relevant_rank=relevant_rank,
        best_temporal_iou=best_iou,
        false_positive_count=false_positive_count,
        negative_false_positive=(
            int(bool(hits)) if not case.relevant_intervals else 0
        ),
        hard_negative_hit=hard_negative_hit,
        hits=hits,
    )


def _checked_overlap(
    overlap: Callable[[float, float, float, float], float],
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    try:
        value = float(overlap(first_start, first_end, second_start, second_end))
    except Exception:
        raise _CaseFailure("overlap_evaluation_failed") from None
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise _CaseFailure("overlap_evaluation_failed")
    return value


def _methodology_identity() -> str:
    return (
        f"portable-retrieval@{BENCHMARK_METHODOLOGY_VERSION};"
        f"temporal-iou-threshold={RELEVANCE_IOU_THRESHOLD:g};"
        "quality-denominator=completed-positive-cases"
    )


def _ordered_identities(
    values: tuple[ComponentIdentity, ...],
) -> tuple[ComponentIdentity, ...]:
    return tuple(sorted(values, key=lambda item: (item.component_id, item.identity)))


def _quality_metrics(
    scores: tuple[_CaseScore, ...],
    *,
    prefix: str | None = None,
) -> tuple[MetricValue, ...]:
    def name(suffix: str) -> str:
        return suffix if prefix is None else _bounded_metric_name(prefix, suffix)

    metrics: list[MetricValue] = []
    positive = tuple(score for score in scores if score.case.relevant_intervals)
    negative = tuple(score for score in scores if not score.case.relevant_intervals)
    hard_negative = tuple(score for score in scores if score.case.hard_negatives)
    if positive:
        metrics.extend(
            (
                MetricValue(
                    name("recall_at_1"),
                    mean(
                        score.relevant_rank is not None and score.relevant_rank <= 1
                        for score in positive
                    ),
                    "ratio",
                ),
                MetricValue(
                    name("recall_at_3"),
                    mean(
                        score.relevant_rank is not None and score.relevant_rank <= 3
                        for score in positive
                    ),
                    "ratio",
                ),
                MetricValue(
                    name("recall_at_5"),
                    mean(
                        score.relevant_rank is not None and score.relevant_rank <= 5
                        for score in positive
                    ),
                    "ratio",
                ),
                MetricValue(
                    name("mrr"),
                    mean(
                        1 / score.relevant_rank if score.relevant_rank else 0.0
                        for score in positive
                    ),
                    "ratio",
                ),
                MetricValue(
                    name("mean_temporal_iou"),
                    mean(score.best_temporal_iou for score in positive),
                    "ratio",
                ),
            )
        )
    if scores:
        result_count = sum(score.result_count for score in scores)
        false_positive_count = sum(score.false_positive_count for score in scores)
        metrics.extend(
            (
                MetricValue(
                    name("false_positive_rate"),
                    false_positive_count / result_count if result_count else 0.0,
                    "ratio",
                ),
                MetricValue(
                    name("mean_latency_ms"),
                    mean(score.latency_ms for score in scores),
                    "milliseconds",
                ),
            )
        )
    if negative:
        metrics.append(
            MetricValue(
                name("negative_case_false_positive_rate"),
                mean(score.negative_false_positive for score in negative),
                "ratio",
            )
        )
    if hard_negative:
        metrics.append(
            MetricValue(
                name("hard_negative_hit_rate"),
                mean(score.hard_negative_hit for score in hard_negative),
                "ratio",
            )
        )
    return tuple(metrics)


def _bounded_metric_name(prefix: str, suffix: str) -> str:
    candidate = f"{prefix}.{suffix}"
    if len(candidate) <= 128:
        return candidate
    digest = sha256(candidate.encode("utf-8")).hexdigest()[:12]
    available = 128 - len(suffix) - len(digest) - 2
    return f"{prefix[:available]}-{digest}.{suffix}"
