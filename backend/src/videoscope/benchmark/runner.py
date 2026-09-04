from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import islice
import math
from statistics import mean
from time import perf_counter
from typing import Callable, Iterable, Literal, Protocol

from videoscope.evaluation import RELEVANCE_IOU_THRESHOLD, temporal_iou
from videoscope.model_manifest import (
    FASTEMBED_ALGORITHM_VERSION,
    FASTEMBED_REPOSITORY,
    FASTEMBED_RUNTIME_VERSION,
    MODEL_REVISIONS,
    QWEN_VIDEO_MODEL,
    SIGLIP_224_MODEL,
    SIGLIP_384_MODEL,
    TEXT_EMBEDDING_DIMENSIONS,
    TEXT_EMBEDDING_MODEL,
    model_identity,
)

from .catalog import AssetResolutionError, LocalAssetResolver, ResolvedAsset
from .measurements import (
    BenchmarkMeasurementFactory,
    BenchmarkMeasurementSession,
    measurement_metrics_from_evidence,
)
from .profiles import (
    PROFILE_IDENTITY_CONTRACT_COMPONENT_ID,
    BenchmarkProfile,
    ProfileIdentityExpectation,
    get_profile,
    profile_identity_contract,
)
from .schema import (
    LEGACY_UNMEASURED_PROTOCOL_IDENTITY,
    MEASUREMENT_PROTOCOL_COMPONENT_ID,
    NOT_MEASURED_PROTOCOL_IDENTITY,
    RUN_SCHEMA_VERSION,
    BenchmarkCaseOutcome,
    BenchmarkDataError,
    BenchmarkDataset,
    BenchmarkMeasurementEvidence,
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
    _run_status_for_outcomes,
)
from .serialization import dataset_revision
from .storage import BenchmarkRunRegistry


BENCHMARK_METHODOLOGY_VERSION = 5
EXECUTION_LIFECYCLE_COMPONENT_ID = "benchmark_execution_lifecycle"
_BENCHMARK_CONFIG_COMPONENT_IDS = frozenset(
    {
        EXECUTION_LIFECYCLE_COMPONENT_ID,
        "benchmark_methodology",
        "benchmark_profile",
        PROFILE_IDENTITY_CONTRACT_COMPONENT_ID,
        "benchmark_search_plan",
    }
)
_COMPLETE_CAPABILITY_STATE = "complete"
_KNOWN_INCOMPLETE_STATES = frozenset(
    {"queued", "running", "cancelled", "failed", "stale", "not_configured"}
)
_SLICE_DIMENSIONS = ("domain", "modality", "label_quality", "split_group")
_CRITICAL_SLICE_DIMENSIONS = (
    "event_class",
    "capture_condition",
    "distribution_shift",
)


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


def _validate_profile_execution_identities(
    identities: ExecutionIdentities,
    *,
    profile: BenchmarkProfile,
    lifecycle: ComponentIdentity,
    execution_mode: Literal["cold", "warm"],
    expected_environment_identity: str | None = None,
    persisted: bool = False,
) -> None:
    """Validate one identity snapshot against the frozen profile contract.

    ``persisted=False`` validates the product-owned identity surface returned by
    a pinned search session. ``persisted=True`` additionally requires the exact
    benchmark-owned audit identities that are added by the runner.
    """

    if not isinstance(identities, ExecutionIdentities):
        raise BenchmarkExecutionError(
            "benchmark profile execution identity contract mismatch"
        )
    if (
        not isinstance(lifecycle, ComponentIdentity)
        or lifecycle.component_id != EXECUTION_LIFECYCLE_COMPONENT_ID
        or execution_mode not in {"cold", "warm"}
        or not lifecycle.identity.startswith(f"{execution_mode}:")
        or not lifecycle.identity.removeprefix(f"{execution_mode}:").strip()
    ):
        raise BenchmarkExecutionError(
            "benchmark profile execution identity contract mismatch"
        )
    contract = profile_identity_contract(profile)
    by_role = {
        "model": {
            item.component_id: item.identity for item in identities.model_identities
        },
        "index": {
            item.component_id: item.identity for item in identities.index_identities
        },
        "config": {
            item.component_id: item.identity for item in identities.config_identities
        },
    }
    expected_by_role = {
        role: set(contract.component_ids(role))
        for role in ("model", "index", "config")
    }
    if persisted:
        expected_by_role["config"].update(_BENCHMARK_CONFIG_COMPONENT_IDS)
    if any(set(by_role[role]) != expected_by_role[role] for role in by_role):
        raise BenchmarkExecutionError(
            "benchmark profile execution identity contract mismatch"
        )
    for expectation in contract.expectations:
        value = by_role[expectation.role][expectation.component_id]
        if not _identity_value_matches_contract(
            expectation,
            value,
            profile=profile,
            lifecycle=lifecycle,
            expected_environment_identity=expected_environment_identity,
        ):
            raise BenchmarkExecutionError(
                "benchmark profile execution identity contract mismatch"
            )
    if not persisted:
        return
    expected_benchmark_values = {
        EXECUTION_LIFECYCLE_COMPONENT_ID: lifecycle.identity,
        "benchmark_methodology": _methodology_identity(),
        "benchmark_profile": profile.identity,
        PROFILE_IDENTITY_CONTRACT_COMPONENT_ID: contract.identity,
        "benchmark_search_plan": profile.search_plan.identity,
    }
    if any(
        by_role["config"].get(component_id) != value
        for component_id, value in expected_benchmark_values.items()
    ):
        raise BenchmarkExecutionError(
            "benchmark profile execution identity contract mismatch"
        )


def _identity_value_matches_contract(
    expectation: ProfileIdentityExpectation,
    value: object,
    *,
    profile: BenchmarkProfile,
    lifecycle: ComponentIdentity,
    expected_environment_identity: str | None,
) -> bool:
    if type(value) is not str or not value or value.strip() != value:
        return False
    value_contract = expectation.value_contract
    if value_contract == "sha256":
        return _is_sha256(value)
    if value_contract == "sha256_prefixed":
        return value.startswith("sha256:") and _is_sha256(
            value.removeprefix("sha256:")
        )
    if value_contract == "fastembed_mpnet_v1":
        return value == (
            f"fastembed@{FASTEMBED_RUNTIME_VERSION}:{FASTEMBED_ALGORITHM_VERSION}:"
            f"{TEXT_EMBEDDING_MODEL}:"
            f"{model_identity(FASTEMBED_REPOSITORY, MODEL_REVISIONS[FASTEMBED_REPOSITORY])}:"
            f"{TEXT_EMBEDDING_DIMENSIONS}"
        )
    if value_contract == "reviewed_siglip_or_not_configured":
        return value == "not-configured" or value in {
            model_identity(model_name, MODEL_REVISIONS[model_name])
            for model_name in (SIGLIP_224_MODEL, SIGLIP_384_MODEL)
        }
    if value_contract == "qwen_verifier_or_not_configured":
        return value == "not-configured" or value == model_identity(
            QWEN_VIDEO_MODEL,
            MODEL_REVISIONS[QWEN_VIDEO_MODEL],
        )
    if value_contract == "internvideo_not_configured":
        return value == "not-configured"
    if value_contract == "provider_digest_or_not_configured":
        return value == "not-configured" or (
            value.startswith("sha256:")
            and _is_sha256(value.removeprefix("sha256:"))
        )
    if value_contract == "benchmark_environment_v2":
        if not value.startswith("benchmark-product-environment@2:"):
            return False
        if not _is_sha256(value.removeprefix("benchmark-product-environment@2:")):
            return False
        return expected_environment_identity is None or (
            value == expected_environment_identity
        )
    if value_contract == "evaluation_search_configuration":
        return value == profile.search_plan.identity.replace(
            "evaluation-search-plan",
            "evaluation-search-configuration",
            1,
        )
    if value_contract == "lifecycle":
        return value == lifecycle.identity
    return False


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class BenchmarkSearchSession(Protocol):
    """One immutable execution snapshot pinned for an entire benchmark run."""

    def identities(self) -> ExecutionIdentities: ...

    def lifecycle_identity(self) -> ComponentIdentity: ...

    def capability_state(
        self,
        asset: ResolvedAsset,
        capability: str,
    ) -> str: ...

    def search(
        self,
        query: str,
        assets: tuple[ResolvedAsset, ...],
        *,
        limit: int,
    ) -> Iterable[BenchmarkSearchHit]: ...

    def close(self) -> None: ...


class BenchmarkSearchAdapter(Protocol):
    """Factory boundary between the portable runner and product search.

    A concrete adapter is responsible for mapping ``ResolvedAsset.video_id``
    to the current search service and translating results back to portable
    asset aliases.  ``open_session`` must pin model/index/config identities and
    capabilities for the complete run; it must not silently follow mutable
    active pointers between cases.
    """

    def open_session(
        self,
        profile: BenchmarkProfile,
        assets: tuple[ResolvedAsset, ...],
        *,
        execution_mode: Literal["cold", "warm"],
    ) -> BenchmarkSearchSession: ...

    def close(self) -> None: ...


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
    relevant_interval_count: int
    covered_relevant_at_10: int
    covered_relevant_at_20: int
    covered_relevant_at_50: int
    precision_at_5: float
    ndcg_at_10: float
    boundary_errors: tuple[tuple[float, float], ...]
    hits: tuple[BenchmarkSearchHit, ...]


class _CaseFailure(RuntimeError):
    def __init__(self, code: str, *, latency_ms: float = 0.0) -> None:
        self.code = code
        self.latency_ms = latency_ms
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
        measurement: BenchmarkMeasurementFactory | None = None,
        clock: Callable[[], datetime] | None = None,
        timer: Callable[[], float] = perf_counter,
        overlap: Callable[[float, float, float, float], float] = temporal_iou,
    ) -> None:
        self.registry = registry
        self._asset_resolver = asset_resolver
        self._search = search
        self._hardware = hardware
        self._code_sha = code_sha
        self._measurement = measurement
        self._clock = clock or (lambda: datetime.now(UTC))
        self._timer = timer
        self._overlap = overlap
        self._prepared_runs: dict[str, BenchmarkRunManifest] = {}

    def close(self) -> None:
        """Retry cleanup of adapter-owned resources after a failed run."""
        self._close_adapter(suppress_error=False)

    def run(
        self,
        dataset: BenchmarkDataset,
        *,
        profile_id: str,
        run_id: str,
        execution_mode: Literal["cold", "warm"],
        publish: bool = True,
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
        if type(publish) is not bool:
            raise BenchmarkDataError("publish must be a boolean")
        profile = get_profile(profile_id)
        if run_id in self._prepared_runs or any(
            entry.run_id == run_id for entry in self.registry.list()
        ):
            raise FileExistsError(f"benchmark run {run_id!r} already exists")
        started_at = self._timestamp()
        measurement_protocol = ComponentIdentity(
            MEASUREMENT_PROTOCOL_COMPONENT_ID,
            NOT_MEASURED_PROTOCOL_IDENTITY,
        )
        measurement_status: Literal["not_measured", "complete", "failed"] = (
            "not_measured"
        )
        measurement_started_at: str | None = None
        measurement_finished_at: str | None = None
        system_metrics: tuple[MetricValue, ...] = ()
        measurement_evidence_status: Literal[
            "not_applicable", "complete", "legacy_unavailable"
        ] = "not_applicable"
        measurement_evidence: BenchmarkMeasurementEvidence | None = None
        measurement_session: BenchmarkMeasurementSession | None = None
        measurement_active = False
        if self._measurement is not None:
            measurement_protocol = self._measurement_protocol(execution_mode)
            measurement_started_at = self._timestamp()
            try:
                measurement_session = self._open_measurement_session()
                measurement_session.start()
                measurement_active = True
            except Exception:
                self._close_measurement_session(
                    measurement_session,
                    suppress_error=True,
                )
                measurement_status = "failed"
                measurement_finished_at = self._timestamp()

        session: BenchmarkSearchSession | None = None
        search_closed = False
        try:
            resolved, resolution_failures = self._resolve_assets(dataset)
            session = self._open_session(
                profile,
                tuple(resolved[key] for key in sorted(resolved)),
                execution_mode=execution_mode,
            )
            identities = self._execution_identities(
                session,
                profile,
                execution_mode=execution_mode,
            )
            capability_failures = self._preflight_capabilities(
                session,
                profile,
                resolved,
            )
            outcomes, scores = self._execute_cases(
                session,
                dataset,
                profile,
                resolved,
                resolution_failures,
                capability_failures,
            )
            ordered_outcomes = tuple(
                sorted(outcomes, key=lambda item: item.case_id)
            )
            metrics = self._run_metrics(dataset, ordered_outcomes, tuple(scores))
            if measurement_active:
                try:
                    if measurement_session is None:
                        raise BenchmarkExecutionError(
                            "benchmark measurement session is unavailable"
                        )
                    candidate_evidence = measurement_session.finish()
                    if not isinstance(
                        candidate_evidence,
                        BenchmarkMeasurementEvidence,
                    ):
                        raise BenchmarkExecutionError(
                            "benchmark measurement did not return portable raw evidence"
                        )
                    candidate_metrics = measurement_metrics_from_evidence(
                        candidate_evidence
                    )
                    system_metrics = self._validate_system_metrics(
                        candidate_metrics,
                        quality_metrics=metrics,
                    )
                except Exception:
                    system_metrics = ()
                    measurement_status = "failed"
                else:
                    measurement_status = "complete"
                    measurement_evidence_status = "complete"
                    measurement_evidence = candidate_evidence
                measurement_finished_at = self._timestamp()
            self._close_session(session, suppress_error=False)
            search_closed = True
            self._close_adapter(suppress_error=False)
        except BaseException as execution_error:
            if session is not None and not search_closed:
                self._close_session(session, suppress_error=True)
            if measurement_active:
                self._close_measurement_session(
                    measurement_session,
                    suppress_error=True,
                )
            try:
                self._close_adapter(suppress_error=False)
            except BaseException as cleanup_error:
                if isinstance(execution_error, (KeyboardInterrupt, SystemExit)):
                    raise execution_error from cleanup_error
                if isinstance(execution_error, Exception) and isinstance(
                    cleanup_error,
                    Exception,
                ):
                    raise BenchmarkExecutionError(
                        "benchmark execution and search adapter cleanup failed; "
                        "resources could not be closed"
                    ) from ExceptionGroup(
                        "benchmark execution and search adapter cleanup failed",
                        [execution_error, cleanup_error],
                    )
                raise cleanup_error from execution_error
            raise

        if measurement_active:
            try:
                self._close_measurement_session(
                    measurement_session,
                    suppress_error=False,
                )
            except Exception:
                system_metrics = ()
                measurement_status = "failed"
                measurement_evidence_status = "not_applicable"
                measurement_evidence = None
        finished_at = self._timestamp()
        run = BenchmarkRunManifest(
            schema_version=RUN_SCHEMA_VERSION,
            run_id=run_id,
            created_at=finished_at,
            started_at=started_at,
            finished_at=finished_at,
            run_status=_run_status_for_outcomes(ordered_outcomes),
            code_sha=self._code_sha,
            dataset_revision=dataset_revision(dataset),
            model_identities=identities.model_identities,
            index_identities=identities.index_identities,
            config_identities=identities.config_identities,
            hardware=self._hardware,
            execution_mode=execution_mode,
            quality_metrics=metrics,
            system_metrics=system_metrics,
            measurement_protocol=measurement_protocol,
            measurement_status=measurement_status,
            measurement_started_at=measurement_started_at,
            measurement_finished_at=measurement_finished_at,
            case_outcomes=ordered_outcomes,
            measurement_evidence_status=measurement_evidence_status,
            measurement_evidence=measurement_evidence,
        )
        _validate_persisted_profile_identity_contract(run)
        if publish:
            self.registry.add(run)
        else:
            self._prepared_runs[run.run_id] = run
        return run

    def publish_prepared_run(self, run: BenchmarkRunManifest) -> None:
        """Commit one exact prepared run after its external owner is closed."""
        if not isinstance(run, BenchmarkRunManifest):
            raise BenchmarkExecutionError("benchmark run is not prepared")
        prepared = self._prepared_runs.get(run.run_id)
        if prepared is None or prepared != run:
            raise BenchmarkExecutionError("benchmark run is not prepared")
        if any(entry.run_id == run.run_id for entry in self.registry.list()):
            raise FileExistsError(f"benchmark run {run.run_id!r} already exists")
        _validate_persisted_profile_identity_contract(run)
        self.registry.add(run)
        del self._prepared_runs[run.run_id]

    def _open_session(
        self,
        profile: BenchmarkProfile,
        assets: tuple[ResolvedAsset, ...],
        *,
        execution_mode: Literal["cold", "warm"],
    ) -> BenchmarkSearchSession:
        try:
            session = self._search.open_session(
                profile,
                assets,
                execution_mode=execution_mode,
            )
        except Exception:
            raise BenchmarkExecutionError(
                "pinned benchmark search session is unavailable"
            ) from None
        methods = (
            "identities",
            "lifecycle_identity",
            "capability_state",
            "search",
            "close",
        )
        if any(not callable(getattr(session, name, None)) for name in methods):
            close = getattr(session, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            raise BenchmarkExecutionError(
                "pinned benchmark search session has an invalid contract"
            )
        return session

    @staticmethod
    def _close_session(
        session: BenchmarkSearchSession,
        *,
        suppress_error: bool,
    ) -> None:
        try:
            session.close()
        except Exception:
            if not suppress_error:
                raise BenchmarkExecutionError(
                    "pinned benchmark search session could not be closed"
                ) from None

    def _close_adapter(self, *, suppress_error: bool) -> None:
        close = getattr(self._search, "close", None)
        if not callable(close):
            if suppress_error:
                return
            raise BenchmarkExecutionError(
                "benchmark search adapter has an invalid cleanup contract"
            )
        last_error: Exception | None = None
        for _attempt in range(3):
            try:
                close()
            except Exception as error:
                last_error = error
                continue
            return
        if not suppress_error:
            assert last_error is not None
            raise BenchmarkExecutionError(
                "benchmark search adapter could not be closed"
            ) from last_error

    def _execution_identities(
        self,
        session: BenchmarkSearchSession,
        profile: BenchmarkProfile,
        *,
        execution_mode: Literal["cold", "warm"],
    ) -> ExecutionIdentities:
        try:
            identities = session.identities()
        except Exception:
            raise BenchmarkExecutionError(
                "stable benchmark execution identity is unavailable"
            ) from None
        if not isinstance(identities, ExecutionIdentities):
            raise BenchmarkExecutionError(
                "stable benchmark execution identity has an invalid contract"
            )
        reserved = _BENCHMARK_CONFIG_COMPONENT_IDS
        if reserved & {
            identity.component_id for identity in identities.config_identities
        }:
            raise BenchmarkExecutionError(
                "execution identity uses a reserved benchmark component id"
            )
        lifecycle = self._execution_lifecycle_identity(
            session,
            execution_mode=execution_mode,
        )
        _validate_profile_execution_identities(
            identities,
            profile=profile,
            lifecycle=lifecycle,
            execution_mode=execution_mode,
        )
        measurement = self._measurement
        if measurement is not None and lifecycle.identity != (
            f"{execution_mode}:{measurement.cache_policy_identity}"
        ):
            raise BenchmarkExecutionError(
                "benchmark measurement and execution lifecycle policies disagree"
            )
        config_identities = (
            *identities.config_identities,
            lifecycle,
            ComponentIdentity("benchmark_profile", profile.identity),
            ComponentIdentity(
                "benchmark_search_plan",
                profile.search_plan.identity,
            ),
            ComponentIdentity(
                "benchmark_methodology",
                _methodology_identity(),
            ),
            ComponentIdentity(
                PROFILE_IDENTITY_CONTRACT_COMPONENT_ID,
                profile_identity_contract(profile).identity,
            ),
        )
        complete = ExecutionIdentities(
            model_identities=_ordered_identities(identities.model_identities),
            index_identities=_ordered_identities(identities.index_identities),
            config_identities=_ordered_identities(config_identities),
        )
        _validate_profile_execution_identities(
            complete,
            profile=profile,
            lifecycle=lifecycle,
            execution_mode=execution_mode,
            persisted=True,
        )
        return complete

    @staticmethod
    def _execution_lifecycle_identity(
        session: BenchmarkSearchSession,
        *,
        execution_mode: Literal["cold", "warm"],
    ) -> ComponentIdentity:
        try:
            identity = session.lifecycle_identity()
        except Exception:
            raise BenchmarkExecutionError(
                "benchmark execution lifecycle identity is unavailable"
            ) from None
        if (
            not isinstance(identity, ComponentIdentity)
            or identity.component_id != EXECUTION_LIFECYCLE_COMPONENT_ID
            or not identity.identity.startswith(f"{execution_mode}:")
            or not identity.identity.removeprefix(f"{execution_mode}:").strip()
        ):
            raise BenchmarkExecutionError(
                "benchmark execution lifecycle identity does not attest the run mode"
            )
        return identity

    def _measurement_protocol(
        self,
        execution_mode: Literal["cold", "warm"],
    ) -> ComponentIdentity:
        factory = self._measurement
        if factory is None:
            raise BenchmarkExecutionError("benchmark measurement factory is unavailable")
        if getattr(factory, "execution_mode", None) != execution_mode:
            raise BenchmarkExecutionError(
                "benchmark measurement lifecycle does not match execution_mode"
            )
        try:
            protocol = factory.protocol_identity()
        except Exception:
            raise BenchmarkExecutionError(
                "benchmark measurement protocol identity is unavailable"
            ) from None
        if (
            not isinstance(protocol, ComponentIdentity)
            or protocol.component_id != MEASUREMENT_PROTOCOL_COMPONENT_ID
            or protocol.identity
            in {
                NOT_MEASURED_PROTOCOL_IDENTITY,
                LEGACY_UNMEASURED_PROTOCOL_IDENTITY,
            }
        ):
            raise BenchmarkExecutionError(
                "benchmark measurement protocol identity is invalid"
            )
        return protocol

    def _open_measurement_session(self) -> BenchmarkMeasurementSession:
        factory = self._measurement
        assert factory is not None
        session = factory.open_session()
        methods = ("start", "finish", "close")
        if any(not callable(getattr(session, name, None)) for name in methods):
            self._close_measurement_session(session, suppress_error=True)
            raise BenchmarkExecutionError(
                "benchmark measurement session has an invalid contract"
            )
        return session

    @staticmethod
    def _close_measurement_session(
        session: BenchmarkMeasurementSession | object | None,
        *,
        suppress_error: bool,
    ) -> None:
        if session is None:
            return
        close = getattr(session, "close", None)
        if not callable(close):
            if not suppress_error:
                raise BenchmarkExecutionError(
                    "benchmark measurement session could not be closed"
                )
            return
        try:
            close()
        except Exception:
            if not suppress_error:
                raise BenchmarkExecutionError(
                    "benchmark measurement session could not be closed"
                ) from None

    @staticmethod
    def _validate_system_metrics(
        values: object,
        *,
        quality_metrics: tuple[MetricValue, ...],
    ) -> tuple[MetricValue, ...]:
        if not isinstance(values, tuple) or not values:
            raise BenchmarkExecutionError(
                "benchmark measurement did not return complete system metrics"
            )
        if any(not isinstance(value, MetricValue) for value in values):
            raise BenchmarkExecutionError(
                "benchmark measurement returned an invalid system metric"
            )
        names = tuple(value.name for value in values)
        quality_names = {value.name for value in quality_metrics}
        if len(names) != len(set(names)) or set(names) & quality_names:
            raise BenchmarkExecutionError(
                "benchmark measurement metric names are invalid"
            )
        return tuple(sorted(values, key=lambda item: item.name))

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

    def _preflight_capabilities(
        self,
        session: BenchmarkSearchSession,
        profile: BenchmarkProfile,
        assets: dict[str, ResolvedAsset],
    ) -> dict[tuple[str, str], str]:
        failures: dict[tuple[str, str], str] = {}
        for asset_id in sorted(assets):
            asset = assets[asset_id]
            for capability in profile.required_capabilities:
                try:
                    state = session.capability_state(
                        asset,
                        capability,
                    )
                except Exception:
                    failures[(asset.asset_id, capability)] = (
                        "capability_check_failed"
                    )
                    continue
                if state == _COMPLETE_CAPABILITY_STATE:
                    continue
                if state == "stale":
                    diagnostic = "capability_stale"
                elif state == "missing":
                    diagnostic = "capability_missing"
                elif state == "failed":
                    diagnostic = "capability_failed"
                elif state == "not_configured":
                    diagnostic = "capability_not_configured"
                elif isinstance(state, str) and state in _KNOWN_INCOMPLETE_STATES:
                    diagnostic = "capability_incomplete"
                else:
                    diagnostic = "capability_invalid_state"
                failures[(asset.asset_id, capability)] = diagnostic
        return failures

    @staticmethod
    def _require_capabilities(
        profile: BenchmarkProfile,
        assets: tuple[ResolvedAsset, ...],
        failures: dict[tuple[str, str], str],
    ) -> None:
        for asset in assets:
            for capability in profile.required_capabilities:
                diagnostic = failures.get((asset.asset_id, capability))
                if diagnostic is not None:
                    raise _CaseFailure(diagnostic)

    def _execute_cases(
        self,
        session: BenchmarkSearchSession,
        dataset: BenchmarkDataset,
        profile: BenchmarkProfile,
        resolved: dict[str, ResolvedAsset],
        resolution_failures: dict[str, str],
        capability_failures: dict[tuple[str, str], str],
    ) -> tuple[list[BenchmarkCaseOutcome], list[_CaseScore]]:
        outcomes: list[BenchmarkCaseOutcome] = []
        scores: list[_CaseScore] = []
        for case in sorted(dataset.cases, key=lambda item: item.case_id):
            score: _CaseScore | None = None
            diagnostic_code: str | None = None
            latency_ms = 0.0
            try:
                case_assets = self._assets_for_case(
                    case,
                    resolved,
                    resolution_failures,
                )
                self._require_capabilities(
                    profile,
                    case_assets,
                    capability_failures,
                )
                hits, latency_ms = self._search_case(
                    session,
                    profile,
                    case,
                    case_assets,
                )
                self._validate_hits(profile, case, case_assets, hits)
                score = self._score_case(case, hits, latency_ms=latency_ms)
            except _CaseFailure as exc:
                diagnostic_code = exc.code
                latency_ms = max(latency_ms, exc.latency_ms)
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
            scores.append(score)
            outcomes.append(self._complete_outcome(score))
        return outcomes, scores

    def _search_case(
        self,
        session: BenchmarkSearchSession,
        profile: BenchmarkProfile,
        case: QueryCase,
        assets: tuple[ResolvedAsset, ...],
    ) -> tuple[tuple[BenchmarkSearchHit, ...], float]:
        started = self._read_timer()
        try:
            value = session.search(
                case.query,
                assets,
                limit=profile.result_limit,
            )
            hits = tuple(islice(iter(value), profile.result_limit + 1))
        except Exception:
            raise _CaseFailure(
                "provider_failed",
                latency_ms=self._elapsed_ms(started),
            ) from None
        return hits, self._elapsed_ms(started)

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
                    MetricValue(
                        "precision_at_5",
                        score.precision_at_5,
                        "ratio",
                    ),
                    MetricValue(
                        "recall_at_10",
                        score.covered_relevant_at_10
                        / score.relevant_interval_count,
                        "ratio",
                    ),
                    MetricValue(
                        "recall_at_20",
                        score.covered_relevant_at_20
                        / score.relevant_interval_count,
                        "ratio",
                    ),
                    MetricValue(
                        "candidate_recall_at_50",
                        score.covered_relevant_at_50
                        / score.relevant_interval_count,
                        "ratio",
                    ),
                    MetricValue(
                        "ndcg_at_10",
                        score.ndcg_at_10,
                        "ratio",
                    ),
                )
            )
            if score.boundary_errors:
                starts = tuple(value[0] for value in score.boundary_errors)
                ends = tuple(value[1] for value in score.boundary_errors)
                case_metrics.extend(
                    (
                        MetricValue(
                            "mean_start_boundary_error_seconds",
                            mean(starts),
                            "seconds",
                        ),
                        MetricValue(
                            "mean_end_boundary_error_seconds",
                            mean(ends),
                            "seconds",
                        ),
                        MetricValue(
                            "mean_boundary_error_seconds",
                            mean((*starts, *ends)),
                            "seconds",
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
            MetricValue(
                "infrastructure_failure_count",
                error_count,
                "count",
            ),
            MetricValue(
                "infrastructure_failure_rate",
                error_count / total_count if total_count else 0.0,
                "ratio",
            ),
        ]
        failure_counts: dict[str, int] = {}
        for outcome in outcomes:
            if outcome.status == "complete":
                continue
            diagnostic = outcome.diagnostic_code or "execution_failed"
            failure_counts[diagnostic] = failure_counts.get(diagnostic, 0) + 1
        for diagnostic, count in sorted(failure_counts.items()):
            metrics.extend(
                (
                    MetricValue(f"failure.{diagnostic}.count", count, "count"),
                    MetricValue(
                        f"failure.{diagnostic}.rate",
                        count / total_count if total_count else 0.0,
                        "ratio",
                    ),
                )
            )
        completed_positive = tuple(
            score for score in scores if score.case.relevant_intervals
        )
        model_miss_count = sum(
            score.relevant_rank is None for score in completed_positive
        )
        metrics.extend(
            (
                MetricValue("model_miss_count", model_miss_count, "count"),
                MetricValue(
                    "model_miss_rate",
                    model_miss_count / len(completed_positive)
                    if completed_positive
                    else 0.0,
                    "ratio",
                ),
            )
        )
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
            dimensions = _SLICE_DIMENSIONS
            if case.critical_slices is not None:
                values.update(
                    {
                        "event_class": case.critical_slices.event_class,
                        "capture_condition": case.critical_slices.capture_condition,
                        "distribution_shift": case.critical_slices.distribution_shift,
                    }
                )
                dimensions += _CRITICAL_SLICE_DIMENSIONS
            for dimension in dimensions:
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

    def _timestamp(self) -> str:
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


def _validate_persisted_profile_identity_contract(
    run: BenchmarkRunManifest,
) -> BenchmarkProfile:
    if not isinstance(run, BenchmarkRunManifest):
        raise BenchmarkExecutionError(
            "benchmark profile execution identity contract mismatch"
        )
    config = {
        identity.component_id: identity.identity
        for identity in run.config_identities
    }
    profile_identity = config.get("benchmark_profile")
    if type(profile_identity) is not str:
        raise BenchmarkExecutionError(
            "benchmark profile execution identity contract mismatch"
        )
    profile_id, separator, _version = profile_identity.rpartition("@")
    if not separator:
        raise BenchmarkExecutionError(
            "benchmark profile execution identity contract mismatch"
        )
    try:
        profile = get_profile(profile_id)
    except BenchmarkDataError:
        raise BenchmarkExecutionError(
            "profile identity mismatch"
        ) from None
    if profile.identity != profile_identity:
        raise BenchmarkExecutionError("profile identity mismatch")
    if config.get("benchmark_methodology") != _methodology_identity():
        raise BenchmarkExecutionError("methodology identity mismatch")
    if config.get("benchmark_search_plan") != profile.search_plan.identity:
        raise BenchmarkExecutionError("search plan identity mismatch")
    lifecycle_value = config.get(EXECUTION_LIFECYCLE_COMPONENT_ID)
    if type(lifecycle_value) is not str:
        raise BenchmarkExecutionError("execution lifecycle identity mismatch")
    lifecycle = ComponentIdentity(
        EXECUTION_LIFECYCLE_COMPONENT_ID,
        lifecycle_value,
    )
    _validate_profile_execution_identities(
        ExecutionIdentities(
            model_identities=run.model_identities,
            index_identities=run.index_identities,
            config_identities=run.config_identities,
        ),
        profile=profile,
        lifecycle=lifecycle,
        execution_mode=run.execution_mode,
        persisted=True,
    )
    return profile


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
    try:
        profile = _validate_persisted_profile_identity_contract(run)
    except BenchmarkExecutionError as error:
        raise BenchmarkExecutionError(
            f"benchmark run audit failed: {error}"
        ) from error

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
    if expected_metrics != run.quality_metrics:
        raise BenchmarkExecutionError(
            "benchmark run audit failed: aggregate metrics mismatch"
        )
    if run.measurement_status == "complete":
        if (
            run.measurement_evidence_status != "complete"
            or run.measurement_evidence is None
        ):
            raise BenchmarkExecutionError(
                "benchmark run audit failed: raw measurement evidence is unavailable"
            )
        try:
            expected_system_metrics = measurement_metrics_from_evidence(
                run.measurement_evidence
            )
        except BenchmarkDataError:
            raise BenchmarkExecutionError(
                "benchmark run audit failed: invalid raw measurement evidence"
            ) from None
        if expected_system_metrics != run.system_metrics:
            raise BenchmarkExecutionError(
                "benchmark run audit failed: system metrics mismatch"
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
    hard_negative_hit = 0
    unique_relevance_gains: list[int] = []
    covered_relevant_at_10 = 0
    covered_relevant_at_20 = 0
    covered_relevant_at_50 = 0
    hit_matches: list[tuple[tuple[int, float], ...]] = []
    gold_to_hit: dict[int, int] = {}

    def augment(hit_index: int, visited_gold: set[int]) -> bool:
        for gold_index, _iou in hit_matches[hit_index]:
            if gold_index in visited_gold:
                continue
            visited_gold.add(gold_index)
            previous_hit = gold_to_hit.get(gold_index)
            if previous_hit is None or augment(previous_hit, visited_gold):
                gold_to_hit[gold_index] = hit_index
                return True
        return False

    for hit_index, hit in enumerate(hits):
        rank = hit_index + 1
        overlaps = {
            index: _checked_overlap(
                overlap,
                interval.start_seconds,
                interval.end_seconds,
                hit.start_seconds,
                hit.end_seconds,
            )
            for index, interval in enumerate(case.relevant_intervals)
            if interval.asset_id == hit.asset_id
        }
        relevant_overlap = max(overlaps.values(), default=0.0)
        best_iou = max(best_iou, relevant_overlap)
        hit_matches.append(
            tuple(
                sorted(
                    (
                        (index, value)
                        for index, value in overlaps.items()
                        if value >= RELEVANCE_IOU_THRESHOLD
                    ),
                    key=lambda item: (-item[1], item[0]),
                )
            )
        )
        gained = int(augment(hit_index, set()))
        unique_relevance_gains.append(gained)
        if gained and relevant_rank is None:
            relevant_rank = rank
        if rank <= 10:
            covered_relevant_at_10 = len(gold_to_hit)
        if rank <= 20:
            covered_relevant_at_20 = len(gold_to_hit)
        if rank <= 50:
            covered_relevant_at_50 = len(gold_to_hit)
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

    boundary_errors: list[tuple[float, float]] = []
    for gold_index, hit_index in sorted(gold_to_hit.items()):
        interval = case.relevant_intervals[gold_index]
        selected_hit = hits[hit_index]
        boundary_errors.append(
            (
                abs(selected_hit.start_seconds - interval.start_seconds),
                abs(selected_hit.end_seconds - interval.end_seconds),
            )
        )
    dcg = sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(unique_relevance_gains[:10], start=1)
    )
    ideal_count = min(len(case.relevant_intervals), 10)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return _CaseScore(
        case=case,
        latency_ms=latency_ms,
        result_count=len(hits),
        relevant_rank=relevant_rank,
        best_temporal_iou=best_iou,
        false_positive_count=len(hits) - sum(unique_relevance_gains),
        negative_false_positive=(
            int(bool(hits)) if not case.relevant_intervals else 0
        ),
        hard_negative_hit=hard_negative_hit,
        relevant_interval_count=len(case.relevant_intervals),
        covered_relevant_at_10=covered_relevant_at_10,
        covered_relevant_at_20=covered_relevant_at_20,
        covered_relevant_at_50=covered_relevant_at_50,
        precision_at_5=sum(unique_relevance_gains[:5]) / 5,
        ndcg_at_10=dcg / ideal_dcg if ideal_dcg else 0.0,
        boundary_errors=tuple(boundary_errors),
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
        "quality-denominator=completed-positive-cases;"
        "matching=rank-prefix-maximum-cardinality-one-to-one;"
        "candidate-recall=one-to-one-gold-interval-coverage-at-50;"
        "precision=one-to-one-prefix-gains-over-five;"
        "ndcg=one-to-one-prefix-gains-at-10;"
        "latency-p95=nearest-rank"
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
        relevant_interval_count = sum(
            score.relevant_interval_count for score in positive
        )
        covered_at_10 = sum(score.covered_relevant_at_10 for score in positive)
        covered_at_20 = sum(score.covered_relevant_at_20 for score in positive)
        covered_at_50 = sum(score.covered_relevant_at_50 for score in positive)
        boundary_errors = tuple(
            error for score in positive for error in score.boundary_errors
        )
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
                MetricValue(
                    name("precision_at_5"),
                    mean(score.precision_at_5 for score in positive),
                    "ratio",
                ),
                MetricValue(
                    name("recall_at_10"),
                    covered_at_10 / relevant_interval_count,
                    "ratio",
                ),
                MetricValue(
                    name("recall_at_20"),
                    covered_at_20 / relevant_interval_count,
                    "ratio",
                ),
                MetricValue(
                    name("candidate_recall_at_50"),
                    covered_at_50 / relevant_interval_count,
                    "ratio",
                ),
                MetricValue(
                    name("ndcg_at_10"),
                    mean(score.ndcg_at_10 for score in positive),
                    "ratio",
                ),
            )
        )
        if boundary_errors:
            starts = tuple(value[0] for value in boundary_errors)
            ends = tuple(value[1] for value in boundary_errors)
            metrics.extend(
                (
                    MetricValue(
                        name("mean_start_boundary_error_seconds"),
                        mean(starts),
                        "seconds",
                    ),
                    MetricValue(
                        name("mean_end_boundary_error_seconds"),
                        mean(ends),
                        "seconds",
                    ),
                    MetricValue(
                        name("mean_boundary_error_seconds"),
                        mean((*starts, *ends)),
                        "seconds",
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
                MetricValue(
                    name("p50_latency_ms"),
                    _nearest_rank_percentile(
                        tuple(score.latency_ms for score in scores),
                        0.50,
                    ),
                    "milliseconds",
                ),
                MetricValue(
                    name("p95_latency_ms"),
                    _nearest_rank_percentile(
                        tuple(score.latency_ms for score in scores),
                        0.95,
                    ),
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


def _nearest_rank_percentile(values: tuple[float, ...], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = tuple(sorted(values))
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _bounded_metric_name(prefix: str, suffix: str) -> str:
    candidate = f"{prefix}.{suffix}"
    if len(candidate) <= 128:
        return candidate
    digest = sha256(candidate.encode("utf-8")).hexdigest()[:12]
    available = 128 - len(suffix) - len(digest) - 2
    return f"{prefix[:available]}-{digest}.{suffix}"
