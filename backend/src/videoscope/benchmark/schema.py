from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from ipaddress import ip_address
import math
import re
from typing import Literal
from urllib.parse import urlsplit


DATASET_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
MAX_RESULT_EVIDENCE_PER_CASE = 100

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CODE_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_MAX_TEXT_LENGTH = 10_000
_PORTABLE_URI_SCHEMES = frozenset({"http", "https", "urn"})
_NONPUBLIC_HOST_SUFFIXES = (".internal", ".lan", ".local")
_LOCAL_PATH_RE = re.compile(r"^(?:/|~[/\\]|[A-Za-z]:[/\\]|\\\\)")


class BenchmarkDataError(ValueError):
    """A benchmark manifest does not satisfy the portable data contract."""


def _require_string(
    value: object,
    field: str,
    *,
    max_length: int = _MAX_TEXT_LENGTH,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise BenchmarkDataError(f"{field} must be a string")
    if len(value) > max_length:
        raise BenchmarkDataError(f"{field} exceeds {max_length} characters")
    if not allow_empty and not value.strip():
        raise BenchmarkDataError(f"{field} must not be empty")
    if "\x00" in value:
        raise BenchmarkDataError(f"{field} must not contain NUL characters")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise BenchmarkDataError(f"{field} contains invalid Unicode text") from exc
    return value


def _require_id(value: object, field: str) -> str:
    value = _require_string(value, field, max_length=128)
    if not _SAFE_ID_RE.fullmatch(value):
        raise BenchmarkDataError(
            f"{field} must be a portable identifier containing only letters, "
            "digits, dot, underscore or hyphen"
        )
    return value


def _require_portable_uri(value: object, field: str) -> str:
    uri = _require_string(value, field, max_length=2_048)
    if any(character.isspace() or ord(character) < 32 for character in uri):
        raise BenchmarkDataError(f"{field} must not contain whitespace or control characters")
    if "\\" in uri:
        raise BenchmarkDataError(f"{field} must use URI separators, not backslashes")
    try:
        parsed = urlsplit(uri)
    except ValueError as exc:
        raise BenchmarkDataError(f"{field} is not a valid URI") from exc
    if parsed.scheme not in _PORTABLE_URI_SCHEMES:
        raise BenchmarkDataError(
            f"{field} must use http, https or urn; local file paths are not portable"
        )
    if parsed.query or parsed.fragment:
        raise BenchmarkDataError(
            f"{field} must not contain a query or fragment that could expose credentials"
        )
    if parsed.scheme == "urn":
        if parsed.netloc or not parsed.path or ":" not in parsed.path:
            raise BenchmarkDataError(f"{field} must be a valid URN")
        return uri
    if parsed.username is not None or parsed.password is not None:
        raise BenchmarkDataError(f"{field} must not contain embedded credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise BenchmarkDataError(f"{field} contains an invalid port") from exc
    try:
        hostname = parsed.hostname
    except ValueError as exc:
        raise BenchmarkDataError(f"{field} contains an invalid host") from exc
    if not hostname:
        raise BenchmarkDataError(f"{field} must contain a host")
    normalized_host = hostname.rstrip(".").lower()
    try:
        address = ip_address(normalized_host)
    except ValueError:
        if (
            normalized_host == "localhost"
            or normalized_host.endswith(_NONPUBLIC_HOST_SUFFIXES)
            or "." not in normalized_host
        ):
            raise BenchmarkDataError(
                f"{field} must not expose a local or internal host"
            )
    else:
        if not address.is_global:
            raise BenchmarkDataError(
                f"{field} must not expose a local or non-public IP address"
            )
    if port is not None and not 1 <= port <= 65_535:
        raise BenchmarkDataError(f"{field} contains an invalid port")
    return uri


def _require_portable_source(value: object) -> str:
    source = _require_string(value, "provenance.source")
    candidate = source.strip()
    if candidate.lower().startswith("file:") or _LOCAL_PATH_RE.match(candidate):
        raise BenchmarkDataError(
            "provenance.source must describe the source, not expose a local file path"
        )
    return source


def _require_tuple(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise BenchmarkDataError(f"{field} must be an immutable tuple")
    return value


def _require_unique(values: tuple[object, ...], field: str) -> None:
    try:
        if len(values) != len(set(values)):
            raise BenchmarkDataError(f"{field} must contain unique values")
    except TypeError as exc:
        raise BenchmarkDataError(f"{field} contains an invalid value") from exc


def _require_positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise BenchmarkDataError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise BenchmarkDataError(f"{field} must be a non-negative integer")
    return value


def _require_finite_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkDataError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise BenchmarkDataError(f"{field} must be finite")
    if strictly_positive and result <= 0:
        raise BenchmarkDataError(f"{field} must be greater than zero")
    if minimum is not None and result < minimum:
        raise BenchmarkDataError(f"{field} must be at least {minimum}")
    return 0.0 if result == 0 else result


def _validate_typed_tuple(
    values: object,
    expected_type: type,
    field: str,
) -> tuple[object, ...]:
    result = _require_tuple(values, field)
    if any(not isinstance(item, expected_type) for item in result):
        raise BenchmarkDataError(
            f"{field} must contain only {expected_type.__name__} values"
        )
    return result


@dataclass(frozen=True, slots=True)
class AssetProvenance:
    source: str
    license_id: str
    source_uri: str | None = None
    license_uri: str | None = None
    attribution: str | None = None

    def __post_init__(self) -> None:
        _require_portable_source(self.source)
        _require_string(self.license_id, "provenance.license_id", max_length=256)
        for field_name in ("source_uri", "license_uri", "attribution"):
            value = getattr(self, field_name)
            if value is not None:
                if field_name.endswith("_uri"):
                    _require_portable_uri(value, f"provenance.{field_name}")
                else:
                    _require_string(value, f"provenance.{field_name}", max_length=2_048)


@dataclass(frozen=True, slots=True)
class BenchmarkAsset:
    asset_id: str
    sha256: str
    byte_size: int
    duration_seconds: float
    provenance: AssetProvenance

    def __post_init__(self) -> None:
        _require_id(self.asset_id, "asset_id")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise BenchmarkDataError("sha256 must be a lowercase 64-character SHA-256")
        _require_positive_int(self.byte_size, "byte_size")
        duration = _require_finite_float(
            self.duration_seconds,
            "duration_seconds",
            strictly_positive=True,
        )
        object.__setattr__(self, "duration_seconds", duration)
        if not isinstance(self.provenance, AssetProvenance):
            raise BenchmarkDataError("provenance must be an AssetProvenance value")


@dataclass(frozen=True, slots=True)
class BenchmarkInterval:
    asset_id: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        _require_id(self.asset_id, "interval.asset_id")
        start = _require_finite_float(
            self.start_seconds,
            "interval.start_seconds",
            minimum=0,
        )
        end = _require_finite_float(
            self.end_seconds,
            "interval.end_seconds",
            minimum=0,
        )
        if end <= start:
            raise BenchmarkDataError("interval.end_seconds must be greater than start_seconds")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)


@dataclass(frozen=True, slots=True)
class HardNegative:
    asset_id: str
    start_seconds: float
    end_seconds: float
    reason: str

    def __post_init__(self) -> None:
        _require_id(self.asset_id, "hard_negative.asset_id")
        start = _require_finite_float(
            self.start_seconds,
            "hard_negative.start_seconds",
            minimum=0,
        )
        end = _require_finite_float(
            self.end_seconds,
            "hard_negative.end_seconds",
            minimum=0,
        )
        if end <= start:
            raise BenchmarkDataError(
                "hard_negative.end_seconds must be greater than start_seconds"
            )
        _require_string(self.reason, "hard_negative.reason", max_length=2_000)
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)


@dataclass(frozen=True, slots=True)
class QueryCase:
    case_id: str
    query: str
    asset_ids: tuple[str, ...]
    domain: str
    modalities: tuple[str, ...]
    label_quality: Literal["gold", "silver"]
    split_group: str
    relevant_intervals: tuple[BenchmarkInterval, ...] = ()
    hard_negatives: tuple[HardNegative, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        _require_id(self.case_id, "case_id")
        _require_string(self.query, "query", max_length=2_000)
        asset_ids = _require_tuple(self.asset_ids, "asset_ids")
        if not asset_ids:
            raise BenchmarkDataError("asset_ids must not be empty")
        for asset_id in asset_ids:
            _require_id(asset_id, "asset_ids item")
        _require_unique(asset_ids, "asset_ids")
        _require_id(self.domain, "domain")
        modalities = _require_tuple(self.modalities, "modalities")
        if not modalities:
            raise BenchmarkDataError("modalities must not be empty")
        for modality in modalities:
            _require_id(modality, "modalities item")
        _require_unique(modalities, "modalities")
        if self.label_quality not in {"gold", "silver"}:
            raise BenchmarkDataError("label_quality must be gold or silver")
        _require_id(self.split_group, "split_group")
        relevant = _validate_typed_tuple(
            self.relevant_intervals,
            BenchmarkInterval,
            "relevant_intervals",
        )
        negatives = _validate_typed_tuple(
            self.hard_negatives,
            HardNegative,
            "hard_negatives",
        )
        _require_unique(relevant, "relevant_intervals")
        negative_ranges = tuple(
            (negative.asset_id, negative.start_seconds, negative.end_seconds)
            for negative in negatives
        )
        _require_unique(negative_ranges, "hard_negatives")
        scoped_assets = set(asset_ids)
        for interval in (*relevant, *negatives):
            if interval.asset_id not in scoped_assets:
                raise BenchmarkDataError(
                    f"interval asset {interval.asset_id!r} is not present in asset_ids"
                )
        _require_string(self.notes, "notes", allow_empty=True)


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    schema_version: int
    dataset_id: str
    dataset_version: str
    description: str
    assets: tuple[BenchmarkAsset, ...]
    cases: tuple[QueryCase, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != DATASET_SCHEMA_VERSION:
            raise BenchmarkDataError(
                f"schema_version must be the supported version {DATASET_SCHEMA_VERSION}"
            )
        _require_id(self.dataset_id, "dataset_id")
        _require_id(self.dataset_version, "dataset_version")
        _require_string(self.description, "description", allow_empty=True)
        assets = _validate_typed_tuple(self.assets, BenchmarkAsset, "assets")
        cases = _validate_typed_tuple(self.cases, QueryCase, "cases")
        if not assets:
            raise BenchmarkDataError("assets must not be empty")
        _require_unique(tuple(asset.asset_id for asset in assets), "asset ids")
        digests = tuple(asset.sha256 for asset in assets)
        if len(digests) != len(set(digests)):
            raise BenchmarkDataError("asset SHA-256 identities must be unique")
        _require_unique(tuple(case.case_id for case in cases), "case ids")
        self._validate_references(assets, cases)

    @staticmethod
    def _validate_references(
        assets: tuple[object, ...],
        cases: tuple[object, ...],
    ) -> None:
        typed_assets = tuple(asset for asset in assets if isinstance(asset, BenchmarkAsset))
        typed_cases = tuple(case for case in cases if isinstance(case, QueryCase))
        assets_by_id = {asset.asset_id: asset for asset in typed_assets}
        split_by_asset: dict[str, str] = {}
        for case in typed_cases:
            for asset_id in case.asset_ids:
                asset = assets_by_id.get(asset_id)
                if asset is None:
                    raise BenchmarkDataError(
                        f"case {case.case_id!r} references unknown asset {asset_id!r}"
                    )
                previous_split = split_by_asset.setdefault(asset_id, case.split_group)
                if previous_split != case.split_group:
                    raise BenchmarkDataError(
                        f"asset {asset_id!r} cannot belong to more than one split_group"
                    )
            for interval in (*case.relevant_intervals, *case.hard_negatives):
                asset = assets_by_id[interval.asset_id]
                if interval.end_seconds > asset.duration_seconds:
                    raise BenchmarkDataError(
                        f"interval for {interval.asset_id!r} exceeds asset duration"
                    )
            for relevant in case.relevant_intervals:
                for negative in case.hard_negatives:
                    if relevant.asset_id != negative.asset_id:
                        continue
                    if max(relevant.start_seconds, negative.start_seconds) < min(
                        relevant.end_seconds,
                        negative.end_seconds,
                    ):
                        raise BenchmarkDataError(
                            f"hard negative overlaps a relevant interval in case {case.case_id!r}"
                        )


@dataclass(frozen=True, slots=True)
class ComponentIdentity:
    component_id: str
    identity: str

    def __post_init__(self) -> None:
        _require_id(self.component_id, "component_id")
        _require_string(self.identity, "identity", max_length=2_048)


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    operating_system: str
    architecture: str
    processor: str
    memory_bytes: int
    accelerator: str | None = None

    def __post_init__(self) -> None:
        _require_string(self.operating_system, "hardware.operating_system", max_length=512)
        _require_string(self.architecture, "hardware.architecture", max_length=128)
        _require_string(self.processor, "hardware.processor", max_length=512)
        _require_positive_int(self.memory_bytes, "hardware.memory_bytes")
        if self.accelerator is not None:
            _require_string(self.accelerator, "hardware.accelerator", max_length=512)


@dataclass(frozen=True, slots=True)
class MetricValue:
    name: str
    value: float
    unit: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.name, "metric.name")
        value = _require_finite_float(self.value, "metric.value")
        object.__setattr__(self, "value", value)
        if self.unit is not None:
            _require_id(self.unit, "metric.unit")


@dataclass(frozen=True, slots=True)
class BenchmarkResultEvidence:
    rank: int
    asset_id: str
    start_seconds: float
    end_seconds: float
    score: float

    def __post_init__(self) -> None:
        _require_positive_int(self.rank, "result evidence rank")
        _require_id(self.asset_id, "result evidence asset_id")
        start = _require_finite_float(
            self.start_seconds,
            "result evidence start_seconds",
            minimum=0,
        )
        end = _require_finite_float(
            self.end_seconds,
            "result evidence end_seconds",
            minimum=0,
        )
        if end <= start:
            raise BenchmarkDataError(
                "result evidence end_seconds must be greater than start_seconds"
            )
        score = _require_finite_float(
            self.score,
            "result evidence score",
            minimum=0,
        )
        if score > 1:
            raise BenchmarkDataError("result evidence score must be between zero and one")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(self, "score", score)


@dataclass(frozen=True, slots=True)
class BenchmarkCaseOutcome:
    case_id: str
    status: Literal["complete", "failed", "skipped"]
    latency_ms: float
    result_count: int
    metrics: tuple[MetricValue, ...] = ()
    result_evidence: tuple[BenchmarkResultEvidence, ...] = ()
    diagnostic_code: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.case_id, "case_outcome.case_id")
        if self.status not in {"complete", "failed", "skipped"}:
            raise BenchmarkDataError(
                "case_outcome.status must be complete, failed or skipped"
            )
        latency = _require_finite_float(
            self.latency_ms,
            "case_outcome.latency_ms",
            minimum=0,
        )
        object.__setattr__(self, "latency_ms", latency)
        _require_nonnegative_int(self.result_count, "case_outcome.result_count")
        metrics = _validate_typed_tuple(self.metrics, MetricValue, "case_outcome.metrics")
        _require_unique(tuple(metric.name for metric in metrics), "case_outcome.metrics")
        if self.status == "complete":
            if self.diagnostic_code is not None:
                raise BenchmarkDataError(
                    "diagnostic_code must be absent for a complete case outcome"
                )
        else:
            if self.diagnostic_code is None:
                raise BenchmarkDataError(
                    "diagnostic_code is required for failed or skipped case outcomes"
                )
            _require_id(self.diagnostic_code, "case_outcome.diagnostic_code")
        evidence = _validate_typed_tuple(
            self.result_evidence,
            BenchmarkResultEvidence,
            "case_outcome.result_evidence",
        )
        if len(evidence) > MAX_RESULT_EVIDENCE_PER_CASE:
            raise BenchmarkDataError(
                "case_outcome.result_evidence exceeds the per-case limit"
            )
        ranks = tuple(item.rank for item in evidence)
        if ranks != tuple(range(1, len(evidence) + 1)):
            raise BenchmarkDataError(
                "case_outcome.result_evidence ranks must be consecutive canonical order"
            )
        exact_hits = tuple(
            (item.asset_id, item.start_seconds, item.end_seconds) for item in evidence
        )
        if len(exact_hits) != len(set(exact_hits)):
            raise BenchmarkDataError(
                "case_outcome.result_evidence contains duplicate exact hits"
            )
        if self.status == "complete":
            if self.result_count != len(evidence):
                raise BenchmarkDataError(
                    "case_outcome.result_count must match persisted result evidence"
                )
        elif self.result_count != 0 or evidence or metrics:
            raise BenchmarkDataError(
                f"{self.status} case outcomes must not contain result evidence or metrics"
            )


@dataclass(frozen=True, slots=True)
class BenchmarkRunManifest:
    schema_version: int
    run_id: str
    created_at: str
    code_sha: str
    dataset_revision: str
    model_identities: tuple[ComponentIdentity, ...]
    index_identities: tuple[ComponentIdentity, ...]
    config_identities: tuple[ComponentIdentity, ...]
    hardware: HardwareProfile
    execution_mode: Literal["cold", "warm"]
    metrics: tuple[MetricValue, ...]
    case_outcomes: tuple[BenchmarkCaseOutcome, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != RUN_SCHEMA_VERSION:
            raise BenchmarkDataError(
                f"schema_version must be the supported version {RUN_SCHEMA_VERSION}"
            )
        _require_id(self.run_id, "run_id")
        _validate_utc_timestamp(self.created_at)
        if not isinstance(self.code_sha, str) or not _CODE_SHA_RE.fullmatch(self.code_sha):
            raise BenchmarkDataError("code_sha must be a lowercase 40- or 64-character commit SHA")
        if (
            not isinstance(self.dataset_revision, str)
            or not _SHA256_RE.fullmatch(self.dataset_revision)
        ):
            raise BenchmarkDataError(
                "dataset_revision must be a lowercase 64-character SHA-256"
            )
        for field_name in (
            "model_identities",
            "index_identities",
            "config_identities",
        ):
            identities = _validate_typed_tuple(
                getattr(self, field_name),
                ComponentIdentity,
                field_name,
            )
            _require_unique(
                tuple(identity.component_id for identity in identities),
                field_name,
            )
        if not isinstance(self.hardware, HardwareProfile):
            raise BenchmarkDataError("hardware must be a HardwareProfile value")
        if self.execution_mode not in {"cold", "warm"}:
            raise BenchmarkDataError("execution_mode must be cold or warm")
        metrics = _validate_typed_tuple(self.metrics, MetricValue, "metrics")
        _require_unique(tuple(metric.name for metric in metrics), "metrics")
        outcomes = _validate_typed_tuple(
            self.case_outcomes,
            BenchmarkCaseOutcome,
            "case_outcomes",
        )
        _require_unique(tuple(outcome.case_id for outcome in outcomes), "case_outcomes")


def _validate_utc_timestamp(value: object) -> None:
    timestamp = _require_string(value, "created_at", max_length=64)
    if not timestamp.endswith("Z"):
        raise BenchmarkDataError("created_at must be an RFC 3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(f"{timestamp[:-1]}+00:00")
    except ValueError as exc:
        raise BenchmarkDataError("created_at must be a valid RFC 3339 timestamp") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise BenchmarkDataError("created_at must use UTC")
