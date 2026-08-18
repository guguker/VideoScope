from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
import hashlib
import json
import math
import re
from types import MappingProxyType


class StageKind(StrEnum):
    PROBE = "probe"
    SCENES = "scenes"
    SPEECH = "speech"
    OCR = "ocr"
    OBJECTS = "objects"
    TEXT_VECTORS = "text_vectors"
    VISUAL_DENSE = "visual_dense"
    LIGHTHOUSE = "lighthouse"


class StageState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    STALE = "stale"
    NOT_CONFIGURED = "not_configured"
    CANCELLED = "cancelled"


SEGMENT_STAGE_KINDS = frozenset(
    {
        StageKind.SCENES,
        StageKind.SPEECH,
        StageKind.OCR,
        StageKind.OBJECTS,
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ERROR_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ALLOWED_TRANSITIONS: dict[StageState, frozenset[StageState]] = {
    StageState.QUEUED: frozenset(
        {
            StageState.RUNNING,
            StageState.FAILED,
            StageState.NOT_CONFIGURED,
            StageState.CANCELLED,
        }
    ),
    StageState.RUNNING: frozenset(
        {StageState.COMPLETE, StageState.FAILED, StageState.CANCELLED}
    ),
    StageState.COMPLETE: frozenset({StageState.STALE}),
    StageState.FAILED: frozenset(),
    StageState.STALE: frozenset(),
    StageState.NOT_CONFIGURED: frozenset(),
    StageState.CANCELLED: frozenset(),
}


class AssetIdentityError(ValueError):
    pass


def asset_id_for_sha256(sha256: str) -> str:
    if type(sha256) is not str or not _SHA256_PATTERN.fullmatch(sha256):
        raise AssetIdentityError("asset SHA-256 must be a lowercase digest")
    return f"sha256:{sha256}"


def validate_artifact_identifier(value: object, *, field_name: str) -> str:
    if type(value) is not str or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a safe non-empty identifier")
    return value


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _normalise_json(value: object, *, path: str = "$") -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"stage specification contains non-finite JSON at {path}")
        return value
    if isinstance(value, Mapping):
        normalised: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"stage specification keys must be strings at {path}")
            normalised[key] = _normalise_json(item, path=f"{path}.{key}")
        return {key: normalised[key] for key in sorted(normalised)}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"stage specification contains unsupported JSON at {path}")


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _required_identity(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-empty canonical string")
    return value


def _optional_identity(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_identity(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class StageSpecification:
    kind: StageKind
    schema_version: int
    implementation_revision: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    model_identity: str | None = None
    dependencies: Mapping[str, str] = field(default_factory=dict)
    _canonical_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            kind = StageKind(self.kind)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported stage kind") from error
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise ValueError("stage schema version must be an integer")
        if self.schema_version < 1:
            raise ValueError("stage schema version must be positive")
        implementation_revision = _required_identity(
            self.implementation_revision,
            field_name="implementation revision",
        )
        model_identity = _optional_identity(self.model_identity, field_name="model identity")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("stage parameters must be an object")
        if not isinstance(self.dependencies, Mapping):
            raise ValueError("stage dependencies must be an object")
        dependencies: dict[str, str] = {}
        for key, value in self.dependencies.items():
            resolved_key = _required_identity(key, field_name="dependency name")
            dependencies[resolved_key] = _required_identity(
                value,
                field_name="dependency identity",
            )
        parameters = _normalise_json(self.parameters, path="$.parameters")
        payload = _normalise_json(
            {
                "dependencies": dependencies,
                "implementation_revision": implementation_revision,
                "kind": kind.value,
                "model_identity": model_identity,
                "parameters": parameters,
                "schema_version": self.schema_version,
            }
        )
        canonical_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "implementation_revision", implementation_revision)
        object.__setattr__(self, "model_identity", model_identity)
        object.__setattr__(self, "parameters", _freeze_json(parameters))
        object.__setattr__(
            self,
            "dependencies",
            MappingProxyType(dict(sorted(dependencies.items()))),
        )
        object.__setattr__(self, "_canonical_json", canonical_json)

    @property
    def canonical_json(self) -> str:
        return self._canonical_json

    @property
    def specification_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @classmethod
    def from_canonical_json(cls, value: str) -> StageSpecification:
        if type(value) is not str:
            raise ValueError("canonical stage specification must be a string")
        try:
            payload = json.loads(value, parse_constant=_reject_non_finite_json)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("invalid canonical stage specification") from error
        if not isinstance(payload, dict):
            raise ValueError("canonical stage specification must be an object")
        expected_keys = {
            "dependencies",
            "implementation_revision",
            "kind",
            "model_identity",
            "parameters",
            "schema_version",
        }
        if set(payload) != expected_keys:
            raise ValueError("canonical stage specification has unsupported fields")
        specification = cls(
            kind=payload["kind"],
            schema_version=payload["schema_version"],
            implementation_revision=payload["implementation_revision"],
            parameters=payload["parameters"],
            model_identity=payload["model_identity"],
            dependencies=payload["dependencies"],
        )
        if specification.canonical_json != value:
            raise ValueError("stage specification is not canonical")
        return specification


@dataclass(frozen=True, slots=True)
class IndexingSpecifications:
    """Explicit runtime identities for every SQLite/text indexing output."""

    scenes: StageSpecification
    speech: StageSpecification
    ocr: StageSpecification
    objects: StageSpecification
    text_vectors: StageSpecification

    def __post_init__(self) -> None:
        expected = {
            "scenes": StageKind.SCENES,
            "speech": StageKind.SPEECH,
            "ocr": StageKind.OCR,
            "objects": StageKind.OBJECTS,
            "text_vectors": StageKind.TEXT_VECTORS,
        }
        for field_name, kind in expected.items():
            specification = getattr(self, field_name)
            if not isinstance(specification, StageSpecification):
                raise ValueError(f"{field_name} specification must be validated")
            if specification.kind is not kind:
                raise ValueError(
                    f"{field_name} specification must describe the {kind.value} stage"
                )

    @property
    def segment_specifications(self) -> tuple[StageSpecification, ...]:
        return (self.scenes, self.speech, self.ocr, self.objects)

    @property
    def semantic_segment_specifications(self) -> tuple[StageSpecification, ...]:
        return (self.speech, self.ocr, self.objects)

    def for_kind(self, kind: StageKind) -> StageSpecification:
        try:
            resolved = StageKind(kind)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported indexing stage kind") from error
        specifications = {
            StageKind.SCENES: self.scenes,
            StageKind.SPEECH: self.speech,
            StageKind.OCR: self.ocr,
            StageKind.OBJECTS: self.objects,
            StageKind.TEXT_VECTORS: self.text_vectors,
        }
        try:
            return specifications[resolved]
        except KeyError as error:
            raise ValueError("stage is not produced by the indexer") from error


def validate_stage_transition(current: StageState, target: StageState) -> None:
    try:
        resolved_current = StageState(current)
        resolved_target = StageState(target)
    except (TypeError, ValueError) as error:
        raise ValueError("unsupported stage state") from error
    if resolved_target not in _ALLOWED_TRANSITIONS[resolved_current]:
        raise ValueError(
            f"invalid stage state transition: {resolved_current.value} -> {resolved_target.value}"
        )


def validate_error_code(error_code: str | None) -> None:
    if error_code is not None and not _ERROR_CODE_PATTERN.fullmatch(error_code):
        raise ValueError("stage error code must be a sanitized symbolic code")


def _validate_timestamp(value: object, *, field_name: str, required: bool) -> datetime | None:
    if value is None and not required:
        return None
    if type(value) is not str:
        raise ValueError(f"{field_name} must be an ISO timestamp")
    try:
        resolved = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO timestamp") from error
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return resolved


@dataclass(frozen=True, slots=True)
class AssetRecord:
    asset_id: str
    sha256: str
    size_bytes: int
    created_at: str

    def __post_init__(self) -> None:
        expected_id = asset_id_for_sha256(self.sha256)
        if self.asset_id != expected_id:
            raise AssetIdentityError("asset id does not match its content digest")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise AssetIdentityError("asset size must be a positive integer")
        _validate_timestamp(self.created_at, field_name="created_at", required=True)

    @classmethod
    def from_digest(
        cls,
        *,
        sha256: str,
        size_bytes: int,
        created_at: str,
    ) -> AssetRecord:
        return cls(
            asset_id=asset_id_for_sha256(sha256),
            sha256=sha256,
            size_bytes=size_bytes,
            created_at=created_at,
        )


@dataclass(frozen=True, slots=True)
class SegmentGeneration:
    generation_id: str
    video_id: str
    stage_kind: StageKind
    specification_hash: str
    source_sha256: str
    run_id: str
    segment_count: int
    completed_at: str

    def __post_init__(self) -> None:
        validate_artifact_identifier(
            self.generation_id,
            field_name="segment generation id",
        )
        validate_artifact_identifier(
            self.video_id,
            field_name="segment generation video id",
        )
        try:
            stage_kind = StageKind(self.stage_kind)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported segment generation stage") from error
        if stage_kind not in SEGMENT_STAGE_KINDS:
            raise ValueError("stage does not produce SQLite segment generations")
        if type(self.specification_hash) is not str or not _SHA256_PATTERN.fullmatch(
            self.specification_hash
        ):
            raise ValueError("segment generation specification hash must be lowercase SHA-256")
        if type(self.source_sha256) is not str or not _SHA256_PATTERN.fullmatch(
            self.source_sha256
        ):
            raise ValueError("segment generation source hash must be lowercase SHA-256")
        validate_artifact_identifier(self.run_id, field_name="segment generation run id")
        if (
            isinstance(self.segment_count, bool)
            or not isinstance(self.segment_count, int)
            or self.segment_count < 0
        ):
            raise ValueError("segment generation count must be a non-negative integer")
        _validate_timestamp(self.completed_at, field_name="completed_at", required=True)
        object.__setattr__(self, "stage_kind", stage_kind)


@dataclass(frozen=True, slots=True)
class StageRun:
    run_id: str
    video_id: str
    stage_kind: StageKind
    state: StageState
    specification_hash: str
    source_sha256: str
    attempt: int
    output_generation: str | None
    error_code: str | None
    retry_of_run_id: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    updated_at: str

    def __post_init__(self) -> None:
        if type(self.run_id) is not str or not _IDENTIFIER_PATTERN.fullmatch(self.run_id):
            raise ValueError("stage run id must be a safe non-empty identifier")
        if type(self.video_id) is not str or not self.video_id:
            raise ValueError("stage run video id must not be empty")
        try:
            stage_kind = StageKind(self.stage_kind)
            state = StageState(self.state)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported stage run kind or state") from error
        if type(self.specification_hash) is not str or not _SHA256_PATTERN.fullmatch(
            self.specification_hash
        ):
            raise ValueError("stage specification hash must be lowercase SHA-256")
        if type(self.source_sha256) is not str or not _SHA256_PATTERN.fullmatch(
            self.source_sha256
        ):
            raise ValueError("stage source hash must be lowercase SHA-256")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("stage attempt must be a positive integer")
        if self.output_generation is not None and (
            type(self.output_generation) is not str
            or not _IDENTIFIER_PATTERN.fullmatch(self.output_generation)
        ):
            raise ValueError("stage output generation must be a safe non-empty identifier")
        validate_error_code(self.error_code)
        if self.retry_of_run_id is not None and (
            type(self.retry_of_run_id) is not str
            or not _IDENTIFIER_PATTERN.fullmatch(self.retry_of_run_id)
            or self.retry_of_run_id == self.run_id
        ):
            raise ValueError("stage retry lineage must reference another safe run id")
        if self.retry_of_run_id is not None and self.attempt < 2:
            raise ValueError("a retry must have an attempt greater than one")

        created_at = _validate_timestamp(self.created_at, field_name="created_at", required=True)
        updated_at = _validate_timestamp(self.updated_at, field_name="updated_at", required=True)
        started_at = _validate_timestamp(self.started_at, field_name="started_at", required=False)
        finished_at = _validate_timestamp(
            self.finished_at,
            field_name="finished_at",
            required=False,
        )
        assert created_at is not None and updated_at is not None
        if updated_at < created_at:
            raise ValueError("stage updated timestamp cannot precede creation")
        if started_at is not None and updated_at < started_at:
            raise ValueError("stage updated timestamp cannot precede start")
        if finished_at is not None and updated_at < finished_at:
            raise ValueError("stage updated timestamp cannot precede outcome")
        if started_at is not None and started_at < created_at:
            raise ValueError("stage start timestamp cannot precede creation")
        if finished_at is not None and finished_at < created_at:
            raise ValueError("stage finish timestamp cannot precede creation")
        if started_at is not None and finished_at is not None and finished_at < started_at:
            raise ValueError("stage finish timestamp cannot precede start")

        if state is StageState.QUEUED:
            if any((started_at, finished_at, self.output_generation, self.error_code)):
                raise ValueError("queued stage run cannot contain outcome fields")
        elif state is StageState.RUNNING:
            if started_at is None or finished_at is not None:
                raise ValueError("running stage run requires only a start timestamp")
            if self.output_generation is not None or self.error_code is not None:
                raise ValueError("running stage run cannot contain outcome fields")
        elif state is StageState.COMPLETE:
            if started_at is None or finished_at is None or self.output_generation is None:
                raise ValueError("complete stage run requires timestamps and an output generation")
            if self.error_code is not None:
                raise ValueError("complete stage run cannot contain an error code")
        elif state is StageState.FAILED:
            if finished_at is None or self.error_code is None:
                raise ValueError("failed stage run requires a finish timestamp and error code")
            if self.output_generation is not None:
                raise ValueError("failed stage run cannot contain an output generation")
        elif state in {StageState.CANCELLED, StageState.NOT_CONFIGURED}:
            if finished_at is None:
                raise ValueError("terminal stage run requires a finish timestamp")
            if self.output_generation is not None or self.error_code is not None:
                raise ValueError("terminal stage run cannot contain outcome details")
        elif state is StageState.STALE:
            if started_at is None:
                raise ValueError("stale stage run requires a start timestamp")
            if finished_at is None or self.output_generation is None:
                raise ValueError(
                    "stale stage run requires a finish timestamp and output generation"
                )
            if self.error_code is not None:
                raise ValueError("stale stage run cannot contain an error code")

        object.__setattr__(self, "stage_kind", stage_kind)
        object.__setattr__(self, "state", state)
