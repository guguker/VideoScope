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


class ArtifactGCOutcome(StrEnum):
    SUCCESS = "success"
    TRANSIENT_FAILURE = "transient_failure"
    PERMANENT_FAILURE = "permanent_failure"


class ArtifactGCAttemptOutcome(StrEnum):
    SUCCESS = "success"
    TRANSIENT_FAILURE = "transient_failure"
    PERMANENT_FAILURE = "permanent_failure"
    LEASE_EXPIRED = "lease_expired"


SEGMENT_STAGE_KINDS = frozenset(
    {
        StageKind.SCENES,
        StageKind.SPEECH,
        StageKind.OCR,
        StageKind.OBJECTS,
    }
)
TEXT_VECTOR_INPUT_STAGE_KINDS = (
    StageKind.SPEECH,
    StageKind.OCR,
    StageKind.OBJECTS,
)
TEXT_VECTOR_MODALITY_BY_STAGE = {
    StageKind.SPEECH: "speech",
    StageKind.OCR: "ocr",
    StageKind.OBJECTS: "objects",
}
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
        except (TypeError, ValueError, RecursionError, json.JSONDecodeError) as error:
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
        try:
            specification = cls(
                kind=payload["kind"],
                schema_version=payload["schema_version"],
                implementation_revision=payload["implementation_revision"],
                parameters=payload["parameters"],
                model_identity=payload["model_identity"],
                dependencies=payload["dependencies"],
            )
        except RecursionError as error:
            raise ValueError("invalid canonical stage specification") from error
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
class TextVectorIndexSpecification:
    """Physical Qdrant contract shared by immutable per-video generations."""

    embedding_identity: str
    dimensions: int
    distance: str = "cosine"
    payload_schema_version: int = 1
    point_id_algorithm: str = "uuid5-generation-segment-v1"
    _canonical_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        embedding_identity = _required_identity(
            self.embedding_identity,
            field_name="text vector embedding identity",
        )
        if (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, int)
            or self.dimensions <= 0
        ):
            raise ValueError("text vector dimensions must be a positive integer")
        if self.distance != "cosine":
            raise ValueError("unsupported text vector distance")
        if (
            isinstance(self.payload_schema_version, bool)
            or not isinstance(self.payload_schema_version, int)
            or self.payload_schema_version < 1
        ):
            raise ValueError("text vector payload schema version must be positive")
        point_id_algorithm = validate_artifact_identifier(
            self.point_id_algorithm,
            field_name="text vector point id algorithm",
        )
        payload = {
            "dimensions": self.dimensions,
            "distance": self.distance,
            "embedding_identity": embedding_identity,
            "payload_schema_version": self.payload_schema_version,
            "point_id_algorithm": point_id_algorithm,
        }
        canonical_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        object.__setattr__(self, "embedding_identity", embedding_identity)
        object.__setattr__(self, "point_id_algorithm", point_id_algorithm)
        object.__setattr__(self, "_canonical_json", canonical_json)

    @property
    def canonical_json(self) -> str:
        return self._canonical_json

    @property
    def specification_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def collection_name(self) -> str:
        return f"videoscope_text_v1_{self.specification_hash[:32]}"

    @classmethod
    def from_canonical_json(cls, value: str) -> TextVectorIndexSpecification:
        if type(value) is not str:
            raise ValueError("canonical text vector index specification must be a string")
        try:
            payload = json.loads(value, parse_constant=_reject_non_finite_json)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("invalid canonical text vector index specification") from error
        if not isinstance(payload, dict) or set(payload) != {
            "dimensions",
            "distance",
            "embedding_identity",
            "payload_schema_version",
            "point_id_algorithm",
        }:
            raise ValueError("canonical text vector index specification has unsupported fields")
        specification = cls(**payload)
        if specification.canonical_json != value:
            raise ValueError("text vector index specification is not canonical")
        return specification


def _validate_sha256(value: object, *, field_name: str) -> str:
    if type(value) is not str or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be lowercase SHA-256")
    return value


def _validate_non_negative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class TextVectorGenerationInput:
    stage_kind: StageKind
    specification_hash: str
    segment_generation_id: str | None
    segment_run_id: str | None
    source_sha256: str | None
    segment_count: int
    content_manifest_sha256: str

    def __post_init__(self) -> None:
        try:
            stage_kind = StageKind(self.stage_kind)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported text vector input stage") from error
        if stage_kind not in TEXT_VECTOR_INPUT_STAGE_KINDS:
            raise ValueError("text vector input must be a semantic segment stage")
        _validate_sha256(
            self.specification_hash,
            field_name="text vector input specification hash",
        )
        _validate_non_negative_integer(
            self.segment_count,
            field_name="text vector input segment count",
        )
        _validate_sha256(
            self.content_manifest_sha256,
            field_name="text vector input content manifest",
        )
        if self.segment_generation_id is None:
            if self.segment_run_id is not None or self.source_sha256 is not None:
                raise ValueError("absent text vector input cannot claim generation lineage")
            if self.segment_count != 0:
                raise ValueError("absent text vector input must have zero segments")
        else:
            validate_artifact_identifier(
                self.segment_generation_id,
                field_name="text vector input generation id",
            )
            if self.segment_run_id is None or self.source_sha256 is None:
                raise ValueError("present text vector input requires complete lineage")
            validate_artifact_identifier(
                self.segment_run_id,
                field_name="text vector input run id",
            )
            _validate_sha256(
                self.source_sha256,
                field_name="text vector input source hash",
            )
        object.__setattr__(self, "stage_kind", stage_kind)

    @property
    def modality(self) -> str:
        return TEXT_VECTOR_MODALITY_BY_STAGE[self.stage_kind]

    @property
    def present(self) -> bool:
        return self.segment_generation_id is not None


@dataclass(frozen=True, slots=True)
class TextVectorPointSource:
    video_id: str
    segment_id: str
    modality: str
    text: str
    segment_generation_id: str
    text_sha256: str

    def __post_init__(self) -> None:
        validate_artifact_identifier(self.video_id, field_name="text vector point video id")
        validate_artifact_identifier(
            self.segment_id,
            field_name="text vector point segment id",
        )
        if self.modality not in set(TEXT_VECTOR_MODALITY_BY_STAGE.values()):
            raise ValueError("unsupported text vector point modality")
        if type(self.text) is not str or not self.text.strip():
            raise ValueError("text vector point text must not be empty")
        validate_artifact_identifier(
            self.segment_generation_id,
            field_name="text vector point segment generation id",
        )
        _validate_sha256(self.text_sha256, field_name="text vector point text hash")
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.text_sha256:
            raise ValueError("text vector point text hash does not match text")


@dataclass(frozen=True, slots=True)
class TextVectorBuildPlan:
    generation_id: str
    run_id: str
    video_id: str
    stage_specification_hash: str
    source_sha256: str
    index_specification: TextVectorIndexSpecification
    expected_previous_generation_id: str | None
    inputs: tuple[TextVectorGenerationInput, ...]
    points: tuple[TextVectorPointSource, ...]
    input_manifest_sha256: str
    point_manifest_sha256: str
    reserved_at: str
    lease_expires_at: str

    def __post_init__(self) -> None:
        validate_artifact_identifier(self.generation_id, field_name="text vector generation id")
        validate_artifact_identifier(self.run_id, field_name="text vector build run id")
        validate_artifact_identifier(self.video_id, field_name="text vector build video id")
        _validate_sha256(
            self.stage_specification_hash,
            field_name="text vector stage specification hash",
        )
        _validate_sha256(self.source_sha256, field_name="text vector source hash")
        if not isinstance(self.index_specification, TextVectorIndexSpecification):
            raise ValueError("text vector build index specification must be validated")
        if self.expected_previous_generation_id is not None:
            validate_artifact_identifier(
                self.expected_previous_generation_id,
                field_name="expected previous text vector generation id",
            )
        if type(self.inputs) is not tuple or any(
            not isinstance(item, TextVectorGenerationInput) for item in self.inputs
        ):
            raise ValueError("text vector build inputs must be validated")
        if tuple(item.stage_kind for item in self.inputs) != TEXT_VECTOR_INPUT_STAGE_KINDS:
            raise ValueError("text vector build inputs must cover every semantic stage in order")
        if type(self.points) is not tuple or any(
            not isinstance(item, TextVectorPointSource) for item in self.points
        ):
            raise ValueError("text vector build points must be validated")
        if any(item.video_id != self.video_id for item in self.points):
            raise ValueError("text vector points must match build video")
        if len({item.segment_id for item in self.points}) != len(self.points):
            raise ValueError("text vector point segment ids must be unique")
        _validate_sha256(
            self.input_manifest_sha256,
            field_name="text vector input manifest",
        )
        _validate_sha256(
            self.point_manifest_sha256,
            field_name="text vector point manifest",
        )
        reserved_at = _validate_timestamp(
            self.reserved_at,
            field_name="reserved_at",
            required=True,
        )
        lease_expires_at = _validate_timestamp(
            self.lease_expires_at,
            field_name="lease_expires_at",
            required=True,
        )
        assert reserved_at is not None and lease_expires_at is not None
        if lease_expires_at <= reserved_at:
            raise ValueError("text vector build lease must expire after reservation")


@dataclass(frozen=True, slots=True)
class TextVectorBuildReceipt:
    generation_id: str
    index_specification_hash: str
    point_count: int
    point_manifest_sha256: str
    vector_manifest_sha256: str

    def __post_init__(self) -> None:
        validate_artifact_identifier(self.generation_id, field_name="text vector receipt generation id")
        _validate_sha256(
            self.index_specification_hash,
            field_name="text vector receipt index specification hash",
        )
        _validate_non_negative_integer(
            self.point_count,
            field_name="text vector receipt point count",
        )
        _validate_sha256(
            self.point_manifest_sha256,
            field_name="text vector receipt point manifest",
        )
        _validate_sha256(
            self.vector_manifest_sha256,
            field_name="text vector receipt vector manifest",
        )


@dataclass(frozen=True, slots=True)
class TextVectorGeneration:
    generation_id: str
    video_id: str
    specification_hash: str
    source_sha256: str
    run_id: str
    index_specification_hash: str
    collection_name: str
    input_manifest_sha256: str
    point_manifest_sha256: str
    vector_manifest_sha256: str
    point_count: int
    completed_at: str

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.generation_id, "text vector generation id"),
            (self.video_id, "text vector generation video id"),
            (self.run_id, "text vector generation run id"),
            (self.collection_name, "text vector collection name"),
        ):
            validate_artifact_identifier(value, field_name=field_name)
        for value, field_name in (
            (self.specification_hash, "text vector stage specification hash"),
            (self.source_sha256, "text vector source hash"),
            (self.index_specification_hash, "text vector index specification hash"),
            (self.input_manifest_sha256, "text vector input manifest"),
            (self.point_manifest_sha256, "text vector point manifest"),
            (self.vector_manifest_sha256, "text vector vector manifest"),
        ):
            _validate_sha256(value, field_name=field_name)
        _validate_non_negative_integer(self.point_count, field_name="text vector point count")
        _validate_timestamp(self.completed_at, field_name="completed_at", required=True)


@dataclass(frozen=True, slots=True)
class TextVectorSearchBinding:
    generation: TextVectorGeneration
    index_specification: TextVectorIndexSpecification
    inputs: tuple[TextVectorGenerationInput, ...]
    points: tuple[TextVectorPointSource, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.generation, TextVectorGeneration):
            raise ValueError("text vector search generation must be validated")
        if not isinstance(self.index_specification, TextVectorIndexSpecification):
            raise ValueError("text vector search index specification must be validated")
        if (
            self.generation.index_specification_hash
            != self.index_specification.specification_hash
            or self.generation.collection_name != self.index_specification.collection_name
        ):
            raise ValueError("text vector search index identity is inconsistent")
        if tuple(item.stage_kind for item in self.inputs) != TEXT_VECTOR_INPUT_STAGE_KINDS:
            raise ValueError("text vector search inputs are incomplete")
        if type(self.points) is not tuple or any(
            not isinstance(item, TextVectorPointSource) for item in self.points
        ):
            raise ValueError("text vector search points must be validated")
        if len(self.points) != self.generation.point_count or any(
            item.video_id != self.generation.video_id for item in self.points
        ):
            raise ValueError("text vector search points do not match generation")
        input_generation_ids = {
            item.segment_generation_id for item in self.inputs if item.present
        }
        if any(
            item.segment_generation_id not in input_generation_ids
            for item in self.points
        ):
            raise ValueError("text vector search point lineage is inconsistent")

    @property
    def video_id(self) -> str:
        return self.generation.video_id

    @property
    def generation_id(self) -> str:
        return self.generation.generation_id

    def supports_modality(self, modality: str) -> bool:
        return any(item.modality == modality and item.present for item in self.inputs)


@dataclass(frozen=True, slots=True)
class TextVectorSearchHit:
    video_id: str
    generation_id: str
    segment_id: str
    modality: str
    score: float

    def __post_init__(self) -> None:
        validate_artifact_identifier(self.video_id, field_name="text vector hit video id")
        validate_artifact_identifier(
            self.generation_id,
            field_name="text vector hit generation id",
        )
        validate_artifact_identifier(self.segment_id, field_name="text vector hit segment id")
        if self.modality not in set(TEXT_VECTOR_MODALITY_BY_STAGE.values()):
            raise ValueError("unsupported text vector hit modality")
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(float(self.score))
        ):
            raise ValueError("text vector hit score must be finite")
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True, slots=True)
class ArtifactGCJob:
    job_id: str
    artifact_kind: str
    generation_id: str
    index_specification_hash: str
    collection_name: str
    reason: str
    state: str
    attempt: int
    backoff_level: int
    available_at: str
    worker_id: str | None
    lease_token: str | None
    lease_expires_at: str | None
    error_code: str | None
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.job_id, "artifact GC job id"),
            (self.generation_id, "artifact GC generation id"),
            (self.collection_name, "artifact GC collection name"),
            (self.reason, "artifact GC reason"),
        ):
            validate_artifact_identifier(value, field_name=field_name)
        if self.artifact_kind != StageKind.TEXT_VECTORS.value:
            raise ValueError("unsupported artifact GC kind")
        _validate_sha256(
            self.index_specification_hash,
            field_name="artifact GC index specification hash",
        )
        if self.state not in {"pending", "running", "complete", "failed"}:
            raise ValueError("unsupported artifact GC state")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 0:
            raise ValueError("artifact GC attempt must be non-negative")
        if (
            isinstance(self.backoff_level, bool)
            or not isinstance(self.backoff_level, int)
            or not 0 <= self.backoff_level <= 7
        ):
            raise ValueError("artifact GC backoff level is invalid")
        created = _validate_timestamp(self.created_at, field_name="created_at", required=True)
        updated = _validate_timestamp(self.updated_at, field_name="updated_at", required=True)
        available = _validate_timestamp(
            self.available_at,
            field_name="available_at",
            required=True,
        )
        lease_expires = _validate_timestamp(
            self.lease_expires_at,
            field_name="lease_expires_at",
            required=False,
        )
        assert created is not None and updated is not None and available is not None
        if updated < created:
            raise ValueError("artifact GC update cannot precede creation")
        if available < created:
            raise ValueError("artifact GC availability cannot precede creation")
        validate_error_code(self.error_code)
        if self.state == "running":
            if self.attempt < 1:
                raise ValueError("running artifact GC job requires an attempt")
            if self.worker_id is None or self.lease_token is None or lease_expires is None:
                raise ValueError("running artifact GC job requires a complete lease")
            validate_artifact_identifier(
                self.worker_id,
                field_name="artifact GC worker id",
            )
            validate_artifact_identifier(
                self.lease_token,
                field_name="artifact GC lease token",
            )
            if lease_expires <= updated:
                raise ValueError("artifact GC lease must expire after its update")
            if self.error_code is not None:
                raise ValueError("running artifact GC job cannot contain an error")
        else:
            if any(
                value is not None
                for value in (self.worker_id, self.lease_token, self.lease_expires_at)
            ):
                raise ValueError("non-running artifact GC job cannot retain a lease")
            if self.state == "complete" and self.error_code is not None:
                raise ValueError("complete artifact GC job cannot contain an error")
            if self.state == "failed" and self.error_code is None:
                raise ValueError("failed artifact GC job requires an error code")


@dataclass(frozen=True, slots=True)
class ArtifactGCAttempt:
    job_id: str
    attempt: int
    worker_id: str
    lease_token: str
    claimed_at: str
    lease_expires_at: str
    finished_at: str | None
    outcome: str | None
    error_code: str | None

    def __post_init__(self) -> None:
        validate_artifact_identifier(self.job_id, field_name="artifact GC audit job id")
        validate_artifact_identifier(
            self.worker_id,
            field_name="artifact GC audit worker id",
        )
        validate_artifact_identifier(
            self.lease_token,
            field_name="artifact GC audit lease token",
        )
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError("artifact GC audit attempt must be positive")
        claimed = _validate_timestamp(
            self.claimed_at,
            field_name="claimed_at",
            required=True,
        )
        lease_expires = _validate_timestamp(
            self.lease_expires_at,
            field_name="lease_expires_at",
            required=True,
        )
        finished = _validate_timestamp(
            self.finished_at,
            field_name="finished_at",
            required=False,
        )
        assert claimed is not None and lease_expires is not None
        if lease_expires <= claimed:
            raise ValueError("artifact GC audit lease must expire after claim")
        if finished is not None and finished < claimed:
            raise ValueError("artifact GC audit finish cannot precede claim")
        validate_error_code(self.error_code)
        if self.outcome is None:
            if finished is not None or self.error_code is not None:
                raise ValueError("unfinished artifact GC audit cannot have an outcome")
            return
        try:
            outcome = ArtifactGCAttemptOutcome(self.outcome)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported artifact GC audit outcome") from error
        if finished is None:
            raise ValueError("finished artifact GC audit requires a timestamp")
        if outcome is ArtifactGCAttemptOutcome.SUCCESS:
            if self.error_code is not None:
                raise ValueError("successful artifact GC audit cannot contain an error")
        elif self.error_code is None:
            raise ValueError("failed artifact GC audit requires an error code")
        if (
            outcome is ArtifactGCAttemptOutcome.LEASE_EXPIRED
            and finished < lease_expires
        ):
            raise ValueError("expired artifact GC audit finished before lease expiry")
        object.__setattr__(self, "outcome", outcome.value)


@dataclass(frozen=True, slots=True)
class ArtifactGCQuarantine:
    sequence: int
    error_code: str
    quarantined_at: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise ValueError("artifact GC quarantine sequence must be positive")
        validate_error_code(self.error_code)
        if self.error_code is None:
            raise ValueError("artifact GC quarantine requires an error code")
        _validate_timestamp(
            self.quarantined_at,
            field_name="quarantined_at",
            required=True,
        )


@dataclass(frozen=True, slots=True)
class TextVectorBuildQuarantine:
    row_id: int
    error_code: str
    quarantined_at: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.row_id, bool)
            or not isinstance(self.row_id, int)
            or self.row_id < 1
        ):
            raise ValueError("text vector build quarantine row id must be positive")
        validate_error_code(self.error_code)
        if self.error_code is None:
            raise ValueError("text vector build quarantine requires an error code")
        _validate_timestamp(
            self.quarantined_at,
            field_name="quarantined_at",
            required=True,
        )


@dataclass(frozen=True, slots=True)
class TextVectorBuildRecoveryReport:
    examined_count: int
    terminalized_generation_ids: tuple[str, ...]
    quarantined_count: int
    has_more: bool

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.examined_count, "text vector recovery examined count"),
            (self.quarantined_count, "text vector recovery quarantine count"),
        ):
            _validate_non_negative_integer(value, field_name=field_name)
        if type(self.terminalized_generation_ids) is not tuple:
            raise ValueError("text vector recovered generation ids must be a tuple")
        for generation_id in self.terminalized_generation_ids:
            validate_artifact_identifier(
                generation_id,
                field_name="recovered text vector generation id",
            )
        if len(set(self.terminalized_generation_ids)) != len(
            self.terminalized_generation_ids
        ):
            raise ValueError("recovered text vector generation ids must be unique")
        if self.examined_count != (
            len(self.terminalized_generation_ids) + self.quarantined_count
        ):
            raise ValueError("text vector recovery counts are inconsistent")
        if type(self.has_more) is not bool:
            raise ValueError("text vector recovery has_more must be boolean")


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
