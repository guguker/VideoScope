from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Literal

from .schema import (
    MAX_DATASET_ASSETS,
    MAX_DATASET_MEDIA_BYTES,
    AssetProvenance,
    BenchmarkAsset,
    BenchmarkDataError,
    BenchmarkInterval,
    ComponentIdentity,
    _require_id,
    _require_positive_int,
    _require_string,
    _require_unique,
    _validate_typed_tuple,
)
from .serialization import (
    JsonObject,
    canonical_json_bytes,
    expect_fields,
    expect_list,
    expect_object,
    parse_json_object,
)


VIDEO_VERIFIER_DATASET_SCHEMA_VERSION = 1
MAX_PREPARED_INPUT_BYTES = 128 * 1024**2
MAX_VIDEO_VERIFIER_CASES = 10_000
MAX_VIDEO_VERIFIER_MANIFEST_BYTES = 8 * 1024**2
MAX_VIDEO_VERIFIER_PREPARATION_PROTOCOLS = 128

VerifierStratum = Literal["basketball_facts", "generic_visual"]
VerifierInputKind = Literal["native_video", "storyboard"]
VerifierLabelQuality = Literal["gold", "silver"]
VerifierSplit = Literal[
    "regression_seen",
    "train",
    "validation",
    "promotion_holdout",
]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JERSEY_RE = re.compile(r"^(?:0|00|[1-9][0-9]?)$")
_EXPECTED_FACTS_BY_STRATUM: dict[str, frozenset[str]] = {
    "basketball_facts": frozenset(
        {
            "shot_attempt",
            "ball_through_hoop",
            "shooter_outside_arc",
            "three_point_signal",
        }
    ),
    "generic_visual": frozenset({"matches_query"}),
}
_VALID_INPUT_KINDS = frozenset({"native_video", "storyboard"})
_VALID_LABEL_QUALITIES = frozenset({"gold", "silver"})
_VALID_SPLITS = frozenset(
    {"regression_seen", "train", "validation", "promotion_holdout"}
)


@dataclass(frozen=True, slots=True)
class VideoVerifierExpectedFact:
    fact_id: str
    expected: bool | None

    def __post_init__(self) -> None:
        _require_id(self.fact_id, "expected_fact.fact_id")
        if self.expected is not None and type(self.expected) is not bool:
            raise BenchmarkDataError(
                "expected_fact.expected must be true, false or null"
            )


@dataclass(frozen=True, slots=True)
class PreparedInputProtocol:
    protocol_id: str
    input_kind: VerifierInputKind
    preparer: ComponentIdentity
    parameters: tuple[ComponentIdentity, ...]

    def __post_init__(self) -> None:
        _require_id(self.protocol_id, "preparation_protocol.protocol_id")
        if (
            not isinstance(self.input_kind, str)
            or self.input_kind not in _VALID_INPUT_KINDS
        ):
            raise BenchmarkDataError(
                "preparation_protocol.input_kind must be native_video or storyboard"
            )
        if not isinstance(self.preparer, ComponentIdentity):
            raise BenchmarkDataError(
                "preparation_protocol.preparer must be a ComponentIdentity value"
            )
        parameters = _validate_typed_tuple(
            self.parameters,
            ComponentIdentity,
            "preparation_protocol.parameters",
        )
        if not parameters:
            raise BenchmarkDataError(
                "preparation_protocol.parameters must not be empty"
            )
        _require_unique(
            tuple(parameter.component_id for parameter in parameters),
            "preparation_protocol parameter ids",
        )
        object.__setattr__(
            self,
            "parameters",
            tuple(sorted(parameters, key=lambda item: item.component_id)),
        )


@dataclass(frozen=True, slots=True)
class VideoVerifierCase:
    case_id: str
    stratum: VerifierStratum
    input_kind: VerifierInputKind
    source_interval: BenchmarkInterval
    prepared_input_protocol_id: str
    prepared_input_sha256: str
    prepared_input_byte_size: int
    query: str | None
    expected_facts: tuple[VideoVerifierExpectedFact, ...]
    expected_jersey: str | None
    label_quality: VerifierLabelQuality
    split: VerifierSplit
    split_group: str
    notes: str = ""

    def __post_init__(self) -> None:
        _require_id(self.case_id, "verifier_case.case_id")
        expected_fact_ids = (
            _EXPECTED_FACTS_BY_STRATUM.get(self.stratum)
            if isinstance(self.stratum, str)
            else None
        )
        if expected_fact_ids is None:
            raise BenchmarkDataError(
                "verifier_case.stratum must be basketball_facts or generic_visual"
            )
        if (
            not isinstance(self.input_kind, str)
            or self.input_kind not in _VALID_INPUT_KINDS
        ):
            raise BenchmarkDataError(
                "verifier_case.input_kind must be native_video or storyboard"
            )
        if not isinstance(self.source_interval, BenchmarkInterval):
            raise BenchmarkDataError(
                "verifier_case.source_interval must be a BenchmarkInterval value"
            )
        _require_id(
            self.prepared_input_protocol_id,
            "verifier_case.prepared_input_protocol_id",
        )
        if (
            not isinstance(self.prepared_input_sha256, str)
            or not _SHA256_RE.fullmatch(self.prepared_input_sha256)
        ):
            raise BenchmarkDataError(
                "verifier_case.prepared_input_sha256 must be a lowercase "
                "64-character SHA-256"
            )
        prepared_size = _require_positive_int(
            self.prepared_input_byte_size,
            "verifier_case.prepared_input_byte_size",
        )
        if prepared_size > MAX_PREPARED_INPUT_BYTES:
            raise BenchmarkDataError(
                "verifier_case prepared input exceeds the byte size limit"
            )
        if self.stratum == "generic_visual":
            if self.query is None:
                raise BenchmarkDataError(
                    "verifier_case.query is required for generic_visual"
                )
            _require_string(self.query, "verifier_case.query", max_length=2_000)
        elif self.query is not None:
            raise BenchmarkDataError(
                "verifier_case.query must be null for basketball_facts"
            )
        facts = _validate_typed_tuple(
            self.expected_facts,
            VideoVerifierExpectedFact,
            "verifier_case.expected_facts",
        )
        _require_unique(
            tuple(fact.fact_id for fact in facts),
            "verifier_case expected fact ids",
        )
        if frozenset(fact.fact_id for fact in facts) != expected_fact_ids:
            required = ", ".join(sorted(expected_fact_ids))
            raise BenchmarkDataError(
                "verifier_case.expected_facts must contain exactly: " + required
            )
        object.__setattr__(
            self,
            "expected_facts",
            tuple(sorted(facts, key=lambda fact: fact.fact_id)),
        )
        if self.expected_jersey is not None:
            if (
                not isinstance(self.expected_jersey, str)
                or not _JERSEY_RE.fullmatch(self.expected_jersey)
            ):
                raise BenchmarkDataError(
                    "verifier_case.expected_jersey must be 0, 00, 1-99 or null"
                )
        if (
            not isinstance(self.label_quality, str)
            or self.label_quality not in _VALID_LABEL_QUALITIES
        ):
            raise BenchmarkDataError(
                "verifier_case.label_quality must be gold or silver"
            )
        if not isinstance(self.split, str) or self.split not in _VALID_SPLITS:
            raise BenchmarkDataError(
                "verifier_case.split must be regression_seen, train, validation "
                "or promotion_holdout"
            )
        _require_id(self.split_group, "verifier_case.split_group")
        _require_string(self.notes, "verifier_case.notes", allow_empty=True)

    @property
    def is_promotion_holdout(self) -> bool:
        return self.split == "promotion_holdout"


@dataclass(frozen=True, slots=True)
class VideoVerifierDataset:
    schema_version: int
    dataset_id: str
    dataset_version: str
    description: str
    assets: tuple[BenchmarkAsset, ...]
    preparation_protocols: tuple[PreparedInputProtocol, ...]
    cases: tuple[VideoVerifierCase, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != VIDEO_VERIFIER_DATASET_SCHEMA_VERSION
        ):
            raise BenchmarkDataError(
                "video verifier schema_version must be the supported version "
                f"{VIDEO_VERIFIER_DATASET_SCHEMA_VERSION}"
            )
        _require_id(self.dataset_id, "video_verifier_dataset.dataset_id")
        _require_id(
            self.dataset_version,
            "video_verifier_dataset.dataset_version",
        )
        _require_string(
            self.description,
            "video_verifier_dataset.description",
            allow_empty=True,
        )
        assets = _validate_typed_tuple(
            self.assets,
            BenchmarkAsset,
            "video_verifier_dataset.assets",
        )
        protocols = _validate_typed_tuple(
            self.preparation_protocols,
            PreparedInputProtocol,
            "video_verifier_dataset.preparation_protocols",
        )
        cases = _validate_typed_tuple(
            self.cases,
            VideoVerifierCase,
            "video_verifier_dataset.cases",
        )
        if not assets:
            raise BenchmarkDataError("video_verifier_dataset.assets must not be empty")
        if not protocols:
            raise BenchmarkDataError(
                "video_verifier_dataset.preparation_protocols must not be empty"
            )
        if not cases:
            raise BenchmarkDataError("video_verifier_dataset.cases must not be empty")
        if len(assets) > MAX_DATASET_ASSETS:
            raise BenchmarkDataError(
                "video verifier dataset exceeds the benchmark asset count limit"
            )
        if len(protocols) > MAX_VIDEO_VERIFIER_PREPARATION_PROTOCOLS:
            raise BenchmarkDataError(
                "video verifier dataset exceeds the preparation protocol count limit"
            )
        if len(cases) > MAX_VIDEO_VERIFIER_CASES:
            raise BenchmarkDataError(
                "video verifier dataset exceeds the case count limit"
            )
        if sum(asset.byte_size for asset in assets) > MAX_DATASET_MEDIA_BYTES:
            raise BenchmarkDataError(
                "video verifier dataset exceeds the aggregate media byte limit"
            )
        if (
            sum(case.prepared_input_byte_size for case in cases)
            > MAX_DATASET_MEDIA_BYTES
        ):
            raise BenchmarkDataError(
                "video verifier dataset exceeds the aggregate prepared input byte "
                "limit"
            )
        _require_unique(
            tuple(asset.asset_id for asset in assets),
            "video verifier asset ids",
        )
        _require_unique(
            tuple(asset.sha256 for asset in assets),
            "video verifier asset SHA-256 identities",
        )
        _require_unique(
            tuple(protocol.protocol_id for protocol in protocols),
            "video verifier preparation protocol ids",
        )
        structural_protocols = tuple(
            _protocol_structural_signature(protocol) for protocol in protocols
        )
        if len(structural_protocols) != len(set(structural_protocols)):
            raise BenchmarkDataError(
                "video verifier preparation protocols must have unique structural "
                "identities"
            )
        _require_unique(
            tuple(case.case_id for case in cases),
            "video verifier case ids",
        )
        prepared_digests = tuple(case.prepared_input_sha256 for case in cases)
        if len(prepared_digests) != len(set(prepared_digests)):
            raise BenchmarkDataError(
                "video verifier prepared input SHA-256 identities must be unique"
            )
        self._validate_references(assets, protocols, cases)
        object.__setattr__(
            self,
            "assets",
            tuple(sorted(assets, key=lambda asset: asset.asset_id)),
        )
        object.__setattr__(
            self,
            "preparation_protocols",
            tuple(sorted(protocols, key=lambda protocol: protocol.protocol_id)),
        )
        object.__setattr__(
            self,
            "cases",
            tuple(sorted(cases, key=lambda case: case.case_id)),
        )

    @staticmethod
    def _validate_references(
        assets: tuple[object, ...],
        protocols: tuple[object, ...],
        cases: tuple[object, ...],
    ) -> None:
        assets_by_id = {
            asset.asset_id: asset
            for asset in assets
            if isinstance(asset, BenchmarkAsset)
        }
        protocols_by_id = {
            protocol.protocol_id: protocol
            for protocol in protocols
            if isinstance(protocol, PreparedInputProtocol)
        }
        typed_cases = tuple(
            case for case in cases if isinstance(case, VideoVerifierCase)
        )
        asset_assignment: dict[str, tuple[str, str]] = {}
        split_by_group: dict[str, str] = {}
        cases_by_asset: dict[str, list[VideoVerifierCase]] = {}
        used_asset_ids: set[str] = set()
        used_protocol_ids: set[str] = set()

        for case in typed_cases:
            asset_id = case.source_interval.asset_id
            asset = assets_by_id.get(asset_id)
            if asset is None:
                raise BenchmarkDataError(
                    f"verifier case {case.case_id!r} references unknown asset "
                    f"{asset_id!r}"
                )
            protocol = protocols_by_id.get(case.prepared_input_protocol_id)
            if protocol is None:
                raise BenchmarkDataError(
                    f"verifier case {case.case_id!r} references unknown preparation "
                    f"protocol {case.prepared_input_protocol_id!r}"
                )
            if protocol.input_kind != case.input_kind:
                raise BenchmarkDataError(
                    f"verifier case {case.case_id!r} input_kind does not match its "
                    "preparation protocol"
                )
            if case.source_interval.end_seconds > asset.duration_seconds:
                raise BenchmarkDataError(
                    f"verifier case {case.case_id!r} interval exceeds asset duration"
                )

            assignment = (case.split, case.split_group)
            previous_assignment = asset_assignment.setdefault(asset_id, assignment)
            if previous_assignment[0] != case.split:
                raise BenchmarkDataError(
                    f"asset {asset_id!r} has cross-split leakage"
                )
            if previous_assignment[1] != case.split_group:
                raise BenchmarkDataError(
                    f"asset {asset_id!r} must remain in one whole-asset split_group"
                )
            previous_group_split = split_by_group.setdefault(
                case.split_group,
                case.split,
            )
            if previous_group_split != case.split:
                raise BenchmarkDataError(
                    f"split_group {case.split_group!r} has cross-split leakage"
                )
            cases_by_asset.setdefault(asset_id, []).append(case)
            used_asset_ids.add(asset_id)
            used_protocol_ids.add(protocol.protocol_id)

        for asset_id, asset_cases in cases_by_asset.items():
            grouped: dict[tuple[float, float], list[VideoVerifierCase]] = {}
            for case in asset_cases:
                interval_key = (
                    case.source_interval.start_seconds,
                    case.source_interval.end_seconds,
                )
                grouped.setdefault(interval_key, []).append(case)

            ordered_groups = sorted(grouped.items())
            for _, comparison_cases in ordered_groups:
                if len(comparison_cases) > 1:
                    self_protocols = {
                        case.prepared_input_protocol_id for case in comparison_cases
                    }
                    if len(self_protocols) != len(comparison_cases):
                        raise BenchmarkDataError(
                            "exact-interval verifier comparisons require distinct "
                            "preparation protocols"
                        )
                    label_signatures = {
                        _case_label_signature(case) for case in comparison_cases
                    }
                    if len(label_signatures) != 1:
                        raise BenchmarkDataError(
                            "exact-interval verifier comparisons require identical "
                            "labels"
                        )

            for previous, current in zip(
                ordered_groups,
                ordered_groups[1:],
                strict=False,
            ):
                if current[0][0] < previous[0][1]:
                    raise BenchmarkDataError(
                        f"verifier source intervals overlap for asset {asset_id!r}: "
                        f"{previous[0]!r} and {current[0]!r}"
                    )

        unused_assets = set(assets_by_id) - used_asset_ids
        if unused_assets:
            raise BenchmarkDataError(
                "video verifier dataset contains unused asset: "
                + ", ".join(sorted(unused_assets))
            )
        unused_protocols = set(protocols_by_id) - used_protocol_ids
        if unused_protocols:
            raise BenchmarkDataError(
                "video verifier dataset contains unused preparation protocol: "
                + ", ".join(sorted(unused_protocols))
            )


def _case_label_signature(case: VideoVerifierCase) -> tuple[object, ...]:
    return (
        case.stratum,
        case.query,
        tuple((fact.fact_id, fact.expected) for fact in case.expected_facts),
        case.expected_jersey,
        case.label_quality,
        case.split,
        case.split_group,
    )


def _protocol_structural_signature(
    protocol: PreparedInputProtocol,
) -> tuple[object, ...]:
    return (
        protocol.input_kind,
        protocol.preparer.component_id,
        protocol.preparer.identity,
        tuple(
            (parameter.component_id, parameter.identity)
            for parameter in protocol.parameters
        ),
    )


def video_verifier_dataset_to_dict(
    dataset: VideoVerifierDataset,
    *,
    canonical: bool = False,
) -> JsonObject:
    if not isinstance(dataset, VideoVerifierDataset):
        raise BenchmarkDataError("dataset must be a VideoVerifierDataset value")
    assets = [_asset_to_dict(asset) for asset in dataset.assets]
    protocols = [
        _protocol_to_dict(protocol, canonical=canonical)
        for protocol in dataset.preparation_protocols
    ]
    cases = [_case_to_dict(case, canonical=canonical) for case in dataset.cases]
    if canonical:
        assets.sort(key=lambda item: str(item["asset_id"]))
        protocols.sort(key=lambda item: str(item["protocol_id"]))
        cases.sort(key=lambda item: str(item["case_id"]))
    return {
        "schema_version": dataset.schema_version,
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "description": dataset.description,
        "assets": assets,
        "preparation_protocols": protocols,
        "cases": cases,
    }


def video_verifier_dataset_from_dict(value: JsonObject) -> VideoVerifierDataset:
    expect_fields(
        value,
        {
            "schema_version",
            "dataset_id",
            "dataset_version",
            "description",
            "assets",
            "preparation_protocols",
            "cases",
        },
        "video verifier dataset",
    )
    return VideoVerifierDataset(
        schema_version=value["schema_version"],  # type: ignore[arg-type]
        dataset_id=value["dataset_id"],  # type: ignore[arg-type]
        dataset_version=value["dataset_version"],  # type: ignore[arg-type]
        description=value["description"],  # type: ignore[arg-type]
        assets=tuple(
            _asset_from_dict(expect_object(item, f"assets[{index}]"))
            for index, item in enumerate(expect_list(value["assets"], "assets"))
        ),
        preparation_protocols=tuple(
            _protocol_from_dict(
                expect_object(item, f"preparation_protocols[{index}]")
            )
            for index, item in enumerate(
                expect_list(
                    value["preparation_protocols"],
                    "preparation_protocols",
                )
            )
        ),
        cases=tuple(
            _case_from_dict(expect_object(item, f"cases[{index}]"))
            for index, item in enumerate(expect_list(value["cases"], "cases"))
        ),
    )


def video_verifier_dataset_from_json(
    data: bytes | str,
) -> VideoVerifierDataset:
    if isinstance(data, bytes):
        input_size = len(data)
    elif isinstance(data, str):
        if len(data) > MAX_VIDEO_VERIFIER_MANIFEST_BYTES:
            raise BenchmarkDataError(
                "video verifier manifest exceeds the JSON byte size limit"
            )
        try:
            input_size = len(data.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as exc:
            raise BenchmarkDataError(
                "video verifier manifest contains invalid Unicode text"
            ) from exc
    else:
        raise BenchmarkDataError("video verifier manifest must be bytes or a string")
    if input_size > MAX_VIDEO_VERIFIER_MANIFEST_BYTES:
        raise BenchmarkDataError(
            "video verifier manifest exceeds the JSON byte size limit"
        )
    return video_verifier_dataset_from_dict(
        parse_json_object(data, "video verifier dataset")
    )


def video_verifier_dataset_to_json_bytes(
    dataset: VideoVerifierDataset,
) -> bytes:
    return canonical_json_bytes(
        video_verifier_dataset_to_dict(dataset, canonical=True)
    )


def video_verifier_dataset_revision(dataset: VideoVerifierDataset) -> str:
    payload = json.dumps(
        video_verifier_dataset_to_dict(dataset, canonical=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _asset_to_dict(asset: BenchmarkAsset) -> JsonObject:
    return {
        "asset_id": asset.asset_id,
        "sha256": asset.sha256,
        "byte_size": asset.byte_size,
        "duration_seconds": asset.duration_seconds,
        "provenance": {
            "source": asset.provenance.source,
            "source_uri": asset.provenance.source_uri,
            "license_id": asset.provenance.license_id,
            "license_uri": asset.provenance.license_uri,
            "attribution": asset.provenance.attribution,
        },
    }


def _asset_from_dict(value: JsonObject) -> BenchmarkAsset:
    expect_fields(
        value,
        {"asset_id", "sha256", "byte_size", "duration_seconds", "provenance"},
        "video verifier asset",
    )
    provenance = expect_object(value["provenance"], "video verifier asset.provenance")
    expect_fields(
        provenance,
        {"source", "source_uri", "license_id", "license_uri", "attribution"},
        "video verifier asset.provenance",
    )
    return BenchmarkAsset(
        asset_id=value["asset_id"],  # type: ignore[arg-type]
        sha256=value["sha256"],  # type: ignore[arg-type]
        byte_size=value["byte_size"],  # type: ignore[arg-type]
        duration_seconds=value["duration_seconds"],  # type: ignore[arg-type]
        provenance=AssetProvenance(
            source=provenance["source"],  # type: ignore[arg-type]
            source_uri=provenance["source_uri"],  # type: ignore[arg-type]
            license_id=provenance["license_id"],  # type: ignore[arg-type]
            license_uri=provenance["license_uri"],  # type: ignore[arg-type]
            attribution=provenance["attribution"],  # type: ignore[arg-type]
        ),
    )


def _identity_to_dict(identity: ComponentIdentity) -> JsonObject:
    return {"component_id": identity.component_id, "identity": identity.identity}


def _identity_from_dict(value: JsonObject, context: str) -> ComponentIdentity:
    expect_fields(value, {"component_id", "identity"}, context)
    return ComponentIdentity(
        component_id=value["component_id"],  # type: ignore[arg-type]
        identity=value["identity"],  # type: ignore[arg-type]
    )


def _protocol_to_dict(
    protocol: PreparedInputProtocol,
    *,
    canonical: bool,
) -> JsonObject:
    parameters = [_identity_to_dict(item) for item in protocol.parameters]
    if canonical:
        parameters.sort(key=lambda item: str(item["component_id"]))
    return {
        "protocol_id": protocol.protocol_id,
        "input_kind": protocol.input_kind,
        "preparer": _identity_to_dict(protocol.preparer),
        "parameters": parameters,
    }


def _protocol_from_dict(value: JsonObject) -> PreparedInputProtocol:
    expect_fields(
        value,
        {"protocol_id", "input_kind", "preparer", "parameters"},
        "preparation protocol",
    )
    return PreparedInputProtocol(
        protocol_id=value["protocol_id"],  # type: ignore[arg-type]
        input_kind=value["input_kind"],  # type: ignore[arg-type]
        preparer=_identity_from_dict(
            expect_object(value["preparer"], "preparation protocol.preparer"),
            "preparation protocol.preparer",
        ),
        parameters=tuple(
            _identity_from_dict(
                expect_object(item, f"preparation protocol.parameters[{index}]"),
                f"preparation protocol.parameters[{index}]",
            )
            for index, item in enumerate(
                expect_list(value["parameters"], "preparation protocol.parameters")
            )
        ),
    )


def _case_to_dict(
    case: VideoVerifierCase,
    *,
    canonical: bool,
) -> JsonObject:
    facts = [
        {"fact_id": fact.fact_id, "expected": fact.expected}
        for fact in case.expected_facts
    ]
    if canonical:
        facts.sort(key=lambda item: str(item["fact_id"]))
    return {
        "case_id": case.case_id,
        "stratum": case.stratum,
        "input_kind": case.input_kind,
        "source_interval": {
            "asset_id": case.source_interval.asset_id,
            "start_seconds": case.source_interval.start_seconds,
            "end_seconds": case.source_interval.end_seconds,
        },
        "prepared_input_protocol_id": case.prepared_input_protocol_id,
        "prepared_input_sha256": case.prepared_input_sha256,
        "prepared_input_byte_size": case.prepared_input_byte_size,
        "query": case.query,
        "expected_facts": facts,
        "expected_jersey": case.expected_jersey,
        "label_quality": case.label_quality,
        "split": case.split,
        "split_group": case.split_group,
        "notes": case.notes,
    }


def _case_from_dict(value: JsonObject) -> VideoVerifierCase:
    expect_fields(
        value,
        {
            "case_id",
            "stratum",
            "input_kind",
            "source_interval",
            "prepared_input_protocol_id",
            "prepared_input_sha256",
            "prepared_input_byte_size",
            "query",
            "expected_facts",
            "expected_jersey",
            "label_quality",
            "split",
            "split_group",
            "notes",
        },
        "verifier case",
    )
    interval = expect_object(value["source_interval"], "verifier case.source_interval")
    expect_fields(
        interval,
        {"asset_id", "start_seconds", "end_seconds"},
        "verifier case.source_interval",
    )
    return VideoVerifierCase(
        case_id=value["case_id"],  # type: ignore[arg-type]
        stratum=value["stratum"],  # type: ignore[arg-type]
        input_kind=value["input_kind"],  # type: ignore[arg-type]
        source_interval=BenchmarkInterval(
            asset_id=interval["asset_id"],  # type: ignore[arg-type]
            start_seconds=interval["start_seconds"],  # type: ignore[arg-type]
            end_seconds=interval["end_seconds"],  # type: ignore[arg-type]
        ),
        prepared_input_protocol_id=value["prepared_input_protocol_id"],  # type: ignore[arg-type]
        prepared_input_sha256=value["prepared_input_sha256"],  # type: ignore[arg-type]
        prepared_input_byte_size=value["prepared_input_byte_size"],  # type: ignore[arg-type]
        query=value["query"],  # type: ignore[arg-type]
        expected_facts=tuple(
            _fact_from_dict(expect_object(item, f"expected_facts[{index}]"))
            for index, item in enumerate(
                expect_list(value["expected_facts"], "expected_facts")
            )
        ),
        expected_jersey=value["expected_jersey"],  # type: ignore[arg-type]
        label_quality=value["label_quality"],  # type: ignore[arg-type]
        split=value["split"],  # type: ignore[arg-type]
        split_group=value["split_group"],  # type: ignore[arg-type]
        notes=value["notes"],  # type: ignore[arg-type]
    )


def _fact_from_dict(value: JsonObject) -> VideoVerifierExpectedFact:
    expect_fields(value, {"fact_id", "expected"}, "expected fact")
    return VideoVerifierExpectedFact(
        fact_id=value["fact_id"],  # type: ignore[arg-type]
        expected=value["expected"],  # type: ignore[arg-type]
    )


__all__ = [
    "MAX_PREPARED_INPUT_BYTES",
    "MAX_VIDEO_VERIFIER_CASES",
    "MAX_VIDEO_VERIFIER_MANIFEST_BYTES",
    "MAX_VIDEO_VERIFIER_PREPARATION_PROTOCOLS",
    "VIDEO_VERIFIER_DATASET_SCHEMA_VERSION",
    "PreparedInputProtocol",
    "VerifierInputKind",
    "VerifierLabelQuality",
    "VerifierSplit",
    "VerifierStratum",
    "VideoVerifierCase",
    "VideoVerifierDataset",
    "VideoVerifierExpectedFact",
    "video_verifier_dataset_from_dict",
    "video_verifier_dataset_from_json",
    "video_verifier_dataset_revision",
    "video_verifier_dataset_to_dict",
    "video_verifier_dataset_to_json_bytes",
]
