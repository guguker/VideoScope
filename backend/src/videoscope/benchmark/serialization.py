from __future__ import annotations

from hashlib import sha256
import json
from typing import Callable

from .schema import (
    MEASUREMENT_PROTOCOL_COMPONENT_ID,
    LEGACY_UNMEASURED_PROTOCOL_IDENTITY,
    RUN_SCHEMA_VERSION,
    AssetProvenance,
    BenchmarkAsset,
    BenchmarkCaseOutcome,
    BenchmarkDataError,
    BenchmarkDataset,
    BenchmarkInterval,
    BenchmarkMeasurementEvidence,
    BenchmarkRunManifest,
    BenchmarkResultEvidence,
    BenchmarkStorageSnapshot,
    ComponentIdentity,
    CriticalSliceLabels,
    HardNegative,
    HardwareProfile,
    MetricValue,
    QueryCase,
    _run_status_for_outcomes,
)


JsonObject = dict[str, object]


def parse_json_object(data: bytes | str, context: str) -> JsonObject:
    try:
        value = json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except BenchmarkDataError:
        raise
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise BenchmarkDataError(f"{context} is not valid JSON") from exc
    return expect_object(value, context)


def _unique_object(pairs: list[tuple[str, object]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkDataError(f"JSON object contains duplicate field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise BenchmarkDataError(f"JSON number {value} is not finite")


def expect_object(value: object, context: str) -> JsonObject:
    if type(value) is not dict:
        raise BenchmarkDataError(f"{context} must be a JSON object")
    return value  # type: ignore[return-value]


def expect_list(value: object, context: str) -> list[object]:
    if type(value) is not list:
        raise BenchmarkDataError(f"{context} must be a JSON array")
    return value


def expect_fields(value: JsonObject, fields: set[str], context: str) -> None:
    present = set(value)
    missing = fields - present
    unexpected = present - fields
    if missing:
        raise BenchmarkDataError(f"{context} is missing fields: {', '.join(sorted(missing))}")
    if unexpected:
        raise BenchmarkDataError(
            f"{context} has unexpected fields: {', '.join(sorted(unexpected))}"
        )


def dataset_to_dict(dataset: BenchmarkDataset, *, canonical: bool = False) -> JsonObject:
    assets = [_asset_to_dict(asset) for asset in dataset.assets]
    cases = [_case_to_dict(case, canonical=canonical) for case in dataset.cases]
    if canonical:
        assets.sort(key=lambda item: _canonical_sort_key(item, "asset_id"))
        cases.sort(key=lambda item: _canonical_sort_key(item, "case_id"))
    return {
        "schema_version": dataset.schema_version,
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "description": dataset.description,
        "assets": assets,
        "cases": cases,
    }


def dataset_from_dict(value: JsonObject) -> BenchmarkDataset:
    expect_fields(
        value,
        {"schema_version", "dataset_id", "dataset_version", "description", "assets", "cases"},
        "dataset",
    )
    assets = tuple(
        _asset_from_dict(expect_object(item, f"assets[{index}]"))
        for index, item in enumerate(expect_list(value["assets"], "assets"))
    )
    cases = tuple(
        _case_from_dict(expect_object(item, f"cases[{index}]"))
        for index, item in enumerate(expect_list(value["cases"], "cases"))
    )
    return BenchmarkDataset(
        schema_version=value["schema_version"],  # type: ignore[arg-type]
        dataset_id=value["dataset_id"],  # type: ignore[arg-type]
        dataset_version=value["dataset_version"],  # type: ignore[arg-type]
        description=value["description"],  # type: ignore[arg-type]
        assets=assets,
        cases=cases,
    )


def dataset_revision(dataset: BenchmarkDataset) -> str:
    payload = json.dumps(
        dataset_to_dict(dataset, canonical=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def run_to_dict(run: BenchmarkRunManifest) -> JsonObject:
    return {
        "schema_version": run.schema_version,
        "run_id": run.run_id,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "run_status": run.run_status,
        "code_sha": run.code_sha,
        "dataset_revision": run.dataset_revision,
        "model_identities": [_identity_to_dict(item) for item in run.model_identities],
        "index_identities": [_identity_to_dict(item) for item in run.index_identities],
        "config_identities": [_identity_to_dict(item) for item in run.config_identities],
        "hardware": {
            "operating_system": run.hardware.operating_system,
            "architecture": run.hardware.architecture,
            "processor": run.hardware.processor,
            "memory_bytes": run.hardware.memory_bytes,
            "accelerator": run.hardware.accelerator,
        },
        "execution_mode": run.execution_mode,
        "quality_metrics": [
            _metric_to_dict(metric) for metric in run.quality_metrics
        ],
        "system_metrics": [
            _metric_to_dict(metric) for metric in run.system_metrics
        ],
        "measurement_protocol": _identity_to_dict(run.measurement_protocol),
        "measurement_status": run.measurement_status,
        "measurement_started_at": run.measurement_started_at,
        "measurement_finished_at": run.measurement_finished_at,
        "measurement_evidence_status": run.measurement_evidence_status,
        "measurement_evidence": (
            _measurement_evidence_to_dict(run.measurement_evidence)
            if run.measurement_evidence is not None
            else None
        ),
        "case_outcomes": [_outcome_to_dict(outcome) for outcome in run.case_outcomes],
    }


def run_from_dict(value: JsonObject) -> BenchmarkRunManifest:
    schema_version = value.get("schema_version")
    if type(schema_version) is not int:
        raise BenchmarkDataError("run manifest schema_version must be an integer")
    if schema_version == 1:
        return _legacy_run_from_dict(value)
    if schema_version == 2:
        return _v2_run_from_dict(value)
    if schema_version != RUN_SCHEMA_VERSION:
        raise BenchmarkDataError(
            f"run manifest schema_version must be 1, 2 or {RUN_SCHEMA_VERSION}"
        )
    fields = {
        "schema_version",
        "run_id",
        "created_at",
        "started_at",
        "finished_at",
        "run_status",
        "code_sha",
        "dataset_revision",
        "model_identities",
        "index_identities",
        "config_identities",
        "hardware",
        "execution_mode",
        "quality_metrics",
        "system_metrics",
        "measurement_protocol",
        "measurement_status",
        "measurement_started_at",
        "measurement_finished_at",
        "measurement_evidence_status",
        "measurement_evidence",
        "case_outcomes",
    }
    expect_fields(value, fields, "run manifest")
    outcomes = _parse_tuple(
        value["case_outcomes"],
        "case_outcomes",
        _outcome_from_dict,
    )
    return BenchmarkRunManifest(
        schema_version=RUN_SCHEMA_VERSION,
        run_id=value["run_id"],  # type: ignore[arg-type]
        created_at=value["created_at"],  # type: ignore[arg-type]
        started_at=value["started_at"],  # type: ignore[arg-type]
        finished_at=value["finished_at"],  # type: ignore[arg-type]
        run_status=value["run_status"],  # type: ignore[arg-type]
        code_sha=value["code_sha"],  # type: ignore[arg-type]
        dataset_revision=value["dataset_revision"],  # type: ignore[arg-type]
        model_identities=_parse_tuple(
            value["model_identities"],
            "model_identities",
            _identity_from_dict,
        ),
        index_identities=_parse_tuple(
            value["index_identities"],
            "index_identities",
            _identity_from_dict,
        ),
        config_identities=_parse_tuple(
            value["config_identities"],
            "config_identities",
            _identity_from_dict,
        ),
        hardware=_hardware_from_dict(
            expect_object(value["hardware"], "hardware")
        ),
        execution_mode=value["execution_mode"],  # type: ignore[arg-type]
        quality_metrics=_parse_tuple(
            value["quality_metrics"],
            "quality_metrics",
            _metric_from_dict,
        ),
        system_metrics=_parse_tuple(
            value["system_metrics"],
            "system_metrics",
            _metric_from_dict,
        ),
        measurement_protocol=_identity_from_dict(
            expect_object(value["measurement_protocol"], "measurement_protocol")
        ),
        measurement_status=value["measurement_status"],  # type: ignore[arg-type]
        measurement_started_at=value["measurement_started_at"],  # type: ignore[arg-type]
        measurement_finished_at=value["measurement_finished_at"],  # type: ignore[arg-type]
        case_outcomes=outcomes,
        measurement_evidence_status=value["measurement_evidence_status"],  # type: ignore[arg-type]
        measurement_evidence=(
            None
            if value["measurement_evidence"] is None
            else _measurement_evidence_from_dict(
                expect_object(
                    value["measurement_evidence"],
                    "measurement_evidence",
                )
            )
        ),
    )


def _v2_run_from_dict(value: JsonObject) -> BenchmarkRunManifest:
    """Read the exact aggregate-only v2 shape without inventing raw evidence."""

    fields = {
        "schema_version",
        "run_id",
        "created_at",
        "started_at",
        "finished_at",
        "run_status",
        "code_sha",
        "dataset_revision",
        "model_identities",
        "index_identities",
        "config_identities",
        "hardware",
        "execution_mode",
        "quality_metrics",
        "system_metrics",
        "measurement_protocol",
        "measurement_status",
        "measurement_started_at",
        "measurement_finished_at",
        "case_outcomes",
    }
    expect_fields(value, fields, "run manifest")
    outcomes = _parse_tuple(
        value["case_outcomes"],
        "case_outcomes",
        _outcome_from_dict,
    )
    measurement_status = value["measurement_status"]
    return BenchmarkRunManifest(
        schema_version=RUN_SCHEMA_VERSION,
        run_id=value["run_id"],  # type: ignore[arg-type]
        created_at=value["created_at"],  # type: ignore[arg-type]
        started_at=value["started_at"],  # type: ignore[arg-type]
        finished_at=value["finished_at"],  # type: ignore[arg-type]
        run_status=value["run_status"],  # type: ignore[arg-type]
        code_sha=value["code_sha"],  # type: ignore[arg-type]
        dataset_revision=value["dataset_revision"],  # type: ignore[arg-type]
        model_identities=_parse_tuple(
            value["model_identities"],
            "model_identities",
            _identity_from_dict,
        ),
        index_identities=_parse_tuple(
            value["index_identities"],
            "index_identities",
            _identity_from_dict,
        ),
        config_identities=_parse_tuple(
            value["config_identities"],
            "config_identities",
            _identity_from_dict,
        ),
        hardware=_hardware_from_dict(expect_object(value["hardware"], "hardware")),
        execution_mode=value["execution_mode"],  # type: ignore[arg-type]
        quality_metrics=_parse_tuple(
            value["quality_metrics"],
            "quality_metrics",
            _metric_from_dict,
        ),
        system_metrics=_parse_tuple(
            value["system_metrics"],
            "system_metrics",
            _metric_from_dict,
        ),
        measurement_protocol=_identity_from_dict(
            expect_object(value["measurement_protocol"], "measurement_protocol")
        ),
        measurement_status=measurement_status,  # type: ignore[arg-type]
        measurement_started_at=value["measurement_started_at"],  # type: ignore[arg-type]
        measurement_finished_at=value["measurement_finished_at"],  # type: ignore[arg-type]
        case_outcomes=outcomes,
        measurement_evidence_status=(
            "legacy_unavailable"
            if measurement_status == "complete"
            else "not_applicable"
        ),
        measurement_evidence=None,
    )


def _legacy_run_from_dict(value: JsonObject) -> BenchmarkRunManifest:
    """Read the exact released v1 shape without rewriting immutable bytes.

    A v1 manifest has no explicit run/measurement timeline.  It is therefore
    represented in memory as a conservative v3 ``not_measured`` run.  New
    writes always use v3; callers must not treat this compatibility read as an
    evidence-preserving upgrade of the old methodology.
    """

    fields = {
        "schema_version",
        "run_id",
        "created_at",
        "code_sha",
        "dataset_revision",
        "model_identities",
        "index_identities",
        "config_identities",
        "hardware",
        "execution_mode",
        "metrics",
        "case_outcomes",
    }
    expect_fields(value, fields, "run manifest")
    outcomes = _parse_tuple(
        value["case_outcomes"],
        "case_outcomes",
        _outcome_from_dict,
    )
    created_at = value["created_at"]
    return BenchmarkRunManifest(
        schema_version=RUN_SCHEMA_VERSION,
        run_id=value["run_id"],  # type: ignore[arg-type]
        created_at=created_at,  # type: ignore[arg-type]
        started_at=created_at,  # type: ignore[arg-type]
        finished_at=created_at,  # type: ignore[arg-type]
        run_status=_run_status_for_outcomes(outcomes),
        code_sha=value["code_sha"],  # type: ignore[arg-type]
        dataset_revision=value["dataset_revision"],  # type: ignore[arg-type]
        model_identities=_parse_tuple(
            value["model_identities"],
            "model_identities",
            _identity_from_dict,
        ),
        index_identities=_parse_tuple(
            value["index_identities"],
            "index_identities",
            _identity_from_dict,
        ),
        config_identities=_parse_tuple(
            value["config_identities"],
            "config_identities",
            _identity_from_dict,
        ),
        hardware=_hardware_from_dict(
            expect_object(value["hardware"], "hardware")
        ),
        execution_mode=value["execution_mode"],  # type: ignore[arg-type]
        quality_metrics=_parse_tuple(
            value["metrics"],
            "metrics",
            _metric_from_dict,
        ),
        system_metrics=(),
        measurement_protocol=ComponentIdentity(
            MEASUREMENT_PROTOCOL_COMPONENT_ID,
            LEGACY_UNMEASURED_PROTOCOL_IDENTITY,
        ),
        measurement_status="not_measured",
        measurement_started_at=None,
        measurement_finished_at=None,
        case_outcomes=outcomes,
        measurement_evidence_status="not_applicable",
        measurement_evidence=None,
    )


def _hardware_from_dict(value: JsonObject) -> HardwareProfile:
    expect_fields(
        value,
        {
            "operating_system",
            "architecture",
            "processor",
            "memory_bytes",
            "accelerator",
        },
        "hardware",
    )
    return HardwareProfile(
        operating_system=value["operating_system"],  # type: ignore[arg-type]
        architecture=value["architecture"],  # type: ignore[arg-type]
        processor=value["processor"],  # type: ignore[arg-type]
        memory_bytes=value["memory_bytes"],  # type: ignore[arg-type]
        accelerator=value["accelerator"],  # type: ignore[arg-type]
    )


def canonical_json_bytes(value: JsonObject) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


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
        "asset",
    )
    provenance = expect_object(value["provenance"], "asset.provenance")
    expect_fields(
        provenance,
        {"source", "source_uri", "license_id", "license_uri", "attribution"},
        "asset.provenance",
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


def _case_to_dict(case: QueryCase, *, canonical: bool) -> JsonObject:
    asset_ids = list(case.asset_ids)
    modalities = list(case.modalities)
    relevant = [_interval_to_dict(interval) for interval in case.relevant_intervals]
    negatives = [_negative_to_dict(negative) for negative in case.hard_negatives]
    if canonical:
        asset_ids.sort()
        modalities.sort()
        relevant.sort(key=_canonical_json_key)
        negatives.sort(key=_canonical_json_key)
    result: JsonObject = {
        "case_id": case.case_id,
        "query": case.query,
        "asset_ids": asset_ids,
        "domain": case.domain,
        "modalities": modalities,
        "label_quality": case.label_quality,
        "split_group": case.split_group,
        "relevant_intervals": relevant,
        "hard_negatives": negatives,
        "notes": case.notes,
    }
    if case.critical_slices is not None:
        result["critical_slices"] = _critical_slice_labels_to_dict(
            case.critical_slices,
            canonical=canonical,
        )
    return result


def _case_from_dict(value: JsonObject) -> QueryCase:
    fields = {
        "case_id",
        "query",
        "asset_ids",
        "domain",
        "modalities",
        "label_quality",
        "split_group",
        "relevant_intervals",
        "hard_negatives",
        "notes",
    }
    if "critical_slices" in value:
        fields.add("critical_slices")
    expect_fields(value, fields, "case")
    asset_ids = tuple(expect_list(value["asset_ids"], "case.asset_ids"))
    modalities = tuple(expect_list(value["modalities"], "case.modalities"))
    relevant = _parse_tuple(
        value["relevant_intervals"],
        "case.relevant_intervals",
        _interval_from_dict,
    )
    negatives = _parse_tuple(
        value["hard_negatives"],
        "case.hard_negatives",
        _negative_from_dict,
    )
    return QueryCase(
        case_id=value["case_id"],  # type: ignore[arg-type]
        query=value["query"],  # type: ignore[arg-type]
        asset_ids=asset_ids,  # type: ignore[arg-type]
        domain=value["domain"],  # type: ignore[arg-type]
        modalities=modalities,  # type: ignore[arg-type]
        label_quality=value["label_quality"],  # type: ignore[arg-type]
        split_group=value["split_group"],  # type: ignore[arg-type]
        relevant_intervals=relevant,
        hard_negatives=negatives,
        notes=value["notes"],  # type: ignore[arg-type]
        critical_slices=(
            _critical_slice_labels_from_dict(
                expect_object(value["critical_slices"], "case.critical_slices")
            )
            if "critical_slices" in value
            else None
        ),
    )


def _critical_slice_labels_to_dict(
    labels: CriticalSliceLabels,
    *,
    canonical: bool,
) -> JsonObject:
    event_class = list(labels.event_class)
    capture_condition = list(labels.capture_condition)
    distribution_shift = list(labels.distribution_shift)
    if canonical:
        event_class.sort()
        capture_condition.sort()
        distribution_shift.sort()
    return {
        "schema_version": labels.schema_version,
        "event_class": event_class,
        "capture_condition": capture_condition,
        "distribution_shift": distribution_shift,
    }


def _critical_slice_labels_from_dict(value: JsonObject) -> CriticalSliceLabels:
    expect_fields(
        value,
        {
            "schema_version",
            "event_class",
            "capture_condition",
            "distribution_shift",
        },
        "case.critical_slices",
    )
    return CriticalSliceLabels(
        schema_version=value["schema_version"],  # type: ignore[arg-type]
        event_class=tuple(
            expect_list(
                value["event_class"],
                "case.critical_slices.event_class",
            )
        ),  # type: ignore[arg-type]
        capture_condition=tuple(
            expect_list(
                value["capture_condition"],
                "case.critical_slices.capture_condition",
            )
        ),  # type: ignore[arg-type]
        distribution_shift=tuple(
            expect_list(
                value["distribution_shift"],
                "case.critical_slices.distribution_shift",
            )
        ),  # type: ignore[arg-type]
    )


def _interval_to_dict(interval: BenchmarkInterval) -> JsonObject:
    return {
        "asset_id": interval.asset_id,
        "start_seconds": interval.start_seconds,
        "end_seconds": interval.end_seconds,
    }


def _interval_from_dict(value: JsonObject) -> BenchmarkInterval:
    expect_fields(value, {"asset_id", "start_seconds", "end_seconds"}, "interval")
    return BenchmarkInterval(
        asset_id=value["asset_id"],  # type: ignore[arg-type]
        start_seconds=value["start_seconds"],  # type: ignore[arg-type]
        end_seconds=value["end_seconds"],  # type: ignore[arg-type]
    )


def _negative_to_dict(negative: HardNegative) -> JsonObject:
    return {
        "asset_id": negative.asset_id,
        "start_seconds": negative.start_seconds,
        "end_seconds": negative.end_seconds,
        "reason": negative.reason,
    }


def _negative_from_dict(value: JsonObject) -> HardNegative:
    expect_fields(
        value,
        {"asset_id", "start_seconds", "end_seconds", "reason"},
        "hard negative",
    )
    return HardNegative(
        asset_id=value["asset_id"],  # type: ignore[arg-type]
        start_seconds=value["start_seconds"],  # type: ignore[arg-type]
        end_seconds=value["end_seconds"],  # type: ignore[arg-type]
        reason=value["reason"],  # type: ignore[arg-type]
    )


def _identity_to_dict(identity: ComponentIdentity) -> JsonObject:
    return {"component_id": identity.component_id, "identity": identity.identity}


def _identity_from_dict(value: JsonObject) -> ComponentIdentity:
    expect_fields(value, {"component_id", "identity"}, "component identity")
    return ComponentIdentity(
        component_id=value["component_id"],  # type: ignore[arg-type]
        identity=value["identity"],  # type: ignore[arg-type]
    )


def _measurement_evidence_to_dict(
    evidence: BenchmarkMeasurementEvidence,
) -> JsonObject:
    return {
        "schema_version": evidence.schema_version,
        "rss_samples_bytes": list(evidence.rss_samples_bytes),
        "storage_before": [
            _storage_snapshot_to_dict(item) for item in evidence.storage_before
        ],
        "storage_after": [
            _storage_snapshot_to_dict(item) for item in evidence.storage_after
        ],
        "metal_telemetry_status": evidence.metal_telemetry_status,
    }


def _measurement_evidence_from_dict(
    value: JsonObject,
) -> BenchmarkMeasurementEvidence:
    expect_fields(
        value,
        {
            "schema_version",
            "rss_samples_bytes",
            "storage_before",
            "storage_after",
            "metal_telemetry_status",
        },
        "measurement evidence",
    )
    return BenchmarkMeasurementEvidence(
        schema_version=value["schema_version"],  # type: ignore[arg-type]
        rss_samples_bytes=tuple(
            expect_list(
                value["rss_samples_bytes"],
                "measurement evidence RSS samples",
            )
        ),  # type: ignore[arg-type]
        storage_before=_parse_tuple(
            value["storage_before"],
            "measurement evidence storage_before",
            _storage_snapshot_from_dict,
        ),
        storage_after=_parse_tuple(
            value["storage_after"],
            "measurement evidence storage_after",
            _storage_snapshot_from_dict,
        ),
        metal_telemetry_status=value["metal_telemetry_status"],  # type: ignore[arg-type]
    )


def _storage_snapshot_to_dict(snapshot: BenchmarkStorageSnapshot) -> JsonObject:
    return {
        "root_id": snapshot.root_id,
        "purpose": snapshot.purpose,
        "file_count": snapshot.file_count,
        "directory_count": snapshot.directory_count,
        "logical_bytes": snapshot.logical_bytes,
        "allocated_bytes": snapshot.allocated_bytes,
        "tree_digest": snapshot.tree_digest,
    }


def _storage_snapshot_from_dict(value: JsonObject) -> BenchmarkStorageSnapshot:
    expect_fields(
        value,
        {
            "root_id",
            "purpose",
            "file_count",
            "directory_count",
            "logical_bytes",
            "allocated_bytes",
            "tree_digest",
        },
        "measurement storage snapshot",
    )
    return BenchmarkStorageSnapshot(
        root_id=value["root_id"],  # type: ignore[arg-type]
        purpose=value["purpose"],  # type: ignore[arg-type]
        file_count=value["file_count"],  # type: ignore[arg-type]
        directory_count=value["directory_count"],  # type: ignore[arg-type]
        logical_bytes=value["logical_bytes"],  # type: ignore[arg-type]
        allocated_bytes=value["allocated_bytes"],  # type: ignore[arg-type]
        tree_digest=value["tree_digest"],  # type: ignore[arg-type]
    )


def _metric_to_dict(metric: MetricValue) -> JsonObject:
    return {"name": metric.name, "value": metric.value, "unit": metric.unit}


def _metric_from_dict(value: JsonObject) -> MetricValue:
    expect_fields(value, {"name", "value", "unit"}, "metric")
    return MetricValue(
        name=value["name"],  # type: ignore[arg-type]
        value=value["value"],  # type: ignore[arg-type]
        unit=value["unit"],  # type: ignore[arg-type]
    )


def _outcome_to_dict(outcome: BenchmarkCaseOutcome) -> JsonObject:
    return {
        "case_id": outcome.case_id,
        "status": outcome.status,
        "latency_ms": outcome.latency_ms,
        "result_count": outcome.result_count,
        "metrics": [_metric_to_dict(metric) for metric in outcome.metrics],
        "result_evidence": [
            _result_evidence_to_dict(item) for item in outcome.result_evidence
        ],
        "diagnostic_code": outcome.diagnostic_code,
    }


def _outcome_from_dict(value: JsonObject) -> BenchmarkCaseOutcome:
    expect_fields(
        value,
        {
            "case_id",
            "status",
            "latency_ms",
            "result_count",
            "metrics",
            "result_evidence",
            "diagnostic_code",
        },
        "case outcome",
    )
    return BenchmarkCaseOutcome(
        case_id=value["case_id"],  # type: ignore[arg-type]
        status=value["status"],  # type: ignore[arg-type]
        latency_ms=value["latency_ms"],  # type: ignore[arg-type]
        result_count=value["result_count"],  # type: ignore[arg-type]
        metrics=_parse_tuple(value["metrics"], "case outcome metrics", _metric_from_dict),
        result_evidence=_parse_tuple(
            value["result_evidence"],
            "case outcome result_evidence",
            _result_evidence_from_dict,
        ),
        diagnostic_code=value["diagnostic_code"],  # type: ignore[arg-type]
    )


def _result_evidence_to_dict(evidence: BenchmarkResultEvidence) -> JsonObject:
    return {
        "rank": evidence.rank,
        "asset_id": evidence.asset_id,
        "start_seconds": evidence.start_seconds,
        "end_seconds": evidence.end_seconds,
        "score": evidence.score,
    }


def _result_evidence_from_dict(value: JsonObject) -> BenchmarkResultEvidence:
    expect_fields(
        value,
        {"rank", "asset_id", "start_seconds", "end_seconds", "score"},
        "result evidence",
    )
    return BenchmarkResultEvidence(
        rank=value["rank"],  # type: ignore[arg-type]
        asset_id=value["asset_id"],  # type: ignore[arg-type]
        start_seconds=value["start_seconds"],  # type: ignore[arg-type]
        end_seconds=value["end_seconds"],  # type: ignore[arg-type]
        score=value["score"],  # type: ignore[arg-type]
    )


def _parse_tuple(
    value: object,
    context: str,
    parser: Callable[[JsonObject], object],
) -> tuple:
    return tuple(
        parser(expect_object(item, f"{context}[{index}]"))
        for index, item in enumerate(expect_list(value, context))
    )


def _canonical_sort_key(value: JsonObject, identity_field: str) -> tuple[str, str]:
    return str(value[identity_field]), _canonical_json_key(value)


def _canonical_json_key(value: JsonObject) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
