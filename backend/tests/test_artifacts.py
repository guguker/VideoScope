from __future__ import annotations

from dataclasses import replace

import pytest

from videoscope.artifacts import (
    AssetRecord,
    SEGMENT_STAGE_KINDS,
    SegmentGeneration,
    StageKind,
    StageRun,
    StageSpecification,
    StageState,
    asset_id_for_sha256,
)


def test_asset_identity_is_stable_content_identity() -> None:
    digest = "a" * 64
    asset = AssetRecord.from_digest(
        sha256=digest,
        size_bytes=2048,
        created_at="2026-08-18T10:00:00+00:00",
    )

    assert asset.asset_id == f"sha256:{digest}"
    assert asset.asset_id == asset_id_for_sha256(digest)
    assert asset.sha256 == digest
    assert asset.size_bytes == 2048


@pytest.mark.parametrize(
    "changes",
    [
        {"asset_id": "asset-1"},
        {"sha256": "A" * 64},
        {"sha256": "not-a-digest"},
        {"size_bytes": 0},
        {"size_bytes": True},
        {"created_at": "2026-08-18"},
    ],
)
def test_asset_record_rejects_noncanonical_identity(changes) -> None:  # type: ignore[no-untyped-def]
    digest = "a" * 64
    values = {
        "asset_id": f"sha256:{digest}",
        "sha256": digest,
        "size_bytes": 10,
        "created_at": "2026-08-18T10:00:00+00:00",
    }
    values.update(changes)

    with pytest.raises(ValueError):
        AssetRecord(**values)


def test_stage_kind_and_state_contracts_are_explicit() -> None:
    assert {kind.value for kind in StageKind} == {
        "probe",
        "scenes",
        "speech",
        "ocr",
        "objects",
        "text_vectors",
        "visual_dense",
        "lighthouse",
    }
    assert {state.value for state in StageState} == {
        "queued",
        "running",
        "complete",
        "failed",
        "stale",
        "not_configured",
        "cancelled",
    }
    assert SEGMENT_STAGE_KINDS == {
        StageKind.SCENES,
        StageKind.SPEECH,
        StageKind.OCR,
        StageKind.OBJECTS,
    }


def test_segment_generation_allows_a_complete_empty_snapshot() -> None:
    generation = SegmentGeneration(
        generation_id="speech-generation-1",
        video_id="video-1",
        stage_kind=StageKind.SPEECH,
        specification_hash="a" * 64,
        source_sha256="b" * 64,
        run_id="speech-run-1",
        segment_count=0,
        completed_at="2026-08-18T10:00:00+00:00",
    )

    assert generation.stage_kind is StageKind.SPEECH
    assert generation.segment_count == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"generation_id": ""},
        {"generation_id": "../../unsafe"},
        {"video_id": ""},
        {"stage_kind": StageKind.PROBE},
        {"stage_kind": StageKind.VISUAL_DENSE},
        {"specification_hash": "A" * 64},
        {"source_sha256": "not-a-digest"},
        {"run_id": "unsafe/run"},
        {"segment_count": -1},
        {"segment_count": True},
        {"completed_at": "2026-08-18"},
    ],
)
def test_segment_generation_rejects_ambiguous_identity(changes) -> None:  # type: ignore[no-untyped-def]
    values = {
        "generation_id": "speech-generation-1",
        "video_id": "video-1",
        "stage_kind": StageKind.SPEECH,
        "specification_hash": "a" * 64,
        "source_sha256": "b" * 64,
        "run_id": "speech-run-1",
        "segment_count": 1,
        "completed_at": "2026-08-18T10:00:00+00:00",
    }
    values.update(changes)

    with pytest.raises(ValueError):
        SegmentGeneration(**values)


def test_stage_specification_hash_is_canonical_and_complete() -> None:
    first = StageSpecification(
        kind=StageKind.VISUAL_DENSE,
        schema_version=2,
        implementation_revision="visual-index-v3",
        model_identity="google/siglip@revision",
        parameters={
            "step_seconds": 1.0,
            "frame": {"width": 768, "format": "rgb", "channels": ["r", "g", "b"]},
        },
        dependencies={"numpy": "2.2.1", "transformers": "5.8.0"},
    )
    reordered = StageSpecification(
        kind=StageKind.VISUAL_DENSE,
        schema_version=2,
        implementation_revision="visual-index-v3",
        model_identity="google/siglip@revision",
        parameters={
            "frame": {"channels": ["r", "g", "b"], "format": "rgb", "width": 768},
            "step_seconds": 1.0,
        },
        dependencies={"transformers": "5.8.0", "numpy": "2.2.1"},
    )

    assert first.canonical_json == reordered.canonical_json
    assert first.specification_hash == reordered.specification_hash
    assert len(first.specification_hash) == 64
    assert StageSpecification.from_canonical_json(first.canonical_json) == first

    changed = replace(first, parameters={"step_seconds": 0.5})
    assert changed.specification_hash != first.specification_hash


def test_stage_specification_identity_cannot_mutate_after_hashing() -> None:
    specification = StageSpecification(
        kind=StageKind.VISUAL_DENSE,
        schema_version=1,
        implementation_revision="visual-v1",
        parameters={"sampling": {"step_seconds": 1.0}},
        dependencies={},
    )

    with pytest.raises(TypeError):
        specification.parameters["sampling"] = {"step_seconds": 2.0}  # type: ignore[index]
    with pytest.raises(TypeError):
        specification.parameters["sampling"]["step_seconds"] = 2.0  # type: ignore[index]


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 0},
        {"schema_version": True},
        {"kind": "unknown"},
        {"implementation_revision": ""},
        {"model_identity": ""},
        {"parameters": {"step_seconds": float("nan")}},
        {"parameters": {1: "non-string-key"}},
        {"parameters": []},
        {"parameters": {"unsupported": object()}},
        {"dependencies": {"numpy": ""}},
        {"dependencies": []},
    ],
)
def test_stage_specification_rejects_ambiguous_identity(
    changes,
) -> None:  # type: ignore[no-untyped-def]
    values = {
        "kind": StageKind.SPEECH,
        "schema_version": 1,
        "implementation_revision": "speech-v1",
        "model_identity": "whisper@revision",
        "parameters": {},
        "dependencies": {},
    }
    values.update(changes)

    with pytest.raises(ValueError):
        StageSpecification(**values)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not-json",
        "[]",
        '{"kind":"speech"}',
        '{"dependencies":{},"implementation_revision":"speech-v1","kind":"speech",'
        '"model_identity":null,"parameters":{"temperature":NaN},"schema_version":1}',
    ],
)
def test_stage_specification_rejects_noncanonical_persisted_data(
    value,
) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        StageSpecification.from_canonical_json(value)


def test_stage_specification_normalizes_deeply_nested_persisted_data() -> None:
    nested_parameters = '{"nested":' * 1_200 + "null" + "}" * 1_200
    persisted = (
        '{"dependencies":{},"implementation_revision":"speech-v1",'
        '"kind":"speech","model_identity":null,"parameters":'
        + nested_parameters
        + ',"schema_version":1}'
    )

    with pytest.raises(ValueError, match="invalid canonical stage specification"):
        StageSpecification.from_canonical_json(persisted)


def test_stage_run_rejects_raw_error_details() -> None:
    with pytest.raises(ValueError, match="error code"):
        StageRun(
            run_id="run-1",
            video_id="video-1",
            stage_kind=StageKind.OCR,
            state=StageState.FAILED,
            specification_hash="a" * 64,
            source_sha256="b" * 64,
            attempt=1,
            output_generation=None,
            error_code="Could not read /Users/person/private.mp4",
            retry_of_run_id=None,
            created_at="2026-08-18T10:00:00+00:00",
            started_at="2026-08-18T10:00:01+00:00",
            finished_at="2026-08-18T10:00:02+00:00",
            updated_at="2026-08-18T10:00:02+00:00",
        )


def test_complete_stage_run_requires_an_output_generation() -> None:
    with pytest.raises(ValueError, match="output generation"):
        StageRun(
            run_id="run-1",
            video_id="video-1",
            stage_kind=StageKind.SPEECH,
            state=StageState.COMPLETE,
            specification_hash="a" * 64,
            source_sha256="b" * 64,
            attempt=1,
            output_generation=None,
            error_code=None,
            retry_of_run_id=None,
            created_at="2026-08-18T10:00:00+00:00",
            started_at="2026-08-18T10:00:01+00:00",
            finished_at="2026-08-18T10:00:02+00:00",
            updated_at="2026-08-18T10:00:02+00:00",
        )


def test_stale_stage_run_requires_the_generation_that_became_stale() -> None:
    with pytest.raises(ValueError, match="output generation"):
        StageRun(
            run_id="run-1",
            video_id="video-1",
            stage_kind=StageKind.VISUAL_DENSE,
            state=StageState.STALE,
            specification_hash="a" * 64,
            source_sha256="b" * 64,
            attempt=1,
            output_generation=None,
            error_code=None,
            retry_of_run_id=None,
            created_at="2026-08-18T10:00:00+00:00",
            started_at="2026-08-18T10:00:01+00:00",
            finished_at="2026-08-18T10:00:02+00:00",
            updated_at="2026-08-18T10:00:03+00:00",
        )


def test_stage_run_rejects_timestamps_that_claim_update_before_outcome() -> None:
    with pytest.raises(ValueError, match="updated timestamp"):
        StageRun(
            run_id="run-1",
            video_id="video-1",
            stage_kind=StageKind.SPEECH,
            state=StageState.COMPLETE,
            specification_hash="a" * 64,
            source_sha256="b" * 64,
            attempt=1,
            output_generation="generation-1",
            error_code=None,
            retry_of_run_id=None,
            created_at="2026-08-18T10:00:00+00:00",
            started_at="2026-08-18T10:00:01+00:00",
            finished_at="2026-08-18T10:00:03+00:00",
            updated_at="2026-08-18T10:00:02+00:00",
        )


def test_stale_stage_run_must_have_started_as_a_real_completed_run() -> None:
    with pytest.raises(ValueError, match="start timestamp"):
        StageRun(
            run_id="run-1",
            video_id="video-1",
            stage_kind=StageKind.VISUAL_DENSE,
            state=StageState.STALE,
            specification_hash="a" * 64,
            source_sha256="b" * 64,
            attempt=1,
            output_generation="generation-1",
            error_code=None,
            retry_of_run_id=None,
            created_at="2026-08-18T10:00:00+00:00",
            started_at=None,
            finished_at="2026-08-18T10:00:02+00:00",
            updated_at="2026-08-18T10:00:03+00:00",
        )
