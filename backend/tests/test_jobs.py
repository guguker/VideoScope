from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from videoscope.artifacts import IndexingSpecifications, StageKind, StageSpecification
from videoscope.jobs import (
    JobFenceError,
    JobKind,
    JobState,
    PriorVideoState,
    VideoIndexIntent,
    VideoIndexJob,
    VideoIndexPlanSnapshot,
    advance_job,
    cancel_job,
    complete_job,
    fail_job,
    request_job_cancellation,
    retry_job,
    start_job,
    video_index_idempotency_hash,
)
from videoscope.providers.whisper import WhisperPromptSnapshot


_EXECUTOR_IDENTITY = "sha256:" + "e" * 64


CREATED_AT = "2026-08-19T08:00:00+00:00"
STARTED_AT = "2026-08-19T08:01:00+00:00"
ADVANCED_AT = "2026-08-19T08:02:00+00:00"
CANCEL_REQUESTED_AT = "2026-08-19T08:03:00+00:00"
FINISHED_AT = "2026-08-19T08:04:00+00:00"
TOKEN = "worker-token-0123456789abcdef"


def _stage_specification(
    kind: StageKind,
    *,
    parameters: dict[str, object] | None = None,
) -> StageSpecification:
    return StageSpecification(
        kind=kind,
        schema_version=1,
        implementation_revision=f"videoscope.{kind.value}.v1",
        parameters=parameters or {},
        model_identity=None,
        dependencies={"videoscope": "0.1.0"},
    )


def _plan_snapshot(**changes: object) -> VideoIndexPlanSnapshot:
    prompt = WhisperPromptSnapshot(
        effective_prompt="Мозгов, контратака",
        effective_prompt_sha256=hashlib.sha256(
            "Мозгов, контратака".encode("utf-8")
        ).hexdigest(),
        glossary_state="ready",
    )
    specifications = IndexingSpecifications(
        scenes=_stage_specification(StageKind.SCENES),
        speech=_stage_specification(
            StageKind.SPEECH,
            parameters={
                "effective_prompt_sha256": prompt.effective_prompt_sha256,
                "glossary_state": prompt.glossary_state,
            },
        ),
        ocr=_stage_specification(StageKind.OCR),
        objects=_stage_specification(StageKind.OBJECTS),
        text_vectors=_stage_specification(StageKind.TEXT_VECTORS),
    )
    values: dict[str, object] = {
        "schema_version": 2,
        "specifications": specifications,
        "visual_dense_specification": _stage_specification(StageKind.VISUAL_DENSE),
        "lighthouse_specification": _stage_specification(StageKind.LIGHTHOUSE),
        "whisper_prompt_snapshot": prompt,
        "executor_identity": _EXECUTOR_IDENTITY,
    }
    values.update(changes)
    return VideoIndexPlanSnapshot(**values)  # type: ignore[arg-type]


def _prior_state() -> PriorVideoState:
    return PriorVideoState(
        status="ready",
        progress=1.0,
        stage="ready",
        error_code=None,
    )


def _queued_job(**changes: object) -> VideoIndexJob:
    default_idempotency_hash = video_index_idempotency_hash(
        kind=JobKind.VIDEO_INDEX,
        intent=VideoIndexIntent.INGEST,
        video_id="video-1",
        source_sha256="a" * 64,
        plan_hash="b" * 64,
    )
    values: dict[str, object] = {
        "job_id": "job-1",
        "kind": JobKind.VIDEO_INDEX,
        "intent": VideoIndexIntent.INGEST,
        "video_id": "video-1",
        "source_sha256": "a" * 64,
        "plan_hash": "b" * 64,
        "idempotency_hash": default_idempotency_hash,
        "state": JobState.QUEUED,
        "progress": 0.0,
        "stage": "queued",
        "attempt": 1,
        "retry_of_job_id": None,
        "prior_video_state": None,
        "execution_token": None,
        "cancel_requested_at": None,
        "error_code": None,
        "created_at": CREATED_AT,
        "started_at": None,
        "finished_at": None,
        "updated_at": CREATED_AT,
    }
    values.update(changes)
    if "idempotency_hash" not in changes and any(
        field in changes
        for field in ("kind", "intent", "video_id", "source_sha256", "plan_hash")
    ):
        values["idempotency_hash"] = video_index_idempotency_hash(
            kind=values["kind"],  # type: ignore[arg-type]
            intent=values["intent"],  # type: ignore[arg-type]
            video_id=values["video_id"],  # type: ignore[arg-type]
            source_sha256=values["source_sha256"],  # type: ignore[arg-type]
            plan_hash=values["plan_hash"],  # type: ignore[arg-type]
        )
    return VideoIndexJob(**values)  # type: ignore[arg-type]


def _running_job(**changes: object) -> VideoIndexJob:
    job = start_job(
        _queued_job(),
        execution_token=TOKEN,
        stage="probe",
        now=STARTED_AT,
    )
    return replace(job, **changes)


def test_job_enums_are_explicit_and_do_not_conflate_intent_with_state() -> None:
    assert {kind.value for kind in JobKind} == {"video_index"}
    assert {intent.value for intent in VideoIndexIntent} == {"ingest", "reindex"}
    assert {state.value for state in JobState} == {
        "queued",
        "running",
        "complete",
        "failed",
        "cancelled",
    }


def test_idempotency_hash_is_canonical_logical_work_identity() -> None:
    values = {
        "kind": JobKind.VIDEO_INDEX,
        "intent": VideoIndexIntent.REINDEX,
        "video_id": "video-1",
        "source_sha256": "a" * 64,
        "plan_hash": "b" * 64,
    }
    first = video_index_idempotency_hash(**values)
    second = video_index_idempotency_hash(**values)

    assert first == second
    assert len(first) == 64
    assert first != video_index_idempotency_hash(**{**values, "plan_hash": "d" * 64})

    with pytest.raises(ValueError, match="idempotency"):
        _queued_job(idempotency_hash="d" * 64)


def test_video_index_plan_is_canonical_complete_and_round_trippable() -> None:
    plan = _plan_snapshot()
    payload = json.loads(plan.canonical_json)

    assert payload == {
        "executor_identity": _EXECUTOR_IDENTITY,
        "schema_version": 2,
        "stages": {
            "lighthouse": plan.lighthouse_specification.canonical_json,
            "objects": plan.specifications.objects.canonical_json,
            "ocr": plan.specifications.ocr.canonical_json,
            "scenes": plan.specifications.scenes.canonical_json,
            "speech": plan.specifications.speech.canonical_json,
            "text_vectors": plan.specifications.text_vectors.canonical_json,
            "visual_dense": plan.visual_dense_specification.canonical_json,
        },
        "whisper_prompt": {
            "effective_prompt": "Мозгов, контратака",
            "effective_prompt_sha256": hashlib.sha256(
                "Мозгов, контратака".encode("utf-8")
            ).hexdigest(),
            "glossary_state": "ready",
        },
    }
    assert plan.plan_hash == hashlib.sha256(plan.canonical_json.encode()).hexdigest()
    assert VideoIndexPlanSnapshot.from_canonical_json(plan.canonical_json) == plan


def test_video_index_plan_identity_covers_every_stage_prompt_and_executor() -> None:
    baseline = _plan_snapshot()
    changed_executor = replace(baseline, executor_identity="sha256:" + "f" * 64)
    changed_scene = replace(
        baseline,
        specifications=replace(
            baseline.specifications,
            scenes=_stage_specification(
                StageKind.SCENES,
                parameters={"threshold": 2.5},
            ),
        ),
    )
    changed_prompt_value = "другой промпт"
    changed_prompt = WhisperPromptSnapshot(
        effective_prompt=changed_prompt_value,
        effective_prompt_sha256=hashlib.sha256(
            changed_prompt_value.encode("utf-8")
        ).hexdigest(),
        glossary_state="ready",
    )
    changed_prompt_plan = replace(
        baseline,
        whisper_prompt_snapshot=changed_prompt,
        specifications=replace(
            baseline.specifications,
            speech=_stage_specification(
                StageKind.SPEECH,
                parameters={
                    "effective_prompt_sha256": changed_prompt.effective_prompt_sha256,
                    "glossary_state": changed_prompt.glossary_state,
                },
            ),
        ),
    )

    assert len(
        {
            baseline.plan_hash,
            changed_executor.plan_hash,
            changed_scene.plan_hash,
            changed_prompt_plan.plan_hash,
        }
    ) == 4


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 0},
        {"schema_version": 1},
        {"schema_version": 3},
        {"schema_version": True},
        {"specifications": object()},
        {"whisper_prompt_snapshot": object()},
        {"executor_identity": ""},
        {"executor_identity": "private/path"},
        {"executor_identity": "sha256:" + "A" * 64},
        {"executor_identity": "sha256:" + "e" * 63},
        {"executor_identity": "x" * 257},
    ],
)
def test_video_index_plan_rejects_ambiguous_contract_fields(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _plan_snapshot(**changes)


@pytest.mark.parametrize(
    "snapshot",
    [
        WhisperPromptSnapshot(
            effective_prompt="x" * 16_001,
            effective_prompt_sha256=hashlib.sha256(b"x").hexdigest(),
            glossary_state="ready",
        ),
        WhisperPromptSnapshot(
            effective_prompt="prompt",
            effective_prompt_sha256="A" * 64,
            glossary_state="ready",
        ),
        WhisperPromptSnapshot(
            effective_prompt="prompt",
            effective_prompt_sha256=hashlib.sha256(b"different").hexdigest(),
            glossary_state="ready",
        ),
        WhisperPromptSnapshot(
            effective_prompt=None,
            effective_prompt_sha256=hashlib.sha256(b"").hexdigest(),
            glossary_state="unknown",
        ),
    ],
)
def test_video_index_plan_validates_the_frozen_whisper_prompt(
    snapshot: WhisperPromptSnapshot,
) -> None:
    with pytest.raises(ValueError, match="invalid canonical|Whisper"):
        _plan_snapshot(whisper_prompt_snapshot=snapshot)


def test_video_index_plan_requires_speech_stage_to_match_prompt_identity() -> None:
    baseline = _plan_snapshot()
    mismatched = replace(
        baseline.specifications,
        speech=_stage_specification(
            StageKind.SPEECH,
            parameters={
                "effective_prompt_sha256": "d" * 64,
                "glossary_state": "missing",
            },
        ),
    )

    with pytest.raises(ValueError, match="speech stage"):
        replace(baseline, specifications=mismatched)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not-json",
        "[]",
        '{"schema_version":1}',
    ],
)
def test_video_index_plan_rejects_invalid_persisted_json(value: object) -> None:
    with pytest.raises(ValueError, match="canonical|plan"):
        VideoIndexPlanSnapshot.from_canonical_json(value)  # type: ignore[arg-type]


def test_video_index_plan_normalizes_deep_nested_stage_json_failure() -> None:
    plan = _plan_snapshot()
    payload = json.loads(plan.canonical_json)
    nested_parameters = '{"x":' * 1_200 + "0" + "}" * 1_200
    payload["stages"]["scenes"] = (
        '{"dependencies":{},"implementation_revision":"videoscope.scenes.v1",'
        '"kind":"scenes","model_identity":null,"parameters":'
        + nested_parameters
        + ',"schema_version":1}'
    )
    persisted = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    with pytest.raises(ValueError, match="invalid canonical video index plan"):
        VideoIndexPlanSnapshot.from_canonical_json(persisted)


def test_video_index_plan_rejects_noncanonical_or_tampered_persisted_json() -> None:
    plan = _plan_snapshot()
    payload = json.loads(plan.canonical_json)
    payload["unsupported"] = True
    with pytest.raises(ValueError):
        VideoIndexPlanSnapshot.from_canonical_json(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )

    pretty = json.dumps(
        json.loads(plan.canonical_json),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    with pytest.raises(ValueError, match="canonical"):
        VideoIndexPlanSnapshot.from_canonical_json(pretty)

    tampered = json.loads(plan.canonical_json)
    tampered["whisper_prompt"]["effective_prompt"] = "secret replacement"
    with pytest.raises(ValueError, match="invalid canonical|Whisper"):
        VideoIndexPlanSnapshot.from_canonical_json(
            json.dumps(
                tampered,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )


def test_video_index_plan_normalizes_deeply_nested_stage_json_failure() -> None:
    payload = json.loads(_plan_snapshot().canonical_json)
    nested_parameters = '{"nested":' * 1_200 + "null" + "}" * 1_200
    payload["stages"]["scenes"] = (
        '{"dependencies":{},"implementation_revision":"videoscope.scenes.v1",'
        '"kind":"scenes","model_identity":null,"parameters":'
        + nested_parameters
        + ',"schema_version":1}'
    )
    persisted = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    with pytest.raises(ValueError, match="invalid canonical video index plan"):
        VideoIndexPlanSnapshot.from_canonical_json(persisted)


def test_queued_job_is_an_immutable_validated_snapshot() -> None:
    job = _queued_job()

    assert job.kind is JobKind.VIDEO_INDEX
    assert job.intent is VideoIndexIntent.INGEST
    assert job.state is JobState.QUEUED
    assert job.progress == 0.0
    with pytest.raises(FrozenInstanceError):
        job.stage = "running"  # type: ignore[misc]


def test_reindex_requires_a_bounded_prior_video_snapshot() -> None:
    prior = _prior_state()
    job = _queued_job(
        intent=VideoIndexIntent.REINDEX,
        prior_video_state=prior,
    )

    assert job.prior_video_state is prior
    with pytest.raises(ValueError, match="prior video state"):
        _queued_job(intent=VideoIndexIntent.REINDEX)
    with pytest.raises(ValueError, match="ingest"):
        _queued_job(prior_video_state=prior)


@pytest.mark.parametrize("status", ["queued", "processing", "ready", "failed"])
def test_prior_video_state_accepts_only_current_video_states(status: str) -> None:
    assert replace(_prior_state(), status=status).status == status


@pytest.mark.parametrize(
    "changes",
    [
        {"job_id": ""},
        {"job_id": "../unsafe"},
        {"job_id": "j" * 129},
        {"video_id": ""},
        {"video_id": "video/id"},
        {"kind": "unknown"},
        {"intent": "unknown"},
        {"state": "unknown"},
        {"source_sha256": "A" * 64},
        {"source_sha256": "not-a-digest"},
        {"plan_hash": "B" * 64},
        {"idempotency_hash": "c" * 63},
        {"progress": -0.01},
        {"progress": 1.01},
        {"progress": float("nan")},
        {"progress": float("inf")},
        {"progress": True},
        {"stage": ""},
        {"stage": "private/path"},
        {"stage": "s" * 65},
        {"attempt": 0},
        {"attempt": True},
        {"retry_of_job_id": "job-0"},
    ],
)
def test_job_rejects_ambiguous_or_unsafe_identity(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _queued_job(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"status": ""},
        {"status": "archived"},
        {"status": "ready/private"},
        {"status": "s" * 65},
        {"progress": -0.1},
        {"progress": 1.1},
        {"progress": float("nan")},
        {"stage": ""},
        {"stage": "path/to/stage"},
        {"stage": "s" * 65},
        {"error_code": "Read /Users/person/private.mp4"},
        {"error_code": "UPPERCASE"},
        {"error_code": "e" * 65},
    ],
)
def test_prior_video_state_rejects_unbounded_or_raw_values(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "status": "ready",
        "progress": 1.0,
        "stage": "ready",
        "error_code": None,
    }
    values.update(changes)

    with pytest.raises(ValueError):
        PriorVideoState(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    [
        {"progress": 0.1},
        {"stage": "probe"},
        {"execution_token": TOKEN},
        {"cancel_requested_at": CANCEL_REQUESTED_AT},
        {"error_code": "index_failed"},
        {"started_at": STARTED_AT},
        {"finished_at": FINISHED_AT},
        {"updated_at": STARTED_AT},
    ],
)
def test_queued_state_has_no_execution_or_outcome_fields(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _queued_job(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"execution_token": None},
        {"started_at": None},
        {"finished_at": FINISHED_AT},
        {"error_code": "index_failed"},
    ],
)
def test_running_state_requires_only_active_execution_fields(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _running_job(**changes)


def test_start_job_claims_queued_work_using_authoritative_inputs() -> None:
    queued = _queued_job()

    running = start_job(
        queued,
        execution_token=TOKEN,
        stage="probe",
        now=STARTED_AT,
    )

    assert running is not queued
    assert running.state is JobState.RUNNING
    assert running.execution_token == TOKEN
    assert running.started_at == STARTED_AT
    assert running.updated_at == STARTED_AT
    assert running.stage == "probe"
    assert queued.state is JobState.QUEUED


@pytest.mark.parametrize(
    "token",
    ["", "short", "worker token with spaces", "x" * 257],
)
def test_start_job_rejects_unsafe_execution_tokens(token: str) -> None:
    with pytest.raises(ValueError, match="execution token"):
        start_job(
            _queued_job(),
            execution_token=token,
            stage="probe",
            now=STARTED_AT,
        )


def test_progress_updates_are_monotonic_bounded_and_fenced() -> None:
    running = _running_job()
    advanced = advance_job(
        running,
        execution_token=TOKEN,
        progress=0.4,
        stage="speech",
        now=ADVANCED_AT,
    )

    assert advanced.progress == 0.4
    assert advanced.stage == "speech"
    assert advanced.updated_at == ADVANCED_AT

    with pytest.raises(ValueError, match="monotonic"):
        advance_job(
            advanced,
            execution_token=TOKEN,
            progress=0.39,
            stage="speech",
            now=CANCEL_REQUESTED_AT,
        )
    with pytest.raises(ValueError, match="below one"):
        advance_job(
            advanced,
            execution_token=TOKEN,
            progress=1.0,
            stage="index",
            now=CANCEL_REQUESTED_AT,
        )
    with pytest.raises(JobFenceError):
        advance_job(
            advanced,
            execution_token="different-token-0123456789abcdef",
            progress=0.5,
            stage="vision",
            now=CANCEL_REQUESTED_AT,
        )


def test_queued_cancellation_is_immediate_and_running_cancellation_is_requested() -> None:
    queued_cancelled = request_job_cancellation(_queued_job(), now=STARTED_AT)

    assert queued_cancelled.state is JobState.CANCELLED
    assert queued_cancelled.stage == "cancelled"
    assert queued_cancelled.cancel_requested_at == STARTED_AT
    assert queued_cancelled.finished_at == STARTED_AT
    assert queued_cancelled.started_at is None

    running = _running_job()
    requested = request_job_cancellation(running, now=CANCEL_REQUESTED_AT)

    assert requested.state is JobState.RUNNING
    assert requested.execution_token == TOKEN
    assert requested.cancel_requested_at == CANCEL_REQUESTED_AT
    assert requested.finished_at is None
    assert request_job_cancellation(requested, now=FINISHED_AT) is requested


def test_running_cancellation_request_cannot_precede_execution_start() -> None:
    with pytest.raises(ValueError, match="cancellation request.*start"):
        replace(
            _running_job(),
            cancel_requested_at="2026-08-19T08:00:30+00:00",
            updated_at=ADVANCED_AT,
        )


def test_requested_cancellation_blocks_progress_and_success() -> None:
    requested = request_job_cancellation(
        _running_job(),
        now=CANCEL_REQUESTED_AT,
    )

    with pytest.raises(ValueError, match="cancellation"):
        advance_job(
            requested,
            execution_token=TOKEN,
            progress=0.5,
            stage="vision",
            now=FINISHED_AT,
        )
    with pytest.raises(ValueError, match="cancellation"):
        complete_job(requested, execution_token=TOKEN, now=FINISHED_AT)
    with pytest.raises(ValueError, match="cancellation"):
        fail_job(
            requested,
            execution_token=TOKEN,
            error_code="index_failed",
            now=FINISHED_AT,
        )


def test_complete_job_is_terminal_success_and_drops_the_fencing_token() -> None:
    complete = complete_job(
        _running_job(),
        execution_token=TOKEN,
        now=FINISHED_AT,
    )

    assert complete.state is JobState.COMPLETE
    assert complete.progress == 1.0
    assert complete.stage == "complete"
    assert complete.execution_token is None
    assert complete.error_code is None
    assert complete.finished_at == FINISHED_AT

    for transition in (
        lambda: complete_job(complete, execution_token=TOKEN, now=FINISHED_AT),
        lambda: request_job_cancellation(complete, now=FINISHED_AT),
    ):
        with pytest.raises(ValueError, match="terminal|running"):
            transition()


def test_failed_job_requires_only_a_sanitized_error_code() -> None:
    failed = fail_job(
        _running_job(),
        execution_token=TOKEN,
        error_code="video_index_failed",
        now=FINISHED_AT,
    )

    assert failed.state is JobState.FAILED
    assert failed.stage == "failed"
    assert failed.error_code == "video_index_failed"
    assert failed.execution_token is None
    assert failed.finished_at == FINISHED_AT

    with pytest.raises(ValueError, match="error code"):
        fail_job(
            _running_job(),
            execution_token=TOKEN,
            error_code="could not read /private/video.mp4",
            now=FINISHED_AT,
        )


def test_running_cancellation_requires_request_and_matching_fence() -> None:
    running = _running_job()
    with pytest.raises(ValueError, match="requested"):
        cancel_job(running, execution_token=TOKEN, now=FINISHED_AT)

    requested = request_job_cancellation(running, now=CANCEL_REQUESTED_AT)
    with pytest.raises(JobFenceError):
        cancel_job(
            requested,
            execution_token="different-token-0123456789abcdef",
            now=FINISHED_AT,
        )

    cancelled = cancel_job(
        requested,
        execution_token=TOKEN,
        now=FINISHED_AT,
    )
    assert cancelled.state is JobState.CANCELLED
    assert cancelled.stage == "cancelled"
    assert cancelled.execution_token is None
    assert cancelled.error_code is None
    assert cancelled.cancel_requested_at == CANCEL_REQUESTED_AT
    assert cancelled.finished_at == FINISHED_AT


@pytest.mark.parametrize("terminal_state", [JobState.FAILED, JobState.CANCELLED])
def test_retry_links_immediate_parent_and_preserves_logical_work(
    terminal_state: JobState,
) -> None:
    original = _queued_job(
        intent=VideoIndexIntent.REINDEX,
        prior_video_state=_prior_state(),
    )
    running = start_job(
        original,
        execution_token=TOKEN,
        stage="probe",
        now=STARTED_AT,
    )
    if terminal_state is JobState.FAILED:
        parent = fail_job(
            running,
            execution_token=TOKEN,
            error_code="video_index_failed",
            now=FINISHED_AT,
        )
    else:
        requested = request_job_cancellation(running, now=CANCEL_REQUESTED_AT)
        parent = cancel_job(
            requested,
            execution_token=TOKEN,
            now=FINISHED_AT,
        )

    retry = retry_job(parent, job_id="job-2", now="2026-08-19T08:05:00+00:00")

    assert retry.state is JobState.QUEUED
    assert retry.job_id == "job-2"
    assert retry.attempt == 2
    assert retry.retry_of_job_id == parent.job_id
    assert retry.kind is parent.kind
    assert retry.intent is parent.intent
    assert retry.video_id == parent.video_id
    assert retry.source_sha256 == parent.source_sha256
    assert retry.plan_hash == parent.plan_hash
    assert retry.idempotency_hash == parent.idempotency_hash
    assert retry.prior_video_state == parent.prior_video_state
    assert retry.progress == 0.0
    assert retry.stage == "queued"
    assert retry.execution_token is None
    assert retry.cancel_requested_at is None
    assert retry.error_code is None
    assert retry.started_at is None
    assert retry.finished_at is None


def test_retry_rejects_nonterminal_parent_reused_id_and_bad_time() -> None:
    with pytest.raises(ValueError, match="failed or cancelled"):
        retry_job(_queued_job(), job_id="job-2", now=STARTED_AT)

    failed = fail_job(
        _running_job(),
        execution_token=TOKEN,
        error_code="video_index_failed",
        now=FINISHED_AT,
    )
    with pytest.raises(ValueError, match="another"):
        retry_job(failed, job_id=failed.job_id, now=FINISHED_AT)
    with pytest.raises(ValueError, match="timestamp"):
        retry_job(failed, job_id="job-2", now=STARTED_AT)


@pytest.mark.parametrize(
    "changes",
    [
        {"created_at": "2026-08-19"},
        {"created_at": "not-a-timestamp"},
        {"updated_at": "2026-08-19T07:59:00+00:00"},
    ],
)
def test_job_rejects_naive_invalid_or_reversed_timestamps(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="timestamp|precede"):
        _queued_job(**changes)


@pytest.mark.parametrize(
    ("terminal_state", "changes"),
    [
        (JobState.COMPLETE, {"progress": 0.99}),
        (JobState.COMPLETE, {"cancel_requested_at": CANCEL_REQUESTED_AT}),
        (JobState.COMPLETE, {"error_code": "unexpected_error"}),
        (JobState.COMPLETE, {"execution_token": TOKEN}),
        (JobState.FAILED, {"error_code": None}),
        (JobState.FAILED, {"execution_token": TOKEN}),
        (JobState.CANCELLED, {"error_code": "unexpected_error"}),
        (JobState.CANCELLED, {"cancel_requested_at": None}),
        (JobState.CANCELLED, {"finished_at": None}),
    ],
)
def test_persisted_terminal_snapshots_fail_closed(
    terminal_state: JobState,
    changes: dict[str, object],
) -> None:
    running = _running_job()
    if terminal_state is JobState.COMPLETE:
        terminal = complete_job(running, execution_token=TOKEN, now=FINISHED_AT)
    elif terminal_state is JobState.FAILED:
        terminal = fail_job(
            running,
            execution_token=TOKEN,
            error_code="video_index_failed",
            now=FINISHED_AT,
        )
    else:
        requested = request_job_cancellation(running, now=CANCEL_REQUESTED_AT)
        terminal = cancel_job(
            requested,
            execution_token=TOKEN,
            now=FINISHED_AT,
        )

    with pytest.raises(ValueError):
        replace(terminal, **changes)


@pytest.mark.parametrize(
    "terminal_state",
    [JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED],
)
def test_terminal_updated_timestamp_must_equal_the_outcome_timestamp(
    terminal_state: JobState,
) -> None:
    running = _running_job()
    if terminal_state is JobState.COMPLETE:
        terminal = complete_job(running, execution_token=TOKEN, now=FINISHED_AT)
    elif terminal_state is JobState.FAILED:
        terminal = fail_job(
            running,
            execution_token=TOKEN,
            error_code="video_index_failed",
            now=FINISHED_AT,
        )
    else:
        requested = request_job_cancellation(running, now=CANCEL_REQUESTED_AT)
        terminal = cancel_job(
            requested,
            execution_token=TOKEN,
            now=FINISHED_AT,
        )

    with pytest.raises(ValueError, match="terminal.*updated"):
        replace(terminal, updated_at="2026-08-19T08:05:00+00:00")
