from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
import hashlib
import json
import math
from numbers import Real
import re

from videoscope.artifacts import IndexingSpecifications, StageKind, StageSpecification
from videoscope.providers.whisper import (
    MAX_WHISPER_EFFECTIVE_PROMPT_CHARS,
    WhisperPromptSnapshot,
)


class JobKind(StrEnum):
    VIDEO_INDEX = "video_index"


class VideoIndexIntent(StrEnum):
    INGEST = "ingest"
    REINDEX = "reindex"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobTransitionError(ValueError):
    """A requested transition is incompatible with the durable job state."""


class JobFenceError(JobTransitionError):
    """An obsolete executor attempted to mutate work it no longer owns."""


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SYMBOL_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_EXECUTOR_IDENTITY_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_EXECUTION_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]{15,255}$")
_GLOSSARY_STATES = frozenset(
    {
        "not_configured",
        "missing",
        "unsafe",
        "oversized",
        "changed",
        "ready",
        "invalid",
        "unreadable",
    }
)
_PRIOR_VIDEO_STATUSES = frozenset({"queued", "processing", "ready", "failed"})
_PLAN_STAGE_FIELDS = (
    "scenes",
    "speech",
    "ocr",
    "objects",
    "text_vectors",
    "visual_dense",
    "lighthouse",
)
_MAX_PLAN_CANONICAL_JSON_BYTES = 1024 * 1024


def is_safe_execution_token(value: object) -> bool:
    """Return whether a token satisfies the shared durable-job fence contract."""
    return (
        type(value) is str
        and _EXECUTION_TOKEN_PATTERN.fullmatch(value) is not None
    )


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _validate_digest(value: object, *, field_name: str) -> str:
    if type(value) is not str or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be lowercase SHA-256")
    return value


def _validate_identifier(value: object, *, field_name: str) -> str:
    if type(value) is not str or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a safe non-empty identifier")
    return value


def _validate_symbol(value: object, *, field_name: str) -> str:
    if type(value) is not str or not _SYMBOL_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a bounded symbolic value")
    return value


def _validate_error_code(value: object) -> str | None:
    if value is None:
        return None
    try:
        return _validate_symbol(value, field_name="job error code")
    except ValueError as error:
        raise ValueError("job error code must be a sanitized symbolic code") from error


def _validate_progress(value: object, *, field_name: str = "job progress") -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite number between zero and one")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{field_name} must be a finite number between zero and one")
    return resolved


def _validate_timestamp(
    value: object,
    *,
    field_name: str,
    required: bool,
) -> datetime | None:
    if value is None and not required:
        return None
    if type(value) is not str:
        raise ValueError(f"{field_name} must be an ISO timestamp")
    try:
        resolved = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO timestamp") from error
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ValueError(f"{field_name} timestamp must include a timezone")
    return resolved


def _validate_prompt_snapshot(snapshot: object) -> WhisperPromptSnapshot:
    if not isinstance(snapshot, WhisperPromptSnapshot):
        raise ValueError("Whisper prompt snapshot must be validated")
    prompt = snapshot.effective_prompt
    if prompt is not None and (
        type(prompt) is not str or len(prompt) > MAX_WHISPER_EFFECTIVE_PROMPT_CHARS
    ):
        raise ValueError("Whisper effective prompt exceeds the bounded contract")
    digest = _validate_digest(
        snapshot.effective_prompt_sha256,
        field_name="Whisper effective prompt digest",
    )
    expected_digest = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()
    if digest != expected_digest:
        raise ValueError("Whisper effective prompt digest does not match its content")
    if type(snapshot.glossary_state) is not str or snapshot.glossary_state not in _GLOSSARY_STATES:
        raise ValueError("Whisper glossary state is invalid")
    return snapshot


def video_index_idempotency_hash(
    *,
    kind: JobKind,
    intent: VideoIndexIntent,
    video_id: str,
    source_sha256: str,
    plan_hash: str,
) -> str:
    try:
        resolved_kind = JobKind(kind)
        resolved_intent = VideoIndexIntent(intent)
    except (TypeError, ValueError) as error:
        raise ValueError("unsupported video index idempotency kind or intent") from error
    resolved_video_id = _validate_identifier(
        video_id,
        field_name="idempotency video id",
    )
    resolved_source = _validate_digest(
        source_sha256,
        field_name="idempotency source hash",
    )
    resolved_plan = _validate_digest(
        plan_hash,
        field_name="idempotency plan hash",
    )
    payload = json.dumps(
        {
            "intent": resolved_intent.value,
            "kind": resolved_kind.value,
            "plan_hash": resolved_plan,
            "source_sha256": resolved_source,
            "video_id": resolved_video_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class VideoIndexPlanSnapshot:
    """Canonical, self-contained identity of one video-index execution plan."""

    schema_version: int
    specifications: IndexingSpecifications
    visual_dense_specification: StageSpecification
    lighthouse_specification: StageSpecification
    whisper_prompt_snapshot: WhisperPromptSnapshot
    executor_identity: str
    _canonical_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != 2
        ):
            raise ValueError("video index plan schema version must be exactly two")
        if not isinstance(self.specifications, IndexingSpecifications):
            raise ValueError("video index plan specifications must be validated")
        for specification, expected_kind, label in (
            (
                self.visual_dense_specification,
                StageKind.VISUAL_DENSE,
                "visual dense",
            ),
            (
                self.lighthouse_specification,
                StageKind.LIGHTHOUSE,
                "Lighthouse",
            ),
        ):
            if (
                not isinstance(specification, StageSpecification)
                or specification.kind is not expected_kind
            ):
                raise ValueError(
                    f"video index {label} specification must be validated"
                )
        prompt_snapshot = _validate_prompt_snapshot(self.whisper_prompt_snapshot)
        if (
            type(self.executor_identity) is not str
            or not _EXECUTOR_IDENTITY_PATTERN.fullmatch(self.executor_identity)
        ):
            raise ValueError("video index executor identity must be canonical SHA-256")

        speech_parameters = self.specifications.speech.parameters
        if (
            speech_parameters.get("effective_prompt_sha256")
            != prompt_snapshot.effective_prompt_sha256
            or speech_parameters.get("glossary_state") != prompt_snapshot.glossary_state
        ):
            raise ValueError("video index speech stage does not match its Whisper prompt")

        payload = {
            "executor_identity": self.executor_identity,
            "schema_version": self.schema_version,
            "stages": {
                name: self.for_kind(StageKind(name)).canonical_json
                for name in _PLAN_STAGE_FIELDS
            },
            "whisper_prompt": {
                "effective_prompt": prompt_snapshot.effective_prompt,
                "effective_prompt_sha256": prompt_snapshot.effective_prompt_sha256,
                "glossary_state": prompt_snapshot.glossary_state,
            },
        }
        canonical_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(canonical_json.encode("utf-8")) > _MAX_PLAN_CANONICAL_JSON_BYTES:
            raise ValueError("canonical video index plan exceeds the bounded contract")
        object.__setattr__(self, "_canonical_json", canonical_json)

    @property
    def canonical_json(self) -> str:
        return self._canonical_json

    @property
    def plan_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def for_kind(self, kind: StageKind) -> StageSpecification:
        try:
            resolved = StageKind(kind)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported video index plan stage") from error
        if resolved is StageKind.VISUAL_DENSE:
            return self.visual_dense_specification
        if resolved is StageKind.LIGHTHOUSE:
            return self.lighthouse_specification
        return self.specifications.for_kind(resolved)

    @classmethod
    def from_canonical_json(cls, value: str) -> VideoIndexPlanSnapshot:
        if type(value) is not str:
            raise ValueError("canonical video index plan must be a string")
        if len(value.encode("utf-8")) > _MAX_PLAN_CANONICAL_JSON_BYTES:
            raise ValueError("canonical video index plan exceeds the bounded contract")
        try:
            payload = json.loads(value, parse_constant=_reject_non_finite_json)
        except (ValueError, RecursionError) as error:
            raise ValueError("invalid canonical video index plan") from error
        if not isinstance(payload, dict) or set(payload) != {
            "executor_identity",
            "schema_version",
            "stages",
            "whisper_prompt",
        }:
            raise ValueError("canonical video index plan has unsupported fields")
        stages = payload["stages"]
        if not isinstance(stages, dict) or set(stages) != set(_PLAN_STAGE_FIELDS):
            raise ValueError("canonical video index plan has invalid stages")
        whisper = payload["whisper_prompt"]
        if not isinstance(whisper, dict) or set(whisper) != {
            "effective_prompt",
            "effective_prompt_sha256",
            "glossary_state",
        }:
            raise ValueError("canonical video index plan has invalid Whisper prompt")
        try:
            parsed_stages = {
                name: StageSpecification.from_canonical_json(stages[name])
                for name in _PLAN_STAGE_FIELDS
            }
            specifications = IndexingSpecifications(
                **{
                    name: parsed_stages[name]
                    for name in _PLAN_STAGE_FIELDS[:5]
                }
            )
            prompt_snapshot = WhisperPromptSnapshot(
                effective_prompt=whisper["effective_prompt"],
                effective_prompt_sha256=whisper["effective_prompt_sha256"],
                glossary_state=whisper["glossary_state"],
            )
            plan = cls(
                schema_version=payload["schema_version"],
                specifications=specifications,
                visual_dense_specification=parsed_stages["visual_dense"],
                lighthouse_specification=parsed_stages["lighthouse"],
                whisper_prompt_snapshot=prompt_snapshot,
                executor_identity=payload["executor_identity"],
            )
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("invalid canonical video index plan") from error
        if plan.canonical_json != value:
            raise ValueError("video index plan is not canonical")
        return plan


@dataclass(frozen=True, slots=True)
class PriorVideoState:
    """The bounded video state restored when a reindex ends without release."""

    status: str
    progress: float
    stage: str
    error_code: str | None

    def __post_init__(self) -> None:
        status = _validate_symbol(self.status, field_name="prior video status")
        if status not in _PRIOR_VIDEO_STATUSES:
            raise ValueError("prior video status is unsupported")
        progress = _validate_progress(self.progress, field_name="prior video progress")
        stage = _validate_symbol(self.stage, field_name="prior video stage")
        error_code = _validate_error_code(self.error_code)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "error_code", error_code)


@dataclass(frozen=True, slots=True)
class VideoIndexJob:
    """One immutable snapshot of durable video-index work."""

    job_id: str
    kind: JobKind
    intent: VideoIndexIntent
    video_id: str
    source_sha256: str
    plan_hash: str
    idempotency_hash: str
    state: JobState
    progress: float
    stage: str
    attempt: int
    retry_of_job_id: str | None
    prior_video_state: PriorVideoState | None
    execution_token: str | None
    cancel_requested_at: str | None
    error_code: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    updated_at: str

    def __post_init__(self) -> None:
        job_id = _validate_identifier(self.job_id, field_name="job id")
        video_id = _validate_identifier(self.video_id, field_name="job video id")
        try:
            kind = JobKind(self.kind)
            intent = VideoIndexIntent(self.intent)
            state = JobState(self.state)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported video index job kind, intent, or state") from error
        source_sha256 = _validate_digest(self.source_sha256, field_name="job source hash")
        plan_hash = _validate_digest(self.plan_hash, field_name="job plan hash")
        idempotency_hash = _validate_digest(
            self.idempotency_hash,
            field_name="job idempotency hash",
        )
        expected_idempotency_hash = video_index_idempotency_hash(
            kind=kind,
            intent=intent,
            video_id=video_id,
            source_sha256=source_sha256,
            plan_hash=plan_hash,
        )
        if idempotency_hash != expected_idempotency_hash:
            raise ValueError("job idempotency hash does not match its logical work")
        progress = _validate_progress(self.progress)
        stage = _validate_symbol(self.stage, field_name="job stage")
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError("job attempt must be a positive integer")
        retry_of_job_id = self.retry_of_job_id
        if retry_of_job_id is None:
            if self.attempt != 1:
                raise ValueError("a root job must have attempt one")
        else:
            retry_of_job_id = _validate_identifier(
                retry_of_job_id,
                field_name="job retry parent id",
            )
            if retry_of_job_id == job_id or self.attempt < 2:
                raise ValueError("job retry lineage must reference another earlier attempt")

        prior_video_state = self.prior_video_state
        if intent is VideoIndexIntent.REINDEX:
            if not isinstance(prior_video_state, PriorVideoState):
                raise ValueError("reindex job requires a validated prior video state")
        elif prior_video_state is not None:
            raise ValueError("ingest job cannot contain a prior video state")

        execution_token = self.execution_token
        if execution_token is not None and (
            not is_safe_execution_token(execution_token)
        ):
            raise ValueError("job execution token must be a safe opaque value")
        error_code = _validate_error_code(self.error_code)

        created_at = _validate_timestamp(
            self.created_at,
            field_name="job created_at",
            required=True,
        )
        updated_at = _validate_timestamp(
            self.updated_at,
            field_name="job updated_at",
            required=True,
        )
        started_at = _validate_timestamp(
            self.started_at,
            field_name="job started_at",
            required=False,
        )
        finished_at = _validate_timestamp(
            self.finished_at,
            field_name="job finished_at",
            required=False,
        )
        cancel_requested_at = _validate_timestamp(
            self.cancel_requested_at,
            field_name="job cancel_requested_at",
            required=False,
        )
        assert created_at is not None and updated_at is not None
        if updated_at < created_at:
            raise ValueError("job updated timestamp cannot precede creation")
        for timestamp, description in (
            (started_at, "start"),
            (finished_at, "finish"),
            (cancel_requested_at, "cancellation request"),
        ):
            if timestamp is not None and timestamp < created_at:
                raise ValueError(f"job {description} timestamp cannot precede creation")
            if timestamp is not None and timestamp > updated_at:
                raise ValueError(f"job updated timestamp cannot precede {description}")
        if started_at is not None and finished_at is not None and finished_at < started_at:
            raise ValueError("job finish timestamp cannot precede start")
        if (
            started_at is not None
            and cancel_requested_at is not None
            and cancel_requested_at < started_at
        ):
            raise ValueError("job cancellation request timestamp cannot precede start")
        if (
            cancel_requested_at is not None
            and finished_at is not None
            and finished_at < cancel_requested_at
        ):
            raise ValueError("job finish timestamp cannot precede cancellation request")
        if (
            state in {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}
            and finished_at is not None
            and updated_at != finished_at
        ):
            raise ValueError("terminal job updated timestamp must equal its finish timestamp")

        if state is JobState.QUEUED:
            if (
                progress != 0.0
                or stage != JobState.QUEUED.value
                or any(
                    (
                        execution_token,
                        cancel_requested_at,
                        error_code,
                        started_at,
                        finished_at,
                    )
                )
                or updated_at != created_at
            ):
                raise ValueError("queued job cannot contain execution or outcome fields")
        elif state is JobState.RUNNING:
            if (
                execution_token is None
                or started_at is None
                or finished_at is not None
                or error_code is not None
                or progress >= 1.0
                or stage in {
                    JobState.QUEUED.value,
                    JobState.COMPLETE.value,
                    JobState.FAILED.value,
                    JobState.CANCELLED.value,
                }
            ):
                raise ValueError("running job requires only active execution fields")
        elif state is JobState.COMPLETE:
            if (
                progress != 1.0
                or stage != JobState.COMPLETE.value
                or execution_token is not None
                or cancel_requested_at is not None
                or error_code is not None
                or started_at is None
                or finished_at is None
            ):
                raise ValueError("complete job requires an unfenced successful outcome")
        elif state is JobState.FAILED:
            if (
                stage != JobState.FAILED.value
                or execution_token is not None
                or cancel_requested_at is not None
                or error_code is None
                or started_at is None
                or finished_at is None
            ):
                raise ValueError("failed job requires a sanitized terminal error outcome")
        elif state is JobState.CANCELLED:
            if (
                stage != JobState.CANCELLED.value
                or execution_token is not None
                or cancel_requested_at is None
                or error_code is not None
                or finished_at is None
                or (started_at is None and progress != 0.0)
            ):
                raise ValueError("cancelled job requires a terminal cancellation outcome")

        object.__setattr__(self, "job_id", job_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "intent", intent)
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "plan_hash", plan_hash)
        object.__setattr__(self, "idempotency_hash", idempotency_hash)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "retry_of_job_id", retry_of_job_id)
        object.__setattr__(self, "error_code", error_code)


def _transition_time(job: VideoIndexJob, now: str) -> None:
    resolved = _validate_timestamp(now, field_name="job transition timestamp", required=True)
    current = _validate_timestamp(
        job.updated_at,
        field_name="job updated_at",
        required=True,
    )
    assert resolved is not None and current is not None
    if resolved < current:
        raise JobTransitionError("job transition timestamp cannot precede current state")


def _require_running_and_fence(job: VideoIndexJob, execution_token: str) -> None:
    if job.state in {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}:
        raise JobTransitionError("terminal job is immutable")
    if job.state is not JobState.RUNNING:
        raise JobTransitionError("job must be running")
    if not is_safe_execution_token(execution_token):
        raise JobFenceError("job execution token is invalid")
    if execution_token != job.execution_token:
        raise JobFenceError("job execution token does not own this attempt")


def start_job(
    job: VideoIndexJob,
    *,
    execution_token: str,
    stage: str,
    now: str,
) -> VideoIndexJob:
    if job.state in {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}:
        raise JobTransitionError("terminal job is immutable")
    if job.state is not JobState.QUEUED:
        raise JobTransitionError("only a queued job can start")
    _transition_time(job, now)
    return replace(
        job,
        state=JobState.RUNNING,
        stage=stage,
        execution_token=execution_token,
        started_at=now,
        updated_at=now,
    )


def advance_job(
    job: VideoIndexJob,
    *,
    execution_token: str,
    progress: float,
    stage: str,
    now: str,
) -> VideoIndexJob:
    _require_running_and_fence(job, execution_token)
    if job.cancel_requested_at is not None:
        raise JobTransitionError("job cancellation must be acknowledged before progress")
    resolved_progress = _validate_progress(progress)
    if resolved_progress < job.progress:
        raise JobTransitionError("job progress must be monotonic")
    if resolved_progress >= 1.0:
        raise JobTransitionError("running job progress must remain below one")
    _transition_time(job, now)
    return replace(job, progress=resolved_progress, stage=stage, updated_at=now)


def request_job_cancellation(job: VideoIndexJob, *, now: str) -> VideoIndexJob:
    if job.state in {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}:
        raise JobTransitionError("terminal job is immutable")
    _transition_time(job, now)
    if job.state is JobState.QUEUED:
        return replace(
            job,
            state=JobState.CANCELLED,
            stage=JobState.CANCELLED.value,
            cancel_requested_at=now,
            finished_at=now,
            updated_at=now,
        )
    if job.state is not JobState.RUNNING:
        raise JobTransitionError("only queued or running work can be cancelled")
    if job.cancel_requested_at is not None:
        return job
    return replace(job, cancel_requested_at=now, updated_at=now)


def complete_job(
    job: VideoIndexJob,
    *,
    execution_token: str,
    now: str,
) -> VideoIndexJob:
    _require_running_and_fence(job, execution_token)
    if job.cancel_requested_at is not None:
        raise JobTransitionError("job cannot complete after a cancellation request")
    _transition_time(job, now)
    return replace(
        job,
        state=JobState.COMPLETE,
        progress=1.0,
        stage=JobState.COMPLETE.value,
        execution_token=None,
        finished_at=now,
        updated_at=now,
    )


def fail_job(
    job: VideoIndexJob,
    *,
    execution_token: str,
    error_code: str,
    now: str,
) -> VideoIndexJob:
    _require_running_and_fence(job, execution_token)
    if job.cancel_requested_at is not None:
        raise JobTransitionError("job cannot fail after a cancellation request")
    validated_error = _validate_error_code(error_code)
    if validated_error is None:
        raise ValueError("failed job requires an error code")
    _transition_time(job, now)
    return replace(
        job,
        state=JobState.FAILED,
        stage=JobState.FAILED.value,
        execution_token=None,
        error_code=validated_error,
        finished_at=now,
        updated_at=now,
    )


def cancel_job(
    job: VideoIndexJob,
    *,
    execution_token: str,
    now: str,
) -> VideoIndexJob:
    _require_running_and_fence(job, execution_token)
    if job.cancel_requested_at is None:
        raise JobTransitionError("job cancellation has not been requested")
    _transition_time(job, now)
    return replace(
        job,
        state=JobState.CANCELLED,
        stage=JobState.CANCELLED.value,
        execution_token=None,
        finished_at=now,
        updated_at=now,
    )


def retry_job(job: VideoIndexJob, *, job_id: str, now: str) -> VideoIndexJob:
    if job.state not in {JobState.FAILED, JobState.CANCELLED}:
        raise JobTransitionError("only a failed or cancelled job can be retried")
    resolved_job_id = _validate_identifier(job_id, field_name="retry job id")
    if resolved_job_id == job.job_id:
        raise ValueError("job retry must use another job id")
    _transition_time(job, now)
    return replace(
        job,
        job_id=resolved_job_id,
        state=JobState.QUEUED,
        progress=0.0,
        stage=JobState.QUEUED.value,
        attempt=job.attempt + 1,
        retry_of_job_id=job.job_id,
        execution_token=None,
        cancel_requested_at=None,
        error_code=None,
        created_at=now,
        started_at=None,
        finished_at=None,
        updated_at=now,
    )
