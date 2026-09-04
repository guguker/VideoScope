"""Strict component benchmark for frozen video-verifier prepared inputs.

This deliberately stays separate from the query-to-interval product runner: it
measures the verifier boundary itself and never invokes a fallback-capable path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Callable, Literal, Protocol, Sequence

from videoscope.providers.qwen_video import QwenVideoJudgement
from videoscope.providers.qwen_worker import QwenWorkerClient

from .schema import (
    BenchmarkDataError,
    ComponentIdentity,
    _CODE_SHA_RE,
    _require_finite_float,
    _require_id,
    _require_positive_int,
    _require_string,
    _validate_utc_timestamp,
)
from .serialization import (
    JsonObject,
    canonical_json_bytes,
    expect_fields,
    expect_list,
    expect_object,
    parse_json_object,
)
from .storage import _read_bounded_file, _write_exclusive_bytes
from .video_verifier_schema import (
    MAX_PREPARED_INPUT_BYTES,
    MAX_VIDEO_VERIFIER_CASES,
    MAX_VIDEO_VERIFIER_MANIFEST_BYTES,
    VideoVerifierCase,
    VideoVerifierDataset,
    VideoVerifierExpectedFact,
    video_verifier_dataset_from_json,
    video_verifier_dataset_revision,
)


VIDEO_VERIFIER_CANDIDATE_SCHEMA_VERSION = 1
VIDEO_VERIFIER_RUN_SCHEMA_VERSION = 1
MAX_VIDEO_VERIFIER_RUN_BYTES = 16 * 1024**2

AttemptStatus = Literal["match", "model_miss", "infrastructure_error"]
RunStatus = Literal["complete", "infrastructure_failed"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JERSEY_RE = re.compile(r"^(?:0|00|[1-9][0-9]?)$")
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_INFRASTRUCTURE_ERROR_CODES = frozenset(
    {
        "prepared_input_unavailable",
        "prepared_input_identity_mismatch",
        "prepared_input_changed",
        "verifier_not_ready",
        "unsupported_input_contract",
        "verifier_execution_failed",
        "verifier_contract_invalid",
        "timer_failed",
    }
)


class VideoVerifierExecutionError(RuntimeError):
    """A direct verifier run cannot produce trustworthy benchmark evidence."""


class VideoVerifierInfrastructureError(RuntimeError):
    """A sanitized, machine-readable infrastructure failure from an adapter."""

    def __init__(self, code: str) -> None:
        if code not in _INFRASTRUCTURE_ERROR_CODES:
            raise ValueError("unsupported video verifier infrastructure error code")
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class VideoVerifierCandidateBinding:
    case_id: str
    candidate_id: str
    prepared_input_path: Path
    proposal_rank: int
    proposal_score: float

    def __post_init__(self) -> None:
        _require_id(self.case_id, "candidate.case_id")
        _require_id(self.candidate_id, "candidate.candidate_id")
        if not isinstance(self.prepared_input_path, Path):
            raise BenchmarkDataError("candidate.prepared_input_path must be a path")
        path_text = str(self.prepared_input_path)
        if "\x00" in path_text or not self.prepared_input_path.is_absolute():
            raise BenchmarkDataError(
                "candidate.prepared_input_path must be an absolute local path"
            )
        rank = _require_positive_int(self.proposal_rank, "candidate.proposal_rank")
        score = _require_finite_float(
            self.proposal_score,
            "candidate.proposal_score",
            minimum=0,
        )
        if score > 1:
            raise BenchmarkDataError(
                "candidate.proposal_score must be between zero and one"
            )
        object.__setattr__(self, "proposal_rank", rank)
        object.__setattr__(self, "proposal_score", score)


@dataclass(frozen=True, slots=True)
class VideoVerifierCandidateSet:
    schema_version: int
    dataset_revision: str
    candidate_set_revision: str
    candidates: tuple[VideoVerifierCandidateBinding, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != VIDEO_VERIFIER_CANDIDATE_SCHEMA_VERSION
        ):
            raise BenchmarkDataError(
                "video verifier candidate schema_version is unsupported"
            )
        _require_sha256(self.dataset_revision, "candidate dataset_revision")
        _require_sha256(
            self.candidate_set_revision,
            "candidate_set_revision",
        )
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise BenchmarkDataError("candidate set must not be empty")
        if len(self.candidates) > MAX_VIDEO_VERIFIER_CASES:
            raise BenchmarkDataError("candidate set exceeds the case count limit")
        if any(
            not isinstance(item, VideoVerifierCandidateBinding)
            for item in self.candidates
        ):
            raise BenchmarkDataError("candidate set contains an invalid binding")
        case_ids = tuple(item.case_id for item in self.candidates)
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        ranks = tuple(item.proposal_rank for item in self.candidates)
        if len(case_ids) != len(set(case_ids)):
            raise BenchmarkDataError("candidate case ids must be unique")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise BenchmarkDataError("candidate ids must be unique")
        if len(ranks) != len(set(ranks)):
            raise BenchmarkDataError("candidate proposal ranks must be unique")
        if set(ranks) != set(range(1, len(ranks) + 1)):
            raise BenchmarkDataError(
                "candidate proposal ranks must be contiguous from one"
            )
        object.__setattr__(
            self,
            "candidates",
            tuple(sorted(self.candidates, key=lambda item: item.proposal_rank)),
        )


@dataclass(frozen=True, slots=True)
class VideoVerifierPrediction:
    facts: tuple[VideoVerifierExpectedFact, ...]
    predicted_jersey: str | None
    confidence: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.facts, tuple) or not self.facts:
            raise BenchmarkDataError("verifier prediction facts must not be empty")
        if any(
            not isinstance(item, VideoVerifierExpectedFact) for item in self.facts
        ):
            raise BenchmarkDataError("verifier prediction contains an invalid fact")
        fact_ids = tuple(item.fact_id for item in self.facts)
        if len(fact_ids) != len(set(fact_ids)):
            raise BenchmarkDataError("verifier prediction fact ids must be unique")
        if self.predicted_jersey is not None and (
            not isinstance(self.predicted_jersey, str)
            or not _JERSEY_RE.fullmatch(self.predicted_jersey)
        ):
            raise BenchmarkDataError(
                "verifier prediction jersey must be 0, 00, 1-99 or null"
            )
        if self.confidence is not None:
            confidence = _require_finite_float(
                self.confidence,
                "verifier prediction confidence",
                minimum=0,
            )
            if confidence > 1:
                raise BenchmarkDataError(
                    "verifier prediction confidence must be between zero and one"
                )
            object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self,
            "facts",
            tuple(sorted(self.facts, key=lambda item: item.fact_id)),
        )


@dataclass(frozen=True, slots=True)
class VideoVerifierAttempt:
    case_id: str
    candidate_id: str
    prepared_input_sha256: str
    prepared_input_byte_size: int
    proposal_rank: int
    proposal_score: float
    status: AttemptStatus
    latency_ms: float
    prediction: VideoVerifierPrediction | None
    error_code: str | None

    def __post_init__(self) -> None:
        _require_id(self.case_id, "verifier attempt case_id")
        _require_id(self.candidate_id, "verifier attempt candidate_id")
        _require_sha256(
            self.prepared_input_sha256,
            "verifier attempt prepared_input_sha256",
        )
        _require_positive_int(
            self.prepared_input_byte_size,
            "verifier attempt prepared_input_byte_size",
        )
        _require_positive_int(self.proposal_rank, "verifier attempt proposal_rank")
        score = _require_finite_float(
            self.proposal_score,
            "verifier attempt proposal_score",
            minimum=0,
        )
        if score > 1:
            raise BenchmarkDataError(
                "verifier attempt proposal_score must be between zero and one"
            )
        latency = _require_finite_float(
            self.latency_ms,
            "verifier attempt latency_ms",
            minimum=0,
        )
        object.__setattr__(self, "proposal_score", score)
        object.__setattr__(self, "latency_ms", latency)
        if not isinstance(self.status, str) or self.status not in {
            "match",
            "model_miss",
            "infrastructure_error",
        }:
            raise BenchmarkDataError("verifier attempt status is invalid")
        if self.status == "infrastructure_error":
            if self.prediction is not None:
                raise BenchmarkDataError(
                    "infrastructure errors must not publish a prediction"
                )
            if (
                not isinstance(self.error_code, str)
                or self.error_code not in _INFRASTRUCTURE_ERROR_CODES
            ):
                raise BenchmarkDataError(
                    "infrastructure errors require a supported error code"
                )
        elif (
            not isinstance(self.prediction, VideoVerifierPrediction)
            or self.error_code is not None
        ):
            raise BenchmarkDataError(
                "completed verifier attempts require a prediction and no error code"
            )


@dataclass(frozen=True, slots=True)
class VideoVerifierSummary:
    case_count: int
    match_count: int
    model_miss_count: int
    infrastructure_error_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "case_count",
            "match_count",
            "model_miss_count",
            "infrastructure_error_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise BenchmarkDataError(
                    f"video verifier summary {field_name} must be non-negative"
                )
        if self.case_count <= 0:
            raise BenchmarkDataError("video verifier summary must not be empty")
        if (
            self.match_count
            + self.model_miss_count
            + self.infrastructure_error_count
            != self.case_count
        ):
            raise BenchmarkDataError("video verifier summary counts are inconsistent")


@dataclass(frozen=True, slots=True)
class VideoVerifierRunManifest:
    schema_version: int
    run_id: str
    created_at: str
    finished_at: str
    run_status: RunStatus
    code_sha: str
    dataset_id: str
    dataset_version: str
    dataset_revision: str
    candidate_set_revision: str
    verifier_identity: ComponentIdentity
    strict_no_fallback: bool
    attempts: tuple[VideoVerifierAttempt, ...]
    summary: VideoVerifierSummary

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != VIDEO_VERIFIER_RUN_SCHEMA_VERSION
        ):
            raise BenchmarkDataError("video verifier run schema_version is unsupported")
        _require_id(self.run_id, "video verifier run_id")
        _validate_utc_timestamp(self.created_at)
        _validate_utc_timestamp(self.finished_at)
        if _parse_timestamp(self.finished_at) < _parse_timestamp(self.created_at):
            raise BenchmarkDataError(
                "video verifier finished_at must not precede created_at"
            )
        if not isinstance(self.run_status, str) or self.run_status not in {
            "complete",
            "infrastructure_failed",
        }:
            raise BenchmarkDataError("video verifier run_status is invalid")
        if not isinstance(self.code_sha, str) or not _CODE_SHA_RE.fullmatch(
            self.code_sha
        ):
            raise BenchmarkDataError("video verifier code_sha is invalid")
        _require_id(self.dataset_id, "video verifier dataset_id")
        _require_id(self.dataset_version, "video verifier dataset_version")
        _require_sha256(self.dataset_revision, "video verifier dataset_revision")
        _require_sha256(
            self.candidate_set_revision,
            "video verifier candidate_set_revision",
        )
        if not isinstance(self.verifier_identity, ComponentIdentity):
            raise BenchmarkDataError("video verifier identity is invalid")
        if self.strict_no_fallback is not True:
            raise BenchmarkDataError(
                "video verifier run must attest strict no-fallback"
            )
        if not isinstance(self.attempts, tuple) or not self.attempts:
            raise BenchmarkDataError("video verifier attempts must not be empty")
        if any(not isinstance(item, VideoVerifierAttempt) for item in self.attempts):
            raise BenchmarkDataError("video verifier run contains an invalid attempt")
        case_ids = tuple(item.case_id for item in self.attempts)
        candidate_ids = tuple(item.candidate_id for item in self.attempts)
        if len(case_ids) != len(set(case_ids)) or len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise BenchmarkDataError("video verifier run attempts must be unique")
        ranks = tuple(item.proposal_rank for item in self.attempts)
        if set(ranks) != set(range(1, len(ranks) + 1)):
            raise BenchmarkDataError(
                "video verifier run proposal ranks must be unique and contiguous"
            )
        if not isinstance(self.summary, VideoVerifierSummary):
            raise BenchmarkDataError("video verifier run summary is invalid")
        actual = _summarize(self.attempts)
        if self.summary != actual:
            raise BenchmarkDataError("video verifier run summary is inconsistent")
        expected_status: RunStatus = (
            "infrastructure_failed"
            if actual.infrastructure_error_count
            else "complete"
        )
        if self.run_status != expected_status:
            raise BenchmarkDataError(
                "video verifier run_status is inconsistent with attempts"
            )
        object.__setattr__(
            self,
            "attempts",
            tuple(sorted(self.attempts, key=lambda item: item.proposal_rank)),
        )


class StrictVideoVerifier(Protocol):
    @property
    def identity(self) -> ComponentIdentity: ...

    def verify_strict(
        self,
        case: VideoVerifierCase,
        candidate: VideoVerifierCandidateBinding,
    ) -> VideoVerifierPrediction: ...


class QwenWorkerStrictVerifier:
    """Strict adapter for the isolated Qwen worker; no product fallback is used."""

    def __init__(
        self,
        client: QwenWorkerClient,
        *,
        fps: float = 2.0,
        max_tokens: int = 320,
    ) -> None:
        resolved_fps = _require_finite_float(fps, "verifier fps", minimum=0.5)
        if resolved_fps > 8:
            raise BenchmarkDataError("verifier fps must be at most eight")
        if type(max_tokens) is not int or not 64 <= max_tokens <= 1024:
            raise BenchmarkDataError("verifier max_tokens must be between 64 and 1024")
        self._client = client
        self._fps = resolved_fps
        self._max_tokens = max_tokens
        self._identity = _qwen_component_identity(client, resolved_fps, max_tokens)

    @property
    def identity(self) -> ComponentIdentity:
        return self._identity

    def verify_strict(
        self,
        case: VideoVerifierCase,
        candidate: VideoVerifierCandidateBinding,
    ) -> VideoVerifierPrediction:
        try:
            status = self._client.status()
        except Exception:
            raise VideoVerifierInfrastructureError("verifier_not_ready") from None
        if not getattr(status, "ready", False):
            raise VideoVerifierInfrastructureError("verifier_not_ready")

        if case.stratum == "basketball_facts" and case.input_kind == "native_video":
            judgement = self._client.judge_video(
                candidate.prepared_input_path,
                fps=self._fps,
                max_tokens=self._max_tokens,
                expected_sha256=case.prepared_input_sha256,
                expected_byte_size=case.prepared_input_byte_size,
            )
            return self._basketball_prediction(judgement)
        if case.stratum == "generic_visual" and case.input_kind == "storyboard":
            if case.query is None:
                raise VideoVerifierInfrastructureError("verifier_contract_invalid")
            judgement = self._client.judge_storyboard(
                candidate.prepared_input_path,
                case.query,
                max_tokens=self._max_tokens,
                expected_sha256=case.prepared_input_sha256,
                expected_byte_size=case.prepared_input_byte_size,
            )
            return self._generic_prediction(judgement)
        if case.stratum == "generic_visual" and case.input_kind == "native_video":
            if case.query is None:
                raise VideoVerifierInfrastructureError("verifier_contract_invalid")
            judgement = self._client.judge_video_query(
                candidate.prepared_input_path,
                case.query,
                fps=self._fps,
                max_tokens=self._max_tokens,
                expected_sha256=case.prepared_input_sha256,
                expected_byte_size=case.prepared_input_byte_size,
            )
            return self._generic_prediction(judgement)
        raise VideoVerifierInfrastructureError("unsupported_input_contract")

    @staticmethod
    def _basketball_prediction(
        judgement: QwenVideoJudgement,
    ) -> VideoVerifierPrediction:
        _require_qwen_judgement(judgement)
        return VideoVerifierPrediction(
            facts=tuple(
                VideoVerifierExpectedFact(fact_id, getattr(judgement, fact_id))
                for fact_id in (
                    "shot_attempt",
                    "ball_through_hoop",
                    "shooter_outside_arc",
                    "three_point_signal",
                )
            ),
            predicted_jersey=judgement.shooter_jersey,
            confidence=judgement.confidence,
        )

    @staticmethod
    def _generic_prediction(judgement: QwenVideoJudgement) -> VideoVerifierPrediction:
        _require_qwen_judgement(judgement)
        return VideoVerifierPrediction(
            facts=(
                VideoVerifierExpectedFact("matches_query", judgement.matches_query),
            ),
            predicted_jersey=None,
            confidence=judgement.confidence,
        )


class VideoVerifierRunner:
    def __init__(
        self,
        *,
        verifier: StrictVideoVerifier,
        code_sha: str,
        clock: Callable[[], datetime] | None = None,
        timer: Callable[[], float] = perf_counter,
        code_identity_resolver: Callable[[], str] | None = None,
    ) -> None:
        self._verifier = verifier
        self._code_sha = code_sha
        self._clock = clock or (lambda: datetime.now(UTC))
        self._timer = timer
        self._code_identity_resolver = code_identity_resolver

    def run(
        self,
        dataset: VideoVerifierDataset,
        candidates: VideoVerifierCandidateSet,
        *,
        run_id: str,
        output_path: Path,
    ) -> VideoVerifierRunManifest:
        output_path = Path(output_path)
        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite {output_path}")
        identity = self._validate_run_contract(dataset, candidates, run_id)
        created_at = self._timestamp()
        cases = {case.case_id: case for case in dataset.cases}
        attempts = tuple(
            self._run_candidate(cases[candidate.case_id], candidate)
            for candidate in candidates.candidates
        )
        finished_at = self._timestamp()
        summary = _summarize(attempts)
        manifest = VideoVerifierRunManifest(
            schema_version=VIDEO_VERIFIER_RUN_SCHEMA_VERSION,
            run_id=run_id,
            created_at=created_at,
            finished_at=finished_at,
            run_status=(
                "infrastructure_failed"
                if summary.infrastructure_error_count
                else "complete"
            ),
            code_sha=self._code_sha,
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            dataset_revision=candidates.dataset_revision,
            candidate_set_revision=candidates.candidate_set_revision,
            verifier_identity=identity,
            strict_no_fallback=True,
            attempts=attempts,
            summary=summary,
        )
        self._attest_code_identity()
        _write_exclusive_bytes(
            output_path,
            canonical_json_bytes(_run_to_dict(manifest)),
        )
        return manifest

    def _validate_run_contract(
        self,
        dataset: VideoVerifierDataset,
        candidates: VideoVerifierCandidateSet,
        run_id: str,
    ) -> ComponentIdentity:
        if not isinstance(dataset, VideoVerifierDataset):
            raise VideoVerifierExecutionError("video verifier dataset is invalid")
        if not isinstance(candidates, VideoVerifierCandidateSet):
            raise VideoVerifierExecutionError("video verifier candidates are invalid")
        _require_id(run_id, "video verifier run_id")
        if not isinstance(self._code_sha, str) or not _CODE_SHA_RE.fullmatch(
            self._code_sha
        ):
            raise VideoVerifierExecutionError("video verifier code SHA is invalid")
        self._attest_code_identity()
        if candidates.dataset_revision != video_verifier_dataset_revision(dataset):
            raise VideoVerifierExecutionError(
                "candidate set does not match the frozen dataset revision"
            )
        expected_cases = {case.case_id for case in dataset.cases}
        actual_cases = {candidate.case_id for candidate in candidates.candidates}
        if expected_cases != actual_cases or len(candidates.candidates) != len(
            dataset.cases
        ):
            raise VideoVerifierExecutionError(
                "candidate set must bind exactly every frozen dataset case"
            )
        expected_candidate_revision = _candidate_set_revision(
            dataset,
            candidates.candidates,
            candidates.dataset_revision,
        )
        if candidates.candidate_set_revision != expected_candidate_revision:
            raise VideoVerifierExecutionError(
                "candidate binding revision does not match its frozen inputs"
            )
        strict_method = getattr(self._verifier, "verify_strict", None)
        if not callable(strict_method):
            raise VideoVerifierExecutionError(
                "video verifier must implement the strict no-fallback contract"
            )
        identity = getattr(self._verifier, "identity", None)
        if not isinstance(identity, ComponentIdentity):
            raise VideoVerifierExecutionError("video verifier identity is invalid")
        return identity

    def _attest_code_identity(self) -> None:
        if self._code_identity_resolver is None:
            return
        try:
            actual = self._code_identity_resolver()
        except Exception:
            raise VideoVerifierExecutionError(
                "video verifier code identity attestation failed"
            ) from None
        if actual != self._code_sha:
            raise VideoVerifierExecutionError(
                "video verifier code identity drifted during execution"
            )

    def _run_candidate(
        self,
        case: VideoVerifierCase,
        candidate: VideoVerifierCandidateBinding,
    ) -> VideoVerifierAttempt:
        try:
            before = _read_bounded_file(
                candidate.prepared_input_path,
                MAX_PREPARED_INPUT_BYTES,
                "prepared verifier input",
            )
        except (BenchmarkDataError, OSError):
            return _infrastructure_attempt(
                case,
                candidate,
                "prepared_input_unavailable",
            )
        if not _matches_prepared_identity(case, before):
            return _infrastructure_attempt(
                case,
                candidate,
                "prepared_input_identity_mismatch",
            )

        try:
            started = self._timer()
        except Exception:
            return _infrastructure_attempt(case, candidate, "timer_failed")
        error_code: str | None = None
        try:
            prediction = self._verifier.verify_strict(case, candidate)
        except VideoVerifierInfrastructureError as exc:
            error_code = exc.code
            prediction = None
        except BenchmarkDataError:
            error_code = "verifier_contract_invalid"
            prediction = None
        except Exception:
            error_code = "verifier_execution_failed"
            prediction = None
        try:
            finished = self._timer()
            latency_ms = _latency_ms(started, finished)
        except Exception:
            return _infrastructure_attempt(case, candidate, "timer_failed")

        if prediction is None:
            return _infrastructure_attempt(
                case,
                candidate,
                error_code or "verifier_contract_invalid",
                latency_ms=latency_ms,
            )
        if not _prediction_matches_contract(case, prediction):
            return _infrastructure_attempt(
                case,
                candidate,
                "verifier_contract_invalid",
                latency_ms=latency_ms,
            )
        try:
            after = _read_bounded_file(
                candidate.prepared_input_path,
                MAX_PREPARED_INPUT_BYTES,
                "prepared verifier input",
            )
        except (BenchmarkDataError, OSError):
            return _infrastructure_attempt(
                case,
                candidate,
                "prepared_input_changed",
                latency_ms=latency_ms,
            )
        if before != after or not _matches_prepared_identity(case, after):
            return _infrastructure_attempt(
                case,
                candidate,
                "prepared_input_changed",
                latency_ms=latency_ms,
            )
        return VideoVerifierAttempt(
            case_id=case.case_id,
            candidate_id=candidate.candidate_id,
            prepared_input_sha256=case.prepared_input_sha256,
            prepared_input_byte_size=case.prepared_input_byte_size,
            proposal_rank=candidate.proposal_rank,
            proposal_score=candidate.proposal_score,
            status=(
                "match"
                if _prediction_matches_label(case, prediction)
                else "model_miss"
            ),
            latency_ms=latency_ms,
            prediction=prediction,
            error_code=None,
        )

    def _timestamp(self) -> str:
        try:
            value = self._clock()
        except Exception:
            raise VideoVerifierExecutionError(
                "video verifier benchmark clock failed"
            ) from None
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise VideoVerifierExecutionError(
                "video verifier benchmark clock must return an aware datetime"
            )
        try:
            return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
        except Exception:
            raise VideoVerifierExecutionError(
                "video verifier benchmark clock is invalid"
            ) from None


def load_video_verifier_candidates(
    path: Path,
    dataset: VideoVerifierDataset,
) -> VideoVerifierCandidateSet:
    if not isinstance(dataset, VideoVerifierDataset):
        raise VideoVerifierExecutionError("video verifier dataset is invalid")
    try:
        payload = _read_bounded_file(
            Path(path),
            MAX_VIDEO_VERIFIER_MANIFEST_BYTES,
            "video verifier candidate manifest",
        )
        value = parse_json_object(payload, "video verifier candidate manifest")
        expect_fields(
            value,
            {"schema_version", "dataset_revision", "candidates"},
            "video verifier candidate manifest",
        )
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != VIDEO_VERIFIER_CANDIDATE_SCHEMA_VERSION
        ):
            raise BenchmarkDataError("candidate schema_version is unsupported")
        dataset_revision = _require_sha256(
            value["dataset_revision"],
            "candidate dataset_revision",
        )
        expected_revision = video_verifier_dataset_revision(dataset)
        if dataset_revision != expected_revision:
            raise VideoVerifierExecutionError(
                "candidate manifest does not match the frozen dataset revision"
            )
        raw_candidates = expect_list(value["candidates"], "candidates")
        candidates = tuple(
            _candidate_from_dict(
                expect_object(item, f"candidates[{index}]")
            )
            for index, item in enumerate(raw_candidates)
        )
        expected_cases = {case.case_id for case in dataset.cases}
        actual_cases = {candidate.case_id for candidate in candidates}
        if expected_cases != actual_cases or len(candidates) != len(dataset.cases):
            raise VideoVerifierExecutionError(
                "candidate manifest must bind exactly every frozen dataset case"
            )
        candidate_revision = _candidate_set_revision(
            dataset,
            candidates,
            dataset_revision,
        )
        return VideoVerifierCandidateSet(
            schema_version=VIDEO_VERIFIER_CANDIDATE_SCHEMA_VERSION,
            dataset_revision=dataset_revision,
            candidate_set_revision=candidate_revision,
            candidates=candidates,
        )
    except VideoVerifierExecutionError:
        raise
    except BenchmarkDataError:
        raise VideoVerifierExecutionError(
            "video verifier candidate manifest is invalid"
        ) from None


def load_video_verifier_run(path: Path) -> VideoVerifierRunManifest:
    payload = _read_bounded_file(
        Path(path),
        MAX_VIDEO_VERIFIER_RUN_BYTES,
        "video verifier run manifest",
    )
    return _run_from_dict(parse_json_object(payload, "video verifier run manifest"))


def _candidate_from_dict(value: JsonObject) -> VideoVerifierCandidateBinding:
    expect_fields(
        value,
        {
            "case_id",
            "candidate_id",
            "prepared_input_path",
            "proposal_rank",
            "proposal_score",
        },
        "candidate",
    )
    path_value = _require_string(
        value["prepared_input_path"],
        "candidate.prepared_input_path",
        max_length=4_096,
    )
    return VideoVerifierCandidateBinding(
        case_id=value["case_id"],  # type: ignore[arg-type]
        candidate_id=value["candidate_id"],  # type: ignore[arg-type]
        prepared_input_path=Path(path_value),
        proposal_rank=value["proposal_rank"],  # type: ignore[arg-type]
        proposal_score=value["proposal_score"],  # type: ignore[arg-type]
    )


def _candidate_set_revision(
    dataset: VideoVerifierDataset,
    candidates: tuple[VideoVerifierCandidateBinding, ...],
    dataset_revision: str,
) -> str:
    cases = {case.case_id: case for case in dataset.cases}
    portable_candidates: list[object] = []
    for candidate in sorted(candidates, key=lambda item: item.proposal_rank):
        case = cases[candidate.case_id]
        portable_candidates.append(
            {
                "case_id": candidate.case_id,
                "candidate_id": candidate.candidate_id,
                "prepared_input_sha256": case.prepared_input_sha256,
                "prepared_input_byte_size": case.prepared_input_byte_size,
                "proposal_rank": candidate.proposal_rank,
                "proposal_score": candidate.proposal_score,
            }
        )
    payload = canonical_json_bytes(
        {
            "schema_version": VIDEO_VERIFIER_CANDIDATE_SCHEMA_VERSION,
            "dataset_revision": dataset_revision,
            "candidates": portable_candidates,
        }
    )
    return sha256(payload).hexdigest()


def _matches_prepared_identity(case: VideoVerifierCase, payload: bytes) -> bool:
    return (
        len(payload) == case.prepared_input_byte_size
        and sha256(payload).hexdigest() == case.prepared_input_sha256
    )


def _prediction_matches_contract(
    case: VideoVerifierCase,
    prediction: object,
) -> bool:
    if not isinstance(prediction, VideoVerifierPrediction):
        return False
    expected_ids = tuple(item.fact_id for item in case.expected_facts)
    predicted_ids = tuple(item.fact_id for item in prediction.facts)
    return expected_ids == predicted_ids


def _prediction_matches_label(
    case: VideoVerifierCase,
    prediction: VideoVerifierPrediction,
) -> bool:
    expected = tuple((item.fact_id, item.expected) for item in case.expected_facts)
    actual = tuple((item.fact_id, item.expected) for item in prediction.facts)
    return expected == actual and case.expected_jersey == prediction.predicted_jersey


def _latency_ms(started: object, finished: object) -> float:
    start = _require_finite_float(started, "verifier timer start")
    end = _require_finite_float(finished, "verifier timer finish")
    if end < start:
        raise BenchmarkDataError("verifier timer moved backwards")
    return (end - start) * 1_000


def _infrastructure_attempt(
    case: VideoVerifierCase,
    candidate: VideoVerifierCandidateBinding,
    error_code: str,
    *,
    latency_ms: float = 0.0,
) -> VideoVerifierAttempt:
    return VideoVerifierAttempt(
        case_id=case.case_id,
        candidate_id=candidate.candidate_id,
        prepared_input_sha256=case.prepared_input_sha256,
        prepared_input_byte_size=case.prepared_input_byte_size,
        proposal_rank=candidate.proposal_rank,
        proposal_score=candidate.proposal_score,
        status="infrastructure_error",
        latency_ms=latency_ms,
        prediction=None,
        error_code=error_code,
    )


def _summarize(
    attempts: tuple[VideoVerifierAttempt, ...],
) -> VideoVerifierSummary:
    return VideoVerifierSummary(
        case_count=len(attempts),
        match_count=sum(item.status == "match" for item in attempts),
        model_miss_count=sum(item.status == "model_miss" for item in attempts),
        infrastructure_error_count=sum(
            item.status == "infrastructure_error" for item in attempts
        ),
    )


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise BenchmarkDataError(f"{field} must be a lowercase SHA-256")
    return value


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(f"{value[:-1]}+00:00")


def _require_qwen_judgement(value: object) -> QwenVideoJudgement:
    if not isinstance(value, QwenVideoJudgement):
        raise BenchmarkDataError("Qwen worker returned an invalid judgement")
    return value


def _qwen_component_identity(
    client: QwenWorkerClient,
    fps: float,
    max_tokens: int,
) -> ComponentIdentity:
    raw = getattr(client, "identity", None)
    if type(raw) is not dict:
        raise BenchmarkDataError("Qwen worker identity is invalid")
    selected: dict[str, object] = {}
    for key in ("mode", "contract", "model", "runtime_identity"):
        selected[key] = _require_string(
            raw.get(key),
            f"Qwen worker identity {key}",
            max_length=512,
        )
    for key in (
        "source_bundle_sha256",
        "prompt_protocol_sha256",
        "input_root_sha256",
    ):
        selected[key] = _require_sha256(
            raw.get(key),
            f"Qwen worker identity {key}",
        )
    selected["fps"] = fps
    selected["max_tokens"] = max_tokens
    identity = json.dumps(
        selected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return ComponentIdentity("video_verifier", identity)


def _prediction_to_dict(prediction: VideoVerifierPrediction) -> JsonObject:
    return {
        "facts": [
            {"fact_id": fact.fact_id, "predicted": fact.expected}
            for fact in prediction.facts
        ],
        "predicted_jersey": prediction.predicted_jersey,
        "confidence": prediction.confidence,
    }


def _prediction_from_dict(value: JsonObject) -> VideoVerifierPrediction:
    expect_fields(value, {"facts", "predicted_jersey", "confidence"}, "prediction")
    facts: list[VideoVerifierExpectedFact] = []
    for index, item in enumerate(expect_list(value["facts"], "prediction.facts")):
        fact = expect_object(item, f"prediction.facts[{index}]")
        expect_fields(
            fact,
            {"fact_id", "predicted"},
            f"prediction.facts[{index}]",
        )
        facts.append(
            VideoVerifierExpectedFact(
                fact_id=fact["fact_id"],  # type: ignore[arg-type]
                expected=fact["predicted"],  # type: ignore[arg-type]
            )
        )
    return VideoVerifierPrediction(
        facts=tuple(facts),
        predicted_jersey=value["predicted_jersey"],  # type: ignore[arg-type]
        confidence=value["confidence"],  # type: ignore[arg-type]
    )


def _attempt_to_dict(attempt: VideoVerifierAttempt) -> JsonObject:
    return {
        "case_id": attempt.case_id,
        "candidate_id": attempt.candidate_id,
        "prepared_input_sha256": attempt.prepared_input_sha256,
        "prepared_input_byte_size": attempt.prepared_input_byte_size,
        "proposal_rank": attempt.proposal_rank,
        "proposal_score": attempt.proposal_score,
        "status": attempt.status,
        "latency_ms": attempt.latency_ms,
        "prediction": (
            _prediction_to_dict(attempt.prediction)
            if attempt.prediction is not None
            else None
        ),
        "error_code": attempt.error_code,
    }


def _attempt_from_dict(value: JsonObject) -> VideoVerifierAttempt:
    expect_fields(
        value,
        {
            "case_id",
            "candidate_id",
            "prepared_input_sha256",
            "prepared_input_byte_size",
            "proposal_rank",
            "proposal_score",
            "status",
            "latency_ms",
            "prediction",
            "error_code",
        },
        "verifier attempt",
    )
    prediction_value = value["prediction"]
    prediction = (
        None
        if prediction_value is None
        else _prediction_from_dict(expect_object(prediction_value, "prediction"))
    )
    return VideoVerifierAttempt(
        case_id=value["case_id"],  # type: ignore[arg-type]
        candidate_id=value["candidate_id"],  # type: ignore[arg-type]
        prepared_input_sha256=value["prepared_input_sha256"],  # type: ignore[arg-type]
        prepared_input_byte_size=value[
            "prepared_input_byte_size"
        ],  # type: ignore[arg-type]
        proposal_rank=value["proposal_rank"],  # type: ignore[arg-type]
        proposal_score=value["proposal_score"],  # type: ignore[arg-type]
        status=value["status"],  # type: ignore[arg-type]
        latency_ms=value["latency_ms"],  # type: ignore[arg-type]
        prediction=prediction,
        error_code=value["error_code"],  # type: ignore[arg-type]
    )


def _run_to_dict(run: VideoVerifierRunManifest) -> JsonObject:
    return {
        "schema_version": run.schema_version,
        "run_id": run.run_id,
        "created_at": run.created_at,
        "finished_at": run.finished_at,
        "run_status": run.run_status,
        "code_sha": run.code_sha,
        "dataset_id": run.dataset_id,
        "dataset_version": run.dataset_version,
        "dataset_revision": run.dataset_revision,
        "candidate_set_revision": run.candidate_set_revision,
        "verifier_identity": {
            "component_id": run.verifier_identity.component_id,
            "identity": run.verifier_identity.identity,
        },
        "strict_no_fallback": run.strict_no_fallback,
        "attempts": [_attempt_to_dict(item) for item in run.attempts],
        "summary": {
            "case_count": run.summary.case_count,
            "match_count": run.summary.match_count,
            "model_miss_count": run.summary.model_miss_count,
            "infrastructure_error_count": run.summary.infrastructure_error_count,
        },
    }


def _run_from_dict(value: JsonObject) -> VideoVerifierRunManifest:
    expect_fields(
        value,
        {
            "schema_version",
            "run_id",
            "created_at",
            "finished_at",
            "run_status",
            "code_sha",
            "dataset_id",
            "dataset_version",
            "dataset_revision",
            "candidate_set_revision",
            "verifier_identity",
            "strict_no_fallback",
            "attempts",
            "summary",
        },
        "video verifier run",
    )
    identity = expect_object(value["verifier_identity"], "verifier_identity")
    expect_fields(identity, {"component_id", "identity"}, "verifier_identity")
    summary = expect_object(value["summary"], "summary")
    expect_fields(
        summary,
        {
            "case_count",
            "match_count",
            "model_miss_count",
            "infrastructure_error_count",
        },
        "summary",
    )
    return VideoVerifierRunManifest(
        schema_version=value["schema_version"],  # type: ignore[arg-type]
        run_id=value["run_id"],  # type: ignore[arg-type]
        created_at=value["created_at"],  # type: ignore[arg-type]
        finished_at=value["finished_at"],  # type: ignore[arg-type]
        run_status=value["run_status"],  # type: ignore[arg-type]
        code_sha=value["code_sha"],  # type: ignore[arg-type]
        dataset_id=value["dataset_id"],  # type: ignore[arg-type]
        dataset_version=value["dataset_version"],  # type: ignore[arg-type]
        dataset_revision=value["dataset_revision"],  # type: ignore[arg-type]
        candidate_set_revision=value[
            "candidate_set_revision"
        ],  # type: ignore[arg-type]
        verifier_identity=ComponentIdentity(
            component_id=identity["component_id"],  # type: ignore[arg-type]
            identity=identity["identity"],  # type: ignore[arg-type]
        ),
        strict_no_fallback=value["strict_no_fallback"],  # type: ignore[arg-type]
        attempts=tuple(
            _attempt_from_dict(expect_object(item, f"attempts[{index}]"))
            for index, item in enumerate(expect_list(value["attempts"], "attempts"))
        ),
        summary=VideoVerifierSummary(
            case_count=summary["case_count"],  # type: ignore[arg-type]
            match_count=summary["match_count"],  # type: ignore[arg-type]
            model_miss_count=summary["model_miss_count"],  # type: ignore[arg-type]
            infrastructure_error_count=summary[
                "infrastructure_error_count"
            ],  # type: ignore[arg-type]
        ),
    )


class _CliUsageError(ValueError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise _CliUsageError("invalid command arguments")


def _parser() -> _ArgumentParser:
    parser = _ArgumentParser(prog="videoscope-video-verifier-benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--candidates", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--code-sha", required=True)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--input-root", type=Path, required=True)
    run.add_argument("--model-identity", required=True)
    run.add_argument("--api-key-env", required=True)
    run.add_argument("--fps", type=float, default=2.0)
    run.add_argument("--max-tokens", type=int, default=320)
    run.add_argument("--timeout", type=float, default=180.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.command != "run":
            raise _CliUsageError("invalid command")
        # Reuse the product benchmark's fail-closed Git attestation.  Keeping
        # ``--code-sha`` explicit makes the intended identity reviewable, while
        # equality with a clean current HEAD prevents a caller from labelling a
        # dirty or different implementation as that revision.
        from .cli import _current_code_sha

        if _current_code_sha() != arguments.code_sha:
            raise VideoVerifierExecutionError(
                "video verifier code identity does not match clean Git HEAD"
            )
        if not _ENVIRONMENT_NAME_RE.fullmatch(arguments.api_key_env):
            raise _CliUsageError("invalid API key environment name")
        api_key = os.environ.get(arguments.api_key_env)
        if api_key is None:
            raise VideoVerifierExecutionError("Qwen worker API key is unavailable")
        dataset_payload = _read_bounded_file(
            arguments.dataset,
            MAX_VIDEO_VERIFIER_MANIFEST_BYTES,
            "video verifier dataset",
        )
        dataset = video_verifier_dataset_from_json(dataset_payload)
        candidates = load_video_verifier_candidates(arguments.candidates, dataset)
        client = QwenWorkerClient(
            endpoint=arguments.endpoint,
            api_key=api_key,
            input_root=arguments.input_root,
            expected_model_identity=arguments.model_identity,
            timeout=arguments.timeout,
        )
        verifier = QwenWorkerStrictVerifier(
            client,
            fps=arguments.fps,
            max_tokens=arguments.max_tokens,
        )
        manifest = VideoVerifierRunner(
            verifier=verifier,
            code_sha=arguments.code_sha,
            code_identity_resolver=_current_code_sha,
        ).run(
            dataset,
            candidates,
            run_id=arguments.run_id,
            output_path=arguments.output,
        )
        _write_json(
            sys.stdout,
            {
                "status": "written",
                "run_id": manifest.run_id,
                "run_status": manifest.run_status,
                "attempt_count": len(manifest.attempts),
            },
        )
        return 0 if manifest.run_status == "complete" else 8
    except _CliUsageError:
        return _fail(2, "usage_error", "invalid verifier benchmark arguments")
    except FileNotFoundError:
        return _fail(4, "not_found", "verifier benchmark input was not found")
    except FileExistsError:
        return _fail(5, "conflict", "verifier benchmark output already exists")
    except (BenchmarkDataError, ValueError):
        return _fail(3, "invalid_data", "verifier benchmark data is invalid")
    except VideoVerifierExecutionError:
        return _fail(8, "execution_failed", "verifier benchmark execution failed")
    except OSError:
        return _fail(7, "io_error", "verifier benchmark storage failed")
    except KeyboardInterrupt:
        return _fail(130, "interrupted", "verifier benchmark was interrupted")
    except Exception:
        return _fail(70, "internal_error", "verifier benchmark failed")


def _write_json(stream, value: JsonObject) -> None:  # type: ignore[no-untyped-def]
    json.dump(
        value,
        stream,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    stream.write("\n")


def _fail(code: int, error: str, message: str) -> int:
    _write_json(sys.stderr, {"error": error, "message": message})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
