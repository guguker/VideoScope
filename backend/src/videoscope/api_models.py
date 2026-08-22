from __future__ import annotations

from typing import Annotated, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from videoscope.evaluation import (
    EVALUATION_REPORT_SCHEMA_VERSION,
    EVALUATION_VIDEO_ID_PATTERN,
    EvaluationCase,
    evaluation_cases_revision,
    evaluation_revision,
)
from videoscope.providers.base import ProviderState


SearchMode: TypeAlias = Literal["all", "speech", "visual", "ocr"]
VIDEO_ID_PATTERN = EVALUATION_VIDEO_ID_PATTERN
VideoId: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=VIDEO_ID_PATTERN),
]
JobId: TypeAlias = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    ),
]
JobIntent: TypeAlias = Literal["ingest", "reindex"]
JobState: TypeAlias = Literal["queued", "running", "complete", "failed", "cancelled"]
JobSymbol: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]{0,63}$"),
]
EvaluationLabelSource: TypeAlias = Literal["gold", "silver"]
EvaluationVariantName: TypeAlias = Literal[
    "auto",
    "auto_lighthouse",
    "speech",
    "visual",
    "visual_lighthouse",
    "ocr",
]
Revision: TypeAlias = Annotated[
    str,
    Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
]
VideoStatus: TypeAlias = Literal["queued", "processing", "ready", "failed"]
EvidenceDetailValue: TypeAlias = str | bool | float | list[str] | None


class ContractModel(BaseModel):
    """Shared strictness for the public HTTP contract."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class SearchRequest(ContractModel):
    query: str = Field(min_length=1, max_length=500)
    video_ids: list[VideoId] | None = Field(default=None, min_length=1, max_length=100)
    limit: int = Field(default=20, ge=1, le=50)
    use_lighthouse: bool = True
    mode: SearchMode = "all"

    @field_validator("video_ids")
    @classmethod
    def validate_unique_video_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("video_ids must be unique")
        return value


class RenameVideoRequest(ContractModel):
    name: str = Field(min_length=1, max_length=160)


class GlossaryRequest(ContractModel):
    entries: dict[str, list[str]]


class EvaluationCaseRequest(ContractModel):
    id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=500)
    video_id: VideoId
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    mode: SearchMode = "all"
    label_source: EvaluationLabelSource = "gold"
    notes: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class EvaluationCasesRequest(ContractModel):
    cases: list[EvaluationCaseRequest] = Field(max_length=200)

    @field_validator("cases")
    @classmethod
    def validate_unique_case_ids(
        cls,
        value: list[EvaluationCaseRequest],
    ) -> list[EvaluationCaseRequest]:
        identifiers = [case.id for case in value]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("evaluation case ids must be unique")
        return value


class EvaluationRunRequest(ContractModel):
    variants: list[EvaluationVariantName] = Field(
        default_factory=lambda: ["auto"],
        min_length=1,
        max_length=6,
    )

    @field_validator("variants")
    @classmethod
    def validate_unique_variants(
        cls,
        value: list[EvaluationVariantName],
    ) -> list[EvaluationVariantName]:
        if len(set(value)) != len(value):
            raise ValueError("evaluation variants must be unique")
        return value


class ClipSelectionRequest(ContractModel):
    video_id: VideoId
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class ExportRequest(ContractModel):
    name: str = Field(default="videoscope-export", min_length=1, max_length=120)
    selections: list[ClipSelectionRequest] = Field(min_length=1, max_length=30)


class HealthResponse(ContractModel):
    status: Literal["ok"]
    service: Literal["videoscope"]


class ErrorResponse(ContractModel):
    detail: str


class JobSummaryResponse(ContractModel):
    job_id: JobId
    intent: JobIntent
    state: JobState
    progress: float = Field(ge=0, le=1)
    stage: JobSymbol
    attempt: int = Field(ge=1, le=1_000_000)
    cancel_requested_at: str | None = Field(default=None, max_length=64)
    error_code: JobSymbol | None = None
    created_at: str = Field(min_length=1, max_length=64)
    started_at: str | None = Field(default=None, max_length=64)
    finished_at: str | None = Field(default=None, max_length=64)
    updated_at: str = Field(min_length=1, max_length=64)


class JobResponse(JobSummaryResponse):
    video_id: VideoId
    retry_of_job_id: JobId | None = None


class VideoResponse(ContractModel):
    id: str
    original_name: str
    display_name: str | None
    size_bytes: int
    status: VideoStatus
    progress: float
    stage: str
    duration: float | None
    width: int | None
    height: int | None
    fps: float | None
    error: str | None
    created_at: str
    updated_at: str
    media_url: str
    thumbnail_url: str | None
    latest_job: JobSummaryResponse | None = None


class ReindexResponse(ContractModel):
    status: Literal["queued"]
    video_id: str


class SearchEvidenceResponse(ContractModel):
    modality: str
    score: float
    text: str
    source: str
    confidence: float | None
    start: float
    end: float
    raw_score: float
    matched_terms: list[str]
    details: dict[str, EvidenceDetailValue]


class SearchResultResponse(ContractModel):
    id: str
    video_id: str
    video_name: str
    start: float
    end: float
    score: float
    modalities: list[str]
    evidence: list[SearchEvidenceResponse]
    thumbnail_url: str | None
    intent: str
    explanation: str
    refined: bool


class GlossaryResponse(ContractModel):
    entries: dict[str, list[str]]


class EvaluationCaseResponse(ContractModel):
    id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=500)
    video_id: VideoId
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    mode: SearchMode
    label_source: EvaluationLabelSource
    notes: str = Field(max_length=500)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class EvaluationCasesResponse(ContractModel):
    cases: list[EvaluationCaseResponse]
    runtime_revision: Revision
    evaluation_revision: Revision

    @computed_field
    @property
    def cases_revision(self) -> str:
        return _evaluation_case_responses_revision(self.cases)

    @model_validator(mode="after")
    def validate_evaluation_revision(self) -> Self:
        expected = _evaluation_case_responses_freshness_revision(
            self.cases,
            self.runtime_revision,
        )
        if self.evaluation_revision != expected:
            raise ValueError("evaluation revision does not match current runtime")
        return self


class CaseEvaluationResponse(ContractModel):
    case_id: str
    query: str
    relevant_rank: int | None = Field(default=None, ge=1)
    temporal_iou: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    result_count: int = Field(ge=0)
    error: str | None


class EvaluationVariantResponse(ContractModel):
    name: EvaluationVariantName
    total_case_count: int = Field(ge=0)
    successful_case_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    status: Literal["complete", "partial", "failed"]
    recall_at_1: float | None = Field(default=None, ge=0, le=1)
    recall_at_3: float | None = Field(default=None, ge=0, le=1)
    recall_at_5: float | None = Field(default=None, ge=0, le=1)
    mrr: float | None = Field(default=None, ge=0, le=1)
    mean_temporal_iou: float | None = Field(default=None, ge=0, le=1)
    mean_latency_ms: float | None = Field(default=None, ge=0)
    cases: list[CaseEvaluationResponse]

    @model_validator(mode="after")
    def validate_counts_and_status(self) -> Self:
        if len(self.cases) != self.total_case_count:
            raise ValueError("evaluation case results must match total count")
        if self.successful_case_count + self.error_count != self.total_case_count:
            raise ValueError("evaluation variant counts must add up to total")
        expected_status = (
            "complete"
            if self.error_count == 0
            else "failed"
            if self.successful_case_count == 0
            else "partial"
        )
        if self.status != expected_status:
            raise ValueError("evaluation variant status does not match counts")
        metrics = (
            self.recall_at_1,
            self.recall_at_3,
            self.recall_at_5,
            self.mrr,
            self.mean_temporal_iou,
            self.mean_latency_ms,
        )
        if self.successful_case_count == 0 and any(value is not None for value in metrics):
            raise ValueError("evaluation metrics require a successful case")
        if self.successful_case_count > 0 and any(value is None for value in metrics):
            raise ValueError("evaluation metrics are required for successful cases")
        return self


class EvaluationReportResponse(ContractModel):
    generated_at: str
    schema_version: int | None = Field(default=None, ge=1)
    methodology_version: int | None = Field(default=None, ge=1)
    evaluation_revision: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    runtime_revision: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    cases_revision: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    temporal_iou_threshold: float | None = Field(default=None, ge=0, le=1)
    variants: list[EvaluationVariantResponse]

    @model_validator(mode="after")
    def validate_current_report_revisions(self) -> Self:
        if self.schema_version == EVALUATION_REPORT_SCHEMA_VERSION and any(
            value is None
            for value in (
                self.methodology_version,
                self.evaluation_revision,
                self.runtime_revision,
                self.cases_revision,
                self.temporal_iou_threshold,
            )
        ):
            raise ValueError("current evaluation report is missing revision metadata")
        return self


class EvaluationPayloadResponse(ContractModel):
    cases: list[EvaluationCaseResponse]
    runtime_revision: Revision
    evaluation_revision: Revision
    report: EvaluationReportResponse | None

    @computed_field
    @property
    def cases_revision(self) -> str:
        return _evaluation_case_responses_revision(self.cases)

    @model_validator(mode="after")
    def validate_evaluation_revision(self) -> Self:
        expected = _evaluation_case_responses_freshness_revision(
            self.cases,
            self.runtime_revision,
        )
        if self.evaluation_revision != expected:
            raise ValueError("evaluation revision does not match current runtime")
        return self


def _evaluation_case_responses_revision(cases: list[EvaluationCaseResponse]) -> str:
    return evaluation_cases_revision([
        EvaluationCase(**case.model_dump())
        for case in cases
    ])


def _evaluation_case_responses_freshness_revision(
    cases: list[EvaluationCaseResponse],
    runtime_revision: str,
) -> str:
    return evaluation_revision([
        EvaluationCase(**case.model_dump())
        for case in cases
    ], runtime_revision)


class ProviderResponse(ContractModel):
    id: str
    label: str
    state: ProviderState
    detail: str
    optional: bool


class ExportResponse(ContractModel):
    name: str
    duration: float
    created_at: str
    url: str
