"""Strict, versioned contracts for a human-reviewed development pilot."""

import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
EventIdentifier = Annotated[str, Field(pattern=r"^(primary|event-[0-9a-f]{32})$")]
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


def canonical_json(value: dict) -> bytes:
    """Canonical UTF-8 identity bytes, deliberately without a trailing newline."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def batch_revision(value: dict) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ReviewSource(StrictModel):
    source_id: Identifier
    sha256: Digest
    byte_size: Annotated[int, Field(gt=0)]
    duration_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    source_group: Identifier
    usage: Literal["development_review"]
    review_allowed: Literal[True]
    training_rights: Literal["unknown"]

    @field_validator("review_allowed", mode="before")
    @classmethod
    def real_boolean(cls, value):
        if type(value) is not bool:
            raise ValueError("review_allowed must be a boolean")
        return value


class ReviewExample(StrictModel):
    example_id: Identifier
    source_id: Identifier
    source_start_seconds: Seconds
    source_end_seconds: Seconds
    clip_duration_seconds: Annotated[float, Field(gt=0, le=60, allow_inf_nan=False)]
    prepared_input_sha256: Digest
    prepared_input_byte_size: Annotated[int, Field(gt=0, le=1024**3)]
    clip_path: str
    poster_path: str
    selection_method: Annotated[str, Field(min_length=1, max_length=100)]
    selection_notes: Annotated[str, Field(max_length=2000)]

    @model_validator(mode="after")
    def valid_interval_and_paths(self):
        duration = self.source_end_seconds - self.source_start_seconds
        if not 0 < duration <= 60:
            raise ValueError("source interval must be positive and at most 60 seconds")
        if abs(duration - self.clip_duration_seconds) > 0.5:
            raise ValueError("prepared duration must match the source interval within 0.5s")
        if self.clip_path != f"clips/{self.example_id}.mp4":
            raise ValueError("clip path must name the contained example MP4")
        if self.poster_path != f"posters/{self.example_id}.jpg":
            raise ValueError("poster path must name the contained example JPG")
        return self


class BatchManifest(StrictModel):
    schema_version: Literal[1]
    batch_id: Identifier
    title: Annotated[str, Field(min_length=1, max_length=200)]
    created_at: Annotated[str, Field(max_length=40)]
    code_sha: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    purpose: Literal["annotation_pilot"]
    training_allowed: Literal[False]
    promotion_allowed: Literal[False]
    sources: Annotated[list[ReviewSource], Field(min_length=1, max_length=100)]
    examples: Annotated[list[ReviewExample], Field(min_length=1, max_length=100)]

    @field_validator("training_allowed", "promotion_allowed", mode="before")
    @classmethod
    def real_boolean(cls, value):
        if type(value) is not bool:
            raise ValueError("usage flags must be booleans")
        return value

    @field_validator("schema_version", mode="before")
    @classmethod
    def real_integer(cls, value):
        if type(value) is not int:
            raise ValueError("schema version must be an integer")
        return value

    @field_validator("created_at")
    @classmethod
    def timezone_required(cls, value):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("created_at requires timezone")
        return value

    @model_validator(mode="after")
    def valid_source_membership(self):
        sources = {source.source_id: source for source in self.sources}
        if len(sources) != len(self.sources):
            raise ValueError("duplicate source identifier")
        if len({source.sha256 for source in self.sources}) != len(self.sources):
            raise ValueError("duplicate source bytes require one source identity")
        if len({example.example_id for example in self.examples}) != len(self.examples):
            raise ValueError("duplicate example identifier")
        seen_inputs = set()
        for example in self.examples:
            source = sources.get(example.source_id)
            if source is None or example.source_end_seconds > source.duration_seconds:
                raise ValueError("example must be bounded by its declared source")
            key = (example.source_id, example.source_start_seconds, example.source_end_seconds)
            if key in seen_inputs:
                raise ValueError("duplicate source interval")
            seen_inputs.add(key)
        return self


class AnnotationFieldsV1(StrictModel):
    shot_type: Literal["two", "three", "free_throw", "non_shot", "unclear"] | None
    outcome: Literal["made", "miss", "not_applicable", "unclear"] | None
    presentation: Literal["live", "replay", "unclear"] | None
    boundary_status: Literal["complete", "too_short", "unclear"] | None
    start_seconds: Seconds | None
    end_seconds: Seconds | None
    notes: Annotated[str, Field(max_length=2000)]

    @model_validator(mode="after")
    def valid_boundaries(self):
        if (self.start_seconds is None) != (self.end_seconds is None):
            raise ValueError("both boundaries must be provided together")
        if self.start_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("end must follow start")
        return self


class AnnotationFields(AnnotationFieldsV1):
    # These are separate human observations, never inferred from the visible outcome.
    scoring_decision: Literal["counted", "not_counted", "not_applicable", "unclear"] | None = Field(
        default=None, description="Whether official points were awarded, independent of the visible ball outcome."
    )
    play_context: Literal[
        "in_play", "foul_on_shot", "after_whistle", "other_dead_ball", "not_applicable", "unclear"
    ] | None = Field(default=None, description=(
        "after_whistle is an entirely new shot begun after play stopped; it excludes continuation. "
        "foul_on_shot may coexist with counted points. null is unanswered; unclear is an explicit answer."
    ))

    @model_validator(mode="after")
    def valid_scoring_context(self):
        # after_whistle denotes a NEW shot begun after play stopped, not continuation.
        if self.scoring_decision == "counted" and self.play_context in {"after_whistle", "other_dead_ball"}:
            raise ValueError("a new dead-ball shot cannot have counted points")
        return self


class AnnotationRequest(AnnotationFields):
    schema_version: Literal[2]
    batch_revision: Digest
    example_id: Identifier
    expected_revision: Annotated[int, Field(ge=0, le=10000)]

    @field_validator("schema_version", mode="before")
    @classmethod
    def real_integer(cls, value):
        if type(value) is not int:
            raise ValueError("schema version must be an integer")
        return value


class EventAnnotationRequest(AnnotationRequest):
    schema_version: Literal[3]
    event_id: EventIdentifier


def annotation_complete(fields: dict, *, schema_version: int = 2) -> bool:
    """Explicit unclear is an answer; null is an unfinished review field."""
    names = ("shot_type", "outcome", "presentation", "boundary_status", "start_seconds", "end_seconds")
    if schema_version >= 2:
        names += ("scoring_decision", "play_context")
    return all(fields.get(name) is not None for name in names)


class AnnotationRecordV1(AnnotationFieldsV1):
    """Historical v1 records retain their original fields and completion semantics."""

    schema_version: Literal[1]
    batch_id: Identifier
    batch_revision: Digest
    example_id: Identifier
    revision: Annotated[int, Field(gt=0, le=10000)]
    created_at: str
    reviewer: Literal["local_owner"]
    label_status: Literal["draft", "human_reviewed"]
    destination: Literal["annotation_inbox"]
    gold: Literal[False]
    training_allowed: Literal[False]
    promotion_allowed: Literal[False]
    source_id: Identifier
    source_sha256: Digest
    source_start_seconds: Seconds
    source_end_seconds: Seconds
    prepared_input_sha256: Digest

    @model_validator(mode="after")
    def status_matches_completeness(self):
        complete = annotation_complete(self.model_dump(), schema_version=self.schema_version)
        expected = "human_reviewed" if complete else "draft"
        if self.label_status != expected:
            raise ValueError("annotation status must match explicit answers and boundaries")
        return self


class AnnotationRecord(AnnotationFields, AnnotationRecordV1):
    """New revisions use v2; older revisions are validated separately, never upgraded."""

    schema_version: Literal[2]


class EventAnnotationRecord(AnnotationRecord):
    schema_version: Literal[3]
    event_id: EventIdentifier


ANNOTATION_FIELDS = tuple(AnnotationFields.model_fields)
