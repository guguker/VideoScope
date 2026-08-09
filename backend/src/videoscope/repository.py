from __future__ import annotations

import json
import logging
import math
from numbers import Real
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from videoscope.search.text_matching import lexical_match


logger = logging.getLogger(__name__)
SEGMENT_MODALITIES = frozenset({"scene", "speech", "ocr", "objects"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


@dataclass(frozen=True, slots=True)
class VideoRecord:
    id: str
    original_name: str
    display_name: str | None
    stored_name: str
    media_path: str
    size_bytes: int
    status: str
    progress: float
    stage: str
    duration: float | None
    width: int | None
    height: int | None
    fps: float | None
    thumbnail_path: str | None
    error: str | None
    created_at: str
    updated_at: str

    @property
    def name(self) -> str:
        return self.display_name or self.original_name


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    id: str
    video_id: str
    start: float
    end: float
    modality: str
    text: str
    confidence: float
    metadata: dict[str, Any]
    thumbnail_path: str | None


class Repository:
    _VIDEO_UPDATE_FIELDS = {
        "display_name",
        "status",
        "progress",
        "stage",
        "duration",
        "width",
        "height",
        "fps",
        "thumbnail_path",
        "error",
    }

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    display_name TEXT,
                    stored_name TEXT NOT NULL UNIQUE,
                    media_path TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    stage TEXT NOT NULL,
                    duration REAL,
                    width INTEGER,
                    height INTEGER,
                    fps REAL,
                    thumbnail_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS segments (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                    start REAL NOT NULL,
                    end REAL NOT NULL,
                    modality TEXT NOT NULL,
                    text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    thumbnail_path TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_segments_video_time
                    ON segments(video_id, start, end);
                CREATE INDEX IF NOT EXISTS idx_segments_modality
                    ON segments(modality);
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(videos)").fetchall()
            }
            if "display_name" not in columns:
                connection.execute("ALTER TABLE videos ADD COLUMN display_name TEXT")

    def create_video(
        self,
        *,
        video_id: str,
        original_name: str,
        stored_name: str,
        media_path: str,
        size_bytes: int,
    ) -> VideoRecord:
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO videos (
                    id, original_name, stored_name, media_path, size_bytes,
                    status, progress, stage, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    original_name,
                    stored_name,
                    media_path,
                    size_bytes,
                    "queued",
                    0.0,
                    "queued",
                    timestamp,
                    timestamp,
                ),
            )
        record = self.get_video(video_id)
        if record is None:
            raise RuntimeError("video was not persisted")
        return record

    def update_video(self, video_id: str, **changes: object) -> VideoRecord:
        unknown = set(changes) - self._VIDEO_UPDATE_FIELDS
        if unknown:
            raise ValueError(f"unsupported video fields: {sorted(unknown)}")
        if not changes:
            record = self.get_video(video_id)
            if record is None:
                raise KeyError(video_id)
            return record

        assignments = [f"{field} = ?" for field in changes]
        assignments.append("updated_at = ?")
        values = [*changes.values(), _now(), video_id]
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE videos SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            if cursor.rowcount != 1:
                raise KeyError(video_id)
        record = self.get_video(video_id)
        if record is None:
            raise KeyError(video_id)
        return record

    def get_video(self, video_id: str) -> VideoRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        return VideoRecord(**dict(row)) if row else None

    def list_videos(self) -> list[VideoRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
        return [VideoRecord(**dict(row)) for row in rows]

    def add_segment(
        self,
        *,
        segment_id: str,
        video_id: str,
        start: float,
        end: float,
        modality: str,
        text: str,
        confidence: float,
        metadata: dict[str, Any] | None = None,
        thumbnail_path: str | None = None,
    ) -> SegmentRecord:
        record = self._validated_segment(
            segment_id=segment_id,
            video_id=video_id,
            start=start,
            end=end,
            modality=modality,
            text=text,
            confidence=confidence,
            metadata={} if metadata is None else metadata,
            thumbnail_path=thumbnail_path,
        )
        try:
            payload = json.dumps(
                record.metadata,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("segment metadata must be finite JSON") from error
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO segments (
                    id, video_id, start, end, modality, text,
                    confidence, metadata_json, thumbnail_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.video_id,
                    record.start,
                    record.end,
                    record.modality,
                    record.text,
                    record.confidence,
                    payload,
                    record.thumbnail_path,
                ),
            )
        return record

    @staticmethod
    def _validated_segment(
        *,
        segment_id: object,
        video_id: object,
        start: object,
        end: object,
        modality: object,
        text: object,
        confidence: object,
        metadata: object,
        thumbnail_path: object,
    ) -> SegmentRecord:
        if type(segment_id) is not str or not segment_id:
            raise ValueError("segment id must not be empty")
        if type(video_id) is not str or not video_id:
            raise ValueError("segment video id must not be empty")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, Real)
            or not isinstance(end, Real)
        ):
            raise ValueError("segment interval must be numeric")
        resolved_start = float(start)
        resolved_end = float(end)
        if (
            not math.isfinite(resolved_start)
            or not math.isfinite(resolved_end)
            or resolved_start < 0
            or resolved_end <= resolved_start
        ):
            raise ValueError("segment interval must be finite, non-negative and ordered")
        if modality not in SEGMENT_MODALITIES:
            raise ValueError("unsupported segment modality")
        if type(text) is not str:
            raise ValueError("segment text must be a string")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, Real)
            or not math.isfinite(resolved_confidence := float(confidence))
            or not 0 <= resolved_confidence <= 1
        ):
            raise ValueError("segment confidence must be finite and between zero and one")
        if not isinstance(metadata, dict):
            raise ValueError("segment metadata must be an object")
        if thumbnail_path is not None and type(thumbnail_path) is not str:
            raise ValueError("segment thumbnail path must be a string")
        return SegmentRecord(
            id=segment_id,
            video_id=video_id,
            start=resolved_start,
            end=resolved_end,
            modality=modality,
            text=text,
            confidence=resolved_confidence,
            metadata=dict(metadata),
            thumbnail_path=thumbnail_path,
        )

    def list_segments(self, video_id: str | None = None) -> list[SegmentRecord]:
        query = "SELECT * FROM segments"
        parameters: tuple[object, ...] = ()
        if video_id is not None:
            query += " WHERE video_id = ?"
            parameters = (video_id,)
        query += " ORDER BY video_id, start, end"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        segments: list[SegmentRecord] = []
        for row in rows:
            try:
                metadata = json.loads(
                    row["metadata_json"],
                    parse_constant=_reject_non_finite_json,
                )
                segments.append(
                    self._validated_segment(
                        segment_id=row["id"],
                        video_id=row["video_id"],
                        start=row["start"],
                        end=row["end"],
                        modality=row["modality"],
                        text=row["text"],
                        confidence=row["confidence"],
                        metadata=metadata,
                        thumbnail_path=row["thumbnail_path"],
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                logger.warning("Ignoring invalid persisted segment %s", row["id"])
        return segments

    def clear_segments(self, video_id: str, *, modality: str | None = None) -> None:
        with self._connect() as connection:
            if modality is None:
                connection.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))
            else:
                connection.execute(
                    "DELETE FROM segments WHERE video_id = ? AND modality = ?",
                    (video_id, modality),
                )

    def search_segments_lexical(
        self,
        query: str,
        *,
        video_ids: list[str] | None = None,
        limit: int = 50,
        query_variants: list[str] | None = None,
    ) -> list[tuple[SegmentRecord, float]]:
        variants = query_variants or [query]
        if not any(value.strip() for value in variants):
            return []
        candidates = self.list_segments()
        ranked: list[tuple[SegmentRecord, float]] = []
        for segment in candidates:
            if video_ids and segment.video_id not in video_ids:
                continue
            score = max(lexical_match(variant, segment.text).score for variant in variants)
            if score <= 0:
                continue
            ranked.append((segment, score))
        return sorted(ranked, key=lambda item: item[1], reverse=True)[:limit]
