from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Protocol

from videoscope.media.ffmpeg import SampledFrame
from videoscope.repository import Repository
from videoscope.search.fusion import EvidenceHit


class FrameExtractor(Protocol):
    def extract_frames(
        self,
        source: Path,
        destination: Path,
        start: float,
        end: float,
        *,
        step: float,
        max_width: int = 640,
    ) -> list[SampledFrame]: ...


class FrameScorer(Protocol):
    def score_images(self, query: str, paths: list[Path]) -> list[float]: ...


class TemporalRefiner:
    _IMPLEMENTATION_REVISION = "dense-temporal-refinement-v1"
    _SCRATCH_POLICY_IDENTITY = "private-temporary-directory-delete-on-exit-v1"

    def __init__(
        self,
        *,
        repository: Repository,
        extractor: FrameExtractor,
        scorer: FrameScorer,
        temp_dir: Path,
        top_candidates: int = 2,
        sample_step: float = 1.5,
        min_score: float = 0.50,
        score_drop: float = 0.08,
        max_candidate_seconds: float = 36.0,
        ffmpeg_identity: str | None = None,
        scorer_identity: str | None = None,
        runtime_identity: str | None = None,
    ) -> None:
        self.repository = repository
        self.extractor = extractor
        self.scorer = scorer
        self.temp_dir = Path(temp_dir)
        self.top_candidates = max(0, top_candidates)
        self.sample_step = max(0.25, sample_step)
        self.min_score = min_score
        self.score_drop = max(0.0, score_drop)
        self.max_candidate_seconds = max(2.0, max_candidate_seconds)
        self.ffmpeg_identity = self._optional_identity(ffmpeg_identity)
        self.scorer_identity = self._optional_identity(scorer_identity)
        self.runtime_identity = self._optional_identity(runtime_identity)

    @staticmethod
    def _optional_identity(value: str | None) -> str | None:
        if value is None:
            return None
        if type(value) is not str or not value.strip() or len(value) > 2_048:
            raise ValueError("benchmark runtime identities must be non-empty strings")
        return value.strip()

    @property
    def implementation_identity(self) -> str:
        payload = {
            "max_candidate_seconds": self.max_candidate_seconds,
            "min_score": self.min_score,
            "revision": self._IMPLEMENTATION_REVISION,
            "sample_step": self.sample_step,
            "score_drop": self.score_drop,
            "top_candidates": self.top_candidates,
        }
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in payload.values()
        ):
            raise ValueError("temporal refinement configuration must be finite")
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def benchmark_attestation(self) -> dict[str, object] | None:
        """Return a path-free strict benchmark identity or decline pinning.

        Production construction has to provide the exact attested FFmpeg and
        scorer-runtime identities.  A generic refiner must never become a
        benchmark-capable component merely because it implements ``refine``.
        """
        if (
            self.ffmpeg_identity is None
            or self.scorer_identity is None
            or self.runtime_identity is None
            or self.temp_dir.is_symlink()
        ):
            return None
        try:
            implementation_identity = self.implementation_identity
        except ValueError:
            return None
        return {
            "ffmpeg_identity": self.ffmpeg_identity,
            "implementation_identity": implementation_identity,
            "runtime_identity": self.runtime_identity,
            "scorer_identity": self.scorer_identity,
            "scratch_policy_identity": self._SCRATCH_POLICY_IDENTITY,
            "source_bound": True,
            "strict_complete": True,
            "top_candidates": self.top_candidates,
        }

    def refine(self, query: str, hits: list[EvidenceHit]) -> list[EvidenceHit]:
        return self._refine(query, hits, raise_on_error=False)

    def refine_strict(self, query: str, hits: list[EvidenceHit]) -> list[EvidenceHit]:
        return self._refine(query, hits, raise_on_error=True)

    def _refine(
        self,
        query: str,
        hits: list[EvidenceHit],
        *,
        raise_on_error: bool,
    ) -> list[EvidenceHit]:
        if not hits or self.top_candidates <= 0:
            return hits
        selected_ids = {
            hit.segment_id
            for hit in sorted(hits, key=lambda item: item.score, reverse=True)[: self.top_candidates]
        }
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        output: list[EvidenceHit] = []
        with tempfile.TemporaryDirectory(prefix="videoscope-refine-", dir=self.temp_dir) as directory:
            workspace = Path(directory)
            for index, hit in enumerate(hits):
                if hit.segment_id not in selected_ids or hit.modality != "visual":
                    output.append(hit)
                    continue
                try:
                    output.append(self._refine_hit(query, hit, workspace / str(index)))
                except Exception:
                    if raise_on_error:
                        raise
                    output.append(hit)
        return output

    def _refine_hit(self, query: str, hit: EvidenceHit, workspace: Path) -> EvidenceHit:
        video = self.repository.get_video(hit.video_id)
        if video is None:
            return hit
        duration = float(video.duration or hit.end)
        start = max(0.0, hit.start)
        end = min(duration, hit.end)
        if end <= start:
            return hit
        if end - start > self.max_candidate_seconds:
            midpoint = (start + end) / 2
            half = self.max_candidate_seconds / 2
            start = max(start, midpoint - half)
            end = min(end, midpoint + half)

        frames = self.extractor.extract_frames(
            Path(video.media_path),
            workspace,
            start,
            end,
            step=self.sample_step,
        )
        if not frames:
            return hit
        scores = self.scorer.score_images(query, [frame.path for frame in frames])
        if len(scores) != len(frames) or not scores:
            return hit
        peak_index = max(range(len(scores)), key=scores.__getitem__)
        peak_score = float(scores[peak_index])
        if peak_score < self.min_score:
            return hit

        threshold = max(self.min_score, peak_score - self.score_drop)
        first = peak_index
        last = peak_index
        while first > 0 and scores[first - 1] >= threshold:
            first -= 1
        while last + 1 < len(scores) and scores[last + 1] >= threshold:
            last += 1

        refined_start = max(hit.start, frames[first].timestamp - self.sample_step / 2)
        refined_end = min(hit.end, frames[last].timestamp + self.sample_step / 2)
        if refined_end - refined_start < min(2.0, hit.end - hit.start):
            center = frames[peak_index].timestamp
            refined_start = max(hit.start, center - 1.0)
            refined_end = min(hit.end, center + 1.0)
        return replace(
            hit,
            start=refined_start,
            end=refined_end,
            score=max(hit.score, peak_score),
            metadata={
                **hit.metadata,
                "source": "siglip2-dense",
                "temporal_refinement": True,
                "coarse_start": hit.start,
                "coarse_end": hit.end,
                "peak_timestamp": frames[peak_index].timestamp,
                "sample_step": self.sample_step,
                "dense_peak_score": peak_score,
                "dense_frame_count": len(frames),
            },
        )
