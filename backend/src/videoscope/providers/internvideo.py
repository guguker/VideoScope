from __future__ import annotations

import base64
from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Protocol

from videoscope.media.ffmpeg import SampledFrame
from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.repository import Repository
from videoscope.search.fusion import EvidenceHit, FusedResult


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


class HTTPClient(Protocol):
    def post(self, url: str, *, json: object, headers: dict[str, str], timeout: float): ...  # type: ignore[no-untyped-def]


class InternVideoReranker:
    """Optional GPU reranker using an InternVideo 2.5 inference endpoint."""

    id = "internvideo"
    model_name = "OpenGVLab/InternVideo2_5_Chat_8B"

    def __init__(
        self,
        *,
        endpoint: str | None,
        api_key: str | None,
        repository: Repository,
        extractor: FrameExtractor,
        temp_dir: Path,
        top_candidates: int = 4,
        frame_count: int = 8,
        timeout: float = 180.0,
        client: HTTPClient | None = None,
    ) -> None:
        self.endpoint = endpoint.strip() if endpoint else None
        self.api_key = api_key
        self.repository = repository
        self.extractor = extractor
        self.temp_dir = Path(temp_dir)
        self.top_candidates = max(1, top_candidates)
        self.frame_count = max(2, frame_count)
        self.timeout = timeout
        self.client = client

    def status(self) -> ProviderStatus:
        if not self.endpoint:
            return ProviderStatus(
                self.id,
                "InternVideo 2.5",
                ProviderState.NEEDS_CONFIGURATION,
                "Optional 8B GPU reranker; configure INTERNVIDEO_ENDPOINT",
                optional=True,
            )
        return ProviderStatus(
            self.id,
            "InternVideo 2.5",
            ProviderState.READY,
            f"GPU candidate reranker: {self.model_name}",
            optional=True,
        )

    def _http_client(self) -> HTTPClient:
        if self.client is not None:
            return self.client
        import httpx

        return httpx  # type: ignore[return-value]

    @staticmethod
    def _candidate_id(candidate: FusedResult) -> str:
        return f"{candidate.video_id}:{candidate.start:.3f}:{candidate.end:.3f}"

    def rerank(self, query: str, candidates: list[FusedResult]) -> list[FusedResult]:
        if not self.endpoint or not candidates:
            return candidates
        selected = candidates[: self.top_candidates]
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        payload_candidates: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="videoscope-internvideo-", dir=self.temp_dir) as directory:
            workspace = Path(directory)
            for index, candidate in enumerate(selected):
                video = self.repository.get_video(candidate.video_id)
                if video is None or candidate.end <= candidate.start:
                    continue
                step = max(0.4, (candidate.end - candidate.start) / self.frame_count)
                frames = self.extractor.extract_frames(
                    Path(video.media_path),
                    workspace / str(index),
                    candidate.start,
                    candidate.end,
                    step=step,
                    max_width=640,
                )[: self.frame_count]
                if not frames:
                    continue
                payload_candidates.append(
                    {
                        "id": self._candidate_id(candidate),
                        "video_id": candidate.video_id,
                        "start": candidate.start,
                        "end": candidate.end,
                        "frames": [
                            {
                                "timestamp": frame.timestamp,
                                "jpeg_base64": base64.b64encode(frame.path.read_bytes()).decode("ascii"),
                            }
                            for frame in frames
                        ],
                    }
                )

            if not payload_candidates:
                return candidates
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            url = self.endpoint if self.endpoint.rstrip("/").endswith("/rerank") else f"{self.endpoint.rstrip('/')}/rerank"
            response = self._http_client().post(
                url,
                json={
                    "model": self.model_name,
                    "task": "temporal_relevance",
                    "query": query,
                    "candidates": payload_candidates,
                },
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            response_payload = response.json()

        scores = {
            str(item.get("id")): (
                max(0.0, min(1.0, float(item.get("score") or 0.0))),
                str(item.get("reason") or "InternVideo 2.5 confirms the moment"),
            )
            for item in response_payload.get("scores", [])
            if isinstance(item, dict) and item.get("id")
        }
        output: list[FusedResult] = []
        for candidate in candidates:
            resolved = scores.get(self._candidate_id(candidate))
            if resolved is None:
                output.append(candidate)
                continue
            score, reason = resolved
            evidence = EvidenceHit(
                video_id=candidate.video_id,
                segment_id=f"internvideo:{self._candidate_id(candidate)}",
                start=candidate.start,
                end=candidate.end,
                modality="internvideo",
                score=score,
                text=reason,
                metadata={"source": "internvideo2.5", "model": self.model_name},
            )
            output.append(
                replace(
                    candidate,
                    score=min(1.0, candidate.score * 0.58 + score * 0.42),
                    modalities=sorted({*candidate.modalities, "internvideo"}),
                    evidence=[evidence, *candidate.evidence],
                )
            )
        return sorted(output, key=lambda candidate: candidate.score, reverse=True)
