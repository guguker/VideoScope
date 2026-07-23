from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace


MODALITY_WEIGHTS = {
    "internvideo": 1.30,
    "lighthouse": 1.20,
    "visual": 1.05,
    "speech": 1.00,
    "ocr": 0.90,
    "objects": 0.90,
    "scene": 0.65,
}


@dataclass(frozen=True, slots=True)
class EvidenceHit:
    video_id: str
    segment_id: str
    start: float
    end: float
    modality: str
    score: float
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FusedResult:
    video_id: str
    start: float
    end: float
    score: float
    modalities: list[str]
    evidence: list[EvidenceHit]


def _belongs_to_cluster(hit: EvidenceHit, cluster: list[EvidenceHit], tolerance: float) -> bool:
    cluster_start = min(item.start for item in cluster)
    cluster_end = max(item.end for item in cluster)
    return hit.start <= cluster_end + tolerance and hit.end >= cluster_start - tolerance


def calibrate_hits(hits: list[EvidenceHit], *, rank_blend: float = 0.16) -> list[EvidenceHit]:
    """Объединяет несопоставимые оценки моделей с учётом ранга внутри каждой модальности."""
    grouped: dict[str, list[EvidenceHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.modality, []).append(hit)

    calibrated: list[EvidenceHit] = []
    for modality_hits in grouped.values():
        ranked = sorted(modality_hits, key=lambda item: item.score, reverse=True)
        denominator = max(1, len(ranked) - 1)
        for rank, hit in enumerate(ranked):
            rank_score = 1.0 - 0.35 * (rank / denominator)
            raw_score = max(0.0, min(1.0, hit.score))
            score = raw_score * (1.0 - rank_blend) + rank_score * rank_blend
            calibrated.append(
                replace(
                    hit,
                    score=score,
                    metadata={
                        **hit.metadata,
                        "raw_score": raw_score,
                        "calibrated_score": score,
                        "modality_rank": rank + 1,
                    },
                )
            )
    return calibrated


def _score(
    cluster: list[EvidenceHit],
    modality_weights: dict[str, float] | None = None,
) -> float:
    weighted_sum = 0.0
    weights = modality_weights or MODALITY_WEIGHTS
    for hit in cluster:
        weight = weights.get(hit.modality, MODALITY_WEIGHTS.get(hit.modality, 0.75))
        # Веса, заданные типом запроса, — априорные приоритеты, а не множители уверенности.
        gain = 0.75 + 0.25 * max(0.0, min(2.0, weight))
        weighted_sum += max(0.0, min(1.0, hit.score)) * gain
    average = weighted_sum / len(cluster) if cluster else 0.0
    diversity_bonus = 0.075 * (len({hit.modality for hit in cluster}) - 1)
    corroboration_bonus = 0.015 * min(3, len(cluster) - 1)
    return min(1.0, average + diversity_bonus + corroboration_bonus)


def fuse_hits(
    hits: list[EvidenceHit],
    *,
    limit: int = 20,
    temporal_tolerance: float = 1.5,
    max_cluster_seconds: float = 30.0,
    modality_weights: dict[str, float] | None = None,
) -> list[FusedResult]:
    clusters: list[list[EvidenceHit]] = []
    for hit in sorted(hits, key=lambda item: (item.video_id, item.start, item.end)):
        matching = next(
            (
                cluster
                for cluster in reversed(clusters)
                if cluster[0].video_id == hit.video_id
                and _belongs_to_cluster(hit, cluster, temporal_tolerance)
                and max(max(item.end for item in cluster), hit.end)
                - min(min(item.start for item in cluster), hit.start)
                <= max_cluster_seconds
            ),
            None,
        )
        if matching is None:
            clusters.append([hit])
        else:
            matching.append(hit)

    results = [
        FusedResult(
            video_id=cluster[0].video_id,
            start=min(hit.start for hit in cluster),
            end=max(hit.end for hit in cluster),
            score=_score(cluster, modality_weights),
            modalities=sorted({hit.modality for hit in cluster}),
            evidence=sorted(cluster, key=lambda hit: hit.score, reverse=True),
        )
        for cluster in clusters
    ]
    return sorted(results, key=lambda result: result.score, reverse=True)[: max(0, limit)]
