from __future__ import annotations

from types import SimpleNamespace

import pytest

from videoscope.repository import ExternalIndexReleaseSnapshot
from videoscope.search.fusion import EvidenceHit
from videoscope.search.service import SearchDependencyError, SearchService


def _descriptor(video_id: str, generation_id: str, *, lighthouse: bool) -> dict[str, object]:
    descriptor: dict[str, object] = {
        "duration_seconds": 10.0,
        "generation_id": generation_id,
        "source_sha256": "a" * 64,
        "source_size_bytes": 10,
        "specification_hash": "b" * 64,
        "video_id": video_id,
    }
    descriptor["manifest_sha256" if lighthouse else "content_sha256"] = "c" * 64
    return descriptor


class _Repository:
    def __init__(self, snapshot: ExternalIndexReleaseSnapshot) -> None:
        self.snapshot = snapshot
        self.snapshot_calls = 0

    def list_videos(self):  # type: ignore[no-untyped-def]
        return [
            SimpleNamespace(id="video-1", status="ready", name="video-1.mp4")
        ]

    def get_external_index_release_snapshot(self, video_ids):  # type: ignore[no-untyped-def]
        assert tuple(video_ids) == ("video-1",)
        self.snapshot_calls += 1
        return self.snapshot


class _VectorIndex:
    available = True
    supports_generation_provenance = False

    def search(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return []


class _Visual:
    def __init__(self, on_exact=None) -> None:  # type: ignore[no-untyped-def]
        self.exact_calls: list[dict[str, dict[str, object]]] = []
        self.legacy_calls: list[list[str] | None] = []
        self.on_exact = on_exact

    def search_generations(self, query: str, *, generation_bindings, limit: int):  # type: ignore[no-untyped-def]
        del limit
        self.exact_calls.append(generation_bindings)
        if self.on_exact is not None:
            self.on_exact()
        return [
            EvidenceHit(
                "video-1",
                f"visual:{generation_bindings['video-1']['generation_id']}",
                1.0,
                3.0,
                "visual",
                0.9,
                query,
            )
        ]

    def search(self, query: str, *, video_ids=None, limit: int):  # type: ignore[no-untyped-def]
        del query, limit
        self.legacy_calls.append(video_ids)
        return []


class _Lighthouse:
    def __init__(self) -> None:
        self.exact_calls: list[dict[str, dict[str, object]]] = []
        self.legacy_calls: list[list[str]] = []

    def search_generations(self, query: str, generation_bindings, *, limit: int):  # type: ignore[no-untyped-def]
        del limit
        self.exact_calls.append(generation_bindings)
        return [
            EvidenceHit(
                "video-1",
                f"lighthouse:{generation_bindings['video-1']['generation_id']}",
                1.0,
                3.0,
                "lighthouse",
                0.8,
                query,
            )
        ]

    def search(self, query: str, video_ids: list[str], *, limit: int):  # type: ignore[no-untyped-def]
        del query, limit
        self.legacy_calls.append(video_ids)
        return []


def _snapshot(visual_id: str, lighthouse_id: str) -> ExternalIndexReleaseSnapshot:
    return ExternalIndexReleaseSnapshot(
        job_backed_video_ids=frozenset({"video-1"}),
        visual_dense={
            "video-1": _descriptor("video-1", visual_id, lighthouse=False)
        },
        lighthouse={
            "video-1": _descriptor("video-1", lighthouse_id, lighthouse=True)
        },
    )


def _service(repository: _Repository, visual: _Visual, lighthouse: _Lighthouse) -> SearchService:
    return SearchService(
        repository,  # type: ignore[arg-type]
        _VectorIndex(),  # type: ignore[arg-type]
        visual_search=visual,
        moment_search=lighthouse,
    )


def test_product_search_ignores_provider_pointer_and_uses_one_db_release_snapshot() -> None:
    repository = _Repository(_snapshot("1" * 32, "2" * 32))
    visual = _Visual()
    lighthouse = _Lighthouse()

    results = _service(repository, visual, lighthouse).search(
        "player shoots",
        mode="visual",
        use_lighthouse=True,
    )

    assert results
    assert repository.snapshot_calls == 1
    assert visual.exact_calls[0]["video-1"]["generation_id"] == "1" * 32
    assert lighthouse.exact_calls[0]["video-1"]["generation_id"] == "2" * 32
    assert visual.legacy_calls == lighthouse.legacy_calls == []


def test_product_search_pins_all_old_then_next_query_observes_all_new() -> None:
    old = _snapshot("1" * 32, "2" * 32)
    new = _snapshot("3" * 32, "4" * 32)
    repository = _Repository(old)
    visual = _Visual(on_exact=lambda: setattr(repository, "snapshot", new))
    lighthouse = _Lighthouse()
    service = _service(repository, visual, lighthouse)

    service.search("player shoots", mode="visual", use_lighthouse=True)
    service.search("player shoots", mode="visual", use_lighthouse=True)

    assert [
        call["video-1"]["generation_id"] for call in visual.exact_calls
    ] == ["1" * 32, "3" * 32]
    assert [
        call["video-1"]["generation_id"] for call in lighthouse.exact_calls
    ] == ["2" * 32, "4" * 32]


def test_job_backed_search_missing_exact_binding_fails_without_legacy_fallback() -> None:
    repository = _Repository(
        ExternalIndexReleaseSnapshot(
            job_backed_video_ids=frozenset({"video-1"}),
            visual_dense={},
            lighthouse={},
        )
    )
    visual = _Visual()
    lighthouse = _Lighthouse()

    with pytest.raises(SearchDependencyError, match="release binding"):
        _service(repository, visual, lighthouse).search(
            "player shoots",
            mode="visual",
            use_lighthouse=True,
        )

    assert visual.legacy_calls == lighthouse.legacy_calls == []


def test_legacy_video_retains_provider_active_pointer_search() -> None:
    repository = _Repository(ExternalIndexReleaseSnapshot(frozenset(), {}, {}))
    visual = _Visual()
    lighthouse = _Lighthouse()

    _service(repository, visual, lighthouse).search(
        "player shoots",
        mode="visual",
        use_lighthouse=True,
    )

    assert visual.exact_calls == lighthouse.exact_calls == []
    assert visual.legacy_calls == [["video-1"]]
    assert lighthouse.legacy_calls == [["video-1"]]
