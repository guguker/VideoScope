from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from videoscope.providers.qwen_video import QwenVideoJudgement, QwenVideoReranker
from videoscope.repository import Repository
from videoscope.search.fusion import EvidenceHit, FusedResult
from videoscope.search.query_router import QueryPlan
from videoscope.search.service import SearchDependencyError, SearchService
from videoscope.search.vector_index import EmptyVectorIndex


def _candidates() -> list[FusedResult]:
    return [
        FusedResult(
            video_id="video-1", start=start, end=start + 2, score=0.9 - index / 10,
            modalities=["visual"],
            evidence=[EvidenceHit(
                video_id="video-1", segment_id=f"source-{index}",
                start=start, end=start + 2, modality="visual", score=0.9,
                text=f"original evidence {index}",
                metadata={"source": "synthetic-visual", "nested": {"values": ["original"]}},
            )],
        )
        for index, start in enumerate((10.0, 30.0))
    ]


def _service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack):  # type: ignore[no-untyped-def]
    import videoscope.search.service as service_module

    repository = Repository(tmp_path / "database.sqlite3")
    repository.initialize()
    for video_id in ("video-1", "video-2"):
        source = tmp_path / f"{video_id}.mp4"
        source.write_bytes(b"synthetic media; no decoder is called")
        repository.create_video(
            video_id=video_id, original_name=source.name, stored_name=source.name,
            media_path=str(source), size_bytes=source.stat().st_size,
        )
        repository.update_video(video_id, status="ready", duration=60.0)
    fused = _candidates()
    calls: list[bool] = []

    class AdversarialQwen(QwenVideoReranker):
        def rerank(self, query, candidates):  # type: ignore[no-untyped-def]
            calls.append(True)
            return attack(self, query, candidates)

        rerank_strict = rerank

    reranker = AdversarialQwen(
        model_name="synthetic-model", repository=repository,
        extractor=SimpleNamespace(), temp_dir=tmp_path / "reranker-tmp",
        context_seconds=4, min_clip_seconds=7, max_clip_seconds=12,
        allow_in_process=True,
    )
    router = SimpleNamespace(route=lambda query, **_kwargs: QueryPlan(
        query=query, intent="action", modalities=frozenset({"visual"}),
        modality_weights={"visual": 1.0}, use_lighthouse=False,
        refine_temporally=False, explanation="synthetic candidate contract test",
    ))
    visual = SimpleNamespace(search=lambda *_args, **_kwargs: [])
    service = SearchService(
        repository, EmptyVectorIndex(), visual_search=visual,
        query_router=router, candidate_reranker=reranker,
    )
    monkeypatch.setattr(service_module, "fuse_hits", lambda *_args, **_kwargs: fused)
    return service, reranker, fused, calls


@pytest.mark.parametrize(
    "attack_kind",
    ["drop_in_place", "append_in_place", "duplicate", "foreign_same_shape", "cross_video", "missing_evidence", "changed_evidence", "nested_metadata_mutation"],
)
def test_reranker_rejects_membership_or_original_evidence_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack_kind: str
) -> None:
    def attack(_reranker, _query, candidates):  # type: ignore[no-untyped-def]
        first = candidates[0]
        if attack_kind == "drop_in_place":
            candidates.pop()
        elif attack_kind == "append_in_place":
            candidates.append(replace(first, video_id="video-2"))
        elif attack_kind == "duplicate":
            candidates[1] = first
        elif attack_kind == "foreign_same_shape":
            candidates[0] = FusedResult(
                first.video_id, first.start, first.end, first.score,
                list(first.modalities), list(first.evidence),
            )
        elif attack_kind == "cross_video":
            candidates[0] = replace(first, video_id="video-2")
        elif attack_kind == "missing_evidence":
            candidates[0] = replace(first, evidence=[])
        elif attack_kind == "changed_evidence":
            candidates[0] = replace(first, evidence=[replace(first.evidence[0], text="replacement")])
        else:
            first.evidence[0].metadata["nested"]["values"][0] = "attacked"
        return candidates

    service, _reranker, fused, calls = _service(tmp_path, monkeypatch, attack)
    original = deepcopy(fused)

    with pytest.raises(SearchDependencyError, match="Candidate reranking failed"):
        service.search_for_evaluation("person waves", mode="visual", use_lighthouse=False)

    assert calls == [True]
    assert fused == original


@pytest.mark.parametrize("start,end", [(0.0, 1.0), (59.0, 60.0), (float("nan"), 11.0), (10.0, float("inf")), (11.0, 11.0)])
def test_reranker_rejects_invalid_interval_despite_matching_qwen_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: float, end: float
) -> None:
    def attack(reranker, query, candidates):  # type: ignore[no-untyped-def]
        first = candidates[0]
        clip_start, clip_end = reranker._clip_interval(first)
        evidence = EvidenceHit(
            first.video_id, f"qwen-video:{reranker._candidate_id(first)}",
            start, end, "qwen_video", 0.9, "verified",
            {"source": "qwen-video-verifier", "model": reranker.model_identity,
             "prompt_version": reranker._prompt_version(query, False),
             "matches_query": True, "clip_start": clip_start, "clip_end": clip_end},
        )
        return [replace(first, start=start, end=end, evidence=[evidence, *first.evidence]), *candidates[1:]]

    service, _reranker, fused, calls = _service(tmp_path, monkeypatch, attack)
    original = deepcopy(fused)
    with pytest.raises(SearchDependencyError, match="Candidate reranking failed"):
        service.search_for_evaluation("person waves", mode="visual", use_lighthouse=False)
    assert calls == [True]
    assert fused == original


@pytest.mark.parametrize("exit_kind", ["raise", "invalid_result"])
def test_fallback_restores_original_candidates_after_in_place_attack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_kind: str
) -> None:
    def attack(_reranker, _query, candidates):  # type: ignore[no-untyped-def]
        first = candidates[0]
        first.evidence[0].metadata["nested"]["values"].append("attacked")
        first.evidence[0].metadata["source"] = "attacked"
        first.evidence.clear()
        first.modalities.clear()
        candidates.clear()
        if exit_kind == "raise":
            raise RuntimeError("synthetic malicious reranker")
        return "invalid result"

    service, _reranker, fused, calls = _service(tmp_path, monkeypatch, attack)
    original = deepcopy(fused)
    result = service.search("person waves", mode="visual", use_lighthouse=False)

    assert calls == [True]
    assert fused == original
    assert [(item.start, item.end, item.modalities) for item in result] == [
        (10.0, 12.0, ["visual"]), (30.0, 32.0, ["visual"])
    ]
    assert [item.evidence[0].text for item in result] == ["original evidence 0", "original evidence 1"]
    assert all(item.evidence[0].source == "synthetic-visual" for item in result)


def test_qwen_candidate_ids_preserve_exact_interval_without_ephemeral_token() -> None:
    first = replace(_candidates()[0], start=10.0001)
    distinct = replace(first, start=10.0002)
    assert QwenVideoReranker._candidate_id(first) != QwenVideoReranker._candidate_id(distinct)
    tagged_a = replace(first, _rerank_token=object())
    tagged_b = replace(first, _rerank_token=object())
    expected = QwenVideoReranker._candidate_id(first)
    assert QwenVideoReranker._candidate_id(tagged_a) == QwenVideoReranker._candidate_id(tagged_b) == expected
    assert "object at" not in expected


def test_actual_qwen_evidence_id_uses_original_interval_after_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _search, reranker, _fused, _calls = _service(tmp_path, monkeypatch, lambda *_args: None)
    item = replace(_candidates()[0], start=10.0001, _rerank_token=object())
    expected_id = f"qwen-video:{QwenVideoReranker._candidate_id(item)}"
    reranker.extractor = SimpleNamespace(extract_frames=lambda *_args, **_kwargs: [])
    monkeypatch.setattr(reranker, "_build_storyboard", lambda *_args: tmp_path / "storyboard.jpg")
    monkeypatch.setattr(reranker, "_judge_storyboard", lambda *_args: QwenVideoJudgement(
        matches_query=True, confidence=0.9, event_start=2.0, event_end=4.0,
    ))

    result = QwenVideoReranker.rerank_strict(reranker, "person waves", [item])[0]

    assert (result.start, result.end) != (item.start, item.end)
    assert result.evidence[0].segment_id == expected_id
    assert result.evidence[1:] == item.evidence
    assert "object at" not in result.evidence[0].segment_id


@pytest.mark.parametrize("field", ["score", "metadata"])
@pytest.mark.parametrize("strict", [True, False])
def test_reranker_preserves_original_evidence_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, strict: bool
) -> None:
    def attack(_reranker, _query, candidates):  # type: ignore[no-untyped-def]
        first = candidates[0]
        if field == "score":
            first.evidence[0] = replace(first.evidence[0], score=True)
        else:
            first.evidence[0].metadata["nested"]["visible_count"] = True
        return candidates

    service, _reranker, fused, calls = _service(tmp_path, monkeypatch, attack)
    fused[0].evidence[0] = replace(fused[0].evidence[0], score=1.0)
    fused[0].evidence[0].metadata["nested"]["visible_count"] = 1
    original = deepcopy(fused)
    if strict:
        with pytest.raises(SearchDependencyError, match="Candidate reranking failed"):
            service.search_for_evaluation("person waves", mode="visual", use_lighthouse=False)
    else:
        # Rejection must restore the original objects before views coerce values.
        import videoscope.search.service as service_module

        validated: list[FusedResult] = []
        original_validate = service_module._validate_reranked_candidates

        def recording_validate(candidates, bindings):  # type: ignore[no-untyped-def]
            result = original_validate(candidates, bindings)
            validated.extend(result)
            return result

        monkeypatch.setattr(service_module, "_validate_reranked_candidates", recording_validate)
        result = service.search("person waves", mode="visual", use_lighthouse=False)
        assert not validated
        assert len(result) == 2
        assert result[0].evidence[0].text == "original evidence 0"
    assert calls == [True]
    assert fused == original
    assert type(fused[0].evidence[0].score) is float
    assert type(fused[0].evidence[0].metadata["nested"]["visible_count"]) is int


@pytest.mark.parametrize("modalities", [[], ["qwen_video"], "visual", ["visual", True], ["visual", ""]])
@pytest.mark.parametrize("strict", [True, False])
def test_reranker_rejects_lost_or_invalid_modality_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modalities: object, strict: bool
) -> None:
    def attack(_reranker, _query, candidates):  # type: ignore[no-untyped-def]
        candidates[0] = replace(candidates[0], modalities=modalities)
        return candidates

    service, _reranker, fused, calls = _service(tmp_path, monkeypatch, attack)
    original = deepcopy(fused)
    if strict:
        with pytest.raises(SearchDependencyError, match="Candidate reranking failed"):
            service.search_for_evaluation("person waves", mode="visual", use_lighthouse=False)
    else:
        result = service.search("person waves", mode="visual", use_lighthouse=False)
        assert [item.modalities for item in result] == [["visual"], ["visual"]]
    assert calls == [True]
    assert fused == original


def test_reranker_allows_reordered_original_and_added_modality_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def attack(_reranker, _query, candidates):  # type: ignore[no-untyped-def]
        candidates[0] = replace(candidates[0], modalities=["qwen_video", "ocr", "visual"])
        return candidates

    service, _reranker, fused, calls = _service(tmp_path, monkeypatch, attack)
    fused[0].modalities.append("ocr")
    result = service.search_for_evaluation("person waves", mode="visual", use_lighthouse=False)
    assert calls == [True]
    assert result[0].modalities == ["qwen_video", "ocr", "visual"]


@pytest.mark.parametrize("attack_kind", ["clear_input", "append_input", "reverse_valid"])
def test_pinned_candidate_limit_keeps_tail_outside_provider_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack_kind: str
) -> None:
    import videoscope.search.service as service_module
    from test_search_service import (
        _PinnedLighthouseProvider, _PinnedReranker, _PinnedTemporalRefiner,
        _PinnedVisualProvider, _full_evaluation_configuration, _pinned_asset_binding,
        _ready_generation_search_service,
    )

    repository, specifications, index, _search = _ready_generation_search_service(tmp_path)
    fused = [replace(_candidates()[0], start=float(i), end=float(i + 1)) for i in range(6)]
    original = deepcopy(fused)
    received: list[list[float]] = []

    class InputMutatingReranker(_PinnedReranker):
        def rerank_strict(self, _query, candidates):  # type: ignore[no-untyped-def]
            received.append([item.start for item in candidates])
            if attack_kind == "clear_input":
                candidates[0].evidence[0].metadata["nested"]["values"].append("attacked")
                candidates.clear()
            elif attack_kind == "append_input":
                candidates.append(candidates[0])
            else:
                candidates.reverse()
            return candidates

    service = SearchService(
        repository, index, moment_search=_PinnedLighthouseProvider(),
        visual_search=_PinnedVisualProvider(), temporal_refiner=_PinnedTemporalRefiner(),
        evaluation_rerankers={"internvideo": InputMutatingReranker("internvideo", 4)},
        specification_resolver=lambda: specifications, media_root=tmp_path,
    )
    monkeypatch.setattr(service_module, "fuse_hits", lambda *_args, **_kwargs: fused)
    session = service.open_pinned_evaluation(
        _full_evaluation_configuration("internvideo"), (_pinned_asset_binding(),),
    )
    try:
        if attack_kind == "reverse_valid":
            result = session.search("person waves", ("video-1",), limit=20)
            assert [item.start for item in result] == [3.0, 2.0, 1.0, 0.0, 4.0, 5.0]
        else:
            with pytest.raises(SearchDependencyError, match="Candidate reranking failed"):
                session.search("person waves", ("video-1",), limit=20)
    finally:
        session.close()
    assert received == [[0.0, 1.0, 2.0, 3.0]]
    assert fused == original
    assert all(item._rerank_token is None for item in fused)
