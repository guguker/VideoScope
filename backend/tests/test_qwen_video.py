import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import videoscope.providers.qwen_video as qwen_video_module
from videoscope.media.ffmpeg import SampledFrame
from videoscope.providers.base import ProviderState
from videoscope.providers.qwen_video import (
    QwenVideoJudgement,
    QwenVideoReranker,
    parse_qwen_judgement,
)
from videoscope.repository import Repository
from videoscope.search.fusion import EvidenceHit, FusedResult


class RecordingMediaExtractor:
    def __init__(self) -> None:
        self.frame_intervals: list[tuple[float, float]] = []
        self.clip_intervals: list[tuple[float, float]] = []

    def export_clip(
        self,
        _source: Path,
        destination: Path,
        start: float,
        end: float,
    ) -> None:
        self.clip_intervals.append((start, end))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"clip")

    def extract_frames(
        self,
        _source: Path,
        destination: Path,
        start: float,
        end: float,
        *,
        step: float,
        max_width: int = 640,
    ) -> list[SampledFrame]:
        self.frame_intervals.append((start, end))
        destination.mkdir(parents=True, exist_ok=True)
        frames: list[SampledFrame] = []
        for index, timestamp in enumerate((start, (start + end) / 2, end)):
            path = destination / f"frame-{index}.jpg"
            from PIL import Image

            Image.new("RGB", (max_width, 360), "black").save(path)
            frames.append(SampledFrame(timestamp, path))
        return frames


def repository_with_video(tmp_path: Path) -> Repository:
    repository = Repository(tmp_path / "videos.sqlite3")
    repository.initialize()
    source = tmp_path / "match.mp4"
    source.write_bytes(b"video")
    repository.create_video(
        video_id="match",
        original_name="match.mp4",
        stored_name="match.mp4",
        media_path=str(source),
        size_bytes=5,
    )
    repository.update_video("match", duration=120)
    return repository


def candidate(
    segment_id: str,
    start: float,
    end: float,
    score: float,
    *,
    event_type: str | None = None,
) -> FusedResult:
    metadata: dict[str, object] = {"source": "siglip2"}
    if event_type:
        metadata["event_type"] = event_type
    evidence = EvidenceHit(
        "match",
        segment_id,
        start,
        end,
        "visual",
        score,
        "query",
        metadata,
    )
    return FusedResult("match", start, end, score, ["visual"], [evidence])


def make_reranker(
    tmp_path: Path,
    extractor: RecordingMediaExtractor,
    *,
    model_name: str = "test-model",
    top_candidates: int = 12,
) -> QwenVideoReranker:
    return QwenVideoReranker(
        model_name=model_name,
        repository=repository_with_video(tmp_path),
        extractor=extractor,
        temp_dir=tmp_path / "tmp",
        cache_dir=tmp_path / "cache",
        top_candidates=top_candidates,
        context_seconds=4,
        min_clip_seconds=7,
        max_clip_seconds=12,
        video_fps=2,
    )


def test_parser_accepts_independent_fact_only_json() -> None:
    result = parse_qwen_judgement(
        """```json
        {"shot_attempt": "yes", "ball_through_hoop": true,
         "shooter_outside_arc": false, "three_point_signal": null,
         "shooter_jersey": "15", "evidence": "ball passes through rim"}
        ```"""
    )

    assert result.shot_attempt is True
    assert result.ball_through_hoop is True
    assert result.shooter_outside_arc is False
    assert result.three_point_signal is None
    assert result.shooter_jersey == "15"


@pytest.mark.parametrize(
    ("query", "event_type"),
    [
        ("игрок забивает трехочковый", "made_three_point"),
        ("player makes a three-point shot", "made_three_point"),
        ("игрок забивает двухочковый", "made_two_point"),
        ("player makes a two-point shot", "made_two_point"),
        ("игрок забивает штрафной бросок", "made_free_throw"),
        ("player makes a free throw", "made_free_throw"),
    ],
)
def test_sports_query_is_verified_by_matching_event_and_made_basket(
    query: str,
    event_type: str,
) -> None:
    result = QwenVideoReranker.resolve_match(
        query,
        candidate("made-basket", 30, 36, 0.70, event_type=event_type),
        QwenVideoJudgement(
            shot_attempt=True,
            ball_through_hoop=True,
            shooter_outside_arc=False,
            evidence="мяч проходит через кольцо",
        ),
    )

    assert result.matches_query is True


@pytest.mark.parametrize(
    ("query", "candidate_event"),
    [
        ("игрок забивает трехочковый", "made_two_point"),
        ("player makes a two-point shot", "made_free_throw"),
        ("игрок забивает штрафной бросок", "made_three_point"),
        ("player makes a free throw", None),
    ],
)
def test_sports_query_rejects_mismatched_candidate_event(
    query: str,
    candidate_event: str | None,
) -> None:
    result = QwenVideoReranker.resolve_match(
        query,
        candidate(
            "wrong-event",
            30,
            36,
            0.70,
            event_type=candidate_event,
        ),
        QwenVideoJudgement(
            shot_attempt=True,
            ball_through_hoop=True,
            shooter_outside_arc=True,
            three_point_signal=True,
            shooter_jersey="15",
            evidence="модель ошибочно считает штрафной дальним броском",
        ),
    )

    assert result.matches_query is False


@pytest.mark.parametrize(
    ("query", "event_type"),
    [
        ("игрок забивает двухочковый", "made_two_point"),
        ("player makes a free throw", "made_free_throw"),
    ],
)
def test_unknown_made_basket_stays_inconclusive(
    query: str,
    event_type: str,
) -> None:
    result = QwenVideoReranker.resolve_match(
        query,
        candidate("made-basket", 30, 36, 0.70, event_type=event_type),
        QwenVideoJudgement(shot_attempt=True, ball_through_hoop=None),
    )

    assert result.matches_query is None


def test_status_accepts_cached_model_with_sharded_weights(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        qwen_video_module.importlib.util,
        "find_spec",
        lambda _name: object(),
    )
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *_args, **_kwargs: str(snapshot),
    )
    reranker = QwenVideoReranker(
        model_name="organization/sharded-model",
        repository=repository_with_video(tmp_path),
        extractor=RecordingMediaExtractor(),
        temp_dir=tmp_path / "tmp",
        cache_dir=tmp_path / "cache",
    )

    assert reranker.status().state is ProviderState.READY


def test_native_video_judge_disables_thinking_and_uses_two_fps(
    tmp_path: Path, monkeypatch
) -> None:
    extractor = RecordingMediaExtractor()
    reranker = make_reranker(tmp_path, extractor)
    calls: dict[str, object] = {}
    fake_model = SimpleNamespace(config=SimpleNamespace())
    fake_processor = object()

    def fake_template(processor, config, prompts, **kwargs):
        calls["template"] = (processor, config, prompts, kwargs)
        return "formatted"

    def fake_generate(model, processor, prompt, **kwargs):
        calls["generate"] = (model, processor, prompt, kwargs)
        return SimpleNamespace(
            text=(
                '{"shot_attempt":true,"ball_through_hoop":true,'
                '"shooter_outside_arc":null,"three_point_signal":null,'
                '"shooter_jersey":null,"evidence":"rim"}'
            )
        )

    fake_module = SimpleNamespace(
        apply_chat_template=fake_template,
        generate=fake_generate,
    )
    monkeypatch.setitem(__import__("sys").modules, "mlx_vlm", fake_module)
    monkeypatch.setattr(reranker, "_load", lambda: (fake_model, fake_processor))
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")

    result = reranker._judge_video(clip)

    assert result.ball_through_hoop is True
    template_kwargs = calls["template"][3]
    assert template_kwargs["video"] == str(clip)
    assert template_kwargs["fps"] == 2
    assert template_kwargs["enable_thinking"] is False
    generate_kwargs = calls["generate"][3]
    assert generate_kwargs["video"] == [str(clip)]
    assert generate_kwargs["fps"] == 2
    assert generate_kwargs["enable_thinking"] is False
    prompt_text = calls["template"][2][0]
    assert "matches_query" not in prompt_text
    assert "confidence" not in prompt_text


@pytest.mark.parametrize(
    ("query", "event_type"),
    [
        ("игрок забивает трехочковый", "made_three_point"),
        ("player makes a two-point shot", "made_two_point"),
        ("игрок забивает штрафной бросок", "made_free_throw"),
    ],
)
def test_sports_event_uses_native_clip_and_marks_jersey_as_possible(
    tmp_path: Path,
    monkeypatch,
    query: str,
    event_type: str,
) -> None:
    extractor = RecordingMediaExtractor()
    reranker = make_reranker(tmp_path, extractor)
    monkeypatch.setattr(
        reranker,
        "_judge_video",
        lambda _clip: QwenVideoJudgement(
            shot_attempt=True,
            ball_through_hoop=True,
            shooter_outside_arc=None,
            shooter_jersey="15",
            evidence="мяч проходит через кольцо",
        ),
    )

    reranked = reranker.rerank(
        query,
        [candidate("made-basket", 30, 36, 0.70, event_type=event_type)],
    )

    assert extractor.clip_intervals == [(27, 39)]
    assert extractor.frame_intervals == []
    assert reranked[0].score > 0.70
    evidence = reranked[0].evidence[0]
    assert evidence.metadata["possible_shooter_jersey"] == "15"
    assert evidence.metadata["shooter_jersey_confirmed"] is False
    assert evidence.metadata["model_evidence"] == "мяч проходит через кольцо"
    assert "номер игрока" not in evidence.text
    assert "мяч проходит через кольцо" not in evidence.text


def test_other_action_query_keeps_generic_storyboard_flow(
    tmp_path: Path, monkeypatch
) -> None:
    extractor = RecordingMediaExtractor()
    reranker = make_reranker(tmp_path, extractor)
    monkeypatch.setattr(
        reranker,
        "_judge_storyboard",
        lambda _storyboard, _query: QwenVideoJudgement(
            matches_query=True,
            confidence=0.8,
            shot_attempt=True,
            evidence="игрок выполняет данк",
        ),
    )

    reranker.rerank(
        "игрок выполняет данк",
        [candidate("dunk", 10, 11, 0.65)],
    )

    assert extractor.frame_intervals == [(6, 15)]
    assert extractor.clip_intervals == []


def test_persistent_cache_avoids_second_export_and_model_call(
    tmp_path: Path, monkeypatch
) -> None:
    first_extractor = RecordingMediaExtractor()
    first = make_reranker(tmp_path, first_extractor)
    calls = 0

    def judge_once(_clip: Path) -> QwenVideoJudgement:
        nonlocal calls
        calls += 1
        return QwenVideoJudgement(
            shot_attempt=True,
            ball_through_hoop=True,
            evidence="мяч проходит через кольцо",
        )

    monkeypatch.setattr(first, "_judge_video", judge_once)
    item = candidate(
        "made-three",
        30,
        36,
        0.70,
        event_type="made_three_point",
    )
    first.rerank("игрок забивает трехочковый", [item])

    second_extractor = RecordingMediaExtractor()
    second = QwenVideoReranker(
        model_name="test-model",
        repository=first.repository,
        extractor=second_extractor,
        temp_dir=tmp_path / "tmp-2",
        cache_dir=tmp_path / "cache",
        context_seconds=4,
        min_clip_seconds=7,
        max_clip_seconds=12,
        video_fps=2,
    )
    monkeypatch.setattr(
        second,
        "_judge_video",
        lambda _clip: (_ for _ in ()).throw(AssertionError("cache miss")),
    )

    second.rerank("игрок забивает трехочковый", [item])

    assert calls == 1
    assert len(first_extractor.clip_intervals) == 1
    assert second_extractor.clip_intervals == []
    cache_files = list((tmp_path / "cache").glob("*.json"))
    assert len(cache_files) == 1
    payload = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert payload["key"]["model"] == "test-model"
    assert payload["key"]["video_id"] == "match"
    assert payload["key"]["prompt_version"].startswith("made-basket-facts-")


def test_current_fact_prompt_reuses_and_promotes_legacy_v2_cache(
    tmp_path: Path, monkeypatch
) -> None:
    extractor = RecordingMediaExtractor()
    reranker = make_reranker(tmp_path, extractor)
    item = candidate(
        "made-three",
        30,
        36,
        0.70,
        event_type="made_three_point",
    )
    interval = reranker._clip_interval(item)
    assert interval is not None
    legacy_key = reranker._cache_key(
        video_id=item.video_id,
        interval=interval,
        prompt_version="made-three-facts-v2-fps2",
    )
    reranker._save_cached(
        legacy_key,
        QwenVideoJudgement(
            shot_attempt=True,
            ball_through_hoop=True,
            evidence="мяч проходит через кольцо",
        ),
    )
    monkeypatch.setattr(
        reranker,
        "_judge_video",
        lambda _clip: (_ for _ in ()).throw(AssertionError("model call")),
    )

    reranked = reranker.rerank("игрок забивает трехочковый", [item])

    assert extractor.clip_intervals == []
    assert reranked[0].evidence[0].metadata["cache_hit"] is True
    current_key = reranker._cache_key(
        video_id=item.video_id,
        interval=interval,
        prompt_version="made-basket-facts-v3-fps2",
    )
    current_path = reranker._cache_path(current_key)
    assert current_path.is_file()
    promoted = json.loads(current_path.read_text(encoding="utf-8"))
    assert promoted["key"] == current_key


def test_cache_key_changes_with_model_name(tmp_path: Path, monkeypatch) -> None:
    extractor = RecordingMediaExtractor()
    first = make_reranker(tmp_path, extractor, model_name="model-a")
    monkeypatch.setattr(
        first,
        "_judge_video",
        lambda _clip: QwenVideoJudgement(
            shot_attempt=True,
            ball_through_hoop=True,
        ),
    )
    item = candidate(
        "made-three",
        30,
        36,
        0.70,
        event_type="made_three_point",
    )
    first.rerank("игрок забивает трехочковый", [item])

    second = QwenVideoReranker(
        model_name="model-b",
        repository=first.repository,
        extractor=extractor,
        temp_dir=tmp_path / "tmp-2",
        cache_dir=tmp_path / "cache",
    )
    model_b_calls = 0

    def judge_model_b(_clip: Path) -> QwenVideoJudgement:
        nonlocal model_b_calls
        model_b_calls += 1
        return QwenVideoJudgement(shot_attempt=True, ball_through_hoop=True)

    monkeypatch.setattr(second, "_judge_video", judge_model_b)
    second.rerank("игрок забивает трехочковый", [item])

    assert model_b_calls == 1
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2


def test_conservative_scoring_keeps_inconclusive_and_unverified_candidates(
    tmp_path: Path, monkeypatch
) -> None:
    extractor = RecordingMediaExtractor()
    reranker = make_reranker(tmp_path, extractor, top_candidates=2)
    judgements = iter(
        [
            QwenVideoJudgement(
                shot_attempt=True,
                ball_through_hoop=False,
                evidence="попадание не видно",
            ),
            QwenVideoJudgement(
                shot_attempt=False,
                ball_through_hoop=False,
                evidence="броска нет",
            ),
        ]
    )
    monkeypatch.setattr(reranker, "_judge_video", lambda _clip: next(judgements))

    reranked = reranker.rerank(
        "игрок забивает трехочковый",
        [
            candidate("inconclusive", 10, 16, 0.90, event_type="made_three_point"),
            candidate("no-shot", 30, 36, 0.80, event_type="made_three_point"),
            candidate("tail", 50, 56, 0.70, event_type="made_three_point"),
        ],
    )
    by_start = {item.start: item for item in reranked}

    assert by_start[10].score >= 0.81
    assert by_start[30].score < 0.80
    assert by_start[50].score == 0.70


def test_reranker_selects_strongest_candidates_instead_of_input_order(
    tmp_path: Path, monkeypatch
) -> None:
    extractor = RecordingMediaExtractor()
    reranker = make_reranker(tmp_path, extractor, top_candidates=1)
    monkeypatch.setattr(
        reranker,
        "_judge_video",
        lambda _clip: QwenVideoJudgement(
            shot_attempt=True,
            ball_through_hoop=True,
            evidence="мяч проходит через кольцо",
        ),
    )

    reranked = reranker.rerank(
        "игрок забивает трехочковый",
        [
            candidate("weaker-first", 10, 16, 0.70, event_type="made_three_point"),
            candidate("stronger-second", 30, 36, 0.90, event_type="made_three_point"),
        ],
    )
    by_start = {item.start: item for item in reranked}

    assert "qwen_video" not in by_start[10].modalities
    assert "qwen_video" in by_start[30].modalities


def test_verified_candidate_precedes_higher_scored_unverified_noise(
    tmp_path: Path, monkeypatch
) -> None:
    extractor = RecordingMediaExtractor()
    reranker = make_reranker(tmp_path, extractor, top_candidates=1)
    judgement = QwenVideoJudgement(
            shot_attempt=True,
            ball_through_hoop=True,
            evidence="мяч проходит через кольцо",
        )
    monkeypatch.setattr(
        reranker,
        "_judge_video",
        lambda _clip: judgement,
    )
    monkeypatch.setattr(
        reranker,
        "_judge_storyboard",
        lambda _storyboard, _query: judgement,
    )

    reranked = reranker.rerank(
        "игрок забивает трехочковый",
        [
            candidate("unverified-noise", 10, 16, 0.99),
            candidate("verified-event", 30, 36, 0.80, event_type="made_three_point"),
        ],
    )

    assert reranked[0].start == 30
    assert reranked[0].evidence[0].metadata["matches_query"] is True


def test_explicit_no_shot_is_sorted_below_unverified_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    extractor = RecordingMediaExtractor()
    reranker = make_reranker(tmp_path, extractor, top_candidates=1)
    monkeypatch.setattr(
        reranker,
        "_judge_video",
        lambda _clip: QwenVideoJudgement(
            shot_attempt=False,
            ball_through_hoop=False,
            evidence="броска нет",
        ),
    )

    reranked = reranker.rerank(
        "игрок забивает трехочковый",
        [
            candidate("no-shot", 10, 16, 0.99, event_type="made_three_point"),
            candidate("unverified", 30, 36, 0.80, event_type="made_three_point"),
        ],
    )

    assert reranked[0].start == 30
    assert reranked[-1].evidence[0].metadata["shot_attempt"] is False
