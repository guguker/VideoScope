from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from threading import Barrier, Lock
import time
from types import SimpleNamespace

import pytest

import videoscope.providers.qwen_video as qwen_video_module
from videoscope.media.ffmpeg import SampledFrame
from videoscope.providers.base import ProviderState
from videoscope.providers.qwen_video import (
    QwenInferenceStatus,
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


class FakeQwenInference:
    identity = {
        "mode": "isolated-worker",
        "contract": "qwen-worker-v4",
        "model": "test-model",
        "runtime_identity": "mlx-vlm==0.6.7",
        "source_bundle_sha256": "1" * 64,
        "prompt_protocol_sha256": "2" * 64,
        "input_root_sha256": "3" * 64,
    }

    def __init__(self) -> None:
        self.video_calls: list[tuple[Path, float, int]] = []
        self.storyboard_calls: list[tuple[Path, str, int]] = []

    def status(self) -> QwenInferenceStatus:
        return QwenInferenceStatus(True, "worker ready")

    def judge_video(
        self,
        source: Path,
        *,
        fps: float,
        max_tokens: int,
    ) -> QwenVideoJudgement:
        self.video_calls.append((source, fps, max_tokens))
        return QwenVideoJudgement(
            shot_attempt=True,
            ball_through_hoop=True,
            evidence="ball passes through hoop",
        )

    def judge_video_query(
        self,
        source: Path,
        query: str,
        *,
        fps: float,
        max_tokens: int,
    ) -> QwenVideoJudgement:
        self.video_calls.append((source, fps, max_tokens))
        return QwenVideoJudgement(
            matches_query=True,
            confidence=0.9,
            evidence=query,
        )

    def judge_storyboard(
        self,
        source: Path,
        query: str,
        *,
        max_tokens: int,
    ) -> QwenVideoJudgement:
        self.storyboard_calls.append((source, query, max_tokens))
        return QwenVideoJudgement(
            matches_query=True,
            confidence=0.9,
            evidence="visible dunk",
        )


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
    model_revision: str | None = None,
    top_candidates: int = 12,
    frame_count: int = 12,
    max_tokens: int = 320,
    video_fps: float = 2,
) -> QwenVideoReranker:
    return QwenVideoReranker(
        model_name=model_name,
        model_revision=model_revision,
        repository=repository_with_video(tmp_path),
        extractor=extractor,
        temp_dir=tmp_path / "tmp",
        cache_dir=tmp_path / "cache",
        top_candidates=top_candidates,
        frame_count=frame_count,
        max_tokens=max_tokens,
        context_seconds=4,
        min_clip_seconds=7,
        max_clip_seconds=12,
        video_fps=video_fps,
        allow_in_process=True,
    )


def test_benchmark_attestation_is_bound_to_exact_isolated_worker(tmp_path: Path) -> None:
    reranker = QwenVideoReranker(
        model_name="test-model",
        repository=repository_with_video(tmp_path),
        extractor=RecordingMediaExtractor(),
        temp_dir=tmp_path / "tmp",
        cache_dir=tmp_path / "cache",
        top_candidates=7,
        inference_client=FakeQwenInference(),
    )

    attestation = reranker.benchmark_attestation

    assert attestation is not None
    assert attestation["provider"] == "qwen-video"
    assert attestation["model_identity"] == "test-model"
    assert attestation["runtime_identity"] == "mlx-vlm==0.6.7"
    assert attestation["source_bundle_sha256"] == "1" * 64
    assert attestation["prompt_protocol_sha256"] == "2" * 64
    assert attestation["input_root_sha256"] == "3" * 64
    assert attestation["candidate_limit"] == 7
    assert attestation["source_bound"] is True
    assert attestation["strict_complete"] is True
    assert str(attestation["protocol_identity"]).startswith("sha256:")
    assert "endpoint" not in attestation


def test_prompt_protocol_identity_covers_exact_prompt_templates() -> None:
    encoded = json.dumps(
        qwen_video_module._QWEN_PROMPT_PROTOCOL,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert qwen_video_module.QWEN_PROMPT_PROTOCOL_SHA256 == sha256(encoded).hexdigest()
    assert (
        qwen_video_module.QwenVideoReranker._fact_prompt()
        == qwen_video_module._QWEN_PROMPT_PROTOCOL["basketball_prompt"]
    )


def test_benchmark_attestation_rejects_disabled_or_wrong_model_worker(
    tmp_path: Path,
) -> None:
    disabled = QwenVideoReranker(
        model_name="test-model",
        repository=repository_with_video(tmp_path / "disabled"),
        extractor=RecordingMediaExtractor(),
        temp_dir=tmp_path / "disabled" / "tmp",
    )
    mismatched_worker = FakeQwenInference()
    mismatched_worker.identity = {
        **mismatched_worker.identity,
        "model": "different-model",
    }
    mismatched = QwenVideoReranker(
        model_name="test-model",
        repository=repository_with_video(tmp_path / "mismatch"),
        extractor=RecordingMediaExtractor(),
        temp_dir=tmp_path / "mismatch" / "tmp",
        inference_client=mismatched_worker,
    )

    assert disabled.benchmark_attestation is None
    assert mismatched.benchmark_attestation is None


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


def test_parser_rejects_truncated_fenced_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        parse_qwen_judgement(
            '```json\n{"evidence": "The image is a solid'
        )


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_parser_rejects_non_finite_numeric_values(literal: str) -> None:
    result = parse_qwen_judgement(
        f'{{"confidence":{literal},"event_start":{literal},"event_end":{literal}}}'
    )

    assert result.confidence == 0.0
    assert result.event_start is None
    assert result.event_end is None


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

    revision = "c" * 40
    calls: list[dict[str, object]] = []

    def snapshot_download(*_args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        snapshot_download,
    )
    reranker = QwenVideoReranker(
        model_name="organization/sharded-model",
        model_revision=revision,
        repository=repository_with_video(tmp_path),
        extractor=RecordingMediaExtractor(),
        temp_dir=tmp_path / "tmp",
        cache_dir=tmp_path / "cache",
        allow_in_process=True,
    )

    assert reranker.status().state is ProviderState.READY
    assert calls == [{"revision": revision, "local_files_only": True}]


def test_default_qwen_path_never_imports_mlx_and_requires_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reranker = QwenVideoReranker(
        model_name="test-model",
        repository=repository_with_video(tmp_path),
        extractor=RecordingMediaExtractor(),
        temp_dir=tmp_path / "tmp",
    )
    monkeypatch.setattr(
        qwen_video_module.importlib.util,
        "find_spec",
        lambda _name: (_ for _ in ()).throw(AssertionError("MLX probe escaped")),
    )
    item = candidate("dunk", 10, 16, 0.7)

    assert reranker.status().state is ProviderState.NEEDS_CONFIGURATION
    assert reranker.rerank("игрок выполняет данк", [item]) == [item]


def test_isolated_worker_preserves_native_and_storyboard_candidate_paths(
    tmp_path: Path,
) -> None:
    extractor = RecordingMediaExtractor()
    inference = FakeQwenInference()
    reranker = QwenVideoReranker(
        model_name="test-model",
        repository=repository_with_video(tmp_path),
        extractor=extractor,
        temp_dir=tmp_path / "tmp",
        cache_dir=tmp_path / "cache",
        inference_client=inference,
    )

    sports = reranker.rerank_strict(
        "игрок забивает трехочковый",
        [candidate("three", 10, 16, 0.7, event_type="made_three_point")],
    )
    generic = reranker.rerank_strict(
        "игрок выполняет данк",
        [candidate("dunk", 30, 36, 0.65)],
    )

    assert reranker.status().state is ProviderState.READY
    assert len(inference.video_calls) == 1
    assert inference.video_calls[0][1:] == (2.0, 320)
    assert len(inference.storyboard_calls) == 1
    assert inference.storyboard_calls[0][1:] == ("игрок выполняет данк", 320)
    assert "qwen_video" in sports[0].modalities
    assert "qwen_video" in generic[0].modalities
    assert reranker.identity["boundary"] == inference.identity


def test_model_is_loaded_once_when_first_requests_arrive_concurrently(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reranker = make_reranker(tmp_path, RecordingMediaExtractor())
    start = Barrier(3)
    calls = 0
    calls_lock = Lock()

    def load(_model_name: str):  # type: ignore[no-untyped-def]
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        time.sleep(0.05)
        return f"model-{call_number}", f"processor-{call_number}"

    monkeypatch.setitem(__import__("sys").modules, "mlx_vlm", SimpleNamespace(load=load))

    def first_request():  # type: ignore[no-untyped-def]
        start.wait()
        return reranker._load()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(first_request) for _ in range(2)]
        start.wait()
        results = [future.result(timeout=2) for future in futures]

    assert calls == 1
    assert results == [("model-1", "processor-1"), ("model-1", "processor-1")]


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


def test_generic_prompt_is_input_neutral_for_storyboards_and_native_video() -> None:
    prompt = QwenVideoReranker._generic_prompt("a person waves")

    assert "a person waves" in prompt
    assert "visual evidence" in prompt
    assert "storyboard" not in prompt.casefold()


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


def test_generic_storyboard_ignores_event_interval_outside_clip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    extractor = RecordingMediaExtractor()
    reranker = make_reranker(tmp_path, extractor)
    monkeypatch.setattr(
        reranker,
        "_judge_storyboard",
        lambda _storyboard, _query: QwenVideoJudgement(
            matches_query=True,
            confidence=0.8,
            event_start=100.0,
            event_end=101.0,
        ),
    )
    original = candidate("dunk", 10, 11, 0.65)

    reranked = reranker.rerank("игрок выполняет данк", [original])

    assert (reranked[0].start, reranked[0].end) == (original.start, original.end)
    assert reranked[0].evidence[0].end > reranked[0].evidence[0].start


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
        allow_in_process=True,
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
        prompt_version=f"{qwen_video_module.MADE_BASKET_PROMPT_VERSION}-fps2",
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
        allow_in_process=True,
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


def test_cache_key_changes_with_model_revision(tmp_path: Path) -> None:
    extractor = RecordingMediaExtractor()
    first = make_reranker(
        tmp_path,
        extractor,
        model_name="organization/model",
        model_revision="a" * 40,
    )
    second = QwenVideoReranker(
        model_name="organization/model",
        model_revision="b" * 40,
        repository=first.repository,
        extractor=extractor,
        temp_dir=tmp_path / "tmp-second",
        cache_dir=tmp_path / "cache",
        allow_in_process=True,
    )
    parameters = {
        "video_id": "video-1",
        "interval": (1.0, 4.0),
        "prompt_version": "prompt-v1",
    }

    first_key = first._cache_key(**parameters)
    second_key = second._cache_key(**parameters)

    assert first_key["model"] == "organization/model@" + "a" * 40
    assert second_key["model"] == "organization/model@" + "b" * 40
    assert first._cache_path(first_key) != second._cache_path(second_key)


def test_cache_key_includes_versioned_inference_config(tmp_path: Path) -> None:
    extractor = RecordingMediaExtractor()
    first = make_reranker(tmp_path, extractor)

    def configured(**kwargs):  # type: ignore[no-untyped-def]
        return QwenVideoReranker(
            model_name="test-model",
            repository=first.repository,
            extractor=extractor,
            temp_dir=tmp_path / "tmp",
            cache_dir=tmp_path / "cache",
            context_seconds=4,
            min_clip_seconds=7,
            max_clip_seconds=12,
            allow_in_process=True,
            **kwargs,
        )

    variants = [
        first,
        configured(frame_count=8),
        configured(max_tokens=256),
        configured(video_fps=1),
    ]
    parameters = {
        "video_id": "video-1",
        "interval": (1.0, 4.0),
        "prompt_version": "prompt-v1",
    }

    keys = [reranker._cache_key(**parameters) for reranker in variants]

    assert keys[0]["inference_config"] == {
        "execution_identity": {
            "mode": "deprecated-in-process",
            "runtime_identity": qwen_video_module.QWEN_IN_PROCESS_RUNTIME_IDENTITY,
        },
        "input_schema": "qwen-video-input-v3",
        "runtime_identity": qwen_video_module.QWEN_IN_PROCESS_RUNTIME_IDENTITY,
        "frame_count": 12,
        "max_tokens": 320,
        "video_fps": 2,
    }
    assert len({reranker._cache_path(key) for reranker, key in zip(variants, keys)}) == 4


@pytest.mark.parametrize(
    "identity_field",
    (
        "contract",
        "runtime_identity",
        "source_bundle_sha256",
        "prompt_protocol_sha256",
        "input_root_sha256",
    ),
)
def test_worker_cache_key_binds_complete_sanitized_execution_identity(
    tmp_path: Path,
    identity_field: str,
) -> None:
    extractor = RecordingMediaExtractor()
    repository = repository_with_video(tmp_path)
    first_client = FakeQwenInference()
    first_client.identity = dict(FakeQwenInference.identity)
    second_client = FakeQwenInference()
    second_client.identity = dict(FakeQwenInference.identity)
    second_client.identity[identity_field] = (
        "qwen-worker-v5"
        if identity_field == "contract"
        else "mlx-vlm==0.6.8"
        if identity_field == "runtime_identity"
        else "4" * 64
    )

    def reranker(client: FakeQwenInference, suffix: str) -> QwenVideoReranker:
        return QwenVideoReranker(
            model_name="test-model",
            repository=repository,
            extractor=extractor,
            temp_dir=tmp_path / f"tmp-{suffix}",
            cache_dir=tmp_path / "cache",
            inference_client=client,
        )

    parameters = {
        "video_id": "video-1",
        "interval": (1.0, 4.0),
        "prompt_version": "prompt-v1",
    }
    first = reranker(first_client, "first")
    second = reranker(second_client, "second")
    first_key = first._cache_key(**parameters)
    second_key = second._cache_key(**parameters)

    assert first._cache_path(first_key) != second._cache_path(second_key)
    boundary = first_key["inference_config"]["execution_identity"]
    assert boundary == first.benchmark_attestation
    assert "endpoint" not in boundary


def test_rerank_strict_surfaces_candidate_failure(tmp_path: Path, monkeypatch) -> None:
    extractor = RecordingMediaExtractor()
    reranker = make_reranker(tmp_path, extractor)
    item = candidate("dunk", 10, 16, 0.7)
    monkeypatch.setattr(
        reranker,
        "_judge_storyboard",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("Qwen inference failed")),
    )

    assert reranker.rerank("игрок выполняет данк", [item]) == [item]
    with pytest.raises(RuntimeError, match="Qwen inference failed"):
        reranker.rerank_strict("игрок выполняет данк", [item])


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
