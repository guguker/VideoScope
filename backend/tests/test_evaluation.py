from types import SimpleNamespace

import pytest

from videoscope.evaluation import (
    EvaluationCase,
    EvaluationService,
    EvaluationStore,
    temporal_iou,
)


class FakeSearch:
    def __init__(self, results):  # type: ignore[no-untyped-def]
        self.results = results
        self.calls: list[tuple[str, str, bool]] = []

    def search(self, query, *, video_ids=None, limit=20, use_lighthouse=True, mode="all"):  # type: ignore[no-untyped-def]
        del video_ids, limit
        self.calls.append((query, mode, use_lighthouse))
        return self.results.get(query, [])


def test_temporal_iou_handles_overlap_and_disjoint_intervals() -> None:
    assert temporal_iou(10, 20, 15, 25) == pytest.approx(5 / 15)
    assert temporal_iou(0, 5, 8, 10) == 0


def test_evaluation_computes_retrieval_and_temporal_metrics(tmp_path) -> None:
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")
    store.replace_cases([
        EvaluationCase("speech-1", "Мозгов", "video-1", 10, 14, "speech", "gold"),
        EvaluationCase("action-1", "поднимает руку", "video-1", 30, 34, "visual", "gold"),
    ])
    search = FakeSearch({
        "Мозгов": [
            SimpleNamespace(video_id="video-2", start=10, end=14),
            SimpleNamespace(video_id="video-1", start=9, end=15),
        ],
        "поднимает руку": [SimpleNamespace(video_id="video-1", start=31, end=35)],
    })
    service = EvaluationService(search, store)  # type: ignore[arg-type]

    report = service.run(["auto"])
    metrics = report.variants[0]

    assert metrics.case_count == 2
    assert metrics.recall_at_1 == pytest.approx(0.5)
    assert metrics.recall_at_3 == pytest.approx(1.0)
    assert metrics.mrr == pytest.approx(0.75)
    assert metrics.mean_temporal_iou > 0.5
    assert store.read_report() is not None


def test_evaluation_runs_requested_ablation_variants(tmp_path) -> None:
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")
    store.replace_cases([
        EvaluationCase("case", "query", "video-1", 0, 2, "all", "gold"),
    ])
    search = FakeSearch({"query": []})

    EvaluationService(search, store).run(["speech", "visual_lighthouse"])

    assert search.calls == [
        ("query", "speech", False),
        ("query", "visual", True),
    ]


def test_store_rejects_invalid_case_interval(tmp_path) -> None:
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")

    with pytest.raises(ValueError, match="interval"):
        store.replace_cases([
            EvaluationCase("bad", "query", "video-1", 10, 10, "all", "gold"),
        ])
