import json
import logging
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from videoscope.api_models import EvaluationPayloadResponse
from videoscope.evaluation import (
    EVALUATION_METHODOLOGY_VERSION,
    EVALUATION_REPORT_SCHEMA_VERSION,
    EvaluationCase,
    EvaluationDataError,
    EvaluationService,
    EvaluationStore,
    evaluation_cases_revision,
    evaluation_revision,
    evaluation_runtime_revision,
    temporal_iou,
)
from videoscope.repository import Repository
from videoscope.search.service import SearchService


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


def test_cases_revision_is_canonical_and_detects_same_count_changes() -> None:
    cases = [
        EvaluationCase("case-a", "бросок", "video-1", 1, 3, "visual", "gold"),
        EvaluationCase(
            "case-b",
            "пас",
            "video-1",
            4,
            6,
            "all",
            "silver",
            "контроль",
        ),
    ]
    changed = [
        cases[0],
        EvaluationCase(
            "case-b",
            "пас",
            "video-1",
            4,
            7,
            "all",
            "silver",
            "контроль",
        ),
    ]

    assert evaluation_cases_revision(cases) == (
        "0f1edb7e91b42677aafd9128ceb79032bc18e0f8afc749c73d479037c6bdeed1"
    )
    assert evaluation_cases_revision(
        list(reversed(cases))
    ) == evaluation_cases_revision(cases)
    assert evaluation_cases_revision(changed) != evaluation_cases_revision(cases)


def test_runtime_revision_tracks_retrieval_config_and_model_identity() -> None:
    first = FakeSearch({})
    first.semantic_text_min_score = 0.42
    first.vector_index = SimpleNamespace(
        available=True,
        embedding_identity="model@example-revision",
        dimensions=384,
    )
    second = FakeSearch({})
    second.semantic_text_min_score = 0.55
    second.vector_index = SimpleNamespace(
        available=True,
        embedding_identity="model@example-revision",
        dimensions=384,
    )

    assert evaluation_runtime_revision(first) != evaluation_runtime_revision(second)


def test_evaluation_revision_changes_with_runtime_for_same_cases() -> None:
    cases = [EvaluationCase("case", "query", "video-1", 0, 1)]

    assert evaluation_revision(cases, "a" * 64) != evaluation_revision(cases, "b" * 64)


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

    assert metrics.total_case_count == 2
    assert metrics.successful_case_count == 2
    assert metrics.error_count == 0
    assert metrics.status == "complete"
    assert metrics.recall_at_1 == pytest.approx(0.5)
    assert metrics.recall_at_3 == pytest.approx(1.0)
    assert metrics.mrr == pytest.approx(0.75)
    assert metrics.mean_temporal_iou > 0.5
    assert report.cases_revision == evaluation_cases_revision(store.read_cases())
    assert report.evaluation_revision == evaluation_revision(
        store.read_cases(),
        report.runtime_revision,
    )
    assert len(report.runtime_revision) == 64
    assert report.schema_version == EVALUATION_REPORT_SCHEMA_VERSION
    assert report.methodology_version == EVALUATION_METHODOLOGY_VERSION
    assert store.read_report() == json.loads(store.report_path.read_text(encoding="utf-8"))


def test_recall_requires_meaningful_temporal_overlap(tmp_path) -> None:
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")
    store.replace_cases([
        EvaluationCase("case", "query", "video-1", 0, 10, "visual", "gold"),
    ])
    search = FakeSearch({
        "query": [
            SimpleNamespace(video_id="video-1", start=9.9, end=20),
            SimpleNamespace(video_id="video-1", start=2, end=8),
        ],
    })

    report = EvaluationService(search, store).run(["auto"])
    result = report.variants[0].cases[0]

    assert report.temporal_iou_threshold == 0.3
    assert result.relevant_rank == 2
    assert result.temporal_iou == pytest.approx(0.6)


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


def test_evaluation_hides_search_failure_details_and_logs_them(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")
    store.replace_cases([
        EvaluationCase("case", "query", "video-1", 0, 2, "all", "gold"),
    ])

    class FailingSearch:
        def search(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("private model path: /secret/models/provider")

    with caplog.at_level(logging.ERROR, logger="videoscope.evaluation"):
        report = EvaluationService(FailingSearch(), store).run(["auto"])  # type: ignore[arg-type]

    metrics = report.variants[0]
    assert metrics.cases[0].error == "Search failed"
    assert "/secret/models/provider" not in metrics.cases[0].error
    assert metrics.total_case_count == 1
    assert metrics.successful_case_count == 0
    assert metrics.error_count == 1
    assert metrics.status == "failed"
    assert metrics.recall_at_1 is None
    assert metrics.recall_at_3 is None
    assert metrics.recall_at_5 is None
    assert metrics.mrr is None
    assert metrics.mean_temporal_iou is None
    assert metrics.mean_latency_ms is None
    assert any(
        record.exc_info
        and "/secret/models/provider" in str(record.exc_info[1])
        for record in caplog.records
    )


def test_failed_searches_are_excluded_from_partial_aggregates(tmp_path) -> None:
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")
    store.replace_cases([
        EvaluationCase("good", "good query", "video-1", 0, 2),
        EvaluationCase("failed", "failed query", "video-1", 0, 2),
    ])

    class PartialSearch:
        def search(self, query, **_kwargs):  # type: ignore[no-untyped-def]
            if query == "failed query":
                raise RuntimeError("provider unavailable")
            return [SimpleNamespace(video_id="video-1", start=0, end=2)]

    metrics = EvaluationService(PartialSearch(), store).run(["auto"]).variants[0]  # type: ignore[arg-type]

    assert metrics.total_case_count == 2
    assert metrics.successful_case_count == 1
    assert metrics.error_count == 1
    assert metrics.status == "partial"
    assert metrics.recall_at_1 == 1
    assert metrics.recall_at_3 == 1
    assert metrics.recall_at_5 == 1
    assert metrics.mrr == 1
    assert metrics.mean_temporal_iou == 1


def test_malformed_search_result_is_an_evaluation_error(tmp_path) -> None:
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")
    store.replace_cases([
        EvaluationCase("case", "query", "video-1", 0, 2),
    ])
    search = FakeSearch({
        "query": [SimpleNamespace(video_id="video-1", start=float("nan"), end=2)],
    })

    metrics = EvaluationService(search, store).run(["auto"]).variants[0]  # type: ignore[arg-type]

    assert metrics.status == "failed"
    assert metrics.error_count == 1
    assert metrics.recall_at_1 is None
    assert metrics.cases[0].error == "Search failed"


def test_evaluation_strict_mode_surfaces_search_dependency_failure(tmp_path) -> None:
    class FailingIndex:
        def search(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("vector database unavailable")

    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="match.mp4",
        media_path=str(tmp_path / "match.mp4"),
        size_bytes=1,
    )
    repository.update_video(
        "video-1",
        status="ready",
        stage="ready",
        duration=2.0,
    )
    search = SearchService(repository, FailingIndex())  # type: ignore[arg-type]
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")
    store.replace_cases([
        EvaluationCase("case", "query", "video-1", 0, 2, "speech"),
    ])

    # Interactive search keeps its existing best-effort/partial-result behavior.
    assert search.search("query", mode="speech") == []

    metrics = EvaluationService(search, store).run(["speech"]).variants[0]

    assert metrics.status == "failed"
    assert metrics.error_count == 1
    assert metrics.cases[0].error == "Search failed"


def test_missing_search_capability_is_not_reported_as_a_completed_miss(tmp_path) -> None:
    from videoscope.search.vector_index import EmptyVectorIndex

    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    search = SearchService(repository, EmptyVectorIndex())
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")
    store.replace_cases([
        EvaluationCase("case", "query", "video-1", 0, 2, "speech"),
    ])

    metrics = EvaluationService(search, store).run(["speech"]).variants[0]

    assert metrics.status == "failed"
    assert metrics.error_count == 1
    assert metrics.cases[0].error == "Search failed"


def test_evaluation_rejects_empty_case_set(tmp_path) -> None:
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")

    with pytest.raises(ValueError, match="at least one"):
        EvaluationService(FakeSearch({}), store).run(["auto"])  # type: ignore[arg-type]


def test_evaluation_validates_video_state_and_label_bounds_before_search(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="match.mp4",
        media_path=str(tmp_path / "match.mp4"),
        size_bytes=1,
    )
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")
    store.replace_cases([
        EvaluationCase("case", "query", "video-1", 0, 2),
    ])
    search = FakeSearch({})
    service = EvaluationService(search, store, repository=repository)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="not ready"):
        service.run(["auto"])
    assert search.calls == []

    repository.update_video("video-1", status="ready", stage="ready", duration=1.0)
    with pytest.raises(ValueError, match="duration"):
        service.run(["auto"])
    assert search.calls == []


@pytest.mark.parametrize(
    "cases",
    [
        [{"id": "bad", "query": "query", "video_id": "../secret", "start": 0, "end": 1}],
        [{"id": "bad", "query": "query", "video_id": "video-1", "start": "0", "end": 1}],
        [{"id": "bad", "query": "query", "video_id": "video-1", "start": float("nan"), "end": 1}],
        [{"id": "bad", "query": "query", "video_id": "video-1", "start": 2, "end": 1}],
        [{"id": "bad", "query": "query", "video_id": "video-1", "start": 0, "end": 1, "mode": "admin"}],
        [{"id": "bad", "query": "query", "video_id": "video-1", "start": 0, "end": 1, "label_source": "guess"}],
        [
            {"id": "duplicate", "query": "one", "video_id": "video-1", "start": 0, "end": 1},
            {"id": "duplicate", "query": "two", "video_id": "video-1", "start": 1, "end": 2},
        ],
    ],
)
def test_corrupt_persisted_cases_fail_closed_before_search(tmp_path, cases) -> None:  # type: ignore[no-untyped-def]
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")
    store.cases_path.write_text(json.dumps({"version": 1, "cases": cases}), encoding="utf-8")
    search = FakeSearch({})

    with pytest.raises(EvaluationDataError, match="invalid") as caught:
        EvaluationService(search, store).run(["auto"])  # type: ignore[arg-type]

    assert str(store.cases_path) not in str(caught.value)
    assert search.calls == []


def test_case_validation_normalizes_strings_and_enforces_api_bounds(tmp_path) -> None:
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")

    store.replace_cases([
        EvaluationCase(" case ", " query ", " video-1 ", 0, 1, " all ", " gold ", " note "),
    ])

    assert store.read_cases() == [
        EvaluationCase("case", "query", "video-1", 0.0, 1.0, "all", "gold", "note"),
    ]

    with pytest.raises(EvaluationDataError, match="invalid"):
        store.replace_cases([
            EvaluationCase("x" * 101, "query", "video-1", 0, 1),
        ])


def test_evaluation_store_delegates_writes_to_atomic_storage(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[tuple[object, object]] = []
    monkeypatch.setattr(
        "videoscope.evaluation.atomic_write_json",
        lambda path, payload: writes.append((path, payload)),
    )
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")

    store.replace_cases([
        EvaluationCase("case", "query", "video-1", 0, 2, "all", "gold"),
    ])

    assert writes == [
        (
            store.cases_path,
            {
                "version": 1,
                "cases": [
                    {
                        "id": "case",
                        "query": "query",
                        "video_id": "video-1",
                        "start": 0,
                        "end": 2,
                        "mode": "all",
                        "label_source": "gold",
                        "notes": "",
                    }
                ],
            },
        )
    ]


def test_store_rejects_invalid_case_interval(tmp_path) -> None:
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")

    with pytest.raises(ValueError, match="interval"):
        store.replace_cases([
            EvaluationCase("bad", "query", "video-1", 10, 10, "all", "gold"),
        ])


def test_legacy_report_is_normalized_and_marked_for_stale_detection(tmp_path) -> None:
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")
    store.report_path.write_text(
        json.dumps({
            "generated_at": "2026-08-01T00:00:00Z",
            "variants": [{
                "name": "auto",
                "case_count": 2,
                "recall_at_1": 0.5,
                "recall_at_3": 0.5,
                "recall_at_5": 0.5,
                "mrr": 0.5,
                "mean_temporal_iou": 0.5,
                "mean_latency_ms": 10,
                "cases": [
                    {
                        "case_id": "ok",
                        "query": "q",
                        "relevant_rank": 1,
                        "temporal_iou": 1,
                        "latency_ms": 10,
                        "result_count": 1,
                        "error": None,
                    },
                    {
                        "case_id": "bad",
                        "query": "q",
                        "relevant_rank": None,
                        "temporal_iou": 0,
                        "latency_ms": 10,
                        "result_count": 0,
                        "error": "Search failed",
                    },
                ],
            }],
        }),
        encoding="utf-8",
    )

    report = store.read_report()
    assert report is not None
    assert report["schema_version"] is None
    assert report["methodology_version"] is None
    assert report["evaluation_revision"] is None
    assert report["runtime_revision"] is None
    assert report["cases_revision"] is None
    assert report["temporal_iou_threshold"] is None
    variant = report["variants"][0]  # type: ignore[index]
    assert variant["total_case_count"] == 2
    assert variant["successful_case_count"] == 1
    assert variant["error_count"] == 1
    assert variant["status"] == "partial"
    assert variant["recall_at_1"] == 1
    assert variant["recall_at_3"] == 1
    assert variant["recall_at_5"] == 1
    assert variant["mrr"] == 1
    assert variant["mean_temporal_iou"] == 1
    assert "case_count" not in variant


def test_evaluation_payload_exposes_revision_for_current_cases(tmp_path) -> None:
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")
    store.replace_cases([
        EvaluationCase("case", "query", "video-1", 0, 2, "all", "gold"),
    ])

    payload = EvaluationPayloadResponse.model_validate({
        "cases": [asdict(case) for case in store.read_cases()],
        "runtime_revision": "a" * 64,
        "evaluation_revision": evaluation_revision(
            store.read_cases(),
            "a" * 64,
        ),
        "report": {
            "generated_at": "2026-08-01T00:00:00Z",
            "schema_version": None,
            "methodology_version": None,
            "evaluation_revision": None,
            "runtime_revision": None,
            "cases_revision": None,
            "temporal_iou_threshold": None,
            "variants": [],
        },
    }).model_dump()

    assert payload["cases_revision"] == evaluation_cases_revision(store.read_cases())
    assert payload["evaluation_revision"] == evaluation_revision(
        store.read_cases(),
        "a" * 64,
    )
    assert payload["report"]["cases_revision"] is None
    assert payload["report"]["evaluation_revision"] is None
    assert payload["report"]["runtime_revision"] is None


def test_methodology_is_part_of_evaluation_freshness_revision(tmp_path) -> None:
    cases = [EvaluationCase("case", "query", "video-1", 0, 1)]

    assert evaluation_revision(cases) != evaluation_cases_revision(cases)
    assert len(evaluation_revision(cases)) == 64


def test_corrupt_derived_report_degrades_to_none(tmp_path) -> None:
    store = EvaluationStore(tmp_path / "cases.json", tmp_path / "report.json")
    store.report_path.write_text(
        json.dumps({
            "generated_at": "2026-08-01T00:00:00Z",
            "schema_version": 2,
            "methodology_version": 2,
            "evaluation_revision": "a" * 64,
            "runtime_revision": "b" * 64,
            "cases_revision": "c" * 64,
            "temporal_iou_threshold": 0.3,
            "variants": [{
                "name": "auto",
                "total_case_count": 1,
                "successful_case_count": 1,
                "error_count": 0,
                "status": "failed",
                "recall_at_1": float("nan"),
                "recall_at_3": 0,
                "recall_at_5": 0,
                "mrr": 0,
                "mean_temporal_iou": 0,
                "mean_latency_ms": 1,
                "cases": [],
            }],
        }),
        encoding="utf-8",
    )

    assert store.read_report() is None
