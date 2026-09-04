from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from videoscope.benchmark import (
    AssetProvenance,
    BenchmarkAsset,
    BenchmarkDataError,
    BenchmarkDataset,
    BenchmarkDurabilityError,
    BenchmarkInterval,
    CRITICAL_SLICE_LABELS_SCHEMA_VERSION,
    CriticalSliceLabels,
    HardNegative,
    QueryCase,
    dataset_revision,
    load_dataset,
    write_dataset,
)
from videoscope.benchmark.serialization import dataset_from_dict, dataset_to_dict


def _asset(
    asset_id: str = "game-a",
    *,
    digest: str = "a" * 64,
    duration_seconds: float = 120.0,
) -> BenchmarkAsset:
    return BenchmarkAsset(
        asset_id=asset_id,
        sha256=digest,
        byte_size=1_024,
        duration_seconds=duration_seconds,
        provenance=AssetProvenance(
            source="University practice recording",
            source_uri="https://example.test/assets/game-a",
            license_id="CC-BY-4.0",
            license_uri="https://creativecommons.org/licenses/by/4.0/",
            attribution="Example camera operator",
        ),
    )


def _dataset() -> BenchmarkDataset:
    first = _asset()
    second = _asset("game-b", digest="b" * 64, duration_seconds=90.0)
    return BenchmarkDataset(
        schema_version=1,
        dataset_id="basketball-core",
        dataset_version="1.0.0",
        description="Portable retrieval benchmark",
        assets=(first, second),
        cases=(
            QueryCase(
                case_id="made-shot",
                query="player makes a basket",
                asset_ids=("game-a",),
                domain="basketball",
                modalities=("visual", "sports"),
                label_quality="gold",
                split_group="match-a",
                relevant_intervals=(
                    BenchmarkInterval("game-a", 10.0, 12.5),
                    BenchmarkInterval("game-a", 42.0, 44.0),
                ),
                hard_negatives=(
                    HardNegative("game-a", 20.0, 22.0, "missed shot"),
                    HardNegative("game-a", 70.0, 72.0, "defensive rebound"),
                ),
                notes="Two valid occurrences",
            ),
            QueryCase(
                case_id="no-dunk",
                query="a dunk occurs",
                asset_ids=("game-b",),
                domain="basketball",
                modalities=("sports", "visual"),
                label_quality="silver",
                split_group="match-b",
                relevant_intervals=(),
                hard_negatives=(
                    HardNegative("game-b", 50.0, 53.0, "layup near the rim"),
                ),
            ),
        ),
    )


def test_dataset_supports_zero_one_and_multiple_relevant_intervals() -> None:
    dataset = _dataset()
    one_interval = QueryCase(
        case_id="one",
        query="scoreboard changes",
        asset_ids=("game-a",),
        domain="basketball",
        modalities=("ocr",),
        label_quality="gold",
        split_group="match-a",
        relevant_intervals=(BenchmarkInterval("game-a", 60.0, 61.0),),
    )
    dataset = BenchmarkDataset(
        schema_version=dataset.schema_version,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        description=dataset.description,
        assets=dataset.assets,
        cases=(*dataset.cases, one_interval),
    )

    interval_counts = {
        case.case_id: len(case.relevant_intervals) for case in dataset.cases
    }

    assert interval_counts == {"made-shot": 2, "no-dunk": 0, "one": 1}


def test_dataset_revision_is_canonical_and_order_independent() -> None:
    dataset = _dataset()
    reordered_cases = tuple(
        QueryCase(
            case_id=case.case_id,
            query=case.query,
            asset_ids=tuple(reversed(case.asset_ids)),
            domain=case.domain,
            modalities=tuple(reversed(case.modalities)),
            label_quality=case.label_quality,
            split_group=case.split_group,
            relevant_intervals=tuple(reversed(case.relevant_intervals)),
            hard_negatives=tuple(reversed(case.hard_negatives)),
            notes=case.notes,
        )
        for case in reversed(dataset.cases)
    )
    reordered = BenchmarkDataset(
        schema_version=dataset.schema_version,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        description=dataset.description,
        assets=tuple(reversed(dataset.assets)),
        cases=reordered_cases,
    )

    assert dataset_revision(dataset) == dataset_revision(reordered)
    assert len(dataset_revision(dataset)) == 64


def test_dataset_revision_tracks_labels_and_provenance() -> None:
    dataset = _dataset()
    changed_asset = BenchmarkAsset(
        asset_id=dataset.assets[0].asset_id,
        sha256=dataset.assets[0].sha256,
        byte_size=dataset.assets[0].byte_size,
        duration_seconds=dataset.assets[0].duration_seconds,
        provenance=AssetProvenance(
            source="Different source",
            source_uri=dataset.assets[0].provenance.source_uri,
            license_id=dataset.assets[0].provenance.license_id,
            license_uri=dataset.assets[0].provenance.license_uri,
            attribution=dataset.assets[0].provenance.attribution,
        ),
    )
    changed = BenchmarkDataset(
        schema_version=dataset.schema_version,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        description=dataset.description,
        assets=(changed_asset, dataset.assets[1]),
        cases=dataset.cases,
    )

    assert dataset_revision(dataset) != dataset_revision(changed)


def _critical_slice_labels(
    *,
    event_class: tuple[str, ...] = ("made_3",),
    capture_condition: tuple[str, ...] = ("low_resolution", "scoreboard_hidden"),
    distribution_shift: tuple[str, ...] = ("different_camera_or_league",),
) -> CriticalSliceLabels:
    return CriticalSliceLabels(
        schema_version=CRITICAL_SLICE_LABELS_SCHEMA_VERSION,
        event_class=event_class,
        capture_condition=capture_condition,
        distribution_shift=distribution_shift,
    )


def test_versioned_critical_slices_round_trip_and_canonicalize_revision() -> None:
    dataset = _dataset()
    labeled = replace(
        dataset,
        cases=(
            replace(dataset.cases[0], critical_slices=_critical_slice_labels()),
            replace(
                dataset.cases[1],
                critical_slices=_critical_slice_labels(
                    event_class=("miss", "replay"),
                    capture_condition=("standard",),
                    distribution_shift=("in_distribution",),
                ),
            ),
        ),
    )
    reordered = replace(
        labeled,
        cases=(
            replace(
                labeled.cases[0],
                critical_slices=_critical_slice_labels(
                    capture_condition=("scoreboard_hidden", "low_resolution"),
                ),
            ),
            replace(
                labeled.cases[1],
                critical_slices=_critical_slice_labels(
                    event_class=("replay", "miss"),
                    capture_condition=("standard",),
                    distribution_shift=("in_distribution",),
                ),
            ),
        ),
    )

    payload = dataset_to_dict(labeled)

    assert payload["cases"][0]["critical_slices"] == {  # type: ignore[index]
        "schema_version": 1,
        "event_class": ["made_3"],
        "capture_condition": ["low_resolution", "scoreboard_hidden"],
        "distribution_shift": ["different_camera_or_league"],
    }
    assert dataset_from_dict(payload) == labeled
    assert dataset_revision(labeled) == dataset_revision(reordered)
    assert dataset_revision(labeled) != dataset_revision(
        replace(
            labeled,
            cases=(
                replace(
                    labeled.cases[0],
                    critical_slices=_critical_slice_labels(
                        event_class=("made_2",),
                    ),
                ),
                labeled.cases[1],
            ),
        )
    )


def test_legacy_dataset_serialization_and_revision_remain_unchanged() -> None:
    dataset = _dataset()
    payload = dataset_to_dict(dataset)

    assert all("critical_slices" not in case for case in payload["cases"])  # type: ignore[union-attr]
    assert dataset_from_dict(payload) == dataset
    assert (
        dataset_revision(dataset)
        == "bc23ec29550de06102a73c80a52fe009c452901183265e8f38bc6015d37b6c9b"
    )


@pytest.mark.parametrize(
    "critical_slices",
    [
        {
            "schema_version": 2,
            "event_class": ["made_3"],
            "capture_condition": ["standard"],
            "distribution_shift": ["in_distribution"],
        },
        {
            "schema_version": 1,
            "event_class": [],
            "capture_condition": ["standard"],
            "distribution_shift": ["in_distribution"],
        },
        {
            "schema_version": 1,
            "event_class": ["made_3", "made_3"],
            "capture_condition": ["standard"],
            "distribution_shift": ["in_distribution"],
        },
        {
            "schema_version": 1,
            "event_class": ["made/3"],
            "capture_condition": ["standard"],
            "distribution_shift": ["in_distribution"],
        },
        {
            "schema_version": 1,
            "event_class": "made_3",
            "capture_condition": ["standard"],
            "distribution_shift": ["in_distribution"],
        },
        {
            "schema_version": 1,
            "event_class": ["made_3"],
            "capture_condition": ["standard"],
        },
        {
            "schema_version": 1,
            "event_class": ["made_3"],
            "capture_condition": ["standard"],
            "distribution_shift": ["in_distribution"],
            "unexpected": True,
        },
    ],
)
def test_critical_slice_labels_reject_malformed_contracts(
    critical_slices: object,
) -> None:
    payload = dataset_to_dict(_dataset())
    payload["cases"][0]["critical_slices"] = critical_slices  # type: ignore[index]

    with pytest.raises(BenchmarkDataError, match="critical_slices"):
        dataset_from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sha256", "not-a-digest"),
        ("byte_size", 0),
        ("byte_size", True),
        ("duration_seconds", float("nan")),
        ("duration_seconds", float("inf")),
        ("duration_seconds", 0.0),
    ],
)
def test_asset_rejects_invalid_hashes_and_numeric_boundaries(
    field,
    value,
) -> None:  # type: ignore[no-untyped-def]
    values = {
        "asset_id": "asset",
        "sha256": "c" * 64,
        "byte_size": 100,
        "duration_seconds": 10.0,
        "provenance": AssetProvenance(source="camera", license_id="MIT"),
    }
    values[field] = value

    with pytest.raises(BenchmarkDataError):
        BenchmarkAsset(**values)


def test_dataset_bounds_asset_count_and_declared_media_bytes() -> None:
    provenance = AssetProvenance(source="camera", license_id="MIT")

    with pytest.raises(BenchmarkDataError, match="byte size limit"):
        BenchmarkAsset(
            asset_id="oversized",
            sha256="f" * 64,
            byte_size=16 * 1024**3 + 1,
            duration_seconds=10.0,
            provenance=provenance,
        )

    too_many = tuple(
        BenchmarkAsset(
            asset_id=f"asset-{index}",
            sha256=f"{index + 1:064x}",
            byte_size=1,
            duration_seconds=10.0,
            provenance=provenance,
        )
        for index in range(129)
    )
    with pytest.raises(BenchmarkDataError, match="asset count limit"):
        BenchmarkDataset(1, "bounded", "1", "", too_many, ())

    aggregate_oversized = tuple(
        BenchmarkAsset(
            asset_id=f"large-{index}",
            sha256=f"{index + 1:064x}",
            byte_size=16 * 1024**3,
            duration_seconds=10.0,
            provenance=provenance,
        )
        for index in range(9)
    )
    with pytest.raises(BenchmarkDataError, match="aggregate media byte limit"):
        BenchmarkDataset(1, "bounded", "1", "", aggregate_oversized, ())


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (-0.1, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (float("nan"), 1.0),
        (0.0, float("inf")),
    ],
)
def test_interval_rejects_non_finite_and_invalid_ranges(
    start: float,
    end: float,
) -> None:
    with pytest.raises(BenchmarkDataError):
        BenchmarkInterval("asset", start, end)


def test_dataset_rejects_unknown_references_and_out_of_bounds_intervals() -> None:
    asset = _asset(duration_seconds=10.0)
    unknown_reference = QueryCase(
        case_id="unknown",
        query="query",
        asset_ids=("missing",),
        domain="basketball",
        modalities=("visual",),
        label_quality="gold",
        split_group="match",
    )
    out_of_bounds = QueryCase(
        case_id="bounds",
        query="query",
        asset_ids=(asset.asset_id,),
        domain="basketball",
        modalities=("visual",),
        label_quality="gold",
        split_group="match",
        relevant_intervals=(BenchmarkInterval(asset.asset_id, 9.0, 11.0),),
    )

    with pytest.raises(BenchmarkDataError, match="unknown asset"):
        BenchmarkDataset(1, "dataset", "1", "", (asset,), (unknown_reference,))
    with pytest.raises(BenchmarkDataError, match="duration"):
        BenchmarkDataset(1, "dataset", "1", "", (asset,), (out_of_bounds,))


def test_dataset_rejects_duplicate_identities_and_conflicting_labels() -> None:
    asset = _asset()
    duplicate_asset = _asset("alias")
    case = QueryCase(
        case_id="case",
        query="shot",
        asset_ids=(asset.asset_id,),
        domain="basketball",
        modalities=("visual",),
        label_quality="gold",
        split_group="match-a",
        relevant_intervals=(BenchmarkInterval(asset.asset_id, 10.0, 20.0),),
        hard_negatives=(HardNegative(asset.asset_id, 15.0, 17.0, "conflict"),),
    )

    with pytest.raises(BenchmarkDataError, match="SHA-256"):
        BenchmarkDataset(1, "dataset", "1", "", (asset, duplicate_asset), ())
    with pytest.raises(BenchmarkDataError, match="overlaps"):
        BenchmarkDataset(1, "dataset", "1", "", (asset,), (case,))


def test_case_rejects_duplicate_assets_modalities_and_intervals() -> None:
    interval = BenchmarkInterval("asset", 1.0, 2.0)

    with pytest.raises(BenchmarkDataError, match="asset_ids"):
        QueryCase(
            "case",
            "query",
            ("asset", "asset"),
            "domain",
            ("visual",),
            "gold",
            "group",
        )
    with pytest.raises(BenchmarkDataError, match="modalities"):
        QueryCase(
            "case",
            "query",
            ("asset",),
            "domain",
            ("visual", "visual"),
            "gold",
            "group",
        )
    with pytest.raises(BenchmarkDataError, match="relevant_intervals"):
        QueryCase(
            "case",
            "query",
            ("asset",),
            "domain",
            ("visual",),
            "gold",
            "group",
            (interval, interval),
        )


def test_case_rejects_duplicate_hard_negative_ranges_with_different_notes() -> None:
    first = HardNegative("asset", 1.0, 2.0, "missed shot")
    second = HardNegative("asset", 1.0, 2.0, "layup")

    with pytest.raises(BenchmarkDataError, match="hard_negatives"):
        QueryCase(
            "case",
            "query",
            ("asset",),
            "domain",
            ("visual",),
            "gold",
            "group",
            hard_negatives=(first, second),
        )


def test_dataset_keeps_every_asset_in_one_complete_video_split_group() -> None:
    asset = _asset()
    first = QueryCase(
        "first",
        "query one",
        (asset.asset_id,),
        "basketball",
        ("visual",),
        "gold",
        "train-match-a",
    )
    second = QueryCase(
        "second",
        "query two",
        (asset.asset_id,),
        "basketball",
        ("visual",),
        "gold",
        "test-match-a",
    )

    with pytest.raises(BenchmarkDataError, match="split_group"):
        BenchmarkDataset(1, "dataset", "1", "", (asset,), (first, second))


def test_dataset_round_trips_and_refuses_to_overwrite(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    dataset = _dataset()

    write_dataset(path, dataset)
    original = path.read_bytes()

    assert load_dataset(path) == dataset
    with pytest.raises(FileExistsError):
        write_dataset(path, BenchmarkDataset(
            schema_version=dataset.schema_version,
            dataset_id=dataset.dataset_id,
            dataset_version="2.0.0",
            description=dataset.description,
            assets=dataset.assets,
            cases=dataset.cases,
        ))
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_dataset_loader_rejects_unknown_fields_and_non_json_numbers(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(path, _dataset())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkDataError, match="unexpected fields"):
        load_dataset(path)

    path.write_text("{\"schema_version\": NaN}", encoding="utf-8")
    with pytest.raises(BenchmarkDataError):
        load_dataset(path)


@pytest.mark.parametrize(
    "source_uri",
    [
        "file:///Users/example/private/game.mp4",
        "https://user:password@example.test/game.mp4",
        "https://example.test/game.mp4?access_token=secret",
        "http://localhost/private/game.mp4",
        "http://127.0.0.1/private/game.mp4",
        "http://10.0.0.8/private/game.mp4",
        "http://[::1]/private/game.mp4",
        "https://[invalid/game.mp4",
        "../private/game.mp4",
    ],
)
def test_provenance_rejects_nonportable_or_secret_bearing_uris(source_uri: str) -> None:
    with pytest.raises(BenchmarkDataError, match="source_uri"):
        AssetProvenance(
            source="Private source",
            source_uri=source_uri,
            license_id="LicenseRef-Private",
        )


def test_provenance_accepts_public_web_and_urn_references() -> None:
    web = AssetProvenance(
        source="Public source",
        source_uri="https://example.test/game.mp4",
        license_id="CC-BY-4.0",
        license_uri="https://creativecommons.org/licenses/by/4.0/",
    )
    urn = AssetProvenance(
        source="DOI source",
        source_uri="urn:doi:10.1000/example",
        license_id="CC-BY-4.0",
    )

    assert web.source_uri == "https://example.test/game.mp4"
    assert urn.source_uri == "urn:doi:10.1000/example"


def test_provenance_rejects_a_local_path_in_the_portable_source_field() -> None:
    with pytest.raises(BenchmarkDataError, match="source"):
        AssetProvenance(
            source="/Users/example/private/game.mp4",
            license_id="LicenseRef-Private",
        )


def test_dataset_contract_is_deeply_immutable() -> None:
    dataset = _dataset()

    with pytest.raises(FrozenInstanceError):
        dataset.assets[0].provenance.source = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        dataset.cases[0].relevant_intervals[0].start_seconds = 0  # type: ignore[misc]


def test_dataset_loader_rejects_non_regular_files(tmp_path) -> None:
    with pytest.raises(BenchmarkDataError, match="regular file"):
        load_dataset(tmp_path)


def test_dataset_loader_enforces_a_bounded_manifest_size(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "dataset.json"
    path.write_bytes(b"{}" * 100)

    from videoscope.benchmark import storage

    monkeypatch.setattr(storage, "MAX_DATASET_MANIFEST_BYTES", 32)

    with pytest.raises(BenchmarkDataError, match="byte limit"):
        load_dataset(path)


def test_dataset_loader_rejects_unencodable_unicode_text(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(path, _dataset())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["assets"][0]["provenance"]["source"] = "\ud800"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkDataError, match="Unicode"):
        load_dataset(path)


def test_dataset_loader_wraps_oversized_json_integer_errors(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text('{"schema_version":' + "9" * 5_000 + "}", encoding="utf-8")

    with pytest.raises(BenchmarkDataError, match="valid JSON"):
        load_dataset(path)


def test_concurrent_dataset_writers_publish_exactly_one_complete_manifest(
    tmp_path,
) -> None:
    path = tmp_path / "dataset.json"
    dataset = _dataset()

    def write_once(_index: int) -> str:
        try:
            write_dataset(path, dataset)
            return "written"
        except FileExistsError:
            return "exists"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(write_once, range(20)))

    assert results.count("written") == 1
    assert results.count("exists") == 19
    assert load_dataset(path) == dataset
    assert not list(tmp_path.glob(".*.tmp"))


def test_dataset_reports_post_commit_durability_failure_without_losing_data(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "dataset.json"
    dataset = _dataset()

    from videoscope.benchmark import storage

    monkeypatch.setattr(
        storage,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(OSError("simulated fsync failure")),
    )

    with pytest.raises(BenchmarkDurabilityError, match="committed"):
        write_dataset(path, dataset)

    assert load_dataset(path) == dataset
