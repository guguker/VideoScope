from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

from videoscope.benchmark import (
    AssetProvenance,
    BenchmarkAsset,
    BenchmarkDataError,
    BenchmarkInterval,
    ComponentIdentity,
    MAX_PREPARED_INPUT_BYTES,
    MAX_VIDEO_VERIFIER_CASES,
    MAX_VIDEO_VERIFIER_MANIFEST_BYTES,
    MAX_VIDEO_VERIFIER_PREPARATION_PROTOCOLS,
    PreparedInputProtocol,
    VideoVerifierCase,
    VideoVerifierDataset,
    VideoVerifierExpectedFact,
    video_verifier_dataset_from_json,
    video_verifier_dataset_revision,
    video_verifier_dataset_to_dict,
    video_verifier_dataset_to_json_bytes,
)


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
            source="Recorded evaluation fixture",
            source_uri=f"https://example.test/assets/{asset_id}",
            license_id="CC-BY-4.0",
            license_uri="https://creativecommons.org/licenses/by/4.0/",
            attribution="Example camera operator",
        ),
    )


def _protocol(
    protocol_id: str = "native-mp4-v1",
    *,
    input_kind: str = "native_video",
) -> PreparedInputProtocol:
    return PreparedInputProtocol(
        protocol_id=protocol_id,
        input_kind=input_kind,  # type: ignore[arg-type]
        preparer=ComponentIdentity("ffmpeg-clip", "ffmpeg@7;video=h264;audio=none"),
        parameters=(
            ComponentIdentity("pixel-format", "yuv420p"),
            ComponentIdentity("timestamp-policy", "source-relative-ms-v1"),
        ),
    )


def _basketball_facts() -> tuple[VideoVerifierExpectedFact, ...]:
    return (
        VideoVerifierExpectedFact("shot_attempt", True),
        VideoVerifierExpectedFact("ball_through_hoop", True),
        VideoVerifierExpectedFact("shooter_outside_arc", None),
        VideoVerifierExpectedFact("three_point_signal", False),
    )


def _case(
    case_id: str = "made-shot-a",
    *,
    asset_id: str = "game-a",
    protocol_id: str = "native-mp4-v1",
    start_seconds: float = 10.0,
    end_seconds: float = 18.0,
    prepared_digest: str = "c" * 64,
    split: str = "regression_seen",
    split_group: str = "match-a",
) -> VideoVerifierCase:
    return VideoVerifierCase(
        case_id=case_id,
        stratum="basketball_facts",
        input_kind="native_video",
        source_interval=BenchmarkInterval(asset_id, start_seconds, end_seconds),
        prepared_input_protocol_id=protocol_id,
        prepared_input_sha256=prepared_digest,
        prepared_input_byte_size=4_096,
        query=None,
        expected_facts=_basketball_facts(),
        expected_jersey="15",
        label_quality="gold",
        split=split,  # type: ignore[arg-type]
        split_group=split_group,
        notes="Observed independently by two reviewers",
    )


def _dataset() -> VideoVerifierDataset:
    return VideoVerifierDataset(
        schema_version=1,
        dataset_id="direct-video-verifier",
        dataset_version="1.0.0",
        description="Pinned direct verifier inputs and independent labels",
        assets=(_asset(),),
        preparation_protocols=(_protocol(),),
        cases=(_case(),),
    )


def test_verifier_dataset_is_immutable_and_round_trips_canonical_json() -> None:
    dataset = _dataset()

    payload = video_verifier_dataset_to_json_bytes(dataset)
    restored = video_verifier_dataset_from_json(payload)

    assert restored == dataset
    assert payload.endswith(b"\n")
    assert video_verifier_dataset_to_json_bytes(restored) == payload
    assert restored.cases[0].is_promotion_holdout is False
    with pytest.raises(FrozenInstanceError):
        restored.dataset_id = "mutated"  # type: ignore[misc]


def test_revision_is_order_independent_but_tracks_labels_and_protocol() -> None:
    first_asset = _asset()
    second_asset = _asset("game-b", digest="b" * 64)
    native = _protocol()
    storyboard = PreparedInputProtocol(
        protocol_id="storyboard-jpeg-v1",
        input_kind="storyboard",
        preparer=ComponentIdentity("storyboard-builder", "pillow@11;layout=v1"),
        parameters=(
            ComponentIdentity("jpeg-quality", "90"),
            ComponentIdentity("frame-count", "12"),
        ),
    )
    generic = VideoVerifierCase(
        case_id="visible-handshake",
        stratum="generic_visual",
        input_kind="storyboard",
        source_interval=BenchmarkInterval("game-b", 20.0, 27.0),
        prepared_input_protocol_id="storyboard-jpeg-v1",
        prepared_input_sha256="d" * 64,
        prepared_input_byte_size=8_192,
        query="two people shake hands",
        expected_facts=(VideoVerifierExpectedFact("matches_query", False),),
        expected_jersey=None,
        label_quality="silver",
        split="validation",
        split_group="interview-b",
    )
    original = VideoVerifierDataset(
        1,
        "verifier",
        "1",
        "",
        (first_asset, second_asset),
        (native, storyboard),
        (_case(), generic),
    )
    reordered = VideoVerifierDataset(
        1,
        "verifier",
        "1",
        "",
        (second_asset, first_asset),
        (
            replace(storyboard, parameters=tuple(reversed(storyboard.parameters))),
            replace(native, parameters=tuple(reversed(native.parameters))),
        ),
        (
            generic,
            replace(_case(), expected_facts=tuple(reversed(_basketball_facts()))),
        ),
    )

    revision = video_verifier_dataset_revision(original)

    assert revision == video_verifier_dataset_revision(reordered)
    assert len(revision) == 64
    changed_label = replace(
        original,
        cases=(replace(original.cases[0], expected_jersey="9"), original.cases[1]),
    )
    changed_protocol = replace(
        original,
        preparation_protocols=(
            replace(
                native,
                preparer=ComponentIdentity("ffmpeg-clip", "ffmpeg@8;video=h264"),
            ),
            storyboard,
        ),
    )
    assert video_verifier_dataset_revision(changed_label) != revision
    assert video_verifier_dataset_revision(changed_protocol) != revision


def test_generic_visual_requires_query_and_only_matches_query_fact() -> None:
    common = {
        "case_id": "generic",
        "stratum": "generic_visual",
        "input_kind": "storyboard",
        "source_interval": BenchmarkInterval("asset", 1.0, 2.0),
        "prepared_input_protocol_id": "storyboard-v1",
        "prepared_input_sha256": "e" * 64,
        "prepared_input_byte_size": 100,
        "expected_facts": (VideoVerifierExpectedFact("matches_query", None),),
        "expected_jersey": None,
        "label_quality": "gold",
        "split": "regression_seen",
        "split_group": "group",
    }

    with pytest.raises(BenchmarkDataError, match="query"):
        VideoVerifierCase(query=None, **common)  # type: ignore[arg-type]
    with pytest.raises(BenchmarkDataError, match="expected_facts"):
        VideoVerifierCase(
            query="a visible action",
            **{
                **common,
                "expected_facts": (VideoVerifierExpectedFact("shot_attempt", True),),
            },
        )  # type: ignore[arg-type]


def test_basketball_facts_require_fixed_fact_set_and_no_query() -> None:
    with pytest.raises(BenchmarkDataError, match="query"):
        replace(_case(), query="made three pointer")
    with pytest.raises(BenchmarkDataError, match="expected_facts"):
        replace(
            _case(),
            expected_facts=(VideoVerifierExpectedFact("shot_attempt", True),),
        )


@pytest.mark.parametrize("expected", [0, 1, "true", [], object()])
def test_expected_fact_rejects_values_outside_strict_tristate(expected: object) -> None:
    with pytest.raises(BenchmarkDataError, match="true, false or null"):
        VideoVerifierExpectedFact("shot_attempt", expected)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", "contains spaces"),
        ("prepared_input_sha256", "not-a-digest"),
        ("prepared_input_byte_size", 0),
        ("prepared_input_byte_size", True),
        ("expected_jersey", "01"),
        ("expected_jersey", "100"),
        ("label_quality", "bronze"),
        ("split", "test"),
    ],
)
def test_case_rejects_invalid_identifiers_numbers_and_enums(
    field: str,
    value: object,
) -> None:
    with pytest.raises(BenchmarkDataError):
        replace(_case(), **{field: value})


def test_protocol_rejects_mutable_duplicate_or_invalid_parameters() -> None:
    parameter = ComponentIdentity("frame-count", "12")

    with pytest.raises(BenchmarkDataError, match="immutable tuple"):
        PreparedInputProtocol(
            "storyboard-v1",
            "storyboard",
            ComponentIdentity("builder", "v1"),
            [parameter],  # type: ignore[arg-type]
        )
    with pytest.raises(BenchmarkDataError, match="unique"):
        PreparedInputProtocol(
            "storyboard-v1",
            "storyboard",
            ComponentIdentity("builder", "v1"),
            (parameter, parameter),
        )
    with pytest.raises(BenchmarkDataError, match="input_kind"):
        replace(_protocol(), input_kind="frames")  # type: ignore[arg-type]
    with pytest.raises(BenchmarkDataError, match="preparer"):
        replace(_protocol(), preparer="ffmpeg@7")  # type: ignore[arg-type]
    with pytest.raises(BenchmarkDataError, match="must not be empty"):
        replace(_protocol(), parameters=())


def test_case_rejects_invalid_type_discriminators_and_prepared_input_limit() -> None:
    with pytest.raises(BenchmarkDataError, match="stratum"):
        replace(_case(), stratum="speech")  # type: ignore[arg-type]
    with pytest.raises(BenchmarkDataError, match="input_kind"):
        replace(_case(), input_kind="frames")  # type: ignore[arg-type]
    with pytest.raises(BenchmarkDataError, match="source_interval"):
        replace(_case(), source_interval=(10.0, 18.0))  # type: ignore[arg-type]
    with pytest.raises(BenchmarkDataError, match="byte size limit"):
        replace(
            _case(),
            prepared_input_byte_size=MAX_PREPARED_INPUT_BYTES + 1,
        )


def test_enum_discriminators_reject_unhashable_json_values_as_data_errors() -> None:
    with pytest.raises(BenchmarkDataError, match="input_kind"):
        replace(_protocol(), input_kind=[])  # type: ignore[arg-type]
    with pytest.raises(BenchmarkDataError, match="stratum"):
        replace(_case(), stratum=[])  # type: ignore[arg-type]
    with pytest.raises(BenchmarkDataError, match="input_kind"):
        replace(_case(), input_kind=[])  # type: ignore[arg-type]
    with pytest.raises(BenchmarkDataError, match="label_quality"):
        replace(_case(), label_quality=[])  # type: ignore[arg-type]
    with pytest.raises(BenchmarkDataError, match="split"):
        replace(_case(), split=[])  # type: ignore[arg-type]


def test_dataset_rejects_protocol_kind_mismatch_unknown_references_and_bounds() -> None:
    with pytest.raises(BenchmarkDataError, match="input_kind"):
        replace(
            _dataset(),
            preparation_protocols=(
                replace(_protocol(), input_kind="storyboard"),
            ),
        )
    with pytest.raises(BenchmarkDataError, match="unknown asset"):
        replace(
            _dataset(),
            cases=(replace(_case(), source_interval=BenchmarkInterval("missing", 1, 2)),),
        )
    with pytest.raises(BenchmarkDataError, match="unknown preparation protocol"):
        replace(
            _dataset(),
            cases=(replace(_case(), prepared_input_protocol_id="missing"),),
        )
    with pytest.raises(BenchmarkDataError, match="duration"):
        replace(
            _dataset(),
            cases=(replace(_case(), source_interval=BenchmarkInterval("game-a", 119, 121)),),
        )


def test_dataset_rejects_partial_overlaps_even_inside_one_split() -> None:
    overlapping = _case(
        "overlap",
        start_seconds=17.999,
        end_seconds=20.0,
        prepared_digest="d" * 64,
    )

    with pytest.raises(BenchmarkDataError, match="overlap"):
        replace(_dataset(), cases=(_case(), overlapping))


def test_dataset_allows_exact_interval_protocol_ab_with_identical_labels() -> None:
    storyboard_protocol = PreparedInputProtocol(
        "storyboard-v1",
        "storyboard",
        ComponentIdentity("builder", "storyboard-builder@1"),
        (ComponentIdentity("frame-count", "12"),),
    )
    storyboard_case = replace(
        _case(),
        case_id="made-shot-a-storyboard",
        input_kind="storyboard",
        prepared_input_protocol_id="storyboard-v1",
        prepared_input_sha256="d" * 64,
        notes="Same labels, alternate prepared representation",
    )

    dataset = replace(
        _dataset(),
        preparation_protocols=(_protocol(), storyboard_protocol),
        cases=(_case(), storyboard_case),
    )

    assert {case.input_kind for case in dataset.cases} == {
        "native_video",
        "storyboard",
    }


def test_exact_interval_protocol_ab_rejects_duplicate_protocol_or_conflicting_labels() -> None:
    storyboard_protocol = PreparedInputProtocol(
        "storyboard-v1",
        "storyboard",
        ComponentIdentity("builder", "storyboard-builder@1"),
        (ComponentIdentity("frame-count", "12"),),
    )
    storyboard_case = replace(
        _case(),
        case_id="made-shot-a-storyboard",
        input_kind="storyboard",
        prepared_input_protocol_id="storyboard-v1",
        prepared_input_sha256="d" * 64,
    )
    duplicate_protocol_case = replace(
        _case(),
        case_id="made-shot-a-second-native",
        prepared_input_sha256="e" * 64,
    )
    conflicting_labels = replace(
        storyboard_case,
        expected_facts=tuple(
            replace(fact, expected=False)
            if fact.fact_id == "ball_through_hoop"
            else fact
            for fact in storyboard_case.expected_facts
        ),
    )

    with pytest.raises(BenchmarkDataError, match="distinct preparation protocols"):
        replace(_dataset(), cases=(_case(), duplicate_protocol_case))
    with pytest.raises(BenchmarkDataError, match="identical labels"):
        replace(
            _dataset(),
            preparation_protocols=(_protocol(), storyboard_protocol),
            cases=(_case(), conflicting_labels),
        )

    duplicate_semantic_protocol = replace(
        _protocol(),
        protocol_id="native-mp4-alias-v1",
    )
    semantic_duplicate_case = replace(
        _case(),
        case_id="made-shot-a-native-alias",
        prepared_input_protocol_id="native-mp4-alias-v1",
        prepared_input_sha256="e" * 64,
    )
    with pytest.raises(BenchmarkDataError, match="structural identities"):
        replace(
            _dataset(),
            preparation_protocols=(_protocol(), duplicate_semantic_protocol),
            cases=(_case(), semantic_duplicate_case),
        )


def test_dataset_allows_touching_non_overlapping_intervals() -> None:
    adjacent = _case(
        "adjacent",
        start_seconds=18.0,
        end_seconds=20.0,
        prepared_digest="d" * 64,
    )

    dataset = replace(_dataset(), cases=(_case(), adjacent))

    assert len(dataset.cases) == 2


def test_dataset_rejects_whole_asset_and_group_cross_split_leakage() -> None:
    same_asset_other_split = _case(
        "leaked-asset",
        start_seconds=30.0,
        end_seconds=35.0,
        prepared_digest="d" * 64,
        split="promotion_holdout",
        split_group="match-a",
    )
    second_asset = _asset("game-b", digest="b" * 64)
    same_group_other_split = _case(
        "leaked-group",
        asset_id="game-b",
        start_seconds=30.0,
        end_seconds=35.0,
        prepared_digest="d" * 64,
        split="promotion_holdout",
        split_group="match-a",
    )

    with pytest.raises(BenchmarkDataError, match="cross-split"):
        replace(_dataset(), cases=(_case(), same_asset_other_split))
    with pytest.raises(BenchmarkDataError, match="cross-split"):
        replace(
            _dataset(),
            assets=(_asset(), second_asset),
            cases=(_case(), same_group_other_split),
        )
    with pytest.raises(BenchmarkDataError, match="whole-asset split_group"):
        replace(
            _dataset(),
            cases=(
                _case(),
                _case(
                    "other-group",
                    start_seconds=30.0,
                    end_seconds=35.0,
                    prepared_digest="d" * 64,
                    split_group="different-match",
                ),
            ),
        )


def test_promotion_holdout_is_an_explicit_usage_guard() -> None:
    regression = _case()
    promotion = replace(regression, split="promotion_holdout")

    assert regression.is_promotion_holdout is False
    assert promotion.is_promotion_holdout is True


def test_dataset_rejects_duplicate_prepared_content_and_unused_entries() -> None:
    duplicate_content = _case(
        "duplicate-content",
        start_seconds=30.0,
        end_seconds=35.0,
    )

    with pytest.raises(BenchmarkDataError, match="prepared input SHA-256"):
        replace(_dataset(), cases=(_case(), duplicate_content))
    with pytest.raises(BenchmarkDataError, match="unused asset"):
        replace(
            _dataset(),
            assets=(_asset(), _asset("unused", digest="b" * 64)),
        )
    with pytest.raises(BenchmarkDataError, match="unused preparation protocol"):
        replace(
            _dataset(),
            preparation_protocols=(
                _protocol(),
                _protocol("unused-v1", input_kind="storyboard"),
            ),
        )


def test_dataset_rejects_invalid_schema_empty_collections_and_resource_bounds() -> None:
    with pytest.raises(BenchmarkDataError, match="schema_version"):
        replace(_dataset(), schema_version=2)
    with pytest.raises(BenchmarkDataError, match="assets must not be empty"):
        replace(_dataset(), assets=())
    with pytest.raises(BenchmarkDataError, match="protocols must not be empty"):
        replace(_dataset(), preparation_protocols=())
    with pytest.raises(BenchmarkDataError, match="cases must not be empty"):
        replace(_dataset(), cases=())

    provenance = AssetProvenance(source="fixture", license_id="MIT")
    too_many_assets = tuple(
        BenchmarkAsset(
            asset_id=f"asset-{index}",
            sha256=f"{index + 1:064x}",
            byte_size=1,
            duration_seconds=1.0,
            provenance=provenance,
        )
        for index in range(129)
    )
    with pytest.raises(BenchmarkDataError, match="asset count limit"):
        replace(_dataset(), assets=too_many_assets)

    with pytest.raises(BenchmarkDataError, match="protocol count limit"):
        replace(
            _dataset(),
            preparation_protocols=tuple(
                _protocol(f"protocol-{index}")
                for index in range(MAX_VIDEO_VERIFIER_PREPARATION_PROTOCOLS + 1)
            ),
        )

    with pytest.raises(BenchmarkDataError, match="case count limit"):
        replace(
            _dataset(),
            cases=(_case(),) * (MAX_VIDEO_VERIFIER_CASES + 1),
        )

    with pytest.raises(BenchmarkDataError, match="prepared input byte limit"):
        replace(
            _dataset(),
            cases=(
                replace(_case(), prepared_input_byte_size=MAX_PREPARED_INPUT_BYTES),
            )
            * 1_025,
        )

    oversized_assets = tuple(
        BenchmarkAsset(
            asset_id=f"large-{index}",
            sha256=f"{index + 1:064x}",
            byte_size=16 * 1024**3,
            duration_seconds=1.0,
            provenance=provenance,
        )
        for index in range(9)
    )
    with pytest.raises(BenchmarkDataError, match="aggregate media byte limit"):
        replace(_dataset(), assets=oversized_assets)


def test_to_dict_requires_a_video_verifier_dataset() -> None:
    with pytest.raises(BenchmarkDataError, match="VideoVerifierDataset"):
        video_verifier_dataset_to_dict(object())  # type: ignore[arg-type]


def test_deserialize_rejects_unknown_nested_fields_duplicate_keys_and_nonfinite_json() -> None:
    value = json.loads(video_verifier_dataset_to_json_bytes(_dataset()))
    value["cases"][0]["unexpected"] = True

    with pytest.raises(BenchmarkDataError, match="unexpected fields"):
        video_verifier_dataset_from_json(json.dumps(value))
    with pytest.raises(BenchmarkDataError, match="duplicate field"):
        video_verifier_dataset_from_json(
            '{"schema_version":1,"schema_version":1}'
        )
    with pytest.raises(BenchmarkDataError, match="not finite"):
        video_verifier_dataset_from_json(
            video_verifier_dataset_to_json_bytes(_dataset()).decode().replace(
                '"start_seconds": 10.0',
                '"start_seconds": NaN',
            )
        )


def test_deserialize_rejects_non_array_expected_facts() -> None:
    value = json.loads(video_verifier_dataset_to_json_bytes(_dataset()))
    value["cases"][0]["expected_facts"] = {"shot_attempt": True}

    with pytest.raises(BenchmarkDataError, match="JSON array"):
        video_verifier_dataset_from_json(json.dumps(value))


def test_deserialize_rejects_oversized_or_non_text_manifest_input() -> None:
    with pytest.raises(BenchmarkDataError, match="JSON byte size limit"):
        video_verifier_dataset_from_json(
            b" " * (MAX_VIDEO_VERIFIER_MANIFEST_BYTES + 1)
        )
    with pytest.raises(BenchmarkDataError, match="JSON byte size limit"):
        video_verifier_dataset_from_json(
            " " * (MAX_VIDEO_VERIFIER_MANIFEST_BYTES + 1)
        )
    with pytest.raises(BenchmarkDataError, match="invalid Unicode"):
        video_verifier_dataset_from_json("\ud800")
    with pytest.raises(BenchmarkDataError, match="bytes or a string"):
        video_verifier_dataset_from_json(object())  # type: ignore[arg-type]


def test_committed_private_regression_seed_is_canonical_and_not_holdout() -> None:
    repository_root = Path(__file__).parents[2]
    path = repository_root / "docs/benchmarks/video-verifier/seed-v1.json"

    payload = path.read_bytes()
    dataset = video_verifier_dataset_from_json(payload)

    assert dataset.dataset_id == "video-verifier-private-regression"
    assert len(dataset.assets) == 2
    assert len(dataset.cases) == 10
    assert all(case.split == "regression_seen" for case in dataset.cases)
    assert all(not case.is_promotion_holdout for case in dataset.cases)
    assert video_verifier_dataset_to_json_bytes(dataset) == payload
    assert (
        video_verifier_dataset_revision(dataset)
        == "2758ce6fd86c4ad921ae29bff83126a31a5ddbfabdca79748c58fa3d42f1763b"
    )


def test_committed_exploratory_evidence_is_sanitized_and_self_disqualifying() -> None:
    repository_root = Path(__file__).parents[2]
    path = (
        repository_root
        / "docs/benchmarks/video-verifier/exploratory-ab-2026-08-23.json"
    )

    evidence_text = path.read_text(encoding="utf-8")
    evidence = json.loads(evidence_text)

    assert evidence["status"] == "exploratory_confounded_not_for_promotion"
    assert evidence["code_context"]["runner_committed"] is False
    assert len(evidence["results"]) == 3
    assert sum(
        attempt["status"] == "metal_oom"
        for result in evidence["results"]
        for attempt in result.get("attempts", [])
    ) == 2
    completed_cases = [
        case
        for result in evidence["results"]
        for case in result.get("cases", [])
        if case["status"] == "completed"
    ]
    assert len(completed_cases) == 4
    assert all("facts_subset" in case and "facts" not in case for case in completed_cases)
    assert "/Users/" not in evidence_text
    assert "data/media/" not in evidence_text
