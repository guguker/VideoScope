from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pytest

from videoscope.benchmark.schema import (
    AssetProvenance,
    BenchmarkAsset,
    BenchmarkInterval,
    ComponentIdentity,
)
from videoscope.benchmark.video_verifier_runner import (
    QwenWorkerStrictVerifier,
    VideoVerifierCandidateBinding,
    VideoVerifierExecutionError,
    VideoVerifierPrediction,
    VideoVerifierRunner,
    load_video_verifier_candidates,
    load_video_verifier_run,
    main,
)
from videoscope.benchmark.video_verifier_schema import (
    PreparedInputProtocol,
    VideoVerifierCase,
    VideoVerifierDataset,
    VideoVerifierExpectedFact,
    video_verifier_dataset_revision,
    video_verifier_dataset_to_json_bytes,
)
from videoscope.providers.qwen_video import QwenVideoJudgement


CODE_SHA = "a" * 40


def _facts(**values: bool | None) -> tuple[VideoVerifierExpectedFact, ...]:
    return tuple(
        VideoVerifierExpectedFact(fact_id, value)
        for fact_id, value in values.items()
    )


def _fixture(tmp_path: Path):  # type: ignore[no-untyped-def]
    native = tmp_path / "sports.mp4"
    storyboard = tmp_path / "generic.jpg"
    native.write_bytes(b"sports prepared input")
    storyboard.write_bytes(b"generic prepared input")
    dataset = VideoVerifierDataset(
        schema_version=1,
        dataset_id="direct-verifier-test",
        dataset_version="1.0.0",
        description="Strict direct verifier fixture",
        assets=(
            BenchmarkAsset(
                asset_id="asset-a",
                sha256="f" * 64,
                byte_size=1_024,
                duration_seconds=20.0,
                provenance=AssetProvenance(
                    source="test fixture",
                    source_uri="https://example.test/asset-a",
                    license_id="CC0-1.0",
                    license_uri="https://creativecommons.org/publicdomain/zero/1.0/",
                    attribution="Synthetic test fixture",
                ),
            ),
        ),
        preparation_protocols=(
            PreparedInputProtocol(
                protocol_id="native-v1",
                input_kind="native_video",
                preparer=ComponentIdentity("clipper", "fake@1"),
                parameters=(ComponentIdentity("container", "mp4"),),
            ),
            PreparedInputProtocol(
                protocol_id="storyboard-v1",
                input_kind="storyboard",
                preparer=ComponentIdentity("storyboard", "fake@1"),
                parameters=(ComponentIdentity("container", "jpeg"),),
            ),
        ),
        cases=(
            VideoVerifierCase(
                case_id="sports-positive",
                stratum="basketball_facts",
                input_kind="native_video",
                source_interval=BenchmarkInterval("asset-a", 0.0, 4.0),
                prepared_input_protocol_id="native-v1",
                prepared_input_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                prepared_input_byte_size=native.stat().st_size,
                query=None,
                expected_facts=_facts(
                    shot_attempt=True,
                    ball_through_hoop=True,
                    shooter_outside_arc=True,
                    three_point_signal=True,
                ),
                expected_jersey="15",
                label_quality="gold",
                split="regression_seen",
                split_group="asset-a",
            ),
            VideoVerifierCase(
                case_id="generic-positive",
                stratum="generic_visual",
                input_kind="storyboard",
                source_interval=BenchmarkInterval("asset-a", 5.0, 9.0),
                prepared_input_protocol_id="storyboard-v1",
                prepared_input_sha256=hashlib.sha256(
                    storyboard.read_bytes()
                ).hexdigest(),
                prepared_input_byte_size=storyboard.stat().st_size,
                query="a person waves",
                expected_facts=_facts(matches_query=True),
                expected_jersey=None,
                label_quality="gold",
                split="regression_seen",
                split_group="asset-a",
            ),
        ),
    )
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_revision": video_verifier_dataset_revision(dataset),
                "candidates": [
                    {
                        "case_id": "sports-positive",
                        "candidate_id": "proposal-sports",
                        "prepared_input_path": str(native),
                        "proposal_rank": 1,
                        "proposal_score": 0.9,
                    },
                    {
                        "case_id": "generic-positive",
                        "candidate_id": "proposal-generic",
                        "prepared_input_path": str(storyboard),
                        "proposal_rank": 2,
                        "proposal_score": 0.8,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return dataset, candidates_path, native, storyboard


class _StrictVerifier:
    identity = ComponentIdentity("video_verifier", "fake-strict-verifier@1")

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def verify_strict(self, case, candidate):  # type: ignore[no-untyped-def]
        self.calls.append((case.case_id, candidate.candidate_id))
        if case.stratum == "basketball_facts":
            return VideoVerifierPrediction(
                facts=_facts(
                    shot_attempt=True,
                    ball_through_hoop=True,
                    shooter_outside_arc=True,
                    three_point_signal=True,
                ),
                predicted_jersey="15",
                confidence=0.95,
            )
        return VideoVerifierPrediction(
            facts=_facts(matches_query=False),
            predicted_jersey=None,
            confidence=0.7,
        )


def test_runner_calls_only_strict_contract_and_separates_model_miss(
    tmp_path: Path,
) -> None:
    dataset, candidates_path, _native, _storyboard = _fixture(tmp_path)
    verifier = _StrictVerifier()
    candidates = load_video_verifier_candidates(candidates_path, dataset)
    ticks = iter((10.0, 10.125, 20.0, 20.25))
    output = tmp_path / "result.json"
    runner = VideoVerifierRunner(
        verifier=verifier,
        code_sha=CODE_SHA,
        clock=lambda: datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
        timer=lambda: next(ticks),
    )

    manifest = runner.run(
        dataset,
        candidates,
        run_id="strict-run",
        output_path=output,
    )

    assert verifier.calls == [
        ("sports-positive", "proposal-sports"),
        ("generic-positive", "proposal-generic"),
    ]
    assert manifest.run_status == "complete"
    assert [(item.case_id, item.status) for item in manifest.attempts] == [
        ("sports-positive", "match"),
        ("generic-positive", "model_miss"),
    ]
    assert manifest.summary.match_count == 1
    assert manifest.summary.model_miss_count == 1
    assert manifest.summary.infrastructure_error_count == 0
    assert manifest.strict_no_fallback is True
    assert manifest.attempts[0].latency_ms == 125.0
    assert manifest.attempts[1].latency_ms == 250.0
    assert load_video_verifier_run(output) == manifest
    payload = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in payload
    assert "prepared_input_path" not in payload
    assert "Strict direct verifier fixture" not in payload
    assert "a person waves" not in payload
    with pytest.raises(FrozenInstanceError):
        manifest.run_id = "mutated"  # type: ignore[misc]

    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        runner.run(
            dataset,
            candidates,
            run_id="second-run",
            output_path=output,
        )
    assert output.read_bytes() == original


class _FailingVerifier:
    identity = ComponentIdentity("video_verifier", "fake-failing-verifier@1")

    def __init__(self) -> None:
        self.calls: list[str] = []

    def verify_strict(self, case, _candidate):  # type: ignore[no-untyped-def]
        self.calls.append(case.case_id)
        raise RuntimeError("private failure at /Users/alice/private/movie.mp4")


def test_runner_records_sanitized_infrastructure_errors_not_model_misses(
    tmp_path: Path,
) -> None:
    dataset, candidates_path, _native, storyboard = _fixture(tmp_path)
    candidate_payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    storyboard.write_bytes(b"changed after candidate manifest was frozen")
    candidates_path.write_text(json.dumps(candidate_payload), encoding="utf-8")
    verifier = _FailingVerifier()
    candidates = load_video_verifier_candidates(candidates_path, dataset)
    output = tmp_path / "infra-result.json"

    manifest = VideoVerifierRunner(
        verifier=verifier,
        code_sha=CODE_SHA,
        timer=lambda: 1.0,
    ).run(
        dataset,
        candidates,
        run_id="infra-run",
        output_path=output,
    )

    assert manifest.run_status == "infrastructure_failed"
    assert [
        (item.case_id, item.status, item.error_code) for item in manifest.attempts
    ] == [
        ("sports-positive", "infrastructure_error", "verifier_execution_failed"),
        (
            "generic-positive",
            "infrastructure_error",
            "prepared_input_identity_mismatch",
        ),
    ]
    assert verifier.calls == ["sports-positive"]
    assert manifest.summary.model_miss_count == 0
    assert manifest.summary.infrastructure_error_count == 2
    serialized = output.read_text(encoding="utf-8")
    assert "/Users/alice" not in serialized
    assert "private failure" not in serialized


def test_runner_rejects_non_strict_adapter_before_publishing(tmp_path: Path) -> None:
    dataset, candidates_path, _native, _storyboard = _fixture(tmp_path)
    candidates = load_video_verifier_candidates(candidates_path, dataset)

    class FallbackOnlyVerifier:
        identity = ComponentIdentity("video_verifier", "fallback-only@1")

        def verify(self, *_args):  # type: ignore[no-untyped-def]
            raise AssertionError("fallback must not be called")

    output = tmp_path / "must-not-exist.json"
    runner = VideoVerifierRunner(
        verifier=FallbackOnlyVerifier(),  # type: ignore[arg-type]
        code_sha=CODE_SHA,
    )

    with pytest.raises(VideoVerifierExecutionError, match="strict"):
        runner.run(
            dataset,
            candidates,
            run_id="invalid-run",
            output_path=output,
        )
    assert not output.exists()


def test_candidate_manifest_requires_exact_frozen_case_coverage(tmp_path: Path) -> None:
    dataset, candidates_path, _native, _storyboard = _fixture(tmp_path)
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    payload["candidates"].pop()
    candidates_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VideoVerifierExecutionError, match="exactly"):
        load_video_verifier_candidates(candidates_path, dataset)


def test_runner_recomputes_candidate_binding_revision_before_inference(
    tmp_path: Path,
) -> None:
    dataset, candidates_path, _native, _storyboard = _fixture(tmp_path)
    verifier = _StrictVerifier()
    candidates = replace(
        load_video_verifier_candidates(candidates_path, dataset),
        candidate_set_revision="0" * 64,
    )
    output = tmp_path / "tampered-binding-result.json"

    with pytest.raises(VideoVerifierExecutionError, match="revision"):
        VideoVerifierRunner(verifier=verifier, code_sha=CODE_SHA).run(
            dataset,
            candidates,
            run_id="tampered-binding",
            output_path=output,
        )

    assert verifier.calls == []
    assert not output.exists()


def test_invalid_strict_return_is_an_infrastructure_error(tmp_path: Path) -> None:
    dataset, candidates_path, _native, _storyboard = _fixture(tmp_path)
    candidates = load_video_verifier_candidates(candidates_path, dataset)

    class NoneVerifier:
        identity = ComponentIdentity("video_verifier", "none-verifier@1")

        def verify_strict(self, _case, _candidate):  # type: ignore[no-untyped-def]
            return None

    manifest = VideoVerifierRunner(
        verifier=NoneVerifier(),  # type: ignore[arg-type]
        code_sha=CODE_SHA,
        timer=lambda: 1.0,
    ).run(
        dataset,
        candidates,
        run_id="invalid-contract-run",
        output_path=tmp_path / "invalid-contract-result.json",
    )

    assert manifest.summary.model_miss_count == 0
    assert manifest.summary.infrastructure_error_count == 2
    assert {item.error_code for item in manifest.attempts} == {
        "verifier_contract_invalid"
    }


def test_prediction_is_discarded_if_prepared_input_changes_during_inference(
    tmp_path: Path,
) -> None:
    dataset, candidates_path, native, _storyboard = _fixture(tmp_path)
    candidates = load_video_verifier_candidates(candidates_path, dataset)

    class MutatingVerifier(_StrictVerifier):
        def verify_strict(self, case, candidate):  # type: ignore[no-untyped-def]
            prediction = super().verify_strict(case, candidate)
            if case.case_id == "sports-positive":
                native.write_bytes(b"mutated during inference")
            return prediction

    manifest = VideoVerifierRunner(
        verifier=MutatingVerifier(),
        code_sha=CODE_SHA,
        timer=lambda: 1.0,
    ).run(
        dataset,
        candidates,
        run_id="mutating-input-run",
        output_path=tmp_path / "mutating-input-result.json",
    )

    sports = next(
        item for item in manifest.attempts if item.case_id == "sports-positive"
    )
    assert sports.status == "infrastructure_error"
    assert sports.error_code == "prepared_input_changed"
    assert sports.prediction is None


def test_qwen_adapter_routes_generic_native_video_through_versioned_query_path(
    tmp_path: Path,
) -> None:
    dataset, _candidates_path, native, _storyboard = _fixture(tmp_path)
    generic_case = replace(
        next(case for case in dataset.cases if case.stratum == "generic_visual"),
        input_kind="native_video",
        prepared_input_protocol_id="native-v1",
    )
    candidate = VideoVerifierCandidateBinding(
        case_id=generic_case.case_id,
        candidate_id="generic-native-proposal",
        prepared_input_path=native,
        proposal_rank=1,
        proposal_score=0.8,
    )

    class FakeWorkerClient:
        identity = {
            "mode": "isolated-worker",
            "contract": "qwen-worker-v4",
            "model": "test/model@revision",
            "runtime_identity": "test-runtime@3",
            "source_bundle_sha256": "1" * 64,
            "prompt_protocol_sha256": "2" * 64,
            "input_root_sha256": "3" * 64,
        }

        def __init__(self) -> None:
            self.calls: list[tuple[Path, str, float, int, str, int]] = []

        def status(self):  # type: ignore[no-untyped-def]
            return type("Status", (), {"ready": True})()

        def judge_video_query(
            self,
            source: Path,
            query: str,
            *,
            fps: float,
            max_tokens: int,
            expected_sha256: str | None = None,
            expected_byte_size: int | None = None,
        ) -> QwenVideoJudgement:
            assert expected_sha256 is not None
            assert expected_byte_size is not None
            self.calls.append(
                (source, query, fps, max_tokens, expected_sha256, expected_byte_size)
            )
            return QwenVideoJudgement(matches_query=True, confidence=0.91)

    client = FakeWorkerClient()
    verifier = QwenWorkerStrictVerifier(
        client,  # type: ignore[arg-type]
        fps=2.0,
        max_tokens=320,
    )

    prediction = verifier.verify_strict(generic_case, candidate)

    assert client.calls == [
        (
            native,
            "a person waves",
            2.0,
            320,
            generic_case.prepared_input_sha256,
            generic_case.prepared_input_byte_size,
        )
    ]
    assert prediction.facts == (VideoVerifierExpectedFact("matches_query", True),)
    assert prediction.confidence == 0.91


def test_cli_runs_qwen_worker_adapter_without_serializing_secret_or_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    dataset, candidates_path, _native, _storyboard = _fixture(tmp_path)
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_bytes(video_verifier_dataset_to_json_bytes(dataset))
    output = tmp_path / "cli-result.json"
    monkeypatch.setenv("TEST_QWEN_KEY", "x" * 32)

    class FakeWorkerClient:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            assert kwargs["api_key"] == "x" * 32
            self.identity = {
                "mode": "isolated-worker",
                "contract": "qwen-worker-v4",
                "model": kwargs["expected_model_identity"],
                "runtime_identity": "test-runtime@1",
                "source_bundle_sha256": "1" * 64,
                "prompt_protocol_sha256": "2" * 64,
                "input_root_sha256": "3" * 64,
            }

        def status(self):  # type: ignore[no-untyped-def]
            return type("Status", (), {"ready": True})()

        def judge_video(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return QwenVideoJudgement(
                shot_attempt=True,
                ball_through_hoop=True,
                shooter_outside_arc=True,
                three_point_signal=True,
                shooter_jersey="15",
            )

        def judge_storyboard(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return QwenVideoJudgement(matches_query=True, confidence=0.9)

        def judge_video_query(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return QwenVideoJudgement(matches_query=True, confidence=0.9)

    monkeypatch.setattr(
        "videoscope.benchmark.video_verifier_runner.QwenWorkerClient",
        FakeWorkerClient,
    )
    monkeypatch.setattr(
        "videoscope.benchmark.cli._current_code_sha",
        lambda: CODE_SHA,
    )

    exit_code = main(
        [
            "run",
            "--dataset",
            str(dataset_path),
            "--candidates",
            str(candidates_path),
            "--output",
            str(output),
            "--run-id",
            "cli-run",
            "--code-sha",
            CODE_SHA,
            "--endpoint",
            "http://127.0.0.1:9085",
            "--input-root",
            str(tmp_path),
            "--model-identity",
            "test/model@revision",
            "--api-key-env",
            "TEST_QWEN_KEY",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "attempt_count": 2,
        "run_id": "cli-run",
        "run_status": "complete",
        "status": "written",
    }
    serialized = output.read_text(encoding="utf-8")
    assert "x" * 32 not in serialized
    assert str(tmp_path) not in serialized
    execution_identity = json.loads(load_video_verifier_run(output).verifier_identity.identity)
    assert execution_identity["source_bundle_sha256"] == "1" * 64
    assert execution_identity["prompt_protocol_sha256"] == "2" * 64
    assert execution_identity["input_root_sha256"] == "3" * 64


def test_cli_rejects_supplied_code_sha_that_is_not_clean_current_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset, candidates_path, _native, _storyboard = _fixture(tmp_path)
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_bytes(video_verifier_dataset_to_json_bytes(dataset))
    output = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(
        "videoscope.benchmark.cli._current_code_sha",
        lambda: "b" * 40,
    )
    monkeypatch.setenv("TEST_QWEN_KEY", "x" * 32)

    exit_code = main(
        [
            "run",
            "--dataset",
            str(dataset_path),
            "--candidates",
            str(candidates_path),
            "--output",
            str(output),
            "--run-id",
            "wrong-code",
            "--code-sha",
            CODE_SHA,
            "--endpoint",
            "http://127.0.0.1:9085",
            "--input-root",
            str(tmp_path),
            "--model-identity",
            "test/model@revision",
            "--api-key-env",
            "TEST_QWEN_KEY",
        ]
    )

    assert exit_code == 8
    assert not output.exists()
    assert json.loads(capsys.readouterr().err) == {
        "error": "execution_failed",
        "message": "verifier benchmark execution failed",
    }


def test_runner_rechecks_clean_code_immediately_before_create_once_publication(
    tmp_path: Path,
) -> None:
    dataset, candidates_path, _native, _storyboard = _fixture(tmp_path)
    candidates = load_video_verifier_candidates(candidates_path, dataset)
    observed = iter((CODE_SHA, "b" * 40))
    output = tmp_path / "must-not-publish-after-code-drift.json"

    with pytest.raises(VideoVerifierExecutionError, match="drifted"):
        VideoVerifierRunner(
            verifier=_StrictVerifier(),
            code_sha=CODE_SHA,
            timer=lambda: 1.0,
            code_identity_resolver=lambda: next(observed),
        ).run(
            dataset,
            candidates,
            run_id="code-drift-during-inference",
            output_path=output,
        )

    assert not output.exists()
