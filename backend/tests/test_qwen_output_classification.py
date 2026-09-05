from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from videoscope.benchmark.video_verifier_runner import (
    QwenWorkerStrictVerifier,
    VideoVerifierRunner,
    load_video_verifier_candidates,
    load_video_verifier_run,
)
from videoscope.benchmark.video_verifier_schema import (
    video_verifier_dataset_from_json,
    video_verifier_dataset_revision,
)
from videoscope.providers import qwen_worker


_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/benchmarks/video-verifier/seed-v1.json"
)
_TOKEN = "t" * 32
_MODEL_IDENTITY = "synthetic/qwen@" + "a" * 40
_EVIDENCE = "Synthetic raw model evidence must not enter the benchmark manifest."


def _explicit_abstention(case_id: str) -> dict[str, object]:
    if case_id.startswith("sports-"):
        return {
            "shot_attempt": None,
            "ball_through_hoop": None,
            "shooter_outside_arc": None,
            "three_point_signal": None,
            "shooter_jersey": None,
            "evidence": _EVIDENCE,
        }
    return {
        "matches_query": None,
        "confidence": 0.0,
        "event_start": None,
        "event_end": None,
        "shot_attempt": None,
        "made": None,
        "three_point": None,
        "shooter_jersey": None,
        "evidence": _EVIDENCE,
    }


def _classify_raw_response(tmp_path: Path, case_id: str, text: str):  # type: ignore[no-untyped-def]
    """Run the real parser, worker HTTP contract, client and direct classifier."""
    frozen = video_verifier_dataset_from_json(_DATASET_PATH.read_bytes())
    selected = next(case for case in frozen.cases if case.case_id == case_id)
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    source = input_root / "synthetic.mp4"
    content = b"synthetic input; no video decoder or model is used"
    source.write_bytes(content)
    before = source.stat()
    selected = replace(
        selected,
        prepared_input_sha256=sha256(content).hexdigest(),
        prepared_input_byte_size=len(content),
    )
    dataset = replace(
        frozen,
        dataset_id="synthetic-qwen-output-classification",
        description="Synthetic bytes with frozen labels for output classification only",
        assets=tuple(
            asset for asset in frozen.assets
            if asset.asset_id == selected.source_interval.asset_id
        ),
        preparation_protocols=tuple(
            protocol for protocol in frozen.preparation_protocols
            if protocol.protocol_id == selected.prepared_input_protocol_id
        ),
        cases=(selected,),
    )
    candidate_file = tmp_path / "candidates.json"
    candidate_file.write_text(json.dumps({
        "schema_version": 1,
        "dataset_revision": video_verifier_dataset_revision(dataset),
        "candidates": [{
            "case_id": case_id,
            "candidate_id": "synthetic-proposal",
            "prepared_input_path": str(source),
            "proposal_rank": 1,
            "proposal_score": 0.0,
        }],
    }))
    materialized: list[Path] = []
    judge_responses: list[tuple[int, object]] = []

    class TextRuntime:
        model_identity = _MODEL_IDENTITY
        available = True
        loaded = True
        judge = qwen_worker.MLXQwenWorkerRuntime.judge

        def _generate_video(self, path: Path, _request: object) -> str:
            assert path != source
            assert path.read_bytes() == content
            materialized.append(path)
            return text

    app = qwen_worker.create_qwen_worker_app(
        runtime=TextRuntime(), input_root=input_root, api_key=_TOKEN
    )
    output = tmp_path / "run.json"
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as transport:
        class RecordingHTTP:
            def get(self, url: str, **kwargs):  # type: ignore[no-untyped-def]
                kwargs.pop("timeout", None)
                return transport.get(url, **kwargs)

            def post(self, url: str, **kwargs):  # type: ignore[no-untyped-def]
                kwargs.pop("timeout", None)
                response = transport.post(url, **kwargs)
                judge_responses.append((response.status_code, response.json()))
                return response

        client = qwen_worker.QwenWorkerClient(
            endpoint="http://127.0.0.1",
            api_key=_TOKEN,
            input_root=input_root,
            expected_model_identity=_MODEL_IDENTITY,
            client=RecordingHTTP(),
        )
        run = VideoVerifierRunner(
            verifier=QwenWorkerStrictVerifier(client), code_sha="a" * 40
        ).run(
            dataset,
            load_video_verifier_candidates(candidate_file, dataset),
            run_id="synthetic-output-classification",
            output_path=output,
        )
    assert load_video_verifier_run(output) == run
    assert len(materialized) == 1
    assert not materialized[0].exists()
    after = source.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
        after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
    )
    assert source.read_bytes() == content
    retained = output.read_text()
    assert _EVIDENCE not in retained
    assert str(tmp_path) not in retained
    assert _TOKEN not in retained
    return run, judge_responses


@pytest.mark.parametrize("case_id", ["sports-abstain", "panel-roundtable-positive"])
@pytest.mark.parametrize(
    "malformation",
    ["empty", "unknown_only", "truncated", "missing_field", "wrong_type", "extra_field", "duplicate_key"],
)
def test_malformed_model_output_is_worker_failure_and_runner_infrastructure(
    tmp_path: Path, case_id: str, malformation: str
) -> None:
    payload = _explicit_abstention(case_id)
    fact = "shot_attempt" if case_id.startswith("sports-") else "matches_query"
    if malformation == "empty":
        text = "{}"
    elif malformation == "unknown_only":
        text = '{"unrelated":"answer"}'
    elif malformation == "truncated":
        text = '{"evidence":"truncated'
    elif malformation == "duplicate_key":
        text = json.dumps(payload)[:-1] + f', "{fact}": true}}'
    else:
        if malformation == "missing_field":
            del payload[fact]
        elif malformation == "wrong_type":
            payload[fact] = 1
        elif malformation == "extra_field":
            payload["unexpected"] = True
        text = json.dumps(payload)

    run, responses = _classify_raw_response(tmp_path, case_id, text)

    assert responses == [(503, {"detail": "Qwen worker inference failed"})]
    assert run.run_status == "infrastructure_failed"
    assert run.summary.infrastructure_error_count == 1
    assert run.summary.match_count == run.summary.model_miss_count == 0
    assert run.attempts[0].prediction is None
    assert run.attempts[0].error_code == "verifier_execution_failed"


@pytest.mark.parametrize(
    ("case_id", "expected_status"),
    [
        ("sports-abstain", "match"),
        ("sports-made-free-throw", "model_miss"),
        ("panel-roundtable-positive", "model_miss"),
    ],
)
def test_explicit_typed_null_remains_valid_abstention(
    tmp_path: Path, case_id: str, expected_status: str
) -> None:
    run, responses = _classify_raw_response(
        tmp_path, case_id, json.dumps(_explicit_abstention(case_id))
    )

    assert responses[0][0] == 200
    assert run.run_status == "complete"
    assert run.summary.infrastructure_error_count == 0
    attempt = run.attempts[0]
    assert attempt.status == expected_status
    assert attempt.error_code is None
    assert attempt.prediction is not None
    assert all(fact.expected is None for fact in attempt.prediction.facts)
    assert attempt.prediction.predicted_jersey is None
