from __future__ import annotations

from hashlib import sha256
import json

import pytest

from videoscope.providers import qwen_video


def _response(prompt_kind: str) -> dict[str, object]:
    if prompt_kind == "basketball_facts":
        return {
            "shot_attempt": None,
            "ball_through_hoop": None,
            "shooter_outside_arc": None,
            "three_point_signal": None,
            "shooter_jersey": None,
            "evidence": "The visible frames do not establish the facts.",
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
        "evidence": "The visible frames do not establish the facts.",
    }


@pytest.mark.parametrize("prompt_kind", ["basketball_facts", "generic_query"])
def test_strict_output_preserves_explicit_null_abstention(prompt_kind: str) -> None:
    result = qwen_video.parse_qwen_worker_judgement(
        json.dumps(_response(prompt_kind)), prompt_kind=prompt_kind
    )

    assert result.shot_attempt is None
    assert result.matches_query is None
    assert result.shooter_jersey is None
    assert result.evidence == _response(prompt_kind)["evidence"]


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        '{"unrelated":"answer"}',
        "[]",
        '```json\n{}\n```',
        "prefix {} suffix",
        '{"matches_query": true',
    ],
)
def test_strict_output_rejects_missing_or_non_object_response(text: str) -> None:
    with pytest.raises(ValueError):
        qwen_video.parse_qwen_worker_judgement(text, prompt_kind="generic_query")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("matches_query", 1),
        ("matches_query", "true"),
        ("matches_query", {}),
        ("confidence", None),
        ("confidence", True),
        ("confidence", "0.9"),
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("confidence", float("nan")),
        ("confidence", float("inf")),
        ("event_start", False),
        ("event_start", -1),
        ("event_end", "2.0"),
        ("shooter_jersey", 15),
        ("shooter_jersey", "1234"),
        ("shooter_jersey", "１５"),
        ("evidence", None),
        ("evidence", ["visible"]),
        ("evidence", "\ud800"),
        ("evidence", "x" * 241),
        ("unknown", None),
    ],
)
def test_strict_output_never_coerces_invalid_values(key: str, value: object) -> None:
    text = json.dumps({**_response("generic_query"), key: value})

    with pytest.raises(ValueError):
        qwen_video.parse_qwen_worker_judgement(text, prompt_kind="generic_query")


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e9999"])
def test_strict_output_rejects_nonfinite_numbers(literal: str) -> None:
    text = json.dumps(_response("generic_query")).replace('"confidence": 0.0', f'"confidence": {literal}')

    with pytest.raises(ValueError):
        qwen_video.parse_qwen_worker_judgement(text, prompt_kind="generic_query")


def test_strict_output_rejects_duplicates_and_invalid_intervals() -> None:
    text = json.dumps(_response("generic_query"))
    duplicates = text[:-1] + ', "matches_query": true}'
    invalid_interval = json.dumps({**_response("generic_query"), "event_start": 2, "event_end": 1})

    for invalid in (duplicates, invalid_interval, text + text, text + " " * 4096):
        with pytest.raises(ValueError):
            qwen_video.parse_qwen_worker_judgement(invalid, prompt_kind="generic_query")


def test_strict_output_preserves_valid_values_without_normalizing() -> None:
    payload = {
        **_response("generic_query"),
        "matches_query": True,
        "confidence": 0.75,
        "event_start": 0.25,
        "event_end": 2.5,
        "shooter_jersey": "015",
        "evidence": " " + "x" * 238 + " ",
    }

    result = qwen_video.parse_qwen_worker_judgement(json.dumps(payload), prompt_kind="generic_query")

    assert result.matches_query is True
    assert result.confidence == 0.75
    assert result.event_start == 0.25
    assert result.event_end == 2.5
    assert result.shooter_jersey == "015"
    assert result.evidence == payload["evidence"]


def test_output_schemas_are_fresh_flat_and_bound_to_protocol_identity() -> None:
    protocol = qwen_video._QWEN_PROMPT_PROTOCOL
    output = protocol["structured_output"]
    assert output["decoder"] == "mlx-vlm==0.6.7:llguidance==1.7.6:json-schema-v1"
    assert output["completion"] == "finish_reason=stop"
    for prompt_kind in ("basketball_facts", "generic_query"):
        schema = qwen_video.qwen_response_schema(prompt_kind)
        assert schema == output["schemas"][prompt_kind]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(_response(prompt_kind))
        assert schema["properties"]["evidence"]["maxLength"] == 240
        schema["properties"]["evidence"]["maxLength"] = 999
        assert qwen_video.qwen_response_schema(prompt_kind)["properties"]["evidence"]["maxLength"] == 240
    encoded = json.dumps(protocol, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert qwen_video.QWEN_PROMPT_PROTOCOL_SHA256 == sha256(encoded).hexdigest()


def test_strict_output_rejects_unsupported_prompt_contract() -> None:
    with pytest.raises(ValueError):
        qwen_video.qwen_response_schema("unknown")
    with pytest.raises(ValueError):
        qwen_video.parse_qwen_worker_judgement("{}", prompt_kind="unknown")


@pytest.mark.parametrize("jersey", ["0", "00", "1", "99"])
def test_basketball_output_jersey_matches_frozen_direct_contract(jersey: str) -> None:
    from videoscope.benchmark.video_verifier_runner import QwenWorkerStrictVerifier

    result = qwen_video.parse_qwen_worker_judgement(
        json.dumps({**_response("basketball_facts"), "shooter_jersey": jersey}),
        prompt_kind="basketball_facts",
    )

    prediction = QwenWorkerStrictVerifier._basketball_prediction(result)
    assert prediction.predicted_jersey == jersey
    assert qwen_video.qwen_response_schema("basketball_facts")["properties"]["shooter_jersey"]["pattern"] == "^(?:0|00|[1-9][0-9]?)$"


@pytest.mark.parametrize("jersey", ["015", "123"])
def test_basketball_output_rejects_jerseys_outside_frozen_contract_only(jersey: str) -> None:
    with pytest.raises(ValueError):
        qwen_video.parse_qwen_worker_judgement(
            json.dumps({**_response("basketball_facts"), "shooter_jersey": jersey}),
            prompt_kind="basketball_facts",
        )
    generic = qwen_video.parse_qwen_worker_judgement(
        json.dumps({**_response("generic_query"), "shooter_jersey": jersey}),
        prompt_kind="generic_query",
    )
    assert generic.shooter_jersey == jersey
