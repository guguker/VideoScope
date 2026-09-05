from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from videoscope.providers.vision_worker import (
    LocalVisionWorkerRuntime,
    VisionWorkerSettings,
    create_vision_worker_app,
)
from videoscope.providers.vision_worker_contract import (
    MAX_DETECTIONS,
    VisionDetectionResponse,
    identity_fields,
    worker_input_root_identity,
)


TOKEN = "d" * 32
CLIPPED_ADAPTER_REVISION = "rfdetr-coco-rgb-clipped-center-box-v3"


@pytest.fixture
def proof16_edge_predictions() -> list[tuple[list[float], float, str]]:
    # Numerical output only from the isolated proof16 diagnostic. No media fixture.
    return [
        (
            [1.9924163818359375, -0.25765299797058105, 824.6556396484375, 282.8080749511719],
            0.44864892959594727,
            "tv",
        ),
        (
            [2.2537994384765625, 485.2027893066406, 962.1627197265625, 539.3308715820312],
            0.4118783175945282,
            "dining table",
        ),
    ]


def _prediction_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    predictions: list[tuple[object, float, str]],
) -> tuple[TestClient, LocalVisionWorkerRuntime, dict[str, object], list[dict[str, object]]]:
    source = tmp_path / "frame.png"
    with Image.new("RGB", (960, 540)) as frame:
        frame.save(source)
    specification = VisionWorkerSettings(api_key=TOKEN, _env_file=None).specification()
    runtime = LocalVisionWorkerRuntime(
        specification=specification,
        detector_checkpoint=tmp_path / "unused-checkpoint.pth",
    )
    calls: list[dict[str, object]] = []

    class Detections:
        xyxy = [row[0] for row in predictions]
        confidence = [row[1] for row in predictions]
        class_id = list(range(len(predictions)))
        data = {"class_name": [row[2] for row in predictions]}

        def __len__(self) -> int:
            return len(predictions)

    def predict(frame: Image.Image, **kwargs: object) -> Detections:
        calls.append({"size": frame.size, "mode": frame.mode, **kwargs})
        return Detections()

    monkeypatch.setattr(runtime, "_load_detector", lambda: SimpleNamespace(predict=predict))
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(__path__=[]))
    monkeypatch.setitem(sys.modules, "rfdetr.assets", SimpleNamespace(__path__=[]))
    monkeypatch.setitem(
        sys.modules,
        "rfdetr.assets.coco_classes",
        SimpleNamespace(COCO_CLASSES={}),
    )
    app = create_vision_worker_app(runtime=runtime, input_root=tmp_path, api_key=TOKEN)
    client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50100))
    payload = source.read_bytes()
    request = {
        **identity_fields(specification, input_root_identity=worker_input_root_identity(tmp_path)),
        "request_id": "e" * 32,
        "source": {
            "item_id": "detect-source",
            "relative_path": source.name,
            "expected_sha256": sha256(payload).hexdigest(),
            "expected_size_bytes": len(payload),
        },
        "minimum_confidence": specification.minimum_confidence,
    }
    return client, runtime, request, calls


def _post(client: TestClient, request: dict[str, object]):  # type: ignore[no-untyped-def]
    return client.post(
        "/v1/detect", json=request, headers={"Authorization": f"Bearer {TOKEN}"}
    )


def test_real_edge_predictions_survive_the_strict_worker_response_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proof16_edge_predictions: list[tuple[list[float], float, str]],
) -> None:
    client, _runtime, request, calls = _prediction_client(
        tmp_path, monkeypatch, proof16_edge_predictions
    )

    response = _post(client, request)

    assert response.status_code == 200
    payload = VisionDetectionResponse.model_validate(response.json())
    assert (payload.image_width, payload.image_height) == (960, 540)
    assert [row.label for row in payload.detections] == ["tv", "dining table"]
    assert [row.confidence for row in payload.detections] == [
        0.44864892959594727, 0.4118783175945282
    ]
    expected_boxes = [
        [1.9924163818359375, 0.0, 824.6556396484375, 282.8080749511719],
        [2.2537994384765625, 485.2027893066406, 960.0, 539.3308715820312],
    ]
    for detection, expected in zip(payload.detections, expected_boxes, strict=True):
        assert [
            detection.x - detection.width / 2,
            detection.y - detection.height / 2,
            detection.x + detection.width / 2,
            detection.y + detection.height / 2,
        ] == expected
    assert calls == [{
        "size": (960, 540), "mode": "RGB", "threshold": 0.25,
        "include_source_image": False,
    }]


@pytest.mark.parametrize(
    "box",
    [
        [-8.0, -9.0, -1.0, -2.0],
        [961.0, 1.0, 970.0, 20.0],
        [1.0, 541.0, 20.0, 550.0],
        [-8.0, 1.0, 0.0, 20.0],
        [960.0, 1.0, 970.0, 20.0],
        [1.0, -9.0, 20.0, 0.0],
        [1.0, 540.0, 20.0, 550.0],
        [12.0, 13.0, 12.0, 20.0],
        [12.0, 13.0, 20.0, 13.0],
    ],
)
def test_empty_or_outside_predictions_are_discarded_without_losing_valid_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, box: list[float]
) -> None:
    client, _runtime, request, _calls = _prediction_client(
        tmp_path, monkeypatch,
        [(box, 0.9, "outside"), ([1.0, 2.0, 5.0, 8.0], 0.8, "person")],
    )

    response = _post(client, request)

    assert response.status_code == 200
    assert response.json()["detections"] == [{
        "label": "person", "confidence": 0.8,
        "x": 3.0, "y": 5.0, "width": 4.0, "height": 6.0,
    }]


def test_prediction_spanning_all_frame_edges_is_clipped_to_the_whole_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _runtime, request, _calls = _prediction_client(
        tmp_path, monkeypatch, [([-10.0, -20.0, 970.0, 560.0], 0.75, "person")]
    )

    response = _post(client, request)

    assert response.status_code == 200
    assert response.json()["detections"] == [{
        "label": "person", "confidence": 0.75,
        "x": 480.0, "y": 270.0, "width": 960.0, "height": 540.0,
    }]


def test_numpy_float32_predictions_keep_the_serving_numeric_type_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    client, _runtime, request, _calls = _prediction_client(
        tmp_path, monkeypatch,
        [(np.asarray([-10.0, -20.0, 970.0, 560.0], dtype=np.float32), 0.75, "person")],
    )

    response = _post(client, request)

    assert response.status_code == 200
    assert response.json()["detections"] == [{
        "label": "person", "confidence": 0.75,
        "x": 480.0, "y": 270.0, "width": 960.0, "height": 540.0,
    }]


def test_a_request_with_only_outside_predictions_returns_no_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _runtime, request, _calls = _prediction_client(
        tmp_path, monkeypatch, [([-8.0, -9.0, -1.0, -2.0], 0.9, "outside")],
    )

    response = _post(client, request)

    assert response.status_code == 200
    assert response.json()["detections"] == []


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.1, 1.1])
def test_discarding_an_outside_box_does_not_hide_invalid_confidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confidence: float
) -> None:
    client, _runtime, request, _calls = _prediction_client(
        tmp_path, monkeypatch,
        [([-8.0, -9.0, -1.0, -2.0], confidence, "outside")],
    )

    response = _post(client, request)

    assert response.status_code == 503
    assert response.json() == {"detail": "Vision worker detection failed"}


@pytest.mark.parametrize(
    "box",
    [
        [float("nan"), 2.0, 5.0, 8.0],
        [1.0, float("-inf"), 5.0, 8.0],
        [1.0, 2.0, float("inf"), 8.0],
        [1.0, 2.0, 5.0, float("nan")],
        [5.0, 2.0, 1.0, 8.0],
        [1.0, 8.0, 5.0, 2.0],
        [1.0, 2.0, 5.0],
        [1.0, 2.0, 5.0, 8.0, 9.0],
        [None, 2.0, 5.0, 8.0],
        ["bad", 2.0, 5.0, 8.0],
        ["1", 2.0, 5.0, 8.0],
        [True, 2.0, 5.0, 8.0],
    ],
)
def test_malformed_nonfinite_and_reversed_boxes_still_fail_the_whole_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, box: object
) -> None:
    client, _runtime, request, _calls = _prediction_client(
        tmp_path, monkeypatch,
        [([1.0, 2.0, 5.0, 8.0], 0.8, "person"), (box, 0.9, "invalid")],
    )

    response = _post(client, request)

    assert response.status_code == 503
    assert response.json() == {"detail": "Vision worker detection failed"}


def test_outside_predictions_cannot_bypass_the_raw_detection_count_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _runtime, request, _calls = _prediction_client(
        tmp_path, monkeypatch,
        [([-8.0, -9.0, -1.0, -2.0], 0.9, "outside")] * (MAX_DETECTIONS + 1),
    )

    response = _post(client, request)

    assert response.status_code == 503
    assert response.json() == {"detail": "Vision worker detection failed"}


def test_clipped_adapter_has_a_new_detector_identity_and_rejects_old_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, runtime, request, calls = _prediction_client(tmp_path, monkeypatch, [])
    specification = runtime.specification
    assert specification.detector_adapter_revision == CLIPPED_ADAPTER_REVISION
    with pytest.raises(ValueError, match="adapter revision is not implemented"):
        replace(specification, detector_adapter_revision="rfdetr-coco-rgb-center-box-v2")
    old_projection = {
        **specification.detector_projection(),
        "adapter_revision": "rfdetr-coco-rgb-center-box-v2",
    }
    old_identity = sha256(json.dumps(
        old_projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    assert old_identity != specification.detector_identity
    old_request = {**request, "detector_specification_hash": old_identity}

    response = _post(client, old_request)

    assert response.status_code == 409
    assert response.json() == {"detail": "Vision worker identity does not match"}
    assert calls == []
