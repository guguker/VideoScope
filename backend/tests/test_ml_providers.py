import sys
import types
from pathlib import Path

import pytest

from videoscope.providers.base import ProviderState
from videoscope.providers.lighthouse import LighthouseRetriever
from videoscope.providers.paddle_ocr import PaddleOCRReader, _result_payload
from videoscope.providers.roboflow import RoboflowDetector
from videoscope.providers.whisper import WhisperTranscriber


def test_whisper_transcriber_maps_mlx_segments(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    module = types.ModuleType("mlx_whisper")

    def transcribe(*_args, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {
            "language": "ru",
            "segments": [
                {
                    "start": 1.5,
                    "end": 3.0,
                    "text": "  бросок  ",
                    "avg_logprob": -0.1,
                    "words": [
                        {"word": " бросок", "start": 1.6, "end": 2.8, "probability": 0.92},
                    ],
                },
                {"start": 3.0, "end": 3.0, "text": "empty"},
            ],
        }

    module.transcribe = transcribe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    glossary = tmp_path / "glossary.json"
    glossary.write_text('{"Мозгов":["Mozgov"]}', encoding="utf-8")
    provider = WhisperTranscriber(
        "test-model",
        initial_prompt="Точная русская речь.",
        glossary_path=glossary,
    )

    segments = provider.transcribe(tmp_path / "video.mp4")

    assert provider.status().state is ProviderState.READY
    assert len(segments) == 1
    assert segments[0].text == "бросок"
    assert segments[0].metadata["language"] == "ru"
    assert segments[0].metadata["words"] == [
        {"word": "бросок", "start": 1.6, "end": 2.8, "probability": 0.92}
    ]
    assert "Мозгов" in str(captured["initial_prompt"])
    assert "Mozgov" in str(captured["initial_prompt"])
    assert 0 < segments[0].confidence < 1


def test_paddle_reader_parses_current_result_shape(monkeypatch, tmp_path) -> None:
    class FakeResult:
        json = {"res": {"rec_texts": [" SCORE 87 ", "noise"], "rec_scores": [0.94, 0.1]}}

    class FakeModel:
        def predict(self, _path: str):  # type: ignore[no-untyped-def]
            return [FakeResult()]

    class FakePaddleOCR:
        def __init__(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def predict(self, path: str):  # type: ignore[no-untyped-def]
            return FakeModel().predict(path)

    module = types.ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    reader = PaddleOCRReader(minimum_confidence=0.35)

    output = reader.read(tmp_path / "frame.jpg")

    assert reader.status().state is ProviderState.READY
    assert output == [("SCORE 87", 0.94)]
    assert _result_payload(types.SimpleNamespace(json=lambda: '{"res":{"rec_texts":[]}}')) == {
        "rec_texts": []
    }


def test_roboflow_requires_explicit_configuration() -> None:
    provider = RoboflowDetector(api_key=None, model_id=None)

    assert provider.status().state is ProviderState.NEEDS_CONFIGURATION
    with pytest.raises(RuntimeError, match="not configured"):
        provider.detect(Path("frame.jpg"))


def test_roboflow_detector_uses_sdk_and_supervision(monkeypatch, tmp_path) -> None:
    response = {
        "predictions": [
            {"class": "basketball", "x": 4, "y": 5, "width": 6, "height": 7},
            {"class": "noise"},
        ]
    }

    class FakeClient:
        def __init__(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def infer(self, _image: str, *, model_id: str):  # type: ignore[no-untyped-def]
            assert model_id == "basketball/1"
            return response

    class FakeDetections:
        confidence = [0.91, 0.1]
        class_id = [1, 2]

        @classmethod
        def from_inference(cls, payload):  # type: ignore[no-untyped-def]
            assert payload is response
            return cls()

        def with_nms(self, *, threshold: float):
            assert threshold == 0.5
            return self

        def __len__(self) -> int:
            return 2

    sdk = types.ModuleType("inference_sdk")
    sdk.InferenceHTTPClient = FakeClient  # type: ignore[attr-defined]
    supervision = types.ModuleType("supervision")
    supervision.Detections = FakeDetections  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "inference_sdk", sdk)
    monkeypatch.setitem(sys.modules, "supervision", supervision)
    provider = RoboflowDetector(api_key="secret", model_id="basketball/1")

    tags = provider.detect(tmp_path / "frame.jpg")

    assert provider.status().state is ProviderState.READY
    assert [(tag.label, tag.confidence) for tag in tags] == [("basketball", 0.91)]
    assert tags[0].metadata["width"] == 6


def test_roboflow_detector_runs_local_rfdetr_without_api_key(monkeypatch, tmp_path) -> None:
    class FakeDetections:
        confidence = [0.93]
        class_id = [1]
        xyxy = [[10, 20, 50, 80]]

        def __len__(self) -> int:
            return 1

    class FakeRFDETRSmall:
        def predict(self, image: str, **kwargs):  # type: ignore[no-untyped-def]
            assert image.endswith("frame.jpg")
            assert kwargs == {"threshold": 0.25, "include_source_image": False}
            return FakeDetections()

    rfdetr = types.ModuleType("rfdetr")
    rfdetr.__path__ = []  # type: ignore[attr-defined]
    rfdetr.RFDETRSmall = FakeRFDETRSmall  # type: ignore[attr-defined]
    assets = types.ModuleType("rfdetr.assets")
    assets.__path__ = []  # type: ignore[attr-defined]
    coco = types.ModuleType("rfdetr.assets.coco_classes")
    coco.COCO_CLASSES = {1: "person"}  # type: ignore[attr-defined]
    supervision = types.ModuleType("supervision")
    monkeypatch.setitem(sys.modules, "rfdetr", rfdetr)
    monkeypatch.setitem(sys.modules, "rfdetr.assets", assets)
    monkeypatch.setitem(sys.modules, "rfdetr.assets.coco_classes", coco)
    monkeypatch.setitem(sys.modules, "supervision", supervision)

    provider = RoboflowDetector(
        api_key=None,
        model_id="rfdetr-small",
        cache_dir=tmp_path / "models",
    )

    tags = provider.detect(tmp_path / "frame.jpg")

    assert provider.status().state is ProviderState.READY
    assert [(tag.label, tag.confidence) for tag in tags] == [("person", 0.93)]
    assert tags[0].metadata == {"x": 30.0, "y": 50.0, "width": 40.0, "height": 60.0}


def test_lighthouse_prepares_windows_and_restores_global_timestamps(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")
    saved: dict[str, object] = {}

    class FakePredictor:
        def __init__(self, checkpoint_path: str, **kwargs) -> None:  # type: ignore[no-untyped-def]
            assert checkpoint_path == str(checkpoint)
            assert kwargs["feature_name"] == "clip"

        def encode_video(self, path: str):
            return {"path": path}

        def predict(self, query: str, _features):  # type: ignore[no-untyped-def]
            assert query == "player shoots"
            return {"pred_relevant_windows": [[1.0, 3.0, 0.9]]}

    lighthouse_package = types.ModuleType("lighthouse")
    lighthouse_package.__path__ = []  # type: ignore[attr-defined]
    lighthouse_models = types.ModuleType("lighthouse.models")
    lighthouse_models.QDDETRPredictor = FakePredictor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lighthouse", lighthouse_package)
    monkeypatch.setitem(sys.modules, "lighthouse.models", lighthouse_models)

    torch = types.ModuleType("torch")

    def save(payload, path) -> None:  # type: ignore[no-untyped-def]
        saved[str(path)] = payload
        Path(path).write_bytes(b"cache")

    def load(path, **_kwargs):  # type: ignore[no-untyped-def]
        return saved[str(path)]

    torch.save = save  # type: ignore[attr-defined]
    torch.load = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)

    class FakeFFmpeg:
        def export_clip(self, _source: Path, destination: Path, _start: float, _end: float) -> None:
            destination.write_bytes(b"window")

    retriever = LighthouseRetriever(
        checkpoint=checkpoint,
        cache_dir=tmp_path / "cache",
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
    )

    retriever.prepare("video-1", tmp_path / "source.mp4", 154.0)
    hits = retriever.search("player shoots", ["video-1"])

    assert retriever.status().state is ProviderState.READY
    assert len(saved) == 2
    assert [hit.start for hit in hits] == [1.0, 151.0]
    assert all(hit.modality == "lighthouse" for hit in hits)
