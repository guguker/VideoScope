import sys
import types
from collections import deque
from hashlib import sha256
import json
from pathlib import Path
import struct
import zlib

import pytest
import torch

from videoscope.providers.base import ProviderState
from videoscope.providers.lighthouse import LighthouseRetriever
from videoscope.providers.paddle_ocr import PaddleOCRReader, _result_payload
from videoscope.providers.roboflow import RoboflowDetector
from videoscope.providers.whisper import WhisperTranscriber


def _tiny_png() -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
        + chunk(b"IEND", b"")
    )


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
    snapshot = tmp_path / "whisper-snapshot"
    snapshot.mkdir()
    revision = "a" * 40
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda model, **kwargs: (
            str(snapshot)
            if model == "test-model"
            and kwargs == {"revision": revision, "local_files_only": True}
            else pytest.fail("unexpected snapshot request")
        ),
    )
    glossary = tmp_path / "glossary.json"
    glossary.write_text('{"Мозгов":["Mozgov"]}', encoding="utf-8")
    provider = WhisperTranscriber(
        "test-model",
        initial_prompt="Точная русская речь.",
        glossary_path=glossary,
        model_revision=revision,
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
    assert captured["path_or_hf_repo"] == str(snapshot)
    assert 0 < segments[0].confidence < 1


def test_whisper_drops_non_finite_segments_and_words(monkeypatch, tmp_path) -> None:
    module = types.ModuleType("mlx_whisper")
    module.transcribe = lambda *_args, **_kwargs: {  # type: ignore[attr-defined]
        "language": "en",
        "segments": [
            {"start": float("nan"), "end": 2.0, "text": "bad"},
            {"start": -1.0, "end": 2.0, "text": "bad"},
            {
                "start": 1.0,
                "end": 3.0,
                "text": "valid",
                "avg_logprob": -0.2,
                "words": [
                    {
                        "word": "bad",
                        "start": 1.0,
                        "end": float("inf"),
                        "probability": float("nan"),
                    }
                ],
            },
        ],
    }
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    model = tmp_path / "model"
    model.mkdir()
    provider = WhisperTranscriber(str(model))

    segments = provider.transcribe(tmp_path / "video.mp4")

    assert len(segments) == 1
    assert segments[0].start == 1.0
    assert segments[0].metadata.get("words") is None


def test_paddle_reader_parses_current_result_shape(monkeypatch, tmp_path) -> None:
    observed: list[object] = []

    class FakeResult:
        json = {"res": {"rec_texts": [" SCORE 87 ", "noise"], "rec_scores": [0.94, 0.1]}}

    class FakeModel:
        def predict(self, decoded: object):
            observed.append(decoded)
            return [FakeResult()]

    class FakePaddleOCR:
        def __init__(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def predict(self, decoded: object):
            return FakeModel().predict(decoded)

    module = types.ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    reader = PaddleOCRReader(minimum_confidence=0.35)
    image = tmp_path / "frame.png"
    image.write_bytes(_tiny_png())

    output = reader.read(image)

    assert reader.status().state is ProviderState.READY
    assert output == [("SCORE 87", 0.94)]
    assert observed and not isinstance(observed[0], (str, Path))
    assert _result_payload(types.SimpleNamespace(json=lambda: '{"res":{"rec_texts":[]}}')) == {
        "rec_texts": []
    }


def test_paddle_reader_uses_isolated_worker_protocol(monkeypatch, tmp_path) -> None:
    python = tmp_path / "python"
    worker = tmp_path / "worker.py"
    image = tmp_path / "frame.jpg"
    for path in (python, worker):
        path.write_bytes(b"fixture")
    image.write_bytes(_tiny_png())
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        "demo==1.0 \\\n    --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    dependency_hash = sha256(lock.read_bytes()).hexdigest()
    model_root = tmp_path / "models"
    model_directory = model_root / "detector"
    model_directory.mkdir(parents=True)
    model_artifact = model_directory / "model.bin"
    model_artifact.write_bytes(b"model")
    manifest_payload = {
        "engine": "transformers",
        "models": [
            {
                "artifacts": [
                    {
                        "name": model_artifact.name,
                        "sha256": sha256(model_artifact.read_bytes()).hexdigest(),
                        "size": model_artifact.stat().st_size,
                    }
                ],
                "directory": model_directory.name,
                "role": "detection",
            }
        ],
        "profile": "test",
        "schema_version": 1,
    }
    manifest = tmp_path / "models.lock.json"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    manifest_hash = sha256(
        json.dumps(
            manifest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    writes: list[bytes] = []

    reader = PaddleOCRReader(
        minimum_confidence=0.5,
        worker_python=python,
        worker_script=worker,
        worker_dependency_lock=lock,
        worker_model_manifest=manifest,
        worker_model_root=model_root,
        expected_dependency_identity=f"paddleocr-deps-v1:sha256:{dependency_hash}",
        expected_runtime_identity=(
            "videoscope-paddleocr-worker-v1|python==3.12.13|"
            "platform==aarch64-apple-darwin-macos14plus|"
            f"lock-sha256:{dependency_hash}"
        ),
        expected_model_identity=f"paddleocr-models-v1:sha256:{manifest_hash}",
        expected_script_sha256=sha256(worker.read_bytes()).hexdigest(),
    )
    attestation = reader.worker_attestation

    class FakeInput:
        def write(self, value: bytes) -> int:
            writes.append(value)
            request = json.loads(value)
            output.lines.append(
                json.dumps(
                    {
                        "attestation": request["attestation"],
                        "items": [[" SCORE 90 ", 0.96]],
                        "ok": True,
                        "request_id": request["request_id"],
                        "type": "result",
                    }
                ).encode("utf-8")
                + b"\n"
            )
            return len(value)

        def flush(self) -> None:
            pass

    class FakeOutput:
        def __init__(self) -> None:
            self.lines = deque(
                [
                    json.dumps(
                        {"attestation": attestation, "ok": True, "type": "hello"}
                    ).encode("utf-8")
                    + b"\n"
                ]
            )

        def readline(self, limit: int) -> bytes:
            return self.lines.popleft()[:limit] if self.lines else b""

    output = FakeOutput()

    class FakeProcess:
        stdin = FakeInput()
        stdout = output
        returncode = None

        def poll(self):  # type: ignore[no-untyped-def]
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def wait(self, timeout=None):  # type: ignore[no-untyped-def]
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(
        "videoscope.providers.paddle_ocr.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )

    assert reader.status().state is ProviderState.READY
    assert reader.read(image) == [("SCORE 90", 0.96)]
    request = json.loads(writes[0])
    assert "path" not in request
    assert request["frame"]["sha256"] == sha256(image.read_bytes()).hexdigest()
    assert request["frame"]["compressed_bytes"] == image.stat().st_size
    assert request["attestation"] == attestation


def test_roboflow_requires_explicit_configuration() -> None:
    provider = RoboflowDetector(api_key=None, model_id=None)

    assert provider.status().state is ProviderState.NEEDS_CONFIGURATION
    with pytest.raises(RuntimeError, match="not configured"):
        provider.detect(Path("frame.jpg"))


def test_roboflow_detector_uses_bounded_http_contract(monkeypatch, tmp_path) -> None:
    response = {
        "predictions": [
            {
                "class": "basketball",
                "confidence": 0.91,
                "x": 4,
                "y": 5,
                "width": 6,
                "height": 7,
            },
            {"class": "noise", "confidence": 0.1},
        ]
    }

    class FakeResponse:
        headers = {"content-length": "512"}
        content = json.dumps(response).encode("utf-8")

        def raise_for_status(self) -> None:
            pass

    class FakeClient:
        def post(self, url: str, **kwargs):  # type: ignore[no-untyped-def]
            assert url == "https://serverless.roboflow.com/basketball/1"
            assert kwargs["params"] == {"api_key": "secret"}
            assert kwargs["content"] == "aW1hZ2U="
            return FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "Client", lambda **_kwargs: FakeClient())
    provider = RoboflowDetector(api_key="secret", model_id="basketball/1")
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")

    tags = provider.detect(image)

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
            assert path.endswith(".mp4")
            return {
                "video_feats": torch.ones((1, 2, 4), dtype=torch.float32),
                "video_mask": torch.ones((1, 2), dtype=torch.float32),
                "audio_feats": None,
            }

        def predict(self, query: str, _features):  # type: ignore[no-untyped-def]
            assert query == "player shoots"
            return {"pred_relevant_windows": [[1.0, 3.0, 0.9]]}

    lighthouse_package = types.ModuleType("lighthouse")
    lighthouse_package.__path__ = []  # type: ignore[attr-defined]
    lighthouse_models = types.ModuleType("lighthouse.models")
    lighthouse_models.QDDETRPredictor = FakePredictor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lighthouse", lighthouse_package)
    monkeypatch.setitem(sys.modules, "lighthouse.models", lighthouse_models)

    def save(payload, path) -> None:  # type: ignore[no-untyped-def]
        saved[str(path)] = payload
        Path(path).write_bytes(b"cache")

    def load(path, **_kwargs):  # type: ignore[no-untyped-def]
        return saved[str(path)]

    monkeypatch.setattr(torch, "save", save)
    monkeypatch.setattr(torch, "load", load)

    class FakeFFmpeg:
        def export_clip(self, _source: Path, destination: Path, _start: float, _end: float) -> None:
            destination.write_bytes(b"window")

    retriever = LighthouseRetriever(
        checkpoint=checkpoint,
        checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
        cache_dir=tmp_path / "cache",
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
    )

    assert retriever.cache_is_current("video-1") is False
    retriever.prepare("video-1", tmp_path / "source.mp4", 154.0)
    hits = retriever.search("player shoots", ["video-1"])

    assert retriever.status().state is ProviderState.READY
    assert len(saved) == 2
    assert [hit.start for hit in hits] == [1.0, 151.0]
    assert all(hit.modality == "lighthouse" for hit in hits)

    manifest = tmp_path / "cache" / "lighthouse" / "video-1" / "manifest.json"
    assert json.loads(manifest.read_text(encoding="utf-8")) == retriever.cache_identity
    assert retriever.cache_is_current("video-1") is True
    manifest.write_text('{"schema_version":0}', encoding="utf-8")
    assert retriever.cache_is_current("video-1") is False
    assert retriever.search("player shoots", ["video-1"]) == []

    stale_window = manifest.parent / "window-9999.pt"
    stale_window.write_bytes(b"stale")
    retriever.prepare("video-1", tmp_path / "source.mp4", 3.0)
    assert not stale_window.exists()
    assert retriever.cache_is_current("video-1") is True
    for cache_path in manifest.parent.glob("window-*.pt"):
        cache_path.unlink()
    assert retriever.cache_is_current("video-1") is False


def test_lighthouse_readiness_rejects_corrupt_or_invalid_cache_windows(
    monkeypatch,
    tmp_path,
) -> None:
    class FakePredictor:
        pass

    class FakeFFmpeg:
        pass

    retriever = LighthouseRetriever(
        checkpoint=None,
        cache_dir=tmp_path / "cache",
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        retriever,
        "_predictor_class",
        lambda: (FakePredictor, "test predictor"),
    )
    target_dir = tmp_path / "cache" / "lighthouse" / "video-1"
    target_dir.mkdir(parents=True)
    manifest = target_dir / "manifest.json"
    manifest.write_text(json.dumps(retriever.cache_identity), encoding="utf-8")
    features = {
        "video_feats": torch.ones((1, 2, 4), dtype=torch.float32),
        "video_mask": torch.ones((1, 2), dtype=torch.float32),
        "audio_feats": None,
    }
    first = target_dir / "window-0000.pt"
    second = target_dir / "window-0001.pt"
    torch.save({"offset": 0.0, "end": 10.0, "features": features}, first)
    torch.save({"offset": 10.0, "end": 20.0, "features": features}, second)

    assert retriever.cache_is_current("video-1") is True

    second.write_bytes(b"truncated torch cache")
    assert retriever.cache_is_current("video-1") is False

    torch.save({"offset": 5.0, "end": 20.0, "features": features}, second)
    assert retriever.cache_is_current("video-1") is False

    invalid_features = {
        **features,
        "video_feats": torch.tensor([[[float("nan"), 1.0, 2.0, 3.0]]]),
        "video_mask": torch.ones((1, 1), dtype=torch.float32),
    }
    torch.save(
        {"offset": 10.0, "end": 20.0, "features": invalid_features},
        second,
    )
    assert retriever.cache_is_current("video-1") is False


def test_lighthouse_rejects_checkpoint_outside_sha256_allowlist(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"untrusted pickle")
    predictor_loaded = False

    class FakeFFmpeg:
        pass

    retriever = LighthouseRetriever(
        checkpoint=checkpoint,
        cache_dir=tmp_path / "cache",
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
    )

    def predictor_class():  # type: ignore[no-untyped-def]
        nonlocal predictor_loaded
        predictor_loaded = True
        raise AssertionError("untrusted checkpoint reached pickle loader")

    monkeypatch.setattr(retriever, "_predictor_class", predictor_class)

    assert retriever.status().state is ProviderState.NEEDS_CONFIGURATION
    with pytest.raises(RuntimeError, match="checksum"):
        retriever._load()
    assert predictor_loaded is False
