from __future__ import annotations

import sys
import types
from pathlib import Path
from hashlib import sha256

import numpy as np
import pytest
import torch

from videoscope.providers.lighthouse_qdetr import QDDETRPredictor


class FakeClipModel:
    def __init__(self) -> None:
        self.visual = types.SimpleNamespace(
            conv1=types.SimpleNamespace(weight=torch.ones(1, dtype=torch.float32))
        )
        self.positional_embedding = torch.zeros(4, 3)

    def eval(self):  # type: ignore[no-untyped-def]
        return self

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.mean(dim=(-1, -2))

    def token_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.ones(tokens.shape[0], tokens.shape[1], 3)

    def transformer(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def ln_final(self, features: torch.Tensor) -> torch.Tensor:
        return features


class FakeQDModel:
    def __init__(self) -> None:
        self.loaded: object | None = None
        self.inputs: dict[str, object] = {}

    def load_state_dict(self, state: object) -> None:
        self.loaded = state

    def to(self, _device: str):  # type: ignore[no-untyped-def]
        return self

    def eval(self):  # type: ignore[no-untyped-def]
        return self

    def __call__(self, **inputs):  # type: ignore[no-untyped-def]
        self.inputs = inputs
        return {
            "pred_logits": torch.tensor([[[4.0, 0.0], [1.0, 2.0]]]),
            "pred_spans": torch.tensor([[[0.5, 0.4], [0.1, 0.5]]]),
            "saliency_scores": torch.tensor([[0.8, 0.2]]),
        }


@pytest.fixture
def predictor(monkeypatch, tmp_path: Path) -> QDDETRPredictor:
    fake_clip = types.ModuleType("clip")
    fake_clip.load = lambda *_args, **_kwargs: (FakeClipModel(), None)  # type: ignore[attr-defined]
    fake_clip.tokenize = lambda _queries: torch.tensor([[1, 2, 0, 0]])  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "clip", fake_clip)

    model = FakeQDModel()
    lighthouse = types.ModuleType("lighthouse")
    lighthouse.__path__ = []  # type: ignore[attr-defined]
    common = types.ModuleType("lighthouse.common")
    common.__path__ = []  # type: ignore[attr-defined]
    qd_detr = types.ModuleType("lighthouse.common.qd_detr")
    qd_detr.build_model = lambda _options: (model, None)  # type: ignore[attr-defined]
    lighthouse.common = common  # type: ignore[attr-defined]
    common.qd_detr = qd_detr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lighthouse", lighthouse)
    monkeypatch.setitem(sys.modules, "lighthouse.common", common)
    monkeypatch.setitem(sys.modules, "lighthouse.common.qd_detr", qd_detr)
    options = types.SimpleNamespace(clip_length=2.0, device="old")
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {"opt": options, "model": {"weight": 1}},
    )

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"test checkpoint")
    instance = QDDETRPredictor(
        str(checkpoint),
        checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
    )
    assert options.device == "cpu"
    assert model.loaded == {"weight": 1}
    return instance


def test_resize_and_center_crop_keeps_target_dimensions() -> None:
    frame = np.zeros((40, 80, 3), dtype=np.uint8)

    resized = QDDETRPredictor._resize_and_crop(frame, 24)

    assert resized.shape == (24, 24, 3)


def test_sample_frames_uses_temporal_centers(monkeypatch) -> None:
    captured_timestamps: list[float] = []

    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            return 5.0 if property_id == 1 else 1.0

        def set(self, _property_id: int, value: float) -> None:
            captured_timestamps.append(value)

        def read(self):  # type: ignore[no-untyped-def]
            return True, np.zeros((4, 6, 3), dtype=np.uint8)

        def release(self) -> None:
            pass

    cv2 = types.ModuleType("cv2")
    cv2.CAP_PROP_FRAME_COUNT = 1  # type: ignore[attr-defined]
    cv2.CAP_PROP_FPS = 2  # type: ignore[attr-defined]
    cv2.CAP_PROP_POS_MSEC = 3  # type: ignore[attr-defined]
    cv2.COLOR_BGR2RGB = 4  # type: ignore[attr-defined]
    cv2.VideoCapture = lambda _source: FakeCapture()  # type: ignore[attr-defined]
    cv2.cvtColor = lambda frame, _mode: frame  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr(QDDETRPredictor, "_resize_and_crop", staticmethod(lambda frame, _size: frame))

    instance = object.__new__(QDDETRPredictor)
    instance._torch = torch
    instance._clip_length = 2.0
    frames = instance._sample_frames("video.mp4")

    assert frames.shape == (3, 3, 4, 6)
    assert captured_timestamps == [1000.0, 3000.0, 4999.0]


def test_clip_encoding_video_features_and_prediction(predictor, monkeypatch) -> None:
    raw_frames = torch.full((2, 3, 8, 8), 127.0)
    image_features = predictor._encode_images(raw_frames)
    text_features, text_mask = predictor._encode_text("a jump shot")

    assert image_features.shape == (2, 3)
    assert text_features.shape == (1, 4, 3)
    assert text_mask.tolist() == [[1.0, 1.0, 0.0, 0.0]]

    monkeypatch.setattr(predictor, "_sample_frames", lambda _source: raw_frames)
    encoded = predictor.encode_video("video.mp4")
    assert encoded["video_feats"].shape == (1, 2, 5)
    assert encoded["video_mask"].tolist() == [[1.0, 1.0]]

    result = predictor.predict("a jump shot", encoded)
    assert result["pred_relevant_windows"][0] == [1.2, 2.8, pytest.approx(0.982, abs=0.001)]
    assert result["pred_relevant_windows"][1][0] == 0.0
    assert result["pred_saliency_scores"] == pytest.approx([0.8, 0.2])


def test_rejects_non_clip_lighthouse_features(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="feature_name='clip'"):
        QDDETRPredictor(str(tmp_path / "model.ckpt"), feature_name="clip_slowfast")


def test_rejects_checkpoint_before_torch_load(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"untrusted pickle")
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("pickle load")),
    )

    with pytest.raises(RuntimeError, match="checksum"):
        QDDETRPredictor(str(checkpoint))


def test_rejects_clip_weights_before_clip_load(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted test checkpoint")
    clip_checkpoint = tmp_path / "ViT-B-32.pt"
    clip_checkpoint.write_bytes(b"untrusted clip weights")
    fake_clip = types.ModuleType("clip")
    fake_clip.load = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[attr-defined]
        AssertionError("CLIP weights reached decoder")
    )
    monkeypatch.setitem(sys.modules, "clip", fake_clip)

    with pytest.raises(RuntimeError, match="CLIP checkpoint checksum"):
        QDDETRPredictor(
            str(checkpoint),
            checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
            clip_checkpoint_path=str(clip_checkpoint),
        )


def test_decoders_receive_verified_private_snapshots(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted qd detr")
    clip_checkpoint = tmp_path / "ViT-B-32.pt"
    clip_checkpoint.write_bytes(b"trusted clip")

    options = types.SimpleNamespace(clip_length=2.0, device="old")
    model = FakeQDModel()
    lighthouse = types.ModuleType("lighthouse")
    lighthouse.__path__ = []  # type: ignore[attr-defined]
    common = types.ModuleType("lighthouse.common")
    common.__path__ = []  # type: ignore[attr-defined]
    qd_detr = types.ModuleType("lighthouse.common.qd_detr")
    qd_detr.build_model = lambda _options: (model, None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lighthouse", lighthouse)
    monkeypatch.setitem(sys.modules, "lighthouse.common", common)
    monkeypatch.setitem(sys.modules, "lighthouse.common.qd_detr", qd_detr)

    def load_checkpoint(snapshot, **_kwargs):  # type: ignore[no-untyped-def]
        checkpoint.write_bytes(b"replacement pickle")
        assert Path(snapshot.name) != checkpoint
        assert snapshot.read() == b"trusted qd detr"
        return {"opt": options, "model": {"weight": 1}}

    monkeypatch.setattr(torch, "load", load_checkpoint)
    fake_clip = types.ModuleType("clip")

    def load_clip(snapshot_path, **_kwargs):  # type: ignore[no-untyped-def]
        clip_checkpoint.write_bytes(b"replacement clip")
        snapshot = Path(snapshot_path)
        assert snapshot != clip_checkpoint
        assert snapshot.read_bytes() == b"trusted clip"
        return FakeClipModel(), None

    fake_clip.load = load_clip  # type: ignore[attr-defined]
    fake_clip.tokenize = lambda _queries: torch.tensor([[1, 2, 0, 0]])  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "clip", fake_clip)

    predictor = QDDETRPredictor(
        str(checkpoint),
        checkpoint_sha256=sha256(b"trusted qd detr").hexdigest(),
        clip_checkpoint_path=str(clip_checkpoint),
        clip_checkpoint_sha256=sha256(b"trusted clip").hexdigest(),
    )

    assert predictor._clip_length == 2.0
    assert model.loaded == {"weight": 1}
