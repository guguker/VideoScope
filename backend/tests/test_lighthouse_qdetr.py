from __future__ import annotations

import sys
import types
from pathlib import Path

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
    import lighthouse.common.qd_detr as qd_detr

    monkeypatch.setattr(qd_detr, "build_model", lambda _options: (model, None))
    options = types.SimpleNamespace(clip_length=2.0, device="old")
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {"opt": options, "model": {"weight": 1}},
    )

    instance = QDDETRPredictor(str(tmp_path / "model.ckpt"))
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
