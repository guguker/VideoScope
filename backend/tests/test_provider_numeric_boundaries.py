import json
from pathlib import Path

import pytest
import torch

from videoscope.providers.lighthouse import LighthouseRetriever


@pytest.mark.parametrize(
    "window",
    [
        [1.0, 3.0, float("nan")],
        [float("nan"), 3.0, 0.9],
        [1.0, float("inf"), 0.9],
        [3.0, 1.0, 0.9],
    ],
)
def test_lighthouse_drops_non_finite_or_invalid_windows(
    monkeypatch,
    tmp_path: Path,
    window: list[float],
) -> None:
    cache_dir = tmp_path / "lighthouse"
    video_dir = cache_dir / "video-1"
    video_dir.mkdir(parents=True)
    cache_path = video_dir / "window-0000.pt"
    cache_path.write_bytes(b"cache")

    class FakeModel:
        @staticmethod
        def predict(_query: str, _features: object) -> dict[str, list[list[float]]]:
            return {"pred_relevant_windows": [window]}

    class FakeFFmpeg:
        pass

    retriever = LighthouseRetriever(
        checkpoint=None,
        cache_dir=tmp_path,
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(retriever, "_load", lambda: FakeModel())
    monkeypatch.setattr(
        retriever,
        "_predictor_class",
        lambda: (FakeModel, "test predictor"),
    )
    (video_dir / "manifest.json").write_text(
        json.dumps(retriever.cache_identity),
        encoding="utf-8",
    )
    features = {
        "video_feats": torch.ones((1, 2, 4), dtype=torch.float32),
        "video_mask": torch.ones((1, 2), dtype=torch.float32),
        "audio_feats": None,
    }

    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {
            "offset": 0.0,
            "end": 10.0,
            "features": features,
        },
    )

    assert retriever.search("query", ["video-1"]) == []
