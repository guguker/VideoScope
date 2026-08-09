from pathlib import Path

import pytest
from pydantic import ValidationError

from videoscope.config import AppSettings


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_upload_bytes", 0),
        ("port", 0),
        ("port", 65_536),
        ("host", "0.0.0.0"),
        ("host", "::1"),
        ("text_embedding_dimensions", 0),
        ("visual_index_step", 0),
        ("semantic_text_min_score", 1.1),
        ("qwen_video_frame_count", 0),
        ("internvideo_top_candidates", 5),
    ],
)
def test_settings_reject_invalid_runtime_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        AppSettings(**{field: value})


def test_settings_reject_inverted_qwen_clip_bounds() -> None:
    with pytest.raises(ValidationError):
        AppSettings(qwen_video_min_clip_seconds=12, qwen_video_max_clip_seconds=7)


def test_settings_accept_an_alternate_ipv4_loopback_and_port() -> None:
    settings = AppSettings(host="127.0.0.2", port=9876)

    assert settings.host == "127.0.0.2"
    assert settings.port == 9876


def test_default_text_embedding_rejects_an_incompatible_dimension() -> None:
    with pytest.raises(ValidationError, match="768"):
        AppSettings(text_embedding_dimensions=384)


def test_settings_model_contains_only_runtime_options() -> None:
    assert "search_limit" not in AppSettings.model_fields
    assert "siglip_quality_model" not in AppSettings.model_fields


def test_blank_optional_env_values_remain_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LIGHTHOUSE_ROOT", raising=False)
    monkeypatch.delenv("INTERNVIDEO_ENDPOINT", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("LIGHTHOUSE_ROOT=\nINTERNVIDEO_ENDPOINT=\n", encoding="utf-8")

    settings = AppSettings(_env_file=env_path)

    assert settings.lighthouse_root is None
    assert settings.internvideo_endpoint is None


def test_remote_internvideo_requires_https_and_strong_api_key() -> None:
    key = "x" * 32
    with pytest.raises(ValidationError, match="internvideo_api_key"):
        AppSettings(internvideo_endpoint="https://gpu.example/rerank")
    with pytest.raises(ValidationError, match="HTTPS"):
        AppSettings(
            internvideo_endpoint="http://gpu.example/rerank",
            internvideo_api_key=key,
        )

    settings = AppSettings(
        internvideo_endpoint="https://gpu.example/rerank",
        internvideo_api_key=key,
    )
    loopback = AppSettings(
        internvideo_endpoint="http://127.0.0.1:8780/rerank",
        internvideo_api_key=key,
    )

    assert settings.internvideo_endpoint == "https://gpu.example/rerank"
    assert loopback.internvideo_endpoint == "http://127.0.0.1:8780/rerank"
