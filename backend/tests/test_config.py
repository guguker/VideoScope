from pathlib import Path

import pytest
from pydantic import ValidationError

from videoscope.config import AppSettings
from videoscope.model_manifest import QWEN_VIDEO_MODEL, SIGLIP_224_MODEL, WHISPER_MODEL
from videoscope.providers.vision_worker_contract import (
    RFDETR_SMALL_CHECKPOINT_SHA256,
)


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
        ("visual_index_max_width", 8_193),
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


def test_settings_accept_the_maximum_worker_image_width() -> None:
    assert AppSettings(visual_index_max_width=8_192).visual_index_max_width == 8_192


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


def test_qwen_worker_requires_loopback_http_and_strong_api_key() -> None:
    key = "q" * 32
    with pytest.raises(ValidationError, match="qwen_video_api_key"):
        AppSettings(qwen_video_endpoint="http://127.0.0.1:8781")
    with pytest.raises(ValidationError, match="qwen_video_model"):
        AppSettings(
            qwen_video_endpoint="http://127.0.0.1:8781",
            qwen_video_api_key=key,
            qwen_video_model=None,
        )
    with pytest.raises(ValidationError, match="127.0.0.1"):
        AppSettings(
            qwen_video_endpoint="http://gpu.example:8781",
            qwen_video_api_key=key,
            qwen_video_model="organization/qwen",
        )
    with pytest.raises(ValidationError, match="plain loopback HTTP"):
        AppSettings(
            qwen_video_endpoint="https://127.0.0.1:8781",
            qwen_video_api_key=key,
            qwen_video_model="organization/qwen",
        )
    with pytest.raises(ValidationError, match="127.0.0.1"):
        AppSettings(
            qwen_video_endpoint="http://localhost:8781",
            qwen_video_api_key=key,
            qwen_video_model="organization/qwen",
        )

    settings = AppSettings(
        qwen_video_endpoint="http://127.0.0.1:8781",
        qwen_video_api_key=key,
        qwen_video_model=QWEN_VIDEO_MODEL,
    )

    assert settings.qwen_video_endpoint == "http://127.0.0.1:8781"
    assert key not in repr(settings)

    with pytest.raises(ValidationError, match="pinned"):
        AppSettings(
            qwen_video_endpoint="http://127.0.0.1:8781",
            qwen_video_api_key=key,
            qwen_video_model="organization/unpinned-qwen",
        )


def test_lighthouse_worker_requires_literal_loopback_and_strong_api_key() -> None:
    key = "l" * 32
    with pytest.raises(ValidationError, match="lighthouse_api_key"):
        AppSettings(lighthouse_endpoint="http://127.0.0.1:8782")
    with pytest.raises(ValidationError, match="127.0.0.1"):
        AppSettings(
            lighthouse_endpoint="http://localhost:8782",
            lighthouse_api_key=key,
        )
    with pytest.raises(ValidationError, match="plain loopback HTTP"):
        AppSettings(
            lighthouse_endpoint="https://127.0.0.1:8782",
            lighthouse_api_key=key,
        )

    settings = AppSettings(
        lighthouse_endpoint="http://127.0.0.1:8782",
        lighthouse_api_key=key,
    )

    assert settings.lighthouse_endpoint == "http://127.0.0.1:8782"
    assert key not in repr(settings)


def test_lighthouse_in_process_compatibility_is_explicit_and_unambiguous() -> None:
    assert AppSettings().lighthouse_allow_in_process is False
    with pytest.raises(ValidationError, match="cannot be combined"):
        AppSettings(
            lighthouse_endpoint="http://127.0.0.1:8782",
            lighthouse_api_key="l" * 32,
            lighthouse_allow_in_process=True,
        )


def test_qwen_in_process_compatibility_is_explicit_and_unambiguous() -> None:
    assert AppSettings().qwen_video_allow_in_process is False
    with pytest.raises(ValidationError, match="cannot be combined"):
        AppSettings(
            qwen_video_endpoint="http://127.0.0.1:8781",
            qwen_video_api_key="q" * 32,
            qwen_video_model=QWEN_VIDEO_MODEL,
            qwen_video_allow_in_process=True,
        )


def test_isolated_vision_and_whisper_workers_are_disabled_by_default() -> None:
    settings = AppSettings(_env_file=None)

    assert settings.vision_worker_endpoint is None
    assert settings.vision_worker_api_key is None
    assert settings.whisper_worker_endpoint is None
    assert settings.whisper_worker_api_key is None
    assert settings.roboflow_model_id is None
    assert settings.vision_detector_model_id == "rfdetr-small"
    assert (
        settings.vision_detector_checkpoint_sha256
        == RFDETR_SMALL_CHECKPOINT_SHA256
    )
    assert "vision_worker_allow_in_process" not in AppSettings.model_fields
    assert "whisper_worker_allow_in_process" not in AppSettings.model_fields


@pytest.mark.parametrize(
    ("field_prefix", "port"),
    [("vision_worker", 8783), ("whisper_worker", 8784)],
)
def test_isolated_ml_workers_require_literal_loopback_and_strong_paired_token(
    field_prefix: str,
    port: int,
) -> None:
    endpoint_field = f"{field_prefix}_endpoint"
    key_field = f"{field_prefix}_api_key"
    endpoint = f"http://127.0.0.1:{port}"
    key = field_prefix[0] * 32

    with pytest.raises(ValidationError, match=key_field):
        AppSettings(_env_file=None, **{endpoint_field: endpoint})
    with pytest.raises(ValidationError, match=endpoint_field):
        AppSettings(_env_file=None, **{key_field: key})
    with pytest.raises(ValidationError):
        AppSettings(
            _env_file=None,
            **{endpoint_field: endpoint, key_field: "short"},
        )
    for invalid in (
        f"http://localhost:{port}",
        f"http://127.0.0.2:{port}",
        f"https://127.0.0.1:{port}",
        f"http://127.0.0.1:{port}/v1",
        f"http://user@127.0.0.1:{port}",
        f"http://127.0.0.1:{port}?debug=1",
        f"http://127.0.0.1:{port}#fragment",
    ):
        with pytest.raises(ValidationError):
            AppSettings(
                _env_file=None,
                **{endpoint_field: invalid, key_field: key},
            )

    settings = AppSettings(
        _env_file=None,
        **{endpoint_field: endpoint, key_field: key},
    )

    assert getattr(settings, endpoint_field) == endpoint
    assert key not in repr(settings)


def test_worker_settings_accept_prefixed_and_unprefixed_environment_aliases(
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "VISION_WORKER_ENDPOINT=http://127.0.0.1:8783\n"
        f"VISION_WORKER_API_KEY={'v' * 32}\n"
        "VIDEOSCOPE_VISION_DETECTOR_MODEL_ID=rfdetr-medium\n"
        f"VIDEOSCOPE_VISION_DETECTOR_CHECKPOINT_SHA256={'a' * 64}\n"
        "VIDEOSCOPE_WHISPER_WORKER_ENDPOINT=http://127.0.0.1:8784\n"
        f"VIDEOSCOPE_WHISPER_WORKER_API_KEY={'w' * 32}\n",
        encoding="utf-8",
    )

    settings = AppSettings(_env_file=env_path)

    assert settings.vision_worker_endpoint == "http://127.0.0.1:8783"
    assert settings.vision_detector_model_id == "rfdetr-medium"
    assert settings.vision_detector_checkpoint_sha256 == "a" * 64
    assert settings.whisper_worker_endpoint == "http://127.0.0.1:8784"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"siglip_model": WHISPER_MODEL}, "reviewed SigLIP"),
        ({"whisper_model": QWEN_VIDEO_MODEL}, "reviewed Whisper"),
        ({"whisper_language": "RU"}, "whisper_language"),
        ({"whisper_initial_prompt": "x" * 16_001}, "whisper_initial_prompt"),
        ({"siglip_batch_size": 33}, "siglip_batch_size"),
    ],
)
def test_worker_configuration_rejects_values_outside_the_http_contract(
    overrides: dict[str, object],
    message: str,
) -> None:
    worker_settings = {
        "vision_worker_endpoint": "http://127.0.0.1:8783",
        "vision_worker_api_key": "v" * 32,
        "whisper_worker_endpoint": "http://127.0.0.1:8784",
        "whisper_worker_api_key": "w" * 32,
        **overrides,
    }

    with pytest.raises(ValidationError, match=message):
        AppSettings(_env_file=None, **worker_settings)


def test_worker_configuration_exposes_one_shared_detector_threshold() -> None:
    settings = AppSettings(
        _env_file=None,
        siglip_model=SIGLIP_224_MODEL,
        vision_worker_endpoint="http://127.0.0.1:8783",
        vision_worker_api_key="v" * 32,
        vision_worker_minimum_confidence=0.4,
    )

    assert settings.vision_worker_minimum_confidence == 0.4


def test_hosted_roboflow_is_fail_closed_and_mutually_exclusive_with_vision_worker() -> None:
    key_only = AppSettings(
        _env_file=None,
        roboflow_api_key="hosted-secret",
    )
    model_only = AppSettings(
        _env_file=None,
        roboflow_model_id="basketball/1",
    )
    assert key_only.roboflow_model_id is None
    assert model_only.roboflow_api_key is None
    with pytest.raises(ValidationError, match="project/version"):
        AppSettings(
            _env_file=None,
            roboflow_api_key="hosted-secret",
            roboflow_model_id="rfdetr-small",
        )
    with pytest.raises(ValidationError, match="cannot be combined"):
        AppSettings(
            _env_file=None,
            vision_worker_endpoint="http://127.0.0.1:8783",
            vision_worker_api_key="v" * 32,
            roboflow_api_key="hosted-secret",
            roboflow_model_id="basketball/1",
        )

    settings = AppSettings(
        _env_file=None,
        roboflow_api_key="hosted-secret",
        roboflow_model_id="basketball/1",
    )

    assert settings.roboflow_model_id == "basketball/1"
