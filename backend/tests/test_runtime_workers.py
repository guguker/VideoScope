from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from videoscope.config import AppSettings
from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.providers.roboflow import RoboflowDetector
from videoscope.repository import Repository
from videoscope.providers.vision_worker_client import (
    VisionCapabilityStatus,
    VisionWorkerClient,
)
from videoscope.providers.whisper_worker import WhisperWorkerClient
import videoscope.runtime as runtime_module


def _worker_settings(tmp_path: Path) -> AppSettings:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        vision_worker_endpoint="http://127.0.0.1:8783",
        vision_worker_api_key="v" * 32,
        vision_worker_timeout=43,
        vision_worker_minimum_confidence=0.4,
        whisper_worker_endpoint="http://127.0.0.1:8784",
        whisper_worker_api_key="w" * 32,
        whisper_worker_timeout=47,
    )
    settings.ensure_directories()
    return settings


def test_runtime_worker_factories_fail_closed_without_endpoints(tmp_path: Path) -> None:
    settings = AppSettings(_env_file=None, data_dir=tmp_path / "data")
    settings.ensure_directories()

    vision_client = runtime_module.create_vision_worker_client(settings)
    visual = runtime_module.create_visual_index(
        settings,
        inference_client=vision_client,
    )

    assert vision_client is None
    assert visual.inference_client is None
    assert visual.status(check_index=False).state is ProviderState.NEEDS_CONFIGURATION
    assert runtime_module.create_object_detector(
        settings,
        vision_client=vision_client,
    ) is None
    assert runtime_module.create_whisper_transcriber(settings) is None


def test_runtime_attaches_one_shared_vision_client_and_isolated_whisper(
    tmp_path: Path,
) -> None:
    settings = _worker_settings(tmp_path)

    vision_client = runtime_module.create_vision_worker_client(settings)
    visual = runtime_module.create_visual_index(
        settings,
        inference_client=vision_client,
    )
    objects = runtime_module.create_object_detector(
        settings,
        vision_client=vision_client,
    )
    speech = runtime_module.create_whisper_transcriber(settings)

    assert isinstance(vision_client, VisionWorkerClient)
    assert vision_client.timeout == 43
    assert vision_client.input_root == settings.data_dir.resolve()
    assert vision_client.specification.minimum_confidence == 0.4
    assert visual.inference_client is vision_client
    assert objects is vision_client
    assert speech is not None
    assert isinstance(speech.inference_client, WhisperWorkerClient)
    assert speech.inference_client.timeout == 47
    assert speech.inference_client.input_root == settings.media_dir.resolve()
    assert vision_client.identity["siglip_specification_hash"] != (
        vision_client.identity["detector_specification_hash"]
    )


def test_configured_unavailable_workers_remain_attached_without_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _worker_settings(tmp_path)
    vision_client = runtime_module.create_vision_worker_client(settings)
    assert vision_client is not None
    monkeypatch.setattr(
        vision_client,
        "capability",
        lambda: VisionCapabilityStatus(False, "worker intentionally down"),
    )
    visual = runtime_module.create_visual_index(
        settings,
        inference_client=vision_client,
    )
    objects = runtime_module.create_object_detector(
        settings,
        vision_client=vision_client,
    )
    speech = runtime_module.create_whisper_transcriber(settings)
    assert speech is not None and speech.inference_client is not None
    monkeypatch.setattr(
        speech.inference_client,
        "status",
        lambda: type("Down", (), {"ready": False, "detail": "worker intentionally down"})(),
    )

    assert visual.inference_client is vision_client
    assert objects is vision_client
    assert visual.status(check_index=False).state is ProviderState.UNAVAILABLE
    assert speech.status().state is ProviderState.UNAVAILABLE


def test_hosted_roboflow_is_an_explicit_separate_provider(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        roboflow_api_key="hosted-secret",
        roboflow_model_id="basketball/1",
        vision_worker_minimum_confidence=0.4,
    )
    settings.ensure_directories()

    vision_client = runtime_module.create_vision_worker_client(settings)
    objects = runtime_module.create_object_detector(
        settings,
        vision_client=vision_client,
    )

    assert vision_client is None
    assert isinstance(objects, RoboflowDetector)
    assert objects.is_local is False
    assert objects.model_id == "basketball/1"
    assert objects.minimum_confidence == 0.4


def test_partial_hosted_roboflow_configuration_never_falls_back(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        roboflow_api_key="hosted-secret",
    )
    settings.ensure_directories()

    assert runtime_module.create_vision_worker_client(settings) is None
    assert runtime_module.create_object_detector(
        settings,
        vision_client=None,
    ) is None


def test_build_runtime_keeps_configured_down_workers_on_the_indexer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _worker_settings(tmp_path)
    repository = Repository(settings.database_path)
    repository.initialize()

    class FakeProvider:
        def __init__(self, provider_id: str, state: ProviderState) -> None:
            self.id = provider_id
            self.state = state

        def status(self, **_kwargs) -> ProviderStatus:
            return ProviderStatus(
                self.id,
                self.id,
                self.state,
                "fixture provider",
            )

    class FakeQdrant(FakeProvider):
        available = True
        supports_generation_provenance = False

        def close(self) -> None:
            pass

    vision = FakeProvider("vision-worker", ProviderState.UNAVAILABLE)
    speech = FakeProvider("whisper", ProviderState.UNAVAILABLE)
    visual = FakeProvider("siglip2", ProviderState.UNAVAILABLE)
    visual.inference_client = vision
    scenes = FakeProvider("scenes", ProviderState.READY)
    ocr = FakeProvider("paddleocr", ProviderState.NEEDS_CONFIGURATION)
    lighthouse = FakeProvider("lighthouse", ProviderState.NEEDS_CONFIGURATION)
    qdrant = FakeQdrant("qdrant", ProviderState.READY)
    qwen = FakeProvider("qwen-video", ProviderState.NEEDS_CONFIGURATION)
    internvideo = FakeProvider("internvideo", ProviderState.NEEDS_CONFIGURATION)

    monkeypatch.setattr(runtime_module, "FFmpeg", lambda: SimpleNamespace())
    monkeypatch.setattr(runtime_module, "SceneDetector", lambda **_kwargs: scenes)
    monkeypatch.setattr(runtime_module, "PaddleOCRReader", lambda **_kwargs: ocr)
    monkeypatch.setattr(
        runtime_module,
        "create_vision_worker_client",
        lambda _settings: vision,
    )
    monkeypatch.setattr(
        runtime_module,
        "create_visual_index",
        lambda _settings, *, inference_client: visual,
    )
    monkeypatch.setattr(
        runtime_module,
        "create_object_detector",
        lambda _settings, *, vision_client: vision_client,
    )
    monkeypatch.setattr(
        runtime_module,
        "create_whisper_transcriber",
        lambda _settings: speech,
    )
    monkeypatch.setattr(
        runtime_module,
        "create_lighthouse_retriever",
        lambda _settings, _ffmpeg: lighthouse,
    )
    monkeypatch.setattr(
        runtime_module,
        "create_semantic_embedding",
        lambda **_kwargs: SimpleNamespace(dimensions=2, identity="fixture-embedding"),
    )
    monkeypatch.setattr(
        runtime_module,
        "QdrantVectorIndex",
        lambda *_args, **_kwargs: qdrant,
    )
    monkeypatch.setattr(
        runtime_module,
        "create_qwen_reranker",
        lambda *_args, **_kwargs: qwen,
    )
    monkeypatch.setattr(
        runtime_module,
        "InternVideoReranker",
        lambda **_kwargs: internvideo,
    )
    monkeypatch.setattr(
        runtime_module,
        "SearchService",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        runtime_module,
        "ClipService",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    def unexpected_in_process(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("runtime attempted an in-process ML fallback")

    monkeypatch.setattr(runtime_module, "WhisperTranscriber", unexpected_in_process)
    monkeypatch.setattr(runtime_module, "RoboflowDetector", unexpected_in_process)

    runtime = runtime_module.build_runtime(settings, repository)

    assert runtime.queue.indexer.speech is speech
    assert runtime.queue.indexer.objects is vision
    assert runtime.queue.indexer.visual_index is visual
