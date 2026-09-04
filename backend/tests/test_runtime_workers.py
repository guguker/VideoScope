from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from videoscope.config import AppSettings
from videoscope.model_manifest import fastembed_cache_snapshot_path
from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.providers.roboflow import RoboflowDetector
from videoscope.repository import Repository
from videoscope.providers.vision_worker_client import (
    VisionCapabilityStatus,
    VisionWorkerClient,
)
from videoscope.providers.whisper_worker import WhisperWorkerClient
import videoscope.runtime as runtime_module
from videoscope.search.service import (
    EvaluationSearchConfiguration,
    SearchAssetBinding,
)
from videoscope.search.vector_index import EmptyVectorIndex


class _FakeIndexingToolchain:
    identity = "sha256:" + "1" * 64

    def __init__(self, ffmpeg) -> None:  # type: ignore[no-untyped-def]
        self.ffmpeg = ffmpeg
        self.verify_calls = 0
        self.create_calls = 0

    def verify_current(self) -> str:
        self.verify_calls += 1
        return self.identity

    def create_ffmpeg(self):  # type: ignore[no-untyped-def]
        self.create_calls += 1
        return self.ffmpeg


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


def test_temporal_refiner_identity_projection_is_path_free_and_exact(
    tmp_path: Path,
) -> None:
    settings = _worker_settings(tmp_path)
    vision_client = runtime_module.create_vision_worker_client(settings)
    assert vision_client is not None
    visual = runtime_module.create_visual_index(
        settings,
        inference_client=vision_client,
    )

    scorer_identity, runtime_identity = (
        runtime_module._temporal_refiner_scorer_identities(visual, vision_client)
    )

    assert scorer_identity == (
        f"{visual.model_identity}#{visual.specification_identity}"
    )
    assert runtime_identity == (
        "vision-worker:" + str(vision_client.identity["siglip_specification_hash"])
    )
    assert str(tmp_path) not in scorer_identity
    assert str(tmp_path) not in runtime_identity


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

    attested_ffmpeg = runtime_module.FFmpeg()
    indexing_toolchain = _FakeIndexingToolchain(attested_ffmpeg)
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
    embedding_contract = runtime_module._reviewed_text_embedding_contract(settings)
    verified_model = settings.temp_dir / "verified-fastembed-fixture"
    verified_model.mkdir()
    text_embedding = runtime_module.SemanticEmbedding(
        model_name=embedding_contract.model_name,
        model_repository=embedding_contract.model_repository,
        model_revision=embedding_contract.model_revision,
        expected_runtime_version=embedding_contract.runtime_version,
        algorithm_version=embedding_contract.algorithm_version,
        dimensions=embedding_contract.dimensions,
        strict=True,
        specific_model_path=verified_model,
        model_content_sha256=embedding_contract.model_content_sha256,
        model_verifier=lambda: None,
    )
    monkeypatch.setattr(
        runtime_module,
        "create_reviewed_semantic_embedding_runtime",
        lambda **_kwargs: SimpleNamespace(
            embedding=text_embedding,
            close=lambda: True,
        ),
    )
    qdrant_embeddings: list[object] = []

    def qdrant_factory(*_args, **kwargs):  # type: ignore[no-untyped-def]
        qdrant_embeddings.append(kwargs["embedding"])
        return qdrant

    qdrant_factory.id = "qdrant"  # type: ignore[attr-defined]
    monkeypatch.setattr(runtime_module, "QdrantVectorIndex", qdrant_factory)
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

    runtime = runtime_module.build_runtime(
        settings,
        repository,
        indexing_toolchain=indexing_toolchain,  # type: ignore[arg-type]
    )

    assert runtime.queue.indexer.ffmpeg is attested_ffmpeg
    assert runtime.queue.indexer.speech is speech
    assert runtime.queue.indexer.objects is vision
    assert runtime.queue.indexer.visual_index is visual
    assert runtime.queue.indexer.vector_index is qdrant
    assert runtime.text_embedding_runtime.embedding is text_embedding
    assert text_embedding.strict_no_fallback is True
    assert qdrant_embeddings == [text_embedding]
    assert indexing_toolchain.verify_calls == 1
    assert indexing_toolchain.create_calls == 1

    expected_plan = object()
    captured_plan_inputs: list[tuple[object, object]] = []

    def plan_snapshot(candidate_settings, *, indexing_toolchain):  # type: ignore[no-untyped-def]
        captured_plan_inputs.append((candidate_settings, indexing_toolchain))
        return expected_plan

    monkeypatch.setattr(
        runtime_module,
        "create_video_index_plan_snapshot",
        plan_snapshot,
    )

    assert runtime.video_index_plan_factory() is expected_plan
    assert captured_plan_inputs == [(settings, indexing_toolchain)]
    assert indexing_toolchain.verify_calls == 1
    assert indexing_toolchain.create_calls == 1


@pytest.mark.parametrize("cache_state", ["missing", "corrupt"])
def test_build_runtime_starts_lexical_only_without_reviewed_fastembed_cache(
    tmp_path: Path,
    cache_state: str,
) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "clean-data",
        ocr_worker_python=tmp_path / "missing-ocr-python",
        ocr_worker_script=(
            Path(__file__).parents[2] / "scripts" / "paddle-ocr-worker.py"
        ),
    )
    settings.ensure_directories()
    repository = Repository(settings.database_path)
    repository.initialize()

    cache_snapshot = fastembed_cache_snapshot_path(
        settings.models_dir / "fastembed",
        settings.text_embedding_model,
    )
    assert cache_snapshot is not None
    if cache_state == "corrupt":
        cache_snapshot.mkdir(parents=True)
        (cache_snapshot / "config.json").write_text("{}", encoding="utf-8")
    else:
        assert list((settings.models_dir / "fastembed").glob("**/*")) == []

    runtime = runtime_module.build_runtime(
        settings,
        repository,
        indexing_toolchain=_FakeIndexingToolchain(runtime_module.FFmpeg()),  # type: ignore[arg-type]
    )
    runtime.start()
    try:
        assert runtime.text_embedding_runtime is None
        assert isinstance(runtime.queue.indexer.vector_index, EmptyVectorIndex)
        assert runtime.search.vector_index is runtime.queue.indexer.vector_index
        assert runtime.search.media_root == settings.media_dir
        assert runtime.search.media_access_root is None
        qdrant = runtime.generation_store
        assert qdrant.status().state is ProviderState.UNAVAILABLE
        assert qdrant.embedding.strict_no_fallback is True
        assert qdrant.embedding.backend == "unavailable"
        with pytest.raises(RuntimeError, match="reviewed semantic embedding"):
            qdrant.embedding.embed(["must never use a fallback"])

        content = b"runtime-pinned-media"
        source = settings.media_dir / "runtime-video.mp4"
        source.write_bytes(content)
        source_sha256 = sha256(content).hexdigest()
        repository.create_video_with_asset(
            video_id="runtime-video",
            original_name="runtime-video.mp4",
            stored_name=source.name,
            media_path=str(source),
            size_bytes=len(content),
            source_sha256=source_sha256,
        )
        repository.update_video(
            "runtime-video",
            status="ready",
            duration=1.0,
        )
        session = runtime.search.open_pinned_evaluation(
            EvaluationSearchConfiguration(
                modalities=("visual",),
                modality_weights=(("visual", 1.0),),
                text_search="disabled",
                visual_search="dense_siglip",
                temporal_refinement=False,
                lighthouse=False,
                reranker="none",
                reranker_trigger="disabled",
                reranker_candidate_limit=0,
                result_limit=20,
            ),
            (
                SearchAssetBinding(
                    external_id="runtime-video",
                    video_id="runtime-video",
                    source_sha256=source_sha256,
                    byte_size=len(content),
                    duration_seconds=1.0,
                ),
            ),
        )
        session.close()
        assert source.read_bytes() == content
    finally:
        assert runtime.close() is True
    assert list(settings.temp_dir.glob(".videoscope-fastembed-*")) == []
