from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace

import pytest

from videoscope.artifacts import StageKind
from videoscope.config import AppSettings
from videoscope.jobs import VideoIndexPlanSnapshot
from videoscope.repository import Repository
import videoscope.runtime as runtime_module


_EXECUTOR_IDENTITY_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OCR_DEPENDENCY_IDENTITY = "paddleocr-deps-v1:sha256:" + "d" * 64
_OCR_RUNTIME_IDENTITY = (
    "videoscope-paddleocr-worker-v1|python==3.12.13|"
    "platform==aarch64-apple-darwin-macos14plus|lock-sha256:" + "e" * 64
)
_OCR_MODEL_IDENTITY = "paddleocr-models-v1:sha256:" + "f" * 64
_TOOLCHAIN_IDENTITY = "sha256:" + "1" * 64


class _FakeIndexingToolchain:
    def __init__(self, identity: str = _TOOLCHAIN_IDENTITY) -> None:
        self.identity = identity
        self.verify_calls = 0

    def verify_current(self) -> str:
        self.verify_calls += 1
        return self.identity

    def create_ffmpeg(self):  # type: ignore[no-untyped-def]
        return runtime_module.FFmpeg()


@pytest.fixture(autouse=True)
def _reviewed_ocr_attestation(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_module,
        "OCR_WORKER_DEPENDENCY_IDENTITY",
        _OCR_DEPENDENCY_IDENTITY,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_module,
        "OCR_WORKER_RUNTIME_IDENTITY",
        _OCR_RUNTIME_IDENTITY,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_module,
        "OCR_MODEL_ARTIFACT_IDENTITY",
        _OCR_MODEL_IDENTITY,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_module,
        "attest_indexing_toolchain",
        lambda: _FakeIndexingToolchain(),
        raising=False,
    )


def _isolated_settings(
    tmp_path,
    name: str,
    *,
    vision_port: int = 8783,
    whisper_port: int = 8784,
    lighthouse_port: int = 8785,
    secret_character: str = "a",
    **changes: object,
) -> AppSettings:
    values: dict[str, object] = {
        "_env_file": None,
        "data_dir": tmp_path / name,
        "vision_worker_endpoint": f"http://127.0.0.1:{vision_port}",
        "vision_worker_api_key": secret_character * 32,
        "whisper_worker_endpoint": f"http://127.0.0.1:{whisper_port}",
        "whisper_worker_api_key": secret_character * 32,
        "lighthouse_endpoint": f"http://127.0.0.1:{lighthouse_port}",
        "lighthouse_api_key": secret_character * 32,
    }
    values.update(changes)
    return AppSettings(**values)  # type: ignore[arg-type]


def test_runtime_adopts_legacy_video_rows_in_bounded_batches(tmp_path) -> None:
    data_dir = tmp_path / "legacy-data"
    data_dir.mkdir()
    repository = Repository(data_dir / "videoscope.sqlite3")
    repository.initialize()
    for index in range(2):
        repository.create_video_with_asset(
            video_id=f"legacy-{index}",
            original_name=f"legacy-{index}.mp4",
            stored_name=f"legacy-{index}.mp4",
            media_path=str(data_dir / f"legacy-{index}.mp4"),
            size_bytes=16,
            source_sha256=str(index + 1) * 64,
        )

    plan = runtime_module.create_video_index_plan_snapshot(
        AppSettings(_env_file=None, data_dir=tmp_path / "plan-data")
    )
    plan_calls = 0

    def plan_factory() -> VideoIndexPlanSnapshot:
        nonlocal plan_calls
        plan_calls += 1
        return plan

    adopted = runtime_module.adopt_legacy_video_index_jobs(
        repository,
        plan_factory=plan_factory,
        batch_size=1,
    )

    assert len(adopted) == 2
    assert plan_calls == 2
    assert repository.list_legacy_video_index_candidates(limit=10) == ()
    assert {
        repository.get_video_index_job_plan(job_id).canonical_json
        for job_id in adopted
    } == {plan.canonical_json}
    assert runtime_module.adopt_legacy_video_index_jobs(
        repository,
        plan_factory=plan_factory,
        batch_size=1,
    ) == ()
    assert plan_calls == 2


def test_video_index_plan_factory_freezes_the_actual_whisper_prompt(tmp_path) -> None:
    settings = AppSettings(_env_file=None, data_dir=tmp_path / "data")
    settings.ensure_directories()
    settings.glossary_path.write_text(
        '{"Мозгов":["Mozgov"],"трёхочковый":["three pointer"]}',
        encoding="utf-8",
    )

    snapshot = runtime_module.create_video_index_plan_snapshot(settings)

    assert isinstance(snapshot, VideoIndexPlanSnapshot)
    assert snapshot.schema_version == 2
    assert "Мозгов" in (snapshot.whisper_prompt_snapshot.effective_prompt or "")
    assert snapshot.whisper_prompt_snapshot.effective_prompt_sha256 == hashlib.sha256(
        (snapshot.whisper_prompt_snapshot.effective_prompt or "").encode("utf-8")
    ).hexdigest()
    assert snapshot.specifications == runtime_module.create_indexing_run_plan(
        settings
    ).specifications
    assert snapshot.for_kind(StageKind.VISUAL_DENSE).kind is StageKind.VISUAL_DENSE
    assert snapshot.for_kind(StageKind.LIGHTHOUSE).kind is StageKind.LIGHTHOUSE
    assert VideoIndexPlanSnapshot.from_canonical_json(snapshot.canonical_json) == snapshot
    assert _EXECUTOR_IDENTITY_RE.fullmatch(snapshot.executor_identity)
    assert "Мозгов" in snapshot.canonical_json


def test_executor_identity_is_pathless_and_ignores_local_topology_and_secrets(
    tmp_path,
) -> None:
    first_secret = "s" * 32
    second_secret = "z" * 32
    first = _isolated_settings(
        tmp_path,
        "private-first",
        vision_port=8783,
        whisper_port=8784,
        lighthouse_port=8785,
        secret_character="s",
    )
    second = _isolated_settings(
        tmp_path,
        "private-second",
        vision_port=9783,
        whisper_port=9784,
        lighthouse_port=9785,
        secret_character="z",
    )

    first_snapshot = runtime_module.create_video_index_plan_snapshot(first)
    second_snapshot = runtime_module.create_video_index_plan_snapshot(second)

    assert first_snapshot.executor_identity == second_snapshot.executor_identity
    assert first_snapshot.plan_hash == second_snapshot.plan_hash
    for forbidden in (
        str(tmp_path),
        first_secret,
        second_secret,
        "127.0.0.1",
        "8783",
        "9783",
    ):
        assert forbidden not in first_snapshot.canonical_json
        assert forbidden not in second_snapshot.canonical_json


def test_executor_and_plan_identity_bind_the_injected_attested_toolchain(
    tmp_path,
) -> None:
    settings = AppSettings(_env_file=None, data_dir=tmp_path / "data")
    first = _FakeIndexingToolchain("sha256:" + "1" * 64)
    changed = _FakeIndexingToolchain("sha256:" + "2" * 64)

    first_identity = runtime_module.create_video_index_executor_identity(
        settings,
        indexing_toolchain=first,
    )
    changed_identity = runtime_module.create_video_index_executor_identity(
        settings,
        indexing_toolchain=changed,
    )
    plan = runtime_module.create_video_index_plan_snapshot(
        settings,
        indexing_toolchain=first,
    )

    assert first_identity != changed_identity
    assert plan.executor_identity == first_identity
    assert first.verify_calls == 2
    assert changed.verify_calls == 1
    assert first.identity not in plan.canonical_json


@pytest.mark.parametrize(
    "changes",
    [
        {"siglip_batch_size": 3},
        {"visual_index_step": 1.75},
        {"visual_index_max_width": 704},
        {"vision_worker_timeout": 121.0},
        {"whisper_worker_timeout": 601.0},
        {"lighthouse_timeout": 301.0},
    ],
)
def test_executor_identity_covers_output_and_provider_execution_settings(
    tmp_path,
    changes: dict[str, object],
) -> None:
    baseline = _isolated_settings(tmp_path, "baseline")
    changed = _isolated_settings(tmp_path, "changed", **changes)

    assert runtime_module.create_video_index_executor_identity(
        baseline
    ) != runtime_module.create_video_index_executor_identity(changed)


def test_executor_identity_covers_optional_provider_boundaries(tmp_path) -> None:
    disabled = AppSettings(_env_file=None, data_dir=tmp_path / "disabled")
    visual = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "visual",
        vision_worker_endpoint="http://127.0.0.1:8783",
        vision_worker_api_key="v" * 32,
    )
    lighthouse = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "lighthouse",
        lighthouse_endpoint="http://127.0.0.1:8785",
        lighthouse_api_key="l" * 32,
    )

    disabled_identity = runtime_module.create_video_index_executor_identity(disabled)

    assert disabled_identity != runtime_module.create_video_index_executor_identity(
        visual
    )
    assert disabled_identity != runtime_module.create_video_index_executor_identity(
        lighthouse
    )


def test_executor_identity_covers_pathless_ocr_script_content(tmp_path) -> None:
    first_script = tmp_path / "one" / "worker.py"
    second_script = tmp_path / "two" / "renamed.py"
    changed_script = tmp_path / "three" / "worker.py"
    for script in (first_script, second_script, changed_script):
        script.parent.mkdir()
    first_script.write_text("print('reviewed worker')\n", encoding="utf-8")
    second_script.write_text("print('reviewed worker')\n", encoding="utf-8")
    changed_script.write_text("print('changed worker')\n", encoding="utf-8")
    first = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "first-data",
        ocr_worker_python=tmp_path / "one" / "python",
        ocr_worker_script=first_script,
    )
    renamed = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "second-data",
        ocr_worker_python=tmp_path / "two" / "different-python",
        ocr_worker_script=second_script,
    )
    changed = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "third-data",
        ocr_worker_python=tmp_path / "three" / "python",
        ocr_worker_script=changed_script,
    )

    first_identity = runtime_module.create_video_index_executor_identity(first)

    assert first_identity == runtime_module.create_video_index_executor_identity(renamed)
    assert first_identity != runtime_module.create_video_index_executor_identity(changed)


def test_video_index_plan_fails_closed_without_exact_ocr_attestation(
    tmp_path,
    monkeypatch,
) -> None:
    settings = AppSettings(_env_file=None, data_dir=tmp_path / "data")
    monkeypatch.setattr(runtime_module, "OCR_WORKER_DEPENDENCY_IDENTITY", None)

    with pytest.raises(ValueError, match="reviewed OCR dependency/runtime"):
        runtime_module.create_video_index_plan_snapshot(settings)


def test_executor_identity_covers_the_reviewed_ocr_model_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    settings = AppSettings(_env_file=None, data_dir=tmp_path / "data")
    baseline = runtime_module.create_video_index_executor_identity(settings)
    monkeypatch.setattr(
        runtime_module,
        "OCR_MODEL_ARTIFACT_IDENTITY",
        "paddleocr-models-v1:sha256:" + "c" * 64,
    )

    assert runtime_module.create_video_index_executor_identity(settings) != baseline


def test_video_index_plan_rejects_unsafe_ocr_script_path(tmp_path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    script = real_parent / "worker.py"
    script.write_text("print('worker')\n", encoding="utf-8")
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        ocr_worker_script=linked_parent / "worker.py",
    )

    with pytest.raises(ValueError, match="reviewed OCR worker script"):
        runtime_module.create_video_index_executor_identity(settings)


def test_video_index_plan_rejects_oversized_ocr_script(tmp_path) -> None:
    script = tmp_path / "worker.py"
    script.write_bytes(b"x" * (runtime_module._MAX_OCR_WORKER_SCRIPT_BYTES + 1))
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        ocr_worker_script=script,
    )

    with pytest.raises(ValueError, match="reviewed OCR worker script"):
        runtime_module.create_video_index_executor_identity(settings)


def test_executor_identity_covers_reviewed_implementation_revisions(
    tmp_path,
    monkeypatch,
) -> None:
    settings = AppSettings(_env_file=None, data_dir=tmp_path / "data")
    baseline = runtime_module.create_video_index_executor_identity(settings)
    original_indexer_revision = runtime_module.VIDEO_INDEXER_IMPLEMENTATION_REVISION

    monkeypatch.setattr(
        runtime_module,
        "VIDEO_INDEXER_IMPLEMENTATION_REVISION",
        "videoscope.indexer.test-bump",
    )
    indexer_changed = runtime_module.create_video_index_executor_identity(settings)
    monkeypatch.setattr(
        runtime_module,
        "VIDEO_INDEXER_IMPLEMENTATION_REVISION",
        original_indexer_revision,
    )
    monkeypatch.setattr(
        runtime_module,
        "FFMPEG_INDEXING_CONTRACT_REVISION",
        "videoscope.ffmpeg-indexing.test-bump",
    )
    ffmpeg_changed = runtime_module.create_video_index_executor_identity(settings)
    monkeypatch.setattr(
        runtime_module,
        "FFMPEG_INDEXING_CONTRACT_REVISION",
        "videoscope.ffmpeg-indexing.v1",
    )
    monkeypatch.setattr(
        runtime_module,
        "TEXT_VECTOR_WRITER_IMPLEMENTATION_REVISION",
        "videoscope.qdrant-generation-writer.test-bump",
    )
    text_writer_changed = runtime_module.create_video_index_executor_identity(settings)

    assert indexer_changed != baseline
    assert ffmpeg_changed != baseline
    assert text_writer_changed != baseline


def test_executor_identity_does_not_probe_provider_health(
    tmp_path,
    monkeypatch,
) -> None:
    settings = _isolated_settings(tmp_path, "data")

    def unexpected_status(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("executor identity must not inspect provider health")

    monkeypatch.setattr(runtime_module.SiglipVisualIndex, "status", unexpected_status)
    monkeypatch.setattr(runtime_module.PaddleOCRReader, "status", unexpected_status)
    monkeypatch.setattr(
        runtime_module.LighthouseWorkerClient,
        "status",
        unexpected_status,
    )
    monkeypatch.setattr(
        runtime_module.SiglipVisualIndex,
        "identity",
        property(lambda _self: unexpected_status()),
    )
    monkeypatch.setattr(
        runtime_module.LighthouseWorkerClient,
        "identity",
        property(lambda _self: unexpected_status()),
    )

    identity = runtime_module.create_video_index_executor_identity(settings)

    assert _EXECUTOR_IDENTITY_RE.fullmatch(identity)


def test_video_index_plan_fails_closed_when_visual_factory_lacks_identity(
    tmp_path,
    monkeypatch,
) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        vision_worker_endpoint="http://127.0.0.1:8783",
        vision_worker_api_key="v" * 32,
    )
    monkeypatch.setattr(
        runtime_module,
        "create_visual_index",
        lambda *_args, **_kwargs: SimpleNamespace(
            batch_size=settings.siglip_batch_size,
            specification=None,
        ),
    )

    with pytest.raises(ValueError, match="reviewed visual provider"):
        runtime_module.create_video_index_executor_identity(settings)


def test_video_index_plan_fails_closed_when_visual_factory_is_unavailable(
    tmp_path,
    monkeypatch,
) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        vision_worker_endpoint="http://127.0.0.1:8783",
        vision_worker_api_key="v" * 32,
    )
    monkeypatch.setattr(
        runtime_module,
        "create_vision_worker_client",
        lambda _settings: None,
    )

    with pytest.raises(ValueError, match="reviewed visual provider"):
        runtime_module.create_video_index_executor_identity(settings)


def test_video_index_plan_binds_the_visual_client_timeout(
    tmp_path,
    monkeypatch,
) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        vision_worker_endpoint="http://127.0.0.1:8783",
        vision_worker_api_key="v" * 32,
        vision_worker_timeout=120.0,
    )
    client = runtime_module.create_vision_worker_client(settings)
    assert client is not None
    client.timeout = 121.0
    monkeypatch.setattr(
        runtime_module,
        "create_vision_worker_client",
        lambda _settings: client,
    )

    with pytest.raises(ValueError, match="reviewed visual provider"):
        runtime_module.create_video_index_executor_identity(settings)


def test_video_index_plan_fails_closed_when_lighthouse_factory_lacks_identity(
    tmp_path,
    monkeypatch,
) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        lighthouse_endpoint="http://127.0.0.1:8785",
        lighthouse_api_key="l" * 32,
    )
    provider = runtime_module.create_lighthouse_retriever(settings, runtime_module.FFmpeg())
    assert isinstance(provider, runtime_module.LighthouseWorkerClient)
    provider.specification = runtime_module.LighthouseSpecification(
        max_window_seconds=149.0,
    )
    monkeypatch.setattr(
        runtime_module,
        "create_lighthouse_retriever",
        lambda *_args, **_kwargs: provider,
    )

    with pytest.raises(ValueError, match="reviewed isolated Lighthouse"):
        runtime_module.create_video_index_executor_identity(settings)


def test_video_index_plan_binds_the_lighthouse_client_timeout(
    tmp_path,
    monkeypatch,
) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        lighthouse_endpoint="http://127.0.0.1:8785",
        lighthouse_api_key="l" * 32,
        lighthouse_timeout=300.0,
    )
    provider = runtime_module.create_lighthouse_retriever(
        settings,
        runtime_module.FFmpeg(),
    )
    assert isinstance(provider, runtime_module.LighthouseWorkerClient)
    provider.timeout = 301.0
    monkeypatch.setattr(
        runtime_module,
        "create_lighthouse_retriever",
        lambda *_args, **_kwargs: provider,
    )

    with pytest.raises(ValueError, match="reviewed isolated Lighthouse"):
        runtime_module.create_video_index_executor_identity(settings)


def test_video_index_plan_fails_closed_for_an_unknown_lighthouse_provider(
    tmp_path,
    monkeypatch,
) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        lighthouse_endpoint="http://127.0.0.1:8785",
        lighthouse_api_key="l" * 32,
    )
    monkeypatch.setattr(
        runtime_module,
        "create_lighthouse_retriever",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(ValueError, match="reviewed isolated Lighthouse"):
        runtime_module.create_video_index_executor_identity(settings)


def test_video_index_plan_factory_requires_validated_settings() -> None:
    with pytest.raises(TypeError, match="settings must be validated"):
        runtime_module.create_video_index_executor_identity(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="settings must be validated"):
        runtime_module.create_video_index_plan_snapshot(object())  # type: ignore[arg-type]


def test_video_index_plan_fails_closed_for_unreviewed_in_process_lighthouse(
    tmp_path,
) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        lighthouse_allow_in_process=True,
    )

    with pytest.raises(ValueError, match="reviewed isolated Lighthouse"):
        runtime_module.create_video_index_plan_snapshot(settings)


def test_video_index_plan_fails_closed_for_hosted_object_inference(tmp_path) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        roboflow_api_key="r" * 32,
        roboflow_model_id="basketball-players/1",
    )

    with pytest.raises(ValueError, match="attested isolated object provider"):
        runtime_module.create_video_index_plan_snapshot(settings)


def test_video_index_plan_fails_closed_for_unpinned_text_embedding(tmp_path) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        text_embedding_model="custom/unreviewed-embedding",
        text_embedding_dimensions=512,
    )

    with pytest.raises(ValueError, match="reviewed text embedding"):
        runtime_module.create_video_index_plan_snapshot(settings)


def test_persisted_text_vector_plan_binds_reviewed_model_bytes_and_concrete_type(
    tmp_path,
) -> None:
    from videoscope.benchmark.snapshots import (
        load_reviewed_fastembed_snapshot_manifest,
    )

    settings = AppSettings(_env_file=None, data_dir=tmp_path / "data")
    manifest = load_reviewed_fastembed_snapshot_manifest()

    plan = runtime_module.create_video_index_plan_snapshot(settings)
    text_vectors = plan.specifications.text_vectors

    assert text_vectors.dependencies["model_content_sha256"] == (
        manifest.model_content_sha256
    )
    assert text_vectors.dependencies["semantic_embedding_type"] == (
        "videoscope.search.embeddings.SemanticEmbedding"
    )
    assert manifest.model_content_sha256 in plan.canonical_json


def test_executor_projection_rejects_a_duck_typed_embedding_identity(
    monkeypatch,
    tmp_path,
) -> None:
    settings = AppSettings(_env_file=None, data_dir=tmp_path / "data")
    real = runtime_module.create_semantic_embedding(
        model_name=settings.text_embedding_model,
        dimensions=settings.text_embedding_dimensions,
        cache_dir=settings.models_dir / "fastembed",
    )
    forged = SimpleNamespace(
        identity=real.identity,
        model_repository=real.model_repository,
        model_revision=real.model_revision,
        algorithm_version=real.algorithm_version,
        dimensions=real.dimensions,
    )
    monkeypatch.setattr(
        runtime_module,
        "create_semantic_embedding",
        lambda **_kwargs: forged,
    )

    with pytest.raises(ValueError, match="concrete semantic embedding"):
        runtime_module.create_video_index_executor_identity(settings)


def test_executor_identity_changes_with_reviewed_fastembed_content_digest(
    monkeypatch,
    tmp_path,
) -> None:
    from videoscope.benchmark import snapshots as snapshots_module
    from videoscope.benchmark.snapshots import (
        FastEmbedFileManifestEntry,
        FastEmbedSnapshotManifest,
        load_reviewed_fastembed_snapshot_manifest,
    )

    settings = AppSettings(_env_file=None, data_dir=tmp_path / "data")
    baseline_manifest = load_reviewed_fastembed_snapshot_manifest()
    baseline = runtime_module.create_video_index_executor_identity(settings)
    first, *remaining = baseline_manifest.files
    changed_manifest = FastEmbedSnapshotManifest(
        model_name=baseline_manifest.model_name,
        model_repository=baseline_manifest.model_repository,
        model_revision=baseline_manifest.model_revision,
        runtime_version=baseline_manifest.runtime_version,
        algorithm_version=baseline_manifest.algorithm_version,
        dimensions=baseline_manifest.dimensions,
        files=(
            FastEmbedFileManifestEntry(
                relative_path=first.relative_path,
                size_bytes=first.size_bytes,
                sha256=("0" * 64 if first.sha256 != "0" * 64 else "1" * 64),
            ),
            *remaining,
        ),
    )
    monkeypatch.setattr(
        snapshots_module,
        "load_reviewed_fastembed_snapshot_manifest",
        lambda: changed_manifest,
    )

    changed = runtime_module.create_video_index_executor_identity(settings)
    projection = runtime_module._text_vector_executor_projection(settings)

    assert changed != baseline
    assert projection["strict_no_fallback"] is True
    assert projection["embedding"] == {
        **runtime_module._reviewed_text_embedding_contract(settings).canonical_dict,
    }
    assert projection["embedding"]["model_content_sha256"] == (
        changed_manifest.model_content_sha256
    )


def test_plan_rejects_fastembed_manifest_drift_between_stage_and_executor_snapshots(
    monkeypatch,
    tmp_path,
) -> None:
    from videoscope.benchmark import snapshots as snapshots_module
    from videoscope.benchmark.snapshots import (
        FastEmbedFileManifestEntry,
        FastEmbedSnapshotManifest,
        load_reviewed_fastembed_snapshot_manifest,
    )

    baseline = load_reviewed_fastembed_snapshot_manifest()
    first, *remaining = baseline.files
    changed = FastEmbedSnapshotManifest(
        model_name=baseline.model_name,
        model_repository=baseline.model_repository,
        model_revision=baseline.model_revision,
        runtime_version=baseline.runtime_version,
        algorithm_version=baseline.algorithm_version,
        dimensions=baseline.dimensions,
        files=(
            FastEmbedFileManifestEntry(
                relative_path=first.relative_path,
                size_bytes=first.size_bytes,
                sha256=("2" * 64 if first.sha256 != "2" * 64 else "3" * 64),
            ),
            *remaining,
        ),
    )
    manifests = iter((baseline, changed))
    monkeypatch.setattr(
        snapshots_module,
        "load_reviewed_fastembed_snapshot_manifest",
        lambda: next(manifests),
    )

    with pytest.raises(ValueError, match="text embedding changed"):
        runtime_module.create_video_index_plan_snapshot(
            AppSettings(_env_file=None, data_dir=tmp_path / "data")
        )
