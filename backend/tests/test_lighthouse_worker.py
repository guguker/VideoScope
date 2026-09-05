from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

import videoscope.providers.lighthouse_worker as lighthouse_worker_module
from videoscope.providers.base import ProviderState
from videoscope.providers.lighthouse_worker import (
    DEFAULT_LIGHTHOUSE_SPECIFICATION,
    LIGHTHOUSE_MODEL_IDENTITY,
    LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
    LIGHTHOUSE_WORKER_SCHEMA_VERSION,
    MAX_FEATURE_ARCHIVE_BYTES,
    MAX_RESPONSE_BYTES,
    LighthouseEncodedWindow,
    LighthouseGenerationBindingPayload,
    LighthouseGenerationStore,
    LighthouseHitPayload,
    LighthousePrepareRequest,
    LighthouseWorkerClient,
    LocalLighthouseWorkerRuntime,
    create_lighthouse_worker_app,
)


TOKEN = "l" * 32
SOURCE_SHA256 = "a" * 64


class FakeWorkerRuntime:
    def __init__(self) -> None:
        self.specification = DEFAULT_LIGHTHOUSE_SPECIFICATION
        self.model_identity = LIGHTHOUSE_MODEL_IDENTITY
        self.runtime_identity = LIGHTHOUSE_WORKER_RUNTIME_IDENTITY
        self.loaded = False
        self.is_available = True
        self.build_calls: list[tuple[object, Path]] = []
        self.activate_calls: list[tuple[str, str]] = []
        self.search_calls: list[object] = []
        self.error: Exception | None = None

    @property
    def available(self) -> bool:
        return self.is_available

    def build_generation(self, request: object, source: Path) -> str:
        self.build_calls.append((request, source))
        if self.error is not None:
            raise self.error
        return "b" * 32

    def activate_generation(self, video_id: str, generation_id: str) -> None:
        self.activate_calls.append((video_id, generation_id))

    def generation_descriptor(
        self,
        video_id: str,
        generation_id: str,
        *,
        force_content_validation: bool = False,
    ) -> dict[str, object] | None:
        del force_content_validation
        if not self.build_calls:
            return None
        request, _source = self.build_calls[-1]
        return {
            "duration_seconds": getattr(request, "duration_seconds"),
            "generation_id": generation_id,
            "manifest_sha256": "c" * 64,
            "source_sha256": getattr(request, "source_sha256"),
            "source_size_bytes": getattr(request, "source_size_bytes"),
            "specification_hash": self.specification.identity,
            "video_id": video_id,
        }

    def search(self, request: object) -> list[LighthouseHitPayload]:
        self.search_calls.append(request)
        if self.error is not None:
            raise self.error
        return [
            LighthouseHitPayload(
                video_id="video-1",
                generation_id="b" * 32,
                window_index=0,
                rank=0,
                start=1.25,
                end=3.5,
                score=0.91,
            )
        ]


def _client(
    tmp_path: Path,
    runtime: FakeWorkerRuntime,
    *,
    max_request_bytes: int = 64 * 1024,
) -> TestClient:
    app = create_lighthouse_worker_app(
        runtime=runtime,
        input_root=tmp_path / "media",
        api_key=TOKEN,
        max_input_bytes=4096,
        max_request_bytes=max_request_bytes,
    )
    return TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    )


def _prepare_request(source: Path) -> dict[str, object]:
    payload = source.read_bytes()
    return {
        "schema_version": LIGHTHOUSE_WORKER_SCHEMA_VERSION,
        "request_id": "c" * 32,
        "specification_hash": DEFAULT_LIGHTHOUSE_SPECIFICATION.identity,
        "model_identity": LIGHTHOUSE_MODEL_IDENTITY,
        "runtime_identity": LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
        "video_id": "video-1",
        "relative_path": source.name,
        "duration_seconds": 12.5,
        "source_sha256": sha256(payload).hexdigest(),
        "source_size_bytes": len(payload),
    }


def _search_request() -> dict[str, object]:
    return {
        "schema_version": LIGHTHOUSE_WORKER_SCHEMA_VERSION,
        "request_id": "d" * 32,
        "specification_hash": DEFAULT_LIGHTHOUSE_SPECIFICATION.identity,
        "model_identity": LIGHTHOUSE_MODEL_IDENTITY,
        "runtime_identity": LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
        "query": "player shoots",
        "video_ids": ["video-1"],
        "limit": 20,
    }


def test_worker_requires_bearer_loopback_and_trusted_host(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    runtime = FakeWorkerRuntime()
    app = create_lighthouse_worker_app(
        runtime=runtime,
        input_root=media,
        api_key=TOKEN,
    )
    local = TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    )
    remote = TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("192.0.2.10", 50000),
    )

    assert local.get("/v1/health").status_code == 401
    assert remote.get(
        "/v1/health", headers={"Authorization": f"Bearer {TOKEN}"}
    ).status_code == 403
    assert local.get(
        "/v1/health",
        headers={"Authorization": f"Bearer {TOKEN}", "Host": "attacker.example"},
    ).status_code == 400


def test_worker_bounds_body_before_json_parsing(tmp_path: Path) -> None:
    (tmp_path / "media").mkdir()
    runtime = FakeWorkerRuntime()
    client = _client(tmp_path, runtime, max_request_bytes=32)

    response = client.post(
        "/v1/prepare",
        content=b"{" + b"x" * 64 + b"}",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}
    assert runtime.build_calls == []


def test_health_exposes_exact_capability_without_loading_model(tmp_path: Path) -> None:
    (tmp_path / "media").mkdir()
    runtime = FakeWorkerRuntime()
    client = _client(tmp_path, runtime)

    response = client.get(
        "/v1/health", headers={"Authorization": f"Bearer {TOKEN}"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": LIGHTHOUSE_WORKER_SCHEMA_VERSION,
        "status": "ok",
        "model_identity": LIGHTHOUSE_MODEL_IDENTITY,
        "runtime_identity": LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
        "specification": DEFAULT_LIGHTHOUSE_SPECIFICATION.to_dict(),
        "specification_hash": DEFAULT_LIGHTHOUSE_SPECIFICATION.identity,
        "loaded": False,
        "operations": ["build", "prepare", "search"],
        "max_input_bytes": 4096,
        "max_query_chars": 500,
        "max_video_ids": 100,
        "max_hits": 100,
        "max_concurrency": 1,
    }

    runtime.is_available = False
    unavailable = client.get(
        "/v1/health", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert unavailable.json()["status"] == "unavailable"


def test_prepare_rejects_identity_mismatch_and_unsafe_input(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    source = media / "source.mp4"
    source.write_bytes(b"safe source")
    runtime = FakeWorkerRuntime()
    client = _client(tmp_path, runtime)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    mismatch = _prepare_request(source)
    mismatch["specification_hash"] = "e" * 64
    assert client.post("/v1/prepare", json=mismatch, headers=headers).status_code == 409

    traversal = _prepare_request(source)
    traversal["relative_path"] = "../source.mp4"
    assert client.post("/v1/prepare", json=traversal, headers=headers).status_code == 422

    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    (media / "link.mp4").symlink_to(outside)
    symlink = _prepare_request(source)
    symlink["relative_path"] = "link.mp4"
    symlink["source_sha256"] = sha256(outside.read_bytes()).hexdigest()
    symlink["source_size_bytes"] = outside.stat().st_size
    assert client.post("/v1/prepare", json=symlink, headers=headers).status_code == 400
    assert runtime.build_calls == []


def test_prepare_checks_source_identity_before_activation(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    source = media / "source.mp4"
    source.write_bytes(b"safe source")
    runtime = FakeWorkerRuntime()
    client = _client(tmp_path, runtime)
    request = _prepare_request(source)

    wrong_hash = dict(request, source_sha256="f" * 64)
    assert client.post(
        "/v1/prepare",
        json=wrong_hash,
        headers={"Authorization": f"Bearer {TOKEN}"},
    ).status_code == 409
    assert runtime.build_calls == []

    response = client.post(
        "/v1/prepare",
        json=request,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 200
    assert response.json()["generation_id"] == "b" * 32
    assert runtime.activate_calls == [("video-1", "b" * 32)]


def test_build_endpoint_returns_exact_descriptor_without_activation(
    tmp_path: Path,
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    source = media / "source.mp4"
    source.write_bytes(b"safe source")
    runtime = FakeWorkerRuntime()
    client = _client(tmp_path, runtime)

    response = client.post(
        "/v1/build",
        json=_prepare_request(source),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 200
    descriptor = response.json()["descriptor"]
    assert descriptor == runtime.generation_descriptor("video-1", "b" * 32)
    assert runtime.activate_calls == []


def test_prepare_does_not_activate_when_source_changes_during_build(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    source = media / "source.mp4"
    source.write_bytes(b"safe source")

    class MutatingRuntime(FakeWorkerRuntime):
        def build_generation(self, request: object, current_source: Path) -> str:
            generation = super().build_generation(request, current_source)
            current_source.write_bytes(b"changed source")
            return generation

    runtime = MutatingRuntime()
    client = _client(tmp_path, runtime)

    response = client.post(
        "/v1/prepare",
        json=_prepare_request(source),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Lighthouse source changed during preparation"}
    assert runtime.activate_calls == []


def test_worker_sanitizes_runtime_errors(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    source = media / "source.mp4"
    source.write_bytes(b"safe source")
    runtime = FakeWorkerRuntime()
    runtime.error = RuntimeError("secret checkpoint path /private/model.ckpt")
    client = _client(tmp_path, runtime)

    response = client.post(
        "/v1/prepare",
        json=_prepare_request(source),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Lighthouse worker preparation failed"}
    assert "private" not in response.text
    assert runtime.activate_calls == []


def test_search_contract_returns_bounded_strict_hits(tmp_path: Path) -> None:
    (tmp_path / "media").mkdir()
    runtime = FakeWorkerRuntime()
    client = _client(tmp_path, runtime)

    response = client.post(
        "/v1/search",
        json=_search_request(),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 200
    assert response.json()["hits"] == [
        {
            "video_id": "video-1",
            "generation_id": "b" * 32,
            "window_index": 0,
            "rank": 0,
            "start": 1.25,
            "end": 3.5,
            "score": 0.91,
        }
    ]


def _encoded_window(offset: float = 0.0, end: float = 4.0) -> LighthouseEncodedWindow:
    return LighthouseEncodedWindow(
        offset=offset,
        end=end,
        video_features=np.ones((1, 2, 514), dtype=np.float32),
        video_mask=np.ones((1, 2), dtype=np.float32),
    )


def test_generation_store_preserves_active_generation_on_failed_replacement(
    tmp_path: Path,
) -> None:
    store = LighthouseGenerationStore(
        tmp_path / "cache",
        DEFAULT_LIGHTHOUSE_SPECIFICATION,
    )
    first = store.build_generation(
        video_id="video-1",
        source_sha256=SOURCE_SHA256,
        source_size_bytes=10,
        duration_seconds=4.0,
        windows=[_encoded_window()],
    )
    store.activate("video-1", first)
    pointer = tmp_path / "cache" / "video-1" / "active.json"
    before = pointer.read_bytes()

    invalid = LighthouseEncodedWindow(
        offset=0.0,
        end=4.0,
        video_features=np.array([[[float("nan"), 1, 2, 3]]], dtype=np.float32),
        video_mask=np.ones((1, 1), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="finite"):
        store.build_generation(
            video_id="video-1",
            source_sha256="b" * 64,
            source_size_bytes=11,
            duration_seconds=4.0,
            windows=[invalid],
        )

    assert pointer.read_bytes() == before
    assert store.active_generation_id("video-1") == first
    assert store.cache_is_current("video-1") is True


def test_local_search_rejects_same_generation_id_with_different_content_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = LocalLighthouseWorkerRuntime(
        checkpoint=tmp_path / "checkpoint.ckpt",
        clip_checkpoint=tmp_path / "clip.pt",
        cache_root=tmp_path / "cache",
        work_root=tmp_path / "work",
    )
    generation_id = runtime.store.build_generation(
        video_id="video-1",
        source_sha256=SOURCE_SHA256,
        source_size_bytes=10,
        duration_seconds=4.0,
        windows=[_encoded_window()],
    )
    runtime.store.activate("video-1", generation_id)
    descriptor = runtime.store.generation_descriptor("video-1", generation_id)
    assert descriptor is not None
    descriptor["source_sha256"] = "b" * 64
    request = lighthouse_worker_module.LighthouseSearchRequest(
        schema_version=LIGHTHOUSE_WORKER_SCHEMA_VERSION,
        request_id="9" * 32,
        specification_hash=DEFAULT_LIGHTHOUSE_SPECIFICATION.identity,
        model_identity=LIGHTHOUSE_MODEL_IDENTITY,
        runtime_identity=LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
        query="player shoots",
        video_ids=["video-1"],
        generation_bindings=[
            LighthouseGenerationBindingPayload.model_validate(descriptor)
        ],
        limit=5,
    )
    monkeypatch.setattr(
        runtime,
        "_load",
        lambda: (_ for _ in ()).throw(AssertionError("model load reached")),
    )

    with pytest.raises(ValueError, match="stale or corrupt"):
        runtime.search(request)


@pytest.mark.parametrize("load_error", [None, OSError("transient read failure")])
@pytest.mark.parametrize("descriptor_cached", [True, False])
def test_local_generation_bound_search_fails_when_validated_generation_cannot_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_error: OSError | None,
    descriptor_cached: bool,
) -> None:
    runtime = LocalLighthouseWorkerRuntime(
        checkpoint=tmp_path / "checkpoint.ckpt",
        clip_checkpoint=tmp_path / "clip.pt",
        cache_root=tmp_path / "cache",
        work_root=tmp_path / "work",
    )
    generation_id = runtime.store.build_generation(
        video_id="video-1",
        source_sha256=SOURCE_SHA256,
        source_size_bytes=10,
        duration_seconds=4.0,
        windows=[_encoded_window()],
    )
    descriptor = runtime.store.generation_descriptor("video-1", generation_id)
    assert descriptor is not None
    if not descriptor_cached:
        runtime.store._generation_descriptor_cache.clear()
    request = lighthouse_worker_module.LighthouseSearchRequest(
        schema_version=LIGHTHOUSE_WORKER_SCHEMA_VERSION,
        request_id="8" * 32,
        specification_hash=DEFAULT_LIGHTHOUSE_SPECIFICATION.identity,
        model_identity=LIGHTHOUSE_MODEL_IDENTITY,
        runtime_identity=LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
        query="player shoots",
        video_ids=["video-1"],
        generation_bindings=[
            LighthouseGenerationBindingPayload.model_validate(descriptor)
        ],
        limit=5,
    )
    def fail_load(*_args):  # type: ignore[no-untyped-def]
        if load_error is not None:
            raise load_error
        return None

    monkeypatch.setattr(runtime.store, "load_generation", fail_load)
    monkeypatch.setattr(
        runtime,
        "_load",
        lambda: (_ for _ in ()).throw(AssertionError("model load reached")),
    )

    with pytest.raises(RuntimeError, match="bound Lighthouse generation is unavailable"):
        runtime.search(request)


def test_lighthouse_descriptor_reuses_content_validation_until_forced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LighthouseGenerationStore(
        tmp_path / "cache",
        DEFAULT_LIGHTHOUSE_SPECIFICATION,
    )
    generation_id = store.build_generation(
        video_id="video-1",
        source_sha256=SOURCE_SHA256,
        source_size_bytes=10,
        duration_seconds=4.0,
        windows=[_encoded_window()],
    )
    native_read = lighthouse_worker_module._read_bounded_regular_file_snapshot
    content_reads = 0

    def recording_read(path: Path, max_bytes: int):  # type: ignore[no-untyped-def]
        nonlocal content_reads
        content_reads += 1
        return native_read(path, max_bytes)

    monkeypatch.setattr(
        lighthouse_worker_module,
        "_read_bounded_regular_file_snapshot",
        recording_read,
    )

    first = store.generation_descriptor("video-1", generation_id)
    second = store.generation_descriptor("video-1", generation_id)
    final = store.generation_descriptor(
        "video-1",
        generation_id,
        force_content_validation=True,
    )

    assert first == second == final
    assert content_reads == 4


def test_generation_build_never_deletes_legacy_cache_before_activation(tmp_path: Path) -> None:
    video_dir = tmp_path / "cache" / "video-1"
    video_dir.mkdir(parents=True)
    legacy_manifest = video_dir / "manifest.json"
    legacy_window = video_dir / "window-0000.pt"
    legacy_manifest.write_text('{"schema_version":1}', encoding="utf-8")
    legacy_window.write_bytes(b"legacy torch cache")
    store = LighthouseGenerationStore(
        tmp_path / "cache",
        DEFAULT_LIGHTHOUSE_SPECIFICATION,
    )

    generation = store.build_generation(
        video_id="video-1",
        source_sha256=SOURCE_SHA256,
        source_size_bytes=10,
        duration_seconds=4.0,
        windows=[_encoded_window()],
    )

    assert store.active_generation_id("video-1") is None
    assert legacy_manifest.read_text(encoding="utf-8") == '{"schema_version":1}'
    assert legacy_window.read_bytes() == b"legacy torch cache"
    store.activate("video-1", generation)
    assert legacy_manifest.exists()
    assert legacy_window.exists()


def test_generation_store_rejects_corruption_and_activation_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LighthouseGenerationStore(
        tmp_path / "cache",
        DEFAULT_LIGHTHOUSE_SPECIFICATION,
    )
    first = store.build_generation(
        video_id="video-1",
        source_sha256=SOURCE_SHA256,
        source_size_bytes=10,
        duration_seconds=4.0,
        windows=[_encoded_window()],
    )
    store.activate("video-1", first)
    pointer = tmp_path / "cache" / "video-1" / "active.json"
    before = pointer.read_bytes()
    second = store.build_generation(
        video_id="video-1",
        source_sha256="b" * 64,
        source_size_bytes=11,
        duration_seconds=4.0,
        windows=[_encoded_window()],
    )

    def fail_activation(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(lighthouse_worker_module, "atomic_write_json", fail_activation)
    with pytest.raises(OSError, match="disk full"):
        store.activate("video-1", second)
    assert pointer.read_bytes() == before
    assert store.active_generation_id("video-1") == first

    generation = tmp_path / "cache" / "video-1" / "generations" / first
    (generation / "window-0000.npz").write_bytes(b"corrupt")
    assert store.cache_is_current("video-1") is False
    assert store.load_active("video-1") is None


def test_generation_store_rejects_symlinked_cache_ancestors(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    video_dir = cache / "video-1"
    video_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (video_dir / "generations").symlink_to(outside, target_is_directory=True)
    store = LighthouseGenerationStore(cache, DEFAULT_LIGHTHOUSE_SPECIFICATION)

    with pytest.raises(ValueError, match="unsafe"):
        store.build_generation(
            video_id="video-1",
            source_sha256=SOURCE_SHA256,
            source_size_bytes=10,
            duration_seconds=4.0,
            windows=[_encoded_window()],
        )

    assert list(outside.iterdir()) == []
    assert store.load_active("video-1") is None

    external_root = tmp_path / "external-root"
    external_root.mkdir()
    linked_root = tmp_path / "linked-cache"
    linked_root.symlink_to(external_root, target_is_directory=True)
    linked = LighthouseGenerationStore(linked_root, DEFAULT_LIGHTHOUSE_SPECIFICATION)
    with pytest.raises(ValueError, match="unsafe"):
        linked.build_generation(
            video_id="video-1",
            source_sha256=SOURCE_SHA256,
            source_size_bytes=10,
            duration_seconds=4.0,
            windows=[_encoded_window()],
        )
    assert list(external_root.iterdir()) == []


def test_generation_store_reads_named_generation_through_macos_file_id_path(
    tmp_path: Path,
) -> None:
    logical_root = tmp_path / "cache"
    writer = LighthouseGenerationStore(
        logical_root,
        DEFAULT_LIGHTHOUSE_SPECIFICATION,
    )
    generation_id = writer.build_generation(
        video_id="video-1",
        source_sha256=SOURCE_SHA256,
        source_size_bytes=10,
        duration_seconds=4.0,
        windows=[_encoded_window()],
    )
    expected = writer.generation_descriptor("video-1", generation_id)
    assert expected is not None
    metadata = tmp_path.stat()
    stable_root = Path(f"/.vol/{metadata.st_dev}/{metadata.st_ino}")
    if not stable_root.is_dir():
        pytest.skip("macOS file-id paths are unavailable")

    reader = LighthouseGenerationStore(
        stable_root / "cache",
        DEFAULT_LIGHTHOUSE_SPECIFICATION,
    )

    assert reader.generation_descriptor("video-1", generation_id) == expected


def test_generation_store_fsyncs_generation_and_activation_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced: list[Path] = []
    monkeypatch.setattr(
        lighthouse_worker_module,
        "_fsync_directory",
        lambda path: synced.append(Path(path)),
    )
    store = LighthouseGenerationStore(
        tmp_path / "cache",
        DEFAULT_LIGHTHOUSE_SPECIFICATION,
    )

    generation = store.build_generation(
        video_id="video-1",
        source_sha256=SOURCE_SHA256,
        source_size_bytes=10,
        duration_seconds=4.0,
        windows=[_encoded_window()],
    )
    store.activate("video-1", generation)

    video_dir = tmp_path / "cache" / "video-1"
    assert video_dir / "generations" in synced
    assert video_dir / "generations" / generation in synced
    assert video_dir in synced


def test_generation_store_rejects_oversized_or_compressed_feature_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LighthouseGenerationStore(
        tmp_path / "cache",
        DEFAULT_LIGHTHOUSE_SPECIFICATION,
    )
    generation_id = store.build_generation(
        video_id="video-1",
        source_sha256=SOURCE_SHA256,
        source_size_bytes=10,
        duration_seconds=4.0,
        windows=[_encoded_window()],
    )
    store.activate("video-1", generation_id)
    generation = (
        tmp_path / "cache" / "video-1" / "generations" / generation_id
    )
    archive = generation / "window-0000.npz"
    manifest_path = generation / "manifest.json"

    archive.write_bytes(b"x" * (MAX_FEATURE_ARCHIVE_BYTES + 1))
    monkeypatch.setattr(
        np,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("oversized archive reached NumPy loader")
        ),
    )
    assert store.load_active("video-1") is None

    with archive.open("wb") as handle:
        np.savez_compressed(
            handle,
            video_features=np.ones((1, 2, 514), dtype=np.float32),
            video_mask=np.ones((1, 2), dtype=np.float32),
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["windows"][0]["sha256"] = sha256(archive.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert store.load_active("video-1") is None


def test_local_runtime_rejects_symlinked_work_root_before_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    work_root = tmp_path / "work"
    work_root.symlink_to(outside, target_is_directory=True)
    runtime = LocalLighthouseWorkerRuntime(
        checkpoint=tmp_path / "checkpoint.ckpt",
        clip_checkpoint=tmp_path / "clip.pt",
        cache_root=tmp_path / "cache",
        work_root=work_root,
    )
    monkeypatch.setattr(
        runtime,
        "_load",
        lambda: (_ for _ in ()).throw(AssertionError("model load reached")),
    )
    request = LighthousePrepareRequest(
        schema_version=LIGHTHOUSE_WORKER_SCHEMA_VERSION,
        request_id="e" * 32,
        specification_hash=DEFAULT_LIGHTHOUSE_SPECIFICATION.identity,
        model_identity=LIGHTHOUSE_MODEL_IDENTITY,
        runtime_identity=LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
        video_id="video-1",
        relative_path="source.mp4",
        duration_seconds=4.0,
        source_sha256=SOURCE_SHA256,
        source_size_bytes=10,
    )

    with pytest.raises(ValueError, match="unsafe Lighthouse work"):
        runtime.build_generation(request, tmp_path / "source.mp4")

    assert list(outside.iterdir()) == []


def test_generation_loader_uses_one_bounded_archive_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LighthouseGenerationStore(
        tmp_path / "cache",
        DEFAULT_LIGHTHOUSE_SPECIFICATION,
    )
    generation_id = store.build_generation(
        video_id="video-1",
        source_sha256=SOURCE_SHA256,
        source_size_bytes=10,
        duration_seconds=4.0,
        windows=[_encoded_window()],
    )
    store.activate("video-1", generation_id)
    native_load = np.load
    snapshots: list[int] = []

    def bounded_load(source: object, **kwargs: object):  # type: ignore[no-untyped-def]
        getbuffer = getattr(source, "getbuffer", None)
        assert callable(getbuffer), "NumPy loader reopened a mutable cache path"
        snapshots.append(getbuffer().nbytes)
        return native_load(source, **kwargs)

    monkeypatch.setattr(np, "load", bounded_load)

    assert store.load_active("video-1") is not None
    assert snapshots and max(snapshots) <= MAX_FEATURE_ARCHIVE_BYTES


def test_local_runtime_keeps_only_one_transient_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        @staticmethod
        def encode_video(_source: str) -> dict[str, object]:
            return {
                "audio_feats": None,
                "video_feats": np.ones((1, 2, 514), dtype=np.float32),
                "video_mask": np.ones((1, 2), dtype=np.float32),
            }

    class RecordingFFmpeg:
        def __init__(self) -> None:
            self.preexisting_clips: list[int] = []

        def export_clip(
            self,
            _source: Path,
            destination: Path,
            _start: float,
            _end: float,
        ) -> None:
            self.preexisting_clips.append(len(list(destination.parent.glob("*.mp4"))))
            destination.write_bytes(b"bounded transient clip")

    ffmpeg = RecordingFFmpeg()
    runtime = LocalLighthouseWorkerRuntime(
        checkpoint=tmp_path / "checkpoint.ckpt",
        clip_checkpoint=tmp_path / "clip.pt",
        cache_root=tmp_path / "cache",
        work_root=tmp_path / "work",
        ffmpeg=ffmpeg,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(runtime, "_load", lambda: FakeModel())
    request = LighthousePrepareRequest(
        schema_version=LIGHTHOUSE_WORKER_SCHEMA_VERSION,
        request_id="f" * 32,
        specification_hash=DEFAULT_LIGHTHOUSE_SPECIFICATION.identity,
        model_identity=LIGHTHOUSE_MODEL_IDENTITY,
        runtime_identity=LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
        video_id="video-1",
        relative_path="source.mp4",
        duration_seconds=301.0,
        source_sha256=SOURCE_SHA256,
        source_size_bytes=10,
    )

    generation = runtime.build_generation(request, tmp_path / "source.mp4")

    assert len(generation) == 32
    assert ffmpeg.preexisting_clips == [0, 0, 0]
    assert list((tmp_path / "work").iterdir()) == []


def test_worker_environment_identity_checks_python_lock_and_core_versions(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_bytes(b"exact worker lock")
    lock_sha = sha256(lock.read_bytes()).hexdigest()
    expected_versions = {
        "fastapi": "0.141.1",
        "numpy": "1.23.5",
        "opencv-python": "4.11.0.86",
        "pydantic": "2.13.4",
        "pydantic-settings": "2.15.0",
        "torch": "2.11.0",
        "torchvision": "0.26.0",
        "uvicorn": "0.52.3",
    }

    assert lighthouse_worker_module._worker_environment_is_exact(
        python_version=(3, 11, 14),
        lock_path=lock,
        expected_lock_sha256=lock_sha,
        distribution_version=expected_versions.__getitem__,
    )
    assert not lighthouse_worker_module._worker_environment_is_exact(
        python_version=(3, 11, 15),
        lock_path=lock,
        expected_lock_sha256=lock_sha,
        distribution_version=expected_versions.__getitem__,
    )
    lock.write_bytes(b"tampered worker lock")
    assert not lighthouse_worker_module._worker_environment_is_exact(
        python_version=(3, 11, 14),
        lock_path=lock,
        expected_lock_sha256=lock_sha,
        distribution_version=expected_versions.__getitem__,
    )


class FakeHTTPResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class FakeHTTPClient:
    def __init__(self, health: dict[str, object], search: dict[str, object]) -> None:
        self.health = health
        self.search_payload = search
        self.posts: list[tuple[str, object]] = []
        self.gets = 0

    def get(self, *_args: object, **_kwargs: object) -> FakeHTTPResponse:
        self.gets += 1
        return FakeHTTPResponse(self.health)

    def post(self, url: str, *, json: object, **_kwargs: object) -> FakeHTTPResponse:
        self.posts.append((url, json))
        if url.endswith("/search"):
            return FakeHTTPResponse(self.search_payload)
        request_id = getattr(json, "get", lambda _key: None)("request_id")
        descriptor = None
        if url.endswith("/build") and isinstance(json, dict):
            descriptor = {
                "duration_seconds": json["duration_seconds"],
                "generation_id": "b" * 32,
                "manifest_sha256": "c" * 64,
                "source_sha256": json["source_sha256"],
                "source_size_bytes": json["source_size_bytes"],
                "specification_hash": DEFAULT_LIGHTHOUSE_SPECIFICATION.identity,
                "video_id": json["video_id"],
            }
        return FakeHTTPResponse(
            {
                "schema_version": LIGHTHOUSE_WORKER_SCHEMA_VERSION,
                "request_id": request_id,
                "specification_hash": DEFAULT_LIGHTHOUSE_SPECIFICATION.identity,
                "model_identity": LIGHTHOUSE_MODEL_IDENTITY,
                "runtime_identity": LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
                "video_id": "video-1",
                "generation_id": "b" * 32,
                **({"descriptor": descriptor} if descriptor is not None else {}),
            }
        )


def _health_payload() -> dict[str, object]:
    return {
        "schema_version": LIGHTHOUSE_WORKER_SCHEMA_VERSION,
        "status": "ok",
        "model_identity": LIGHTHOUSE_MODEL_IDENTITY,
        "runtime_identity": LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
        "specification": DEFAULT_LIGHTHOUSE_SPECIFICATION.to_dict(),
        "specification_hash": DEFAULT_LIGHTHOUSE_SPECIFICATION.identity,
        "loaded": False,
        "operations": ["build", "prepare", "search"],
        "max_input_bytes": 4096,
        "max_query_chars": 500,
        "max_video_ids": 100,
        "max_hits": 100,
        "max_concurrency": 1,
    }


def test_backend_client_preserves_retriever_behavior_and_caches_health(
    tmp_path: Path,
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    source = media / "source.mp4"
    source.write_bytes(b"source")
    cache = tmp_path / "cache"
    search_response = {
        "schema_version": LIGHTHOUSE_WORKER_SCHEMA_VERSION,
        "request_id": "placeholder",
        "specification_hash": DEFAULT_LIGHTHOUSE_SPECIFICATION.identity,
        "model_identity": LIGHTHOUSE_MODEL_IDENTITY,
        "runtime_identity": LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
        "hits": [
            {
                "video_id": "video-1",
                "generation_id": "b" * 32,
                "window_index": 2,
                "rank": 0,
                "start": 1.0,
                "end": 3.0,
                "score": 0.9,
            }
        ],
    }
    http = FakeHTTPClient(_health_payload(), search_response)
    client = LighthouseWorkerClient(
        endpoint="http://127.0.0.1:8782",
        api_key=TOKEN,
        input_root=media,
        cache_dir=cache,
        timeout=42,
        client=http,
    )

    assert client.status().state is ProviderState.READY
    assert client.status().state is ProviderState.READY
    assert http.gets == 1

    # Search remains the existing EvidenceHit-facing provider API.
    original_post = http.post

    def post_with_request_id(url: str, *, json: object, **kwargs: object) -> FakeHTTPResponse:
        if url.endswith("/search"):
            assert isinstance(json, dict)
            response = dict(search_response, request_id=json["request_id"])
            return FakeHTTPResponse(response)
        return original_post(url, json=json, **kwargs)

    http.post = post_with_request_id  # type: ignore[method-assign]
    hits = client.search("player shoots", ["video-1"], limit=12)
    client.prepare("video-1", source, 4.0)

    assert [(hit.start, hit.end, hit.score, hit.modality) for hit in hits] == [
        (1.0, 3.0, 0.9, "lighthouse")
    ]
    assert hits[0].segment_id == f"lighthouse:video-1:{'b' * 32}:0002:0000"
    assert client.timeout == 42
    assert client.identity["specification"] == DEFAULT_LIGHTHOUSE_SPECIFICATION.to_dict()


def test_backend_client_builds_without_legacy_activation(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    source = media / "source.mp4"
    source.write_bytes(b"source")
    http = FakeHTTPClient(_health_payload(), {})
    client = LighthouseWorkerClient(
        endpoint="http://127.0.0.1:8782",
        api_key=TOKEN,
        input_root=media,
        cache_dir=tmp_path / "cache",
        client=http,
    )

    descriptor = client.build_video_source("video-1", source, 4.0)

    assert descriptor["generation_id"] == "b" * 32
    assert descriptor["video_id"] == "video-1"
    assert [url for url, _payload in http.posts] == [
        "http://127.0.0.1:8782/v1/build"
    ]


def test_backend_client_rejects_hits_from_an_unbound_generation(
    tmp_path: Path,
) -> None:
    descriptor = {
        "duration_seconds": 4.0,
        "generation_id": "b" * 32,
        "manifest_sha256": "c" * 64,
        "source_sha256": SOURCE_SHA256,
        "source_size_bytes": 10,
        "specification_hash": DEFAULT_LIGHTHOUSE_SPECIFICATION.identity,
        "video_id": "video-1",
    }
    http = FakeHTTPClient(_health_payload(), {})
    client = LighthouseWorkerClient(
        endpoint="http://127.0.0.1:8782",
        api_key=TOKEN,
        input_root=tmp_path,
        cache_dir=tmp_path / "cache",
        client=http,
    )

    def post_with_wrong_generation(
        url: str,
        *,
        json: object,
        **_kwargs: object,
    ) -> FakeHTTPResponse:
        assert url.endswith("/search")
        assert isinstance(json, dict)
        bindings = json["generation_bindings"]
        assert isinstance(bindings, list)
        bindings_identity = sha256(
            json.dumps(
                bindings,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return FakeHTTPResponse(
            {
                "schema_version": LIGHTHOUSE_WORKER_SCHEMA_VERSION,
                "request_id": json["request_id"],
                "specification_hash": DEFAULT_LIGHTHOUSE_SPECIFICATION.identity,
                "model_identity": LIGHTHOUSE_MODEL_IDENTITY,
                "runtime_identity": LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
                "generation_bindings_sha256": bindings_identity,
                "hits": [
                    {
                        "video_id": "video-1",
                        "generation_id": "d" * 32,
                        "window_index": 0,
                        "rank": 0,
                        "start": 1.0,
                        "end": 3.0,
                        "score": 0.9,
                    }
                ],
            }
        )

    http.post = post_with_wrong_generation  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="worker search failed"):
        client.search_generations(
            "player shoots",
            {"video-1": descriptor},
            limit=12,
        )


def test_client_rejects_remote_endpoint_and_response_identity_mismatch(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="127.0.0.1"):
        LighthouseWorkerClient(
            endpoint="http://localhost:8782",
            api_key=TOKEN,
            input_root=tmp_path,
            cache_dir=tmp_path / "cache",
        )

    payload = _health_payload()
    payload["specification_hash"] = "f" * 64
    client = LighthouseWorkerClient(
        endpoint="http://127.0.0.1:8782",
        api_key=TOKEN,
        input_root=tmp_path,
        cache_dir=tmp_path / "cache",
        client=FakeHTTPClient(payload, {}),
    )
    assert client.status().state is ProviderState.UNAVAILABLE


class StreamingResponse:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.chunks = chunks
        self.headers = headers or {}
        self.iterated = False

    def __enter__(self) -> "StreamingResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self, *, chunk_size: int):  # type: ignore[no-untyped-def]
        assert chunk_size == 8192
        self.iterated = True
        yield from self.chunks


class StreamingHTTPClient:
    def __init__(self, response: StreamingResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self.closed = False

    def __enter__(self) -> "StreamingHTTPClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def stream(self, method: str, url: str, **kwargs: object) -> StreamingResponse:
        self.calls.append((method, url, kwargs))
        return self.response


def test_default_transport_ignores_proxy_environment_and_bounds_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    native = StreamingHTTPClient(
        StreamingResponse([json.dumps(_health_payload()).encode("utf-8")])
    )
    options: list[dict[str, object]] = []

    def build_client(**kwargs: object) -> StreamingHTTPClient:
        options.append(kwargs)
        return native

    monkeypatch.setattr(httpx, "Client", build_client)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("top-level HTTP client honors proxy environment")
        ),
    )
    client = LighthouseWorkerClient(
        endpoint="http://127.0.0.1:8782",
        api_key=TOKEN,
        input_root=tmp_path,
        cache_dir=tmp_path / "cache",
    )

    assert client.status().state is ProviderState.READY
    assert options == [{"trust_env": False, "follow_redirects": False}]
    assert native.closed is True

    oversized_native = StreamingHTTPClient(
        StreamingResponse([b"x" * (MAX_RESPONSE_BYTES + 1)])
    )
    monkeypatch.setattr(httpx, "Client", lambda **_kwargs: oversized_native)
    oversized = LighthouseWorkerClient(
        endpoint="http://127.0.0.1:8782",
        api_key=TOKEN,
        input_root=tmp_path,
        cache_dir=tmp_path / "cache-2",
    )
    assert oversized.status().state is ProviderState.UNAVAILABLE

    declared_response = StreamingResponse(
        [json.dumps(_health_payload()).encode("utf-8")],
        headers={"Content-Length": str(MAX_RESPONSE_BYTES + 1)},
    )
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **_kwargs: StreamingHTTPClient(declared_response),
    )
    declared = LighthouseWorkerClient(
        endpoint="http://127.0.0.1:8782",
        api_key=TOKEN,
        input_root=tmp_path,
        cache_dir=tmp_path / "cache-3",
    )
    assert declared.status().state is ProviderState.UNAVAILABLE
    assert declared_response.iterated is False


def test_specification_identity_is_canonical_and_complete() -> None:
    payload = DEFAULT_LIGHTHOUSE_SPECIFICATION.to_dict()

    assert payload["checkpoint_sha256"] == (
        "42798d352dde089a835cb4995eb9c8084a2e97337166abbbb09c12856aec2c55"
    )
    assert payload["clip_checkpoint_sha256"] == (
        "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
    )
    assert payload["lighthouse_revision"] == "d095eaa552cecef240897a8b750306b3b2a08740"
    assert payload["clip_revision"] == "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
    assert payload["runtime_identity"] == LIGHTHOUSE_WORKER_RUNTIME_IDENTITY
    assert payload["preprocessing"] == {
        "color": "rgb",
        "frame_sampling": "temporal-centers-by-checkpoint-clip-length",
        "image_size": 224,
        "max_frames": 75,
        "normalize": "openai-clip-vit-b-32",
        "resize": "short-edge-and-center-crop",
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert DEFAULT_LIGHTHOUSE_SPECIFICATION.identity == sha256(canonical.encode()).hexdigest()
