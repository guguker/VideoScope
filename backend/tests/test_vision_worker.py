from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from hashlib import sha1, sha256
import json
import math
import os
from pathlib import Path
import sys
from threading import Event
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import videoscope.providers.vision_worker as vision_worker_module
import videoscope.providers.vision_worker_contract as vision_worker_contract
from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.providers.types import ObjectTag
from videoscope.providers.vision_worker import (
    LocalVisionWorkerRuntime,
    VisionWorkerSettings,
    create_vision_worker_app,
    vision_worker_environment_is_exact,
)
from videoscope.providers.vision_worker_client import VisionWorkerClient
from videoscope.providers.vision_worker_contract import (
    MAX_RESPONSE_BYTES,
    VISION_WORKER_RUNTIME_IDENTITY,
    VISION_WORKER_SCHEMA_VERSION,
    VisionEmbedImagesRequest,
    VisionWorkerSpecification,
    worker_input_root_identity,
)


TOKEN = "v" * 32
SIGLIP_MODEL = "google/siglip2-base-patch16-224"
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
RF_CHECKPOINT_SHA256 = "8" * 64


def _specification(*, dimensions: int = 4) -> VisionWorkerSpecification:
    return VisionWorkerSpecification(
        siglip_model=SIGLIP_MODEL,
        siglip_revision=SIGLIP_REVISION,
        embedding_dimensions=dimensions,
        detector_model_id="rfdetr-small",
        detector_checkpoint_sha256=RF_CHECKPOINT_SHA256,
        siglip_logit_scale=112.66890226978423,
        siglip_logit_bias=-16.771724700927734,
        minimum_confidence=0.25,
    )


class FakeVisionRuntime:
    def __init__(self, *, dimensions: int = 4) -> None:
        self.specification = _specification(dimensions=dimensions)
        self.siglip_loaded = False
        self.detector_loaded = False
        self.is_available = True
        self.image_calls: list[tuple[Path, ...]] = []
        self.text_calls: list[tuple[str, ...]] = []
        self.detect_calls: list[tuple[Path, float]] = []
        self.release_calls = 0
        self.block: Event | None = None
        self.error: Exception | None = None
        self.release_error: Exception | None = None

    @property
    def available(self) -> bool:
        return self.is_available

    def embed_images(self, sources: tuple[Path, ...]) -> list[list[float]]:
        self.image_calls.append(sources)
        if self.block is not None:
            self.block.wait(timeout=2)
        if self.error is not None:
            raise self.error
        return [
            [1.0, 0.0, 0.0, 0.0] if index == 0 else [0.0, 1.0, 0.0, 0.0]
            for index, _ in enumerate(sources)
        ]

    def embed_texts(self, texts: tuple[str, ...]) -> list[list[float]]:
        self.text_calls.append(texts)
        if self.error is not None:
            raise self.error
        return [
            [1.0, 0.0, 0.0, 0.0] if index == 0 else [0.0, 1.0, 0.0, 0.0]
            for index, _ in enumerate(texts)
        ]

    def detect(self, source: Path, minimum_confidence: float) -> list[dict[str, object]]:
        self.detect_calls.append((source, minimum_confidence))
        if self.error is not None:
            raise self.error
        return [
            {
                "label": "person",
                "confidence": 0.9,
                "x": 5.0,
                "y": 5.0,
                "width": 4.0,
                "height": 6.0,
            }
        ]

    def release_detector(self) -> None:
        self.release_calls += 1
        if self.release_error is not None:
            self.is_available = False
            raise self.release_error
        self.detector_loaded = False


def _write_png(path: Path, *, width: int = 10, height: int = 10) -> bytes:
    from PIL import Image

    Image.new("RGB", (width, height), color=(12, 34, 56)).save(path, format="PNG")
    return path.read_bytes()


def _identity(
    runtime: FakeVisionRuntime,
    input_root: Path | None = None,
) -> dict[str, str]:
    specification = runtime.specification
    return {
        "schema_version": VISION_WORKER_SCHEMA_VERSION,
        "runtime_identity": VISION_WORKER_RUNTIME_IDENTITY,
        "specification_hash": specification.identity,
        "siglip_specification_hash": specification.siglip_identity,
        "detector_specification_hash": specification.detector_identity,
        "siglip_model_identity": specification.siglip_model_identity,
        "detector_model_identity": specification.detector_model_identity,
        "input_root_identity": (
            worker_input_root_identity(input_root)
            if input_root is not None
            else "sha256:" + "f" * 64
        ),
    }


def _image_item(path: Path, *, item_id: str = "frame-1") -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "item_id": item_id,
        "relative_path": path.name,
        "expected_sha256": sha256(payload).hexdigest(),
        "expected_size_bytes": len(payload),
    }


def _images_request(
    runtime: FakeVisionRuntime,
    path: Path,
    *,
    input_root: Path | None = None,
) -> dict[str, object]:
    return {
        **_identity(runtime, input_root or path.parent),
        "request_id": "a" * 32,
        "items": [_image_item(path)],
    }


def _release_request(
    runtime: FakeVisionRuntime,
    input_root: Path,
    *,
    request_id: str = "f" * 32,
) -> dict[str, object]:
    return {
        **_identity(runtime, input_root),
        "request_id": request_id,
    }


def _client(tmp_path: Path, runtime: FakeVisionRuntime, **limits: int) -> TestClient:
    app = create_vision_worker_app(
        runtime=runtime,
        input_root=tmp_path,
        api_key=TOKEN,
        **limits,
    )
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50100))


def test_worker_requires_exact_host_authentication_and_loopback(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    app = create_vision_worker_app(runtime=runtime, input_root=tmp_path, api_key=TOKEN)
    local = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50100))
    remote = TestClient(app, base_url="http://127.0.0.1", client=("192.0.2.10", 50100))

    assert local.get("/v1/health").status_code == 401
    assert remote.get(
        "/v1/health", headers={"Authorization": f"Bearer {TOKEN}"}
    ).status_code == 403
    hostile = local.get(
        "/v1/health",
        headers={"Authorization": f"Bearer {TOKEN}", "Host": "attacker.example"},
    )
    assert hostile.status_code == 400


def test_worker_bounds_body_before_json_or_contract_parsing(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    client = _client(tmp_path, runtime, max_request_bytes=32)

    response = client.post(
        "/v1/embed/images",
        content=b"{" + b"x" * 80 + b"}",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}
    assert runtime.image_calls == []


def test_health_exposes_exact_bounded_capabilities_without_loading(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    client = _client(
        tmp_path,
        runtime,
        max_image_bytes=2048,
        max_image_dimension=512,
        max_image_pixels=100_000,
    )

    response = client.get("/v1/health", headers={"Authorization": f"Bearer {TOKEN}"})

    assert response.status_code == 200
    assert response.json() == {
        **_identity(runtime, tmp_path),
        "status": "ok",
        "siglip_loaded": False,
        "detector_loaded": False,
        "operations": [
            "probe",
            "embed_images",
            "embed_texts",
            "detect",
            "release_detector",
        ],
        "embedding_dimensions": 4,
        "max_images": 32,
        "max_texts": 64,
        "max_image_bytes": 2048,
        "max_image_dimension": 512,
        "max_image_pixels": 100_000,
        "max_batch_image_bytes": 128 * 1024 * 1024,
        "max_batch_image_pixels": 160_000_000,
        "max_detections": 500,
        "max_concurrency": 1,
    }

    runtime.is_available = False
    unavailable = client.get(
        "/v1/health", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert unavailable.json()["status"] == "unavailable"


def test_worker_releases_detector_only_for_an_exact_authenticated_request(
    tmp_path: Path,
) -> None:
    runtime = FakeVisionRuntime()
    runtime.siglip_loaded = True
    runtime.detector_loaded = True
    client = _client(tmp_path, runtime)
    request = _release_request(runtime, tmp_path)

    unauthenticated = client.post(
        "/v1/lifecycle/release-detector",
        json=request,
    )
    stale_request = {**request, "specification_hash": "0" * 64}
    stale = client.post(
        "/v1/lifecycle/release-detector",
        json=stale_request,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    released = client.post(
        "/v1/lifecycle/release-detector",
        json=request,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    released_again = client.post(
        "/v1/lifecycle/release-detector",
        json={**request, "request_id": "e" * 32},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert unauthenticated.status_code == 401
    assert stale.status_code == 409
    assert released.status_code == 200
    assert released.json() == {
        **_identity(runtime, tmp_path),
        "request_id": "f" * 32,
        "released": True,
        "siglip_loaded": True,
        "detector_loaded": False,
    }
    assert released_again.status_code == 200
    assert runtime.release_calls == 2
    assert runtime.siglip_loaded is True
    assert runtime.detector_loaded is False


def test_worker_release_failure_is_sanitized_and_latches_runtime_unavailable(
    tmp_path: Path,
) -> None:
    runtime = FakeVisionRuntime()
    runtime.detector_loaded = True
    runtime.release_error = RuntimeError("private cache failure")
    client = _client(tmp_path, runtime)

    response = client.post(
        "/v1/lifecycle/release-detector",
        json=_release_request(runtime, tmp_path),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    health = client.get(
        "/v1/health",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Vision worker detector release failed"}
    assert health.json()["status"] == "unavailable"


def test_worker_rejects_detector_release_while_inference_owns_capacity(
    tmp_path: Path,
) -> None:
    runtime = FakeVisionRuntime()
    gate = Event()
    runtime.block = gate
    path = tmp_path / "frame.png"
    _write_png(path)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    with _client(tmp_path, runtime) as client, ThreadPoolExecutor(max_workers=2) as pool:
        inference = pool.submit(
            client.post,
            "/v1/embed/images",
            json=_images_request(runtime, path),
            headers=headers,
        )
        deadline = time.monotonic() + 2
        while not runtime.image_calls and time.monotonic() < deadline:
            time.sleep(0.01)
        release = client.post(
            "/v1/lifecycle/release-detector",
            json=_release_request(runtime, tmp_path),
            headers=headers,
        )
        gate.set()
        assert inference.result(timeout=2).status_code == 200

    assert release.status_code == 429
    assert release.json() == {"detail": "Vision worker is busy"}
    assert runtime.release_calls == 0


def test_input_root_identity_is_path_free_and_changes_after_root_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "worker-input"
    root.mkdir()
    first = worker_input_root_identity(root)
    retained = tmp_path / "retained-input"
    root.rename(retained)
    root.mkdir()

    assert first.startswith("sha256:")
    assert str(root) not in first
    assert worker_input_root_identity(root) != first


def test_worker_probe_reads_a_bounded_source_under_the_attested_root(
    tmp_path: Path,
) -> None:
    runtime = FakeVisionRuntime()
    marker = tmp_path / "probe.bin"
    marker.write_bytes(b"videoscope-worker-root-probe-v1")
    payload = marker.read_bytes()
    response = _client(tmp_path, runtime).post(
        "/v1/probe",
        json={
            **_identity(runtime, tmp_path),
            "request_id": "e" * 32,
            "relative_path": marker.name,
            "expected_sha256": sha256(payload).hexdigest(),
            "expected_size_bytes": len(payload),
        },
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 200
    assert response.json() == {
        **_identity(runtime, tmp_path),
        "request_id": "e" * 32,
        "source_sha256": sha256(payload).hexdigest(),
        "source_size_bytes": len(payload),
    }


def test_worker_probe_rejects_a_stale_input_root_identity(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    marker = tmp_path / "probe.bin"
    marker.write_bytes(b"probe")
    request = {
        **_identity(runtime, tmp_path),
        "request_id": "e" * 32,
        "relative_path": marker.name,
        "expected_sha256": sha256(marker.read_bytes()).hexdigest(),
        "expected_size_bytes": marker.stat().st_size,
    }
    request["input_root_identity"] = "sha256:" + "0" * 64

    response = _client(tmp_path, runtime).post(
        "/v1/probe",
        json=request,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Vision worker identity does not match"}


def test_contract_forbids_extra_fields_duplicates_and_non_finite_values(
    tmp_path: Path,
) -> None:
    runtime = FakeVisionRuntime()
    path = tmp_path / "frame.png"
    _write_png(path)
    payload = _images_request(runtime, path)
    client = _client(tmp_path, runtime)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    extra = client.post(
        "/v1/embed/images", json={**payload, "unknown": True}, headers=headers
    )
    assert extra.status_code == 422
    assert extra.json() == {"detail": "Invalid Vision worker request"}
    duplicated = {**payload, "items": [payload["items"][0], payload["items"][0]]}
    duplicate_ids = client.post(
        "/v1/embed/images",
        json=duplicated,
        headers=headers,
    )
    assert duplicate_ids.status_code == 422
    assert duplicate_ids.json() == {"detail": "Invalid Vision worker request"}
    raw = json.dumps(payload).replace('"items":', '"nan": NaN, "items":', 1)
    invalid_numeric = client.post(
        "/v1/embed/images",
        content=raw,
        headers={**headers, "Content-Type": "application/json"},
    )
    assert invalid_numeric.status_code == 400
    assert invalid_numeric.json() == {"detail": "Invalid JSON body"}
    assert runtime.image_calls == []


@pytest.mark.parametrize(
    "relative_path",
    ["../frame.png", "/tmp/frame.png", "nested//frame.png", "frame.txt"],
)
def test_worker_rejects_unsafe_or_unsupported_paths(
    tmp_path: Path,
    relative_path: str,
) -> None:
    runtime = FakeVisionRuntime()
    path = tmp_path / "frame.png"
    payload = _write_png(path)
    request = _images_request(runtime, path)
    request["items"] = [
        {
            **request["items"][0],
            "relative_path": relative_path,
            "expected_sha256": sha256(payload).hexdigest(),
        }
    ]

    response = _client(tmp_path, runtime).post(
        "/v1/embed/images",
        json=request,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code in {400, 422}
    assert runtime.image_calls == []


def test_worker_rejects_symlinks_hash_mismatch_and_size_mismatch(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    real = tmp_path / "real.png"
    _write_png(real)
    link = tmp_path / "frame.png"
    link.symlink_to(real)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    client = _client(tmp_path, runtime)

    symlink_response = client.post(
        "/v1/embed/images", json=_images_request(runtime, link), headers=headers
    )
    assert symlink_response.status_code == 400

    link.unlink()
    _write_png(link)
    wrong_hash = _images_request(runtime, link)
    wrong_hash["items"][0]["expected_sha256"] = "0" * 64
    assert client.post("/v1/embed/images", json=wrong_hash, headers=headers).status_code == 409
    wrong_size = _images_request(runtime, link)
    wrong_size["items"][0]["expected_size_bytes"] += 1
    assert client.post("/v1/embed/images", json=wrong_size, headers=headers).status_code == 409
    assert runtime.image_calls == []


def test_worker_keeps_original_input_tree_after_ancestor_replacement(
    tmp_path: Path,
) -> None:
    runtime = FakeVisionRuntime()
    ancestor = tmp_path / "ancestor"
    input_root = ancestor / "input"
    input_root.mkdir(parents=True)
    trusted = input_root / "frame.png"
    _write_png(trusted, width=10, height=10)
    request = _images_request(runtime, trusted)
    app = create_vision_worker_app(
        runtime=runtime,
        input_root=input_root,
        api_key=TOKEN,
    )

    retained_ancestor = tmp_path / "retained-ancestor"
    ancestor.rename(retained_ancestor)
    replacement_root = ancestor / "input"
    replacement_root.mkdir(parents=True)
    _write_png(replacement_root / "frame.png", width=11, height=10)

    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50100),
    ) as client:
        response = client.post(
            "/v1/embed/images",
            json=request,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 200
    assert len(runtime.image_calls) == 1


def test_worker_closes_retained_input_root_descriptor_on_shutdown(
    tmp_path: Path,
) -> None:
    runtime = FakeVisionRuntime()
    app = create_vision_worker_app(
        runtime=runtime,
        input_root=tmp_path,
        api_key=TOKEN,
    )
    retained_root = app.state.vision_input_root
    descriptor = retained_root.descriptor

    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50100),
    ):
        os.fstat(descriptor)

    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_worker_rejects_oversized_or_overdimensioned_images(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    path = tmp_path / "frame.png"
    payload = _write_png(path, width=20, height=20)
    request = _images_request(runtime, path)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    too_many_bytes = _client(tmp_path, runtime, max_image_bytes=len(payload) - 1).post(
        "/v1/embed/images", json=request, headers=headers
    )
    assert too_many_bytes.status_code == 413
    too_wide = _client(tmp_path, runtime, max_image_dimension=19).post(
        "/v1/embed/images", json=request, headers=headers
    )
    assert too_wide.status_code == 413
    too_many_pixels = _client(tmp_path, runtime, max_image_pixels=399).post(
        "/v1/embed/images", json=request, headers=headers
    )
    assert too_many_pixels.status_code == 413
    assert runtime.image_calls == []


def test_worker_enforces_aggregate_batch_byte_and_pixel_limits(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    payload = _write_png(first, width=20, height=20)
    _write_png(second, width=20, height=20)
    request = _images_request(runtime, first)
    request["items"].append(_image_item(second, item_id="frame-2"))
    headers = {"Authorization": f"Bearer {TOKEN}"}

    bytes_response = _client(
        tmp_path,
        runtime,
        max_image_bytes=len(payload),
        max_batch_image_bytes=len(payload) * 2 - 1,
    ).post("/v1/embed/images", json=request, headers=headers)
    pixels_response = _client(
        tmp_path,
        runtime,
        max_image_pixels=400,
        max_batch_image_pixels=799,
    ).post("/v1/embed/images", json=request, headers=headers)

    assert bytes_response.status_code == 413
    assert pixels_response.status_code == 413
    assert runtime.image_calls == []


def test_worker_enforces_explicit_top_level_input_allowlist(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    forbidden = tmp_path / "media.png"
    _write_png(forbidden)
    allowed_dir = tmp_path / "visual-index"
    allowed_dir.mkdir()
    allowed = allowed_dir / "frame.png"
    _write_png(allowed)
    app = create_vision_worker_app(
        runtime=runtime,
        input_root=tmp_path,
        api_key=TOKEN,
        allowed_input_subdirectories=("visual-index", "thumbnails", "tmp"),
    )
    client = TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50100),
    )
    headers = {"Authorization": f"Bearer {TOKEN}"}

    rejected = client.post(
        "/v1/embed/images",
        json=_images_request(runtime, forbidden, input_root=tmp_path),
        headers=headers,
    )
    accepted_request = _images_request(runtime, allowed, input_root=tmp_path)
    accepted_request["items"][0]["relative_path"] = "visual-index/frame.png"
    accepted = client.post(
        "/v1/embed/images",
        json=accepted_request,
        headers=headers,
    )

    assert rejected.status_code == 400
    assert accepted.status_code == 200


def test_worker_enforces_nested_input_allowlist_without_exposing_product_media(
    tmp_path: Path,
) -> None:
    runtime = FakeVisionRuntime()
    allowed_dir = tmp_path / "product" / "tmp"
    media_dir = tmp_path / "product" / "media"
    collision_dir = tmp_path / "product" / "tmp-elsewhere"
    root_tmp_dir = tmp_path / "tmp"
    for directory in (allowed_dir, media_dir, collision_dir, root_tmp_dir):
        directory.mkdir(parents=True, exist_ok=True)
    allowed = allowed_dir / "frame.png"
    forbidden_media = media_dir / "frame.png"
    forbidden_collision = collision_dir / "frame.png"
    forbidden_root_tmp = root_tmp_dir / "frame.png"
    for source in (
        allowed,
        forbidden_media,
        forbidden_collision,
        forbidden_root_tmp,
    ):
        _write_png(source)
    app = create_vision_worker_app(
        runtime=runtime,
        input_root=tmp_path,
        api_key=TOKEN,
        allowed_input_subdirectories=(
            "product/visual-index",
            "product/thumbnails",
            "product/tmp",
        ),
    )
    client = TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50100),
    )
    headers = {"Authorization": f"Bearer {TOKEN}"}

    def request_for(source: Path) -> dict[str, object]:
        request = _images_request(runtime, source, input_root=tmp_path)
        request["items"][0]["relative_path"] = source.relative_to(  # type: ignore[index]
            tmp_path
        ).as_posix()
        return request

    accepted = client.post(
        "/v1/embed/images",
        json=request_for(allowed),
        headers=headers,
    )
    rejected = [
        client.post(
            "/v1/embed/images",
            json=request_for(source),
            headers=headers,
        )
        for source in (
            forbidden_media,
            forbidden_collision,
            forbidden_root_tmp,
        )
    ]

    assert accepted.status_code == 200
    assert [response.status_code for response in rejected] == [400, 400, 400]
    assert len(runtime.image_calls) == 1


def test_worker_checks_identity_before_reading_sources(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    missing = tmp_path / "missing.png"
    request = {
        **_identity(runtime, tmp_path),
        "request_id": "a" * 32,
        "items": [
            {
                "item_id": "frame-1",
                "relative_path": missing.name,
                "expected_sha256": "1" * 64,
                "expected_size_bytes": 1,
            }
        ],
    }
    request["specification_hash"] = "0" * 64

    response = _client(tmp_path, runtime).post(
        "/v1/embed/images",
        json=request,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Vision worker identity does not match"}


def test_image_and_text_embeddings_preserve_order_and_exact_shape(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_png(first)
    _write_png(second)
    client = _client(tmp_path, runtime)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    images = _images_request(runtime, first)
    images["items"].append(_image_item(second, item_id="frame-2"))

    image_response = client.post("/v1/embed/images", json=images, headers=headers)
    texts = {
        **_identity(runtime, tmp_path),
        "request_id": "b" * 32,
        "items": [
            {"item_id": "prompt-1", "text": "person"},
            {"item_id": "prompt-2", "text": "basketball"},
        ],
    }
    text_response = client.post("/v1/embed/texts", json=texts, headers=headers)

    assert image_response.status_code == 200
    assert [item["item_id"] for item in image_response.json()["items"]] == [
        "frame-1",
        "frame-2",
    ]
    assert all(len(item["vector"]) == 4 for item in image_response.json()["items"])
    assert text_response.status_code == 200
    assert [item["item_id"] for item in text_response.json()["items"]] == [
        "prompt-1",
        "prompt-2",
    ]
    assert runtime.text_calls == [("person", "basketball")]


@pytest.mark.parametrize(
    "bad_vectors",
    [
        [[1.0, 2.0]],
        [[1.0, 2.0, 3.0, float("nan")]],
        [[1.0, 1.0, 0.0, 0.0]],
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
    ],
)
def test_worker_sanitizes_invalid_runtime_vector_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_vectors: list[list[float]],
) -> None:
    runtime = FakeVisionRuntime()
    path = tmp_path / "frame.png"
    _write_png(path)
    monkeypatch.setattr(runtime, "embed_images", lambda _sources: bad_vectors)

    response = _client(tmp_path, runtime).post(
        "/v1/embed/images",
        json=_images_request(runtime, path),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Vision worker image embedding failed"}


def test_detection_is_bounded_and_validated_against_image_dimensions(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    path = tmp_path / "frame.png"
    _write_png(path, width=10, height=10)
    request = {
        **_identity(runtime, tmp_path),
        "request_id": "c" * 32,
        "source": _image_item(path),
        "minimum_confidence": 0.25,
    }

    response = _client(tmp_path, runtime).post(
        "/v1/detect",
        json=request,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 200
    assert response.json()["source_id"] == "frame-1"
    assert response.json()["image_width"] == 10
    assert response.json()["image_height"] == 10
    assert response.json()["detections"] == [
        {
            "label": "person",
            "confidence": 0.9,
            "x": 5.0,
            "y": 5.0,
            "width": 4.0,
            "height": 6.0,
        }
    ]

    runtime.detect = lambda *_args: [  # type: ignore[method-assign]
        {
            "label": "person",
            "confidence": 0.9,
            "x": 50.0,
            "y": 5.0,
            "width": 4.0,
            "height": 6.0,
        }
    ]
    invalid = _client(tmp_path, runtime).post(
        "/v1/detect",
        json=request,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert invalid.status_code == 503
    assert invalid.json() == {"detail": "Vision worker detection failed"}


def test_worker_detects_source_change_during_inference(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    path = tmp_path / "frame.png"
    _write_png(path)
    original = runtime.embed_images

    def mutate(sources: tuple[Path, ...]) -> list[list[float]]:
        vectors = original(sources)
        _write_png(path, width=11, height=10)
        return vectors

    runtime.embed_images = mutate  # type: ignore[method-assign]

    response = _client(tmp_path, runtime).post(
        "/v1/embed/images",
        json=_images_request(runtime, path),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Vision worker input changed during inference"}


def test_worker_rejects_concurrent_inference_without_queueing(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    gate = Event()
    runtime.block = gate
    path = tmp_path / "frame.png"
    _write_png(path)
    request = _images_request(runtime, path)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    with _client(tmp_path, runtime) as client, ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.post, "/v1/embed/images", json=request, headers=headers)
        deadline = time.monotonic() + 2
        while not runtime.image_calls and time.monotonic() < deadline:
            time.sleep(0.01)
        second = client.post("/v1/embed/images", json=request, headers=headers)
        gate.set()
        assert first.result(timeout=2).status_code == 200

    assert second.status_code == 429
    assert second.json() == {"detail": "Vision worker is busy"}


def test_global_capacity_rejects_cross_operation_contention(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    gate = Event()
    runtime.block = gate
    path = tmp_path / "frame.png"
    _write_png(path)
    image_request = _images_request(runtime, path)
    text_request = {
        **_identity(runtime, tmp_path),
        "request_id": "d" * 32,
        "items": [{"item_id": "prompt", "text": "person"}],
    }
    headers = {"Authorization": f"Bearer {TOKEN}"}

    with _client(tmp_path, runtime) as client, ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            client.post,
            "/v1/embed/images",
            json=image_request,
            headers=headers,
        )
        deadline = time.monotonic() + 2
        while not runtime.image_calls and time.monotonic() < deadline:
            time.sleep(0.01)
        competing = client.post(
            "/v1/embed/texts",
            json=text_request,
            headers=headers,
        )
        gate.set()
        assert first.result(timeout=2).status_code == 200

    assert competing.status_code == 429
    assert runtime.text_calls == []


class FakeHTTPResponse:
    def __init__(self, payload: object, *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
        return None

    def json(self) -> object:
        return self.payload


class FakeHTTPClient:
    def __init__(self, runtime: FakeVisionRuntime, input_root: Path) -> None:
        self.runtime = runtime
        self.input_root = input_root
        self.posts: list[tuple[str, object]] = []
        self.payload_override: object | None = None
        self.health_payload_override: object | None = None

    def get(self, url: str, **_kwargs: object) -> FakeHTTPResponse:
        if self.health_payload_override is not None:
            return FakeHTTPResponse(self.health_payload_override)
        specification = self.runtime.specification
        return FakeHTTPResponse(
            {
                **_identity(self.runtime, self.input_root),
                "status": "ok",
                "siglip_loaded": False,
                "detector_loaded": False,
                "operations": [
                    "probe",
                    "embed_images",
                    "embed_texts",
                    "detect",
                    "release_detector",
                ],
                "embedding_dimensions": specification.embedding_dimensions,
                "max_images": 32,
                "max_texts": 64,
                "max_image_bytes": 20 * 1024 * 1024,
                "max_image_dimension": 8192,
                "max_image_pixels": 40_000_000,
                "max_batch_image_bytes": 128 * 1024 * 1024,
                "max_batch_image_pixels": 160_000_000,
                "max_detections": 500,
                "max_concurrency": 1,
            }
        )

    def post(self, url: str, *, json: object, **_kwargs: object) -> FakeHTTPResponse:
        self.posts.append((url, json))
        if self.payload_override is not None:
            return FakeHTTPResponse(self.payload_override)
        request = json
        assert isinstance(request, dict)
        base = {
            **_identity(self.runtime, self.input_root),
            "request_id": request["request_id"],
        }
        if url.endswith("/v1/probe"):
            return FakeHTTPResponse(
                {
                    **base,
                    "source_sha256": request["expected_sha256"],
                    "source_size_bytes": request["expected_size_bytes"],
                }
            )
        if url.endswith("/v1/embed/images") or url.endswith("/v1/embed/texts"):
            return FakeHTTPResponse(
                {
                    **base,
                    "embedding_dimensions": 4,
                    "items": [
                        {
                            "item_id": item["item_id"],
                            "vector": [1.0, 0.0, 0.0, 0.0],
                        }
                        for item in request["items"]
                    ],
                }
            )
        if url.endswith("/v1/lifecycle/release-detector"):
            return FakeHTTPResponse(
                {
                    **base,
                    "released": True,
                    "siglip_loaded": False,
                    "detector_loaded": False,
                }
            )
        return FakeHTTPResponse(
            {
                **base,
                "source_id": request["source"]["item_id"],
                "image_width": 10,
                "image_height": 10,
                "detections": [
                    {
                        "label": "person",
                        "confidence": 0.9,
                        "x": 5.0,
                        "y": 5.0,
                        "width": 4.0,
                        "height": 6.0,
                    }
                ],
            }
        )


def _worker_adapter(tmp_path: Path, runtime: FakeVisionRuntime) -> tuple[VisionWorkerClient, FakeHTTPClient]:
    transport = FakeHTTPClient(runtime, tmp_path)
    adapter = VisionWorkerClient(
        endpoint="http://127.0.0.1:8093",
        api_key=TOKEN,
        input_root=tmp_path,
        specification=runtime.specification,
        client=transport,
    )
    return adapter, transport


def test_client_preserves_visual_encoder_and_object_provider_shapes(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_png(first)
    _write_png(second)
    adapter, transport = _worker_adapter(tmp_path, runtime)

    image_vectors = adapter.image_vectors([first, second])
    text_vectors = adapter.text_vectors(["person", "ball"])
    detections = adapter.detect(first)

    assert image_vectors.shape == (2, 4)
    assert text_vectors.shape == (2, 4)
    assert detections == [
        ObjectTag(
            label="person",
            confidence=0.9,
            metadata={"x": 5.0, "y": 5.0, "width": 4.0, "height": 6.0},
        )
    ]
    image_payload = transport.posts[0][1]
    assert isinstance(image_payload, dict)
    assert image_payload["items"][0]["expected_sha256"] == sha256(
        first.read_bytes()
    ).hexdigest()
    assert image_payload["items"][0]["expected_size_bytes"] == first.stat().st_size


def test_client_releases_ingestion_resources_with_exact_worker_identity(
    tmp_path: Path,
) -> None:
    runtime = FakeVisionRuntime()
    adapter, transport = _worker_adapter(tmp_path, runtime)

    result = adapter.release_ingestion_resources()

    assert result is None
    url, payload = transport.posts[-1]
    assert url.endswith("/v1/lifecycle/release-detector")
    assert isinstance(payload, dict)
    assert payload.keys() == {
        "schema_version",
        "runtime_identity",
        "specification_hash",
        "siglip_specification_hash",
        "detector_specification_hash",
        "siglip_model_identity",
        "detector_model_identity",
        "input_root_identity",
        "request_id",
    }
    assert payload["input_root_identity"] == worker_input_root_identity(tmp_path)


def test_client_retries_busy_detector_release_past_six_attempts_until_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeVisionRuntime()
    clock = [100.0]
    release_attempts = 0
    timeouts: list[float] = []
    request_ids: list[object] = []
    backoffs: list[float] = []

    class BusyThenReadyTransport(FakeHTTPClient):
        def post(
            self,
            url: str,
            *,
            json: object,
            **kwargs: object,
        ) -> FakeHTTPResponse:
            nonlocal release_attempts
            if not url.endswith("/v1/lifecycle/release-detector"):
                return super().post(url, json=json, **kwargs)
            self.posts.append((url, json))
            assert isinstance(json, dict)
            release_attempts += 1
            request_ids.append(json["request_id"])
            timeout = kwargs.get("timeout")
            assert isinstance(timeout, float)
            timeouts.append(timeout)
            if release_attempts <= 7:
                return FakeHTTPResponse(
                    {"detail": "Vision worker is busy"},
                    status_code=429,
                )
            return FakeHTTPResponse(
                {
                    **_identity(runtime, tmp_path),
                    "request_id": json["request_id"],
                    "released": True,
                    "siglip_loaded": False,
                    "detector_loaded": False,
                }
            )

    def advance(delay: float) -> None:
        backoffs.append(delay)
        clock[0] += delay

    monkeypatch.setattr(
        "videoscope.providers.vision_worker_client.monotonic",
        lambda: clock[0],
    )
    monkeypatch.setattr(
        "videoscope.providers.vision_worker_client.sleep",
        advance,
    )
    adapter = VisionWorkerClient(
        endpoint="http://127.0.0.1:8093",
        api_key=TOKEN,
        input_root=tmp_path,
        specification=runtime.specification,
        timeout=3.0,
        client=BusyThenReadyTransport(runtime, tmp_path),
    )

    adapter.release_ingestion_resources()

    assert release_attempts == 8
    assert len(set(request_ids)) == 1
    assert backoffs == [0.05, 0.1, 0.2, 0.25, 0.25, 0.25, 0.25]
    assert len(timeouts) == 8
    assert all(0 < timeout <= 3.0 for timeout in timeouts)
    assert timeouts == sorted(timeouts, reverse=True)


@pytest.mark.parametrize(
    ("status_code", "payload"),
    [
        (429, {"detail": "another busy response"}),
        (409, {"detail": "Vision worker is busy"}),
        (503, {"detail": "Vision worker is busy"}),
    ],
)
def test_client_does_not_retry_non_exact_busy_release_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    payload: object,
) -> None:
    runtime = FakeVisionRuntime()
    attempts = 0
    backoffs: list[float] = []

    class FailingTransport(FakeHTTPClient):
        def post(
            self,
            url: str,
            *,
            json: object,
            **kwargs: object,
        ) -> FakeHTTPResponse:
            nonlocal attempts
            if not url.endswith("/v1/lifecycle/release-detector"):
                return super().post(url, json=json, **kwargs)
            attempts += 1
            return FakeHTTPResponse(payload, status_code=status_code)

    monkeypatch.setattr(
        "videoscope.providers.vision_worker_client.sleep",
        lambda delay: backoffs.append(delay),
        raising=False,
    )
    adapter = VisionWorkerClient(
        endpoint="http://127.0.0.1:8093",
        api_key=TOKEN,
        input_root=tmp_path,
        specification=runtime.specification,
        client=FailingTransport(runtime, tmp_path),
    )

    with pytest.raises(
        RuntimeError,
        match="Vision worker ingestion resource release failed",
    ):
        adapter.release_ingestion_resources()

    assert attempts == 1
    assert backoffs == []


def test_client_stops_busy_release_retries_before_total_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeVisionRuntime()
    clock = [100.0]
    attempts = 0
    timeouts: list[float] = []
    backoffs: list[float] = []

    class AlwaysBusyTransport(FakeHTTPClient):
        def post(
            self,
            url: str,
            *,
            json: object,
            **kwargs: object,
        ) -> FakeHTTPResponse:
            nonlocal attempts
            assert url.endswith("/v1/lifecycle/release-detector")
            attempts += 1
            timeout = kwargs.get("timeout")
            assert isinstance(timeout, float)
            timeouts.append(timeout)
            return FakeHTTPResponse(
                {"detail": "Vision worker is busy"},
                status_code=429,
            )

    def advance(delay: float) -> None:
        backoffs.append(delay)
        clock[0] += delay

    monkeypatch.setattr(
        "videoscope.providers.vision_worker_client.monotonic",
        lambda: clock[0],
    )
    monkeypatch.setattr(
        "videoscope.providers.vision_worker_client.sleep",
        advance,
        raising=False,
    )
    adapter = VisionWorkerClient(
        endpoint="http://127.0.0.1:8093",
        api_key=TOKEN,
        input_root=tmp_path,
        specification=runtime.specification,
        timeout=0.06,
        client=AlwaysBusyTransport(runtime, tmp_path),
    )

    with pytest.raises(
        RuntimeError,
        match="Vision worker ingestion resource release failed",
    ):
        adapter.release_ingestion_resources()

    assert attempts == 2
    assert backoffs == [0.05]
    assert timeouts == pytest.approx([0.06, 0.01])


@pytest.mark.parametrize(
    "mutation",
    ["identity", "request_id", "released", "detector_loaded", "extra"],
)
def test_client_fails_closed_on_untrusted_detector_release_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    runtime = FakeVisionRuntime()
    adapter, transport = _worker_adapter(tmp_path, runtime)
    monkeypatch.setattr(
        "videoscope.providers.vision_worker_client.uuid4",
        lambda: SimpleNamespace(hex="0" * 32),
    )
    payload: dict[str, object] = {
        **_identity(runtime, tmp_path),
        "request_id": "0" * 32,
        "released": True,
        "siglip_loaded": False,
        "detector_loaded": False,
    }
    if mutation == "identity":
        payload["detector_specification_hash"] = "0" * 64
    elif mutation == "request_id":
        payload["request_id"] = "not-a-request-id"
    elif mutation == "released":
        payload["released"] = False
    elif mutation == "detector_loaded":
        payload["detector_loaded"] = True
    else:
        payload["unexpected"] = True
    transport.payload_override = payload

    with pytest.raises(
        RuntimeError,
        match="Vision worker ingestion resource release failed",
    ):
        adapter.release_ingestion_resources()
    assert len(transport.posts) == 1


def test_client_source_probe_is_bound_to_the_same_input_root(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    marker = tmp_path / "probe.bin"
    marker.write_bytes(b"videoscope-worker-root-probe-v1")
    adapter, transport = _worker_adapter(tmp_path, runtime)

    adapter.probe_source(marker)

    url, payload = transport.posts[-1]
    assert url.endswith("/v1/probe")
    assert isinstance(payload, dict)
    assert payload["input_root_identity"] == worker_input_root_identity(tmp_path)
    assert payload["expected_sha256"] == sha256(marker.read_bytes()).hexdigest()


def test_client_exposes_provider_registry_status(tmp_path: Path) -> None:
    runtime = FakeVisionRuntime()
    ready, _transport = _worker_adapter(tmp_path, runtime)

    assert ready.status() == ProviderStatus(
        id="vision-worker",
        label="Vision worker (SigLIP + RF-DETR)",
        state=ProviderState.READY,
        detail="Vision worker is ready",
        optional=True,
    )

    incompatible_transport = FakeHTTPClient(runtime, tmp_path)
    incompatible_transport.health_payload_override = {}
    unavailable = VisionWorkerClient(
        endpoint="http://127.0.0.1:8093",
        api_key=TOKEN,
        input_root=tmp_path,
        specification=runtime.specification,
        client=incompatible_transport,
    )
    assert unavailable.status() == ProviderStatus(
        id="vision-worker",
        label="Vision worker (SigLIP + RF-DETR)",
        state=ProviderState.UNAVAILABLE,
        detail="Vision worker is unreachable, unavailable, or incompatible",
        optional=True,
    )


@pytest.mark.parametrize(
    ("inference_timeout", "health_timeout", "expected_health_timeout"),
    [(120.0, None, 5.0), (120.0, 30.0, 30.0), (12.0, 4.0, 4.0)],
)
def test_client_health_uses_a_bounded_timeout_independent_of_slow_inference(
    tmp_path: Path,
    inference_timeout: float,
    health_timeout: float | None,
    expected_health_timeout: float,
) -> None:
    runtime = FakeVisionRuntime()
    observed_timeouts: list[float] = []

    class RecordingHealthClient(FakeHTTPClient):
        def get(self, url: str, **kwargs: object) -> FakeHTTPResponse:
            timeout = kwargs.get("timeout")
            assert isinstance(timeout, float)
            observed_timeouts.append(timeout)
            return super().get(url, **kwargs)

    transport = RecordingHealthClient(runtime, tmp_path)
    client_options: dict[str, object] = {}
    if health_timeout is not None:
        client_options["health_timeout"] = health_timeout
    adapter = VisionWorkerClient(
        endpoint="http://127.0.0.1:8093",
        api_key=TOKEN,
        input_root=tmp_path,
        specification=runtime.specification,
        timeout=inference_timeout,
        client=transport,
        **client_options,
    )

    assert adapter.capability().ready is True
    assert observed_timeouts == [expected_health_timeout]


def test_client_stalled_health_is_promptly_reported_unavailable(
    tmp_path: Path,
) -> None:
    runtime = FakeVisionRuntime()
    observed_timeouts: list[float] = []

    class StalledHealthClient(FakeHTTPClient):
        def get(self, _url: str, **kwargs: object) -> FakeHTTPResponse:
            timeout = kwargs.get("timeout")
            assert isinstance(timeout, float)
            observed_timeouts.append(timeout)
            raise TimeoutError("health endpoint stalled")

    adapter = VisionWorkerClient(
        endpoint="http://127.0.0.1:8093",
        api_key=TOKEN,
        input_root=tmp_path,
        specification=runtime.specification,
        timeout=600.0,
        client=StalledHealthClient(runtime, tmp_path),
    )

    assert adapter.capability().ready is False
    assert observed_timeouts == [5.0]


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:8093",
        "http://localhost:8093",
        "http://127.0.0.2:8093",
        "http://user@127.0.0.1:8093",
        "http://127.0.0.1:8093/path",
    ],
)
def test_client_accepts_only_plain_literal_loopback_origin(
    tmp_path: Path,
    endpoint: str,
) -> None:
    with pytest.raises(ValueError, match="loopback"):
        VisionWorkerClient(
            endpoint=endpoint,
            api_key=TOKEN,
            input_root=tmp_path,
            specification=_specification(),
        )


@pytest.mark.parametrize("mutation", ["identity", "unknown_id", "shape", "non_finite"])
def test_client_fails_closed_on_untrusted_embedding_response(
    tmp_path: Path,
    mutation: str,
) -> None:
    runtime = FakeVisionRuntime()
    path = tmp_path / "frame.png"
    _write_png(path)
    adapter, transport = _worker_adapter(tmp_path, runtime)
    base = {
        **_identity(runtime, tmp_path),
        "request_id": "0" * 32,
        "embedding_dimensions": 4,
        "items": [{"item_id": "image-0000", "vector": [1.0, 0.0, 0.0, 0.0]}],
    }
    if mutation == "identity":
        base["specification_hash"] = "0" * 64
    elif mutation == "unknown_id":
        base["items"][0]["item_id"] = "attacker"
    elif mutation == "shape":
        base["items"][0]["vector"] = [1.0]
    else:
        base["items"][0]["vector"] = [1.0, 0.0, 0.0, float("inf")]
    transport.payload_override = base

    with pytest.raises(RuntimeError, match="contract violation"):
        adapter.image_vectors([path])


def test_default_http_transport_is_bounded_proxy_free_and_redirect_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeVisionRuntime()
    calls: list[dict[str, object]] = []

    class StreamResponse:
        headers = {"Content-Length": str(MAX_RESPONSE_BYTES + 1)}

        def __enter__(self) -> "StreamResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

    class HTTPXClient:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

        def __enter__(self) -> "HTTPXClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def stream(self, *_args: object, **_kwargs: object) -> StreamResponse:
            return StreamResponse()

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=HTTPXClient))
    adapter = VisionWorkerClient(
        endpoint="http://127.0.0.1:8093",
        api_key=TOKEN,
        input_root=tmp_path,
        specification=runtime.specification,
    )

    assert not adapter.capability().ready
    assert calls == [{"trust_env": False, "follow_redirects": False}]


def test_default_http_transport_preserves_exact_busy_response_for_release_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeVisionRuntime()
    attempts = 0

    class StreamResponse:
        def __init__(self, payload: object, *, status_code: int) -> None:
            self.status_code = status_code
            self.body = json.dumps(payload).encode("utf-8")
            self.headers = {
                "Content-Length": str(len(self.body)),
                "Content-Type": "application/json",
            }

        def __enter__(self) -> "StreamResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise AssertionError("429 must remain available to lifecycle retry")

        def iter_bytes(self, *, chunk_size: int):  # type: ignore[no-untyped-def]
            assert chunk_size == 8192
            yield self.body

    class HTTPXClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def __enter__(self) -> "HTTPXClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def stream(
            self,
            method: str,
            url: str,
            **kwargs: object,
        ) -> StreamResponse:
            nonlocal attempts
            assert method == "POST"
            assert url.endswith("/v1/lifecycle/release-detector")
            request = kwargs["json"]
            assert isinstance(request, dict)
            attempts += 1
            if attempts == 1:
                return StreamResponse(
                    {"detail": "Vision worker is busy"},
                    status_code=429,
                )
            return StreamResponse(
                {
                    **_identity(runtime, tmp_path),
                    "request_id": request["request_id"],
                    "released": True,
                    "siglip_loaded": False,
                    "detector_loaded": False,
                },
                status_code=200,
            )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=HTTPXClient))
    monkeypatch.setattr(
        "videoscope.providers.vision_worker_client.sleep",
        lambda _delay: None,
    )
    adapter = VisionWorkerClient(
        endpoint="http://127.0.0.1:8093",
        api_key=TOKEN,
        input_root=tmp_path,
        specification=runtime.specification,
        timeout=1.0,
    )

    adapter.release_ingestion_resources()

    assert attempts == 2


def test_production_runtime_never_imports_heavy_modules_until_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "rf.pth"
    checkpoint.write_bytes(b"trusted")
    specification = VisionWorkerSpecification(
        siglip_model=SIGLIP_MODEL,
        siglip_revision=SIGLIP_REVISION,
        embedding_dimensions=768,
        detector_model_id="rfdetr-small",
        detector_checkpoint_sha256=sha256(b"trusted").hexdigest(),
        siglip_logit_scale=112.66890226978423,
        siglip_logit_bias=-16.771724700927734,
        minimum_confidence=0.25,
    )
    monkeypatch.setattr(
        "videoscope.providers.vision_worker.vision_worker_environment_is_exact",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "videoscope.providers.vision_worker._siglip_snapshot_is_available",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "videoscope.providers.vision_worker._siglip_compute_environment_is_exact",
        lambda **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        "videoscope.providers.vision_worker._rfdetr_compute_environment_is_exact",
        lambda **_kwargs: True,
        raising=False,
    )

    runtime = LocalVisionWorkerRuntime(
        specification=specification,
        detector_checkpoint=checkpoint,
    )

    assert runtime.available
    assert not runtime.siglip_loaded
    assert not runtime.detector_loaded


def test_siglip_identity_freezes_mps_float32_compute() -> None:
    specification = _specification()

    assert vision_worker_contract.SIGLIP_COMPUTE_BACKEND == "mps"
    assert vision_worker_contract.SIGLIP_COMPUTE_DTYPE == "float32"
    assert vision_worker_contract.RFDETR_COMPUTE_BACKEND == "mps"
    assert vision_worker_contract.RFDETR_COMPUTE_DTYPE == "float32"
    assert vision_worker_contract.VISION_WORKER_MINIMUM_MACOS_MAJOR == 14
    assert "macos-major>=14" in VISION_WORKER_RUNTIME_IDENTITY
    assert "siglip-backend==mps" in VISION_WORKER_RUNTIME_IDENTITY
    assert "siglip-dtype==float32" in VISION_WORKER_RUNTIME_IDENTITY
    assert "rfdetr-backend==mps" in VISION_WORKER_RUNTIME_IDENTITY
    assert "rfdetr-dtype==float32" in VISION_WORKER_RUNTIME_IDENTITY
    assert (
        "mps-residency==exclusive-vision-backbone-v1"
        in VISION_WORKER_RUNTIME_IDENTITY
    )
    assert "detector-release==stage-bound-v1" in VISION_WORKER_RUNTIME_IDENTITY
    assert specification.siglip_projection()["compute_backend"] == "mps"
    assert specification.siglip_projection()["compute_dtype"] == "float32"
    assert specification.detector_projection()["compute_backend"] == "mps"
    assert specification.detector_projection()["compute_dtype"] == "float32"


def test_runtime_fails_closed_without_exact_mps_float32_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "rf.pth"
    checkpoint.write_bytes(b"trusted")
    specification = VisionWorkerSpecification(
        siglip_model=SIGLIP_MODEL,
        siglip_revision=SIGLIP_REVISION,
        embedding_dimensions=768,
        detector_model_id="rfdetr-small",
        detector_checkpoint_sha256=sha256(b"trusted").hexdigest(),
        siglip_logit_scale=112.66890226978423,
        siglip_logit_bias=-16.771724700927734,
        minimum_confidence=0.25,
    )
    monkeypatch.setattr(
        vision_worker_module,
        "vision_worker_environment_is_exact",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        vision_worker_module,
        "_siglip_snapshot_is_available",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        vision_worker_module,
        "_siglip_compute_environment_is_exact",
        lambda **_kwargs: False,
        raising=False,
    )
    runtime = LocalVisionWorkerRuntime(
        specification=specification,
        detector_checkpoint=checkpoint,
    )

    assert not runtime.available


def test_runtime_fails_closed_without_exact_rfdetr_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "rf.pth"
    checkpoint.write_bytes(b"trusted")
    specification = VisionWorkerSpecification(
        siglip_model=SIGLIP_MODEL,
        siglip_revision=SIGLIP_REVISION,
        embedding_dimensions=768,
        detector_model_id="rfdetr-small",
        detector_checkpoint_sha256=sha256(b"trusted").hexdigest(),
        siglip_logit_scale=112.66890226978423,
        siglip_logit_bias=-16.771724700927734,
        minimum_confidence=0.25,
    )
    monkeypatch.setattr(
        vision_worker_module,
        "vision_worker_environment_is_exact",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        vision_worker_module,
        "_siglip_compute_environment_is_exact",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        vision_worker_module,
        "_rfdetr_compute_environment_is_exact",
        lambda **_kwargs: False,
        raising=False,
    )
    monkeypatch.setattr(
        vision_worker_module,
        "_siglip_snapshot_is_available",
        lambda *_args, **_kwargs: True,
    )
    runtime = LocalVisionWorkerRuntime(
        specification=specification,
        detector_checkpoint=checkpoint,
    )

    assert not runtime.available


def test_siglip_compute_probe_requires_real_mps_float32() -> None:
    float32 = object()
    probe = SimpleNamespace(
        device=SimpleNamespace(type="mps"),
        dtype=float32,
    )
    exact_torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(
                is_available=lambda: True,
                is_built=lambda: True,
            )
        ),
        empty=lambda _shape, *, device, dtype: (
            probe if device == "mps" and dtype is float32 else None
        ),
        float32=float32,
    )
    cpu_only_torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(
                is_available=lambda: False,
                is_built=lambda: True,
            )
        ),
        float32=float32,
    )

    assert vision_worker_module._siglip_compute_environment_is_exact(
        torch_module=exact_torch,
    )
    assert vision_worker_module._rfdetr_compute_environment_is_exact(
        torch_module=exact_torch,
    )
    assert not vision_worker_module._siglip_compute_environment_is_exact(
        torch_module=cpu_only_torch,
    )
    assert not vision_worker_module._rfdetr_compute_environment_is_exact(
        torch_module=cpu_only_torch,
    )


@pytest.mark.parametrize(
    ("context_device", "placed_device", "placed_dtype"),
    [
        ("cpu", "mps", "float32"),
        ("mps", "cpu", "float32"),
        ("mps", "mps", "float16"),
    ],
)
def test_rfdetr_compute_validation_rejects_wrong_actual_model(
    context_device: str,
    placed_device: str,
    placed_dtype: str,
) -> None:
    float32 = object()
    float16 = object()

    class Tensor:
        device = SimpleNamespace(type=placed_device)
        dtype = float32 if placed_dtype == "float32" else float16

        @staticmethod
        def is_floating_point() -> bool:
            return True

    class Model:
        def to(self, **_kwargs: object) -> "Model":
            return self

        def parameters(self):  # type: ignore[no-untyped-def]
            return iter((Tensor(),))

        def buffers(self):  # type: ignore[no-untyped-def]
            return iter(())

    detector = SimpleNamespace(
        model=SimpleNamespace(
            device=SimpleNamespace(type=context_device),
            model=Model(),
        )
    )

    with pytest.raises(RuntimeError, match="compute identity"):
        LocalVisionWorkerRuntime._place_detector_on_exact_compute(
            detector,
            SimpleNamespace(float32=float32),
            move=True,
        )


def test_environment_identity_checks_python_platform_lock_and_every_locked_package(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "requirements.lock"
    versions = {
        name: version
        for name, version in vision_worker_module.VISION_WORKER_CORE_DISTRIBUTIONS.items()
    }
    versions["safetensors"] = "0.7.0"
    lock.write_text(
        "".join(f"{name}=={version} \\\n" for name, version in versions.items()),
        encoding="utf-8",
    )

    exact = vision_worker_environment_is_exact(
        lock_path=lock,
        expected_lock_sha256=sha256(lock.read_bytes()).hexdigest(),
        python_version=(3, 12, 13),
        system="Darwin",
        machine="arm64",
        macos_version=(14, 0, 0),
        distribution_version=versions.__getitem__,
    )
    wrong_python = vision_worker_environment_is_exact(
        lock_path=lock,
        expected_lock_sha256=sha256(lock.read_bytes()).hexdigest(),
        python_version=(3, 12, 12),
        system="Darwin",
        machine="arm64",
        macos_version=(14, 0, 0),
        distribution_version=versions.__getitem__,
    )
    wrong_transitive_versions = {**versions, "safetensors": "0.6.2"}
    wrong_transitive = vision_worker_environment_is_exact(
        lock_path=lock,
        expected_lock_sha256=sha256(lock.read_bytes()).hexdigest(),
        python_version=(3, 12, 13),
        system="Darwin",
        machine="arm64",
        macos_version=(14, 0, 0),
        distribution_version=wrong_transitive_versions.__getitem__,
    )
    old_macos = vision_worker_environment_is_exact(
        lock_path=lock,
        expected_lock_sha256=sha256(lock.read_bytes()).hexdigest(),
        python_version=(3, 12, 13),
        system="Darwin",
        machine="arm64",
        macos_version=(13, 6, 9),
        distribution_version=versions.__getitem__,
    )

    assert exact
    assert not wrong_python
    assert not wrong_transitive
    assert not old_macos


def test_siglip_snapshot_availability_is_local_revision_and_blob_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = SIGLIP_REVISION
    root = tmp_path / "models--google--siglip2-base-patch16-224"
    snapshot = root / "snapshots" / revision
    blobs = root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    filenames = (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
    )
    cached: dict[str, str] = {}
    artifacts: list[tuple[str, str, int]] = []
    for filename in filenames:
        content = f"trusted:{filename}".encode("ascii")
        if filename in {"model.safetensors", "tokenizer.json", "tokenizer.model"}:
            blob_name = sha256(content).hexdigest()
        else:
            git_blob = sha1(usedforsecurity=False)
            git_blob.update(f"blob {len(content)}\0".encode("ascii"))
            git_blob.update(content)
            blob_name = git_blob.hexdigest()
        blob = blobs / blob_name
        blob.write_bytes(content)
        link = snapshot / filename
        link.symlink_to(blob)
        cached[filename] = str(link)
        artifacts.append((filename, blob_name, len(content)))
    monkeypatch.setattr(
        vision_worker_module,
        "SIGLIP_ARTIFACT_MANIFEST",
        {(SIGLIP_MODEL, revision): tuple(artifacts)},
        raising=False,
    )
    hub = SimpleNamespace(
        try_to_load_from_cache=lambda _model, filename, *, revision: (
            cached[filename] if revision == SIGLIP_REVISION else None
        )
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert vision_worker_module._siglip_snapshot_is_available(
        SIGLIP_MODEL,
        SIGLIP_REVISION,
    )
    (blobs / artifacts[0][1]).write_bytes(b"X" * artifacts[0][2])
    assert not vision_worker_module._siglip_snapshot_is_available(
        SIGLIP_MODEL,
        SIGLIP_REVISION,
    )
    assert not vision_worker_module._siglip_snapshot_is_available(
        SIGLIP_MODEL,
        "0" * 40,
    )


def test_runtime_loads_siglip_offline_and_validates_frozen_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "rf.pth"
    checkpoint.write_bytes(b"trusted")
    specification = _specification()
    monkeypatch.setattr(vision_worker_module, "siglip_profile_is_reviewed", lambda _spec: True)
    monkeypatch.setattr(
        LocalVisionWorkerRuntime,
        "available",
        property(lambda _self: True),
    )
    monkeypatch.setattr(
        vision_worker_module,
        "_siglip_snapshot_is_available",
        lambda *_args: True,
    )

    class Scalar:
        def __init__(self, value: float) -> None:
            self.value = value

        def detach(self) -> "Scalar":
            return self

        def cpu(self) -> "Scalar":
            return self

        def item(self) -> float:
            return self.value

    class Model:
        logit_scale = Scalar(math.log(specification.siglip_logit_scale))
        logit_bias = Scalar(specification.siglip_logit_bias)

        def eval(self) -> "Model":
            return self

        def to(self, *, device: str, dtype: object) -> "Model":
            assert device == "mps"
            assert dtype is float32
            return self

    model = Model()
    processor = object()
    calls: list[tuple[str, dict[str, object]]] = []

    class AutoModel:
        @staticmethod
        def from_pretrained(name: str, **kwargs: object) -> Model:
            calls.append((name, kwargs))
            return model

    class AutoProcessor:
        @staticmethod
        def from_pretrained(name: str, **kwargs: object) -> object:
            calls.append((name, kwargs))
            return processor

    float32 = object()
    fake_torch = SimpleNamespace(float32=float32)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=AutoModel, AutoProcessor=AutoProcessor),
    )
    runtime = LocalVisionWorkerRuntime(
        specification=specification,
        detector_checkpoint=checkpoint,
    )
    monkeypatch.setattr(
        vision_worker_module,
        "_siglip_compute_environment_is_exact",
        lambda **_kwargs: True,
        raising=False,
    )

    assert runtime._load_siglip() == (model, processor)
    assert all(call[1]["local_files_only"] is True for call in calls)
    assert all(call[1]["trust_remote_code"] is False for call in calls)
    assert all(call[1]["revision"] == SIGLIP_REVISION for call in calls)

    monkeypatch.setattr(
        vision_worker_module,
        "_siglip_snapshot_is_available",
        lambda *_args: False,
    )
    changed_runtime = LocalVisionWorkerRuntime(
        specification=specification,
        detector_checkpoint=checkpoint,
    )
    with pytest.raises(RuntimeError, match="artifacts changed"):
        changed_runtime._load_siglip()
    assert changed_runtime.siglip_loaded is False


def test_runtime_keeps_only_one_vision_backbone_resident_on_mps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "rf.pth"
    checkpoint.write_bytes(b"trusted")
    float32 = object()
    cache_events: list[str] = []

    class Tensor:
        device = SimpleNamespace(type="mps")
        dtype = float32

        @staticmethod
        def is_floating_point() -> bool:
            return True

    class DetectorModel:
        @staticmethod
        def parameters():  # type: ignore[no-untyped-def]
            return iter((Tensor(),))

        @staticmethod
        def buffers():  # type: ignore[no-untyped-def]
            return iter(())

    detector = SimpleNamespace(
        model=SimpleNamespace(
            device=SimpleNamespace(type="mps"),
            model=DetectorModel(),
        )
    )
    fake_torch = SimpleNamespace(
        float32=float32,
        mps=SimpleNamespace(
            synchronize=lambda: cache_events.append("synchronize"),
            empty_cache=lambda: cache_events.append("empty_cache"),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        vision_worker_module,
        "siglip_profile_is_reviewed",
        lambda _specification: True,
    )
    monkeypatch.setattr(
        vision_worker_module,
        "_rfdetr_compute_environment_is_exact",
        lambda **_kwargs: True,
    )

    siglip_runtime = LocalVisionWorkerRuntime(
        specification=_specification(),
        detector_checkpoint=checkpoint,
    )
    siglip_model = object()
    siglip_processor = object()
    siglip_runtime._siglip_model = siglip_model
    siglip_runtime._siglip_processor = siglip_processor
    siglip_runtime._detector = detector

    assert siglip_runtime._load_siglip() == (siglip_model, siglip_processor)
    assert siglip_runtime.siglip_loaded
    assert not siglip_runtime.detector_loaded
    assert cache_events == ["synchronize", "empty_cache"]

    cache_events.clear()
    detector_runtime = LocalVisionWorkerRuntime(
        specification=_specification(),
        detector_checkpoint=checkpoint,
    )
    detector_runtime._siglip_model = object()
    detector_runtime._siglip_processor = object()
    detector_runtime._detector = detector

    assert detector_runtime._load_detector() is detector
    assert detector_runtime.detector_loaded
    assert not detector_runtime.siglip_loaded
    assert cache_events == ["synchronize", "empty_cache"]


def test_runtime_releases_detector_idempotently_under_lifecycle_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "rf.pth"
    checkpoint.write_bytes(b"trusted")
    lifecycle_events: list[str] = []
    cache_events: list[str] = []

    class RecordingLock:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            lifecycle_events.append(f"{self.name}:enter")

        def __exit__(self, *_args: object) -> None:
            lifecycle_events.append(f"{self.name}:exit")

    fake_torch = SimpleNamespace(
        mps=SimpleNamespace(
            synchronize=lambda: cache_events.append("synchronize"),
            empty_cache=lambda: cache_events.append("empty_cache"),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        vision_worker_module,
        "siglip_profile_is_reviewed",
        lambda _specification: True,
    )
    monkeypatch.setattr(
        vision_worker_module.gc,
        "collect",
        lambda: cache_events.append("gc"),
    )
    runtime = LocalVisionWorkerRuntime(
        specification=_specification(),
        detector_checkpoint=checkpoint,
    )
    runtime._inference_lock = RecordingLock("inference")  # type: ignore[assignment]
    runtime._detector_load_lock = RecordingLock("detector")  # type: ignore[assignment]
    runtime._siglip_model = object()
    runtime._siglip_processor = object()
    runtime._detector = object()

    runtime.release_detector()
    runtime.release_detector()

    assert lifecycle_events == [
        "inference:enter",
        "detector:enter",
        "detector:exit",
        "inference:exit",
        "inference:enter",
        "detector:enter",
        "detector:exit",
        "inference:exit",
    ]
    assert cache_events == ["synchronize", "gc", "empty_cache"]
    assert runtime.detector_loaded is False
    assert runtime.siglip_loaded is True


def test_runtime_latches_unhealthy_when_detector_cache_release_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "rf.pth"
    checkpoint.write_bytes(b"trusted")

    def fail_empty_cache() -> None:
        raise RuntimeError("empty cache failed")

    fake_torch = SimpleNamespace(
        mps=SimpleNamespace(
            synchronize=lambda: None,
            empty_cache=fail_empty_cache,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        vision_worker_module,
        "siglip_profile_is_reviewed",
        lambda _specification: True,
    )
    monkeypatch.setattr(
        vision_worker_module,
        "vision_worker_environment_is_exact",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        vision_worker_module,
        "_siglip_compute_environment_is_exact",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        vision_worker_module,
        "_rfdetr_compute_environment_is_exact",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        vision_worker_module,
        "_siglip_snapshot_is_available",
        lambda *_args, **_kwargs: True,
    )
    runtime = LocalVisionWorkerRuntime(
        specification=_specification(),
        detector_checkpoint=checkpoint,
    )
    monkeypatch.setattr(runtime, "_checkpoint_is_exact", lambda: True)
    runtime._siglip_model = object()
    runtime._siglip_processor = object()
    runtime._detector = object()

    assert runtime.available is True
    with pytest.raises(RuntimeError, match="empty cache failed"):
        runtime.release_detector()

    assert runtime.available is False
    assert runtime.detector_loaded is False
    assert runtime.siglip_loaded is True
    with pytest.raises(RuntimeError, match="runtime is unhealthy"):
        runtime._load_siglip()


def test_runtime_blocks_rfdetr_redownload_and_checkpoint_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = b"trusted RF-DETR checkpoint"
    checkpoint = tmp_path / "rf-detr-small.pth"
    checkpoint.write_bytes(trusted)
    specification = VisionWorkerSpecification(
        siglip_model=SIGLIP_MODEL,
        siglip_revision=SIGLIP_REVISION,
        embedding_dimensions=4,
        detector_model_id="rfdetr-small",
        detector_checkpoint_sha256=sha256(trusted).hexdigest(),
        siglip_logit_scale=112.66890226978423,
        siglip_logit_bias=-16.771724700927734,
        minimum_confidence=0.25,
    )
    monkeypatch.setattr(vision_worker_module, "siglip_profile_is_reviewed", lambda _spec: True)
    monkeypatch.setattr(
        LocalVisionWorkerRuntime,
        "available",
        property(lambda _self: True),
    )
    monkeypatch.setattr(
        vision_worker_module,
        "_rfdetr_compute_environment_is_exact",
        lambda **_kwargs: True,
        raising=False,
    )
    calls: list[tuple[str, bool, bool]] = []
    constructor_devices: list[str] = []
    model_compute_moves: list[tuple[str, object]] = []

    class DetectorTensor:
        def __init__(self, torch_module: object) -> None:
            self.device = SimpleNamespace(type="cpu")
            self.dtype = torch_module.float32  # type: ignore[attr-defined]

        def is_floating_point(self) -> bool:
            return True

    class DetectorModel:
        def __init__(self, torch_module: object) -> None:
            self.tensor = DetectorTensor(torch_module)

        def to(self, *, device: str, dtype: object) -> "DetectorModel":
            model_compute_moves.append((device, dtype))
            self.tensor.device = SimpleNamespace(type=device)
            self.tensor.dtype = dtype
            return self

        def parameters(self):  # type: ignore[no-untyped-def]
            return iter((self.tensor,))

        def buffers(self):  # type: ignore[no-untyped-def]
            return iter(())

    def dependency_downloader(
        path: str,
        redownload: bool = False,
        validate_md5: bool = True,
    ) -> None:
        calls.append((path, redownload, validate_md5))
        if redownload:
            Path(path).write_bytes(b"dependency replacement")

    detr_module = SimpleNamespace(download_pretrain_weights=dependency_downloader)
    weights_module = SimpleNamespace(download_pretrain_weights=dependency_downloader)

    class RFDETRSmall:
        def __init__(self, *, pretrain_weights: str, device: str) -> None:
            constructor_devices.append(device)
            weights_module.download_pretrain_weights(
                pretrain_weights,
                redownload=True,
                validate_md5=False,
            )

    rfdetr_module = SimpleNamespace(
        __path__=[],
        RFDETRSmall=RFDETRSmall,
        detr=detr_module,
    )
    models_module = SimpleNamespace(__path__=[], weights=weights_module)
    monkeypatch.setitem(sys.modules, "rfdetr", rfdetr_module)
    monkeypatch.setitem(sys.modules, "rfdetr.detr", detr_module)
    monkeypatch.setitem(sys.modules, "rfdetr.models", models_module)
    monkeypatch.setitem(sys.modules, "rfdetr.models.weights", weights_module)
    runtime = LocalVisionWorkerRuntime(
        specification=specification,
        detector_checkpoint=checkpoint,
    )

    with pytest.raises(RuntimeError, match="download"):
        runtime._load_detector()

    assert checkpoint.read_bytes() == trusted
    assert runtime.detector_loaded is False
    assert detr_module.download_pretrain_weights is dependency_downloader
    assert weights_module.download_pretrain_weights is dependency_downloader
    assert calls == []

    loaded_paths: list[Path] = []
    loaded_payloads: list[bytes] = []

    class OfflineRFDETRSmall:
        def __init__(self, *, pretrain_weights: str, device: str) -> None:
            constructor_devices.append(device)
            torch_module = sys.modules["torch"]
            self.model = SimpleNamespace(
                device=SimpleNamespace(type=device),
                model=DetectorModel(torch_module),
            )
            normalized = str(Path(pretrain_weights).resolve(strict=True))
            detr_module.download_pretrain_weights(normalized)
            weights_module.download_pretrain_weights(normalized)
            loaded_path = Path(normalized)
            loaded_paths.append(loaded_path)
            loaded_payloads.append(loaded_path.read_bytes())

    rfdetr_module.RFDETRSmall = OfflineRFDETRSmall
    offline_runtime = LocalVisionWorkerRuntime(
        specification=specification,
        detector_checkpoint=checkpoint,
    )
    assert isinstance(offline_runtime._load_detector(), OfflineRFDETRSmall)
    assert loaded_paths[0] != checkpoint
    assert loaded_payloads == [trusted]
    assert not loaded_paths[0].exists()
    assert checkpoint.read_bytes() == trusted
    assert calls == []
    assert detr_module.download_pretrain_weights is dependency_downloader
    assert weights_module.download_pretrain_weights is dependency_downloader
    assert constructor_devices == ["mps", "mps"]
    assert len(model_compute_moves) == 1
    assert model_compute_moves[0][0] == "mps"
    assert model_compute_moves[0][1] is sys.modules["torch"].float32

    class MutatingRFDETRSmall:
        def __init__(self, *, pretrain_weights: str, device: str) -> None:
            constructor_devices.append(device)
            loaded_payloads.append(Path(pretrain_weights).read_bytes())
            checkpoint.write_bytes(b"attacker replacement")

    rfdetr_module.RFDETRSmall = MutatingRFDETRSmall
    mutating_runtime = LocalVisionWorkerRuntime(
        specification=specification,
        detector_checkpoint=checkpoint,
    )
    with pytest.raises(RuntimeError, match="changed during loading"):
        mutating_runtime._load_detector()
    assert loaded_payloads[-1] == trusted
    assert mutating_runtime.detector_loaded is False
    assert constructor_devices == ["mps", "mps", "mps"]


def test_runtime_embedding_and_detection_adapters_are_batched_and_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "rf.pth"
    checkpoint.write_bytes(b"trusted")
    monkeypatch.setattr(vision_worker_module, "siglip_profile_is_reviewed", lambda _spec: True)
    runtime = LocalVisionWorkerRuntime(
        specification=_specification(),
        detector_checkpoint=checkpoint,
    )
    image = tmp_path / "frame.png"
    _write_png(image)
    monkeypatch.setattr(
        vision_worker_module,
        "_SIGLIP_IMAGE_INFERENCE_BATCH_SIZE",
        1,
        raising=False,
    )
    monkeypatch.setattr(
        vision_worker_module,
        "_SIGLIP_TEXT_INFERENCE_BATCH_SIZE",
        1,
        raising=False,
    )
    processor_calls: list[tuple[str, int]] = []
    text_processor_options: list[dict[str, object]] = []
    compute_moves: list[tuple[str, object | None]] = []
    float32 = object()
    int64 = object()

    class Input:
        def __init__(self, count: int, *, floating: bool) -> None:
            self.count = count
            self.floating = floating
            self.device = SimpleNamespace(type="cpu")
            self.dtype = float32 if floating else int64

        def is_floating_point(self) -> bool:
            return self.floating

        def to(self, *, device: str, dtype: object | None = None) -> "Input":
            compute_moves.append((device, dtype))
            self.device = SimpleNamespace(type=device)
            if dtype is not None:
                self.dtype = dtype
            return self

    class Encoded:
        def __init__(self, rows: list[list[float]]) -> None:
            self.rows = rows
            self.ndim = 2
            self.shape = (len(rows), 4)

        def detach(self) -> "Encoded":
            return self

        def float(self) -> "Encoded":
            return self

        def cpu(self) -> "Encoded":
            return self

        def tolist(self) -> list[list[float]]:
            return self.rows

    class Processor:
        def __call__(self, *, images=None, text=None, **_kwargs):  # type: ignore[no-untyped-def]
            count = len(images if images is not None else text)
            processor_calls.append(("images" if images is not None else "texts", count))
            if text is not None:
                text_processor_options.append(_kwargs)
            return {"input": Input(count, floating=images is not None)}

    class Model:
        def get_image_features(self, *, input: Input) -> Encoded:
            return Encoded([[1.0, 0.0, 0.0, 0.0] for _ in range(input.count)])

        def get_text_features(self, *, input: Input) -> Encoded:
            return Encoded([[0.0, 1.0, 0.0, 0.0] for _ in range(input.count)])

    fake_torch = SimpleNamespace(float32=float32, inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runtime, "_load_siglip", lambda: (Model(), Processor()))
    monkeypatch.setattr(runtime, "_normalize", lambda encoded: encoded)
    image_vectors = runtime.embed_images((image, image))
    text_vectors = runtime.embed_texts(("слово " * 80, "баскетбол 🏀"))

    class Detections:
        confidence = [0.9]
        class_id = [1]
        xyxy = [[1.0, 2.0, 5.0, 8.0]]
        data = {"class_name": ["person"]}

        def __len__(self) -> int:
            return 1

    detector = SimpleNamespace(
        predict=lambda *_args, **_kwargs: Detections(),
    )
    rfdetr = SimpleNamespace(__path__=[])
    assets = SimpleNamespace(__path__=[])
    coco = SimpleNamespace(COCO_CLASSES={1: "person"})
    monkeypatch.setitem(sys.modules, "rfdetr", rfdetr)
    monkeypatch.setitem(sys.modules, "rfdetr.assets", assets)
    monkeypatch.setitem(sys.modules, "rfdetr.assets.coco_classes", coco)
    monkeypatch.setattr(runtime, "_load_detector", lambda: detector)
    detections = runtime.detect(image, 0.25)

    assert image_vectors == [[1.0, 0.0, 0.0, 0.0]] * 2
    assert text_vectors == [[0.0, 1.0, 0.0, 0.0]] * 2
    assert processor_calls == [
        ("images", 1),
        ("images", 1),
        ("texts", 1),
        ("texts", 1),
    ]
    assert text_processor_options == [
        {
            "max_length": 64,
            "padding": "max_length",
            "return_tensors": "pt",
            "truncation": True,
        },
        {
            "max_length": 64,
            "padding": "max_length",
            "return_tensors": "pt",
            "truncation": True,
        },
    ]
    assert compute_moves == [
        ("mps", float32),
        ("mps", float32),
        ("mps", None),
        ("mps", None),
    ]
    assert detections == [
        {
            "label": "person",
            "confidence": 0.9,
            "x": 3.0,
            "y": 5.0,
            "width": 4.0,
            "height": 6.0,
        }
    ]


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_rfdetr_adapter_always_receives_three_channel_rgb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    from PIL import Image

    checkpoint = tmp_path / "rf.pth"
    checkpoint.write_bytes(b"trusted")
    source = tmp_path / f"{mode}.png"
    Image.new(mode, (8, 6)).save(source, format="PNG")
    monkeypatch.setattr(vision_worker_module, "siglip_profile_is_reviewed", lambda _spec: True)
    runtime = LocalVisionWorkerRuntime(
        specification=_specification(),
        detector_checkpoint=checkpoint,
    )
    seen: list[tuple[str, tuple[int, int]]] = []

    def predict(image: object, **_kwargs: object) -> None:
        seen.append((image.mode, image.size))  # type: ignore[attr-defined]
        return None

    detector = SimpleNamespace(predict=predict)
    rfdetr = SimpleNamespace(__path__=[])
    assets = SimpleNamespace(__path__=[])
    coco = SimpleNamespace(COCO_CLASSES={})
    monkeypatch.setitem(sys.modules, "rfdetr", rfdetr)
    monkeypatch.setitem(sys.modules, "rfdetr.assets", assets)
    monkeypatch.setitem(sys.modules, "rfdetr.assets.coco_classes", coco)
    monkeypatch.setattr(runtime, "_load_detector", lambda: detector)

    assert runtime.detect(source, 0.25) == []
    assert seen == [("RGB", (8, 6))]


def test_request_model_requires_hash_bound_image_identity() -> None:
    runtime = FakeVisionRuntime()
    payload = {
        **_identity(runtime),
        "request_id": "a" * 32,
        "items": [
            {
                "item_id": "frame",
                "relative_path": "frame.png",
                "expected_sha256": "0" * 64,
                "expected_size_bytes": 1,
            }
        ],
    }

    request = VisionEmbedImagesRequest.model_validate(payload)

    assert request.items[0].expected_sha256 == "0" * 64


@pytest.mark.parametrize(
    "field",
    [
        "siglip_preprocessing_revision",
        "siglip_tokenizer_revision",
        "detector_adapter_revision",
    ],
)
def test_worker_specification_rejects_unimplemented_adapter_revisions(field: str) -> None:
    with pytest.raises(ValueError, match="revision"):
        replace(_specification(), **{field: "unimplemented-v999"})


def test_worker_default_port_matches_reserved_worker_layout() -> None:
    settings = VisionWorkerSettings(api_key=TOKEN, _env_file=None)

    assert settings.host == "127.0.0.1"
    assert settings.port == 8783


def test_worker_settings_scope_product_data_to_three_nested_directories() -> None:
    settings = VisionWorkerSettings(
        api_key=TOKEN,
        product_data_subdirectory="product",
        _env_file=None,
    )

    assert settings.allowed_input_subdirectories() == (
        "product/visual-index",
        "product/thumbnails",
        "product/tmp",
    )

    with pytest.raises(ValueError):
        VisionWorkerSettings(
            api_key=TOKEN,
            product_data_subdirectory="../product",
            _env_file=None,
        )


def test_worker_settings_resolve_only_reviewed_siglip_quality_profiles() -> None:
    standard = VisionWorkerSettings(api_key=TOKEN, _env_file=None).specification()
    quality = VisionWorkerSettings(
        api_key=TOKEN,
        siglip_model="google/siglip2-base-patch16-384",
        _env_file=None,
    ).specification()

    assert standard.siglip_model == "google/siglip2-base-patch16-224"
    assert standard.siglip_revision == SIGLIP_REVISION
    assert standard.siglip_logit_scale == 112.66890226978423
    assert quality.siglip_model == "google/siglip2-base-patch16-384"
    assert quality.siglip_revision == "f775b65a79762255128c981547af89addcfe0f88"
    assert quality.siglip_logit_scale == 112.84601055046839
    assert quality.siglip_logit_bias == -16.77224349975586

    with pytest.raises(ValueError, match="reviewed"):
        VisionWorkerSettings(
            api_key=TOKEN,
            siglip_model="attacker/unknown-model",
            _env_file=None,
        )


def test_worker_settings_freeze_explicit_detector_model_and_checksum() -> None:
    specification = VisionWorkerSettings(
        api_key=TOKEN,
        detector_model_id="rfdetr-medium",
        detector_checkpoint_sha256="a" * 64,
        _env_file=None,
    ).specification()

    assert specification.detector_model_id == "rfdetr-medium"
    assert specification.detector_checkpoint_sha256 == "a" * 64

    with pytest.raises(ValueError):
        VisionWorkerSettings(
            api_key=TOKEN,
            detector_model_id="rfdetr-attacker",
            _env_file=None,
        )
