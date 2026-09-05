from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import sys
import time
from weakref import ref

from fastapi.testclient import TestClient
import pytest

import videoscope.providers.qwen_worker as qwen_worker_module
from videoscope.model_manifest import QWEN_VIDEO_MODEL
from videoscope.providers.qwen_video import QWEN_PROMPT_PROTOCOL_SHA256, QwenVideoJudgement
from videoscope.providers.qwen_worker import (
    MAX_RESPONSE_BYTES,
    MLXQwenWorkerRuntime,
    QWEN_INFERENCE_RUNTIME_IDENTITY,
    QWEN_SOURCE_BUNDLE_SHA256,
    QWEN_WORKER_SCHEMA_VERSION,
    QwenJudgeRequest,
    QwenWorkerClient,
    QwenWorkerSettings,
    attest_qwen_worker_startup,
    create_qwen_worker_app,
)


TOKEN = "q" * 32
MODEL_IDENTITY = "organization/qwen@" + "a" * 40


def _generated_response(prompt_kind: str = "generic_query") -> str:
    if prompt_kind == "basketball_facts":
        return json.dumps({
            "shot_attempt": True, "ball_through_hoop": True,
            "shooter_outside_arc": None, "three_point_signal": None,
            "shooter_jersey": None, "evidence": "The ball passes through the rim.",
        })
    return json.dumps({
        "matches_query": True, "confidence": 0.9,
        "event_start": None, "event_end": None, "shot_attempt": None,
        "made": None, "three_point": None, "shooter_jersey": None,
        "evidence": "The requested action is visible.",
    })


def _canonical_sha256(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_source_bundle_identity_is_deterministic_and_path_free() -> None:
    assert qwen_worker_module._qwen_source_bundle_sha256() == (
        QWEN_SOURCE_BUNDLE_SHA256
    )
    assert len(QWEN_SOURCE_BUNDLE_SHA256) == 64
    assert "/Users/" not in QWEN_SOURCE_BUNDLE_SHA256


def test_worker_startup_attests_exact_environment_and_complete_model(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"reviewed-lock")
    runtime_manifest = {
        "backend_lock_sha256": sha256(lock.read_bytes()).hexdigest(),
        "distributions": {"demo-runtime": "1.2.3"},
        "platform": "aarch64-apple-darwin-macos14plus",
        "python": "3.12.13",
        "schema_version": 1,
    }
    artifact = b"reviewed-safe-weights"
    model_manifest = {
        "artifacts": [
            {
                "name": "model.safetensors",
                "sha256": sha256(artifact).hexdigest(),
                "size": len(artifact),
            }
        ],
        "license": "apache-2.0",
        "model": "organization/qwen",
        "revision": "a" * 40,
        "schema_version": 1,
    }
    runtime_path = tmp_path / "runtime.json"
    model_path = tmp_path / "models.json"
    runtime_path.write_text(json.dumps(runtime_manifest), encoding="utf-8")
    model_path.write_text(json.dumps(model_manifest), encoding="utf-8")
    snapshot = tmp_path / "cache" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(artifact)

    attestation = attest_qwen_worker_startup(
        "organization/qwen",
        "a" * 40,
        runtime_manifest_path=runtime_path,
        model_manifest_path=model_path,
        backend_lock_path=lock,
        expected_runtime_manifest_sha256=_canonical_sha256(runtime_manifest),
        expected_model_manifest_sha256=_canonical_sha256(model_manifest),
        python_version=(3, 12, 13),
        system="Darwin",
        machine="arm64",
        macos_version="14.4",
        installed_distributions={"demo-runtime": "1.2.3"},
        snapshot_resolver=lambda _model, _revision: snapshot,
    )

    assert attestation["model_identity"] == "organization/qwen@" + "a" * 40
    assert attestation["runtime_identity"] == QWEN_INFERENCE_RUNTIME_IDENTITY
    assert attestation["source_bundle_sha256"] == QWEN_SOURCE_BUNDLE_SHA256
    assert attestation["prompt_protocol_sha256"] == QWEN_PROMPT_PROTOCOL_SHA256
    assert str(tmp_path) not in json.dumps(attestation)

    with pytest.raises(RuntimeError, match="executable protocol identity"):
        attest_qwen_worker_startup(
            "organization/qwen",
            "a" * 40,
            runtime_manifest_path=runtime_path,
            model_manifest_path=model_path,
            backend_lock_path=lock,
            expected_runtime_manifest_sha256=_canonical_sha256(runtime_manifest),
            expected_model_manifest_sha256=_canonical_sha256(model_manifest),
            expected_source_bundle_sha256="0" * 64,
            python_version=(3, 12, 13),
            system="Darwin",
            machine="arm64",
            macos_version="14.4",
            installed_distributions={"demo-runtime": "1.2.3"},
            snapshot_resolver=lambda _model, _revision: snapshot,
        )

    (snapshot / "model.safetensors").write_bytes(b"tampered-safe-weights")
    with pytest.raises(RuntimeError, match="model artifact"):
        attest_qwen_worker_startup(
            "organization/qwen",
            "a" * 40,
            runtime_manifest_path=runtime_path,
            model_manifest_path=model_path,
            backend_lock_path=lock,
            expected_runtime_manifest_sha256=_canonical_sha256(runtime_manifest),
            expected_model_manifest_sha256=_canonical_sha256(model_manifest),
            python_version=(3, 12, 13),
            system="Darwin",
            machine="arm64",
            macos_version="14.4",
            installed_distributions={"demo-runtime": "1.2.3"},
            snapshot_resolver=lambda _model, _revision: snapshot,
        )


class FakeWorkerRuntime:
    def __init__(self) -> None:
        self.model_identity = MODEL_IDENTITY
        self.loaded = False
        self.calls: list[tuple[object, Path]] = []
        self.error: Exception | None = None
        self.is_available = True

    @property
    def available(self) -> bool:
        return self.is_available

    def judge(self, request: object, source: Path) -> QwenVideoJudgement:
        self.calls.append((request, source))
        if self.error is not None:
            raise self.error
        return QwenVideoJudgement(
            matches_query=True,
            confidence=0.91,
            shot_attempt=True,
            evidence="visible action",
        )


def _worker_client(
    tmp_path: Path,
    runtime: FakeWorkerRuntime,
    *,
    max_input_bytes: int = 1024,
    max_concurrency: int = 1,
) -> TestClient:
    app = create_qwen_worker_app(
        runtime=runtime,
        input_root=tmp_path,
        api_key=TOKEN,
        max_input_bytes=max_input_bytes,
        max_concurrency=max_concurrency,
    )
    return TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    )


def _request(
    input_root: Path,
    relative_path: str = "candidate.mp4",
) -> dict[str, object]:
    source = input_root / relative_path
    try:
        payload = source.read_bytes()
    except OSError:
        payload = b"unavailable"
    return {
        "schema_version": QWEN_WORKER_SCHEMA_VERSION,
        "request_id": "a" * 32,
        "model_identity": MODEL_IDENTITY,
        "runtime_identity": QWEN_INFERENCE_RUNTIME_IDENTITY,
        "source_bundle_sha256": QWEN_SOURCE_BUNDLE_SHA256,
        "prompt_protocol_sha256": QWEN_PROMPT_PROTOCOL_SHA256,
        "input_root_sha256": qwen_worker_module._input_root_identity(input_root),
        "input_kind": "video",
        "relative_path": relative_path,
        "expected_sha256": sha256(payload).hexdigest(),
        "expected_byte_size": len(payload),
        "prompt_kind": "basketball_facts",
        "query": None,
        "fps": 2.0,
        "max_tokens": 320,
    }


def test_worker_requires_authentication_and_loopback_client(tmp_path: Path) -> None:
    runtime = FakeWorkerRuntime()
    app = create_qwen_worker_app(
        runtime=runtime,
        input_root=tmp_path,
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
        client=("192.0.2.12", 50000),
    )

    assert local.get("/v1/health").status_code == 401
    assert remote.get(
        "/v1/health", headers={"Authorization": f"Bearer {TOKEN}"}
    ).status_code == 403

    hostile_host = local.get(
        "/v1/health",
        headers={"Authorization": f"Bearer {TOKEN}", "Host": "attacker.example"},
    )
    assert hostile_host.status_code == 400


def test_worker_bounds_json_body_before_contract_parsing(tmp_path: Path) -> None:
    runtime = FakeWorkerRuntime()
    app = create_qwen_worker_app(
        runtime=runtime,
        input_root=tmp_path,
        api_key=TOKEN,
        max_request_bytes=32,
    )
    client = TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    )

    response = client.post(
        "/v1/judge",
        content=b"{" + b"x" * 64 + b"}",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}
    assert runtime.calls == []


def test_health_exposes_bounded_capability_without_loading_model(tmp_path: Path) -> None:
    runtime = FakeWorkerRuntime()
    client = _worker_client(tmp_path, runtime, max_input_bytes=4096, max_concurrency=2)

    response = client.get(
        "/v1/health", headers={"Authorization": f"Bearer {TOKEN}"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": QWEN_WORKER_SCHEMA_VERSION,
        "status": "ok",
        "model_identity": MODEL_IDENTITY,
        "runtime_identity": QWEN_INFERENCE_RUNTIME_IDENTITY,
        "source_bundle_sha256": QWEN_SOURCE_BUNDLE_SHA256,
        "prompt_protocol_sha256": QWEN_PROMPT_PROTOCOL_SHA256,
        "input_root_sha256": qwen_worker_module._input_root_identity(tmp_path),
        "loaded": False,
        "input_kinds": ["video", "storyboard"],
        "max_input_bytes": 4096,
        "max_query_chars": 500,
        "max_tokens": 1024,
        "max_concurrency": 2,
    }

    runtime.is_available = False
    unavailable = client.get(
        "/v1/health", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert unavailable.json()["status"] == "unavailable"


def test_worker_contract_is_strict_and_bounds_request_before_inference(
    tmp_path: Path,
) -> None:
    runtime = FakeWorkerRuntime()
    source = tmp_path / "candidate.mp4"
    source.write_bytes(b"12345")
    client = _worker_client(tmp_path, runtime, max_input_bytes=4)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    extra = {**_request(tmp_path), "unexpected": True}
    assert client.post("/v1/judge", json=extra, headers=headers).status_code == 422

    oversized = client.post("/v1/judge", json=_request(tmp_path), headers=headers)
    assert oversized.status_code == 413
    assert oversized.json() == {"detail": "Qwen worker input is too large"}
    assert runtime.calls == []


def test_judge_request_supports_only_versioned_prompt_input_combinations(
    tmp_path: Path,
) -> None:
    generic_video = {
        **_request(tmp_path),
        "prompt_kind": "generic_query",
        "query": "a person waves",
    }

    parsed = QwenJudgeRequest.model_validate(generic_video)
    assert parsed.input_kind == "video"
    assert parsed.prompt_kind == "generic_query"
    assert parsed.query == "a person waves"
    assert parsed.fps == 2.0

    invalid_requests = (
        {**generic_video, "query": None},
        {**generic_video, "query": "   "},
        {**generic_video, "fps": None},
        {**_request(tmp_path), "query": "must stay fixed"},
        {
            **generic_video,
            "relative_path": "candidate.jpg",
            "input_kind": "storyboard",
            "fps": 2.0,
        },
        {
            **_request(tmp_path, "candidate.jpg"),
            "input_kind": "storyboard",
            "fps": None,
        },
    )
    for request in invalid_requests:
        with pytest.raises(ValueError):
            QwenJudgeRequest.model_validate(request)


def test_worker_rejects_model_mismatch_before_reading_input(tmp_path: Path) -> None:
    runtime = FakeWorkerRuntime()
    client = _worker_client(tmp_path, runtime)
    request = {
        **_request(tmp_path, "missing.mp4"),
        "model_identity": "wrong/model",
    }

    response = client.post(
        "/v1/judge",
        json=request,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Qwen worker execution identity does not match"}
    assert runtime.calls == []


@pytest.mark.parametrize(
    "relative_path",
    [
        "../candidate.mp4",
        "/tmp/candidate.mp4",
        "nested//candidate.mp4",
        "candidate.txt",
    ],
)
def test_worker_rejects_unsafe_or_unsupported_input_paths(
    tmp_path: Path,
    relative_path: str,
) -> None:
    runtime = FakeWorkerRuntime()
    (tmp_path / "candidate.txt").write_bytes(b"text")
    client = _worker_client(tmp_path, runtime)

    response = client.post(
        "/v1/judge",
        json=_request(tmp_path, relative_path),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code in {400, 422}
    assert runtime.calls == []


def test_worker_rejects_symlink_even_when_target_stays_under_input_root(
    tmp_path: Path,
) -> None:
    runtime = FakeWorkerRuntime()
    target = tmp_path / "real.mp4"
    target.write_bytes(b"video")
    (tmp_path / "candidate.mp4").symlink_to(target)
    client = _worker_client(tmp_path, runtime)

    response = client.post(
        "/v1/judge",
        json=_request(tmp_path),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid Qwen worker input"}
    assert runtime.calls == []


def test_worker_returns_strict_judgement_and_sanitizes_runtime_errors(
    tmp_path: Path,
) -> None:
    runtime = FakeWorkerRuntime()
    source = tmp_path / "candidate.mp4"
    source.write_bytes(b"video")
    client = _worker_client(tmp_path, runtime)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    response = client.post("/v1/judge", json=_request(tmp_path), headers=headers)

    assert response.status_code == 200
    assert response.json()["request_id"] == "a" * 32
    assert response.json()["model_identity"] == MODEL_IDENTITY
    assert response.json()["source_sha256"] == sha256(b"video").hexdigest()
    assert response.json()["byte_size"] == len(b"video")
    assert response.json()["judgement"]["confidence"] == 0.91
    materialized = runtime.calls[0][1]
    assert materialized != source.resolve()
    assert materialized.name == "input.mp4"
    assert not materialized.exists()

    runtime.error = RuntimeError("secret path: /private/model/token")
    failed = client.post("/v1/judge", json=_request(tmp_path), headers=headers)

    assert failed.status_code == 503
    assert failed.json() == {"detail": "Qwen worker inference failed"}
    assert "/private/model/token" not in failed.text


def test_worker_executes_generic_query_on_native_video(tmp_path: Path) -> None:
    runtime = FakeWorkerRuntime()
    source = tmp_path / "candidate.mp4"
    source.write_bytes(b"video")
    client = _worker_client(tmp_path, runtime)
    request = {
        **_request(tmp_path),
        "prompt_kind": "generic_query",
        "query": "a person waves",
    }

    response = client.post(
        "/v1/judge",
        json=request,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 200
    assert len(runtime.calls) == 1
    received, received_source = runtime.calls[0]
    assert isinstance(received, QwenJudgeRequest)
    assert received.prompt_kind == "generic_query"
    assert received.query == "a person waves"
    assert received.fps == 2.0
    assert received_source != source.resolve()
    assert received_source.name == "input.mp4"
    assert not received_source.exists()


class MutatingRuntime(FakeWorkerRuntime):
    def __init__(self, exposed_source: Path) -> None:
        super().__init__()
        self.exposed_source = exposed_source

    def judge(self, request: object, source: Path) -> QwenVideoJudgement:
        result = super().judge(request, source)
        self.exposed_source.write_bytes(b"changed during inference")
        return result


def test_worker_fails_closed_when_input_changes_during_inference(tmp_path: Path) -> None:
    source = tmp_path / "candidate.mp4"
    source.write_bytes(b"video")
    runtime = MutatingRuntime(source)
    client = _worker_client(tmp_path, runtime)

    response = client.post(
        "/v1/judge",
        json=_request(tmp_path),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Qwen worker input changed during inference"}


class RestoringMetadataRuntime(FakeWorkerRuntime):
    def __init__(self, exposed_source: Path) -> None:
        super().__init__()
        self.exposed_source = exposed_source

    def judge(self, request: object, source: Path) -> QwenVideoJudgement:
        result = super().judge(request, source)
        before = self.exposed_source.stat()
        time.sleep(0.002)
        original = self.exposed_source.read_bytes()
        self.exposed_source.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        os.utime(
            self.exposed_source,
            ns=(before.st_atime_ns, before.st_mtime_ns),
        )
        assert self.exposed_source.stat().st_ctime_ns != before.st_ctime_ns
        return result


def test_worker_detects_in_place_change_even_when_size_and_mtime_are_restored(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate.mp4"
    source.write_bytes(b"video")
    runtime = RestoringMetadataRuntime(source)
    client = _worker_client(tmp_path, runtime)

    response = client.post(
        "/v1/judge",
        json=_request(tmp_path),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Qwen worker input changed during inference"}


class RenameSwapRestoringRuntime(FakeWorkerRuntime):
    def __init__(self, exposed_source: Path, replacement: Path) -> None:
        super().__init__()
        self.exposed_source = exposed_source
        self.replacement = replacement
        self.materialized_source: Path | None = None

    def judge(self, request: object, source: Path) -> QwenVideoJudgement:
        self.materialized_source = source
        assert source != self.exposed_source
        assert source.read_bytes() == b"original-video"
        backup = self.exposed_source.with_suffix(".original")
        os.replace(self.exposed_source, backup)
        os.replace(self.replacement, self.exposed_source)
        assert self.exposed_source.read_bytes() == b"substitute-vid"
        os.replace(self.exposed_source, self.replacement)
        os.replace(backup, self.exposed_source)
        return super().judge(request, source)


def test_worker_fails_closed_on_rename_swap_restore_and_uses_private_copy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate.mp4"
    source.write_bytes(b"original-video")
    replacement = tmp_path / "candidate.replacement"
    replacement.write_bytes(b"substitute-vid")
    runtime = RenameSwapRestoringRuntime(source, replacement)
    client = _worker_client(tmp_path, runtime)

    response = client.post(
        "/v1/judge",
        json=_request(tmp_path),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Qwen worker input changed during inference"}
    assert source.read_bytes() == b"original-video"
    assert runtime.materialized_source is not None
    assert not runtime.materialized_source.exists()


def test_worker_rejects_wrong_judge_content_identity_before_inference(
    tmp_path: Path,
) -> None:
    runtime = FakeWorkerRuntime()
    source = tmp_path / "candidate.mp4"
    source.write_bytes(b"video")
    client = _worker_client(tmp_path, runtime)
    request = {**_request(tmp_path), "expected_sha256": "0" * 64}

    response = client.post(
        "/v1/judge",
        json=request,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Qwen worker input identity mismatch"}
    assert runtime.calls == []


class BlockingRuntime(FakeWorkerRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.entered = Event()
        self.release = Event()

    def judge(self, request: object, source: Path) -> QwenVideoJudgement:
        self.entered.set()
        self.release.wait(timeout=2)
        return super().judge(request, source)


def test_worker_rejects_work_above_configured_concurrency(tmp_path: Path) -> None:
    runtime = BlockingRuntime()
    (tmp_path / "candidate.mp4").write_bytes(b"video")
    client = _worker_client(tmp_path, runtime, max_concurrency=1)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            client.post,
            "/v1/judge",
            json=_request(tmp_path),
            headers=headers,
        )
        assert runtime.entered.wait(timeout=1)
        second = client.post("/v1/judge", json=_request(tmp_path), headers=headers)
        runtime.release.set()
        first_response = first.result(timeout=2)

    assert first_response.status_code == 200
    assert second.status_code == 429
    assert second.json() == {"detail": "Qwen worker is busy"}


class FakeResponse:
    def __init__(self, payload: object, status_error: Exception | None = None) -> None:
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self) -> None:
        if self.status_error is not None:
            raise self.status_error

    def json(self) -> object:
        return self.payload


class RecordingHTTPClient:
    def __init__(self, input_root: Path) -> None:
        self.calls: list[tuple[str, str, object | None, dict[str, str], float]] = []
        input_root_sha256 = qwen_worker_module._input_root_identity(input_root)
        self.health_payload: object = {
            "schema_version": QWEN_WORKER_SCHEMA_VERSION,
            "status": "ok",
            "model_identity": MODEL_IDENTITY,
            "runtime_identity": QWEN_INFERENCE_RUNTIME_IDENTITY,
            "source_bundle_sha256": QWEN_SOURCE_BUNDLE_SHA256,
            "prompt_protocol_sha256": QWEN_PROMPT_PROTOCOL_SHA256,
            "input_root_sha256": input_root_sha256,
            "loaded": False,
            "input_kinds": ["video", "storyboard"],
            "max_input_bytes": 1024,
            "max_query_chars": 500,
            "max_tokens": 1024,
            "max_concurrency": 1,
        }
        self.judge_payload: object = {
            "schema_version": QWEN_WORKER_SCHEMA_VERSION,
            "request_id": "b" * 32,
            "model_identity": MODEL_IDENTITY,
            "runtime_identity": QWEN_INFERENCE_RUNTIME_IDENTITY,
            "source_bundle_sha256": QWEN_SOURCE_BUNDLE_SHA256,
            "prompt_protocol_sha256": QWEN_PROMPT_PROTOCOL_SHA256,
            "input_root_sha256": input_root_sha256,
            "judgement": {
                "matches_query": True,
                "confidence": 0.88,
                "event_start": None,
                "event_end": None,
                "shot_attempt": True,
                "ball_through_hoop": None,
                "shooter_outside_arc": None,
                "three_point_signal": None,
                "shooter_jersey": None,
                "evidence": "visible dunk",
                "made": None,
                "three_point": None,
            },
        }

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> FakeResponse:
        self.calls.append(("GET", url, None, headers, timeout))
        return FakeResponse(self.health_payload)

    def post(
        self,
        url: str,
        *,
        json: object,
        headers: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        self.calls.append(("POST", url, json, headers, timeout))
        if isinstance(self.judge_payload, dict):
            self.judge_payload = {
                **self.judge_payload,
                "request_id": json["request_id"],  # type: ignore[index]
                "source_sha256": json["expected_sha256"],  # type: ignore[index]
                "byte_size": json["expected_byte_size"],  # type: ignore[index]
            }
        return FakeResponse(self.judge_payload)


def test_client_uses_relative_paths_auth_timeout_and_validates_capability(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nested" / "candidate.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")
    http = RecordingHTTPClient(tmp_path)
    client = QwenWorkerClient(
        endpoint="http://127.0.0.1:8781",
        api_key=TOKEN,
        input_root=tmp_path,
        expected_model_identity=MODEL_IDENTITY,
        timeout=37,
        client=http,
    )

    status = client.status()
    cached_status = client.status()
    judgement = client.judge_video(source, fps=2, max_tokens=320)

    assert status.ready is True
    assert cached_status is status
    assert judgement.matches_query is True
    assert TOKEN not in repr(client.identity)
    assert http.calls[0][1] == "http://127.0.0.1:8781/v1/health"
    method, url, payload, headers, timeout = http.calls[1]
    assert (method, url, timeout) == ("POST", "http://127.0.0.1:8781/v1/judge", 37)
    assert headers == {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }
    assert payload["relative_path"] == "nested/candidate.mp4"  # type: ignore[index]
    assert payload["model_identity"] == MODEL_IDENTITY  # type: ignore[index]
    assert payload["runtime_identity"] == QWEN_INFERENCE_RUNTIME_IDENTITY  # type: ignore[index]
    assert payload["source_bundle_sha256"] == QWEN_SOURCE_BUNDLE_SHA256  # type: ignore[index]
    assert payload["prompt_protocol_sha256"] == QWEN_PROMPT_PROTOCOL_SHA256  # type: ignore[index]
    assert payload["input_root_sha256"] == client.input_root_sha256  # type: ignore[index]
    assert payload["expected_sha256"] == sha256(b"video").hexdigest()  # type: ignore[index]
    assert payload["expected_byte_size"] == len(b"video")  # type: ignore[index]


def test_client_sends_explicit_generic_native_video_contract(tmp_path: Path) -> None:
    source = tmp_path / "candidate.mp4"
    source.write_bytes(b"video")
    http = RecordingHTTPClient(tmp_path)
    client = QwenWorkerClient(
        endpoint="http://127.0.0.1:8781",
        api_key=TOKEN,
        input_root=tmp_path,
        expected_model_identity=MODEL_IDENTITY,
        client=http,
    )

    judgement = client.judge_video_query(
        source,
        "a person waves",
        fps=1.5,
        max_tokens=256,
    )

    assert judgement.matches_query is True
    payload = http.calls[-1][2]
    assert isinstance(payload, dict)
    assert payload["schema_version"] == QWEN_WORKER_SCHEMA_VERSION
    assert payload["input_kind"] == "video"
    assert payload["prompt_kind"] == "generic_query"
    assert payload["query"] == "a person waves"
    assert payload["fps"] == 1.5
    assert payload["max_tokens"] == 256


@pytest.mark.parametrize(
    "field",
    [
        "source_bundle_sha256",
        "prompt_protocol_sha256",
        "input_root_sha256",
    ],
)
def test_client_fails_closed_on_stale_worker_execution_identity(
    tmp_path: Path,
    field: str,
) -> None:
    source = tmp_path / "candidate.mp4"
    source.write_bytes(b"video")
    http = RecordingHTTPClient(tmp_path)
    client = QwenWorkerClient(
        endpoint="http://127.0.0.1:8781",
        api_key=TOKEN,
        input_root=tmp_path,
        expected_model_identity=MODEL_IDENTITY,
        client=http,
    )
    assert isinstance(http.health_payload, dict)
    http.health_payload = {**http.health_payload, field: "0" * 64}

    assert client.status().ready is False

    assert isinstance(http.judge_payload, dict)
    http.judge_payload = {**http.judge_payload, field: "0" * 64}
    with pytest.raises(RuntimeError, match="response contract violation"):
        client.judge_video(source, fps=2, max_tokens=320)


@pytest.mark.parametrize(
    "field",
    [
        "source_bundle_sha256",
        "prompt_protocol_sha256",
        "input_root_sha256",
    ],
)
def test_worker_rejects_stale_request_execution_identity_before_input_read(
    tmp_path: Path,
    field: str,
) -> None:
    runtime = FakeWorkerRuntime()
    client = _worker_client(tmp_path, runtime)
    request = {**_request(tmp_path, "missing.mp4"), field: "0" * 64}

    response = client.post(
        "/v1/judge",
        json=request,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Qwen worker execution identity does not match"
    }
    assert runtime.calls == []


def test_probe_proves_same_source_root_and_file_without_inference(
    tmp_path: Path,
) -> None:
    runtime = FakeWorkerRuntime()
    marker = tmp_path / "private.marker"
    marker.write_bytes(b"private source-bound marker")
    transport = _worker_client(tmp_path, runtime)
    client = QwenWorkerClient(
        endpoint="http://127.0.0.1",
        api_key=TOKEN,
        input_root=tmp_path,
        expected_model_identity=MODEL_IDENTITY,
        client=transport,
    )

    assert client.probe_source(marker) is True
    assert runtime.calls == []
    assert str(tmp_path) not in json.dumps(client.identity)

    link = tmp_path / "linked.marker"
    link.symlink_to(marker)
    with pytest.raises(RuntimeError, match="probe source is unavailable"):
        client.probe_source(link)


def test_probe_rejects_wrong_source_digest_without_inference(tmp_path: Path) -> None:
    runtime = FakeWorkerRuntime()
    marker = tmp_path / "marker"
    marker.write_bytes(b"source-bound marker")
    transport = _worker_client(tmp_path, runtime)
    request = {
        "schema_version": QWEN_WORKER_SCHEMA_VERSION,
        "request_id": "c" * 32,
        "model_identity": MODEL_IDENTITY,
        "runtime_identity": QWEN_INFERENCE_RUNTIME_IDENTITY,
        "source_bundle_sha256": QWEN_SOURCE_BUNDLE_SHA256,
        "prompt_protocol_sha256": QWEN_PROMPT_PROTOCOL_SHA256,
        "input_root_sha256": qwen_worker_module._input_root_identity(tmp_path),
        "relative_path": "marker",
        "expected_sha256": "0" * 64,
        "expected_byte_size": marker.stat().st_size,
    }

    response = transport.post(
        "/v1/probe",
        json=request,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Qwen worker probe input identity mismatch"
    }
    assert runtime.calls == []


def test_client_fails_closed_on_model_mismatch_and_sanitizes_bad_response(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate.mp4"
    source.write_bytes(b"video")
    http = RecordingHTTPClient(tmp_path)
    client = QwenWorkerClient(
        endpoint="http://127.0.0.1:8781",
        api_key=TOKEN,
        input_root=tmp_path,
        expected_model_identity=MODEL_IDENTITY,
        client=http,
    )
    http.health_payload = {**http.health_payload, "model_identity": "wrong/model"}  # type: ignore[arg-type]

    assert client.status().ready is False

    http.judge_payload = {"private_model_path": "/secret/model"}
    with pytest.raises(RuntimeError, match="Qwen worker response contract violation") as error:
        client.judge_video(source, fps=2, max_tokens=320)
    assert "/secret/model" not in str(error.value)


@pytest.mark.parametrize(
    ("field", "value"),
    (("source_sha256", "0" * 64), ("byte_size", 6)),
)
def test_client_fails_closed_when_worker_echoes_a_different_input_identity(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    source = tmp_path / "candidate.mp4"
    source.write_bytes(b"video")
    http = RecordingHTTPClient(tmp_path)
    client = QwenWorkerClient(
        endpoint="http://127.0.0.1:8781",
        api_key=TOKEN,
        input_root=tmp_path,
        expected_model_identity=MODEL_IDENTITY,
        client=http,
    )
    original_post = http.post

    def changed_post(*args: object, **kwargs: object) -> FakeResponse:
        response = original_post(*args, **kwargs)  # type: ignore[arg-type]
        assert isinstance(response.payload, dict)
        response.payload = {**response.payload, field: value}
        return response

    http.post = changed_post  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="response contract violation"):
        client.judge_video(source, fps=2, max_tokens=320)


@pytest.mark.parametrize(
    ("endpoint", "api_key"),
    [
        ("http://gpu.example:8781", TOKEN),
        ("http://localhost:8781", TOKEN),
        ("http://127.0.0.2:8781", TOKEN),
        ("https://127.0.0.1:8781", TOKEN),
        ("http://127.0.0.1:8781", "short"),
    ],
)
def test_client_rejects_non_loopback_plain_contract_or_weak_key(
    tmp_path: Path,
    endpoint: str,
    api_key: str,
) -> None:
    with pytest.raises(ValueError):
        QwenWorkerClient(
            endpoint=endpoint,
            api_key=api_key,
            input_root=tmp_path,
            expected_model_identity=MODEL_IDENTITY,
        )


def test_client_rejects_input_outside_shared_root(tmp_path: Path) -> None:
    source = tmp_path.parent / "outside.mp4"
    source.write_bytes(b"video")
    client = QwenWorkerClient(
        endpoint="http://127.0.0.1:8781",
        api_key=TOKEN,
        input_root=tmp_path,
        expected_model_identity=MODEL_IDENTITY,
        client=RecordingHTTPClient(tmp_path),
    )

    with pytest.raises(RuntimeError, match="outside configured input root"):
        client.judge_video(source, fps=2, max_tokens=320)


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

    def __enter__(self) -> StreamingResponse:
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

    def __enter__(self) -> StreamingHTTPClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def stream(self, method: str, url: str, **kwargs: object) -> StreamingResponse:
        self.calls.append((method, url, kwargs))
        return self.response


def test_default_transport_ignores_proxy_environment_and_bounds_response_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    valid_health = {
        "schema_version": QWEN_WORKER_SCHEMA_VERSION,
        "status": "ok",
        "model_identity": MODEL_IDENTITY,
        "runtime_identity": QWEN_INFERENCE_RUNTIME_IDENTITY,
        "source_bundle_sha256": QWEN_SOURCE_BUNDLE_SHA256,
        "prompt_protocol_sha256": QWEN_PROMPT_PROTOCOL_SHA256,
        "input_root_sha256": qwen_worker_module._input_root_identity(tmp_path),
        "loaded": False,
        "input_kinds": ["video", "storyboard"],
        "max_input_bytes": 1024,
        "max_query_chars": 500,
        "max_tokens": 1024,
        "max_concurrency": 1,
    }
    native = StreamingHTTPClient(
        StreamingResponse([json.dumps(valid_health).encode("utf-8")])
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
    client = QwenWorkerClient(
        endpoint="http://127.0.0.1:8781",
        api_key=TOKEN,
        input_root=tmp_path,
        expected_model_identity=MODEL_IDENTITY,
    )

    assert client.status().ready is True
    assert options == [{"trust_env": False, "follow_redirects": False}]
    assert native.closed is True
    assert native.calls[0][0:2] == (
        "GET",
        "http://127.0.0.1:8781/v1/health",
    )

    oversized_native = StreamingHTTPClient(
        StreamingResponse([b"x" * (MAX_RESPONSE_BYTES + 1)])
    )
    monkeypatch.setattr(httpx, "Client", lambda **_kwargs: oversized_native)
    oversized = QwenWorkerClient(
        endpoint="http://127.0.0.1:8781",
        api_key=TOKEN,
        input_root=tmp_path,
        expected_model_identity=MODEL_IDENTITY,
    )

    assert oversized.status().ready is False

    declared_oversized_response = StreamingResponse(
        [json.dumps(valid_health).encode("utf-8")],
        headers={"Content-Length": str(MAX_RESPONSE_BYTES + 1)},
    )
    declared_oversized_native = StreamingHTTPClient(
        declared_oversized_response
    )
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **_kwargs: declared_oversized_native,
    )
    declared_oversized = QwenWorkerClient(
        endpoint="http://127.0.0.1:8781",
        api_key=TOKEN,
        input_root=tmp_path,
        expected_model_identity=MODEL_IDENTITY,
    )

    assert declared_oversized.status().ready is False
    assert declared_oversized_response.iterated is False


def test_mlx_runtime_stays_lazy_and_preserves_all_prompt_input_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    video = tmp_path / "candidate.mp4"
    storyboard = tmp_path / "candidate.jpg"
    video.write_bytes(b"video")
    storyboard.write_bytes(b"image")
    calls: list[tuple[str, object]] = []
    fake_model = SimpleNamespace(
        config=SimpleNamespace(),
        language_model=SimpleNamespace(
            _position_ids=None,
            _rope_deltas=None,
        ),
    )
    fake_processor = SimpleNamespace(tokenizer=object())
    grammars: list[object] = []

    def build_grammar(tokenizer, schema):  # type: ignore[no-untyped-def]
        assert tokenizer is fake_processor.tokenizer
        calls.append(("grammar", schema))
        grammar = object()
        grammars.append(grammar)
        return grammar

    monkeypatch.setitem(
        sys.modules, "mlx_vlm.structured",
        SimpleNamespace(build_json_schema_logits_processor=build_grammar),
    )

    def load(reference: str):  # type: ignore[no-untyped-def]
        calls.append(("load", reference))
        return fake_model, fake_processor

    def apply_chat_template(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(("template", {**kwargs, "prompts": args[2]}))
        return args[2][0]

    def generate(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(("generate", kwargs))
        if "video" in kwargs and "User query" not in args[2]:
            return SimpleNamespace(
                text=_generated_response("basketball_facts"), finish_reason="stop",
            )
        return SimpleNamespace(
            text=_generated_response(), finish_reason="stop",
        )

    monkeypatch.setattr(
        qwen_worker_module.importlib.util,
        "find_spec",
        lambda _name: object(),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        SimpleNamespace(
            load=load,
            apply_chat_template=apply_chat_template,
            generate=generate,
        ),
    )
    fake_mlx = SimpleNamespace(
        synchronize=lambda: None,
        clear_cache=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_mlx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx)
    runtime = MLXQwenWorkerRuntime(str(snapshot), None)

    assert runtime.available is True
    assert runtime.loaded is False
    video_result = runtime.judge(
        QwenJudgeRequest.model_validate(_request(tmp_path)),
        video,
    )
    storyboard_result = runtime.judge(
        QwenJudgeRequest.model_validate(
            {
                **_request(tmp_path, "candidate.jpg"),
                "input_kind": "storyboard",
                "prompt_kind": "generic_query",
                "query": "visible dunk",
                "fps": None,
            }
        ),
        storyboard,
    )
    generic_video_result = runtime.judge(
        QwenJudgeRequest.model_validate(
            {
                **_request(tmp_path),
                "prompt_kind": "generic_query",
                "query": "visible dunk",
            }
        ),
        video,
    )

    assert runtime.loaded is True
    assert video_result.ball_through_hoop is True
    assert storyboard_result.matches_query is True
    assert generic_video_result.matches_query is True
    assert [name for name, _ in calls].count("load") == 1
    video_generates = [
        payload for name, payload in calls if name == "generate" and "video" in payload
    ]
    image_generate = next(
        payload for name, payload in calls if name == "generate" and "image" in payload
    )
    assert len(video_generates) == 2
    assert all(payload["fps"] == 2.0 for payload in video_generates)
    assert image_generate["image"] == [str(storyboard)]
    template_prompts = [
        payload["prompts"][0]
        for name, payload in calls
        if name == "template" and "prompts" in payload
    ]
    assert any("visible dunk" in prompt for prompt in template_prompts)
    assert len(grammars) == 3
    assert len({id(grammar) for grammar in grammars}) == 3
    for index, payload in enumerate(item for name, item in calls if name == "generate"):
        assert payload["logits_processors"] == [grammars[index]]
        assert payload["max_tokens"] == 320
        assert payload["temperature"] == 0.0
        assert payload["enable_thinking"] is False
    schemas = [payload for name, payload in calls if name == "grammar"]
    assert "ball_through_hoop" in schemas[0]["required"]
    assert "matches_query" not in schemas[0]["required"]
    assert "matches_query" in schemas[1]["required"]
    assert schemas[1] == schemas[2]


def _mlx_request_lifecycle_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    generate_error: Exception | None = None,
    clear_cache_error: Exception | None = None,
    finish_reason: str | None = "stop",
    generation_text: object = None,
    grammar_error: Exception | None = None,
) -> tuple[MLXQwenWorkerRuntime, object, list[str], list[object]]:
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    events: list[str] = []
    request_references: list[object] = []

    class RequestState:
        pass

    class GenerationResult:
        text = generation_text if generation_text is not None else _generated_response()

    GenerationResult.finish_reason = finish_reason

    language_model = SimpleNamespace(
        _position_ids=None,
        _rope_deltas=None,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(),
        language_model=language_model,
    )
    processor = object()

    def build_grammar(_tokenizer, _schema):  # type: ignore[no-untyped-def]
        grammar = RequestState()
        request_references.append(ref(grammar))
        if grammar_error is not None:
            raise grammar_error
        return grammar

    monkeypatch.setitem(
        sys.modules, "mlx_vlm.structured",
        SimpleNamespace(build_json_schema_logits_processor=build_grammar),
    )

    def load(_reference: str):  # type: ignore[no-untyped-def]
        events.append("load")
        return model, processor

    def apply_chat_template(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return "prompt"

    def generate(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        position_ids = RequestState()
        rope_deltas = RequestState()
        language_model._position_ids = position_ids
        language_model._rope_deltas = rope_deltas
        request_references.extend((ref(position_ids), ref(rope_deltas)))
        events.append("generate")
        if generate_error is not None:
            failed_local = RequestState()
            request_references.append(ref(failed_local))
            raise generate_error
        result = GenerationResult()
        request_references.append(ref(result))
        return result

    def synchronize() -> None:
        events.append("synchronize")

    def clear_cache() -> None:
        assert language_model._position_ids is None
        assert language_model._rope_deltas is None
        if "generate" in events:
            assert request_references
        assert all(reference() is None for reference in request_references)
        events.append("clear_cache")
        if clear_cache_error is not None:
            raise clear_cache_error

    monkeypatch.setattr(
        qwen_worker_module.importlib.util,
        "find_spec",
        lambda _name: object(),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        SimpleNamespace(
            load=load,
            apply_chat_template=apply_chat_template,
            generate=generate,
        ),
    )
    fake_mlx = SimpleNamespace(
        synchronize=synchronize,
        clear_cache=clear_cache,
    )
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_mlx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx)
    monkeypatch.setattr(
        qwen_worker_module.gc,
        "collect",
        lambda: events.append("gc"),
    )
    return MLXQwenWorkerRuntime(str(snapshot), None), model, events, request_references


@pytest.mark.parametrize("finish_reason", ["length", None, "unknown"])
def test_mlx_runtime_rejects_unfinished_generation_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finish_reason: str | None,
) -> None:
    runtime, _model, events, _references = _mlx_request_lifecycle_runtime(
        tmp_path, monkeypatch, finish_reason=finish_reason,
    )
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"video")

    with pytest.raises(RuntimeError, match="completion"):
        runtime.judge(QwenJudgeRequest.model_validate(_request(tmp_path)), video)

    assert events == ["load", "generate", "synchronize", "gc", "clear_cache"]
    assert runtime.available is True


def test_mlx_runtime_rejects_invalid_typed_json_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _model, events, _references = _mlx_request_lifecycle_runtime(
        tmp_path, monkeypatch, generation_text="{}",
    )
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"video")

    with pytest.raises(ValueError):
        runtime.judge(QwenJudgeRequest.model_validate(_request(tmp_path)), video)

    assert events == ["load", "generate", "synchronize", "gc", "clear_cache"]
    assert runtime.available is True


@pytest.mark.parametrize("unavailable_package", [False, True])
def test_mlx_runtime_preserves_cleanup_when_structured_decoder_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unavailable_package: bool,
) -> None:
    runtime, _model, events, _references = _mlx_request_lifecycle_runtime(
        tmp_path, monkeypatch, grammar_error=RuntimeError("grammar unavailable"),
    )
    if unavailable_package:
        monkeypatch.setitem(sys.modules, "mlx_vlm.structured", None)
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"video")

    with pytest.raises((RuntimeError, ModuleNotFoundError)):
        runtime.judge(QwenJudgeRequest.model_validate(_request(tmp_path)), video)

    assert events == ["load", "synchronize", "gc", "clear_cache"]
    assert runtime.available is True


def test_mlx_runtime_does_not_coerce_non_text_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _model, events, _references = _mlx_request_lifecycle_runtime(
        tmp_path, monkeypatch, generation_text=17,
    )
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"video")

    with pytest.raises(ValueError, match="invalid text"):
        runtime.judge(QwenJudgeRequest.model_validate(_request(tmp_path)), video)

    assert events == ["load", "generate", "synchronize", "gc", "clear_cache"]


def test_mlx_runtime_clears_request_memory_without_reloading_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, model, events, _references = _mlx_request_lifecycle_runtime(
        tmp_path,
        monkeypatch,
    )
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"video")
    request = QwenJudgeRequest.model_validate(
        {
            **_request(tmp_path),
            "prompt_kind": "generic_query",
            "query": "visible dunk",
        }
    )

    runtime.judge(request, video)
    runtime.judge(request, video)

    assert runtime.loaded is True
    assert runtime.available is True
    assert runtime._model is model
    assert events == [
        "load",
        "generate",
        "synchronize",
        "gc",
        "clear_cache",
        "generate",
        "synchronize",
        "gc",
        "clear_cache",
    ]


def test_mlx_runtime_clears_failed_generation_traceback_before_allocator_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _model, events, _references = _mlx_request_lifecycle_runtime(
        tmp_path,
        monkeypatch,
        generate_error=RuntimeError("generation failed"),
    )
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"video")

    with pytest.raises(RuntimeError, match="generation failed"):
        runtime.judge(
            QwenJudgeRequest.model_validate(_request(tmp_path)),
            video,
        )

    assert events == ["load", "generate", "synchronize", "gc", "clear_cache"]
    assert runtime.loaded is True
    assert runtime.available is True


def test_mlx_runtime_latches_unavailable_when_request_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _model, events, _references = _mlx_request_lifecycle_runtime(
        tmp_path,
        monkeypatch,
        clear_cache_error=RuntimeError("cache stuck"),
    )
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"video")
    request = QwenJudgeRequest.model_validate(_request(tmp_path))

    with pytest.raises(RuntimeError, match="release MLX request memory"):
        runtime.judge(request, video)

    assert runtime.loaded is True
    assert runtime.available is False
    with pytest.raises(RuntimeError, match="runtime is unavailable"):
        runtime.judge(request, video)
    assert events == ["load", "generate", "synchronize", "gc", "clear_cache"]


def test_pinned_worker_model_cannot_be_shadowed_by_a_relative_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = tmp_path / QWEN_VIDEO_MODEL
    shadow.mkdir(parents=True)
    (shadow / "config.json").write_text("{}", encoding="utf-8")
    (shadow / "shadow.safetensors").write_bytes(b"shadow")
    snapshot = tmp_path / "pinned-snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"pinned")
    revision = qwen_worker_module.model_revision(QWEN_VIDEO_MODEL)
    calls: list[tuple[str, str | None, bool]] = []

    def snapshot_download(
        model_name: str,
        *,
        revision: str | None,
        local_files_only: bool,
    ) -> str:
        calls.append((model_name, revision, local_files_only))
        return str(snapshot)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    runtime = MLXQwenWorkerRuntime(QWEN_VIDEO_MODEL, revision)

    assert runtime._resolve_local_reference() == str(snapshot)
    assert calls == [(QWEN_VIDEO_MODEL, revision, True)]


def test_worker_main_hardcodes_loopback_and_does_not_load_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        model_name=QWEN_VIDEO_MODEL,
        input_root=tmp_path,
        api_key=TOKEN,
        max_concurrency=1,
        port=8781,
    )
    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(qwen_worker_module, "QwenWorkerSettings", lambda: settings)
    monkeypatch.setattr(
        qwen_worker_module,
        "attest_qwen_worker_startup",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        qwen_worker_module.MLXQwenWorkerRuntime,
        "_load",
        lambda _self: (_ for _ in ()).throw(AssertionError("model loaded at startup")),
    )
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda app, **kwargs: calls.append((app, kwargs))),
    )

    qwen_worker_module.main()

    assert len(calls) == 1
    assert calls[0][1] == {"host": "127.0.0.1", "port": 8781, "workers": 1}


def test_worker_main_rejects_unpinned_model_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        model_name="organization/unpinned-qwen",
        input_root=tmp_path,
        api_key=TOKEN,
        max_concurrency=1,
        port=8781,
    )
    monkeypatch.setattr(qwen_worker_module, "QwenWorkerSettings", lambda: settings)

    with pytest.raises(RuntimeError, match="pinned revision"):
        qwen_worker_module.main()


def test_worker_input_root_is_derived_from_shared_data_directory(tmp_path: Path) -> None:
    settings = QwenWorkerSettings(
        api_key=TOKEN,
        model_name="organization/qwen",
        data_dir=tmp_path,
        _env_file=None,
    )

    assert settings.input_root == tmp_path / "tmp"


def test_worker_input_root_can_be_explicitly_shared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = tmp_path / "shared-qwen-spool"
    monkeypatch.setenv("VIDEOSCOPE_QWEN_WORKER_INPUT_ROOT", str(shared))
    settings = QwenWorkerSettings(
        api_key=TOKEN,
        model_name="organization/qwen",
        data_dir=tmp_path / "other-data",
        _env_file=None,
    )

    assert settings.input_root == shared
