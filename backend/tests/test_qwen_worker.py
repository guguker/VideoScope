from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import sys
import time

from fastapi.testclient import TestClient
import pytest

import videoscope.providers.qwen_worker as qwen_worker_module
from videoscope.model_manifest import QWEN_VIDEO_MODEL
from videoscope.providers.qwen_video import QwenVideoJudgement
from videoscope.providers.qwen_worker import (
    MAX_RESPONSE_BYTES,
    MLXQwenWorkerRuntime,
    QWEN_INFERENCE_RUNTIME_IDENTITY,
    QWEN_WORKER_SCHEMA_VERSION,
    QwenJudgeRequest,
    QwenWorkerClient,
    QwenWorkerSettings,
    create_qwen_worker_app,
)


TOKEN = "q" * 32
MODEL_IDENTITY = "organization/qwen@" + "a" * 40


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


def _request(relative_path: str = "candidate.mp4") -> dict[str, object]:
    return {
        "schema_version": QWEN_WORKER_SCHEMA_VERSION,
        "request_id": "a" * 32,
        "model_identity": MODEL_IDENTITY,
        "runtime_identity": QWEN_INFERENCE_RUNTIME_IDENTITY,
        "input_kind": "video",
        "relative_path": relative_path,
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

    extra = {**_request(), "unexpected": True}
    assert client.post("/v1/judge", json=extra, headers=headers).status_code == 422

    oversized = client.post("/v1/judge", json=_request(), headers=headers)
    assert oversized.status_code == 413
    assert oversized.json() == {"detail": "Qwen worker input is too large"}
    assert runtime.calls == []


def test_worker_rejects_model_mismatch_before_reading_input(tmp_path: Path) -> None:
    runtime = FakeWorkerRuntime()
    client = _worker_client(tmp_path, runtime)
    request = {**_request("missing.mp4"), "model_identity": "wrong/model"}

    response = client.post(
        "/v1/judge",
        json=request,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Qwen worker model identity does not match"}
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
        json=_request(relative_path),
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
        json=_request(),
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

    response = client.post("/v1/judge", json=_request(), headers=headers)

    assert response.status_code == 200
    assert response.json()["request_id"] == "a" * 32
    assert response.json()["model_identity"] == MODEL_IDENTITY
    assert response.json()["judgement"]["confidence"] == 0.91
    assert runtime.calls[0][1] == source.resolve()

    runtime.error = RuntimeError("secret path: /private/model/token")
    failed = client.post("/v1/judge", json=_request(), headers=headers)

    assert failed.status_code == 503
    assert failed.json() == {"detail": "Qwen worker inference failed"}
    assert "/private/model/token" not in failed.text


class MutatingRuntime(FakeWorkerRuntime):
    def judge(self, request: object, source: Path) -> QwenVideoJudgement:
        result = super().judge(request, source)
        source.write_bytes(b"changed during inference")
        return result


def test_worker_fails_closed_when_input_changes_during_inference(tmp_path: Path) -> None:
    runtime = MutatingRuntime()
    (tmp_path / "candidate.mp4").write_bytes(b"video")
    client = _worker_client(tmp_path, runtime)

    response = client.post(
        "/v1/judge",
        json=_request(),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Qwen worker input changed during inference"}


class RestoringMetadataRuntime(FakeWorkerRuntime):
    def judge(self, request: object, source: Path) -> QwenVideoJudgement:
        result = super().judge(request, source)
        before = source.stat()
        time.sleep(0.002)
        original = source.read_bytes()
        source.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert source.stat().st_ctime_ns != before.st_ctime_ns
        return result


def test_worker_detects_in_place_change_even_when_size_and_mtime_are_restored(
    tmp_path: Path,
) -> None:
    runtime = RestoringMetadataRuntime()
    (tmp_path / "candidate.mp4").write_bytes(b"video")
    client = _worker_client(tmp_path, runtime)

    response = client.post(
        "/v1/judge",
        json=_request(),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Qwen worker input changed during inference"}


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
        first = executor.submit(client.post, "/v1/judge", json=_request(), headers=headers)
        assert runtime.entered.wait(timeout=1)
        second = client.post("/v1/judge", json=_request(), headers=headers)
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
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object | None, dict[str, str], float]] = []
        self.health_payload: object = {
            "schema_version": QWEN_WORKER_SCHEMA_VERSION,
            "status": "ok",
            "model_identity": MODEL_IDENTITY,
            "runtime_identity": QWEN_INFERENCE_RUNTIME_IDENTITY,
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
            self.judge_payload = {**self.judge_payload, "request_id": json["request_id"]}  # type: ignore[index]
        return FakeResponse(self.judge_payload)


def test_client_uses_relative_paths_auth_timeout_and_validates_capability(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nested" / "candidate.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")
    http = RecordingHTTPClient()
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


def test_client_fails_closed_on_model_mismatch_and_sanitizes_bad_response(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate.mp4"
    source.write_bytes(b"video")
    http = RecordingHTTPClient()
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
        client=RecordingHTTPClient(),
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


def test_mlx_runtime_stays_lazy_and_preserves_both_prompt_paths(
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
    fake_model = SimpleNamespace(config=SimpleNamespace())
    fake_processor = object()

    def load(reference: str):  # type: ignore[no-untyped-def]
        calls.append(("load", reference))
        return fake_model, fake_processor

    def apply_chat_template(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(("template", kwargs))
        return "prompt"

    def generate(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(("generate", kwargs))
        if "video" in kwargs:
            return SimpleNamespace(
                text='{"shot_attempt":true,"ball_through_hoop":true}'
            )
        return SimpleNamespace(
            text='{"matches_query":true,"confidence":0.9}'
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
    runtime = MLXQwenWorkerRuntime(str(snapshot), None)

    assert runtime.available is True
    assert runtime.loaded is False
    video_result = runtime.judge(
        QwenJudgeRequest.model_validate(_request()),
        video,
    )
    storyboard_result = runtime.judge(
        QwenJudgeRequest.model_validate(
            {
                **_request("candidate.jpg"),
                "input_kind": "storyboard",
                "prompt_kind": "generic_query",
                "query": "visible dunk",
                "fps": None,
            }
        ),
        storyboard,
    )

    assert runtime.loaded is True
    assert video_result.ball_through_hoop is True
    assert storyboard_result.matches_query is True
    assert [name for name, _ in calls].count("load") == 1
    video_generate = next(
        payload for name, payload in calls if name == "generate" and "video" in payload
    )
    image_generate = next(
        payload for name, payload in calls if name == "generate" and "image" in payload
    )
    assert video_generate["fps"] == 2.0
    assert image_generate["image"] == [str(storyboard)]


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
