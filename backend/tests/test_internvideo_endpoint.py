from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest


SERVER_PATH = Path(__file__).parents[2] / "deploy" / "internvideo" / "server.py"
API_KEY = "test-key-" + ("x" * 48)
REVISION = "a" * 40


def load_server(monkeypatch: pytest.MonkeyPatch, *, api_key: str | None = API_KEY, revision: str | None = REVISION):
    if api_key is None:
        monkeypatch.delenv("INTERNVIDEO_API_KEY", raising=False)
    else:
        monkeypatch.setenv("INTERNVIDEO_API_KEY", api_key)
    if revision is None:
        monkeypatch.delenv("INTERNVIDEO_REVISION", raising=False)
    else:
        monkeypatch.setenv("INTERNVIDEO_REVISION", revision)

    module_name = f"videoscope_internvideo_server_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    try:
        spec.loader.exec_module(module)
    except Exception:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
        raise
    return module


def authorization_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


def valid_payload(module) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return {
        "model": module.MODEL_NAME,
        "task": "temporal_relevance",
        "query": "three-point shot",
        "candidates": [
            {
                "id": "video-1:1.000:2.000",
                "video_id": "video-1",
                "start": 1.0,
                "end": 2.0,
                "frames": [
                    {"timestamp": 1.5, "jpeg_base64": "YWJjZA=="},
                ],
            }
        ],
    }


def test_import_fails_closed_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="INTERNVIDEO_API_KEY"):
        load_server(monkeypatch, api_key=None)


def test_import_rejects_weak_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="at least 32 characters"):
        load_server(monkeypatch, api_key="too-short")


@pytest.mark.parametrize("revision", [None, "main", "abc123"])
def test_import_requires_full_commit_revision(monkeypatch: pytest.MonkeyPatch, revision: str | None) -> None:
    with pytest.raises(RuntimeError, match="INTERNVIDEO_REVISION"):
        load_server(monkeypatch, revision=revision)


def test_authentication_runs_before_request_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_server(monkeypatch)
    client = TestClient(module.app)

    response = client.post("/rerank", content=b"not-json", headers={"Content-Type": "application/json"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}
    assert response.headers["www-authenticate"] == "Bearer"


def test_invalid_bearer_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_server(monkeypatch)
    client = TestClient(module.app)

    response = client.post(
        "/rerank",
        json=valid_payload(module),
        headers={"Authorization": "Bearer wrong-key"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_declared_oversized_body_is_rejected_before_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_server(monkeypatch)
    client = TestClient(module.app)

    response = client.post(
        "/rerank",
        content=b"{}",
        headers={
            **authorization_headers(),
            "Content-Type": "application/json",
            "Content-Length": str(module.MAX_REQUEST_BYTES + 1),
        },
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}


@pytest.mark.asyncio
async def test_streamed_body_limit_does_not_depend_on_content_length(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_server(monkeypatch)
    downstream_called = False

    async def downstream(scope, receive, send):  # type: ignore[no-untyped-def]
        nonlocal downstream_called
        downstream_called = True

    middleware = module.RerankGuardMiddleware(downstream, api_key=API_KEY, max_request_bytes=4)
    messages = iter(
        [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"de", "more_body": False},
        ]
    )
    sent: list[dict[str, object]] = []

    async def receive():  # type: ignore[no-untyped-def]
        return next(messages)

    async def send(message):  # type: ignore[no-untyped-def]
        sent.append(message)

    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": "/rerank",
            "headers": [(b"authorization", f"Bearer {API_KEY}".encode())],
        },
        receive,
        send,
    )

    assert downstream_called is False
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"]) == {"detail": "Request body too large"}


@pytest.mark.parametrize(
    ("mutate", "error_fragment"),
    [
        (lambda payload: payload.update({"unexpected": True}), "extra_forbidden"),
        (
            lambda payload: payload["candidates"][0].update({"start": 2.0, "end": 1.0}),  # type: ignore[index,union-attr]
            "end must be greater than start",
        ),
        (
            lambda payload: payload["candidates"][0]["frames"][0].update({"timestamp": 3.0}),  # type: ignore[index,union-attr]
            "frame timestamps must fall inside the candidate interval",
        ),
        (
            lambda payload: payload["candidates"][0].update({"video_id": "../secret"}),  # type: ignore[index,union-attr]
            "string_pattern_mismatch",
        ),
        (
            lambda payload: payload.update(
                {"candidates": [payload["candidates"][0], payload["candidates"][0]]}  # type: ignore[index]
            ),
            "candidate ids must be unique",
        ),
        (
            lambda payload: payload.update(
                {"candidates": [payload["candidates"][0]] * 5}  # type: ignore[index]
            ),
            "too_long",
        ),
        (
            lambda payload: payload["candidates"][0].update(  # type: ignore[index,union-attr]
                {"frames": payload["candidates"][0]["frames"] * 9}  # type: ignore[index]
            ),
            "too_long",
        ),
    ],
)
def test_request_contract_is_strict_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
    mutate,  # type: ignore[no-untyped-def]
    error_fragment: str,
) -> None:
    module = load_server(monkeypatch)
    client = TestClient(module.app)
    payload = valid_payload(module)
    mutate(payload)

    response = client.post("/rerank", json=payload, headers=authorization_headers())

    assert response.status_code == 422
    assert error_fragment in response.text


def test_validation_errors_do_not_echo_frame_contents(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_server(monkeypatch)
    client = TestClient(module.app)
    payload = valid_payload(module)
    marker = "private-frame-content"
    payload["candidates"][0]["frames"][0]["jpeg_base64"] = marker  # type: ignore[index]

    response = client.post("/rerank", json=payload, headers=authorization_headers())

    assert response.status_code == 422
    assert marker not in response.text


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_request_contract_rejects_non_finite_numbers(
    monkeypatch: pytest.MonkeyPatch,
    value: float,
) -> None:
    module = load_server(monkeypatch)

    with pytest.raises(ValueError, match="finite_number"):
        module.CandidatePayload(
            id="candidate-1",
            video_id="video-1",
            start=value,
            end=2.0,
            frames=[{"timestamp": 1.5, "jpeg_base64": "YWJjZA=="}],
        )


def test_valid_request_uses_typed_response_model(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_server(monkeypatch)
    module.runtime.score = lambda query, candidate: (0.75, f"match: {query}")
    client = TestClient(module.app)

    response = client.post("/rerank", json=valid_payload(module), headers=authorization_headers())

    assert response.status_code == 200
    assert response.json() == {
        "scores": [
            {
                "id": "video-1:1.000:2.000",
                "score": 0.75,
                "reason": "match: three-point shot",
            }
        ]
    }
    response_schema = module.app.openapi()["paths"]["/rerank"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert response_schema["$ref"].endswith("/RerankResponse")


def test_runtime_failures_are_logged_but_not_exposed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = load_server(monkeypatch)

    def fail_score(query, candidate):  # type: ignore[no-untyped-def]
        raise RuntimeError("CUDA device 3 failed with internal path /secret/model")

    module.runtime.score = fail_score
    client = TestClient(module.app, raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR):
        response = client.post("/rerank", json=valid_payload(module), headers=authorization_headers())

    assert response.status_code == 503
    assert response.json() == {"detail": "Reranker temporarily unavailable"}
    assert "/secret/model" not in response.text
    assert "/secret/model" in caplog.text


def test_first_model_load_is_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_server(monkeypatch)
    runtime = module.ModelRuntime()
    load_count = 0

    def build_components():
        nonlocal load_count
        load_count += 1
        return object(), object(), object()

    runtime._build_components = build_components

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: runtime.load(), range(16)))

    assert load_count == 1
    assert all(result == results[0] for result in results)


def test_model_revision_is_passed_to_remote_code_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_server(monkeypatch)
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: object) -> object:
            calls.append(("tokenizer", {"name": name, **kwargs}))
            return object()

    class FakeLoadedModel:
        def eval(self):  # type: ignore[no-untyped-def]
            return self

        def cuda(self):  # type: ignore[no-untyped-def]
            return self

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: object) -> FakeLoadedModel:
            calls.append(("model", {"name": name, **kwargs}))
            return FakeLoadedModel()

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    fake_torch.bfloat16 = object()  # type: ignore[attr-defined]
    fake_transforms = ModuleType("torchvision.transforms")
    fake_transforms.InterpolationMode = SimpleNamespace(BICUBIC="bicubic")  # type: ignore[attr-defined]
    fake_transforms.Resize = lambda *args, **kwargs: ("resize", args, kwargs)  # type: ignore[attr-defined]
    fake_transforms.ToTensor = lambda: "tensor"  # type: ignore[attr-defined]
    fake_transforms.Normalize = lambda **kwargs: ("normalize", kwargs)  # type: ignore[attr-defined]
    fake_transforms.Compose = lambda transforms: tuple(transforms)  # type: ignore[attr-defined]
    fake_torchvision = ModuleType("torchvision")
    fake_torchvision.transforms = fake_transforms  # type: ignore[attr-defined]
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModel = FakeModel  # type: ignore[attr-defined]
    fake_transformers.AutoTokenizer = FakeTokenizer  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", fake_transforms)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    module.ModelRuntime()._build_components()

    assert [call[1]["revision"] for call in calls] == [REVISION, REVISION]


def test_model_output_is_finite_and_has_a_nonblank_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_server(monkeypatch)

    class FakeTensor:
        def to(self, **_kwargs: object):  # type: ignore[no-untyped-def]
            return self

    class FakeModel:
        device = "cuda"

        def __init__(self, response: str) -> None:
            self.response = response

        def chat(self, *_args: object, **_kwargs: object) -> str:
            return self.response

    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = object()  # type: ignore[attr-defined]
    fake_torch.stack = lambda _images: FakeTensor()  # type: ignore[attr-defined]
    fake_torch.inference_mode = nullcontext  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    runtime = module.ModelRuntime()
    runtime.model = FakeModel('{"score": 0.8, "reason": "   "}')
    runtime.tokenizer = object()
    runtime.transform = lambda _image: object()
    runtime._decode_frame = lambda _encoded: object()
    candidate = module.CandidatePayload(
        id="candidate-1",
        video_id="video-1",
        start=1.0,
        end=2.0,
        frames=[{"timestamp": 1.5, "jpeg_base64": "YWJjZA=="}],
    )

    score, reason = runtime.score("query", candidate)

    assert score == 0.8
    assert reason == "InternVideo visual match"

    runtime.model.response = '{"score": NaN, "reason": "bad"}'
    with pytest.raises(ValueError, match="non-finite"):
        runtime.score("query", candidate)
