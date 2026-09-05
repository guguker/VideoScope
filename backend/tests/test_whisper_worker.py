from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import errno
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Event
import textwrap
from types import SimpleNamespace
from weakref import ref

from fastapi.testclient import TestClient
import pytest

import videoscope.providers.whisper as whisper_module
import videoscope.providers.whisper_worker as whisper_worker_module
from videoscope.providers.base import ProviderState
from videoscope.providers.types import TimedText
from videoscope.providers.whisper import (
    WhisperPromptSnapshot,
    WhisperTranscriber,
    snapshot_whisper_prompt,
    snapshot_whisper_prompt_from_content,
)
from videoscope.providers.whisper_worker import (
    MAX_RESPONSE_BYTES,
    MLXWhisperWorkerRuntime,
    WHISPER_DEPENDENCY_IDENTITY,
    WHISPER_INFERENCE_RUNTIME_IDENTITY,
    WHISPER_WORKER_SCHEMA_VERSION,
    WhisperTranscript,
    WhisperSegmentPayload,
    WhisperTranscribeRequest,
    WhisperTranscribeResponse,
    WhisperWordPayload,
    WhisperWorkerClient,
    WhisperWorkerSettings,
    create_whisper_worker_app,
)


TOKEN = "w" * 32
MODEL_IDENTITY = "mlx-community/whisper-test@" + "a" * 40
SOURCE_BYTES = b"safe-video-fixture"
SOURCE_SHA256 = sha256(SOURCE_BYTES).hexdigest()
PROMPT = "Names: Mozgov."
PROMPT_SHA256 = sha256(PROMPT.encode("utf-8")).hexdigest()


class FakeWorkerRuntime:
    model_identity = MODEL_IDENTITY
    runtime_identity = WHISPER_INFERENCE_RUNTIME_IDENTITY
    dependency_identity = WHISPER_DEPENDENCY_IDENTITY

    def __init__(self) -> None:
        self.loaded = False
        self.is_available = True
        self.calls: list[tuple[object, Path, float]] = []
        self.error: Exception | None = None
        self.result = WhisperTranscript(
            language="ru",
            segments=(
                TimedText(
                    start=1.0,
                    end=2.5,
                    text="точный бросок",
                    confidence=0.91,
                    metadata={
                        "language": "ru",
                        "engine": "mlx-whisper",
                        "words": [
                            {
                                "word": "точный",
                                "start": 1.0,
                                "end": 1.5,
                                "probability": 0.92,
                            },
                            {
                                "word": "бросок",
                                "start": 1.6,
                                "end": 2.5,
                                "probability": 0.9,
                            },
                        ],
                    },
                ),
            ),
        )

    @property
    def available(self) -> bool:
        return self.is_available

    def transcribe(
        self,
        request: object,
        source: Path,
        duration_seconds: float,
    ) -> WhisperTranscript:
        self.calls.append((request, source, duration_seconds))
        if self.error is not None:
            raise self.error
        return self.result


def _source(tmp_path: Path, name: str = "source.mp4") -> Path:
    source = tmp_path / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(SOURCE_BYTES)
    return source


def _request(relative_path: str = "source.mp4") -> dict[str, object]:
    return {
        "schema_version": WHISPER_WORKER_SCHEMA_VERSION,
        "request_id": "b" * 32,
        "model_identity": MODEL_IDENTITY,
        "runtime_identity": WHISPER_INFERENCE_RUNTIME_IDENTITY,
        "dependency_identity": WHISPER_DEPENDENCY_IDENTITY,
        "relative_path": relative_path,
        "source_size": len(SOURCE_BYTES),
        "source_sha256": SOURCE_SHA256,
        "language": "ru",
        "effective_prompt": PROMPT,
        "effective_prompt_sha256": PROMPT_SHA256,
        "glossary_state": "ready",
    }


def _worker_client(
    tmp_path: Path,
    runtime: FakeWorkerRuntime,
    *,
    duration: float = 12.0,
    max_input_bytes: int = 4096,
    max_duration_seconds: float = 3600.0,
    max_concurrency: int = 1,
) -> TestClient:
    work_root = tmp_path / "worker-tmp"
    work_root.mkdir(exist_ok=True)
    app = create_whisper_worker_app(
        runtime=runtime,
        input_root=tmp_path,
        work_root=work_root,
        api_key=TOKEN,
        duration_probe=lambda _source: duration,
        max_input_bytes=max_input_bytes,
        max_duration_seconds=max_duration_seconds,
        max_concurrency=max_concurrency,
    )
    return TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 51000),
    )


def test_worker_rejects_remote_unauthenticated_and_hostile_host(
    tmp_path: Path,
) -> None:
    runtime = FakeWorkerRuntime()
    app = create_whisper_worker_app(
        runtime=runtime,
        input_root=tmp_path,
        work_root=tmp_path,
        api_key=TOKEN,
        duration_probe=lambda _source: 12.0,
    )
    local = TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 51000),
    )
    remote = TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("192.0.2.10", 51000),
    )

    assert local.get("/v1/health").status_code == 401
    assert remote.get(
        "/v1/health", headers={"Authorization": f"Bearer {TOKEN}"}
    ).status_code == 403
    hostile = local.get(
        "/v1/health",
        headers={"Authorization": f"Bearer {TOKEN}", "Host": "attacker.invalid"},
    )
    assert hostile.status_code == 400


def test_worker_lifespan_removes_private_request_directory(tmp_path: Path) -> None:
    runtime = FakeWorkerRuntime()
    work_root = tmp_path / "work"
    work_root.mkdir()
    app = create_whisper_worker_app(
        runtime=runtime,
        input_root=tmp_path,
        work_root=work_root,
        api_key=TOKEN,
        duration_probe=lambda _source: 12.0,
    )
    temporary_root = Path(app.state.whisper_temporary_root.name)

    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 51000),
    ):
        assert temporary_root.is_dir()

    assert temporary_root.exists() is False


def test_worker_work_root_is_single_instance_and_clean_shutdown_releases_lease(
    tmp_path: Path,
) -> None:
    runtime = FakeWorkerRuntime()
    work_root = tmp_path / "work"
    work_root.mkdir()
    first = create_whisper_worker_app(
        runtime=runtime,
        input_root=tmp_path,
        work_root=work_root,
        api_key=TOKEN,
        duration_probe=lambda _source: 12.0,
    )
    root_lease_descriptor = first.state.whisper_work_root_lease.descriptor
    instance_lease_descriptor = first.state.whisper_temporary_root.lease_descriptor

    with pytest.raises(ValueError, match="already in use"):
        create_whisper_worker_app(
            runtime=FakeWorkerRuntime(),
            input_root=tmp_path,
            work_root=work_root,
            api_key=TOKEN,
            duration_probe=lambda _source: 12.0,
        )

    with TestClient(
        first,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 51000),
    ):
        assert os.fstat(root_lease_descriptor).st_nlink == 1
        assert os.fstat(instance_lease_descriptor).st_nlink == 1

    for descriptor in (root_lease_descriptor, instance_lease_descriptor):
        with pytest.raises(OSError) as closed:
            os.fstat(descriptor)
        assert closed.value.errno == errno.EBADF

    replacement = create_whisper_worker_app(
        runtime=FakeWorkerRuntime(),
        input_root=tmp_path,
        work_root=work_root,
        api_key=TOKEN,
        duration_probe=lambda _source: 12.0,
    )
    with TestClient(
        replacement,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 51000),
    ):
        assert Path(replacement.state.whisper_temporary_root.name).is_dir()


def test_worker_startup_removes_only_proven_stale_owned_directories(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "work"
    work_root.mkdir()
    owner_file = ".videoscope-whisper-owner"
    lease_file = ".videoscope-whisper-lease"
    owner_prefix = "videoscope-whisper-worker-owned-v1:"

    def owned_directory(name: str) -> Path:
        candidate = work_root / name
        candidate.mkdir(mode=0o700)
        marker = candidate / owner_file
        marker.write_text(f"{owner_prefix}{name}\n", encoding="ascii")
        marker.chmod(0o600)
        lease = candidate / lease_file
        lease.write_bytes(b"")
        lease.chmod(0o600)
        return candidate

    stale = owned_directory("videoscope-whisper-worker-" + "a" * 32)
    stale_snapshot = stale / "input-dead.mp4"
    stale_snapshot.write_bytes(b"orphaned private bytes")
    stale_snapshot.chmod(0o600)

    locked = owned_directory("videoscope-whisper-worker-" + "b" * 32)
    locked_descriptor = os.open(locked / lease_file, os.O_RDWR)
    fcntl.flock(locked_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    foreign = work_root / ("videoscope-whisper-worker-" + "c" * 32)
    foreign.mkdir()
    (foreign / "do-not-delete").write_text("foreign", encoding="utf-8")

    forged = owned_directory("videoscope-whisper-worker-" + "d" * 32)
    (forged / owner_file).write_text("not-owned\n", encoding="ascii")

    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    symlink = work_root / ("videoscope-whisper-worker-" + "e" * 32)
    symlink.symlink_to(symlink_target, target_is_directory=True)
    special = work_root / ("videoscope-whisper-worker-" + "f" * 32)
    os.mkfifo(special, mode=0o600)

    try:
        app = create_whisper_worker_app(
            runtime=FakeWorkerRuntime(),
            input_root=tmp_path,
            work_root=work_root,
            api_key=TOKEN,
            duration_probe=lambda _source: 12.0,
        )

        assert stale.exists() is False
        assert locked.is_dir()
        assert foreign.is_dir()
        assert forged.is_dir()
        assert symlink.is_symlink()
        assert symlink_target.is_dir()
        assert special.exists()

        with TestClient(
            app,
            base_url="http://127.0.0.1",
            client=("127.0.0.1", 51000),
        ):
            pass
    finally:
        fcntl.flock(locked_descriptor, fcntl.LOCK_UN)
        os.close(locked_descriptor)


def test_worker_fails_closed_on_unbounded_work_root_and_releases_lease(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "work"
    work_root.mkdir()
    for index in range(whisper_worker_module._MAX_WORK_ROOT_ENTRIES):
        (work_root / f"foreign-{index}").touch(mode=0o600)

    with pytest.raises(ValueError, match="cannot be inspected"):
        create_whisper_worker_app(
            runtime=FakeWorkerRuntime(),
            input_root=tmp_path,
            work_root=work_root,
            api_key=TOKEN,
            duration_probe=lambda _source: 12.0,
        )
    assert not tuple(work_root.glob("videoscope-whisper-worker-*"))

    (work_root / "foreign-0").unlink()
    replacement = create_whisper_worker_app(
        runtime=FakeWorkerRuntime(),
        input_root=tmp_path,
        work_root=work_root,
        api_key=TOKEN,
        duration_probe=lambda _source: 12.0,
    )
    with TestClient(
        replacement,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 51000),
    ):
        pass


def test_worker_recovers_private_directory_orphaned_by_sigkill(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    work_root.mkdir()
    orphan_record = tmp_path / "orphan-path"
    script = textwrap.dedent(
        """
        import os
        from pathlib import Path
        import signal
        import sys

        from videoscope.providers.whisper_worker import (
            WHISPER_DEPENDENCY_IDENTITY,
            WHISPER_INFERENCE_RUNTIME_IDENTITY,
            create_whisper_worker_app,
        )

        class Runtime:
            model_identity = "mlx-community/whisper-test@" + "a" * 40
            runtime_identity = WHISPER_INFERENCE_RUNTIME_IDENTITY
            dependency_identity = WHISPER_DEPENDENCY_IDENTITY
            available = True
            loaded = False

        root = Path(sys.argv[1])
        app = create_whisper_worker_app(
            runtime=Runtime(),
            input_root=root.parent,
            work_root=root,
            api_key="w" * 32,
            duration_probe=lambda _source: 12.0,
        )
        record = Path(sys.argv[2])
        record.write_text(app.state.whisper_temporary_root.name, encoding="utf-8")
        with record.open("rb") as stream:
            os.fsync(stream.fileno())
        os.kill(os.getpid(), signal.SIGKILL)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(work_root), str(orphan_record)],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    assert completed.returncode == -signal.SIGKILL
    orphan = Path(orphan_record.read_text(encoding="utf-8"))
    assert orphan.is_dir()

    replacement = create_whisper_worker_app(
        runtime=FakeWorkerRuntime(),
        input_root=tmp_path,
        work_root=work_root,
        api_key=TOKEN,
        duration_probe=lambda _source: 12.0,
    )
    assert orphan.exists() is False
    with TestClient(
        replacement,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 51000),
    ):
        pass


def test_worker_retains_original_input_root_identity_until_shutdown(
    tmp_path: Path,
) -> None:
    runtime = FakeWorkerRuntime()
    input_root = tmp_path / "input"
    work_root = tmp_path / "work"
    input_root.mkdir()
    work_root.mkdir()
    _source(input_root, "nested/source.mp4")
    app = create_whisper_worker_app(
        runtime=runtime,
        input_root=input_root,
        work_root=work_root,
        api_key=TOKEN,
        duration_probe=lambda _source: 12.0,
        max_input_bytes=4096,
    )
    retained_descriptor = app.state.whisper_input_root.descriptor

    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 51000),
    ) as client:
        original_root = tmp_path / "original-input"
        input_root.rename(original_root)
        (input_root / "nested").mkdir(parents=True)
        (input_root / "nested/source.mp4").write_bytes(b"redirected-fixture")

        response = client.post(
            "/v1/transcribe",
            json=_request("nested/source.mp4"),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

        assert response.status_code == 200
        assert len(runtime.calls) == 1
        assert os.fstat(retained_descriptor).st_ino == original_root.stat().st_ino

    with pytest.raises(OSError) as closed:
        os.fstat(retained_descriptor)
    assert closed.value.errno == errno.EBADF


def test_snapshot_cleanup_error_is_contained_and_releases_worker_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = FakeWorkerRuntime()
    _source(tmp_path)
    work_root = tmp_path / "work"
    work_root.mkdir()
    app = create_whisper_worker_app(
        runtime=runtime,
        input_root=tmp_path,
        work_root=work_root,
        api_key=TOKEN,
        duration_probe=lambda _source: 12.0,
        max_input_bytes=4096,
    )
    temporary_root = Path(app.state.whisper_temporary_root.name)
    original_unlink = Path.unlink
    failed = False

    def fail_first_snapshot_unlink(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal failed
        if path.name.startswith("input-") and not failed:
            failed = True
            raise OSError("simulated cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_first_snapshot_unlink)
    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 51000),
    ) as client:
        first = client.post(
            "/v1/transcribe",
            json=_request(),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        second = client.post(
            "/v1/transcribe",
            json=_request(),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

        assert first.status_code == 200
        assert second.status_code == 200

    assert temporary_root.exists() is False


def test_worker_caps_body_before_json_parsing(tmp_path: Path) -> None:
    runtime = FakeWorkerRuntime()
    app = create_whisper_worker_app(
        runtime=runtime,
        input_root=tmp_path,
        work_root=tmp_path,
        api_key=TOKEN,
        duration_probe=lambda _source: 12.0,
        max_request_bytes=32,
    )
    client = TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 51000),
    )

    response = client.post(
        "/v1/transcribe",
        content=b"{" + (b"x" * 64) + b"}",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}
    assert runtime.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_identity", "not-pinned"),
        ("relative_path", "source\\evil.mp4"),
        ("language", "RUSSIAN"),
        ("glossary_state", "invented"),
    ],
)
def test_request_dto_rejects_invalid_bounded_contract_fields(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError):
        WhisperTranscribeRequest.model_validate({**_request(), field: value})


def test_response_dtos_reject_invalid_ordering_language_and_state() -> None:
    with pytest.raises(ValueError):
        WhisperWordPayload(text="word", start=2.0, end=1.0, confidence=0.5)
    with pytest.raises(ValueError):
        WhisperSegmentPayload(
            start=1.0,
            end=2.0,
            text="segment",
            confidence=0.5,
            words=[
                WhisperWordPayload(text="word", start=0.5, end=1.5, confidence=0.5)
            ],
        )
    response = _response()
    with pytest.raises(ValueError):
        WhisperTranscribeResponse.model_validate(
            {**response, "detected_language": "auto"}
        )
    with pytest.raises(ValueError):
        WhisperTranscribeResponse.model_validate(
            {**response, "glossary_state": "invented"}
        )
    duplicate_segment = dict(response["segments"][0])  # type: ignore[index]
    with pytest.raises(ValueError, match="ordered"):
        WhisperTranscribeResponse.model_validate(
            {**response, "segments": [duplicate_segment, duplicate_segment]}
        )
    duplicate_word = dict(duplicate_segment["words"][0])  # type: ignore[index]
    with pytest.raises(ValueError, match="ordered"):
        WhisperSegmentPayload.model_validate(
            {**duplicate_segment, "words": [duplicate_word, duplicate_word]}
        )
    overlapping_segment = {
        **duplicate_segment,
        "start": duplicate_segment["end"] - 0.1,
        "end": duplicate_segment["end"] + 0.5,
    }
    with pytest.raises(ValueError, match="ordered"):
        WhisperTranscribeResponse.model_validate(
            {**response, "segments": [duplicate_segment, overlapping_segment]}
        )
    overlapping_word = {
        **duplicate_word,
        "start": duplicate_word["end"] - 0.1,
        "end": duplicate_word["end"] + 0.1,
    }
    with pytest.raises(ValueError, match="ordered"):
        WhisperSegmentPayload.model_validate(
            {**duplicate_segment, "words": [duplicate_word, overlapping_word]}
        )


def test_health_exposes_exact_identity_and_limits_without_loading_model(
    tmp_path: Path,
) -> None:
    runtime = FakeWorkerRuntime()
    client = _worker_client(
        tmp_path,
        runtime,
        max_input_bytes=4096,
        max_duration_seconds=900.0,
        max_concurrency=1,
    )

    response = client.get(
        "/v1/health", headers={"Authorization": f"Bearer {TOKEN}"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": WHISPER_WORKER_SCHEMA_VERSION,
        "status": "ok",
        "model_identity": MODEL_IDENTITY,
        "runtime_identity": WHISPER_INFERENCE_RUNTIME_IDENTITY,
        "dependency_identity": WHISPER_DEPENDENCY_IDENTITY,
        "loaded": False,
        "max_input_bytes": 4096,
        "max_duration_seconds": 900.0,
        "max_request_bytes": 64 * 1024,
        "max_response_bytes": MAX_RESPONSE_BYTES,
        "max_concurrency": 1,
    }

    runtime.is_available = False
    unavailable = client.get(
        "/v1/health", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert unavailable.json()["status"] == "unavailable"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_identity", "wrong/model@" + "a" * 40),
        ("runtime_identity", "mlx-whisper==0.0.0"),
        ("dependency_identity", "whisper-deps-v1:sha256:" + "0" * 64),
        ("effective_prompt_sha256", "0" * 64),
    ],
)
def test_worker_rejects_identity_or_prompt_mismatch_before_inference(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    runtime = FakeWorkerRuntime()
    _source(tmp_path)
    client = _worker_client(tmp_path, runtime)

    response = client.post(
        "/v1/transcribe",
        json={**_request(), field: value},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code in {409, 422}
    assert runtime.calls == []


@pytest.mark.parametrize(
    "relative_path",
    ["../source.mp4", "/tmp/source.mp4", "nested//source.mp4", "source.txt"],
)
def test_worker_rejects_unsafe_or_unsupported_paths(
    tmp_path: Path,
    relative_path: str,
) -> None:
    runtime = FakeWorkerRuntime()
    (tmp_path / "source.txt").write_bytes(SOURCE_BYTES)
    client = _worker_client(tmp_path, runtime)

    response = client.post(
        "/v1/transcribe",
        json=_request(relative_path),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code in {400, 422}
    assert runtime.calls == []


def test_worker_rejects_symlink_special_or_wrong_source_contract(
    tmp_path: Path,
) -> None:
    runtime = FakeWorkerRuntime()
    real = _source(tmp_path, "real.mp4")
    (tmp_path / "source.mp4").symlink_to(real)
    client = _worker_client(tmp_path, runtime)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    symlink = client.post("/v1/transcribe", json=_request(), headers=headers)
    assert symlink.status_code == 400
    assert symlink.json() == {"detail": "Invalid Whisper worker input"}

    (tmp_path / "source.mp4").unlink()
    _source(tmp_path)
    wrong_hash = client.post(
        "/v1/transcribe",
        json={**_request(), "source_sha256": "0" * 64},
        headers=headers,
    )
    wrong_size = client.post(
        "/v1/transcribe",
        json={**_request(), "source_size": len(SOURCE_BYTES) + 1},
        headers=headers,
    )
    assert wrong_hash.status_code == 409
    assert wrong_size.status_code == 409
    assert runtime.calls == []


def test_worker_bounds_actual_media_duration_before_inference(tmp_path: Path) -> None:
    runtime = FakeWorkerRuntime()
    _source(tmp_path)
    client = _worker_client(
        tmp_path,
        runtime,
        duration=901.0,
        max_duration_seconds=900.0,
    )

    response = client.post(
        "/v1/transcribe",
        json=_request(),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Whisper worker input duration is too large"}
    assert runtime.calls == []


def test_worker_fails_closed_when_explicit_work_root_lacks_reserved_space(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = FakeWorkerRuntime()
    _source(tmp_path)
    client = _worker_client(tmp_path, runtime)
    monkeypatch.setattr(
        whisper_worker_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )

    response = client.post(
        "/v1/transcribe",
        json=_request(),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Whisper worker storage is unavailable"}
    assert runtime.calls == []
    assert list((tmp_path / "worker-tmp").rglob("input-*")) == []


def test_worker_cleans_partial_snapshot_when_work_root_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = FakeWorkerRuntime()
    _source(tmp_path)
    client = _worker_client(tmp_path, runtime)

    def fail_write(_descriptor: int, _chunk: object) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(whisper_worker_module.os, "write", fail_write)
    response = client.post(
        "/v1/transcribe",
        json=_request(),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Whisper worker storage is unavailable"}
    assert runtime.calls == []
    assert list((tmp_path / "worker-tmp").rglob("input-*")) == []


def test_worker_rejects_symlink_ancestor_and_insecure_work_root(
    tmp_path: Path,
) -> None:
    runtime = FakeWorkerRuntime()
    real_parent = tmp_path / "real-parent"
    input_root = real_parent / "media"
    work_root = real_parent / "work"
    input_root.mkdir(parents=True)
    work_root.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        create_whisper_worker_app(
            runtime=runtime,
            input_root=linked_parent / "media",
            work_root=work_root,
            api_key=TOKEN,
        )

    work_root.chmod(0o777)
    with pytest.raises(ValueError, match="permissions"):
        create_whisper_worker_app(
            runtime=runtime,
            input_root=input_root,
            work_root=work_root,
            api_key=TOKEN,
        )


def test_source_snapshot_primitives_never_follow_symlink_ancestors(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-source-parent"
    real_parent.mkdir()
    source = _source(real_parent)
    linked_parent = tmp_path / "linked-source-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    linked_source = linked_parent / source.name
    work_root = tmp_path / "worker-snapshots"
    work_root.mkdir()

    with pytest.raises(whisper_worker_module._InputRejected):
        whisper_worker_module._inspect_regular_source(
            linked_source,
            max_input_bytes=4096,
        )
    with pytest.raises(whisper_worker_module._InputRejected):
        whisper_worker_module._copy_source_snapshot(
            linked_source,
            max_input_bytes=4096,
            temp_root=work_root,
        )
    assert list(work_root.iterdir()) == []


def test_worker_detects_same_size_source_rewrite_with_restored_mtime(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    original_stat = source.stat()
    runtime = FakeWorkerRuntime()
    original = runtime.transcribe

    def rewrite(request: object, path: Path, duration: float) -> WhisperTranscript:
        result = original(request, path, duration)
        source.write_bytes(b"x" * len(SOURCE_BYTES))
        os.utime(
            source,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        return result

    runtime.transcribe = rewrite  # type: ignore[method-assign]
    response = _worker_client(tmp_path, runtime).post(
        "/v1/transcribe",
        json=_request(),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Whisper worker input changed during inference"}


def test_worker_enforces_exact_serialized_response_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _source(tmp_path)
    runtime = FakeWorkerRuntime()
    client = _worker_client(tmp_path, runtime)
    monkeypatch.setattr(whisper_worker_module, "MAX_RESPONSE_BYTES", 128)

    response = client.post(
        "/v1/transcribe",
        json=_request(),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Whisper worker produced invalid output"}

    runtime.result = WhisperTranscript(
        language="ru",
        segments=(object(),),  # type: ignore[arg-type]
    )
    malformed = client.post(
        "/v1/transcribe",
        json=_request(),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert malformed.status_code == 503
    assert malformed.json() == {"detail": "Whisper worker produced invalid output"}


def test_worker_transcribes_and_returns_strict_bounded_payload(tmp_path: Path) -> None:
    runtime = FakeWorkerRuntime()
    source = _source(tmp_path)
    client = _worker_client(tmp_path, runtime)

    response = client.post(
        "/v1/transcribe",
        json=_request(),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": WHISPER_WORKER_SCHEMA_VERSION,
        "request_id": "b" * 32,
        "model_identity": MODEL_IDENTITY,
        "runtime_identity": WHISPER_INFERENCE_RUNTIME_IDENTITY,
        "dependency_identity": WHISPER_DEPENDENCY_IDENTITY,
        "source_size": len(SOURCE_BYTES),
        "source_sha256": SOURCE_SHA256,
        "effective_prompt_sha256": PROMPT_SHA256,
        "glossary_state": "ready",
        "duration_seconds": 12.0,
        "detected_language": "ru",
        "segments": [
            {
                "start": 1.0,
                "end": 2.5,
                "text": "точный бросок",
                "confidence": 0.91,
                "words": [
                    {"text": "точный", "start": 1.0, "end": 1.5, "confidence": 0.92},
                    {"text": "бросок", "start": 1.6, "end": 2.5, "confidence": 0.9},
                ],
            }
        ],
    }
    assert len(runtime.calls) == 1
    sent_request, inference_source, duration = runtime.calls[0]
    assert getattr(sent_request, "effective_prompt", None) == PROMPT
    assert getattr(sent_request, "effective_prompt_sha256", None) == PROMPT_SHA256
    assert inference_source != source.resolve()
    assert inference_source.suffix == ".mp4"
    assert inference_source.exists() is False
    assert duration == 12.0


def test_worker_detects_source_mutation_and_sanitizes_runtime_errors(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    runtime = FakeWorkerRuntime()
    original = runtime.transcribe

    def mutate(request: object, path: Path, duration: float) -> WhisperTranscript:
        result = original(request, path, duration)
        source.write_bytes(b"mutated")
        return result

    runtime.transcribe = mutate  # type: ignore[method-assign]
    client = _worker_client(tmp_path, runtime)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    changed = client.post("/v1/transcribe", json=_request(), headers=headers)
    assert changed.status_code == 409
    assert changed.json() == {"detail": "Whisper worker input changed during inference"}

    source.write_bytes(SOURCE_BYTES)
    runtime.transcribe = original  # type: ignore[method-assign]
    runtime.error = RuntimeError("/private/model/token=secret")
    failed = client.post("/v1/transcribe", json=_request(), headers=headers)
    assert failed.status_code == 503
    assert failed.json() == {"detail": "Whisper worker inference failed"}
    assert "secret" not in failed.text


def test_worker_rejects_unordered_or_out_of_duration_runtime_output(
    tmp_path: Path,
) -> None:
    _source(tmp_path)
    runtime = FakeWorkerRuntime()
    runtime.result = WhisperTranscript(
        language="ru",
        segments=(TimedText(11.5, 12.5, "outside", 0.8),),
    )
    client = _worker_client(tmp_path, runtime, duration=12.0)

    response = client.post(
        "/v1/transcribe",
        json=_request(),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Whisper worker produced invalid output"}


def test_worker_serializes_inference(tmp_path: Path) -> None:
    _source(tmp_path)
    runtime = FakeWorkerRuntime()
    entered = Event()
    release = Event()
    original = runtime.transcribe

    def blocking(request: object, source: Path, duration: float) -> WhisperTranscript:
        entered.set()
        assert release.wait(2)
        return original(request, source, duration)

    runtime.transcribe = blocking  # type: ignore[method-assign]
    client = _worker_client(tmp_path, runtime, max_concurrency=1)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.post, "/v1/transcribe", json=_request(), headers=headers)
        assert entered.wait(1)
        second = client.post(
            "/v1/transcribe",
            json={**_request(), "request_id": "c" * 32},
            headers=headers,
        )
        release.set()
        assert first.result(timeout=2).status_code == 200

    assert second.status_code == 429
    assert second.json() == {"detail": "Whisper worker is busy"}


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class FakeHTTPClient:
    def __init__(
        self,
        *,
        health: object,
        response: object,
        health_timeout: float = 5.0,
    ) -> None:
        self.health = health
        self.response = response
        self.health_timeout = health_timeout
        self.posts: list[tuple[str, object, dict[str, str], float]] = []

    def get(self, _url: str, *, headers: dict[str, str], timeout: float) -> FakeResponse:
        assert headers == {"Authorization": f"Bearer {TOKEN}"}
        assert timeout == self.health_timeout
        return FakeResponse(self.health)

    def post(
        self,
        url: str,
        *,
        json: object,
        headers: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        self.posts.append((url, json, headers, timeout))
        return FakeResponse(self.response)


def _health() -> dict[str, object]:
    return {
        "schema_version": WHISPER_WORKER_SCHEMA_VERSION,
        "status": "ok",
        "model_identity": MODEL_IDENTITY,
        "runtime_identity": WHISPER_INFERENCE_RUNTIME_IDENTITY,
        "dependency_identity": WHISPER_DEPENDENCY_IDENTITY,
        "loaded": False,
        "max_input_bytes": 4096,
        "max_duration_seconds": 900.0,
        "max_request_bytes": 64 * 1024,
        "max_response_bytes": MAX_RESPONSE_BYTES,
        "max_concurrency": 1,
    }


def _response(request_id: str = "d" * 32) -> dict[str, object]:
    return {
        "schema_version": WHISPER_WORKER_SCHEMA_VERSION,
        "request_id": request_id,
        "model_identity": MODEL_IDENTITY,
        "runtime_identity": WHISPER_INFERENCE_RUNTIME_IDENTITY,
        "dependency_identity": WHISPER_DEPENDENCY_IDENTITY,
        "source_size": len(SOURCE_BYTES),
        "source_sha256": SOURCE_SHA256,
        "effective_prompt_sha256": PROMPT_SHA256,
        "glossary_state": "ready",
        "duration_seconds": 12.0,
        "detected_language": "ru",
        "segments": [
            {
                "start": 1.0,
                "end": 2.5,
                "text": "точный бросок",
                "confidence": 0.91,
                "words": [
                    {"text": "точный", "start": 1.0, "end": 1.5, "confidence": 0.92}
                ],
            }
        ],
    }


def _inference_client(
    tmp_path: Path,
    transport: FakeHTTPClient,
) -> WhisperWorkerClient:
    return WhisperWorkerClient(
        endpoint="http://127.0.0.1:8784",
        api_key=TOKEN,
        input_root=tmp_path,
        expected_model_identity=MODEL_IDENTITY,
        timeout=30.0,
        client=transport,
        request_id_factory=lambda: "d" * 32,
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:8784",
        "http://localhost:8784",
        "http://[::1]:8784",
        "http://127.0.0.2:8784",
        "http://user@127.0.0.1:8784",
        "http://127.0.0.1:8784/path",
    ],
)
def test_client_accepts_only_exact_ipv4_loopback_origin(
    tmp_path: Path,
    endpoint: str,
) -> None:
    transport = FakeHTTPClient(health=_health(), response=_response())

    with pytest.raises(ValueError, match="127.0.0.1"):
        WhisperWorkerClient(
            endpoint=endpoint,
            api_key=TOKEN,
            input_root=tmp_path,
            expected_model_identity=MODEL_IDENTITY,
            client=transport,
        )


def test_worker_settings_load_dotenv_and_use_reserved_whisper_port(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                f"VIDEOSCOPE_WHISPER_WORKER_API_KEY={TOKEN}",
                f"VIDEOSCOPE_WHISPER_WORKER_INPUT_ROOT={tmp_path}",
                f"VIDEOSCOPE_WHISPER_WORKER_WORK_ROOT={tmp_path}",
                "WHISPER_MODEL=mlx-community/whisper-test",
                f"VIDEOSCOPE_WHISPER_WORKER_MODEL_REVISION={'a' * 40}",
                "VIDEOSCOPE_WHISPER_WORKER_LOG_LEVEL=",
            ]
        ),
        encoding="utf-8",
    )

    settings = WhisperWorkerSettings()  # type: ignore[call-arg]

    assert settings.port == 8784
    assert settings.api_key == TOKEN
    assert TOKEN not in repr(settings)
    assert settings.model_name == "mlx-community/whisper-test"
    assert settings.log_level == "info"


def test_worker_prepares_private_roots_without_following_symlinks(tmp_path: Path) -> None:
    work_root = tmp_path / "data" / "tmp" / "whisper-worker"

    prepared = whisper_worker_module._ensure_real_directory(
        work_root,
        label="work root",
        require_private_owner=True,
    )

    assert prepared == work_root.resolve()
    assert work_root.is_dir()
    assert work_root.stat().st_mode & 0o777 == 0o700

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="safely"):
        whisper_worker_module._ensure_real_directory(
            linked_parent / "worker",
            label="work root",
            require_private_owner=True,
        )


def test_whisper_runtime_identity_is_bound_to_exact_apple_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(whisper_worker_module.sys, "version_info", (3, 12, 13))
    monkeypatch.setattr(whisper_worker_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(whisper_worker_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(whisper_worker_module.platform, "mac_ver", lambda: ("14.0", (), ""))

    assert whisper_worker_module._runtime_platform_is_exact() is True
    assert "python==3.12.13" in WHISPER_INFERENCE_RUNTIME_IDENTITY
    assert "darwin-arm64" in WHISPER_INFERENCE_RUNTIME_IDENTITY
    assert "model-residency=request-scoped-v1" in WHISPER_INFERENCE_RUNTIME_IDENTITY

    monkeypatch.setattr(whisper_worker_module.platform, "machine", lambda: "x86_64")
    assert whisper_worker_module._runtime_platform_is_exact() is False

    monkeypatch.setattr(whisper_worker_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(whisper_worker_module.platform, "mac_ver", lambda: ("13.6", (), ""))
    assert whisper_worker_module._runtime_platform_is_exact() is False


def test_client_status_requires_exact_health_identity(tmp_path: Path) -> None:
    mismatched = {
        **_health(),
        "dependency_identity": "whisper-deps-v1:sha256:" + "0" * 64,
    }
    client = _inference_client(
        tmp_path,
        FakeHTTPClient(health=mismatched, response=_response()),
    )

    status = client.status()

    assert status.ready is False
    assert status.detail == "Whisper worker identity does not match"

    unavailable_client = _inference_client(
        tmp_path,
        FakeHTTPClient(
            health={**_health(), "status": "unavailable"},
            response=_response(),
        ),
    )
    assert unavailable_client.status().detail == "Whisper worker model is unavailable"


def test_client_status_uses_explicit_health_timeout(tmp_path: Path) -> None:
    transport = FakeHTTPClient(
        health=_health(),
        response=_response(),
        health_timeout=17.0,
    )
    client = WhisperWorkerClient(
        endpoint="http://127.0.0.1:8784",
        api_key=TOKEN,
        input_root=tmp_path,
        expected_model_identity=MODEL_IDENTITY,
        timeout=1.0,
        health_timeout=17.0,
        client=transport,
    )

    assert client.status().ready is True


def test_runtime_dependency_identity_checks_every_locked_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = dict(whisper_worker_module.WHISPER_EXACT_DEPENDENCIES)
    monkeypatch.setattr(
        whisper_worker_module.importlib.metadata,
        "version",
        lambda distribution: expected[distribution],
    )
    assert (
        whisper_worker_module._installed_dependency_identity()  # noqa: SLF001
        == WHISPER_DEPENDENCY_IDENTITY
    )

    monkeypatch.setattr(
        whisper_worker_module.importlib.metadata,
        "version",
        lambda distribution: (
            "0.0.0" if distribution == "mlx-whisper" else expected[distribution]
        ),
    )
    assert (
        whisper_worker_module._installed_dependency_identity()  # noqa: SLF001
        == "unavailable"
    )

    def unreadable_metadata(_distribution: str) -> str:
        raise OSError("broken distribution metadata")

    monkeypatch.setattr(
        whisper_worker_module.importlib.metadata,
        "version",
        unreadable_metadata,
    )
    assert (
        whisper_worker_module._installed_dependency_identity()  # noqa: SLF001
        == "unavailable"
    )


def test_client_accepts_explicit_run_bound_prompt_snapshot_and_maps_timed_text(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    effective = "Словарь имён и терминов: Мозгов, Mozgov."
    effective_sha = sha256(effective.encode("utf-8")).hexdigest()
    payload = {
        **_response(),
        "effective_prompt_sha256": effective_sha,
    }
    transport = FakeHTTPClient(health=_health(), response=payload)
    client = _inference_client(tmp_path, transport)
    prompt_snapshot = WhisperPromptSnapshot(
        effective_prompt=effective,
        effective_prompt_sha256=effective_sha,
        glossary_state="ready",
    )

    segments = client.transcribe(
        source,
        language="ru",
        prompt_snapshot=prompt_snapshot,
    )

    assert client.status().ready is True
    assert "glossary_state" not in client.identity
    assert "effective_prompt_sha256" not in client.identity
    assert segments == [
        TimedText(
            start=1.0,
            end=2.5,
            text="точный бросок",
            confidence=0.91,
            metadata={
                "language": "ru",
                "engine": "mlx-whisper",
                "words": [
                    {"word": "точный", "start": 1.0, "end": 1.5, "probability": 0.92}
                ],
            },
        )
    ]
    posted = transport.posts[0][1]
    assert isinstance(posted, dict)
    assert posted["effective_prompt"] == effective
    assert posted["effective_prompt_sha256"] == effective_sha
    assert posted["glossary_state"] == "ready"
    assert posted["source_size"] == len(SOURCE_BYTES)
    assert posted["source_sha256"] == SOURCE_SHA256


def test_client_never_hashes_a_replacement_input_root(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    source = _source(input_root)
    transport = FakeHTTPClient(health=_health(), response=_response())
    client = _inference_client(input_root, transport)
    original_root = tmp_path / "original-input"
    input_root.rename(original_root)
    input_root.mkdir()
    (input_root / source.name).write_bytes(b"redirected-fixture")

    with pytest.raises(RuntimeError, match="unsafe or unavailable"):
        client.transcribe(
            source,
            language="ru",
            prompt_snapshot=WhisperPromptSnapshot(
                effective_prompt=PROMPT,
                effective_prompt_sha256=PROMPT_SHA256,
                glossary_state="ready",
            ),
        )

    assert transport.posts == []


def test_client_rejects_mismatched_response_without_leaking_payload(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    transport = FakeHTTPClient(
        health=_health(),
        response={**_response(), "source_sha256": "0" * 64, "private": "secret"},
    )
    client = _inference_client(tmp_path, transport)

    with pytest.raises(RuntimeError, match="contract violation") as caught:
        client.transcribe(
            source,
            language="ru",
            prompt_snapshot=WhisperPromptSnapshot(
                effective_prompt=PROMPT,
                effective_prompt_sha256=PROMPT_SHA256,
                glossary_state="ready",
            ),
        )

    assert "secret" not in str(caught.value)


def test_provider_can_delegate_to_isolated_client(tmp_path: Path) -> None:
    source = _source(tmp_path)
    transport = FakeHTTPClient(
        health=_health(),
        response={**_response(), "glossary_state": "not_configured"},
    )
    isolated = _inference_client(tmp_path, transport)
    provider = WhisperTranscriber(
        "ignored-when-client-is-set",
        initial_prompt=PROMPT,
        inference_client=isolated,
    )

    assert provider.status().state is ProviderState.READY
    assert provider.transcribe(source)[0].text == "точный бросок"
    assert provider.identity["boundary"]["mode"] == "isolated-worker"


def test_prompt_snapshot_is_live_per_run_and_reads_glossary_without_following_symlinks(
    tmp_path: Path,
) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text('{"Мозгов":["Mozgov"]}', encoding="utf-8")

    first = snapshot_whisper_prompt(None, glossary)
    glossary.write_text('{"Карри":["Curry"]}', encoding="utf-8")
    second = snapshot_whisper_prompt(None, glossary)

    assert first.glossary_state == "ready"
    assert first.effective_prompt != second.effective_prompt
    assert first.effective_prompt_sha256 != second.effective_prompt_sha256

    target = tmp_path / "real.json"
    target.write_text('{"unsafe":[]}', encoding="utf-8")
    glossary.unlink()
    glossary.symlink_to(target)
    unsafe = snapshot_whisper_prompt(None, glossary)
    assert unsafe.glossary_state == "unsafe"
    assert unsafe.effective_prompt is None

    real_parent = tmp_path / "real-glossary-parent"
    real_parent.mkdir()
    nested_glossary = real_parent / "glossary.json"
    nested_glossary.write_text('{"unsafe":[]}', encoding="utf-8")
    linked_parent = tmp_path / "linked-glossary-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    unsafe_ancestor = snapshot_whisper_prompt(
        None,
        linked_parent / "glossary.json",
    )
    assert unsafe_ancestor.glossary_state == "unsafe"
    assert unsafe_ancestor.effective_prompt is None

    deeply_nested = tmp_path / "deep-glossary.json"
    deeply_nested.write_text("[" * 2000 + "0" + "]" * 2000, encoding="utf-8")
    invalid = snapshot_whisper_prompt(None, deeply_nested)
    assert invalid.glossary_state == "invalid"
    assert invalid.effective_prompt is None

    with pytest.raises(ValueError, match="bounded worker contract"):
        snapshot_whisper_prompt("x" * 16_001, None)


def test_prompt_snapshot_from_frozen_content_matches_the_production_normalizer(
    tmp_path: Path,
) -> None:
    raw = '{"Мозгов":["Mozgov"],"Карри":["Curry"]}'.encode()
    glossary = tmp_path / "glossary.json"
    glossary.write_bytes(raw)

    from_file = snapshot_whisper_prompt("initial", glossary)
    from_content = snapshot_whisper_prompt_from_content(
        "initial",
        raw,
        glossary_state="ready",
    )

    assert from_content == from_file
    assert snapshot_whisper_prompt_from_content(
        None,
        None,
        glossary_state="missing",
    ).glossary_state == "missing"
    with pytest.raises(ValueError, match="content and state"):
        snapshot_whisper_prompt_from_content(
            None,
            raw,
            glossary_state="missing",
        )


def test_prompt_snapshot_contains_glossary_read_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text('{"term":[]}', encoding="utf-8")

    def fail_read(_descriptor: int, _count: int) -> bytes:
        raise OSError("simulated read failure")

    monkeypatch.setattr(whisper_module.os, "read", fail_read)
    snapshot = snapshot_whisper_prompt(None, glossary)

    assert snapshot.glossary_state == "unreadable"
    assert snapshot.effective_prompt is None


def test_mlx_runtime_normalizes_and_bounds_untrusted_model_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "weights.safetensors").write_bytes(b"weights")
    captured: dict[str, object] = {}
    module = SimpleNamespace()
    cache_events: list[str] = []

    class ModelHolder:
        model: object | None = object()
        model_path: str | None = "retained-model"

    def synchronize() -> None:
        cache_events.append("synchronize")

    def clear_cache() -> None:
        assert ModelHolder.model is None
        assert ModelHolder.model_path is None
        cache_events.append("clear_cache")

    def transcribe(path: str, **kwargs: object) -> dict[str, object]:
        captured.update({"path": path, **kwargs})
        return {
            "language": "ru",
            "segments": [
                {
                    "start": 1.0,
                    "end": 2.0,
                    "text": "  valid  ",
                    "avg_logprob": -0.2,
                    "words": [
                        {"word": " valid", "start": 1.0, "end": 2.0, "probability": 0.8}
                    ],
                },
            ],
        }

    module.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    monkeypatch.setitem(
        sys.modules,
        "mlx_whisper.transcribe",
        SimpleNamespace(
            ModelHolder=ModelHolder,
            mx=SimpleNamespace(
                synchronize=synchronize,
                clear_cache=clear_cache,
            ),
            transcribe=transcribe,
        ),
    )
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *args, **kwargs: (
            str(snapshot)
            if kwargs == {"revision": "a" * 40, "local_files_only": True}
            else pytest.fail("unexpected model resolution")
        ),
    )
    monkeypatch.setattr(
        whisper_worker_module,
        "_installed_dependency_identity",
        lambda: WHISPER_DEPENDENCY_IDENTITY,
    )
    monkeypatch.setattr(
        whisper_worker_module,
        "_runtime_platform_is_exact",
        lambda: True,
    )
    monkeypatch.setattr(
        whisper_worker_module,
        "WHISPER_REVIEWED_MODEL_ARTIFACTS",
        {
            ("mlx-community/whisper-test", "a" * 40): (
                ("config.json", 2, sha256(b"{}").hexdigest()),
                ("weights.safetensors", 7, sha256(b"weights").hexdigest()),
            )
        },
    )
    runtime = MLXWhisperWorkerRuntime(
        "mlx-community/whisper-test",
        "a" * 40,
    )
    request = SimpleNamespace(language="ru", effective_prompt=PROMPT)

    result = runtime.transcribe(request, tmp_path / "source.mp4", 12.0)

    assert runtime.available is True
    assert runtime.loaded is False
    assert result.language == "ru"
    assert [segment.text for segment in result.segments] == ["valid"]
    assert result.segments[0].metadata["words"] == [
        {"word": "valid", "start": 1.0, "end": 2.0, "probability": 0.8}
    ]
    assert captured["path_or_hf_repo"] == str(snapshot)
    assert captured["initial_prompt"] == PROMPT
    assert captured["word_timestamps"] is True
    assert ModelHolder.model is None
    assert ModelHolder.model_path is None
    assert cache_events == ["synchronize", "clear_cache"]


def _mlx_lifecycle_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result_or_error: object,
    clear_cache_error: Exception | None = None,
) -> tuple[MLXWhisperWorkerRuntime, type[object], list[str]]:
    events: list[str] = []

    class ModelHolder:
        model: object | None = None
        model_path: str | None = None

    def transcribe(_path: str, **_kwargs: object) -> object:
        assert ModelHolder.model is None
        assert ModelHolder.model_path is None
        ModelHolder.model = object()
        ModelHolder.model_path = "loaded-model"
        events.append("transcribe")
        if isinstance(result_or_error, BaseException):
            raise result_or_error
        return result_or_error

    def synchronize() -> None:
        assert ModelHolder.model is not None
        events.append("synchronize")

    def clear_cache() -> None:
        assert ModelHolder.model is None
        assert ModelHolder.model_path is None
        events.append("clear_cache")
        if clear_cache_error is not None:
            raise clear_cache_error

    package = SimpleNamespace(transcribe=transcribe)
    transcribe_module = SimpleNamespace(
        ModelHolder=ModelHolder,
        mx=SimpleNamespace(
            synchronize=synchronize,
            clear_cache=clear_cache,
        ),
        transcribe=transcribe,
    )
    monkeypatch.setitem(sys.modules, "mlx_whisper", package)
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", transcribe_module)
    monkeypatch.setattr(
        whisper_worker_module,
        "_installed_dependency_identity",
        lambda: WHISPER_DEPENDENCY_IDENTITY,
    )
    monkeypatch.setattr(
        whisper_worker_module,
        "_runtime_platform_is_exact",
        lambda: True,
    )
    monkeypatch.setattr(
        whisper_worker_module.gc,
        "collect",
        lambda: events.append("gc"),
    )
    runtime = MLXWhisperWorkerRuntime("mlx-community/whisper-test", "a" * 40)
    monkeypatch.setattr(
        runtime,
        "_resolve_local_reference",
        lambda *, force_hash=False: "/private/tmp/pinned-whisper",
    )
    return runtime, ModelHolder, events


def _valid_mlx_result() -> dict[str, object]:
    return {
        "language": "ru",
        "segments": [
            {
                "start": 1.0,
                "end": 2.0,
                "text": "valid",
                "avg_logprob": -0.2,
                "words": [],
            }
        ],
    }


@pytest.mark.parametrize(
    "result_or_error, expected_error",
    [
        (RuntimeError("inference failed"), "inference failed"),
        ({"language": "ru", "segments": [{"start": 2.0}]}, "mlx-whisper"),
    ],
)
def test_mlx_runtime_releases_cached_model_after_failed_request(
    monkeypatch: pytest.MonkeyPatch,
    result_or_error: object,
    expected_error: str,
) -> None:
    runtime, holder, events = _mlx_lifecycle_runtime(
        monkeypatch,
        result_or_error=result_or_error,
    )

    with pytest.raises(Exception, match=expected_error):
        runtime.transcribe(
            SimpleNamespace(language="ru", effective_prompt=None),
            Path("source.mp4"),
            12.0,
        )

    assert holder.model is None
    assert holder.model_path is None
    assert events == ["transcribe", "synchronize", "gc", "clear_cache"]
    assert runtime.available is True


def test_mlx_runtime_clears_failed_inference_traceback_before_allocator_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    model_references: list[object] = []

    class Model:
        pass

    class ModelHolder:
        model: object | None = None
        model_path: str | None = None

    def transcribe(_path: str, **_kwargs: object) -> object:
        local_model = Model()
        model_references.append(ref(local_model))
        ModelHolder.model = local_model
        ModelHolder.model_path = "loaded-model"
        raise RuntimeError("inference failed with model frame")

    def synchronize() -> None:
        events.append("synchronize")

    def clear_cache() -> None:
        assert ModelHolder.model is None
        assert model_references and model_references[0]() is None
        events.append("clear_cache")

    transcribe_module = SimpleNamespace(
        ModelHolder=ModelHolder,
        mx=SimpleNamespace(
            synchronize=synchronize,
            clear_cache=clear_cache,
        ),
        transcribe=transcribe,
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_whisper",
        SimpleNamespace(transcribe=transcribe),
    )
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", transcribe_module)
    monkeypatch.setattr(
        whisper_worker_module,
        "_installed_dependency_identity",
        lambda: WHISPER_DEPENDENCY_IDENTITY,
    )
    monkeypatch.setattr(
        whisper_worker_module,
        "_runtime_platform_is_exact",
        lambda: True,
    )
    monkeypatch.setattr(
        whisper_worker_module.gc,
        "collect",
        lambda: events.append("gc"),
    )
    runtime = MLXWhisperWorkerRuntime("mlx-community/whisper-test", "a" * 40)
    monkeypatch.setattr(
        runtime,
        "_resolve_local_reference",
        lambda *, force_hash=False: "/private/tmp/pinned-whisper",
    )

    with pytest.raises(RuntimeError, match="inference failed with model frame"):
        runtime.transcribe(
            SimpleNamespace(language="ru", effective_prompt=None),
            Path("source.mp4"),
            12.0,
        )

    assert events == ["synchronize", "gc", "clear_cache"]
    assert ModelHolder.model is None
    assert model_references[0]() is None


def test_mlx_runtime_releases_model_when_post_inference_artifact_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, holder, events = _mlx_lifecycle_runtime(
        monkeypatch,
        result_or_error=_valid_mlx_result(),
    )

    force_hash_calls = 0

    def resolve(*, force_hash: bool = False) -> str:
        nonlocal force_hash_calls
        if force_hash:
            force_hash_calls += 1
        if force_hash_calls == 2:
            raise RuntimeError("artifact changed")
        return "/private/tmp/pinned-whisper"

    monkeypatch.setattr(runtime, "_resolve_local_reference", resolve)

    with pytest.raises(RuntimeError, match="artifact changed"):
        runtime.transcribe(
            SimpleNamespace(language="ru", effective_prompt=None),
            Path("source.mp4"),
            12.0,
        )

    assert holder.model is None
    assert holder.model_path is None
    assert events == ["transcribe", "synchronize", "gc", "clear_cache"]


def test_mlx_runtime_latches_unavailable_when_cache_release_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, holder, events = _mlx_lifecycle_runtime(
        monkeypatch,
        result_or_error=_valid_mlx_result(),
        clear_cache_error=RuntimeError("cache stuck"),
    )

    with pytest.raises(RuntimeError, match="release MLX model memory"):
        runtime.transcribe(
            SimpleNamespace(language="ru", effective_prompt=None),
            Path("source.mp4"),
            12.0,
        )

    assert holder.model is None
    assert holder.model_path is None
    assert events == ["transcribe", "synchronize", "gc", "clear_cache"]
    assert runtime.available is False
    with pytest.raises(RuntimeError, match="runtime identity is unavailable"):
        runtime.transcribe(
            SimpleNamespace(language="ru", effective_prompt=None),
            Path("source.mp4"),
            12.0,
        )


def test_mlx_runtime_never_reuses_model_between_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, holder, events = _mlx_lifecycle_runtime(
        monkeypatch,
        result_or_error=_valid_mlx_result(),
    )
    request = SimpleNamespace(language="ru", effective_prompt=None)

    runtime.transcribe(request, Path("source.mp4"), 12.0)
    runtime.transcribe(request, Path("source.mp4"), 12.0)

    assert holder.model is None
    assert holder.model_path is None
    assert events == [
        "transcribe",
        "synchronize",
        "gc",
        "clear_cache",
        "transcribe",
        "synchronize",
        "gc",
        "clear_cache",
    ]


def test_mlx_runtime_omits_zero_duration_timestamp_quantization() -> None:
    result = whisper_worker_module._normalize_mlx_transcript(
        {
            "language": "ru",
            "segments": [
                {
                    "start": 0.60,
                    "end": 1.96,
                    "text": "названия команд названия.",
                    "avg_logprob": -0.2,
                    "words": [
                        {
                            "word": " названия",
                            "start": 0.60,
                            "end": 1.28,
                            "probability": 0.8,
                        },
                        {
                            "word": " команд",
                            "start": 1.28,
                            "end": 1.28,
                            "probability": 0.7,
                        },
                        {
                            "word": " названия.",
                            "start": 1.96,
                            "end": 1.96,
                            "probability": 0.6,
                        },
                    ],
                }
            ],
        },
        requested_language="ru",
        duration_seconds=3.0,
    )

    assert result.segments[0].start == 0.60
    assert result.segments[0].end == 1.96
    assert result.segments[0].metadata["words"] == [
        {
            "word": "названия",
            "start": 0.60,
            "end": 1.28,
            "probability": 0.8,
        }
    ]


@pytest.mark.parametrize(
    "zero_word",
    [
        {"word": 7, "start": 1.28, "end": 1.28, "probability": 0.7},
        {
            "word": "команд",
            "start": float("nan"),
            "end": 1.28,
            "probability": 0.7,
        },
        {"word": "команд", "start": 2.0, "end": 2.0, "probability": 0.7},
        {"word": "команд", "start": 1.0, "end": 1.0, "probability": 0.7},
        {"word": "команд", "start": 1.4, "end": 1.3, "probability": 0.7},
    ],
)
def test_mlx_runtime_zero_duration_exception_remains_fail_closed(
    zero_word: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="mlx-whisper word"):
        whisper_worker_module._normalize_mlx_transcript(
            {
                "language": "ru",
                "segments": [
                    {
                        "start": 0.60,
                        "end": 1.96,
                        "text": "названия команд",
                        "avg_logprob": -0.2,
                        "words": [
                            {
                                "word": "названия",
                                "start": 0.60,
                                "end": 1.28,
                                "probability": 0.8,
                            },
                            zero_word,
                        ],
                    }
                ],
            },
            requested_language="ru",
            duration_seconds=3.0,
        )


def test_mlx_runtime_rejects_unreviewed_or_mutated_model_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    weights = snapshot / "weights.safetensors"
    weights.write_bytes(b"weights")
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *_args, **_kwargs: str(snapshot),
    )
    monkeypatch.setattr(
        whisper_worker_module,
        "_installed_dependency_identity",
        lambda: WHISPER_DEPENDENCY_IDENTITY,
    )
    monkeypatch.setattr(
        whisper_worker_module,
        "_runtime_platform_is_exact",
        lambda: True,
    )
    runtime = MLXWhisperWorkerRuntime("mlx-community/whisper-test", "a" * 40)
    monkeypatch.setattr(
        runtime,
        "_mlx_runtime_module",
        lambda: SimpleNamespace(),
    )

    assert runtime.available is False

    monkeypatch.setattr(
        whisper_worker_module,
        "WHISPER_REVIEWED_MODEL_ARTIFACTS",
        {
            ("mlx-community/whisper-test", "a" * 40): (
                ("config.json", 2, sha256(b"{}").hexdigest()),
                ("weights.safetensors", 7, sha256(b"weights").hexdigest()),
            )
        },
    )
    assert runtime.available is True

    weights.write_bytes(b"tamper!")
    assert runtime.available is False


@pytest.mark.parametrize(
    "segments",
    [
        [{"start": float("nan"), "end": 3.0, "text": "bad"}],
        [
            {"start": 1.0, "end": 2.0, "text": "first"},
            {"start": 1.5, "end": 3.0, "text": "overlap"},
        ],
        [
            {
                "start": 1.0,
                "end": 2.0,
                "text": "x" * (whisper_worker_module.MAX_SEGMENT_TEXT_CHARS + 1),
            }
        ],
        [
            {
                "start": 1.0,
                "end": 2.0,
                "text": "bad words",
                "words": [
                    {"word": "x", "start": 1.0, "end": 1.5, "probability": 2.0}
                ],
            }
        ],
    ],
)
def test_mlx_runtime_rejects_the_entire_malformed_transcript(
    segments: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError, match="mlx-whisper"):
        whisper_worker_module._normalize_mlx_transcript(
            {"language": "ru", "segments": segments},
            requested_language="ru",
            duration_seconds=12.0,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(language={"unexpected": 1}),
        lambda payload: payload["segments"][0].update(words=False),
        lambda payload: payload["segments"][0].pop("avg_logprob"),
        lambda payload: payload["segments"][0]["words"][0].pop("probability"),
    ],
)
def test_mlx_runtime_rejects_falsy_or_defaulted_output_fields(
    mutation,
) -> None:  # type: ignore[no-untyped-def]
    payload: dict[str, object] = {
        "language": "ru",
        "segments": [
            {
                "start": 1.0,
                "end": 2.0,
                "text": "valid",
                "avg_logprob": -0.2,
                "words": [
                    {
                        "word": "valid",
                        "start": 1.0,
                        "end": 2.0,
                        "probability": 0.8,
                    }
                ],
            }
        ],
    }
    mutation(payload)

    with pytest.raises(ValueError, match="mlx-whisper"):
        whisper_worker_module._normalize_mlx_transcript(
            payload,
            requested_language="ru",
            duration_seconds=12.0,
        )


def test_default_http_transport_disables_proxies_redirects_and_caps_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class StreamResponse:
        headers = {"Content-Length": str(MAX_RESPONSE_BYTES + 1)}

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, **_kwargs: object):  # type: ignore[no-untyped-def]
            yield b"{}"

    class HTTPXClient:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def stream(self, *_args: object, **_kwargs: object) -> StreamResponse:
            return StreamResponse()

    import httpx

    monkeypatch.setattr(httpx, "Client", HTTPXClient)
    transport = whisper_worker_module._BoundedHTTPClient(  # noqa: SLF001
        max_response_bytes=MAX_RESPONSE_BYTES
    )

    with pytest.raises(RuntimeError, match="too large"):
        transport.get("http://127.0.0.1:8784/v1/health", headers={}, timeout=1.0)

    assert observed == {"trust_env": False, "follow_redirects": False}
