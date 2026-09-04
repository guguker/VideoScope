from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import platform
import stat
import subprocess
import sys

import pytest

import videoscope.ml_environment_attestation as attestation
from videoscope.ml_environment_attestation import (
    DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256,
    HostSnapshot,
    MlEnvironmentManifestError,
    RuntimeSnapshot,
    UvSnapshot,
    attest_ml_environment,
    canonical_json_sha256,
    load_ml_environment_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFLINE_ENVIRONMENT = {
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "UV_OFFLINE": "1",
}


def _distribution_identity(distributions: dict[str, str]) -> str:
    return canonical_json_sha256(distributions)


def _write_executable(path: Path, body: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    path.chmod(stat.S_IRUSR | stat.S_IXUSR)
    return path


def _fixture_manifest(
    root: Path,
    *,
    include_vision_environment: bool = False,
) -> tuple[Path, str, dict[str, str]]:
    base_distributions = {"videoscope-backend": "0.1.0", "uv": "0.12.3"}
    qwen_distributions = {"mlx": "0.32.0", "mlx-vlm": "0.6.7"}
    vision_distributions = {"torch": "2.13.0", "transformers": "5.14.1"}

    backend_lock = root / "backend" / "uv.lock"
    vision_lock = root / "workers" / "vision" / "requirements.lock"
    qwen_runtime = root / "workers" / "qwen" / "runtime.lock.json"
    qwen_models = root / "workers" / "qwen" / "model-artifacts.lock.json"
    backend_lock.parent.mkdir(parents=True)
    vision_lock.parent.mkdir(parents=True)
    qwen_runtime.parent.mkdir(parents=True)
    backend_lock.write_bytes(b"frozen-backend-lock")
    vision_lock.write_text(
        "torch==2.13.0 --hash=sha256:" + "a" * 64 + "\n"
        "transformers==5.14.1 --hash=sha256:" + "b" * 64 + "\n",
        encoding="utf-8",
    )
    runtime_payload = {
        "backend_lock_sha256": sha256(backend_lock.read_bytes()).hexdigest(),
        "distributions": qwen_distributions,
        "platform": "aarch64-apple-darwin-macos14plus",
        "python": "3.12.13",
        "schema_version": 1,
    }
    model_payload = {
        "artifacts": [
            {
                "name": "model.safetensors",
                "sha256": "c" * 64,
                "size": 100,
            }
        ],
        "license": "apache-2.0",
        "model": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "revision": "d" * 40,
        "schema_version": 1,
    }
    qwen_runtime.write_text(json.dumps(runtime_payload), encoding="utf-8")
    qwen_models.write_text(json.dumps(model_payload), encoding="utf-8")

    _write_executable(root / ".venv" / "bin" / "python")
    _write_executable(root / ".venv" / "bin" / "uv", b"uv-binary")
    _write_executable(root / ".venv-qwen" / "bin" / "python")
    if include_vision_environment:
        _write_executable(root / ".venv-vision" / "bin" / "python")

    contracts = [
        {
            "id": "backend-lock",
            "kind": "dependency_lock",
            "path": "backend/uv.lock",
            "sha256": sha256(backend_lock.read_bytes()).hexdigest(),
        },
        {
            "id": "vision-lock",
            "kind": "dependency_lock",
            "path": "workers/vision/requirements.lock",
            "sha256": sha256(vision_lock.read_bytes()).hexdigest(),
        },
        {
            "canonical_json_sha256": canonical_json_sha256(runtime_payload),
            "id": "qwen-runtime-manifest",
            "kind": "worker_manifest",
            "path": "workers/qwen/runtime.lock.json",
            "sha256": sha256(qwen_runtime.read_bytes()).hexdigest(),
        },
        {
            "canonical_json_sha256": canonical_json_sha256(model_payload),
            "id": "qwen-model-manifest",
            "kind": "model_manifest",
            "path": "workers/qwen/model-artifacts.lock.json",
            "sha256": sha256(qwen_models.read_bytes()).hexdigest(),
        },
    ]
    manifest = {
        "attestation_id": "phase0-fixture-v1",
        "capabilities": [
            {
                "contract_ids": ["backend-lock"],
                "environment_ids": ["base"],
                "id": "base_search",
                "model_identities": ["xenova/mpnet@" + "e" * 40],
                "runtime_identity": "base-search-runtime-v1",
            },
            {
                "contract_ids": ["vision-lock"],
                "environment_ids": ["vision"],
                "id": "dense_siglip",
                "model_identities": ["google/siglip@" + "f" * 40],
                "runtime_identity": "vision-runtime-v1",
            },
            {
                "contract_ids": [
                    "backend-lock",
                    "qwen-runtime-manifest",
                    "qwen-model-manifest",
                ],
                "environment_ids": ["qwen"],
                "id": "qwen_verification",
                "model_identities": [
                    "mlx-community/Qwen3.5-9B-MLX-4bit@" + "d" * 40
                ],
                "runtime_identity": (
                    "videoscope-qwen-worker-v4|python==3.12.13|"
                    "runtime-manifest-sha256:"
                    + canonical_json_sha256(runtime_payload)
                    + "|model-manifest-sha256:"
                    + canonical_json_sha256(model_payload)
                ),
            },
        ],
        "contracts": contracts,
        "environments": [
            {
                "dependency_contract_id": "backend-lock",
                "directory": ".venv",
                "distribution_source": {
                    "kind": "frozen_identity",
                    "sha256": _distribution_identity(base_distributions),
                },
                "id": "base",
                "python": "3.12.13",
            },
            {
                "dependency_contract_id": "vision-lock",
                "directory": ".venv-vision",
                "distribution_source": {
                    "contract_id": "vision-lock",
                    "kind": "requirements_lock",
                },
                "id": "vision",
                "python": "3.12.13",
            },
            {
                "dependency_contract_id": "backend-lock",
                "directory": ".venv-qwen",
                "distribution_source": {
                    "contract_id": "qwen-runtime-manifest",
                    "field": "distributions",
                    "kind": "json_mapping",
                },
                "id": "qwen",
                "python": "3.12.13",
            },
        ],
        "host": {
            "chip": "Apple M4 Pro",
            "machine": "arm64",
            "memory_gib": 24,
            "minimum_macos_major": 14,
            "system": "Darwin",
        },
        "offline_environment": OFFLINE_ENVIRONMENT,
        "schema_version": 1,
        "uv": {
            "environment_id": "base",
            "path": ".venv/bin/uv",
            "sha256": sha256(b"uv-binary").hexdigest(),
            "version": "0.12.3",
        },
    }
    manifest_path = root / "workers" / "ml-environment.lock.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, sha256(manifest_path.read_bytes()).hexdigest(), {
        "base": _distribution_identity(base_distributions),
        "qwen": _distribution_identity(qwen_distributions),
        "vision": _distribution_identity(vision_distributions),
    }


def _host() -> HostSnapshot:
    return HostSnapshot(
        system="Darwin",
        machine="arm64",
        macos_version="26.6.2",
        chip="Apple M4 Pro",
        memory_gib=24,
    )


def _runtime(distribution_identity: str, *, python: str = "3.12.13") -> RuntimeSnapshot:
    return RuntimeSnapshot(
        implementation="CPython",
        python_version=python,
        system="Darwin",
        machine="arm64",
        macos_version="26.6.2",
        distribution_identity=distribution_identity,
    )


def test_attestation_is_pathless_tokenless_and_reports_absence_as_not_configured(
    tmp_path: Path,
) -> None:
    manifest_path, manifest_sha, identities = _fixture_manifest(tmp_path)
    secret = "hf_secret_value_that_must_not_escape"
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    report = attest_ml_environment(
        tmp_path,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha,
        environ={**OFFLINE_ENVIRONMENT, "HF_TOKEN": secret},
        host_probe=_host,
        runtime_probe=lambda _python, environment_id: _runtime(
            identities[environment_id]
        ),
        uv_probe=lambda _uv: UvSnapshot(
            version="0.12.3",
            sha256=sha256(b"uv-binary").hexdigest(),
        ),
    )

    encoded = json.dumps(report, sort_keys=True)
    environments = {item["id"]: item for item in report["environments"]}
    capabilities = {item["id"]: item for item in report["capabilities"]}
    assert report["status"] == "partial"
    assert report["manifest_identity"] == "sha256:" + manifest_sha
    assert environments["base"]["status"] == "complete"
    assert environments["qwen"]["status"] == "complete"
    assert environments["vision"] == {
        "diagnostic": "environment_missing",
        "id": "vision",
        "status": "not_configured",
    }
    assert capabilities["dense_siglip"]["status"] == "not_configured"
    assert capabilities["qwen_verification"]["status"] == "complete"
    assert report["failures"] == []
    assert str(tmp_path) not in encoded
    assert secret not in encoded
    assert "HF_TOKEN" not in encoded
    assert "serial" not in encoded.casefold()
    assert "uuid" not in encoded.casefold()
    assert before == {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("drift", "scope", "code"),
    (
        ("host", "host", "host_contract_mismatch"),
        ("offline", "offline_environment", "offline_policy_mismatch"),
        ("python", "environment", "python_version_mismatch"),
        ("distributions", "environment", "distribution_identity_mismatch"),
        ("uv", "uv", "uv_identity_mismatch"),
        ("lock", "contract", "contract_identity_mismatch"),
    ),
)
def test_attestation_classifies_contract_drift_without_exposing_values(
    tmp_path: Path,
    drift: str,
    scope: str,
    code: str,
) -> None:
    manifest_path, manifest_sha, identities = _fixture_manifest(tmp_path)
    environment = dict(OFFLINE_ENVIRONMENT)
    if drift == "offline":
        environment["HF_HUB_OFFLINE"] = "0"
    if drift == "lock":
        (tmp_path / "backend" / "uv.lock").write_bytes(b"changed-lock")

    def host_probe() -> HostSnapshot:
        observed = _host()
        if drift == "host":
            return HostSnapshot(
                system="Darwin",
                machine="x86_64",
                macos_version=observed.macos_version,
                chip=observed.chip,
                memory_gib=observed.memory_gib,
            )
        return observed

    def runtime_probe(_python: Path, environment_id: str) -> RuntimeSnapshot:
        if drift == "python" and environment_id == "base":
            return _runtime(identities[environment_id], python="3.12.14")
        identity = (
            "0" * 64
            if drift == "distributions" and environment_id == "base"
            else identities[environment_id]
        )
        return _runtime(identity)

    report = attest_ml_environment(
        tmp_path,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha,
        environ=environment,
        host_probe=host_probe,
        runtime_probe=runtime_probe,
        uv_probe=lambda _uv: UvSnapshot(
            version="0.12.3",
            sha256=("0" * 64 if drift == "uv" else sha256(b"uv-binary").hexdigest()),
        ),
    )

    assert report["status"] == "failed"
    assert any(
        failure["scope"] == scope
        and failure["kind"] == "drift"
        and failure["code"] == code
        for failure in report["failures"]
    )
    if drift == "python":
        assert report["uv"]["status"] == "complete"
    assert str(tmp_path) not in json.dumps(report)


def test_existing_broken_environment_is_infrastructure_failure_not_absence(
    tmp_path: Path,
) -> None:
    manifest_path, manifest_sha, identities = _fixture_manifest(tmp_path)
    (tmp_path / ".venv-qwen" / "bin" / "python").unlink()

    report = attest_ml_environment(
        tmp_path,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha,
        environ=OFFLINE_ENVIRONMENT,
        host_probe=_host,
        runtime_probe=lambda _python, environment_id: _runtime(
            identities[environment_id]
        ),
        uv_probe=lambda _uv: UvSnapshot(
            version="0.12.3",
            sha256=sha256(b"uv-binary").hexdigest(),
        ),
    )

    qwen = next(item for item in report["environments"] if item["id"] == "qwen")
    assert qwen == {
        "diagnostic": "python_unavailable",
        "failure_kind": "infrastructure",
        "id": "qwen",
        "status": "failed",
    }
    assert any(
        failure == {
            "code": "python_unavailable",
            "id": "qwen",
            "kind": "infrastructure",
            "scope": "environment",
        }
        for failure in report["failures"]
    )


def test_probe_failure_is_sanitized_infrastructure_failure(tmp_path: Path) -> None:
    manifest_path, manifest_sha, identities = _fixture_manifest(tmp_path)
    private_path = tmp_path / "private" / "token-hf_secret"

    def runtime_probe(_python: Path, environment_id: str) -> RuntimeSnapshot:
        if environment_id == "qwen":
            raise OSError(f"cannot execute {private_path}")
        return _runtime(identities[environment_id])

    report = attest_ml_environment(
        tmp_path,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha,
        environ=OFFLINE_ENVIRONMENT,
        host_probe=_host,
        runtime_probe=runtime_probe,
        uv_probe=lambda _uv: UvSnapshot(
            version="0.12.3",
            sha256=sha256(b"uv-binary").hexdigest(),
        ),
    )

    encoded = json.dumps(report)
    assert private_path.as_posix() not in encoded
    assert "hf_secret" not in encoded
    assert any(
        item["kind"] == "infrastructure"
        and item["code"] == "environment_probe_failed"
        for item in report["failures"]
    )


def test_present_requirements_environment_and_direct_distribution_are_verified(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest_sha, identities = _fixture_manifest(
        tmp_path,
        include_vision_environment=True,
    )
    vision_lock = tmp_path / "workers" / "vision" / "requirements.lock"
    vision_lock.write_text(
        vision_lock.read_text(encoding="utf-8")
        + "direct-package @ https://invalid.example/archive.tgz#sha256="
        + "c" * 64
        + "\n",
        encoding="utf-8",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    vision_contract = next(
        item for item in payload["contracts"] if item["id"] == "vision-lock"
    )
    vision_contract["sha256"] = sha256(vision_lock.read_bytes()).hexdigest()
    vision_environment = next(
        item for item in payload["environments"] if item["id"] == "vision"
    )
    vision_environment["distribution_source"]["direct_distributions"] = {
        "direct-package": "1.2.3"
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_sha = sha256(manifest_path.read_bytes()).hexdigest()
    vision_identity = _distribution_identity(
        {
            "direct-package": "1.2.3",
            "torch": "2.13.0",
            "transformers": "5.14.1",
        }
    )

    report = attest_ml_environment(
        tmp_path,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha,
        environ=OFFLINE_ENVIRONMENT,
        host_probe=_host,
        runtime_probe=lambda _python, environment_id: _runtime(
            vision_identity
            if environment_id == "vision"
            else identities[environment_id]
        ),
        uv_probe=lambda _uv: UvSnapshot(
            version="0.12.3",
            sha256=sha256(b"uv-binary").hexdigest(),
        ),
    )

    assert report["status"] == "complete"
    vision = next(item for item in report["environments"] if item["id"] == "vision")
    assert vision["distribution_identity"] == "sha256:" + vision_identity


def test_default_probes_return_only_whitelisted_runtime_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_profiler_payload = {
        "SPHardwareDataType": [
            {
                "chip_type": "Apple M4 Pro",
                "physical_memory": "24 GB",
                "platform_UUID": "private-uuid",
                "serial_number": "private-serial",
            }
        ]
    }

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(system_profiler_payload).encode("utf-8"),
            stderr=b"",
        )

    monkeypatch.setattr(attestation.subprocess, "run", fake_run)
    monkeypatch.setattr(attestation.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(attestation.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(attestation.platform, "mac_ver", lambda: ("26.6.2", (), ""))
    host = attestation._default_host_probe()
    assert host == _host()
    assert "private" not in repr(host)

    monkeypatch.undo()
    runtime = attestation._default_runtime_probe(
        Path(sys.executable),
        "base",
        offline_environment=OFFLINE_ENVIRONMENT,
    )
    assert runtime.implementation == platform.python_implementation()
    assert runtime.python_version == platform.python_version()
    assert len(runtime.distribution_identity) == 64

    uv_executable = _write_executable(
        tmp_path / "uv",
        b"#!/bin/sh\nprintf 'uv 0.12.3 (fixture 2026-01-01)\\n'\n",
    )
    uv = attestation._default_uv_probe(
        uv_executable,
        offline_environment=OFFLINE_ENVIRONMENT,
    )
    assert uv == UvSnapshot(
        version="0.12.3",
        sha256=sha256(uv_executable.read_bytes()).hexdigest(),
    )


@pytest.mark.parametrize(("status", "exit_code"), (("partial", 0), ("failed", 2)))
def test_cli_emits_one_json_document_and_uses_status_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    exit_code: int,
) -> None:
    report = {
        "attestation_id": "fixture",
        "capabilities": [],
        "contracts": [],
        "environments": [],
        "failures": [],
        "host": {"status": "complete"},
        "manifest_identity": "sha256:" + "a" * 64,
        "offline_environment": {"status": "complete"},
        "schema_version": 1,
        "status": status,
        "uv": {"status": "complete"},
    }
    monkeypatch.setattr(attestation, "attest_ml_environment", lambda *_a, **_k: report)

    assert attestation.main([]) == exit_code
    assert json.loads(capsys.readouterr().out) == report


def test_cli_sanitizes_manifest_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise MlEnvironmentManifestError("manifest_identity_mismatch")

    monkeypatch.setattr(attestation, "attest_ml_environment", fail)

    assert attestation.main([]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert report["failures"] == [
        {
            "code": "manifest_identity_mismatch",
            "id": "phase0-environment",
            "kind": "drift",
            "scope": "manifest",
        }
    ]


def test_manifest_is_strict_bounded_and_rejects_nonportable_paths(tmp_path: Path) -> None:
    manifest_path, manifest_sha, _identities = _fixture_manifest(tmp_path)
    loaded = load_ml_environment_manifest(
        manifest_path,
        expected_sha256=manifest_sha,
    )
    assert loaded.attestation_id == "phase0-fixture-v1"

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    changed_sha = sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(MlEnvironmentManifestError) as unexpected:
        load_ml_environment_manifest(manifest_path, expected_sha256=changed_sha)
    assert unexpected.value.code == "manifest_invalid"

    payload.pop("unexpected")
    payload["contracts"][0]["path"] = "../private.lock"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    changed_sha = sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(MlEnvironmentManifestError) as unsafe:
        load_ml_environment_manifest(manifest_path, expected_sha256=changed_sha)
    assert unsafe.value.code == "manifest_invalid"

    with pytest.raises(MlEnvironmentManifestError) as identity:
        load_ml_environment_manifest(manifest_path, expected_sha256="0" * 64)
    assert identity.value.code == "manifest_identity_mismatch"


def test_committed_manifest_binds_qwen_composite_runtime_and_make_is_offline() -> None:
    from videoscope.model_manifest import (
        FASTEMBED_REPOSITORY,
        LIGHTHOUSE_CLIP_CHECKPOINT_SHA256,
        LIGHTHOUSE_CLIP_REVISION,
        LIGHTHOUSE_SOURCE_REVISION,
        MODEL_REVISIONS,
        QWEN_VIDEO_MODEL,
        SIGLIP_224_MODEL,
        SIGLIP_384_MODEL,
        WHISPER_MODEL,
        model_identity,
    )
    from videoscope.providers.lighthouse_worker import (
        LIGHTHOUSE_MODEL_IDENTITY,
        LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
    )
    from videoscope.providers.paddle_ocr import (
        OCR_MODEL_ARTIFACT_IDENTITY,
        OCR_WORKER_RUNTIME_IDENTITY,
    )
    from videoscope.providers.qwen_video import QWEN_INFERENCE_RUNTIME_IDENTITY
    from videoscope.providers.vision_worker_contract import (
        RFDETR_SMALL_CHECKPOINT_SHA256,
        VISION_WORKER_RUNTIME_IDENTITY,
    )
    from videoscope.providers.whisper_worker import WHISPER_INFERENCE_RUNTIME_IDENTITY

    manifest_path = PROJECT_ROOT / "workers" / "ml-environment.lock.json"
    assert sha256(manifest_path.read_bytes()).hexdigest() == (
        DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256
    )
    loaded = load_ml_environment_manifest(
        manifest_path,
        expected_sha256=DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256,
    )
    assert loaded.attestation_id == "videoscope-phase0-ml-environment-v1"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    contracts = {item["id"]: item for item in payload["contracts"]}
    capabilities = {item["id"]: item for item in payload["capabilities"]}
    environments = {item["id"]: item for item in payload["environments"]}
    qwen_runtime = json.loads(
        (PROJECT_ROOT / "workers" / "qwen" / "runtime.lock.json").read_text(
            encoding="utf-8"
        )
    )
    qwen_models = json.loads(
        (PROJECT_ROOT / "workers" / "qwen" / "model-artifacts.lock.json").read_text(
            encoding="utf-8"
        )
    )

    assert contracts["qwen-runtime-manifest"]["canonical_json_sha256"] == (
        canonical_json_sha256(qwen_runtime)
    )
    assert contracts["qwen-model-manifest"]["canonical_json_sha256"] == (
        canonical_json_sha256(qwen_models)
    )
    for contract in payload["contracts"]:
        source = PROJECT_ROOT / contract["path"]
        assert sha256(source.read_bytes()).hexdigest() == contract["sha256"]
        if "canonical_json_sha256" in contract:
            assert canonical_json_sha256(
                json.loads(source.read_text(encoding="utf-8"))
            ) == contract["canonical_json_sha256"]
    assert (
        "runtime-manifest-sha256:"
        + canonical_json_sha256(qwen_runtime)
        in capabilities["qwen_verification"]["runtime_identity"]
    )
    assert (
        "model-manifest-sha256:"
        + canonical_json_sha256(qwen_models)
        in capabilities["qwen_verification"]["runtime_identity"]
    )
    assert set(environments) == {
        "base",
        "lighthouse",
        "ocr",
        "qwen",
        "vision",
        "whisper",
    }
    assert {
        identifier: environment["python"]
        for identifier, environment in environments.items()
    } == {
        "base": "3.12.13",
        "lighthouse": "3.11.14",
        "ocr": "3.12.13",
        "qwen": "3.12.13",
        "vision": "3.12.13",
        "whisper": "3.12.13",
    }
    assert environments["lighthouse"]["distribution_source"][
        "direct_distributions"
    ] == {"clip": "1.0", "lighthouse": "0.1"}
    assert capabilities["dense_siglip"]["runtime_identity"] == (
        VISION_WORKER_RUNTIME_IDENTITY
    )
    assert capabilities["dense_siglip"]["model_identities"] == [
        model_identity(SIGLIP_224_MODEL, MODEL_REVISIONS[SIGLIP_224_MODEL]),
        model_identity(SIGLIP_384_MODEL, MODEL_REVISIONS[SIGLIP_384_MODEL]),
    ]
    assert capabilities["object_detection"]["model_identities"] == [
        "rfdetr-small@sha256:" + RFDETR_SMALL_CHECKPOINT_SHA256
    ]
    assert capabilities["speech_transcription"]["runtime_identity"] == (
        WHISPER_INFERENCE_RUNTIME_IDENTITY
    )
    assert capabilities["speech_transcription"]["model_identities"] == [
        model_identity(WHISPER_MODEL, MODEL_REVISIONS[WHISPER_MODEL])
    ]
    assert capabilities["ocr_text"]["runtime_identity"] == OCR_WORKER_RUNTIME_IDENTITY
    assert capabilities["ocr_text"]["model_identities"] == [
        OCR_MODEL_ARTIFACT_IDENTITY
    ]
    assert capabilities["temporal_refinement"]["runtime_identity"] == (
        LIGHTHOUSE_WORKER_RUNTIME_IDENTITY
    )
    assert capabilities["temporal_refinement"]["model_identities"] == [
        LIGHTHOUSE_MODEL_IDENTITY,
        "lighthouse/source@" + LIGHTHOUSE_SOURCE_REVISION,
        "openai/clip@" + LIGHTHOUSE_CLIP_REVISION,
        "openai/clip-vit-b-32@sha256:" + LIGHTHOUSE_CLIP_CHECKPOINT_SHA256,
    ]
    assert capabilities["qwen_verification"]["runtime_identity"] == (
        QWEN_INFERENCE_RUNTIME_IDENTITY
    )
    assert capabilities["qwen_verification"]["model_identities"] == [
        model_identity(QWEN_VIDEO_MODEL, MODEL_REVISIONS[QWEN_VIDEO_MODEL])
    ]
    assert capabilities["base_text_search"]["model_identities"] == [
        model_identity(FASTEMBED_REPOSITORY, MODEL_REVISIONS[FASTEMBED_REPOSITORY])
    ]

    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    target = makefile.split("ml-attest-offline:\n", 1)[1].split("\n\n", 1)[0]
    assert "/usr/bin/env -i" in target
    assert (
        ".venv/bin/python\" -I -m videoscope.ml_environment_attestation"
        in target
    )
    for name, value in OFFLINE_ENVIRONMENT.items():
        assert f"{name}={value}" in target
    for forbidden in ("download", "snapshot_download", "data/", "inference"):
        assert forbidden not in target
