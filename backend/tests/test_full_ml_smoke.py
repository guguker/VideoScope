from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from videoscope.benchmark.adapter import (
    ProductBenchmarkSearchAdapter,
    ProductSearchExecutionReceipt,
)
from videoscope.benchmark.host_resources import (
    HOST_RESOURCE_IDENTITY,
    HostResourceRawSample,
    HostResourceSampler,
    HostResourceSnapshot,
)
from videoscope.benchmark.profiles import FROZEN_PROFILES
from videoscope.providers.base import ProviderState
from videoscope.providers.qwen_video import QwenVideoJudgement


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SHA = "a" * 64


def _load_script() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "full-ml-smoke.py"
    specification = importlib.util.spec_from_file_location(
        "videoscope_test_full_ml_smoke",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _offline_environment(root: Path, models_root: Path) -> dict[str, str]:
    return {
        "HF_HUB_OFFLINE": "1",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "TRANSFORMERS_OFFLINE": "1",
        "VIDEOSCOPE_DATA_DIR": str(root / "product"),
        "VIDEOSCOPE_VISION_WORKER_INPUT_ROOT": str(root),
        "VIDEOSCOPE_WHISPER_WORKER_INPUT_ROOT": str(
            root / "product" / "media"
        ),
        "VIDEOSCOPE_WHISPER_WORKER_WORK_ROOT": str(
            root / "tmp" / "whisper-worker"
        ),
        "VIDEOSCOPE_FULL_ML_SMOKE_MODELS_ROOT": str(models_root),
    }


def _models_root(tmp_path: Path) -> Path:
    models_root = tmp_path / "immutable-models"
    models_root.mkdir(mode=0o755)
    return models_root


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        vision_worker_endpoint="http://127.0.0.1:8783",
        vision_worker_api_key="v" * 32,
        vision_worker_timeout=120.0,
        whisper_worker_endpoint="http://127.0.0.1:8784",
        whisper_worker_api_key="w" * 32,
        whisper_worker_timeout=600.0,
        whisper_model="mlx-community/whisper-large-v3-turbo",
        whisper_language="auto",
        whisper_initial_prompt=None,
        ocr_worker_python=Path(".venv-ocr/bin/python"),
        ocr_worker_script=Path("scripts/paddle-ocr-worker.py"),
        lighthouse_endpoint="http://127.0.0.1:8782",
        lighthouse_api_key="l" * 32,
        lighthouse_timeout=300.0,
        qwen_video_endpoint="http://127.0.0.1:8781",
        qwen_video_api_key="q" * 32,
        qwen_video_timeout=180.0,
        qwen_video_model="mlx-community/Qwen3.5-9B-MLX-4bit",
        siglip_model="google/siglip2-base-patch16-224",
        siglip_batch_size=1,
        visual_index_step=1.0,
        visual_index_max_width=224,
        vision_worker_minimum_confidence=0.25,
        vision_detector_model_id="rfdetr-small",
        vision_detector_checkpoint_sha256="b" * 64,
    )


class _Toolchain:
    identity = "sha256:" + _SHA

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls = 0

    def verify_current(self) -> str:
        self.calls += 1
        self.events.append(
            "toolchain.verify.pre" if self.calls == 1 else "toolchain.verify.post"
        )
        return self.identity


class _Vision:
    def __init__(
        self,
        events: list[str],
        *,
        fail_image: bool = False,
        oom_image: bool = False,
        fail_ingestion_release: bool = False,
    ) -> None:
        self.events = events
        self.fail_image = fail_image
        self.oom_image = oom_image
        self.fail_ingestion_release = fail_ingestion_release
        self.specification = SimpleNamespace(
            embedding_dimensions=3,
            identity="c" * 64,
        )

    def capability(self) -> SimpleNamespace:
        self.events.append("vision.status")
        return SimpleNamespace(ready=True)

    def image_vectors(self, paths: list[Path]) -> np.ndarray:
        self.events.append("vision.image")
        assert len(paths) == 1
        if self.oom_image:
            raise MemoryError("private allocator diagnostics")
        if self.fail_image:
            raise RuntimeError("secret-token /private/user/path")
        return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    def text_vectors(self, texts: list[str]) -> np.ndarray:
        self.events.append("vision.text")
        assert texts == ["synthetic local video"]
        return np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)

    def detect(self, image: Path) -> list[object]:
        self.events.append("vision.detect")
        assert image.is_file()
        return []

    def release_ingestion_resources(self) -> None:
        self.events.append("vision.release_ingestion_resources")
        if self.fail_ingestion_release:
            raise RuntimeError("private vision worker release diagnostics")


class _Whisper:
    expected_model_identity = "whisper@" + "d" * 40

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def status(self) -> SimpleNamespace:
        self.events.append("whisper.status")
        return SimpleNamespace(ready=True)

    def transcribe(self, source: Path, **kwargs: object) -> list[object]:
        self.events.append("whisper.transcribe")
        assert source.is_file()
        assert kwargs["language"] == "auto"
        return []


class _OCR:
    worker_attestation = {
        "dependency_identity": "paddleocr-deps-v1:sha256:" + "e" * 64,
        "model_identity": "paddleocr-models-v1:sha256:" + "f" * 64,
        "protocol": "videoscope.paddleocr-jsonl.v2",
        "runtime_identity": "ocr-runtime",
        "script_sha256": "1" * 64,
    }

    def __init__(
        self,
        events: list[str],
        *,
        read_outcome: str = "complete",
        fail_ingestion_release: bool = False,
    ) -> None:
        self.events = events
        self.read_outcome = read_outcome
        self.fail_ingestion_release = fail_ingestion_release

    def status(self) -> SimpleNamespace:
        self.events.append("ocr.status")
        return SimpleNamespace(state=ProviderState.READY)

    def read(self, image: Path) -> list[tuple[str, float]]:
        self.events.append("ocr.read")
        assert image.is_file()
        if self.read_outcome == "failed":
            raise RuntimeError("private OCR request diagnostics")
        if self.read_outcome == "invalid":
            return None  # type: ignore[return-value]
        return []

    def release_ingestion_resources(self) -> None:
        self.events.append("ocr.release_ingestion_resources")
        if self.fail_ingestion_release:
            raise RuntimeError("private OCR worker release diagnostics")

    def close(self) -> None:
        self.events.append("ocr.close")


class _Lighthouse:
    specification = SimpleNamespace(identity="2" * 64)

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def status(self) -> SimpleNamespace:
        self.events.append("lighthouse.status")
        return SimpleNamespace(state=ProviderState.READY)

    def build_video_source(
        self,
        video_id: str,
        source: Path,
        duration: float,
    ) -> dict[str, object]:
        self.events.append("lighthouse.build")
        assert source.is_file() and duration > 0
        return {
            "video_id": video_id,
            "generation_id": "3" * 32,
            "source_sha256": "4" * 64,
            "source_size_bytes": source.stat().st_size,
            "duration_seconds": duration,
            "specification_hash": self.specification.identity,
            "manifest_sha256": "5" * 64,
        }

    def search_generations(
        self,
        query: str,
        bindings: dict[str, dict[str, object]],
        *,
        limit: int,
    ) -> list[object]:
        self.events.append("lighthouse.search")
        assert query == "synthetic local video"
        assert len(bindings) == 1 and limit == 1
        return []


class _Qwen:
    expected_model_identity = "qwen@" + "6" * 40

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def status(self) -> SimpleNamespace:
        self.events.append("qwen.status")
        return SimpleNamespace(ready=True)

    def judge_video(
        self,
        source: Path,
        *,
        fps: float,
        max_tokens: int,
    ) -> QwenVideoJudgement:
        self.events.append("qwen.judge")
        assert source.is_file() and fps == 1.0 and max_tokens == 320
        return QwenVideoJudgement(
            matches_query=None,
            confidence=0.0,
            evidence="synthetic smoke",
        )


class _ResourceMonitor:
    def __init__(self) -> None:
        self.started = False
        self.finished = False

    def start(self) -> None:
        assert self.started is False
        self.started = True

    def finish(self) -> dict[str, object]:
        assert self.started is True
        self.finished = True
        return {
            "process_tree": {
                "baseline_bytes": 10_000_000,
                "increment_bytes": 2_345_678,
                "peak_bytes": 12_345_678,
                "sample_count": 3,
                "samples_bytes": (10_000_000, 12_345_678, 11_000_000),
            },
            "host_resources": _host_resource_receipt(),
        }

    def close(self) -> None:
        return None


class _WorkerCluster:
    settings_overrides: dict[str, object] = {}
    managed_workers: tuple[object, ...] = ()
    all_workers_managed = False

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def wait_ready(self) -> None:
        self.events.append("workers.ready")

    def close(self) -> None:
        self.events.append("workers.close")


def _host_resource_receipt() -> object:
    snapshots = (
        HostResourceSnapshot(1_000, 2_000, 10, 20, 30, 16_384),
        HostResourceSnapshot(1_400, 2_500, 11, 22, 33, 16_384),
        HostResourceSnapshot(1_200, 2_300, 12, 25, 37, 16_384),
    )
    samples = tuple(
        HostResourceRawSample(index * 250_000_000, snapshot)
        for index, snapshot in enumerate(snapshots)
    )
    return HostResourceSampler.receipt_from_samples(
        provider_identity=HOST_RESOURCE_IDENTITY,
        sample_interval_milliseconds=250,
        samples=samples,
    )


_EXECUTION_COMPONENT_IDS = (
    "text_vectors",
    "lexical_text",
    "visual_dense",
    "temporal_refinement",
    "lighthouse",
    "qwen_verification",
)


def _execution_receipt(
    profile_id: str,
    *,
    missing_component: str | None = None,
    zero_component: str | None = None,
    include_generic_lexical: bool = False,
) -> ProductSearchExecutionReceipt:
    profile = FROZEN_PROFILES[profile_id]
    observed = set(profile.required_capabilities)
    if profile.search_plan.text_search != "disabled":
        observed.add("lexical_text")
    if missing_component is not None:
        observed.remove(missing_component)
    if include_generic_lexical:
        observed.add("lexical_text")
    output_counts = tuple(
        (
            component_id,
            int(component_id in observed and component_id != zero_component),
        )
        for component_id in _EXECUTION_COMPONENT_IDS
    )
    evidence_counts = tuple(
        (
            component_id,
            int(component_id in observed and component_id != zero_component),
        )
        for component_id in _EXECUTION_COMPONENT_IDS
    )
    input_counts = tuple(
        (component_id, int(component_id in observed))
        for component_id in _EXECUTION_COMPONENT_IDS
    )
    return ProductSearchExecutionReceipt(
        schema_version=1,
        profile_id=profile.profile_id,
        profile_identity=profile.identity,
        search_configuration_identity=(
            ProductBenchmarkSearchAdapter._configuration(profile).identity
        ),
        total_evidence_count=max(1, sum(count for _, count in evidence_counts)),
        invoked_component_ids=tuple(
            component_id
            for component_id in _EXECUTION_COMPONENT_IDS
            if component_id in observed
        ),
        component_input_counts=input_counts,
        component_output_counts=output_counts,
        component_evidence_counts=evidence_counts,
    )


def _serialized_execution_receipt(
    receipt: ProductSearchExecutionReceipt,
) -> dict[str, object]:
    return {
        "schema_version": receipt.schema_version,
        "profile_identity": receipt.profile_identity,
        "search_configuration_identity": receipt.search_configuration_identity,
        "invoked_component_ids": list(receipt.invoked_component_ids),
        "component_input_counts": dict(receipt.component_input_counts),
        "component_output_counts": dict(receipt.component_output_counts),
        "component_evidence_counts": dict(receipt.component_evidence_counts),
    }


def _complete_product_receipt() -> dict[str, object]:
    profiles: dict[str, object] = {}
    evidence_count = 0
    for profile_id in (
        "lexical_qdrant",
        "dense_siglip",
        "temporal_refinement",
        "lighthouse",
        "qwen_verification",
    ):
        execution = _execution_receipt(profile_id)
        evidence_count += execution.total_evidence_count
        profiles[profile_id] = {
            "close": "complete",
            "status": "complete",
            "generation_bound": True,
            "evidence_count": execution.total_evidence_count,
            "component_execution": _serialized_execution_receipt(execution),
            "open": "complete",
            "search": "complete",
        }
    profiles["internvideo"] = {
        "status": "not_configured",
        "reason_code": "provider_not_configured",
    }
    return {
        "status": "complete",
        "upload": "complete",
        "index": "complete",
        "search": "complete",
        "export": {
            "byte_size": 14,
            "container": "mp4",
            "duration_seconds": 1.0,
            "source_profile_id": "lexical_qdrant",
            "status": "complete",
        },
        "generation_bound": True,
        "evidence_count": evidence_count,
        "profiles": profiles,
        "runtime_cleanup": "complete",
    }


def _dependencies(
    script: ModuleType,
    events: list[str],
    *,
    fail_image: bool = False,
    oom_image: bool = False,
    fail_ingestion_release: bool = False,
    ocr_read_outcome: str = "complete",
    fail_ocr_ingestion_release: bool = False,
) -> object:
    toolchain = _Toolchain(events)

    def fixture_factory(_toolchain: object, workspace: object) -> object:
        events.append("fixture")
        for path in (workspace.media_video, workspace.qwen_video, workspace.image):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        return script.SmokeFixture(
            duration_seconds=1.0,
            image=workspace.image,
            media_video=workspace.media_video,
            qwen_video=workspace.qwen_video,
            video_id=workspace.video_id,
        )

    def product_integration(*_args: object) -> dict[str, object]:
        events.append("product.integration")
        return _complete_product_receipt()

    return script.SmokeDependencies(
        attest_ml_environment=lambda _root, _environ: (
            events.append("ml-environment.attest")
            or {
                "environment_bindings": {
                    "owner": "base",
                    "vision": "vision",
                    "whisper": "whisper",
                    "ocr": "ocr",
                    "lighthouse": "lighthouse",
                    "qwen": "qwen",
                },
                "ml_environment_attestation_id": (
                    "videoscope-phase0-ml-environment-v1"
                ),
                "ml_environment_manifest_identity": "sha256:" + "9" * 64,
            }
        ),
        attest_toolchain=lambda _environ: toolchain,
        build_clients=lambda _settings, _root: script.SmokeClients(
            vision=_Vision(
                events,
                fail_image=fail_image,
                oom_image=oom_image,
                fail_ingestion_release=fail_ingestion_release,
            ),
            whisper=_Whisper(events),
            ocr=_OCR(
                events,
                read_outcome=ocr_read_outcome,
                fail_ingestion_release=fail_ocr_ingestion_release,
            ),
            lighthouse=_Lighthouse(events),
            qwen=_Qwen(events),
        ),
        build_fixture=fixture_factory,
        build_resource_monitor=lambda _workers: _ResourceMonitor(),
        code_identity_resolver=lambda: "a" * 40,
        load_settings=lambda _root, _models_root, _overrides: _settings(),
        product_integration=product_integration,
        start_workers=lambda *_args: (
            events.append("workers.start") or _WorkerCluster(events)
        ),
    )


def test_requires_exact_offline_flags_before_attestation(tmp_path: Path) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    models_root = _models_root(tmp_path)
    called = False

    def forbidden_attestation(_environ: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("attestation must not run")

    dependencies = script.SmokeDependencies(
        attest_ml_environment=lambda *_args: pytest.fail(
            "ML environment must not be attested"
        ),
        attest_toolchain=forbidden_attestation,
        build_clients=lambda *_args: pytest.fail("clients must not be built"),
        build_fixture=lambda *_args: pytest.fail("fixture must not be built"),
        build_resource_monitor=lambda _workers: pytest.fail(
            "monitor must not be built"
        ),
        code_identity_resolver=lambda: "a" * 40,
        load_settings=lambda _root, _models_root, _overrides: _settings(),
        product_integration=None,
        start_workers=lambda *_args: pytest.fail("workers must not start"),
    )

    with pytest.raises(script.SmokeConfigurationError) as captured:
        script.execute(
            root,
            models_root=models_root,
            environ={},
            dependencies=dependencies,
        )

    assert captured.value.code == "offline_environment_required"
    assert called is False


def test_requires_exact_external_models_root_binding_before_attestation(
    tmp_path: Path,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    models_root = _models_root(tmp_path)
    events: list[str] = []
    environment = _offline_environment(root, models_root)
    environment["VIDEOSCOPE_FULL_ML_SMOKE_MODELS_ROOT"] = str(
        tmp_path / "different-models"
    )

    with pytest.raises(script.SmokeConfigurationError) as captured:
        script.execute(
            root,
            models_root=models_root,
            environ=environment,
            dependencies=_dependencies(script, events),
        )

    assert captured.value.code == "models_root_binding_required"
    assert events == []

    nested_models = root / "models"
    nested_models.mkdir(mode=0o700)
    with pytest.raises(script.SmokeConfigurationError) as captured:
        script.execute(
            root,
            models_root=nested_models,
            environ=_offline_environment(root, nested_models),
            dependencies=_dependencies(script, events),
        )

    assert captured.value.code == "immutable_model_root_invalid"
    assert events == []

    with pytest.raises(script.SmokeConfigurationError) as captured:
        script.execute(
            root,
            models_root=PROJECT_ROOT,
            environ=_offline_environment(root, PROJECT_ROOT),
            dependencies=_dependencies(script, events),
        )

    assert captured.value.code == "immutable_model_root_invalid"
    assert events == []


def test_settings_use_validated_models_root_without_project_data_coupling(
    tmp_path: Path,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    models_root = _models_root(tmp_path)

    settings = script._load_settings(
        root,
        models_root,
        {
            "ffmpeg_binary": Path("/attested/ffmpeg"),
            "ffprobe_binary": Path("/attested/ffprobe"),
        },
    )

    assert settings.data_dir == root / "product"
    assert settings.ffmpeg_binary == Path("/attested/ffmpeg")
    assert settings.ffprobe_binary == Path("/attested/ffprobe")
    assert settings.models_dir == models_root
    assert settings.models_dir != PROJECT_ROOT / "data" / "models"


def test_exact_environment_binding_rejects_substituted_compatible_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    project_root = tmp_path / "checkout"
    smoke_root = tmp_path / "smoke"
    project_root.mkdir()
    smoke_root.mkdir(mode=0o700)
    directories = {
        "base": ".venv",
        "vision": ".venv-vision",
        "whisper": ".venv-whisper",
        "ocr": ".venv-ocr",
        "lighthouse": ".venv-lighthouse",
        "qwen": ".venv-qwen",
    }
    expected: dict[str, Path] = {}
    for environment_id, directory in directories.items():
        executable = project_root / directory / "bin" / "python"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"reviewed-python")
        executable.chmod(0o700)
        expected[environment_id] = executable
    manifest = SimpleNamespace(
        environments=tuple(
            SimpleNamespace(environment_id=environment_id, directory=directory)
            for environment_id, directory in directories.items()
        )
    )
    environment = {
        script._WORKER_PYTHON_ENVIRONMENT[role]: str(expected[environment_id])
        for role, environment_id in script._ML_ENVIRONMENT_ROLE_BINDINGS.items()
        if role != "owner"
    }
    monkeypatch.setattr(script, "_PROJECT_ROOT", project_root)

    assert script._validate_environment_executable_bindings(
        smoke_root,
        environment,
        manifest,
        owner_executable=expected["base"],
    ) == dict(script._ML_ENVIRONMENT_ROLE_BINDINGS)

    substitute = tmp_path / "compatible-copy" / "bin" / "python"
    substitute.parent.mkdir(parents=True)
    substitute.write_bytes(b"reviewed-python")
    substitute.chmod(0o700)
    environment[script._WORKER_PYTHON_ENVIRONMENT["vision"]] = str(substitute)
    with pytest.raises(script.SmokeConfigurationError) as captured:
        script._validate_environment_executable_bindings(
            smoke_root,
            environment,
            manifest,
            owner_executable=expected["base"],
        )

    assert captured.value.code == "ml_environment_binding_mismatch"
    assert captured.value.component == "vision"


def test_media_toolchain_binding_requires_exact_named_pair(
    tmp_path: Path,
) -> None:
    script = _load_script()
    media_bin = tmp_path / "media-bin"
    media_bin.mkdir()
    environment: dict[str, str] = {}
    for name in ("ffmpeg", "ffprobe"):
        executable = media_bin / name
        executable.write_bytes(b"#!/bin/sh\n")
        executable.chmod(0o700)
        environment[f"VIDEOSCOPE_FULL_ML_SMOKE_{name.upper()}_BINARY"] = str(
            executable
        )

    assert script._validated_media_executables(environment) == (
        media_bin / "ffmpeg",
        media_bin / "ffprobe",
    )

    substitute = tmp_path / "other-bin"
    substitute.mkdir()
    foreign_probe = substitute / "ffprobe"
    foreign_probe.write_bytes(b"#!/bin/sh\n")
    foreign_probe.chmod(0o700)
    environment["VIDEOSCOPE_FULL_ML_SMOKE_FFPROBE_BINARY"] = str(foreign_probe)
    with pytest.raises(script.SmokeConfigurationError) as captured:
        script._validated_media_executables(environment)

    assert captured.value.code == "media_toolchain_binding_required"

    canonical_bin = tmp_path / "cellar" / "bin"
    canonical_bin.mkdir(parents=True)
    for name in ("ffmpeg", "ffprobe"):
        target = canonical_bin / name
        target.write_bytes(b"#!/bin/sh\n")
        target.chmod(0o700)
        (media_bin / name).unlink()
        (media_bin / name).symlink_to(target)
        environment[f"VIDEOSCOPE_FULL_ML_SMOKE_{name.upper()}_BINARY"] = str(
            media_bin / name
        )

    assert script._validated_media_executables(environment) == (
        canonical_bin / "ffmpeg",
        canonical_bin / "ffprobe",
    )


def test_rejects_invalid_ml_environment_receipt_before_toolchain(
    tmp_path: Path,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    models_root = _models_root(tmp_path)
    events: list[str] = []
    dependencies = _dependencies(script, events)._replace(
        attest_ml_environment=lambda *_args: {
            "environment_bindings": dict(script._ML_ENVIRONMENT_ROLE_BINDINGS),
            "ml_environment_attestation_id": (
                "videoscope-phase0-ml-environment-v1"
            ),
            "ml_environment_manifest_identity": "foreign-runtime",
        }
    )

    with pytest.raises(script.SmokeContractError) as captured:
        script.execute(
            root,
            models_root=models_root,
            environ=_offline_environment(root, models_root),
            dependencies=dependencies,
        )

    assert captured.value.code == "ml_environment_attestation_invalid"
    assert events == []


def test_product_settings_ignore_hostile_ambient_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    models_root = _models_root(tmp_path)
    monkeypatch.setenv("VIDEOSCOPE_SCENE_THRESHOLD", "999")
    monkeypatch.setenv("VIDEOSCOPE_VISUAL_INDEX_STEP", "99")
    monkeypatch.setenv("QWEN_VIDEO_MODEL", "attacker/model")
    monkeypatch.setenv("QWEN_VIDEO_ENDPOINT", "https://attacker.invalid")
    monkeypatch.setenv("INTERNVIDEO_ENDPOINT", "https://attacker.invalid")
    monkeypatch.setenv("ROBOFLOW_API_KEY", "ambient-secret")
    overrides = {
        "ffmpeg_binary": Path("/attested/ffmpeg"),
        "ffprobe_binary": Path("/attested/ffprobe"),
        "vision_worker_endpoint": "http://127.0.0.1:8783",
        "vision_worker_api_key": "v" * 32,
        "whisper_worker_endpoint": "http://127.0.0.1:8784",
        "whisper_worker_api_key": "w" * 32,
        "lighthouse_endpoint": "http://127.0.0.1:8782",
        "lighthouse_api_key": "l" * 32,
        "qwen_video_endpoint": "http://127.0.0.1:8781",
        "qwen_video_api_key": "q" * 32,
    }

    settings = script._load_settings(root, models_root, overrides)

    assert settings.scene_threshold == 3.0
    assert settings.visual_index_step == 1.0
    assert settings.qwen_video_model == script.QWEN_VIDEO_MODEL
    assert settings.qwen_video_endpoint == "http://127.0.0.1:8781"
    assert settings.internvideo_endpoint is None
    assert settings.roboflow_api_key is None


def test_rejects_non_private_or_project_roots_before_clients(
    tmp_path: Path,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o755)
    models_root = _models_root(tmp_path)
    dependencies = _dependencies(script, [])

    with pytest.raises(script.SmokeConfigurationError) as captured:
        script.execute(
            root,
            models_root=models_root,
            environ=_offline_environment(root, models_root),
            dependencies=dependencies,
        )

    assert captured.value.code == "disposable_root_invalid"

    project_root = PROJECT_ROOT / ".full-ml-smoke-test-root"
    project_root.mkdir(mode=0o700, exist_ok=False)
    try:
        with pytest.raises(script.SmokeConfigurationError) as captured:
            script.execute(
                project_root,
                models_root=models_root,
                environ=_offline_environment(project_root, models_root),
                dependencies=dependencies,
            )
        assert captured.value.code == "disposable_root_forbidden"
    finally:
        project_root.rmdir()


def test_worker_scaffold_baseline_is_preserved_while_smoke_writes_are_removed(
    tmp_path: Path,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    models_root = _models_root(tmp_path)
    (root / "media").mkdir(mode=0o700)
    work_root = root / "tmp" / "whisper-worker"
    work_root.mkdir(mode=0o700, parents=True)
    lock = work_root / ".videoscope-whisper-worker.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    events: list[str] = []

    report = script.execute(
        root,
        models_root=models_root,
        environ=_offline_environment(root, models_root),
        dependencies=_dependencies(script, events),
    )

    assert report["workspace_cleanup"] == "complete"
    assert lock.is_file() and lock.read_bytes() == b""
    assert sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*")
    ) == [
        "media",
        "tmp",
        "tmp/whisper-worker",
        "tmp/whisper-worker/.videoscope-whisper-worker.lock",
    ]


def test_component_smoke_is_sequential_pathless_and_completes_product_path(
    tmp_path: Path,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    models_root = _models_root(tmp_path)
    events: list[str] = []

    report = script.execute(
        root,
        models_root=models_root,
        environ=_offline_environment(root, models_root),
        dependencies=_dependencies(script, events),
    )

    assert events == [
        "ml-environment.attest",
        "toolchain.verify.pre",
        "workers.start",
        "workers.ready",
        "fixture",
        "vision.status",
        "vision.image",
        "vision.text",
        "vision.detect",
        "vision.release_ingestion_resources",
        "whisper.status",
        "whisper.transcribe",
        "ocr.status",
        "ocr.read",
        "ocr.release_ingestion_resources",
        "lighthouse.status",
        "lighthouse.build",
        "lighthouse.search",
        "product.integration",
        "qwen.status",
        "qwen.judge",
        "ocr.close",
        "toolchain.verify.post",
        "workers.close",
    ]
    assert report["status"] == "ready"
    assert report["code_sha_before"] == "a" * 40
    assert report["code_sha_after"] == "a" * 40
    assert report["ml_environment_attestation_id"] == (
        "videoscope-phase0-ml-environment-v1"
    )
    assert report["ml_environment_manifest_identity"] == "sha256:" + "9" * 64
    assert report["environment_bindings"] == {
        "owner": "base",
        "vision": "vision",
        "whisper": "whisper",
        "ocr": "ocr",
        "lighthouse": "lighthouse",
        "qwen": "qwen",
    }
    assert [step["id"] for step in report["steps"]] == [
        "vision.image_embedding",
        "vision.text_embedding",
        "vision.rfdetr",
        "whisper.transcribe",
        "ocr.read",
        "lighthouse.generation",
        "lighthouse.search",
        "qwen.judge",
    ]
    assert all(step["status"] == "complete" for step in report["steps"])
    assert report["product_integration"] == _complete_product_receipt()
    assert report["workspace_cleanup"] == "complete"
    host_resources = _host_resource_receipt().to_portable_dict()
    assert report["resources"] == {
        "host_resources": host_resources,
        "measurement_caveat": {
            "memory_accounting": (
                "metal_standalone_not_additive_with_process_rss"
            ),
            "scope": "system_wide",
        },
        "oom": {"status": "not_observed"},
        "peak_metal_bytes": {
            "baseline_bytes": 1_000,
            "increment_bytes": 400,
            "scope": "system_wide",
            "status": "measured",
            "value": 1_400,
        },
        "sampled_peak_process_tree_rss_bytes": {
            "baseline_bytes": 10_000_000,
            "external_loopback_workers_included": False,
            "increment_bytes": 2_345_678,
            "managed_worker_roles": [],
            "sample_count": 3,
            "samples_bytes": [10_000_000, 12_345_678, 11_000_000],
            "sampling_interval_ms": 50,
            "scope": "smoke_process_and_descendants",
            "status": "measured",
            "value": 12_345_678,
        },
        "system_wide_pressure_deltas": {
            "metal_recovery_count": 2,
            "swapins_bytes": 5 * 16_384,
            "swapins_pages": 5,
            "swapouts_bytes": 7 * 16_384,
            "swapouts_pages": 7,
        },
    }
    encoded = json.dumps(report, sort_keys=True)
    assert str(root) not in encoded
    assert str(models_root) not in encoded
    assert "vvvvvvvv" not in encoded
    assert list((root / "product" / "media").glob("full-ml-smoke-*")) == []
    assert list((root / "product" / "tmp").glob("full-ml-smoke-*")) == []


def test_vision_ingestion_release_failure_is_infrastructure_and_stops_before_whisper(
    tmp_path: Path,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    models_root = _models_root(tmp_path)
    events: list[str] = []

    with pytest.raises(script.SmokeInfrastructureError) as captured:
        script.execute(
            root,
            models_root=models_root,
            environ=_offline_environment(root, models_root),
            dependencies=_dependencies(
                script,
                events,
                fail_ingestion_release=True,
            ),
        )

    assert captured.value.code == "ingestion_resource_release_failed"
    assert captured.value.component == "vision"
    assert captured.value.kind == "infrastructure"
    assert events == [
        "ml-environment.attest",
        "toolchain.verify.pre",
        "workers.start",
        "workers.ready",
        "fixture",
        "vision.status",
        "vision.image",
        "vision.text",
        "vision.detect",
        "vision.release_ingestion_resources",
        "ocr.close",
        "toolchain.verify.post",
        "workers.close",
    ]


@pytest.mark.parametrize("read_outcome", ["complete", "failed", "invalid"])
def test_ocr_ingestion_release_failure_stops_before_lighthouse(
    tmp_path: Path,
    read_outcome: str,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    models_root = _models_root(tmp_path)
    events: list[str] = []

    with pytest.raises(script.SmokeInfrastructureError) as captured:
        script.execute(
            root,
            models_root=models_root,
            environ=_offline_environment(root, models_root),
            dependencies=_dependencies(
                script,
                events,
                ocr_read_outcome=read_outcome,
                fail_ocr_ingestion_release=True,
            ),
        )

    assert captured.value.code == "ingestion_resource_release_failed"
    assert captured.value.component == "ocr"
    assert captured.value.kind == "infrastructure"
    assert "private OCR" not in str(captured.value)
    assert "lighthouse.status" not in events
    assert "product.integration" not in events
    assert events[-5:] == [
        "ocr.read",
        "ocr.release_ingestion_resources",
        "ocr.close",
        "toolchain.verify.post",
        "workers.close",
    ]


@pytest.mark.parametrize(
    ("read_outcome", "error_code"),
    [("failed", "read_failed"), ("invalid", "read_contract_invalid")],
)
def test_ocr_read_failure_still_releases_ingestion_resources(
    tmp_path: Path,
    read_outcome: str,
    error_code: str,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    models_root = _models_root(tmp_path)
    events: list[str] = []

    with pytest.raises(script.SmokeError) as captured:
        script.execute(
            root,
            models_root=models_root,
            environ=_offline_environment(root, models_root),
            dependencies=_dependencies(
                script,
                events,
                ocr_read_outcome=read_outcome,
            ),
        )

    assert captured.value.code == error_code
    assert captured.value.component == "ocr"
    assert "lighthouse.status" not in events
    assert events[-5:] == [
        "ocr.read",
        "ocr.release_ingestion_resources",
        "ocr.close",
        "toolchain.verify.post",
        "workers.close",
    ]


def test_production_fixture_contains_searchable_text_and_uses_attested_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    workspace = script.SmokeWorkspace(
        image=root / "frame.jpg",
        lighthouse_cache=root / "lighthouse",
        media_video=root / "video.mp4",
        qwen_video=root / "qwen.mp4",
        video_id="smoke_fixture",
    )
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        from PIL import Image

        commands.append(command)
        with Image.open(workspace.image) as source:
            assert source.size == (960, 540)
            assert source.getbbox() is not None
        workspace.media_video.write_bytes(b"attested-synthetic-video")
        assert kwargs["env"] == {}
        return SimpleNamespace(returncode=0)

    class FFmpeg:
        def probe(self, source: Path) -> SimpleNamespace:
            assert source == workspace.media_video
            return SimpleNamespace(duration=2.0, width=960, height=540)

        def extract_frame(
            self,
            source: Path,
            destination: Path,
            timestamp: float,
            *,
            max_width: int,
        ) -> None:
            assert source == workspace.media_video
            assert destination == workspace.image
            assert timestamp == 0.5
            assert max_width == 960

    monkeypatch.setattr(script.subprocess, "run", run)
    toolchain = SimpleNamespace(
        ffmpeg_binary="/attested/ffmpeg",
        create_ffmpeg=lambda: FFmpeg(),
    )

    fixture = script.build_synthetic_fixture(toolchain, workspace)

    assert fixture.duration_seconds == 2.0
    assert fixture.qwen_video.read_bytes() == b"attested-synthetic-video"
    assert commands[0][0] == "/attested/ffmpeg"
    assert commands[0][-1] == str(workspace.media_video)
    assert "testsrc2" not in " ".join(commands[0])


def test_infrastructure_failure_is_typed_and_cli_output_is_sanitized(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    root = tmp_path / "private-user-path"
    root.mkdir(mode=0o700)
    models_root = _models_root(tmp_path)
    environment = _offline_environment(root, models_root)
    dependencies = _dependencies(script, [], fail_image=True)

    real_execute = script.execute

    def execute(
        _root: Path,
        *,
        models_root: Path,
        environ: object,
    ) -> object:
        return real_execute(
            root,
            models_root=models_root,
            environ=environment,
            dependencies=dependencies,
        )

    monkeypatch.setattr(script, "execute", execute)
    exit_code = script.main(
        ["--root", str(root), "--models-root", str(models_root)]
    )
    captured = capsys.readouterr()

    assert exit_code == script.EXIT_INFRASTRUCTURE
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload == {
        "error": {
            "code": "image_embedding_failed",
            "component": "vision",
            "kind": "infrastructure",
        },
        "oom": {
            "reason_code": "worker_failure_contract_missing_oom_code",
            "status": "unknown",
        },
        "schema_version": 2,
        "status": "failed",
    }
    assert "secret-token" not in captured.err
    assert str(root) not in captured.err


def test_product_integration_receipt_can_complete_the_full_smoke(
    tmp_path: Path,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    models_root = _models_root(tmp_path)
    events: list[str] = []
    dependencies = _dependencies(script, events)
    dependencies = script.SmokeDependencies(
        attest_ml_environment=dependencies.attest_ml_environment,
        attest_toolchain=dependencies.attest_toolchain,
        build_clients=dependencies.build_clients,
        build_fixture=dependencies.build_fixture,
        build_resource_monitor=dependencies.build_resource_monitor,
        code_identity_resolver=dependencies.code_identity_resolver,
        load_settings=dependencies.load_settings,
        product_integration=lambda *_args: _complete_product_receipt(),
        start_workers=dependencies.start_workers,
    )

    report = script.execute(
        root,
        models_root=models_root,
        environ=_offline_environment(root, models_root),
        dependencies=dependencies,
    )

    assert report["status"] == "ready"
    assert report["product_integration"]["status"] == "complete"


def test_rejects_code_identity_drift_after_cleanup(tmp_path: Path) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    models_root = _models_root(tmp_path)
    identities = iter(("a" * 40, "b" * 40))
    dependencies = _dependencies(script, [])
    dependencies = dependencies._replace(
        code_identity_resolver=lambda: next(identities)
    )

    with pytest.raises(script.SmokeContractError) as captured:
        script.execute(
            root,
            models_root=models_root,
            environ=_offline_environment(root, models_root),
            dependencies=dependencies,
        )

    assert captured.value.code == "code_identity_changed"


def test_production_product_integration_uses_durable_generation_pinned_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    models = tmp_path / "immutable-models"
    models.mkdir(mode=0o755)
    fixture_path = root / "fixture.mp4"
    fixture_path.write_bytes(b"synthetic-video")
    fixture = script.SmokeFixture(
        duration_seconds=1.0,
        image=root / "frame.jpg",
        media_video=fixture_path,
        qwen_video=root / "qwen.mp4",
        video_id="smoke_fixture",
    )
    settings = script._DisposableProductSettings(
        data_dir=root / "product",
        ffmpeg_binary=Path("/attested/ffmpeg"),
        ffprobe_binary=Path("/attested/ffprobe"),
        smoke_immutable_models_dir=models,
        ocr_worker_environment={
            "HOME": str(root / "tmp" / "worker-ocr-product"),
        },
    )
    events: list[str] = []

    class Repository:
        def __init__(self, path: Path) -> None:
            assert path == root / "product" / "videoscope.sqlite3"

        def initialize(self) -> None:
            events.append("repository.initialize")

        def get_video_index_job(self, job_id: str) -> SimpleNamespace:
            assert job_id == "job-1"
            from videoscope.jobs import JobState

            return SimpleNamespace(state=JobState.COMPLETE)

        def find_assets_by_sha256_bounded(
            self,
            digest: str,
            *,
            limit: int,
            video_id: str,
        ) -> tuple[SimpleNamespace, ...]:
            assert len(digest) == 64 and limit == 1
            return (
                SimpleNamespace(
                    id="sha256:" + digest,
                    sha256=digest,
                    byte_size=len(b"synthetic-video"),
                    duration_seconds=1.0,
                    video_id=video_id,
                ),
            )

    class Queue:
        def wake(self) -> None:
            events.append("queue.wake")

    from videoscope.benchmark.catalog import ResolvedAsset
    from videoscope.benchmark.runner import BenchmarkSearchHit

    expected_probes = {
        "lexical_qdrant": (),
        "dense_siglip": ("vision",),
        "temporal_refinement": ("vision",),
        "lighthouse": ("vision",),
        "qwen_verification": ("qwen", "vision"),
    }

    class Session:
        def __init__(self, profile_id: str) -> None:
            self.profile_id = profile_id

        def capability_state(self, _asset: object, _capability: str) -> str:
            return "complete"

        def identities(self) -> SimpleNamespace:
            return SimpleNamespace(
                index_identities=tuple(
                    SimpleNamespace(component_id=value)
                    for value in (
                        "text_vector_generations",
                        "visual_generations",
                        "lighthouse_generations",
                    )
                ),
                config_identities=(
                    SimpleNamespace(component_id="benchmark_product_environment"),
                ),
            )

        def search(
            self,
            _query: str,
            assets: tuple[ResolvedAsset, ...],
            *,
            limit: int,
        ) -> tuple[BenchmarkSearchHit, ...]:
            events.append(f"search.{self.profile_id}")
            assert limit == 50
            return (
                BenchmarkSearchHit(
                    asset_id=assets[0].asset_id,
                    start_seconds=0.0,
                    end_seconds=1.0,
                    score=0.75,
                ),
            )

        def last_search_execution_receipt(self) -> ProductSearchExecutionReceipt:
            return _execution_receipt(self.profile_id)

        def close(self) -> None:
            events.append(f"session.close.{self.profile_id}")

    class AssetResolver:
        def __init__(self, profile_id: str) -> None:
            self.profile_id = profile_id

        def resolve(self, portable: object) -> ResolvedAsset:
            events.append(f"asset.resolve.{self.profile_id}")
            return ResolvedAsset(
                asset_id=portable.asset_id,
                repository_asset_id="sha256:" + portable.sha256,
                video_id="local-video-a",
                sha256=portable.sha256,
                byte_size=portable.byte_size,
                duration_seconds=portable.duration_seconds,
            )

    class SearchAdapter:
        def __init__(self, profile_id: str) -> None:
            self.profile_id = profile_id

        def open_session(
            self,
            profile: object,
            _assets: tuple[ResolvedAsset, ...],
            *,
            execution_mode: str,
        ) -> Session:
            assert profile.profile_id == self.profile_id
            assert execution_mode == "warm"
            events.append(f"search.open_pinned.{self.profile_id}")
            return Session(self.profile_id)

    class Environment:
        def __init__(self, profile_id: str) -> None:
            self.profile_id = profile_id
            self.asset_resolver = AssetResolver(profile_id)
            self.search_adapter = SearchAdapter(profile_id)
            self.is_closed = False

        def probe_worker_sources(self) -> tuple[str, ...]:
            events.append(f"environment.probe.{self.profile_id}")
            return expected_probes[self.profile_id]

        def close(self) -> None:
            events.append(f"environment.close.{self.profile_id}")
            self.is_closed = True

    class Runtime:
        queue = Queue()
        closed = False

        class Clips:
            @staticmethod
            def export(name: str, selections: list[object]) -> SimpleNamespace:
                events.append("clips.export")
                assert name == "phase0-full-path"
                assert len(selections) == 1
                selection = selections[0]
                assert selection.start == 0.0 and selection.end == 1.0
                destination = settings.clips_dir / "phase0-full-path.mp4"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"synthetic-clip")
                return SimpleNamespace(
                    path=destination,
                    duration=1.0,
                )

        clips = Clips()
        video_index_plan_factory = staticmethod(lambda: object())

        def start(self) -> None:
            events.append("runtime.start")

        def close(self) -> bool:
            events.append("runtime.close")
            self.closed = True
            return True

    class Coordinator:
        def __init__(self, _repository: object, **_kwargs: object) -> None:
            return None

        def create_ingest(self, **kwargs: object) -> tuple[object, SimpleNamespace]:
            events.append("coordinator.create_ingest")
            assert Path(str(kwargs["media_path"])).is_file()
            return object(), SimpleNamespace(job_id="job-1")

    monkeypatch.setattr("videoscope.repository.Repository", Repository)
    product_runtime = Runtime()

    def build_runtime(*_args: object, **kwargs: object) -> Runtime:
        assert kwargs["ocr_worker_environment"] == settings.ocr_worker_environment
        assert kwargs["vision_worker_input_root"] == root
        return product_runtime

    monkeypatch.setattr("videoscope.runtime.build_runtime", build_runtime)
    monkeypatch.setattr(
        "videoscope.processing.coordinator.VideoIndexCoordinator",
        Coordinator,
    )

    def open_environment(
        candidate_settings: object,
        scratch_parent: Path,
        *,
        profile_id: str,
        execution_mode: str,
    ) -> Environment:
        assert candidate_settings is settings
        assert scratch_parent == root
        assert execution_mode == "warm"
        assert product_runtime.closed is True
        events.append(f"environment.open.{profile_id}")
        return Environment(profile_id)

    monkeypatch.setattr(
        "videoscope.benchmark.open_product_benchmark_environment",
        open_environment,
    )

    class ExportLock:
        def __init__(self, data_dir: Path) -> None:
            assert data_dir == settings.data_dir

        def acquire_existing(self) -> None:
            assert all(
                f"environment.close.{profile_id}" in events
                for profile_id in expected_probes
            )
            events.append("export-lock.acquire")

        def close(self) -> None:
            events.append("export-lock.close")

    monkeypatch.setattr(
        "videoscope.runtime_lifecycle.ExclusiveRuntimeLock",
        ExportLock,
    )

    class Toolchain:
        @staticmethod
        def create_ffmpeg() -> SimpleNamespace:
            return SimpleNamespace(
                probe=lambda path: (
                    events.append("clip.probe")
                    or SimpleNamespace(duration=1.0)
                )
            )

    receipt = script.run_disposable_product_integration(
        root,
        fixture,
        script.SmokeClients(*(object() for _ in range(5))),
        Toolchain(),
        settings,
    )

    assert receipt == _complete_product_receipt()
    assert events == [
        "repository.initialize",
        "runtime.start",
        "coordinator.create_ingest",
        "runtime.close",
        "environment.open.lexical_qdrant",
        "environment.probe.lexical_qdrant",
        "asset.resolve.lexical_qdrant",
        "search.open_pinned.lexical_qdrant",
        "search.lexical_qdrant",
        "session.close.lexical_qdrant",
        "environment.close.lexical_qdrant",
        "environment.open.dense_siglip",
        "environment.probe.dense_siglip",
        "asset.resolve.dense_siglip",
        "search.open_pinned.dense_siglip",
        "search.dense_siglip",
        "session.close.dense_siglip",
        "environment.close.dense_siglip",
        "environment.open.temporal_refinement",
        "environment.probe.temporal_refinement",
        "asset.resolve.temporal_refinement",
        "search.open_pinned.temporal_refinement",
        "search.temporal_refinement",
        "session.close.temporal_refinement",
        "environment.close.temporal_refinement",
        "environment.open.lighthouse",
        "environment.probe.lighthouse",
        "asset.resolve.lighthouse",
        "search.open_pinned.lighthouse",
        "search.lighthouse",
        "session.close.lighthouse",
        "environment.close.lighthouse",
        "environment.open.qwen_verification",
        "environment.probe.qwen_verification",
        "asset.resolve.qwen_verification",
        "search.open_pinned.qwen_verification",
        "search.qwen_verification",
        "session.close.qwen_verification",
        "environment.close.qwen_verification",
        "export-lock.acquire",
        "clips.export",
        "export-lock.close",
        "clip.probe",
    ]


@pytest.mark.parametrize(
    ("profile_id", "missing_component"),
    (
        ("lexical_qdrant", "text_vectors"),
        ("dense_siglip", "visual_dense"),
        ("temporal_refinement", "temporal_refinement"),
        ("lighthouse", "lighthouse"),
        ("qwen_verification", "qwen_verification"),
    ),
)
def test_product_profile_rejects_generic_evidence_substitution(
    profile_id: str,
    missing_component: str,
) -> None:
    script = _load_script()
    profile = FROZEN_PROFILES[profile_id]
    receipt = _execution_receipt(
        profile_id,
        zero_component=missing_component,
        include_generic_lexical=True,
    )

    assert receipt.total_evidence_count > 0
    assert receipt.component_evidence_count("lexical_text") == 1
    with pytest.raises(script.SmokeContractError) as captured:
        script._validate_product_search_execution_receipt(profile, receipt)

    assert captured.value.code == "product_search_component_execution_unproven"


@pytest.mark.parametrize("component_id", _EXECUTION_COMPONENT_IDS)
@pytest.mark.parametrize("count_kind", ("output", "evidence"))
def test_cumulative_profile_rejects_each_zero_selected_component_count(
    component_id: str,
    count_kind: str,
) -> None:
    script = _load_script()
    profile = FROZEN_PROFILES["qwen_verification"]
    receipt = _execution_receipt("qwen_verification")
    field_name = f"component_{count_kind}_counts"
    receipt = replace(
        receipt,
        **{
            field_name: tuple(
                (observed_id, 0 if observed_id == component_id else count)
                for observed_id, count in getattr(receipt, field_name)
            )
        },
    )

    assert receipt.total_evidence_count > 0
    assert sum(count for _, count in receipt.component_evidence_counts) > 0
    with pytest.raises(script.SmokeContractError) as captured:
        script._validate_product_search_execution_receipt(profile, receipt)

    assert captured.value.code == "product_search_component_execution_unproven"


def test_product_profile_rejects_execution_receipt_from_another_frozen_profile() -> None:
    script = _load_script()

    with pytest.raises(script.SmokeContractError) as captured:
        script._validate_product_search_execution_receipt(
            FROZEN_PROFILES["temporal_refinement"],
            _execution_receipt("dense_siglip"),
        )

    assert captured.value.code == "product_search_execution_receipt_invalid"


def test_serialized_component_execution_rejects_nested_profile_id() -> None:
    script = _load_script()
    receipt = _complete_product_receipt()
    profiles = receipt["profiles"]
    assert isinstance(profiles, dict)
    profile_receipt = profiles["lexical_qdrant"]
    assert isinstance(profile_receipt, dict)
    component_execution = profile_receipt["component_execution"]
    assert isinstance(component_execution, dict)
    component_execution["profile_id"] = "lexical_qdrant"

    with pytest.raises(script.SmokeContractError) as captured:
        script._validate_product_receipt(receipt)

    assert captured.value.code == "product_integration_receipt_invalid"


def test_product_profile_rejects_zero_input_qwen_invocation_with_generic_evidence() -> None:
    script = _load_script()
    profile = FROZEN_PROFILES["qwen_verification"]
    receipt = _execution_receipt("qwen_verification")
    receipt = replace(
        receipt,
        component_input_counts=tuple(
            (component_id, 0 if component_id == "qwen_verification" else count)
            for component_id, count in receipt.component_input_counts
        ),
    )

    assert receipt.total_evidence_count > 0
    with pytest.raises(script.SmokeContractError) as captured:
        script._validate_product_search_execution_receipt(profile, receipt)

    assert captured.value.code == "product_search_component_execution_unproven"


def test_product_snapshot_cleanup_retries_and_preserves_cancellation() -> None:
    script = _load_script()

    class RecoverableEnvironment:
        is_closed = False
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("retry cleanup")
            self.is_closed = True

    recovered = RecoverableEnvironment()
    script._close_product_snapshot_environment(recovered)
    assert recovered.close_calls == 2
    assert recovered.is_closed is True

    class InterruptedEnvironment:
        is_closed = False
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise KeyboardInterrupt
            self.is_closed = True

    interrupted = InterruptedEnvironment()
    with pytest.raises(KeyboardInterrupt):
        script._close_product_snapshot_environment(interrupted)
    assert interrupted.close_calls == 2
    assert interrupted.is_closed is True

    class ProductCleanup:
        cleanup_pending = True
        retry_calls = 0

        def retry_cleanup(self) -> None:
            self.retry_calls += 1
            if self.retry_calls == 1:
                raise RuntimeError("retry product snapshot cleanup")
            self.cleanup_pending = False

    product_cleanup = ProductCleanup()
    script._retry_product_snapshot_cleanup(product_cleanup)
    assert product_cleanup.retry_calls == 2
    assert product_cleanup.cleanup_pending is False


def test_out_of_memory_is_not_collapsed_into_infrastructure_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    root = tmp_path / "private-user-path"
    root.mkdir(mode=0o700)
    models_root = _models_root(tmp_path)
    environment = _offline_environment(root, models_root)
    dependencies = _dependencies(script, [], oom_image=True)
    real_execute = script.execute

    def execute(
        _root: Path,
        *,
        models_root: Path,
        environ: object,
    ) -> object:
        return real_execute(
            root,
            models_root=models_root,
            environ=environment,
            dependencies=dependencies,
        )

    monkeypatch.setattr(script, "execute", execute)
    exit_code = script.main(
        ["--root", str(root), "--models-root", str(models_root)]
    )
    captured = capsys.readouterr()

    assert exit_code == script.EXIT_OUT_OF_MEMORY
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {
            "code": "out_of_memory",
            "component": "vision",
            "kind": "out_of_memory",
        },
        "oom": {"status": "observed"},
        "schema_version": 2,
        "status": "failed",
    }
    assert "private allocator diagnostics" not in captured.err


def test_production_dependencies_are_real_client_only() -> None:
    script = _load_script()

    dependencies = script.production_dependencies()

    assert (
        dependencies.product_integration
        is script.run_disposable_product_integration
    )
    assert dependencies.attest_toolchain is script._attest_current_toolchain
    assert dependencies.build_clients is script.build_real_clients
    assert dependencies.build_fixture is script.build_synthetic_fixture
    assert dependencies.build_resource_monitor is script.build_resource_monitor
    assert dependencies.code_identity_resolver is script.resolve_clean_code_sha
    assert dependencies.start_workers is script.start_managed_workers


def test_real_clients_use_full_cold_start_health_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    vision_captured: dict[str, object] = {}
    whisper_captured: dict[str, object] = {}
    dummy = SimpleNamespace(close=lambda: None)

    def build_vision(**kwargs: object) -> object:
        vision_captured.update(kwargs)
        return dummy

    def build_whisper(**kwargs: object) -> object:
        whisper_captured.update(kwargs)
        return dummy

    monkeypatch.setattr(script, "VisionWorkerClient", build_vision)
    monkeypatch.setattr(script, "WhisperWorkerClient", build_whisper)
    monkeypatch.setattr(script, "PaddleOCRReader", lambda **_kwargs: dummy)
    monkeypatch.setattr(script, "LighthouseWorkerClient", lambda **_kwargs: dummy)
    monkeypatch.setattr(script, "QwenWorkerClient", lambda **_kwargs: dummy)
    monkeypatch.setattr(script, "_reviewed_ocr_script_sha256", lambda _path: _SHA)
    models_root = tmp_path / "models"
    models_root.mkdir()
    settings = script._DisposableProductSettings(
        data_dir=tmp_path,
        ffmpeg_binary=Path("/attested/ffmpeg"),
        ffprobe_binary=Path("/attested/ffprobe"),
        smoke_immutable_models_dir=models_root,
        vision_worker_endpoint="http://127.0.0.1:8783",
        vision_worker_api_key="v" * 32,
        whisper_worker_endpoint="http://127.0.0.1:8784",
        whisper_worker_api_key="w" * 32,
        lighthouse_endpoint="http://127.0.0.1:8782",
        lighthouse_api_key="l" * 32,
        qwen_video_endpoint="http://127.0.0.1:8781",
        qwen_video_api_key="q" * 32,
        qwen_video_model=script.QWEN_VIDEO_MODEL,
    )

    script.build_real_clients(settings, tmp_path)

    assert vision_captured["timeout"] == settings.vision_worker_timeout
    assert (
        vision_captured["health_timeout"]
        == script._WORKER_START_TIMEOUT_SECONDS
    )
    assert whisper_captured["timeout"] == settings.whisper_worker_timeout
    assert (
        whisper_captured["health_timeout"]
        == script._WORKER_START_TIMEOUT_SECONDS
    )


def test_managed_workers_use_explicit_isolated_pythons_and_native_pid_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    (root / "media").mkdir(mode=0o700)
    (root / "tmp").mkdir(mode=0o700)
    (root / "cache").mkdir(mode=0o700)
    models_root = _models_root(tmp_path)
    environment = _offline_environment(root, models_root)
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    environment["HF_HOME"] = str(hf_home)
    media_bin = tmp_path / "media-bin"
    media_bin.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        executable = media_bin / name
        executable.write_bytes(b"#!/bin/sh\n")
        executable.chmod(0o700)
        environment[f"VIDEOSCOPE_FULL_ML_SMOKE_{name.upper()}_BINARY"] = str(
            executable
        )
    executable_by_role: dict[str, Path] = {}
    for role in ("vision", "whisper", "lighthouse", "qwen", "ocr"):
        executable = tmp_path / f"{role}-python"
        executable.write_bytes(b"#!/bin/sh\n")
        executable.chmod(0o700)
        executable_by_role[role] = executable
        environment[f"VIDEOSCOPE_FULL_ML_SMOKE_{role.upper()}_PYTHON"] = str(
            executable
        )

    launches: list[tuple[list[str], dict[str, object]]] = []

    class Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, *, timeout: float) -> int:
            assert timeout == 5.0
            assert self.returncode is not None
            return self.returncode

    def popen(command: list[str], **kwargs: object) -> Process:
        launches.append((command, kwargs))
        return Process(200 + len(launches))

    records = tuple(
        script.ProcessRecord(
            pid=200 + index,
            parent_pid=os.getpid(),
            rss_bytes=index * 1000,
            start_token=f"start-{index}",
            executable_identity="path-sha256:" + str(index) * 64,
        )
        for index in range(1, 5)
    )
    monkeypatch.setattr(script.subprocess, "Popen", popen)
    monkeypatch.setattr(
        script,
        "_allocate_loopback_ports",
        lambda: {
            "vision": 48101,
            "whisper": 48102,
            "lighthouse": 48103,
            "qwen": 48104,
        },
    )
    snapshots = iter(((), records))
    monkeypatch.setattr(
        script,
        "create_native_process_snapshot_provider",
        lambda **_kwargs: SimpleNamespace(snapshot=lambda: next(snapshots)),
    )

    cluster = script.start_managed_workers(root, models_root, environment)

    assert script._validate_worker_cluster(cluster) is cluster
    assert cluster.all_workers_managed is True
    assert tuple(item.role for item in cluster.managed_workers) == (
        "vision",
        "whisper",
        "lighthouse",
        "qwen",
    )
    assert len(launches) == 4
    for role, (command, kwargs) in zip(
        ("vision", "whisper", "lighthouse", "qwen"),
        launches,
        strict=True,
    ):
        assert command == [
            str(executable_by_role[role]),
            "-m",
            f"videoscope.providers.{role}_worker",
        ]
        assert kwargs["cwd"] == root
        child_environment = kwargs["env"]
        assert child_environment["HF_HUB_OFFLINE"] == "1"
        assert child_environment["TRANSFORMERS_OFFLINE"] == "1"
        assert child_environment["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] == "True"
        assert child_environment["PATH"] == str(media_bin)
        assert child_environment["HOME"] == str(root / "tmp" / f"worker-{role}")
        assert child_environment["TMPDIR"] == str(root / "tmp" / f"worker-{role}")
        assert child_environment["HF_HOME"] == str(hf_home)
    assert launches[0][1]["env"]["VIDEOSCOPE_VISION_WORKER_INPUT_ROOT"] == str(root)
    assert launches[0][1]["env"][
        "VIDEOSCOPE_VISION_WORKER_PRODUCT_DATA_SUBDIRECTORY"
    ] == "product"
    assert launches[1][1]["env"]["VIDEOSCOPE_WHISPER_WORKER_INPUT_ROOT"] == str(
        root / "product" / "media"
    )
    assert launches[2][1]["env"]["VIDEOSCOPE_DATA_DIR"] == str(
        root / "product"
    )
    assert launches[3][1]["env"]["VIDEOSCOPE_QWEN_WORKER_INPUT_ROOT"] == str(
        root
    )
    for setting, role in (
        ("smoke_ocr_client_environment", "ocr-client"),
        ("ocr_worker_environment", "ocr-product"),
    ):
        private_root = root / "tmp" / f"worker-{role}"
        assert cluster.settings_overrides[setting] == {
            "HF_HOME": str(hf_home),
            "HOME": str(private_root),
            "PATH": str(media_bin),
            "TMPDIR": str(private_root),
            "XDG_CACHE_HOME": str(private_root / "cache"),
        }
    assert len(
        {
            cluster.settings_overrides[f"{role if role != 'qwen' else 'qwen_video'}_api_key"]
            for role in ("vision_worker", "whisper_worker", "lighthouse", "qwen")
        }
    ) == 4
    cluster.close()
    assert all(process.poll() == 0 for process in (item.process for item in cluster._processes))


def test_worker_cluster_rejects_non_exact_managed_ocr_environments(
    tmp_path: Path,
) -> None:
    script = _load_script()
    private_root = tmp_path / "worker-ocr-client"
    valid_environment = {
        "HF_HOME": str(tmp_path / "hf-home"),
        "HOME": str(private_root),
        "PATH": str(tmp_path / "media-bin"),
        "TMPDIR": str(private_root),
        "XDG_CACHE_HOME": str(private_root / "cache"),
    }
    settings_overrides = {
        "ocr_worker_python": tmp_path / "ocr-python",
        "ocr_worker_script": tmp_path / "paddle-ocr-worker.py",
        "smoke_ocr_client_environment": dict(valid_environment),
        "ocr_worker_environment": {
            **valid_environment,
            "HOME": str(tmp_path / "worker-ocr-product"),
            "TMPDIR": str(tmp_path / "worker-ocr-product"),
            "XDG_CACHE_HOME": str(
                tmp_path / "worker-ocr-product" / "cache"
            ),
        },
        "vision_worker_endpoint": "http://127.0.0.1:48101",
        "vision_worker_api_key": "vision-token",
        "whisper_worker_endpoint": "http://127.0.0.1:48102",
        "whisper_worker_api_key": "whisper-token",
        "lighthouse_endpoint": "http://127.0.0.1:48103",
        "lighthouse_api_key": "lighthouse-token",
        "qwen_video_endpoint": "http://127.0.0.1:48104",
        "qwen_video_api_key": "qwen-token",
    }
    bindings = tuple(
        script.ManagedProcessBinding(
            pid=300 + index,
            start_token=f"start-{index}",
            executable_identity=f"worker-{index}@1",
            role=role,
        )
        for index, role in enumerate(
            ("vision", "whisper", "lighthouse", "qwen"),
            start=1,
        )
    )

    def cluster(overrides: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(
            settings_overrides=overrides,
            managed_workers=bindings,
            all_workers_managed=True,
            wait_ready=lambda: None,
            close=lambda: None,
        )

    assert script._validate_worker_cluster(cluster(settings_overrides))
    for field, replacement in (
        (
            "smoke_ocr_client_environment",
            {**valid_environment, "UNMANAGED": "1"},
        ),
        (
            "ocr_worker_environment",
            {
                key: value
                for key, value in settings_overrides[
                    "ocr_worker_environment"
                ].items()
                if key != "TMPDIR"
            },
        ),
    ):
        invalid = {**settings_overrides, field: replacement}
        with pytest.raises(script.SmokeContractError) as captured:
            script._validate_worker_cluster(cluster(invalid))
        assert captured.value.code == "managed_worker_cluster_contract_invalid"


def test_smoke_ocr_environment_is_contained_and_ignores_ambient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    (root / "tmp").mkdir(mode=0o700)
    models_root = _models_root(tmp_path)
    environment = _offline_environment(root, models_root)
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    environment["HF_HOME"] = str(hf_home)
    media_bin = tmp_path / "media-bin"
    media_bin.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        executable = media_bin / name
        executable.write_bytes(b"#!/bin/sh\n")
        executable.chmod(0o700)
        environment[f"VIDEOSCOPE_FULL_ML_SMOKE_{name.upper()}_BINARY"] = str(
            executable
        )
    for name in ("HOME", "PATH", "TMPDIR", "XDG_CACHE_HOME"):
        monkeypatch.setenv(name, f"/hostile/{name.casefold()}")

    isolated = script._isolated_ocr_worker_environment(
        root,
        environment,
        role="ocr-client",
    )

    private_root = root / "tmp" / "worker-ocr-client"
    assert isolated == {
        "HF_HOME": str(hf_home),
        "HOME": str(private_root),
        "PATH": str(media_bin),
        "TMPDIR": str(private_root),
        "XDG_CACHE_HOME": str(private_root / "cache"),
    }
    assert private_root.stat().st_mode & 0o777 == 0o700
    assert (private_root / "cache").stat().st_mode & 0o777 == 0o700


def test_native_resource_monitor_rejects_orchestrator_only_measurement() -> None:
    script = _load_script()

    with pytest.raises(script.SmokeConfigurationError) as captured:
        script.build_resource_monitor(())

    assert captured.value.code == "external_worker_process_binding_unavailable"


def test_native_resource_monitor_runs_process_and_host_samplers_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    events: list[str] = []
    bindings = tuple(
        script.ManagedProcessBinding(
            pid=300 + index,
            start_token=f"start-{index}",
            executable_identity=f"worker-{index}@1",
            role=role,
        )
        for index, role in enumerate(
            ("vision", "whisper", "lighthouse", "qwen"),
            start=1,
        )
    )

    class ProcessSampler:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["managed_workers"] == bindings

        def start(self) -> None:
            events.append("process.start")

        def finish(self) -> SimpleNamespace:
            events.append("process.finish")
            return SimpleNamespace(
                baseline_bytes=100,
                increment_bytes=50,
                peak_bytes=150,
                sample_count=2,
                samples_bytes=(100, 150),
            )

        def close(self) -> None:
            events.append("process.close")

    class HostSampler:
        def __init__(self, provider: object) -> None:
            assert provider == "host-provider"

        def start(self) -> None:
            events.append("host.start")

        def finish(self) -> object:
            events.append("host.finish")
            return _host_resource_receipt()

    monkeypatch.setattr(
        script,
        "create_native_process_snapshot_provider",
        lambda **_kwargs: "process-provider",
    )
    monkeypatch.setattr(
        script,
        "create_host_resource_snapshot_provider",
        lambda: "host-provider",
    )
    monkeypatch.setattr(script, "ProcessTreeRssSampler", ProcessSampler)
    monkeypatch.setattr(script, "HostResourceSampler", HostSampler)

    monitor = script.build_resource_monitor(bindings)
    monitor.start()
    measurement = monitor.finish()
    monitor.close()

    assert measurement["host_resources"] == _host_resource_receipt()
    assert measurement["process_tree"].peak_bytes == 150
    assert events == [
        "host.start",
        "process.start",
        "process.finish",
        "host.finish",
        "process.close",
    ]
