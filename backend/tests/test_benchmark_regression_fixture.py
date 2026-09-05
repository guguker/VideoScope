from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from videoscope.benchmark.regression_fixture import (
    BASELINE_BATCH_RECEIPT_FILENAME,
    BENCHMARK_BINDINGS_FILENAME,
    FIXTURE_DATASET_ID,
    FIXTURE_DATASET_REVISION,
    FIXTURE_DATASET_VERSION,
    LOCAL_BINDINGS_SCHEMA_VERSION,
    PROVISION_RECEIPT_FILENAME,
    ProvisionedAsset,
    RegressionFixtureError,
    load_local_input_bindings,
    provision_regression_fixture,
    run_required_product_profiles,
)
from videoscope.benchmark.measurements import ManagedProcessBinding
from videoscope.benchmark.measurements import MeasurementError
from videoscope.benchmark.runner import BenchmarkExecutionError
from videoscope.benchmark.serialization import dataset_revision
from videoscope.benchmark.storage import load_dataset
from videoscope.benchmark.video_verifier_schema import video_verifier_dataset_from_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_FIXTURE = (
    PROJECT_ROOT / "docs" / "benchmarks" / "product-retrieval" / "seed-v1.json"
)
VERIFIER_FIXTURE = (
    PROJECT_ROOT / "docs" / "benchmarks" / "video-verifier" / "seed-v1.json"
)


def _load_verifier_fixture():  # type: ignore[no-untyped-def]
    return video_verifier_dataset_from_json(VERIFIER_FIXTURE.read_bytes())


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _local_inputs(tmp_path: Path) -> tuple[dict[str, Path], dict[str, bytes]]:
    dataset = load_dataset(PRODUCT_FIXTURE)
    paths: dict[str, Path] = {}
    payloads: dict[str, bytes] = {}
    for index, asset in enumerate(dataset.assets):
        payload = f"fixture-{index}".encode()
        # This helper is used only for parser tests; byte attestation is exercised
        # by the provision test with a tiny dataset-specific driver below.
        path = tmp_path / f"{asset.asset_id}.mp4"
        path.write_bytes(payload)
        paths[asset.asset_id] = path
        payloads[asset.asset_id] = payload
    return paths, payloads


def _bindings_payload(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "schema_version": LOCAL_BINDINGS_SCHEMA_VERSION,
        "dataset_revision": FIXTURE_DATASET_REVISION,
        "inputs": [
            {"asset_id": asset_id, "path": str(path.resolve())}
            for asset_id, path in reversed(tuple(paths.items()))
        ],
    }


def test_committed_product_fixture_is_content_addressed_and_non_promotion() -> None:
    product = load_dataset(PRODUCT_FIXTURE)
    verifier = _load_verifier_fixture()

    assert product.dataset_id == FIXTURE_DATASET_ID
    assert product.dataset_version == FIXTURE_DATASET_VERSION
    assert dataset_revision(product) == FIXTURE_DATASET_REVISION
    assert len(product.assets) == len(verifier.cases) == 10
    assert {asset.asset_id for asset in product.assets} == {
        case.case_id for case in verifier.cases
    }
    verifier_by_case = {case.case_id: case for case in verifier.cases}
    for asset in product.assets:
        source = verifier_by_case[asset.asset_id]
        assert asset.sha256 == source.prepared_input_sha256
        assert asset.byte_size == source.prepared_input_byte_size
        assert asset.provenance.license_id == "LicenseRef-All-Rights-Reserved"

    assert product.cases
    assert all(case.label_quality == "gold" for case in product.cases)
    assert all(case.critical_slices is not None for case in product.cases)
    assert "regression_seen" in product.description
    assert "not promotion evidence" in product.description


def test_committed_fixture_keeps_every_prepared_clip_in_its_source_group() -> None:
    product = load_dataset(PRODUCT_FIXTURE)
    verifier = _load_verifier_fixture()
    source_group_by_asset = {case.case_id: case.split_group for case in verifier.cases}

    referenced = set()
    for case in product.cases:
        referenced.update(case.asset_ids)
        assert {source_group_by_asset[item] for item in case.asset_ids} == {
            case.split_group
        }

    assert referenced == {asset.asset_id for asset in product.assets}
    assert {case.split_group for case in product.cases} == {
        "panel-a",
        "sports-game-a",
    }


def test_committed_fixture_contains_no_paths_or_private_transcripts() -> None:
    payload = json.loads(PRODUCT_FIXTURE.read_text(encoding="utf-8"))
    encoded = json.dumps(payload, ensure_ascii=False)

    assert '"path"' not in encoded
    assert "transcript" not in encoded.casefold()
    assert "/Users/" not in encoded
    assert "data/" not in encoded


def test_local_bindings_require_exact_revision_aliases_and_absolute_paths(
    tmp_path: Path,
) -> None:
    dataset = load_dataset(PRODUCT_FIXTURE)
    paths, _payloads = _local_inputs(tmp_path)
    manifest = tmp_path / "bindings.json"
    _write_json(manifest, _bindings_payload(paths))

    loaded = load_local_input_bindings(manifest, dataset)

    assert loaded.dataset_revision == FIXTURE_DATASET_REVISION
    assert tuple(item.asset_id for item in loaded.inputs) == tuple(
        sorted(paths)
    )
    assert all(item.path.is_absolute() for item in loaded.inputs)

    missing = _bindings_payload(paths)
    missing["inputs"] = missing["inputs"][:-1]  # type: ignore[index]
    _write_json(manifest, missing)
    with pytest.raises(RegressionFixtureError, match="bindings_alias_mismatch"):
        load_local_input_bindings(manifest, dataset)

    relative = _bindings_payload(paths)
    relative["inputs"][0]["path"] = "relative.mp4"  # type: ignore[index]
    _write_json(manifest, relative)
    with pytest.raises(RegressionFixtureError, match="binding_path_invalid"):
        load_local_input_bindings(manifest, dataset)


def test_local_bindings_reject_symlink_manifest(tmp_path: Path) -> None:
    dataset = load_dataset(PRODUCT_FIXTURE)
    paths, _payloads = _local_inputs(tmp_path)
    target = tmp_path / "real-bindings.json"
    link = tmp_path / "bindings.json"
    _write_json(target, _bindings_payload(paths))
    link.symlink_to(target)

    with pytest.raises(Exception, match="symbolic link"):
        load_local_input_bindings(link, dataset)


class _FixtureDriver:
    def __init__(self, durations: dict[str, float]) -> None:
        self.durations = durations
        self.received: tuple[object, ...] = ()

    def probe_duration(self, asset_id: str, path: Path) -> float:
        assert path.is_file()
        return self.durations[asset_id]

    def provision(self, *, data_root: Path, assets, timeout_seconds: float):  # type: ignore[no-untyped-def]
        assert timeout_seconds == 45.0
        self.received = tuple(assets)
        return tuple(
            ProvisionedAsset(
                asset_id=item.asset_id,
                video_id=f"reg_{item.asset_id}",
                job_id=f"reg_job_{item.asset_id}",
                plan_hash=sha256(item.asset_id.encode()).hexdigest(),
                sha256=item.sha256,
                byte_size=item.byte_size,
                duration_seconds=item.duration_seconds,
            )
            for item in assets
        )


def test_provision_stages_exact_bytes_and_writes_only_sanitized_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_dataset(PRODUCT_FIXTURE)
    source_root = tmp_path / "private-inputs"
    source_root.mkdir()
    paths: dict[str, Path] = {}
    payload_by_asset: dict[str, bytes] = {}
    # Use small files while retaining the committed fixture contract by replacing
    # the loader only at the I/O boundary. This keeps the safety logic exercised
    # without committing or copying private media into the test suite.
    for index, asset in enumerate(dataset.assets):
        payload = (f"asset-{index}-" * 3).encode()
        path = source_root / f"{asset.asset_id}.mp4"
        path.write_bytes(payload)
        paths[asset.asset_id] = path
        payload_by_asset[asset.asset_id] = payload

    from videoscope.benchmark import regression_fixture as subject

    tiny = replace(
        dataset,
        assets=tuple(
            replace(
                asset,
                sha256=sha256(payload_by_asset[asset.asset_id]).hexdigest(),
                byte_size=len(payload_by_asset[asset.asset_id]),
            )
            for asset in dataset.assets
        ),
    )
    monkeypatch.setattr(subject, "load_regression_fixture", lambda _path: tiny)
    bindings = tmp_path / "local-bindings.json"
    _write_json(
        bindings,
        {
            "schema_version": LOCAL_BINDINGS_SCHEMA_VERSION,
            "dataset_revision": dataset_revision(tiny),
            "inputs": [
                {"asset_id": asset_id, "path": str(path.resolve())}
                for asset_id, path in paths.items()
            ],
        },
    )
    data_root = tmp_path / "disposable-product"
    models_root = tmp_path / "models"
    models_root.mkdir()
    driver = _FixtureDriver(
        {asset.asset_id: asset.duration_seconds for asset in tiny.assets}
    )

    receipt = provision_regression_fixture(
        dataset_path=PRODUCT_FIXTURE,
        bindings_path=bindings,
        data_root=data_root,
        models_root=models_root,
        timeout_seconds=45.0,
        driver=driver,
    )

    assert receipt["status"] == "complete"
    assert receipt["dataset_revision"] == dataset_revision(tiny)
    assert receipt["promotion_eligible"] is False
    assert receipt["evidence_use"] == "regression_only"
    assert len(driver.received) == 10
    for staged in driver.received:
        assert staged.path.parent == data_root / "media"
        assert staged.path.read_bytes() == payload_by_asset[staged.asset_id]
    benchmark_bindings = json.loads(
        (data_root / BENCHMARK_BINDINGS_FILENAME).read_text(encoding="utf-8")
    )
    assert benchmark_bindings == receipt["benchmark_bindings"]
    persisted = json.loads(
        (data_root / PROVISION_RECEIPT_FILENAME).read_text(encoding="utf-8")
    )
    assert persisted == receipt
    rendered = json.dumps(receipt, sort_keys=True)
    assert str(tmp_path) not in rendered
    assert "private-inputs" not in rendered


def test_provision_rejects_nonempty_symlink_and_identity_mismatch_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_dataset(PRODUCT_FIXTURE)
    source_root = tmp_path / "inputs"
    source_root.mkdir()
    paths: dict[str, Path] = {}
    payloads: dict[str, bytes] = {}
    for index, asset in enumerate(dataset.assets):
        payload = f"safe-{index}".encode()
        path = source_root / f"{asset.asset_id}.mp4"
        path.write_bytes(payload)
        paths[asset.asset_id] = path
        payloads[asset.asset_id] = payload
    from videoscope.benchmark import regression_fixture as subject

    tiny = replace(
        dataset,
        assets=tuple(
            replace(
                asset,
                sha256=sha256(payloads[asset.asset_id]).hexdigest(),
                byte_size=len(payloads[asset.asset_id]),
            )
            for asset in dataset.assets
        ),
    )
    monkeypatch.setattr(subject, "load_regression_fixture", lambda _path: tiny)
    bindings = tmp_path / "bindings.json"
    binding_payload = {
        "schema_version": LOCAL_BINDINGS_SCHEMA_VERSION,
        "dataset_revision": dataset_revision(tiny),
        "inputs": [
            {"asset_id": key, "path": str(value.resolve())}
            for key, value in paths.items()
        ],
    }
    _write_json(bindings, binding_payload)
    models = tmp_path / "models"
    models.mkdir()
    driver = _FixtureDriver(
        {asset.asset_id: asset.duration_seconds for asset in tiny.assets}
    )

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_text("do not touch", encoding="utf-8")
    with pytest.raises(RegressionFixtureError, match="data_root_not_empty"):
        provision_regression_fixture(
            dataset_path=PRODUCT_FIXTURE,
            bindings_path=bindings,
            data_root=nonempty,
            models_root=models,
            driver=driver,
        )
    assert (nonempty / "keep.txt").read_text(encoding="utf-8") == "do not touch"

    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises(RegressionFixtureError, match="data_root_unsafe"):
        provision_regression_fixture(
            dataset_path=PRODUCT_FIXTURE,
            bindings_path=bindings,
            data_root=linked,
            models_root=models,
            driver=driver,
        )

    changed = dict(binding_payload)
    changed_inputs = [dict(item) for item in binding_payload["inputs"]]
    changed_inputs[0]["path"] = str((tmp_path / "wrong.mp4").resolve())
    (tmp_path / "wrong.mp4").write_bytes(b"wrong")
    changed["inputs"] = changed_inputs
    _write_json(bindings, changed)
    with pytest.raises(RegressionFixtureError, match="prepared_input_identity_mismatch"):
        provision_regression_fixture(
            dataset_path=PRODUCT_FIXTURE,
            bindings_path=bindings,
            data_root=tmp_path / "new-root",
            models_root=models,
            driver=driver,
        )
    assert not (tmp_path / "new-root").exists()


def test_disposable_and_model_roots_reject_checkout_home_and_overlap(
    tmp_path: Path,
) -> None:
    from videoscope.benchmark import regression_fixture as subject

    external_data = tmp_path / "data"
    external_models = tmp_path / "models"
    external_models.mkdir()

    with pytest.raises(RegressionFixtureError, match="data_root_unsafe"):
        subject._validate_unused_data_root(PROJECT_ROOT / ".phase0-product")
    with pytest.raises(RegressionFixtureError, match="data_root_unsafe"):
        subject._validate_unused_data_root(Path.home() / ".phase0-product")
    with pytest.raises(RegressionFixtureError, match="models_root_unsafe"):
        subject._validate_models_root(PROJECT_ROOT, external_data)
    with pytest.raises(RegressionFixtureError, match="models_root_unsafe"):
        subject._validate_models_root(external_models, external_models / "child")
    with pytest.raises(RegressionFixtureError, match="models_root_unsafe"):
        subject._validate_models_root(external_models, tmp_path)


def test_stage_closes_source_descriptor_when_destination_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from videoscope.benchmark import regression_fixture as subject

    payload = b"private-fixture"
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(payload)
    media_root = tmp_path / "media"
    media_root.mkdir()
    asset = replace(
        load_dataset(PRODUCT_FIXTURE).assets[0],
        sha256=sha256(payload).hexdigest(),
        byte_size=len(payload),
    )
    binding = subject.LocalPreparedInput(asset_id=asset.asset_id, path=source_path)
    real_open = subject.os.open
    real_close = subject.os.close
    opened: list[int] = []
    closed: list[int] = []

    def failing_second_open(path, flags, mode=0o777):  # type: ignore[no-untyped-def]
        if opened:
            raise PermissionError("synthetic destination failure")
        descriptor = real_open(path, flags, mode)
        opened.append(descriptor)
        return descriptor

    def recording_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(subject.os, "open", failing_second_open)
    monkeypatch.setattr(subject.os, "close", recording_close)

    with pytest.raises(RegressionFixtureError, match="prepared_input_stage_failed"):
        subject._stage_asset(binding, asset, media_root)

    assert closed == opened


def test_stage_closes_source_when_destination_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from videoscope.benchmark import regression_fixture as subject

    payload = b"private-fixture"
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(payload)
    media_root = tmp_path / "media"
    media_root.mkdir()
    asset = replace(
        load_dataset(PRODUCT_FIXTURE).assets[0],
        sha256=sha256(payload).hexdigest(),
        byte_size=len(payload),
    )
    binding = subject.LocalPreparedInput(asset_id=asset.asset_id, path=source_path)
    real_open = subject.os.open
    real_close = subject.os.close
    opened: list[int] = []
    closed: list[int] = []

    def recording_open(path, flags, mode=0o777):  # type: ignore[no-untyped-def]
        descriptor = real_open(path, flags, mode)
        opened.append(descriptor)
        return descriptor

    def failing_target_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)
        if len(opened) == 2 and descriptor == opened[1]:
            raise OSError("synthetic target close failure")

    monkeypatch.setattr(subject.os, "open", recording_open)
    monkeypatch.setattr(subject.os, "close", failing_target_close)

    with pytest.raises(OSError, match="synthetic target close failure"):
        subject._stage_asset(binding, asset, media_root)

    assert set(closed) == set(opened)


def test_regression_settings_ignore_ambient_remote_provider_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from videoscope.benchmark import regression_fixture as subject

    for name in (
        "VISION_WORKER_ENDPOINT",
        "VISION_WORKER_API_KEY",
        "WHISPER_WORKER_ENDPOINT",
        "WHISPER_WORKER_API_KEY",
        "LIGHTHOUSE_ENDPOINT",
        "LIGHTHOUSE_API_KEY",
        "QWEN_VIDEO_ENDPOINT",
        "QWEN_VIDEO_API_KEY",
        "INTERNVIDEO_ENDPOINT",
        "INTERNVIDEO_API_KEY",
        "ROBOFLOW_API_KEY",
        "ROBOFLOW_MODEL_ID",
    ):
        monkeypatch.setenv(name, "ambient-must-not-be-read")
    models = tmp_path / "models"
    ocr_models = tmp_path / "ocr-models"
    models.mkdir()
    ocr_models.mkdir()

    settings = subject._RegressionProvisionSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        immutable_models_dir=models,
        ocr_model_root=ocr_models,
        ffmpeg_binary=Path("/usr/bin/false"),
        ffprobe_binary=Path("/usr/bin/false"),
    )

    assert settings.vision_worker_endpoint is None
    assert settings.whisper_worker_endpoint is None
    assert settings.lighthouse_endpoint is None
    assert settings.qwen_video_endpoint is None
    assert settings.internvideo_endpoint is None
    assert settings.roboflow_api_key is None


def test_product_benchmark_attests_the_exact_ffmpeg_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from videoscope.benchmark import environment as subject
    from videoscope.media.ffmpeg import FFmpeg

    ffmpeg_binary = Path("/pinned/toolchain/ffmpeg")
    ffprobe_binary = Path("/pinned/toolchain/ffprobe")
    calls: list[dict[str, object]] = []

    class Toolchain:
        def verify_current(self) -> str:
            return "sha256:" + "a" * 64

        def create_ffmpeg(self) -> FFmpeg:
            return FFmpeg.from_attested_paths(
                Path("/usr/bin/false"),
                Path("/usr/bin/false"),
            )

    def attest(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return Toolchain()

    monkeypatch.setattr(subject, "attest_indexing_toolchain", attest)

    runtime = subject._attested_benchmark_ffmpeg(
        ffmpeg_binary=ffmpeg_binary,
        ffprobe_binary=ffprobe_binary,
    )

    assert runtime is not None
    assert calls == [
        {
            "ffmpeg_binary": ffmpeg_binary,
            "ffprobe_binary": ffprobe_binary,
        }
    ]


def test_batch_prevalidation_attests_the_exact_ffmpeg_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from videoscope.benchmark import regression_fixture as subject
    from videoscope.benchmark.managed_workers import WorkerLaunchConfiguration
    from videoscope import indexing_attestation

    ffmpeg_binary = tmp_path / "ffmpeg"
    ffprobe_binary = tmp_path / "ffprobe"
    calls: list[dict[str, object]] = []

    class Toolchain:
        def verify_current(self) -> str:
            return "sha256:" + "a" * 64

    def attest(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return Toolchain()

    monkeypatch.setattr(indexing_attestation, "attest_indexing_toolchain", attest)
    configuration = WorkerLaunchConfiguration(
        schema_version=1,
        executables=(),
        hf_home=tmp_path,
        ocr_model_root=tmp_path,
        ffmpeg_binary=ffmpeg_binary,
        ffprobe_binary=ffprobe_binary,
    )

    identity = subject._prevalidate_indexing_toolchain(configuration)

    assert identity == "sha256:" + "a" * 64
    assert calls == [
        {
            "ffmpeg_binary": ffmpeg_binary,
            "ffprobe_binary": ffprobe_binary,
        }
    ]


def test_ml_attestation_binds_owner_and_roles_to_manifest_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from videoscope.benchmark import regression_fixture as subject
    from videoscope.benchmark.managed_workers import WorkerLaunchConfiguration
    from videoscope import ml_environment_attestation as ml_attestation

    manifest = ml_attestation.load_ml_environment_manifest(
        PROJECT_ROOT / "workers" / "ml-environment.lock.json",
        expected_sha256=ml_attestation.DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256,
    )
    directories = {
        item.environment_id: item.directory for item in manifest.environments
    }

    def python_for(environment_id: str) -> Path:
        return PROJECT_ROOT / directories[environment_id] / "bin" / "python"

    configuration = WorkerLaunchConfiguration(
        schema_version=1,
        executables=tuple(
            (role, python_for(role))
            for role in ("vision", "whisper", "lighthouse", "qwen", "ocr")
        ),
        hf_home=tmp_path / "hf",
        ocr_model_root=tmp_path / "ocr",
        ffmpeg_binary=tmp_path / "tools" / "ffmpeg",
        ffprobe_binary=tmp_path / "tools" / "ffprobe",
    )
    monkeypatch.setattr(subject.sys, "executable", str(python_for("base")))
    monkeypatch.setattr(
        ml_attestation,
        "attest_ml_environment",
        lambda *_args, **_kwargs: {
            "schema_version": ml_attestation.ML_ENVIRONMENT_REPORT_SCHEMA_VERSION,
            "status": "complete",
            "attestation_id": manifest.attestation_id,
            "manifest_identity": "sha256:" + manifest.raw_sha256,
            "failures": [],
        },
    )

    binding = subject._prevalidate_ml_environment(configuration)

    assert binding == (
        manifest.attestation_id,
        "sha256:" + manifest.raw_sha256,
    )

    monkeypatch.setattr(
        ml_attestation,
        "attest_ml_environment",
        lambda *_args, **_kwargs: {
            "schema_version": ml_attestation.ML_ENVIRONMENT_REPORT_SCHEMA_VERSION,
            "status": "partial",
            "attestation_id": manifest.attestation_id,
            "manifest_identity": "sha256:" + manifest.raw_sha256,
            "failures": [],
        },
    )
    with pytest.raises(
        RegressionFixtureError,
        match="ml_environment_attestation_incomplete",
    ):
        subject._prevalidate_ml_environment(configuration)


def test_ml_attestation_rejects_a_compatible_substitute_worker_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from videoscope.benchmark import regression_fixture as subject
    from videoscope.benchmark.managed_workers import WorkerLaunchConfiguration
    from videoscope import ml_environment_attestation as ml_attestation

    manifest = ml_attestation.load_ml_environment_manifest(
        PROJECT_ROOT / "workers" / "ml-environment.lock.json",
        expected_sha256=ml_attestation.DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256,
    )
    directories = {
        item.environment_id: item.directory for item in manifest.environments
    }

    def python_for(environment_id: str) -> Path:
        return PROJECT_ROOT / directories[environment_id] / "bin" / "python"

    substituted = tmp_path / ".venv-compatible" / "bin" / "python"
    configuration = WorkerLaunchConfiguration(
        schema_version=1,
        executables=tuple(
            (
                role,
                substituted if role == "vision" else python_for(role),
            )
            for role in ("vision", "whisper", "lighthouse", "qwen", "ocr")
        ),
        hf_home=tmp_path / "hf",
        ocr_model_root=tmp_path / "ocr",
        ffmpeg_binary=tmp_path / "tools" / "ffmpeg",
        ffprobe_binary=tmp_path / "tools" / "ffprobe",
    )
    monkeypatch.setattr(subject.sys, "executable", str(python_for("base")))
    monkeypatch.setattr(
        ml_attestation,
        "attest_ml_environment",
        lambda *_args, **_kwargs: pytest.fail("must reject before probing"),
    )

    with pytest.raises(RegressionFixtureError, match="ml_environment_binding_invalid"):
        subject._prevalidate_ml_environment(configuration)

    exact_configuration = replace(
        configuration,
        executables=tuple(
            (role, python_for(role))
            for role in ("vision", "whisper", "lighthouse", "qwen", "ocr")
        ),
    )
    monkeypatch.setattr(subject.sys, "executable", str(substituted))
    with pytest.raises(RegressionFixtureError, match="ml_environment_binding_invalid"):
        subject._prevalidate_ml_environment(exact_configuration)


def test_batch_prevalidation_constructs_inputs_with_toolchain_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from videoscope.benchmark import regression_fixture as subject
    from videoscope.benchmark.managed_workers import WorkerLaunchConfiguration
    from videoscope.benchmark.metric_policy import load_frozen_metric_policy

    dataset = load_dataset(PRODUCT_FIXTURE)
    policy_path = (
        PROJECT_ROOT
        / "docs"
        / "benchmarks"
        / "policies"
        / "phase0-regression-v1.json"
    )
    configuration = WorkerLaunchConfiguration(
        schema_version=1,
        executables=(),
        hf_home=tmp_path / "hf",
        ocr_model_root=tmp_path / "ocr",
        ffmpeg_binary=tmp_path / "tools" / "ffmpeg",
        ffprobe_binary=tmp_path / "tools" / "ffprobe",
    )
    roots = {
        "data": tmp_path / "data",
        "models": tmp_path / "models",
        "scratch": tmp_path / "scratch",
        "registry": tmp_path / "registry",
    }
    monkeypatch.setattr(subject, "load_regression_fixture", lambda _path: dataset)
    monkeypatch.setattr(
        subject,
        "load_frozen_metric_policy",
        lambda _path: load_frozen_metric_policy(policy_path),
    )
    monkeypatch.setattr(
        subject,
        "validate_frozen_metric_policy_product_dataset",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        subject,
        "load_worker_launch_configuration",
        lambda _path: configuration,
    )
    monkeypatch.setattr(
        subject,
        "_prevalidate_ml_environment",
        lambda _configuration: (
            "videoscope-phase0-ml-environment-v1",
            "sha256:" + "b" * 64,
        ),
    )
    monkeypatch.setattr(
        subject,
        "_prevalidate_indexing_toolchain",
        lambda _configuration: "sha256:" + "a" * 64,
    )
    monkeypatch.setattr(subject, "_validate_unused_data_root", lambda path: path)
    monkeypatch.setattr(
        subject,
        "_validate_models_root",
        lambda path, _data: path,
    )
    monkeypatch.setattr(
        subject,
        "_validate_empty_external_directory",
        lambda path, **_kwargs: path,
    )
    monkeypatch.setattr(
        subject,
        "load_local_input_bindings",
        lambda *_args: subject.LocalInputBindings(
            schema_version=1,
            dataset_revision=FIXTURE_DATASET_REVISION,
            inputs=(),
        ),
    )
    monkeypatch.setattr(subject, "_attest_all_sources", lambda *_args: None)

    inputs = subject._prevalidate_batch_inputs(
        dataset_path=PRODUCT_FIXTURE,
        bindings_path=tmp_path / "bindings.json",
        policy_path=policy_path,
        data_root=roots["data"],
        models_root=roots["models"],
        scratch_parent=roots["scratch"],
        registry_root=roots["registry"],
        worker_launch_path=tmp_path / "workers.json",
        code_identity=lambda: "d" * 40,
    )

    assert inputs.indexing_toolchain_identity == "sha256:" + "a" * 64
    assert inputs.ml_environment_attestation_id == (
        "videoscope-phase0-ml-environment-v1"
    )
    assert inputs.ml_environment_manifest_identity == "sha256:" + "b" * 64
    assert inputs.worker_configuration is configuration


def test_owner_batch_retires_ingest_workers_runs_five_profiles_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from videoscope.benchmark import regression_fixture as subject
    from videoscope.benchmark.managed_workers import WorkerLaunchConfiguration
    from videoscope.benchmark.metric_policy import load_frozen_metric_policy

    dataset = load_dataset(PRODUCT_FIXTURE)
    policy_path = (
        PROJECT_ROOT
        / "docs"
        / "benchmarks"
        / "policies"
        / "phase0-regression-v1.json"
    )
    policy = load_frozen_metric_policy(policy_path)
    data_root = tmp_path / "data"
    models_root = tmp_path / "models"
    scratch_root = tmp_path / "scratch"
    registry_root = tmp_path / "registry"
    ocr_root = tmp_path / "ocr"
    for path in (models_root, scratch_root, ocr_root):
        path.mkdir(mode=0o700)
        path.chmod(0o700)
    ocr_script = tmp_path / "ocr-worker.py"
    ocr_script.write_text("# fixture\n", encoding="utf-8")
    configuration = WorkerLaunchConfiguration(
        schema_version=1,
        executables=(),
        hf_home=tmp_path / "hf",
        ocr_model_root=ocr_root,
        ffmpeg_binary=tmp_path / "tools" / "ffmpeg",
        ffprobe_binary=tmp_path / "tools" / "ffprobe",
    )
    code_sha = "d" * 40
    toolchain_identity = "sha256:" + "a" * 64
    ml_attestation_id = "videoscope-phase0-ml-environment-v1"
    ml_manifest_identity = "sha256:" + "b" * 64
    inputs = subject.RegressionBatchInputs(
        dataset=dataset,
        dataset_path=PRODUCT_FIXTURE,
        bindings_path=tmp_path / "private-bindings.json",
        policy=policy,
        data_root=data_root,
        models_root=models_root,
        scratch_parent=scratch_root,
        registry_root=registry_root,
        worker_configuration=configuration,
        code_sha=code_sha,
        indexing_toolchain_identity=toolchain_identity,
        ml_environment_attestation_id=ml_attestation_id,
        ml_environment_manifest_identity=ml_manifest_identity,
    )
    events: list[str] = []
    all_workers = tuple(
        ManagedProcessBinding(
            pid=5000 + index,
            start_token=f"start-{role}",
            executable_identity=f"sha256:{index:064x}",
            role=role,
        )
        for index, role in enumerate(subject.MANAGED_WORKER_ROLES, start=1)
    )
    worker_overrides = {
        "ocr_worker_python": Path("/bin/sh"),
        "ocr_worker_script": ocr_script,
        "ocr_worker_environment": {
            "HF_HOME": str(tmp_path / "hf"),
            "HOME": str(data_root / "tmp" / "worker-ocr"),
            "PATH": str(tmp_path / "tools"),
            "TMPDIR": str(data_root / "tmp" / "worker-ocr"),
            "XDG_CACHE_HOME": str(data_root / "tmp" / "worker-ocr" / "cache"),
        },
        "vision_worker_endpoint": "http://127.0.0.1:31001",
        "vision_worker_api_key": "v" * 32,
        "whisper_worker_endpoint": "http://127.0.0.1:31002",
        "whisper_worker_api_key": "w" * 32,
        "lighthouse_endpoint": "http://127.0.0.1:31003",
        "lighthouse_api_key": "l" * 32,
        "qwen_video_endpoint": "http://127.0.0.1:31004",
        "qwen_video_api_key": "q" * 32,
    }

    class Cluster:
        def __init__(self) -> None:
            self.managed_workers = all_workers
            self.ingest_overrides = dict(worker_overrides)
            self.benchmark_overrides = dict(worker_overrides)
            self.ocr_model_root = ocr_root
            self.retired_worker_roles: tuple[str, ...] = ()

        def retire_ingest_workers(self) -> None:
            events.append("retire")
            self.managed_workers = tuple(
                item
                for item in self.managed_workers
                if item.role not in {"vision_index", "whisper"}
            )
            self.retired_worker_roles = ("vision_index", "whisper")

        def close(self) -> None:
            events.append("close")

    cluster = Cluster()

    def prepare(**_kwargs):  # type: ignore[no-untyped-def]
        events.append("prepare")
        data_root.mkdir(mode=0o700)
        return subject.PreparedRegressionFixture(
            dataset=dataset,
            data_root=data_root,
            models_root=models_root,
            assets=(),
        )

    def complete(_prepared, **_kwargs):  # type: ignore[no-untyped-def]
        events.append("ingest")
        (data_root / subject.BENCHMARK_BINDINGS_FILENAME).write_text(
            "{}",
            encoding="utf-8",
        )
        return {"status": "complete"}

    def create_driver(*_args, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["ffmpeg_binary"] == configuration.ffmpeg_binary
        assert kwargs["ffprobe_binary"] == configuration.ffprobe_binary
        return object()

    def run_profiles(**kwargs):  # type: ignore[no-untyped-def]
        events.append("profiles")
        assert tuple(item.role for item in kwargs["managed_workers"]) == (
            "vision",
            "lighthouse",
            "qwen",
        )
        assert kwargs["settings"].ffmpeg_binary == configuration.ffmpeg_binary
        return tuple(
            {
                "run_id": f"phase0-test-{profile_id}",
                "profile_id": profile_id,
                "run_status": "complete",
                "measurement_status": "complete",
                "measurement_evidence_status": "complete",
                "manifest_sha256": "e" * 64,
                "rss_sample_count": 2,
                "model_identities": [],
            }
            for profile_id in subject.REQUIRED_PROFILE_IDS
        )

    monkeypatch.setattr(subject, "_prevalidate_batch_inputs", lambda **_kwargs: inputs)
    monkeypatch.setattr(subject, "prepare_regression_fixture", prepare)
    monkeypatch.setattr(subject, "complete_regression_fixture", complete)
    monkeypatch.setattr(
        subject,
        "_prevalidate_indexing_toolchain",
        lambda _configuration: toolchain_identity,
    )
    monkeypatch.setattr(
        subject,
        "_prevalidate_ml_environment",
        lambda _configuration: (ml_attestation_id, ml_manifest_identity),
    )

    receipt = subject.execute_phase0_regression_batch(
        dataset_path=PRODUCT_FIXTURE,
        bindings_path=inputs.bindings_path,
        policy_path=policy_path,
        data_root=data_root,
        models_root=models_root,
        scratch_parent=scratch_root,
        registry_root=registry_root,
        worker_launch_path=tmp_path / "worker-launch.json",
        run_id_prefix="phase0-test",
        _code_identity=lambda: code_sha,
        _cluster_factory=lambda **_kwargs: cluster,
        _driver_factory=create_driver,
        _profile_runner=run_profiles,
    )

    assert events == ["prepare", "ingest", "retire", "profiles", "close"]
    assert set(receipt) == {
        "schema_version",
        "status",
        "code_sha",
        "dataset_revision",
        "policy_revision",
        "ml_environment_manifest_identity",
        "ml_environment_attestation_id",
        "environment_bindings",
        "profile_runs",
        "worker_lifecycle",
    }
    assert receipt["ml_environment_manifest_identity"] == ml_manifest_identity
    assert receipt["ml_environment_attestation_id"] == ml_attestation_id
    assert receipt["environment_bindings"] == subject.ENVIRONMENT_BINDINGS
    assert receipt["worker_lifecycle"] == {
        "cleanup_status": "complete",
        "retirement_status": "complete",
    }
    assert len(receipt["profile_runs"]) == 5
    persisted = json.loads(
        (data_root / BASELINE_BATCH_RECEIPT_FILENAME).read_text(encoding="utf-8")
    )
    assert persisted == receipt
    assert str(tmp_path) not in json.dumps(receipt, sort_keys=True)


def _production_driver_id_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    assets: tuple[object, ...],
):  # type: ignore[no-untyped-def]
    from videoscope.benchmark import regression_fixture as subject
    from videoscope.jobs import JobState
    from videoscope.processing import coordinator as coordinator_module
    from videoscope.providers.lighthouse_worker import _validate_video_id
    from videoscope import repository as repository_module
    from videoscope import runtime as runtime_module
    from videoscope.search.visual_index import SiglipVisualIndex

    events: list[str] = []
    submitted: list[dict[str, object]] = []
    by_digest = {item.sha256: item for item in assets}

    class Repository:
        def __init__(self, _path: Path) -> None:
            events.append("repository")

        def initialize(self) -> None:
            pass

        def find_assets_by_sha256_bounded(self, digest, *, limit, video_id):  # type: ignore[no-untyped-def]
            assert limit == 1
            assert video_id == digest
            return [by_digest[digest]]

    class Runtime:
        queue = object()
        video_index_plan_factory = staticmethod(lambda: None)

        def start(self) -> None:
            events.append("runtime_start")

        def close(self) -> bool:
            events.append("runtime_close")
            return True

    class Coordinator:
        def __init__(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            pass

        def create_ingest(self, **kwargs):  # type: ignore[no-untyped-def]
            video_id = kwargs["video_id"]
            # Exercise the unchanged serving contracts without model execution.
            if not SiglipVisualIndex._valid_video_id(video_id):
                raise ValueError("invalid visual index video id")
            assert _validate_video_id(video_id) == video_id
            submitted.append(kwargs)
            return None, SimpleNamespace(job_id=kwargs["job_id"])

    settings = SimpleNamespace(
        ensure_directories=lambda: events.append("create_directories"),
        database_path=tmp_path / "library.sqlite3",
        ocr_worker_environment={},
        max_upload_bytes=1024**3,
    )
    driver = object.__new__(subject.ProductionRegressionDriver)
    driver._models_root = tmp_path / "models"
    driver._ocr_model_root = tmp_path / "ocr-models"
    driver._worker_overrides = {}
    driver._ffmpeg_binary = Path("/usr/bin/false")
    driver._ffprobe_binary = Path("/usr/bin/false")
    driver._toolchain = SimpleNamespace(verify_current=lambda: None)
    monkeypatch.setattr(subject, "_explicit_product_settings", lambda **_kwargs: settings)
    monkeypatch.setattr(repository_module, "Repository", Repository)
    monkeypatch.setattr(runtime_module, "build_runtime", lambda *_args, **_kwargs: Runtime())
    monkeypatch.setattr(coordinator_module, "VideoIndexCoordinator", Coordinator)
    monkeypatch.setattr(
        subject,
        "_wait_for_index_job",
        lambda *_args: SimpleNamespace(state=JobState.COMPLETE, plan_hash="a" * 64),
    )
    return driver, submitted, events


def test_production_fixture_ids_fit_serving_contracts_and_keep_full_source_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from videoscope.benchmark import regression_fixture as subject

    assets = tuple(
        subject.StagedRegressionAsset(
            asset_id=asset.asset_id,
            path=tmp_path / f"{asset.asset_id}.mp4",
            sha256=asset.sha256,
            byte_size=asset.byte_size,
            duration_seconds=asset.duration_seconds,
        )
        for asset in load_dataset(PRODUCT_FIXTURE).assets
    )
    driver, submitted, events = _production_driver_id_harness(tmp_path, monkeypatch, assets)

    first = driver.provision(data_root=tmp_path, assets=assets, timeout_seconds=10.0)
    second = driver.provision(data_root=tmp_path, assets=assets, timeout_seconds=10.0)

    assert first == second
    assert len({item.video_id for item in first}) == len(assets) == 10
    for item, result in zip(assets, first):
        assert result.video_id == result.sha256 == item.sha256
        assert result.job_id == f"regjob_{item.sha256}"
    assert [item["source_sha256"] for item in submitted] == [
        item.sha256 for item in (*assets, *assets)
    ]
    assert events.count("runtime_close") == 2


@pytest.mark.parametrize("digest", ["a" * 63, "a" * 65, "A" * 64, "a" * 63 + "/"])
def test_production_fixture_rejects_invalid_identity_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    digest: str,
) -> None:
    from videoscope.benchmark import regression_fixture as subject

    assets = (subject.StagedRegressionAsset("fixture", tmp_path / "input.mp4", digest, 1, 1.0),)
    driver, submitted, events = _production_driver_id_harness(tmp_path, monkeypatch, assets)

    with pytest.raises(RegressionFixtureError, match="fixture_video_id_invalid"):
        driver.provision(data_root=tmp_path, assets=assets, timeout_seconds=10.0)

    assert submitted == []
    assert events == []


@pytest.mark.parametrize("entry_point", ["prepare", "batch"])
def test_fixture_rejects_invalid_identity_before_preflight_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
) -> None:
    from videoscope.benchmark import regression_fixture as subject

    dataset = SimpleNamespace(assets=(SimpleNamespace(sha256="a" * 65),))
    monkeypatch.setattr(subject, "load_regression_fixture", lambda _path: dataset)

    def unexpected(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        pytest.fail("invalid fixture identity reached later preflight work")

    monkeypatch.setattr(subject, "load_local_input_bindings", unexpected)
    monkeypatch.setattr(subject, "load_frozen_metric_policy", unexpected)
    arguments = {
        "dataset_path": PRODUCT_FIXTURE,
        "bindings_path": tmp_path / "bindings.json",
        "data_root": tmp_path / "data",
        "models_root": tmp_path / "models",
    }
    with pytest.raises(RegressionFixtureError, match="fixture_video_id_invalid"):
        if entry_point == "prepare":
            subject.prepare_regression_fixture(**arguments)
        else:
            subject._prevalidate_batch_inputs(
                **arguments,
                policy_path=tmp_path / "policy.json",
                scratch_parent=tmp_path / "scratch",
                registry_root=tmp_path / "registry",
                worker_launch_path=tmp_path / "workers.json",
                code_identity=lambda: "a" * 40,
            )
    assert tuple(tmp_path.iterdir()) == ()


def test_production_driver_closes_partially_started_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from videoscope.benchmark import regression_fixture as subject
    from videoscope import repository as repository_module
    from videoscope import runtime as runtime_module

    models = tmp_path / "models"
    ocr_models = tmp_path / "ocr-models"
    data_root = tmp_path / "data"
    models.mkdir()
    ocr_models.mkdir()
    data_root.mkdir()
    closed: list[bool] = []

    class Toolchain:
        def verify_current(self) -> str:
            return "sha256:" + "a" * 64

    class Repository:
        def __init__(self, _path: Path) -> None:
            pass

        def initialize(self) -> None:
            pass

    class Runtime:
        def start(self) -> None:
            raise RuntimeError("synthetic partial start")

        def close(self) -> bool:
            closed.append(True)
            return True

    driver = object.__new__(subject.ProductionRegressionDriver)
    driver._models_root = models
    driver._ocr_model_root = ocr_models
    driver._ffmpeg_binary = Path("/usr/bin/false")
    driver._ffprobe_binary = Path("/usr/bin/false")
    driver._toolchain = Toolchain()
    driver._worker_overrides = {
        "ocr_worker_python": Path("/bin/sh"),
        "ocr_worker_script": Path("/bin/sh"),
        "ocr_worker_environment": {
            "HF_HOME": str(tmp_path / "hf"),
            "HOME": str(data_root / "tmp" / "worker-ocr"),
            "PATH": "/usr/bin",
            "TMPDIR": str(data_root / "tmp" / "worker-ocr"),
            "XDG_CACHE_HOME": str(data_root / "tmp" / "worker-ocr" / "cache"),
        },
        "vision_worker_endpoint": "http://127.0.0.1:31001",
        "vision_worker_api_key": "v" * 32,
        "whisper_worker_endpoint": "http://127.0.0.1:31002",
        "whisper_worker_api_key": "w" * 32,
        "lighthouse_endpoint": "http://127.0.0.1:31003",
        "lighthouse_api_key": "l" * 32,
        "qwen_video_endpoint": "http://127.0.0.1:31004",
        "qwen_video_api_key": "q" * 32,
    }
    monkeypatch.setattr(repository_module, "Repository", Repository)
    monkeypatch.setattr(runtime_module, "build_runtime", lambda *_args, **_kwargs: Runtime())

    with pytest.raises(RuntimeError, match="synthetic partial start"):
        driver.provision(data_root=data_root, assets=(), timeout_seconds=10.0)

    assert closed == [True]


def test_required_profile_runner_uses_one_sha_revision_and_owned_workers(
    tmp_path: Path,
) -> None:
    from videoscope.benchmark import regression_fixture as subject

    dataset = load_dataset(PRODUCT_FIXTURE)
    expected_revision = dataset_revision(dataset)
    models = tmp_path / "models"
    ocr_models = tmp_path / "ocr-models"
    models.mkdir()
    ocr_models.mkdir()
    settings = subject._RegressionProvisionSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        immutable_models_dir=models,
        ocr_model_root=ocr_models,
        ffmpeg_binary=Path("/usr/bin/false"),
        ffprobe_binary=Path("/usr/bin/false"),
    )
    roles = ("vision", "lighthouse", "qwen")
    workers = tuple(
        ManagedProcessBinding(
            pid=1000 + index,
            start_token=f"start-{role}",
            executable_identity=f"sha256:{index:064x}",
            role=role,
        )
        for index, role in enumerate(roles, start=1)
    )
    calls: list[tuple[str, bool, object, object, object]] = []
    audited: list[str] = []
    code_sha = "d" * 40

    def executor(arguments, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(
            (
                arguments.profile,
                arguments.preflight,
                kwargs["managed_workers"],
                kwargs["explicit_settings"],
                kwargs["expected_code_sha"],
            )
        )
        if arguments.preflight:
            return {
                "status": "ready",
                "measurement": {"status": "ready"},
                "profile_id": arguments.profile,
                "execution_mode": "warm",
                "dataset_revision": expected_revision,
                "code_sha": code_sha,
            }
        return {
            "status": "published",
            "run_status": "complete",
            "profile_id": arguments.profile,
            "execution_mode": "warm",
            "dataset_revision": expected_revision,
        }

    def auditor(**kwargs):  # type: ignore[no-untyped-def]
        audited.append(kwargs["profile_id"])
        return {
            "run_id": kwargs["run_id"],
            "profile_id": kwargs["profile_id"],
            "run_status": "complete",
            "measurement_status": "complete",
            "measurement_evidence_status": "complete",
            "manifest_sha256": "e" * 64,
            "rss_sample_count": 2,
            "model_identities": [],
        }

    receipts = run_required_product_profiles(
        dataset_path=PRODUCT_FIXTURE,
        bindings_path=tmp_path / "benchmark-bindings.json",
        registry_root=tmp_path / "registry",
        data_root=tmp_path / "data",
        scratch_parent=tmp_path / "scratch",
        run_id_prefix="phase0-test",
        managed_workers=workers,
        settings=settings,
        expected_code_sha=code_sha,
        expected_dataset_revision=expected_revision,
        code_identity=lambda: code_sha,
        _executor=executor,
        _auditor=auditor,
    )

    assert tuple(item[0] for item in calls[::2]) == subject.REQUIRED_PROFILE_IDS
    assert tuple(item[0] for item in calls[1::2]) == subject.REQUIRED_PROFILE_IDS
    assert all(item[1] is (index % 2 == 0) for index, item in enumerate(calls))
    assert all(item[2:] == (workers, settings, code_sha) for item in calls)
    assert tuple(audited) == subject.REQUIRED_PROFILE_IDS
    assert tuple(item["profile_id"] for item in receipts) == subject.REQUIRED_PROFILE_IDS


def test_required_profile_runner_stops_before_run_when_preflight_is_not_ready(
    tmp_path: Path,
) -> None:
    from videoscope.benchmark import regression_fixture as subject

    models = tmp_path / "models"
    ocr_models = tmp_path / "ocr-models"
    models.mkdir()
    ocr_models.mkdir()
    settings = subject._RegressionProvisionSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        immutable_models_dir=models,
        ocr_model_root=ocr_models,
        ffmpeg_binary=Path("/usr/bin/false"),
        ffprobe_binary=Path("/usr/bin/false"),
    )
    workers = tuple(
        ManagedProcessBinding(
            pid=2000 + index,
            start_token=f"start-{role}",
            executable_identity=f"sha256:{index:064x}",
            role=role,
        )
        for index, role in enumerate(
            ("vision", "lighthouse", "qwen"),
            start=1,
        )
    )
    calls: list[bool] = []

    def executor(arguments, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(arguments.preflight)
        return {
            "status": "not_ready",
            "measurement": {"status": "ready"},
            "profile_id": arguments.profile,
            "execution_mode": "warm",
            "dataset_revision": FIXTURE_DATASET_REVISION,
            "code_sha": "d" * 40,
        }

    with pytest.raises(RegressionFixtureError, match="profile_preflight_not_ready"):
        run_required_product_profiles(
            dataset_path=PRODUCT_FIXTURE,
            bindings_path=tmp_path / "bindings.json",
            registry_root=tmp_path / "registry",
            data_root=tmp_path / "data",
            scratch_parent=tmp_path / "scratch",
            run_id_prefix="phase0-test",
            managed_workers=workers,
            settings=settings,
            expected_code_sha="d" * 40,
            expected_dataset_revision=FIXTURE_DATASET_REVISION,
            code_identity=lambda: "d" * 40,
            _executor=executor,
            _auditor=lambda **_kwargs: pytest.fail("must not audit"),
        )

    assert calls == [True]


def test_required_profile_runner_rejects_retired_worker_bindings(
    tmp_path: Path,
) -> None:
    from videoscope.benchmark import regression_fixture as subject

    models = tmp_path / "models"
    ocr_models = tmp_path / "ocr-models"
    models.mkdir()
    ocr_models.mkdir()
    settings = subject._RegressionProvisionSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        immutable_models_dir=models,
        ocr_model_root=ocr_models,
        ffmpeg_binary=Path("/usr/bin/false"),
        ffprobe_binary=Path("/usr/bin/false"),
    )
    workers = tuple(
        ManagedProcessBinding(
            pid=3000 + index,
            start_token=f"start-{role}",
            executable_identity=f"sha256:{index:064x}",
            role=role,
        )
        for index, role in enumerate(
            ("vision_index", "vision", "lighthouse", "qwen"),
            start=1,
        )
    )

    with pytest.raises(RegressionFixtureError, match="managed_worker_bindings_incomplete"):
        run_required_product_profiles(
            dataset_path=PRODUCT_FIXTURE,
            bindings_path=tmp_path / "bindings.json",
            registry_root=tmp_path / "registry",
            data_root=tmp_path / "data",
            scratch_parent=tmp_path / "scratch",
            run_id_prefix="phase0-test",
            managed_workers=workers,
            settings=settings,
            expected_code_sha="d" * 40,
            expected_dataset_revision=FIXTURE_DATASET_REVISION,
            code_identity=lambda: "d" * 40,
            _executor=lambda *_args, **_kwargs: pytest.fail("must not execute"),
        )


def test_public_batch_validates_worker_configuration_before_creating_root(
    tmp_path: Path,
) -> None:
    from videoscope.benchmark import regression_fixture as subject

    data_root = tmp_path / "disposable"
    with pytest.raises(FileNotFoundError):
        subject.execute_phase0_regression_batch(
            dataset_path=PRODUCT_FIXTURE,
            bindings_path=tmp_path / "private-bindings.json",
            policy_path=(
                PROJECT_ROOT
                / "docs"
                / "benchmarks"
                / "policies"
                / "phase0-regression-v1.json"
            ),
            data_root=data_root,
            models_root=tmp_path / "models",
            scratch_parent=tmp_path / "scratch",
            registry_root=tmp_path / "registry",
            worker_launch_path=tmp_path / "missing-worker-launch.json",
            run_id_prefix="phase0-test",
            _code_identity=lambda: "d" * 40,
        )

    assert not data_root.exists()
    assert not (tmp_path / "registry").exists()


def test_driverless_provision_validates_runtime_bindings_before_staging(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "disposable"

    with pytest.raises(
        RegressionFixtureError,
        match="managed_worker_configuration_required",
    ):
        provision_regression_fixture(
            dataset_path=PRODUCT_FIXTURE,
            bindings_path=tmp_path / "private-bindings.json",
            data_root=data_root,
            models_root=tmp_path / "models",
        )

    assert not data_root.exists()


@pytest.mark.parametrize(
    ("error", "exit_code", "reason_code"),
    (
        (
            BenchmarkExecutionError("private /absolute/input must not leak"),
            8,
            "benchmark_execution_failed",
        ),
        (MeasurementError("measurement_failed"), 9, "measurement_failed"),
        (RuntimeError("private /absolute/input must not leak"), 70, "internal_error"),
    ),
)
def test_batch_cli_failures_are_classified_without_traceback_or_private_paths(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    exit_code: int,
    reason_code: str,
) -> None:  # type: ignore[no-untyped-def]
    from videoscope.benchmark import regression_fixture as subject

    def fail(**_kwargs):  # type: ignore[no-untyped-def]
        raise error

    monkeypatch.setattr(subject, "execute_phase0_regression_batch", fail)
    private = tmp_path / "private-input.json"

    actual = subject.main(
        [
            "batch",
            "--dataset",
            str(private),
            "--bindings",
            str(private),
            "--policy",
            str(private),
            "--data-root",
            str(tmp_path / "data"),
            "--models-root",
            str(tmp_path / "models"),
            "--scratch-parent",
            str(tmp_path / "scratch"),
            "--registry",
            str(tmp_path / "registry"),
            "--worker-launch",
            str(private),
            "--run-id-prefix",
            "phase0-test",
        ]
    )

    captured = capsys.readouterr()
    assert actual == exit_code
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "reason_code": reason_code,
        "status": "failed",
    }
    assert str(tmp_path) not in captured.err
    assert "Traceback" not in captured.err
