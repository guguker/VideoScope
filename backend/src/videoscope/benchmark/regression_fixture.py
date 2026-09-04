from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
import argparse
import json
import math
import os
from pathlib import Path
import stat
import sys
from time import monotonic, sleep
from typing import Protocol, Sequence

from pydantic import Field

from videoscope.config import AppSettings
from videoscope.model_manifest import QWEN_VIDEO_MODEL, SIGLIP_224_MODEL, WHISPER_MODEL

from .schema import (
    BenchmarkAsset,
    BenchmarkDataError,
    BenchmarkDataset,
    _require_id,
)
from .managed_workers import (
    MANAGED_WORKER_ROLES,
    ManagedRegressionWorkerCluster,
    ManagedWorkerError,
    WorkerLaunchConfiguration,
    load_worker_launch_configuration,
    start_regression_worker_cluster,
)
from .measurements import ManagedProcessBinding, MeasurementError
from .metric_policy import (
    FrozenMetricPolicy,
    frozen_metric_policy_revision,
    load_frozen_metric_policy,
    validate_frozen_metric_policy_product_dataset,
)
from .runner import BenchmarkExecutionError, audit_run_manifest
from .serialization import (
    canonical_json_bytes,
    dataset_revision,
    expect_fields,
    expect_list,
    expect_object,
    parse_json_object,
)
from .storage import (
    MAX_DATASET_MANIFEST_BYTES,
    _read_bounded_file,
    _write_exclusive_bytes,
    BenchmarkRunRegistry,
    load_dataset,
)


FIXTURE_DATASET_ID = "video-verifier-product-regression"
FIXTURE_DATASET_VERSION = "1.0.0"
FIXTURE_DATASET_REVISION = (
    "8cda261148ce3f459e08c3762017cec86970ab7c85d36eaa77ef9f832a2ae6a1"
)
LOCAL_BINDINGS_SCHEMA_VERSION = 1
PROVISION_RECEIPT_SCHEMA_VERSION = 1
BASELINE_BATCH_RECEIPT_SCHEMA_VERSION = 1
BENCHMARK_BINDINGS_FILENAME = "benchmark-bindings.json"
PROVISION_RECEIPT_FILENAME = "provision-receipt.json"
BASELINE_BATCH_RECEIPT_FILENAME = "phase0-baseline-receipt.json"
REQUIRED_PROFILE_IDS = (
    "lexical_qdrant",
    "dense_siglip",
    "temporal_refinement",
    "lighthouse",
    "qwen_verification",
)
ENVIRONMENT_BINDINGS = {
    "owner": "base",
    "vision_index": "vision",
    "vision": "vision",
    "whisper": "whisper",
    "ocr": "ocr",
    "lighthouse": "lighthouse",
    "qwen": "qwen",
}

_MAX_LOCAL_BINDINGS_BYTES = 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_DURATION_TOLERANCE_SECONDS = 0.05
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


class RegressionFixtureError(RuntimeError):
    """A path-free, machine-readable fixture provisioning failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"regression fixture failed ({code})")


@dataclass(frozen=True, slots=True)
class LocalPreparedInput:
    asset_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class LocalInputBindings:
    schema_version: int
    dataset_revision: str
    inputs: tuple[LocalPreparedInput, ...]


@dataclass(frozen=True, slots=True)
class StagedRegressionAsset:
    asset_id: str
    path: Path
    sha256: str
    byte_size: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class ProvisionedAsset:
    asset_id: str
    video_id: str
    job_id: str
    plan_hash: str
    sha256: str
    byte_size: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class PreparedRegressionFixture:
    dataset: BenchmarkDataset
    data_root: Path
    models_root: Path
    assets: tuple[StagedRegressionAsset, ...]


@dataclass(frozen=True, slots=True)
class RegressionBatchInputs:
    dataset: BenchmarkDataset
    dataset_path: Path
    bindings_path: Path
    policy: FrozenMetricPolicy
    data_root: Path
    models_root: Path
    scratch_parent: Path
    registry_root: Path
    worker_configuration: WorkerLaunchConfiguration
    code_sha: str
    indexing_toolchain_identity: str
    ml_environment_attestation_id: str
    ml_environment_manifest_identity: str


class RegressionProvisionDriver(Protocol):
    def probe_duration(self, asset_id: str, path: Path) -> float: ...

    def provision(
        self,
        *,
        data_root: Path,
        assets: Sequence[StagedRegressionAsset],
        timeout_seconds: float,
    ) -> tuple[ProvisionedAsset, ...]: ...


def load_regression_fixture(path: Path) -> BenchmarkDataset:
    dataset = load_dataset(Path(path))
    if (
        dataset.dataset_id != FIXTURE_DATASET_ID
        or dataset.dataset_version != FIXTURE_DATASET_VERSION
        or dataset_revision(dataset) != FIXTURE_DATASET_REVISION
    ):
        raise RegressionFixtureError("fixture_revision_mismatch")
    return dataset


def load_local_input_bindings(
    path: Path,
    dataset: BenchmarkDataset,
) -> LocalInputBindings:
    payload = _read_bounded_file(
        Path(path),
        _MAX_LOCAL_BINDINGS_BYTES,
        "local prepared-input bindings",
    )
    value = parse_json_object(payload, "local prepared-input bindings")
    expect_fields(
        value,
        {"schema_version", "dataset_revision", "inputs"},
        "local prepared-input bindings",
    )
    schema_version = value["schema_version"]
    if type(schema_version) is not int or schema_version != LOCAL_BINDINGS_SCHEMA_VERSION:
        raise RegressionFixtureError("bindings_schema_unsupported")
    revision = value["dataset_revision"]
    if type(revision) is not str or revision != dataset_revision(dataset):
        raise RegressionFixtureError("bindings_revision_mismatch")

    inputs: list[LocalPreparedInput] = []
    for index, raw_input in enumerate(
        expect_list(value["inputs"], "local prepared-input bindings.inputs")
    ):
        item = expect_object(raw_input, f"local prepared-input bindings.inputs[{index}]")
        expect_fields(
            item,
            {"asset_id", "path"},
            f"local prepared-input bindings.inputs[{index}]",
        )
        asset_id = item["asset_id"]
        raw_path = item["path"]
        if (
            type(asset_id) is not str
            or not asset_id
            or len(asset_id) > 128
            or "\x00" in asset_id
        ):
            raise RegressionFixtureError("binding_asset_id_invalid")
        if (
            type(raw_path) is not str
            or not raw_path
            or len(raw_path) > 4096
            or "\x00" in raw_path
        ):
            raise RegressionFixtureError("binding_path_invalid")
        prepared_path = Path(raw_path)
        if not prepared_path.is_absolute():
            raise RegressionFixtureError("binding_path_invalid")
        inputs.append(LocalPreparedInput(asset_id=asset_id, path=prepared_path))

    expected_aliases = {asset.asset_id for asset in dataset.assets}
    actual_aliases = tuple(item.asset_id for item in inputs)
    if len(actual_aliases) != len(set(actual_aliases)) or set(actual_aliases) != expected_aliases:
        raise RegressionFixtureError("bindings_alias_mismatch")
    return LocalInputBindings(
        schema_version=LOCAL_BINDINGS_SCHEMA_VERSION,
        dataset_revision=revision,
        inputs=tuple(sorted(inputs, key=lambda item: item.asset_id)),
    )


def provision_regression_fixture(
    *,
    dataset_path: Path,
    bindings_path: Path,
    data_root: Path,
    models_root: Path,
    timeout_seconds: float = 1800.0,
    driver: RegressionProvisionDriver | None = None,
    worker_overrides: Mapping[str, object] | None = None,
    ocr_model_root: Path | None = None,
    ffmpeg_binary: Path | None = None,
    ffprobe_binary: Path | None = None,
) -> dict[str, object]:
    """Stage and index the frozen fixture without exposing its private locations."""

    selected_driver = driver
    if selected_driver is None:
        if (
            worker_overrides is None
            or ocr_model_root is None
            or ffmpeg_binary is None
            or ffprobe_binary is None
        ):
            raise RegressionFixtureError("managed_worker_configuration_required")
        # Validate and attest every private runtime binding before creating the
        # disposable root or copying source bytes into it.
        selected_driver = ProductionRegressionDriver(
            Path(models_root),
            worker_overrides=worker_overrides,
            ocr_model_root=ocr_model_root,
            ffmpeg_binary=ffmpeg_binary,
            ffprobe_binary=ffprobe_binary,
        )
    prepared = prepare_regression_fixture(
        dataset_path=dataset_path,
        bindings_path=bindings_path,
        data_root=data_root,
        models_root=models_root,
        timeout_seconds=timeout_seconds,
    )
    return complete_regression_fixture(
        prepared,
        driver=selected_driver,
        timeout_seconds=timeout_seconds,
    )


def prepare_regression_fixture(
    *,
    dataset_path: Path,
    bindings_path: Path,
    data_root: Path,
    models_root: Path,
    timeout_seconds: float = 1800.0,
) -> PreparedRegressionFixture:
    """Validate private inputs and stage exact bytes before workers start."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 1.0 <= float(timeout_seconds) <= 86_400.0
    ):
        raise RegressionFixtureError("timeout_invalid")
    dataset = load_regression_fixture(Path(dataset_path))
    bindings = load_local_input_bindings(Path(bindings_path), dataset)
    resolved_data_root = _validate_unused_data_root(Path(data_root))
    resolved_models_root = _validate_models_root(Path(models_root), resolved_data_root)
    _attest_all_sources(bindings, dataset, resolved_data_root)

    _create_or_seal_empty_data_root(resolved_data_root)
    media_root = resolved_data_root / "media"
    for directory in (
        media_root,
        resolved_data_root / "cache",
        resolved_data_root / "tmp",
    ):
        _create_private_directory(directory)
    staged = tuple(
        _stage_asset(
            binding,
            _asset_by_id(dataset, binding.asset_id),
            media_root,
        )
        for binding in bindings.inputs
    )
    return PreparedRegressionFixture(
        dataset=dataset,
        data_root=resolved_data_root,
        models_root=resolved_models_root,
        assets=staged,
    )


def complete_regression_fixture(
    prepared: PreparedRegressionFixture,
    *,
    driver: RegressionProvisionDriver,
    timeout_seconds: float = 1800.0,
) -> dict[str, object]:
    """Probe, production-index and publish receipts for a staged fixture."""

    if not isinstance(prepared, PreparedRegressionFixture):
        raise RegressionFixtureError("prepared_fixture_invalid")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 1.0 <= float(timeout_seconds) <= 86_400.0
    ):
        raise RegressionFixtureError("timeout_invalid")
    dataset = prepared.dataset
    staged = prepared.assets
    selected_driver = driver
    for item in staged:
        try:
            probed_duration = selected_driver.probe_duration(item.asset_id, item.path)
        except Exception as exc:
            raise RegressionFixtureError("prepared_input_probe_failed") from exc
        if (
            isinstance(probed_duration, bool)
            or not isinstance(probed_duration, (int, float))
            or not math.isfinite(float(probed_duration))
            or abs(float(probed_duration) - item.duration_seconds)
            > _DURATION_TOLERANCE_SECONDS
        ):
            raise RegressionFixtureError("prepared_input_duration_mismatch")

    try:
        provisioned = selected_driver.provision(
            data_root=prepared.data_root,
            assets=staged,
            timeout_seconds=float(timeout_seconds),
        )
    except RegressionFixtureError:
        raise
    except Exception as exc:
        raise RegressionFixtureError("production_ingest_failed") from exc
    _validate_provisioned_assets(staged, provisioned)

    benchmark_bindings = {
        item.asset_id: item.video_id
        for item in sorted(provisioned, key=lambda value: value.asset_id)
    }
    binding_bytes = canonical_json_bytes(benchmark_bindings)
    _write_exclusive_bytes(
        prepared.data_root / BENCHMARK_BINDINGS_FILENAME,
        binding_bytes,
    )
    receipt: dict[str, object] = {
        "schema_version": PROVISION_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "dataset_revision": dataset_revision(dataset),
        "dataset_schema_version": dataset.schema_version,
        "asset_count": len(dataset.assets),
        "case_count": len(dataset.cases),
        "source_group_count": len({case.split_group for case in dataset.cases}),
        "evidence_use": "regression_only",
        "promotion_eligible": False,
        "execution_mode": "warm",
        "required_profiles": list(REQUIRED_PROFILE_IDS),
        "preflight_required": True,
        "benchmark_bindings_file": BENCHMARK_BINDINGS_FILENAME,
        "benchmark_bindings_sha256": sha256(binding_bytes).hexdigest(),
        "benchmark_bindings": benchmark_bindings,
        "assets": [
            {
                "asset_id": item.asset_id,
                "video_id": item.video_id,
                "job_id": item.job_id,
                "plan_hash": item.plan_hash,
                "sha256": item.sha256,
                "byte_size": item.byte_size,
                "duration_seconds": item.duration_seconds,
            }
            for item in sorted(provisioned, key=lambda value: value.asset_id)
        ],
    }
    _write_exclusive_bytes(
        prepared.data_root / PROVISION_RECEIPT_FILENAME,
        canonical_json_bytes(receipt),
    )
    return receipt


def execute_phase0_regression_batch(
    *,
    dataset_path: Path,
    bindings_path: Path,
    policy_path: Path,
    data_root: Path,
    models_root: Path,
    scratch_parent: Path,
    registry_root: Path,
    worker_launch_path: Path,
    run_id_prefix: str,
    timeout_seconds: float = 1800.0,
    _code_identity: Callable[[], str] | None = None,
    _cluster_factory: Callable[..., object] | None = None,
    _driver_factory: Callable[..., RegressionProvisionDriver] | None = None,
    _profile_runner: Callable[..., tuple[dict[str, object], ...]] | None = None,
) -> dict[str, object]:
    """Own prepare → workers → ingest → five measured runs → cleanup."""

    from .cli import _current_code_sha

    code_identity = _code_identity or _current_code_sha
    inputs = _prevalidate_batch_inputs(
        dataset_path=dataset_path,
        bindings_path=bindings_path,
        policy_path=policy_path,
        data_root=data_root,
        models_root=models_root,
        scratch_parent=scratch_parent,
        registry_root=registry_root,
        worker_launch_path=worker_launch_path,
        code_identity=code_identity,
    )
    _require_id(run_id_prefix, "run_id_prefix")
    run_ids = tuple(
        f"{run_id_prefix}-{profile_id}" for profile_id in REQUIRED_PROFILE_IDS
    )
    if any(len(run_id) > 128 for run_id in run_ids):
        raise RegressionFixtureError("run_id_prefix_invalid")

    prepared = prepare_regression_fixture(
        dataset_path=inputs.dataset_path,
        bindings_path=inputs.bindings_path,
        data_root=inputs.data_root,
        models_root=inputs.models_root,
        timeout_seconds=timeout_seconds,
    )
    _initialize_empty_registry(inputs.registry_root)
    cluster_factory = _cluster_factory or start_regression_worker_cluster
    driver_factory = _driver_factory or ProductionRegressionDriver
    profile_runner = _profile_runner or run_required_product_profiles
    cluster: object | None = None
    profile_receipts: tuple[dict[str, object], ...] = ()
    primary_error: BaseException | None = None
    try:
        cluster = cluster_factory(
            data_root=inputs.data_root,
            scratch_parent=inputs.scratch_parent,
            models_root=inputs.models_root,
            configuration=inputs.worker_configuration,
        )
        _validate_owned_cluster(cluster)
        driver = driver_factory(
            inputs.models_root,
            worker_overrides=cluster.ingest_overrides,  # type: ignore[attr-defined]
            ocr_model_root=cluster.ocr_model_root,  # type: ignore[attr-defined]
            ffmpeg_binary=inputs.worker_configuration.ffmpeg_binary,
            ffprobe_binary=inputs.worker_configuration.ffprobe_binary,
        )
        complete_regression_fixture(
            prepared,
            driver=driver,
            timeout_seconds=timeout_seconds,
        )
        cluster.retire_ingest_workers()  # type: ignore[attr-defined]
        _validate_benchmark_cluster(cluster)
        benchmark_settings = _explicit_product_settings(
            data_root=inputs.data_root,
            models_root=inputs.models_root,
            ocr_model_root=cluster.ocr_model_root,  # type: ignore[attr-defined]
            worker_overrides=cluster.benchmark_overrides,  # type: ignore[attr-defined]
            ffmpeg_binary=inputs.worker_configuration.ffmpeg_binary,
            ffprobe_binary=inputs.worker_configuration.ffprobe_binary,
        )
        profile_receipts = profile_runner(
            dataset_path=inputs.dataset_path,
            bindings_path=inputs.data_root / BENCHMARK_BINDINGS_FILENAME,
            registry_root=inputs.registry_root,
            data_root=inputs.data_root,
            scratch_parent=inputs.scratch_parent,
            run_id_prefix=run_id_prefix,
            managed_workers=cluster.managed_workers,  # type: ignore[attr-defined]
            settings=benchmark_settings,
            expected_code_sha=inputs.code_sha,
            expected_dataset_revision=dataset_revision(inputs.dataset),
            code_identity=code_identity,
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if cluster is not None:
            try:
                cluster.close()  # type: ignore[attr-defined]
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise RegressionFixtureError("managed_worker_cleanup_failed") from cleanup_error

    if code_identity() != inputs.code_sha:
        raise RegressionFixtureError("code_identity_changed")
    current_dataset = load_regression_fixture(inputs.dataset_path)
    if dataset_revision(current_dataset) != dataset_revision(inputs.dataset):
        raise RegressionFixtureError("dataset_identity_changed")
    current_policy = load_frozen_metric_policy(policy_path)
    policy_revision = frozen_metric_policy_revision(inputs.policy)  # type: ignore[arg-type]
    if frozen_metric_policy_revision(current_policy) != policy_revision:
        raise RegressionFixtureError("metric_policy_identity_changed")
    if (
        _prevalidate_indexing_toolchain(inputs.worker_configuration)
        != inputs.indexing_toolchain_identity
    ):
        raise RegressionFixtureError("indexing_toolchain_identity_changed")
    if _prevalidate_ml_environment(inputs.worker_configuration) != (
        inputs.ml_environment_attestation_id,
        inputs.ml_environment_manifest_identity,
    ):
        raise RegressionFixtureError("ml_environment_identity_changed")

    receipt = {
        "schema_version": BASELINE_BATCH_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "code_sha": inputs.code_sha,
        "dataset_revision": dataset_revision(inputs.dataset),
        "policy_revision": policy_revision,
        "ml_environment_manifest_identity": (
            inputs.ml_environment_manifest_identity
        ),
        "ml_environment_attestation_id": inputs.ml_environment_attestation_id,
        "environment_bindings": dict(ENVIRONMENT_BINDINGS),
        "profile_runs": _portable_profile_runs(profile_receipts, run_id_prefix),
        "worker_lifecycle": {
            "cleanup_status": "complete",
            "retirement_status": "complete",
        },
    }
    _write_exclusive_bytes(
        inputs.data_root / BASELINE_BATCH_RECEIPT_FILENAME,
        canonical_json_bytes(receipt),
    )
    return receipt


def _portable_profile_runs(
    receipts: tuple[dict[str, object], ...],
    run_id_prefix: str,
) -> list[dict[str, str]]:
    if len(receipts) != len(REQUIRED_PROFILE_IDS):
        raise RegressionFixtureError("profile_batch_incomplete")
    result: list[dict[str, str]] = []
    for expected_profile, item in zip(REQUIRED_PROFILE_IDS, receipts, strict=True):
        run_id = item.get("run_id") if isinstance(item, dict) else None
        profile_id = item.get("profile_id") if isinstance(item, dict) else None
        manifest_sha256 = (
            item.get("manifest_sha256") if isinstance(item, dict) else None
        )
        if (
            profile_id != expected_profile
            or run_id != f"{run_id_prefix}-{expected_profile}"
            or not isinstance(manifest_sha256, str)
            or len(manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in manifest_sha256
            )
        ):
            raise RegressionFixtureError("profile_batch_incomplete")
        result.append(
            {
                "profile_id": expected_profile,
                "run_id": run_id,
                "manifest_sha256": manifest_sha256,
            }
        )
    return result


def run_required_product_profiles(
    *,
    dataset_path: Path,
    bindings_path: Path,
    registry_root: Path,
    data_root: Path,
    scratch_parent: Path,
    run_id_prefix: str,
    managed_workers: tuple[ManagedProcessBinding, ...],
    settings: AppSettings,
    expected_code_sha: str,
    expected_dataset_revision: str,
    code_identity: Callable[[], str],
    _executor: Callable[..., dict[str, object]] | None = None,
    _auditor: Callable[..., dict[str, object]] | None = None,
) -> tuple[dict[str, object], ...]:
    """Run the exact five-profile warm batch through BenchmarkRunner."""

    from .cli import _execute_product_run

    if not isinstance(managed_workers, tuple) or any(
        not isinstance(item, ManagedProcessBinding) for item in managed_workers
    ):
        raise RegressionFixtureError("managed_worker_bindings_invalid")
    roles = tuple(item.role for item in managed_workers)
    if roles != ("vision", "lighthouse", "qwen"):
        raise RegressionFixtureError("managed_worker_bindings_incomplete")
    if not isinstance(settings, AppSettings):
        raise RegressionFixtureError("benchmark_settings_invalid")
    executor = _executor or _execute_product_run
    auditor = _auditor or _audit_published_product_run
    receipts: list[dict[str, object]] = []
    for profile_id in REQUIRED_PROFILE_IDS:
        if code_identity() != expected_code_sha:
            raise RegressionFixtureError("code_identity_changed")
        if dataset_revision(load_regression_fixture(dataset_path)) != expected_dataset_revision:
            raise RegressionFixtureError("dataset_identity_changed")
        arguments = argparse.Namespace(
            dataset=dataset_path,
            registry=registry_root,
            data_dir=data_root,
            scratch_parent=scratch_parent,
            run_id=f"{run_id_prefix}-{profile_id}",
            bindings=bindings_path,
            profile=profile_id,
            execution_mode="warm",
            preflight=True,
        )
        preflight = executor(
            arguments,
            managed_workers=managed_workers,
            explicit_settings=settings,
            expected_code_sha=expected_code_sha,
        )
        measurement = preflight.get("measurement")
        if (
            preflight.get("status") != "ready"
            or not isinstance(measurement, dict)
            or measurement.get("status") != "ready"
            or preflight.get("profile_id") != profile_id
            or preflight.get("execution_mode") != "warm"
            or preflight.get("dataset_revision") != expected_dataset_revision
            or preflight.get("code_sha") != expected_code_sha
        ):
            raise RegressionFixtureError("profile_preflight_not_ready")
        arguments.preflight = False
        published = executor(
            arguments,
            managed_workers=managed_workers,
            explicit_settings=settings,
            expected_code_sha=expected_code_sha,
        )
        if (
            published.get("status") != "published"
            or published.get("run_status") != "complete"
            or published.get("profile_id") != profile_id
            or published.get("execution_mode") != "warm"
            or published.get("dataset_revision") != expected_dataset_revision
        ):
            raise RegressionFixtureError("profile_run_incomplete")
        receipts.append(
            auditor(
                dataset_path=dataset_path,
                registry_root=registry_root,
                run_id=arguments.run_id,
                profile_id=profile_id,
                expected_code_sha=expected_code_sha,
                expected_dataset_revision=expected_dataset_revision,
            )
        )
    if tuple(item.get("profile_id") for item in receipts) != REQUIRED_PROFILE_IDS:
        raise RegressionFixtureError("profile_batch_incomplete")
    return tuple(receipts)


def _audit_published_product_run(
    *,
    dataset_path: Path,
    registry_root: Path,
    run_id: str,
    profile_id: str,
    expected_code_sha: str,
    expected_dataset_revision: str,
) -> dict[str, object]:
    dataset = load_regression_fixture(dataset_path)
    registry = BenchmarkRunRegistry(registry_root)
    run = registry.read(run_id)
    audit_run_manifest(dataset, run)
    if (
        run.run_status != "complete"
        or run.execution_mode != "warm"
        or run.code_sha != expected_code_sha
        or run.dataset_revision != expected_dataset_revision
        or run.measurement_status != "complete"
        or run.measurement_evidence_status != "complete"
        or run.measurement_evidence is None
        or not run.measurement_evidence.rss_samples_bytes
    ):
        raise RegressionFixtureError("published_run_audit_failed")
    entry = next(
        (item for item in registry.list() if item.run_id == run_id),
        None,
    )
    if entry is None:
        raise RegressionFixtureError("published_run_audit_failed")
    return {
        "run_id": run_id,
        "profile_id": profile_id,
        "run_status": "complete",
        "measurement_status": "complete",
        "measurement_evidence_status": "complete",
        "manifest_sha256": entry.manifest_sha256,
        "rss_sample_count": len(run.measurement_evidence.rss_samples_bytes),
        "model_identities": [
            {
                "component_id": item.component_id,
                "identity": item.identity,
            }
            for item in run.model_identities
        ],
    }


def _prevalidate_batch_inputs(
    *,
    dataset_path: Path,
    bindings_path: Path,
    policy_path: Path,
    data_root: Path,
    models_root: Path,
    scratch_parent: Path,
    registry_root: Path,
    worker_launch_path: Path,
    code_identity: Callable[[], str],
) -> RegressionBatchInputs:
    """Finish every private/configuration check before creating output state."""

    code_sha = code_identity()
    dataset = load_regression_fixture(Path(dataset_path))
    policy = load_frozen_metric_policy(Path(policy_path))
    validate_frozen_metric_policy_product_dataset(policy, dataset)
    worker_configuration = load_worker_launch_configuration(Path(worker_launch_path))
    (
        ml_environment_attestation_id,
        ml_environment_manifest_identity,
    ) = _prevalidate_ml_environment(worker_configuration)
    indexing_toolchain_identity = _prevalidate_indexing_toolchain(
        worker_configuration
    )
    resolved_data = _validate_unused_data_root(Path(data_root))
    resolved_models = _validate_models_root(Path(models_root), resolved_data)
    resolved_scratch = _validate_empty_external_directory(
        Path(scratch_parent),
        code="scratch_parent_unsafe",
        must_exist=True,
    )
    resolved_registry = _validate_empty_external_directory(
        Path(registry_root),
        code="registry_root_unsafe",
        must_exist=False,
    )
    roots = (
        resolved_data,
        resolved_models,
        resolved_scratch,
        resolved_registry,
        worker_configuration.hf_home,
        worker_configuration.ocr_model_root,
    )
    for index, root in enumerate(roots):
        for other in roots[index + 1 :]:
            if root == other or root.is_relative_to(other) or other.is_relative_to(root):
                raise RegressionFixtureError("batch_roots_overlap")
    bindings = load_local_input_bindings(Path(bindings_path), dataset)
    _attest_all_sources(bindings, dataset, resolved_data)
    return RegressionBatchInputs(
        dataset=dataset,
        dataset_path=Path(dataset_path),
        bindings_path=Path(bindings_path),
        policy=policy,
        data_root=resolved_data,
        models_root=resolved_models,
        scratch_parent=resolved_scratch,
        registry_root=resolved_registry,
        worker_configuration=worker_configuration,
        code_sha=code_sha,
        indexing_toolchain_identity=indexing_toolchain_identity,
        ml_environment_attestation_id=ml_environment_attestation_id,
        ml_environment_manifest_identity=ml_environment_manifest_identity,
    )


def _prevalidate_ml_environment(
    configuration: WorkerLaunchConfiguration,
) -> tuple[str, str]:
    """Bind this process and every launch role to the pinned complete manifest."""

    from videoscope.ml_environment_attestation import (
        DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256,
        ML_ENVIRONMENT_REPORT_SCHEMA_VERSION,
        attest_ml_environment,
        load_ml_environment_manifest,
    )

    manifest_path = _PROJECT_ROOT / "workers" / "ml-environment.lock.json"
    try:
        manifest = load_ml_environment_manifest(
            manifest_path,
            expected_sha256=DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256,
        )
        environment_directories = {
            item.environment_id: item.directory for item in manifest.environments
        }
        if set(environment_directories) != set(ENVIRONMENT_BINDINGS.values()):
            raise RegressionFixtureError("ml_environment_binding_invalid")

        def expected_python(environment_id: str) -> Path:
            return (
                _PROJECT_ROOT
                / environment_directories[environment_id]
                / "bin"
                / "python"
            )

        owner_python = Path(os.path.abspath(sys.executable))
        if owner_python != expected_python(ENVIRONMENT_BINDINGS["owner"]):
            raise RegressionFixtureError("ml_environment_binding_invalid")
        for role in ("vision", "whisper", "lighthouse", "qwen", "ocr"):
            if configuration.executable(role) != expected_python(
                ENVIRONMENT_BINDINGS[role]
            ):
                raise RegressionFixtureError("ml_environment_binding_invalid")

        report = attest_ml_environment(
            _PROJECT_ROOT,
            manifest_path=manifest_path,
            expected_manifest_sha256=DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256,
        )
        manifest_identity = "sha256:" + manifest.raw_sha256
        if (
            not isinstance(report, dict)
            or report.get("schema_version") != ML_ENVIRONMENT_REPORT_SCHEMA_VERSION
            or report.get("status") != "complete"
            or report.get("attestation_id") != manifest.attestation_id
            or report.get("manifest_identity") != manifest_identity
            or report.get("failures") != []
        ):
            raise RegressionFixtureError("ml_environment_attestation_incomplete")
        return manifest.attestation_id, manifest_identity
    except RegressionFixtureError:
        raise
    except Exception as exc:
        raise RegressionFixtureError("ml_environment_attestation_incomplete") from exc


def _prevalidate_indexing_toolchain(
    configuration: WorkerLaunchConfiguration,
) -> str:
    """Attest the exact media executables before any private bytes are staged."""

    from videoscope.indexing_attestation import attest_indexing_toolchain

    try:
        toolchain = attest_indexing_toolchain(
            ffmpeg_binary=configuration.ffmpeg_binary,
            ffprobe_binary=configuration.ffprobe_binary,
        )
        identity = toolchain.verify_current()
    except Exception as exc:
        raise RegressionFixtureError("indexing_toolchain_unavailable") from exc
    digest = identity.removeprefix("sha256:") if isinstance(identity, str) else ""
    if (
        not isinstance(identity, str)
        or not identity.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RegressionFixtureError("indexing_toolchain_unavailable")
    return identity


def _validate_empty_external_directory(
    path: Path,
    *,
    code: str,
    must_exist: bool,
) -> Path:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path or path.is_symlink():
        raise RegressionFixtureError(code)
    if (
        path == Path(path.anchor)
        or path == Path.home().resolve()
        or path.is_relative_to(Path.home().resolve())
        or path == _PROJECT_ROOT
        or path.is_relative_to(_PROJECT_ROOT)
    ):
        raise RegressionFixtureError(code)
    if not path.exists():
        if must_exist:
            raise RegressionFixtureError(code)
        try:
            parent = path.parent.resolve(strict=True)
        except OSError as exc:
            raise RegressionFixtureError(code) from exc
        if parent != path.parent or not parent.is_dir():
            raise RegressionFixtureError(code)
        return path
    try:
        metadata = os.lstat(path)
        resolved = path.resolve(strict=True)
        empty = not any(path.iterdir())
    except OSError as exc:
        raise RegressionFixtureError(code) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or resolved != path
        or not empty
        or (must_exist and stat.S_IMODE(metadata.st_mode) != 0o700)
    ):
        raise RegressionFixtureError(code)
    return resolved


def _initialize_empty_registry(path: Path) -> None:
    try:
        if not path.exists():
            path.mkdir(mode=0o700)
        os.chmod(path, 0o700)
        registry = BenchmarkRunRegistry(path)
        if registry.rebuild():
            raise RegressionFixtureError("registry_root_not_empty")
    except RegressionFixtureError:
        raise
    except Exception as exc:
        raise RegressionFixtureError("registry_initialization_failed") from exc


def _validate_owned_cluster(cluster: object) -> None:
    workers = getattr(cluster, "managed_workers", None)
    ingest = getattr(cluster, "ingest_overrides", None)
    benchmark = getattr(cluster, "benchmark_overrides", None)
    ocr_root = getattr(cluster, "ocr_model_root", None)
    close = getattr(cluster, "close", None)
    retire = getattr(cluster, "retire_ingest_workers", None)
    if (
        not isinstance(workers, tuple)
        or any(not isinstance(item, ManagedProcessBinding) for item in workers)
        or tuple(item.role for item in workers) != MANAGED_WORKER_ROLES
        or not callable(close)
        or not callable(retire)
        or not isinstance(ocr_root, Path)
    ):
        raise RegressionFixtureError("managed_worker_cluster_invalid")
    _validate_worker_overrides(ingest)
    _validate_worker_overrides(benchmark)


def _validate_benchmark_cluster(cluster: object) -> None:
    workers = getattr(cluster, "managed_workers", None)
    retired = getattr(cluster, "retired_worker_roles", None)
    if (
        not isinstance(workers, tuple)
        or tuple(item.role for item in workers) != ("vision", "lighthouse", "qwen")
        or retired != ("vision_index", "whisper")
    ):
        raise RegressionFixtureError("managed_worker_retirement_invalid")


class _RegressionProvisionSettings(AppSettings):
    immutable_models_dir: Path = Field(exclude=True)
    ocr_model_root: Path = Field(exclude=True)
    ffmpeg_binary: Path = Field(exclude=True)
    ffprobe_binary: Path = Field(exclude=True)
    ocr_worker_environment: dict[str, str] = Field(
        default_factory=dict,
        exclude=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):  # type: ignore[no-untyped-def]
        del cls, settings_cls, env_settings, dotenv_settings, file_secret_settings
        return (init_settings,)

    @property
    def models_dir(self) -> Path:
        return self.immutable_models_dir

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.media_dir,
            self.thumbnails_dir,
            self.clips_dir,
            self.cache_dir,
            self.qdrant_dir,
            self.visual_index_dir,
            self.temp_dir,
        ):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)


_RegressionProvisionSettings.model_rebuild(_types_namespace={"Path": Path})


class _QueueWakeup:
    def __init__(self, queue: object) -> None:
        self._queue = queue

    def wake(self, _video_id: str) -> None:
        wake = getattr(self._queue, "wake", None)
        if not callable(wake):
            raise RegressionFixtureError("production_queue_unavailable")
        wake()


class ProductionRegressionDriver:
    """Provision via the same attested coordinator and runtime used in production."""

    def __init__(
        self,
        models_root: Path,
        *,
        worker_overrides: Mapping[str, object],
        ocr_model_root: Path,
        ffmpeg_binary: Path,
        ffprobe_binary: Path,
    ) -> None:
        from videoscope.indexing_attestation import attest_indexing_toolchain

        self._models_root = Path(models_root)
        self._worker_overrides = _validate_worker_overrides(worker_overrides)
        self._ocr_model_root = _validate_read_only_root(
            Path(ocr_model_root),
            code="ocr_model_root_unsafe",
        )
        self._ffmpeg_binary = Path(ffmpeg_binary)
        self._ffprobe_binary = Path(ffprobe_binary)
        try:
            self._toolchain = attest_indexing_toolchain(
                ffmpeg_binary=self._ffmpeg_binary,
                ffprobe_binary=self._ffprobe_binary,
            )
            self._ffmpeg = self._toolchain.create_ffmpeg()
        except Exception as exc:
            raise RegressionFixtureError("indexing_toolchain_unavailable") from exc

    def probe_duration(self, _asset_id: str, path: Path) -> float:
        probe = self._ffmpeg.probe(path)
        if probe.width <= 0 or probe.height <= 0 or probe.fps <= 0:
            raise RegressionFixtureError("prepared_input_media_invalid")
        return float(probe.duration)

    def provision(
        self,
        *,
        data_root: Path,
        assets: Sequence[StagedRegressionAsset],
        timeout_seconds: float,
    ) -> tuple[ProvisionedAsset, ...]:
        from videoscope.jobs import JobState
        from videoscope.media.uploads import validate_upload
        from videoscope.processing.coordinator import VideoIndexCoordinator
        from videoscope.repository import Repository
        from videoscope.runtime import build_runtime

        runtime: object | None = None
        results: list[ProvisionedAsset] = []
        primary_error: BaseException | None = None
        try:
            self._toolchain.verify_current()
            settings = _explicit_product_settings(
                data_root=data_root,
                models_root=self._models_root,
                ocr_model_root=self._ocr_model_root,
                worker_overrides=self._worker_overrides,
                ffmpeg_binary=self._ffmpeg_binary,
                ffprobe_binary=self._ffprobe_binary,
            )
            settings.ensure_directories()
            repository = Repository(settings.database_path)
            repository.initialize()
            runtime = build_runtime(
                settings,
                repository,
                indexing_toolchain=self._toolchain,
                ocr_worker_environment=settings.ocr_worker_environment,
            )
            runtime.start()
            plan_factory = getattr(runtime, "video_index_plan_factory", None)
            if not callable(plan_factory):
                raise RegressionFixtureError("production_index_plan_unavailable")
            coordinator = VideoIndexCoordinator(
                repository,
                plan_factory=plan_factory,
                wakeup=_QueueWakeup(runtime.queue),
            )
            jobs: list[tuple[StagedRegressionAsset, str, str]] = []
            for item in assets:
                video_id = f"reg_{item.sha256}"
                job_id = f"regjob_{item.sha256}"
                validate_upload(
                    f"{item.asset_id}.mp4",
                    item.byte_size,
                    max_bytes=settings.max_upload_bytes,
                )
                _video, job = coordinator.create_ingest(
                    video_id=video_id,
                    original_name=f"{item.asset_id}.mp4",
                    stored_name=item.path.name,
                    media_path=str(item.path),
                    size_bytes=item.byte_size,
                    source_sha256=item.sha256,
                    job_id=job_id,
                )
                jobs.append((item, video_id, job.job_id))

            deadline = monotonic() + timeout_seconds
            for item, video_id, job_id in jobs:
                completed = _wait_for_index_job(repository, job_id, deadline)
                if completed.state is not JobState.COMPLETE:
                    raise RegressionFixtureError("production_index_failed")
                records = repository.find_assets_by_sha256_bounded(
                    item.sha256,
                    limit=1,
                    video_id=video_id,
                )
                if len(records) != 1:
                    raise RegressionFixtureError("production_asset_binding_invalid")
                record = records[0]
                if (
                    record.sha256 != item.sha256
                    or record.byte_size != item.byte_size
                    or abs(float(record.duration_seconds) - item.duration_seconds)
                    > _DURATION_TOLERANCE_SECONDS
                ):
                    raise RegressionFixtureError("production_asset_binding_invalid")
                results.append(
                    ProvisionedAsset(
                        asset_id=item.asset_id,
                        video_id=video_id,
                        job_id=job_id,
                        plan_hash=completed.plan_hash,
                        sha256=item.sha256,
                        byte_size=item.byte_size,
                        duration_seconds=float(record.duration_seconds),
                    )
                )
            self._toolchain.verify_current()
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            if runtime is not None:
                try:
                    _close_runtime(runtime)
                except Exception:
                    if primary_error is None:
                        raise
        return tuple(results)


def _asset_by_id(dataset: BenchmarkDataset, asset_id: str) -> BenchmarkAsset:
    return next(asset for asset in dataset.assets if asset.asset_id == asset_id)


_WORKER_OVERRIDE_FIELDS = frozenset(
    {
        "ocr_worker_python",
        "ocr_worker_script",
        "ocr_worker_environment",
        "vision_worker_endpoint",
        "vision_worker_api_key",
        "whisper_worker_endpoint",
        "whisper_worker_api_key",
        "lighthouse_endpoint",
        "lighthouse_api_key",
        "qwen_video_endpoint",
        "qwen_video_api_key",
    }
)


def _explicit_product_settings(
    *,
    data_root: Path,
    models_root: Path,
    ocr_model_root: Path,
    worker_overrides: Mapping[str, object],
    ffmpeg_binary: Path,
    ffprobe_binary: Path,
) -> _RegressionProvisionSettings:
    overrides = _validate_worker_overrides(worker_overrides)
    try:
        return _RegressionProvisionSettings(
            _env_file=None,
            data_dir=data_root,
            immutable_models_dir=models_root,
            ocr_model_root=ocr_model_root,
            ffmpeg_binary=ffmpeg_binary,
            ffprobe_binary=ffprobe_binary,
            internvideo_api_key=None,
            internvideo_endpoint=None,
            lighthouse_allow_in_process=False,
            lighthouse_checkpoint=(
                models_root / "lighthouse" / "clip_qd_detr_qvhighlight.ckpt"
            ),
            lighthouse_clip_checkpoint=(
                models_root / "lighthouse" / "ViT-B-32.pt"
            ),
            qwen_video_allow_in_process=False,
            qwen_video_model=QWEN_VIDEO_MODEL,
            roboflow_api_key=None,
            roboflow_model_id=None,
            semantic_text_min_score=0.0,
            siglip_model=SIGLIP_224_MODEL,
            visual_min_score=0.0,
            whisper_model=WHISPER_MODEL,
            **overrides,
        )
    except RegressionFixtureError:
        raise
    except Exception as exc:
        raise RegressionFixtureError("managed_worker_configuration_invalid") from exc


def _validate_worker_overrides(
    values: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(values, Mapping) or set(values) != _WORKER_OVERRIDE_FIELDS:
        raise RegressionFixtureError("managed_worker_configuration_invalid")
    output = dict(values)
    for field_name in ("ocr_worker_python", "ocr_worker_script"):
        value = output[field_name]
        if not isinstance(value, Path) or not value.is_absolute():
            raise RegressionFixtureError("managed_worker_configuration_invalid")
        try:
            lexical = os.lstat(value)
            resolved = value.resolve(strict=True)
            metadata = os.stat(resolved)
        except OSError as exc:
            raise RegressionFixtureError("managed_worker_configuration_invalid") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (
                field_name == "ocr_worker_script"
                and not stat.S_ISREG(lexical.st_mode)
            )
            or (field_name == "ocr_worker_python" and not os.access(value, os.X_OK))
        ):
            raise RegressionFixtureError("managed_worker_configuration_invalid")
    worker_environment = output["ocr_worker_environment"]
    if (
        not isinstance(worker_environment, Mapping)
        or set(worker_environment)
        != {"HF_HOME", "HOME", "PATH", "TMPDIR", "XDG_CACHE_HOME"}
        or any(
            type(name) is not str
            or type(value) is not str
            or not value
            or "\x00" in value
            for name, value in worker_environment.items()
        )
    ):
        raise RegressionFixtureError("managed_worker_configuration_invalid")
    output["ocr_worker_environment"] = dict(worker_environment)
    for field_name in _WORKER_OVERRIDE_FIELDS - {
        "ocr_worker_python",
        "ocr_worker_script",
        "ocr_worker_environment",
    }:
        value = output[field_name]
        if type(value) is not str or not value or "\x00" in value:
            raise RegressionFixtureError("managed_worker_configuration_invalid")
    return output


def _validate_read_only_root(path: Path, *, code: str) -> Path:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path or path.is_symlink():
        raise RegressionFixtureError(code)
    try:
        metadata = os.lstat(path)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RegressionFixtureError(code) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or resolved == Path(resolved.anchor)
        or resolved == _PROJECT_ROOT
        or resolved.is_relative_to(_PROJECT_ROOT)
    ):
        raise RegressionFixtureError(code)
    return resolved


def _validate_unused_data_root(path: Path) -> Path:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise RegressionFixtureError("data_root_unsafe")
    home = Path.home().resolve()
    if (
        path == Path(path.anchor)
        or path == home
        or path.is_relative_to(home)
        or path == _PROJECT_ROOT
        or path.is_relative_to(_PROJECT_ROOT)
    ):
        raise RegressionFixtureError("data_root_unsafe")
    if path.is_symlink():
        raise RegressionFixtureError("data_root_unsafe")
    if path.exists():
        try:
            metadata = os.lstat(path)
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise RegressionFixtureError("data_root_unsafe") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or resolved != path
        ):
            raise RegressionFixtureError("data_root_unsafe")
        try:
            if any(path.iterdir()):
                raise RegressionFixtureError("data_root_not_empty")
        except OSError as exc:
            raise RegressionFixtureError("data_root_unsafe") from exc
        return resolved
    parent = path.parent
    try:
        parent_metadata = os.lstat(parent)
        resolved_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise RegressionFixtureError("data_root_unsafe") from exc
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or stat.S_ISLNK(parent_metadata.st_mode)
        or resolved_parent != parent
    ):
        raise RegressionFixtureError("data_root_unsafe")
    return path


def _validate_models_root(path: Path, data_root: Path) -> Path:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path or path.is_symlink():
        raise RegressionFixtureError("models_root_unsafe")
    try:
        metadata = os.lstat(path)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RegressionFixtureError("models_root_unsafe") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or resolved == Path(resolved.anchor)
        or resolved == data_root
        or resolved.is_relative_to(data_root)
        or data_root.is_relative_to(resolved)
        or resolved == _PROJECT_ROOT
        or resolved.is_relative_to(_PROJECT_ROOT)
    ):
        raise RegressionFixtureError("models_root_unsafe")
    return resolved


def _attest_all_sources(
    bindings: LocalInputBindings,
    dataset: BenchmarkDataset,
    data_root: Path,
) -> None:
    seen_files: set[tuple[int, int]] = set()
    for binding in bindings.inputs:
        asset = _asset_by_id(dataset, binding.asset_id)
        path = binding.path
        if path.is_relative_to(data_root):
            raise RegressionFixtureError("prepared_input_unsafe")
        size, digest, identity = _read_source_identity(path)
        if identity in seen_files:
            raise RegressionFixtureError("prepared_input_unsafe")
        seen_files.add(identity)
        if size != asset.byte_size or digest != asset.sha256:
            raise RegressionFixtureError("prepared_input_identity_mismatch")


def _read_source_identity(path: Path) -> tuple[int, str, tuple[int, int]]:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise RegressionFixtureError("prepared_input_unsafe")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RegressionFixtureError("prepared_input_unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_nlink < 1
        ):
            raise RegressionFixtureError("prepared_input_unsafe")
        digest = sha256()
        total = 0
        while chunk := os.read(descriptor, _COPY_CHUNK_BYTES):
            total += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _file_seal(before) != _file_seal(after) or total != before.st_size:
            raise RegressionFixtureError("prepared_input_changed")
        return total, digest.hexdigest(), (int(before.st_dev), int(before.st_ino))
    finally:
        os.close(descriptor)


def _file_seal(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _create_or_seal_empty_data_root(path: Path) -> None:
    try:
        if not path.exists():
            path.mkdir(mode=0o700)
        metadata = os.lstat(path)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or path.resolve(strict=True) != path
            or any(path.iterdir())
        ):
            raise OSError
        os.chmod(path, 0o700)
    except OSError as exc:
        raise RegressionFixtureError("data_root_unsafe") from exc


def _create_private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
        metadata = os.lstat(path)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise OSError
        os.chmod(path, 0o700)
    except OSError as exc:
        raise RegressionFixtureError("data_root_unsafe") from exc


def _stage_asset(
    binding: LocalPreparedInput,
    asset: BenchmarkAsset,
    media_root: Path,
) -> StagedRegressionAsset:
    destination = media_root / f"{asset.asset_id}.mp4"
    source_flags = (
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    with ExitStack() as descriptors:
        try:
            source = os.open(binding.path, source_flags)
        except OSError as exc:
            raise RegressionFixtureError("prepared_input_stage_failed") from exc
        descriptors.callback(os.close, source)
        try:
            target = os.open(
                destination,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
        except OSError as exc:
            raise RegressionFixtureError("prepared_input_stage_failed") from exc
        descriptors.callback(os.close, target)
        before = os.fstat(source)
        if not stat.S_ISREG(before.st_mode):
            raise RegressionFixtureError("prepared_input_unsafe")
        digest = sha256()
        total = 0
        while chunk := os.read(source, _COPY_CHUNK_BYTES):
            total += len(chunk)
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(target, view)
                if written <= 0:
                    raise RegressionFixtureError("prepared_input_stage_failed")
                view = view[written:]
        os.fsync(target)
        after = os.fstat(source)
    if (
        _file_seal(before) != _file_seal(after)
        or total != asset.byte_size
        or digest.hexdigest() != asset.sha256
    ):
        raise RegressionFixtureError("prepared_input_identity_mismatch")
    return StagedRegressionAsset(
        asset_id=asset.asset_id,
        path=destination,
        sha256=asset.sha256,
        byte_size=asset.byte_size,
        duration_seconds=asset.duration_seconds,
    )


def _validate_provisioned_assets(
    staged: tuple[StagedRegressionAsset, ...],
    provisioned: tuple[ProvisionedAsset, ...],
) -> None:
    if not isinstance(provisioned, tuple) or not all(
        isinstance(item, ProvisionedAsset) for item in provisioned
    ):
        raise RegressionFixtureError("production_receipt_invalid")
    staged_by_id = {item.asset_id: item for item in staged}
    if len(provisioned) != len(staged) or {
        item.asset_id for item in provisioned
    } != set(staged_by_id):
        raise RegressionFixtureError("production_receipt_invalid")
    if len({item.asset_id for item in provisioned}) != len(provisioned) or len(
        {item.video_id for item in provisioned}
    ) != len(provisioned):
        raise RegressionFixtureError("production_receipt_invalid")
    for item in provisioned:
        source = staged_by_id[item.asset_id]
        if (
            item.sha256 != source.sha256
            or item.byte_size != source.byte_size
            or abs(float(item.duration_seconds) - source.duration_seconds)
            > _DURATION_TOLERANCE_SECONDS
            or not item.video_id
            or not item.job_id
            or len(item.plan_hash) != 64
            or any(character not in "0123456789abcdef" for character in item.plan_hash)
        ):
            raise RegressionFixtureError("production_receipt_invalid")


def _wait_for_index_job(repository: object, job_id: str, deadline: float) -> object:
    from videoscope.jobs import JobState

    while monotonic() < deadline:
        job = repository.get_video_index_job(job_id)  # type: ignore[attr-defined]
        if job is None:
            raise RegressionFixtureError("production_index_job_missing")
        if job.state is JobState.COMPLETE:
            return job
        if job.state in {JobState.FAILED, JobState.CANCELLED}:
            raise RegressionFixtureError("production_index_failed")
        sleep(0.1)
    raise RegressionFixtureError("production_index_timeout")


def _close_runtime(runtime: object) -> None:
    close = getattr(runtime, "close", None)
    if not callable(close):
        raise RegressionFixtureError("production_runtime_cleanup_failed")
    deadline = monotonic() + 60.0
    while monotonic() < deadline:
        if close() is True:
            return
        sleep(0.1)
    raise RegressionFixtureError("production_runtime_cleanup_failed")


class _RegressionUsageError(ValueError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise _RegressionUsageError("invalid regression batch arguments")


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Own the frozen Phase-0 five-profile regression batch.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    batch = subparsers.add_parser("batch")
    batch.add_argument("--dataset", required=True, type=Path)
    batch.add_argument("--bindings", required=True, type=Path)
    batch.add_argument("--policy", required=True, type=Path)
    batch.add_argument("--data-root", required=True, type=Path)
    batch.add_argument("--models-root", required=True, type=Path)
    batch.add_argument("--scratch-parent", required=True, type=Path)
    batch.add_argument("--registry", required=True, type=Path)
    batch.add_argument("--worker-launch", required=True, type=Path)
    batch.add_argument("--run-id-prefix", required=True)
    batch.add_argument("--timeout-seconds", type=float, default=1800.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _build_parser().parse_args(argv)
        if arguments.command != "batch":
            raise _RegressionUsageError("invalid regression batch command")
        receipt = execute_phase0_regression_batch(
            dataset_path=arguments.dataset,
            bindings_path=arguments.bindings,
            policy_path=arguments.policy,
            data_root=arguments.data_root,
            models_root=arguments.models_root,
            scratch_parent=arguments.scratch_parent,
            registry_root=arguments.registry,
            worker_launch_path=arguments.worker_launch,
            run_id_prefix=arguments.run_id_prefix,
            timeout_seconds=arguments.timeout_seconds,
        )
    except _RegressionUsageError:
        print(
            '{"reason_code":"usage_error","status":"failed"}',
            file=sys.stderr,
        )
        return 2
    except RegressionFixtureError as error:
        print(
            json.dumps(
                {"status": "failed", "reason_code": error.code},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    except ManagedWorkerError as error:
        print(
            json.dumps(
                {"status": "failed", "reason_code": error.code},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 8
    except MeasurementError as error:
        print(
            json.dumps(
                {"status": "failed", "reason_code": error.code},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 9
    except BenchmarkExecutionError:
        print(
            '{"reason_code":"benchmark_execution_failed","status":"failed"}',
            file=sys.stderr,
        )
        return 8
    except (BenchmarkDataError, OSError, ValueError):
        print(
            '{"reason_code":"fixture_manifest_invalid","status":"failed"}',
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print(
            '{"reason_code":"interrupted","status":"failed"}',
            file=sys.stderr,
        )
        return 130
    except Exception:
        print(
            '{"reason_code":"internal_error","status":"failed"}',
            file=sys.stderr,
        )
        return 70
    print(canonical_json_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
