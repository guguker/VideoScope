from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
from typing import Sequence

from videoscope.config import AppSettings

from .catalog import AssetResolutionError, DatasetCatalog, LocalAssetResolver
from .comparison import compare_runs, comparison_json
from .environment import (
    BenchmarkEnvironmentCleanupError,
    BenchmarkEnvironmentError,
    ProductBenchmarkEnvironment,
    open_product_benchmark_environment,
)
from .measurements import (
    DeclaredStorageRoot,
    ManagedProcessBinding,
    MeasurementError,
    MeasurementUnavailableError,
    SystemMeasurementFactory,
    create_native_process_snapshot_provider,
)
from .policy import load_policy
from .product_runtime import ProductSnapshotCleanupError, ProductSnapshotError
from .profiles import BenchmarkProfile, get_profile
from .runner import (
    EXECUTION_LIFECYCLE_COMPONENT_ID,
    BenchmarkExecutionError,
    BenchmarkRunner,
    ExecutionIdentities,
    _validate_profile_execution_identities,
    audit_run_manifest,
)
from .schema import (
    BenchmarkDataError,
    BenchmarkDataset,
    ComponentIdentity,
    HardwareProfile,
    _require_id,
)
from .serialization import dataset_revision, parse_json_object, run_to_dict
from .storage import (
    BenchmarkDurabilityError,
    BenchmarkRunRegistry,
    RunRegistryEntry,
    _read_bounded_file,
    load_dataset,
)


EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_DATA = 3
EXIT_NOT_FOUND = 4
EXIT_CONFLICT = 5
EXIT_AUDIT = 6
EXIT_IO = 7
EXIT_EXECUTION = 8
EXIT_MEASUREMENT = 9
EXIT_INTERNAL = 70
EXIT_INTERRUPTED = 130
EXIT_TERMINATED = 143

_RUN_PROFILE_ID = "lexical_qdrant"
_RUN_EXECUTION_MODE = "warm"
_MAX_BINDINGS_BYTES = 1024 * 1024
_MAX_ENVIRONMENT_CLOSE_ATTEMPTS = 16
_GIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class _CliUsageError(ValueError):
    pass


class _CodeIdentityError(BenchmarkExecutionError):
    pass


class _TerminationRequested(BaseException):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise _CliUsageError("invalid command arguments")


def main(argv: Sequence[str] | None = None) -> int:
    previous_handler = _install_termination_handler()
    try:
        return _main(argv)
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)


def _main(argv: Sequence[str] | None = None) -> int:
    arguments: argparse.Namespace | None = None
    try:
        arguments = _parser().parse_args(argv)
        value = _dispatch(arguments)
        if isinstance(value, str):
            sys.stdout.write(value)
            sys.stdout.write("\n")
        else:
            _write_json(sys.stdout, value)
        if (
            arguments.command == "run"
            and getattr(arguments, "preflight", False)
        ):
            return _preflight_exit_code(value)
        return EXIT_SUCCESS
    except _CliUsageError:
        return _fail(
            EXIT_USAGE,
            "usage_error",
            "invalid benchmark command arguments",
        )
    except KeyboardInterrupt:
        return _fail(
            EXIT_INTERRUPTED,
            "interrupted",
            "benchmark command was interrupted",
        )
    except _TerminationRequested:
        return _fail(
            EXIT_TERMINATED,
            "terminated",
            "benchmark command was terminated",
        )
    except (
        AssetResolutionError,
        BenchmarkEnvironmentError,
        ProductSnapshotError,
    ):
        return _fail(
            EXIT_EXECUTION,
            "execution_failed",
            "benchmark execution or preflight failed",
        )
    except MeasurementUnavailableError as error:
        return _fail(
            EXIT_MEASUREMENT,
            error.code,
            "benchmark measurement is unavailable",
        )
    except MeasurementError as error:
        return _fail(
            EXIT_MEASUREMENT,
            error.code,
            "benchmark measurement failed",
        )
    except BenchmarkExecutionError:
        if arguments is not None and arguments.command == "run":
            return _fail(
                EXIT_EXECUTION,
                "execution_failed",
                "benchmark execution or preflight failed",
            )
        return _fail(
            EXIT_AUDIT,
            "audit_failed",
            "benchmark run audit failed",
        )
    except (FileNotFoundError, KeyError):
        return _fail(
            EXIT_NOT_FOUND,
            "not_found",
            "requested benchmark data was not found",
        )
    except FileExistsError:
        return _fail(
            EXIT_CONFLICT,
            "conflict",
            "benchmark destination already exists",
        )
    except BenchmarkDataError:
        return _fail(
            EXIT_DATA,
            "invalid_data",
            "benchmark input or stored data is invalid",
        )
    except (BenchmarkDurabilityError, OSError):
        return _fail(
            EXIT_IO,
            "io_error",
            "benchmark storage operation failed",
        )
    except Exception:
        return _fail(
            EXIT_INTERNAL,
            "internal_error",
            "benchmark command failed",
        )


def _preflight_exit_code(value: object) -> int:
    if not isinstance(value, dict):
        raise BenchmarkExecutionError("benchmark preflight result is invalid")
    status = value.get("status")
    measurement = value.get("measurement")
    if not isinstance(measurement, dict):
        raise BenchmarkExecutionError("benchmark preflight result is invalid")
    measurement_status = measurement.get("status")
    if status == "ready" and measurement_status == "ready":
        return EXIT_SUCCESS
    if measurement_status != "ready":
        return EXIT_MEASUREMENT
    if status == "not_ready":
        return EXIT_EXECUTION
    raise BenchmarkExecutionError("benchmark preflight result is invalid")


def _install_termination_handler():  # type: ignore[no-untyped-def]
    try:
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(
            signal.SIGTERM,
            lambda _signum, _frame: (_raise_termination()),
        )
    except (OSError, ValueError):
        return None
    return previous


def _raise_termination() -> None:
    raise _TerminationRequested


def _parser() -> _ArgumentParser:
    parser = _ArgumentParser(prog="videoscope-benchmark")
    commands = parser.add_subparsers(dest="command", required=True)

    dataset = commands.add_parser("dataset")
    dataset_commands = dataset.add_subparsers(
        dest="dataset_command",
        required=True,
    )
    validate = dataset_commands.add_parser("validate")
    validate.add_argument("file", type=Path)
    import_command = dataset_commands.add_parser("import")
    import_command.add_argument("file", type=Path)
    import_command.add_argument("--catalog", type=Path, required=True)

    runs = commands.add_parser("runs")
    run_commands = runs.add_subparsers(dest="runs_command", required=True)
    list_command = run_commands.add_parser("list")
    list_command.add_argument("--registry", type=Path, required=True)
    show = run_commands.add_parser("show")
    show.add_argument("run_id")
    show.add_argument("--registry", type=Path, required=True)
    audit = run_commands.add_parser("audit")
    audit.add_argument("run_id")
    audit.add_argument("--registry", type=Path, required=True)
    audit.add_argument("--dataset", type=Path, required=True)

    compare = commands.add_parser("compare")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--registry", type=Path, required=True)
    compare.add_argument("--policy", type=Path, required=True)
    compare.add_argument("--dataset", type=Path, required=True)

    run = commands.add_parser("run")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--registry", type=Path, required=True)
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--scratch-parent", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--bindings", type=Path)
    run.add_argument("--profile", default=_RUN_PROFILE_ID)
    run.add_argument("--execution-mode", default=_RUN_EXECUTION_MODE)
    run.add_argument("--preflight", action="store_true")
    return parser


def _dispatch(arguments: argparse.Namespace) -> dict[str, object] | str:
    if arguments.command == "dataset":
        if arguments.dataset_command == "validate":
            dataset = load_dataset(arguments.file)
            return {"status": "valid", "dataset": _dataset_summary(dataset)}
        if arguments.dataset_command == "import":
            imported = DatasetCatalog(arguments.catalog).import_file(arguments.file)
            return {
                "status": "imported",
                "catalog_entry": (
                    f"{imported.dataset.dataset_id}/{imported.revision}.json"
                ),
                "dataset": _dataset_summary(imported.dataset),
            }
    if arguments.command == "runs":
        registry = _existing_registry(arguments.registry)
        if arguments.runs_command == "list":
            return {
                "runs": [_registry_entry(item) for item in registry.list()],
            }
        if arguments.runs_command == "show":
            return run_to_dict(registry.read(arguments.run_id))
        if arguments.runs_command == "audit":
            run = registry.read(arguments.run_id)
            dataset = load_dataset(arguments.dataset)
            audit_run_manifest(dataset, run)
            return {
                "status": "valid",
                "run_id": run.run_id,
                "dataset_revision": run.dataset_revision,
                "case_count": len(run.case_outcomes),
            }
    if arguments.command == "compare":
        registry = _existing_registry(arguments.registry)
        baseline = registry.read(arguments.baseline)
        candidate = registry.read(arguments.candidate)
        policy = load_policy(arguments.policy)
        dataset = load_dataset(arguments.dataset)
        return comparison_json(compare_runs(dataset, baseline, candidate, policy))
    if arguments.command == "run":
        return _execute_product_run(arguments)
    raise _CliUsageError("unsupported command")


def _execute_product_run(
    arguments: argparse.Namespace,
    *,
    managed_workers: tuple[ManagedProcessBinding, ...] = (),
    explicit_settings: AppSettings | None = None,
    expected_code_sha: str | None = None,
) -> dict[str, object]:
    if arguments.execution_mode != _RUN_EXECUTION_MODE:
        raise BenchmarkExecutionError(
            "only warm benchmark execution is supported"
        )
    _require_id(arguments.run_id, "run_id")
    dataset = load_dataset(arguments.dataset)
    profile = get_profile(arguments.profile)
    registry = _existing_registry(arguments.registry)
    run_path = registry.runs_path / arguments.run_id
    if (
        any(entry.run_id == arguments.run_id for entry in registry.list())
        or run_path.exists()
        or run_path.is_symlink()
    ):
        raise FileExistsError("benchmark run already exists")
    bindings = (
        None
        if arguments.bindings is None
        else _load_exact_bindings(arguments.bindings, dataset)
    )
    code_sha = _current_code_sha()
    if expected_code_sha is not None and code_sha != expected_code_sha:
        raise _CodeIdentityError(
            "Git HEAD identity differs from the batch identity"
        )
    hardware = _current_hardware_profile()
    if explicit_settings is None:
        try:
            settings = AppSettings(data_dir=arguments.data_dir)
        except Exception as error:
            raise BenchmarkExecutionError(
                "validated product benchmark settings are unavailable"
            ) from error
    else:
        if not isinstance(explicit_settings, AppSettings):
            raise BenchmarkExecutionError(
                "explicit product benchmark settings are invalid"
            )
        expected_data_dir = Path(os.path.abspath(arguments.data_dir))
        actual_data_dir = Path(os.path.abspath(explicit_settings.data_dir))
        if expected_data_dir != actual_data_dir:
            raise BenchmarkExecutionError(
                "explicit product benchmark data root differs from the request"
            )
        settings = explicit_settings
    environment = _open_environment(
        settings,
        arguments.scratch_parent,
        profile_id=profile.profile_id,
        execution_mode=arguments.execution_mode,
    )
    try:
        resolver = _environment_resolver(environment, bindings)
        source_bound_workers = environment.probe_worker_sources()
        if arguments.preflight:
            summary = _preflight_environment(
                environment,
                resolver,
                dataset,
                profile=profile,
                run_id=arguments.run_id,
                code_sha=code_sha,
                execution_mode=arguments.execution_mode,
            )
            if managed_workers:
                measurement_summary = _preflight_measurement(
                    settings,
                    environment,
                    profile,
                    managed_workers=managed_workers,
                )
            else:
                measurement_summary = _preflight_measurement(
                    settings,
                    environment,
                    profile,
                )
            summary["measurement"] = measurement_summary
            if source_bound_workers:
                summary["source_bound_worker_probes"] = list(
                    source_bound_workers
                )
            if measurement_summary["status"] != "ready":
                summary["status"] = "not_ready"
        else:
            if managed_workers:
                measurement = _create_product_measurement_factory(
                    settings,
                    environment,
                    profile,
                    managed_workers=managed_workers,
                )
            else:
                measurement = _create_product_measurement_factory(
                    settings,
                    environment,
                    profile,
                )
            runner = BenchmarkRunner(
                registry=registry,
                asset_resolver=resolver,
                search=environment.search_adapter,
                hardware=hardware,
                code_sha=code_sha,
                measurement=measurement,
            )
            run = runner.run(
                dataset,
                profile_id=profile.profile_id,
                run_id=arguments.run_id,
                execution_mode=arguments.execution_mode,
                publish=False,
            )
    except BaseException as error:
        _raise_after_environment_cleanup(environment, error)
    _close_environment_fully(environment)
    if arguments.preflight:
        if _current_code_sha() != code_sha or (
            expected_code_sha is not None and code_sha != expected_code_sha
        ):
            raise _CodeIdentityError(
                "Git HEAD identity changed during benchmark preflight"
            )
        return summary
    if run.run_status != "complete":
        raise BenchmarkExecutionError(
            "only a complete benchmark run may be published"
        )
    if run.measurement_status != "complete":
        raise MeasurementError("benchmark_measurement_incomplete")
    audit_run_manifest(dataset, run)
    if _current_code_sha() != code_sha or (
        expected_code_sha is not None and code_sha != expected_code_sha
    ):
        raise _CodeIdentityError(
            "Git HEAD identity changed during benchmark execution"
        )
    runner.publish_prepared_run(run)
    return {
        "status": "published",
        "run_id": run.run_id,
        "run_status": run.run_status,
        "profile_id": profile.profile_id,
        "execution_mode": arguments.execution_mode,
        "dataset_revision": run.dataset_revision,
        "case_count": len(run.case_outcomes),
    }


def _environment_resolver(
    environment: ProductBenchmarkEnvironment,
    bindings: dict[str, str] | None,
) -> LocalAssetResolver:
    if bindings is None:
        return environment.asset_resolver
    return LocalAssetResolver(environment.repository, bindings=bindings)


def _external_worker_roles(profile: BenchmarkProfile) -> tuple[str, ...]:
    roles: set[str] = set()
    plan = profile.search_plan
    if plan.visual_search != "disabled":
        roles.add("vision")
    if plan.lighthouse:
        roles.add("lighthouse")
    if plan.reranker == "qwen":
        roles.add("qwen")
    elif plan.reranker == "internvideo":
        roles.add("internvideo")
    return tuple(sorted(roles))


def _create_product_measurement_factory(
    settings: AppSettings,
    environment: ProductBenchmarkEnvironment,
    profile: BenchmarkProfile,
    *,
    managed_workers: tuple[ManagedProcessBinding, ...] = (),
) -> SystemMeasurementFactory:
    external_roles = _external_worker_roles(profile)
    managed_roles = {
        worker.role
        for worker in managed_workers
        if isinstance(worker, ManagedProcessBinding)
    }
    if not set(external_roles) <= managed_roles:
        raise MeasurementUnavailableError(
            "external_worker_process_binding_unavailable"
        )
    data_root = Path(os.path.abspath(os.fspath(settings.data_dir)))
    scratch_root = Path(environment.scratch_root)
    provider = create_native_process_snapshot_provider(root_pid=os.getpid())
    return SystemMeasurementFactory(
        storage_roots=(
            DeclaredStorageRoot(
                root_id="active-text-vector-storage",
                path=data_root / "qdrant" / "text",
                purpose="active_immutable_artifacts",
            ),
            DeclaredStorageRoot(
                root_id="benchmark-private-scratch",
                path=scratch_root,
                purpose="benchmark_scratch",
            ),
        ),
        execution_mode="warm",
        cache_policy_identity="process-cache-preserved@1",
        process_provider=provider,
        process_root_pid=os.getpid(),
        managed_workers=managed_workers,
    )


def _measurement_preflight_failure(
    *,
    status: str,
    reason_code: str,
    external_roles: tuple[str, ...],
) -> dict[str, object]:
    return {
        "external_worker_roles": list(external_roles),
        "measurement_protocol_identity": None,
        "metal_telemetry_status": "unavailable",
        "process_provider_identity": None,
        "reason_code": reason_code,
        "status": status,
        "storage_root_ids": [],
    }


def _preflight_measurement(
    settings: AppSettings,
    environment: ProductBenchmarkEnvironment,
    profile: BenchmarkProfile,
    *,
    managed_workers: tuple[ManagedProcessBinding, ...] = (),
) -> dict[str, object]:
    external_roles = _external_worker_roles(profile)
    try:
        factory = _create_product_measurement_factory(
            settings,
            environment,
            profile,
            managed_workers=managed_workers,
        )
    except MeasurementUnavailableError as error:
        return _measurement_preflight_failure(
            status="not_configured" if external_roles else "unavailable",
            reason_code=error.code,
            external_roles=external_roles,
        )
    except MeasurementError as error:
        return _measurement_preflight_failure(
            status="failed",
            reason_code=error.code,
            external_roles=external_roles,
        )

    session = None
    evidence = None
    failure: dict[str, object] | None = None
    try:
        session = factory.open_session()
        session.start()
        evidence = session.finish()
    except MeasurementUnavailableError as error:
        failure = _measurement_preflight_failure(
            status="unavailable",
            reason_code=error.code,
            external_roles=external_roles,
        )
    except MeasurementError as error:
        failure = _measurement_preflight_failure(
            status="failed",
            reason_code=error.code,
            external_roles=external_roles,
        )
    except (KeyboardInterrupt, SystemExit, _TerminationRequested):
        raise
    except Exception:
        failure = _measurement_preflight_failure(
            status="failed",
            reason_code="measurement_probe_failed",
            external_roles=external_roles,
        )
    if session is not None:
        try:
            session.close()
        except (KeyboardInterrupt, SystemExit, _TerminationRequested):
            raise
        except Exception:
            return _measurement_preflight_failure(
                status="failed",
                reason_code="measurement_probe_cleanup_failed",
                external_roles=external_roles,
            )
    if failure is not None:
        return failure
    if evidence is None:
        return _measurement_preflight_failure(
            status="failed",
            reason_code="measurement_probe_failed",
            external_roles=external_roles,
        )

    provider = factory.process_provider
    provider_identity = getattr(provider, "identity", None)
    return {
        "external_worker_roles": list(external_roles),
        "managed_worker_roles": [
            item.role for item in sorted(managed_workers, key=lambda value: value.role)
        ],
        "measurement_protocol_identity": factory.protocol_identity().identity,
        "metal_telemetry_status": evidence.metal_telemetry_status,
        "process_provider_identity": provider_identity,
        "reason_code": None,
        "status": "ready",
        "storage_root_ids": [root.root_id for root in factory.storage_roots],
    }


def _preflight_environment(
    environment: ProductBenchmarkEnvironment,
    resolver: LocalAssetResolver,
    dataset: BenchmarkDataset,
    *,
    profile: BenchmarkProfile,
    run_id: str,
    code_sha: str,
    execution_mode: str,
) -> dict[str, object]:
    try:
        assets = tuple(
            resolver.resolve(asset)
            for asset in sorted(dataset.assets, key=lambda item: item.asset_id)
        )
    except Exception as error:
        raise BenchmarkExecutionError(
            "benchmark asset preflight failed"
        ) from error
    session = None
    try:
        session = environment.search_adapter.open_session(
            profile,
            assets,
            execution_mode=execution_mode,
        )
        required_methods = (
            "identities",
            "lifecycle_identity",
            "capability_state",
            "search",
            "close",
        )
        if any(not callable(getattr(session, name, None)) for name in required_methods):
            raise BenchmarkExecutionError(
                "benchmark preflight session contract is invalid"
            )
        identities = session.identities()
        if not isinstance(identities, ExecutionIdentities):
            raise BenchmarkExecutionError(
                "benchmark preflight identities are invalid"
            )
        lifecycle = session.lifecycle_identity()
        if (
            not isinstance(lifecycle, ComponentIdentity)
            or lifecycle.component_id != EXECUTION_LIFECYCLE_COMPONENT_ID
            or not lifecycle.identity.startswith(f"{execution_mode}:")
            or not lifecycle.identity.removeprefix(
                f"{execution_mode}:"
            ).strip()
        ):
            raise BenchmarkExecutionError(
                "benchmark preflight lifecycle identity is invalid"
            )
        _validate_preflight_identities(
            identities,
            lifecycle=lifecycle,
            environment=environment,
            profile=profile,
            execution_mode=execution_mode,
        )
        capability_matrix: list[dict[str, object]] = []
        all_ready = True
        for asset in assets:
            states: dict[str, str] = {}
            for capability in profile.required_capabilities:
                state = session.capability_state(asset, capability)
                if state not in {
                    "cancelled",
                    "complete",
                    "failed",
                    "missing",
                    "not_configured",
                    "queued",
                    "running",
                    "stale",
                }:
                    raise BenchmarkExecutionError(
                        "benchmark preflight capability state is invalid"
                    )
                states[capability] = state
            asset_ready = all(state == "complete" for state in states.values())
            all_ready = all_ready and asset_ready
            capability_matrix.append(
                {
                    "asset_id": asset.asset_id,
                    "capabilities": states,
                    "status": "ready" if asset_ready else "not_ready",
                }
            )
    except BenchmarkExecutionError:
        raise
    except Exception as error:
        raise BenchmarkExecutionError("benchmark preflight failed") from error
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as error:
                raise BenchmarkExecutionError(
                    "benchmark preflight session could not be closed"
                ) from error
    return {
        "status": "ready" if all_ready else "not_ready",
        "run_id": run_id,
        "profile_id": profile.profile_id,
        "execution_mode": execution_mode,
        "dataset_revision": dataset_revision(dataset),
        "code_sha": code_sha,
        "environment_identity": environment.identity.identity,
        "asset_count": len(assets),
        "capability_count": len(assets) * len(profile.required_capabilities),
        "capability_matrix": capability_matrix,
    }


def _validate_preflight_identities(
    identities: ExecutionIdentities,
    *,
    lifecycle: ComponentIdentity,
    environment: ProductBenchmarkEnvironment,
    profile: BenchmarkProfile,
    execution_mode: str,
) -> None:
    try:
        _validate_profile_execution_identities(
            identities,
            profile=profile,
            lifecycle=lifecycle,
            execution_mode=execution_mode,
            expected_environment_identity=environment.identity.identity,
        )
    except BenchmarkExecutionError:
        raise BenchmarkExecutionError(
            "benchmark preflight product identities are invalid"
        ) from None


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _open_environment(
    settings: AppSettings,
    scratch_parent: Path,
    *,
    profile_id: str,
    execution_mode: str,
) -> ProductBenchmarkEnvironment:
    try:
        return open_product_benchmark_environment(
            settings,
            scratch_parent,
            profile_id=profile_id,
            execution_mode=execution_mode,
        )
    except BenchmarkEnvironmentCleanupError as error:
        try:
            _close_environment_fully(error.environment)
        except (KeyboardInterrupt, SystemExit, _TerminationRequested):
            raise
        except BaseException as cleanup_error:
            raise BenchmarkExecutionError(
                "benchmark environment setup cleanup failed"
            ) from cleanup_error
        raise BenchmarkExecutionError(
            "benchmark environment setup failed"
        ) from error
    except ProductSnapshotCleanupError as error:
        last_error: BaseException = error
        cancellation: BaseException | None = None
        for _attempt in range(_MAX_ENVIRONMENT_CLOSE_ATTEMPTS):
            try:
                error.retry_cleanup()
            except (
                KeyboardInterrupt,
                SystemExit,
                _TerminationRequested,
            ) as interrupted:
                cancellation = interrupted
                last_error = interrupted
                continue
            except BaseException as cleanup_error:
                last_error = cleanup_error
                continue
            if not error.cleanup_pending:
                break
        if error.cleanup_pending:
            raise BenchmarkExecutionError(
                "product benchmark setup cleanup failed"
            ) from last_error
        if cancellation is not None:
            raise cancellation
        raise BenchmarkExecutionError(
            "product benchmark environment setup failed"
        ) from error


def _close_environment_fully(environment: ProductBenchmarkEnvironment) -> None:
    last_error: BaseException | None = None
    cancellation: BaseException | None = None
    for _attempt in range(_MAX_ENVIRONMENT_CLOSE_ATTEMPTS):
        try:
            environment.close()
        except (
            KeyboardInterrupt,
            SystemExit,
            _TerminationRequested,
        ) as interrupted:
            cancellation = interrupted
            last_error = interrupted
            if environment.is_closed:
                raise interrupted
            continue
        except BaseException as error:
            last_error = error
            if environment.is_closed:
                raise BenchmarkExecutionError(
                    "benchmark final attestation failed"
                ) from error
            continue
        if environment.is_closed:
            if cancellation is not None:
                raise cancellation
            return
        last_error = BenchmarkExecutionError(
            "benchmark environment close contract is incomplete"
        )
    raise BenchmarkExecutionError(
        "benchmark environment could not be closed completely"
    ) from last_error


def _raise_after_environment_cleanup(
    environment: ProductBenchmarkEnvironment,
    primary_error: BaseException,
) -> None:
    try:
        _close_environment_fully(environment)
    except BaseException as cleanup_error:
        if isinstance(
            primary_error,
            (KeyboardInterrupt, SystemExit, _TerminationRequested),
        ):
            raise primary_error from cleanup_error
        if isinstance(primary_error, Exception) and isinstance(
            cleanup_error,
            Exception,
        ):
            raise BenchmarkExecutionError(
                "benchmark execution and environment cleanup failed"
            ) from ExceptionGroup(
                "benchmark execution and environment cleanup failed",
                [primary_error, cleanup_error],
            )
        raise cleanup_error from primary_error
    raise primary_error


def _load_exact_bindings(
    path: Path,
    dataset: BenchmarkDataset,
) -> dict[str, str]:
    value = parse_json_object(
        _read_bounded_file(path, _MAX_BINDINGS_BYTES, "benchmark asset bindings"),
        "benchmark asset bindings",
    )
    expected_aliases = {asset.asset_id for asset in dataset.assets}
    if set(value) != expected_aliases:
        raise BenchmarkDataError(
            "benchmark asset bindings must name every dataset alias exactly"
        )
    if any(type(video_id) is not str for video_id in value.values()):
        raise BenchmarkDataError(
            "benchmark asset bindings must map aliases to video ids"
        )
    bindings = {
        alias: video_id
        for alias, video_id in value.items()
        if isinstance(video_id, str)
    }
    # Reuse the resolver's strict portable/local identifier validation without
    # opening a repository or touching product state.
    LocalAssetResolver(_NoopAssetLookup(), bindings=bindings)
    return bindings


class _NoopAssetLookup:
    def find_assets_by_sha256(self, _digest: str) -> tuple[object, ...]:
        return ()


def _source_repository_root() -> Path:
    root = Path(__file__).resolve().parents[4]
    if not (root / ".git").exists():
        raise _CodeIdentityError("Git source identity is unavailable")
    return root


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(root), *arguments],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise _CodeIdentityError("Git source identity is unavailable") from error
    if completed.returncode != 0:
        raise _CodeIdentityError("Git source identity is unavailable")
    return completed.stdout


def _current_code_sha() -> str:
    root = _source_repository_root()
    head = _git_bytes(root, "rev-parse", "--verify", "HEAD^{commit}").strip().decode(
        "ascii",
        errors="strict",
    )
    if not _GIT_SHA_RE.fullmatch(head):
        raise _CodeIdentityError("Git HEAD identity is invalid")
    status = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    for record in status.split(b"\0"):
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise _CodeIdentityError("Git worktree status is invalid")
        state = record[:2]
        if state != b"??":
            raise _CodeIdentityError("tracked Git worktree changes are not allowed")
        raise _CodeIdentityError("untracked Git worktree files are not allowed")
    return head


def _current_hardware_profile() -> HardwareProfile:
    system = platform.system().strip()
    release = platform.release().strip()
    architecture = platform.machine().strip()
    processor = platform.processor().strip() or architecture
    if not system or not release or not architecture or not processor:
        raise BenchmarkExecutionError("current hardware identity is unavailable")
    try:
        memory_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(
            os.sysconf("SC_PAGE_SIZE")
        )
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise BenchmarkExecutionError(
            "current hardware memory identity is unavailable"
        ) from error
    if memory_bytes <= 0:
        raise BenchmarkExecutionError(
            "current hardware memory identity is unavailable"
        )
    return HardwareProfile(
        operating_system=f"{system} {release}",
        architecture=architecture,
        processor=processor,
        memory_bytes=memory_bytes,
        accelerator="Metal" if system == "Darwin" else None,
    )


def _existing_registry(root: Path) -> BenchmarkRunRegistry:
    root = Path(root)
    if root.is_symlink():
        raise BenchmarkDataError("benchmark registry root must not be a symbolic link")
    if not root.exists():
        raise FileNotFoundError("benchmark registry is missing")
    if not root.is_dir():
        raise BenchmarkDataError("benchmark registry root must be a directory")
    registry = BenchmarkRunRegistry(root)
    if not (registry.lock_path.exists() or registry.lock_path.is_symlink()):
        raise FileNotFoundError("benchmark registry lock is missing")
    return registry


def _dataset_summary(dataset: BenchmarkDataset) -> dict[str, object]:
    positive_count = sum(bool(case.relevant_intervals) for case in dataset.cases)
    return {
        "schema_version": dataset.schema_version,
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "revision": dataset_revision(dataset),
        "asset_count": len(dataset.assets),
        "case_count": len(dataset.cases),
        "positive_case_count": positive_count,
        "negative_case_count": len(dataset.cases) - positive_count,
        "relevant_interval_count": sum(
            len(case.relevant_intervals) for case in dataset.cases
        ),
        "hard_negative_count": sum(
            len(case.hard_negatives) for case in dataset.cases
        ),
    }


def _registry_entry(entry: RunRegistryEntry) -> dict[str, object]:
    return {
        "run_id": entry.run_id,
        "created_at": entry.created_at,
        "code_sha": entry.code_sha,
        "dataset_revision": entry.dataset_revision,
        "execution_mode": entry.execution_mode,
        "manifest_sha256": entry.manifest_sha256,
        "manifest_path": entry.manifest_path,
    }


def _write_json(stream, value: object) -> None:  # type: ignore[no-untyped-def]
    stream.write(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    stream.write("\n")


def _fail(exit_code: int, code: str, message: str) -> int:
    _write_json(
        sys.stderr,
        {"error": {"code": code, "message": message}},
    )
    return exit_code
