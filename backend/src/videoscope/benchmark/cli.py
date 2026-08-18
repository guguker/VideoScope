from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .catalog import DatasetCatalog
from .comparison import compare_runs, comparison_json
from .policy import load_policy
from .runner import BenchmarkExecutionError, audit_run_manifest
from .schema import BenchmarkDataError, BenchmarkDataset
from .serialization import dataset_revision, run_to_dict
from .storage import (
    BenchmarkDurabilityError,
    BenchmarkRunRegistry,
    RunRegistryEntry,
    load_dataset,
)


EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_DATA = 3
EXIT_NOT_FOUND = 4
EXIT_CONFLICT = 5
EXIT_AUDIT = 6
EXIT_IO = 7
EXIT_INTERNAL = 70


class _CliUsageError(ValueError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise _CliUsageError("invalid command arguments")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        value = _dispatch(arguments)
        if isinstance(value, str):
            sys.stdout.write(value)
            sys.stdout.write("\n")
        else:
            _write_json(sys.stdout, value)
        return EXIT_SUCCESS
    except _CliUsageError:
        return _fail(
            EXIT_USAGE,
            "usage_error",
            "invalid benchmark command arguments",
        )
    except BenchmarkExecutionError:
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
        return comparison_json(compare_runs(baseline, candidate, policy))
    raise _CliUsageError("unsupported command")


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
