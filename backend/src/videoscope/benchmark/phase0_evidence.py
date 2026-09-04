"""Fail-closed assembly of the portable Phase-0 baseline evidence bundle.

The collector never runs a model and never copies source media, queries, labels,
paths, tokens, or raw model output into the committed report.  It validates
already-produced evidence, extracts only bounded portable measurements and
stable identities, and refuses to publish a partial bundle.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Final

from videoscope.ml_environment_attestation import (
    MlEnvironmentManifestError,
    _contract_observations,
    _expected_distribution_identity,
    load_ml_environment_manifest,
)
from videoscope.providers.qwen_video import (
    QWEN_INFERENCE_RUNTIME_IDENTITY,
    QWEN_PROMPT_PROTOCOL_SHA256,
)
from videoscope.providers.qwen_worker import (
    QWEN_SOURCE_BUNDLE_SHA256,
    QWEN_WORKER_SCHEMA_VERSION,
)

from .metric_policy import (
    FrozenMetricPolicy,
    frozen_metric_policy_from_dict,
    frozen_metric_policy_revision,
    validate_frozen_metric_policy_dataset,
    validate_frozen_metric_policy_product_dataset,
)
from .host_resources import HOST_RESOURCE_IDENTITY
from .profiles import get_profile
from .runner import (
    BenchmarkExecutionError,
    _validate_persisted_profile_identity_contract,
    audit_run_manifest,
)
from .schema import (
    BenchmarkDataError,
    BenchmarkDataset,
    BenchmarkRunManifest,
    _CODE_SHA_RE,
    _require_id,
)
from .serialization import (
    JsonObject,
    canonical_json_bytes,
    dataset_from_dict,
    dataset_revision,
    expect_fields,
    expect_list,
    expect_object,
    parse_json_object,
    run_from_dict,
    run_to_dict,
)
from .storage import (
    MAX_DATASET_MANIFEST_BYTES,
    MAX_RUN_MANIFEST_BYTES,
    _read_bounded_file,
)
from .video_verifier_runner import (
    MAX_VIDEO_VERIFIER_RUN_BYTES,
    VideoVerifierRunManifest,
    _run_from_dict as video_verifier_run_from_dict,
)
from .video_verifier_schema import (
    MAX_VIDEO_VERIFIER_MANIFEST_BYTES,
    VideoVerifierDataset,
    video_verifier_dataset_from_json,
    video_verifier_dataset_revision,
)


PHASE0_EVIDENCE_SCHEMA_VERSION: Final = 1
PHASE0_BASELINE_SNAPSHOT_SCHEMA_VERSION: Final = 2
PHASE0_RAW_MEASUREMENTS_SCHEMA_VERSION: Final = 1
PHASE0_SANITIZED_REPORT_SCHEMA_VERSION: Final = 2
PHASE0_ERROR_LEDGER_SCHEMA_VERSION: Final = 1
PHASE0_ROLLBACK_PROOF_SCHEMA_VERSION: Final = 1
PHASE0_BASELINE_BATCH_SCHEMA_VERSION: Final = 1
FULL_ML_SMOKE_SCHEMA_VERSION: Final = 2
COMPONENT_EXECUTION_SCHEMA_VERSION: Final = 1

MAX_PHASE0_EVIDENCE_BYTES: Final = 64 * 1024**2
MAX_PHASE0_OUTPUT_BYTES: Final = 64 * 1024**2
PHASE0_MEMORY_LIMIT_BYTES: Final = 16 * 1024**3
ML_ENVIRONMENT_MANIFEST_SHA256: Final = (
    "ec4d3feb390d6c6bdffffe747491ad0f11ad83b954cc58cc9cb74698be3992ae"
)

REQUIRED_EXECUTED_PROFILE_IDS: Final = (
    "lexical_qdrant",
    "dense_siglip",
    "temporal_refinement",
    "lighthouse",
    "qwen_verification",
)
ALL_PHASE0_PROFILE_IDS: Final = (*REQUIRED_EXECUTED_PROFILE_IDS, "internvideo")
COMPONENT_EXECUTION_IDS: Final = (
    "text_vectors",
    "lexical_text",
    "visual_dense",
    "temporal_refinement",
    "lighthouse",
    "qwen_verification",
)
REQUIRED_ENVIRONMENT_IDS: Final = frozenset(
    {"base", "vision", "whisper", "ocr", "lighthouse", "qwen"}
)
REQUIRED_CAPABILITY_IDS: Final = frozenset(
    {
        "base_text_search",
        "dense_siglip",
        "object_detection",
        "speech_transcription",
        "ocr_text",
        "temporal_refinement",
        "qwen_verification",
    }
)
REQUIRED_MANAGED_WORKER_ROLES: Final = (
    "vision",
    "whisper",
    "lighthouse",
    "qwen",
)
BASELINE_BATCH_ENVIRONMENT_BINDINGS: Final = {
    "owner": "base",
    "vision_index": "vision",
    "vision": "vision",
    "whisper": "whisper",
    "ocr": "ocr",
    "lighthouse": "lighthouse",
    "qwen": "qwen",
}
FULL_ML_SMOKE_ENVIRONMENT_BINDINGS: Final = {
    key: value
    for key, value in BASELINE_BATCH_ENVIRONMENT_BINDINGS.items()
    if key != "vision_index"
}
ROLLBACK_TEST_NODE_ID: Final = (
    "backend/tests/test_video_index_jobs_repository.py::"
    "test_failed_full_job_publish_keeps_prior_release_searchable_after_restart"
)
ROLLBACK_TEST_SOURCE: Final = Path(
    "backend/tests/test_video_index_jobs_repository.py"
)
ROLLBACK_PROOF_ID: Final = "phase0-persisted-generation-rollback-v1"
ROLLBACK_FORCED_FAILURE: Final = "injected_pre_commit_publication_failure"
ROLLBACK_SEMANTIC_ASSERTIONS: Final = (
    "candidate_release_not_activated_after_forced_failure",
    "prior_release_active_after_process_restart",
    "prior_release_searchable_after_process_restart",
    "candidate_generation_addressable_but_inactive",
    "source_asset_identity_unchanged",
    "source_reindex_not_required",
)

# These identities describe product state shared by the progressive profile
# ladder.  The benchmark environment itself is deliberately absent: its v2
# digest binds ``profile_id`` and ``profile_identity`` and therefore must vary
# between profiles.
_SHARED_MODEL_COMPONENT_IDS: Final = frozenset(
    {"text_embedding", "visual_embedding", "lighthouse_model"}
)
_SHARED_INDEX_COMPONENT_IDS: Final = frozenset(
    {
        "text_vector_index",
        "text_vector_generations",
        "visual_generations",
        "lighthouse_generations",
    }
)
_SHARED_CONFIG_COMPONENT_IDS: Final = frozenset(
    {
        "benchmark_execution_lifecycle",
        "benchmark_methodology",
        "product_search_lifecycle",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
class Phase0EvidenceError(BenchmarkDataError):
    """A sanitized Phase-0 evidence failure safe for a CLI response."""

    def __init__(self, code: str) -> None:
        if not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,127}", code):
            raise ValueError("invalid Phase-0 evidence error code")
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class LoadedEvidence:
    payload: JsonObject
    sha256: str


@dataclass(frozen=True, slots=True)
class Phase0EvidenceArtifacts:
    baseline_snapshot: JsonObject
    raw_measurements: JsonObject
    sanitized_report: JsonObject
    error_ledger: JsonObject

    def files(self) -> tuple[tuple[str, JsonObject], ...]:
        return (
            ("baseline-snapshot.json", self.baseline_snapshot),
            ("raw-measurements.json", self.raw_measurements),
            ("sanitized-report.json", self.sanitized_report),
            ("error-ledger.json", self.error_ledger),
        )


def _fail(code: str) -> None:
    raise Phase0EvidenceError(code)


def _is_digest(value: object, *, prefixed: bool = False) -> bool:
    if prefixed:
        return isinstance(value, str) and value.startswith("sha256:") and bool(
            _SHA256_RE.fullmatch(value.removeprefix("sha256:"))
        )
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _integer(
    value: object,
    *,
    minimum: int = 0,
    maximum: int = (1 << 63) - 1,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("invalid bounded integer")
    return value


def _number(value: object, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError("invalid finite number")
    return result


def _identity(value: object) -> str:
    lowered = value.casefold() if isinstance(value, str) else ""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2_048
        or value.strip() != value
        or "\\" in value
        or value.startswith(("/", "~"))
        or re.match(r"^[A-Za-z]:/", value) is not None
        or "://" in value
        or any(
            marker in lowered
            for marker in (
                "authorization:",
                "api-key=",
                "api_key=",
                "bearer ",
                "secret=",
                "token=",
            )
        )
        or any(
            marker in value
            for marker in ("/Users/", "/home/", "/private/", "/tmp/", "/var/folders/")
        )
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("invalid portable identity")
    return value


def _exact_id_set(values: object, *, expected: frozenset[str]) -> dict[str, JsonObject]:
    items = expect_list(values, "Phase-0 identity list")
    result: dict[str, JsonObject] = {}
    for item in items:
        value = expect_object(item, "Phase-0 identity item")
        identifier = _require_id(value.get("id"), "Phase-0 identity id")
        if identifier in result:
            raise ValueError("duplicate Phase-0 identity")
        result[identifier] = value
    if set(result) != expected:
        raise ValueError("Phase-0 identity set mismatch")
    return result


def _current_ml_environment_contract() -> JsonObject:
    project_root = Path(__file__).resolve().parents[4]
    manifest_path = project_root / "workers/ml-environment.lock.json"
    manifest = load_ml_environment_manifest(
        manifest_path,
        expected_sha256=ML_ENVIRONMENT_MANIFEST_SHA256,
    )
    if manifest.raw_sha256 != ML_ENVIRONMENT_MANIFEST_SHA256:
        raise MlEnvironmentManifestError("manifest_identity_mismatch")
    failures: list[dict[str, str]] = []
    contract_reports, observations = _contract_observations(
        project_root,
        manifest,
        failures,
    )
    if failures:
        raise MlEnvironmentManifestError("contract_identity_mismatch")
    contracts_by_id = {
        contract.contract_id: contract for contract in manifest.contracts
    }
    environments: list[JsonObject] = []
    for environment in sorted(
        manifest.environments,
        key=lambda item: item.environment_id,
    ):
        dependency = contracts_by_id[environment.dependency_contract_id]
        environments.append(
            {
                "dependency_identity": "sha256:" + dependency.sha256,
                "distribution_identity": "sha256:"
                + _expected_distribution_identity(
                    environment.distribution_source,
                    observations,
                ),
                "id": environment.environment_id,
                "python": environment.python,
                "status": "complete",
            }
        )
    capabilities = [
        {
            "id": capability.capability_id,
            "model_identities": sorted(capability.model_identities),
            "runtime_identity": capability.runtime_identity,
            "status": "complete",
        }
        for capability in sorted(
            manifest.capabilities,
            key=lambda item: item.capability_id,
        )
    ]
    return {
        "attestation_id": manifest.attestation_id,
        "capabilities": capabilities,
        "contracts": contract_reports,
        "environments": environments,
        "host": {
            "chip": manifest.host.chip,
            "machine": manifest.host.machine,
            "memory_gib": manifest.host.memory_gib,
            "minimum_macos_major": manifest.host.minimum_macos_major,
            "system": manifest.host.system,
        },
        "manifest_identity": "sha256:" + manifest.raw_sha256,
        "offline_flags": sorted(manifest.offline_environment),
        "uv": {
            "identity": "sha256:" + manifest.uv.sha256,
            "status": "complete",
            "version": manifest.uv.version,
        },
    }


def validate_ml_environment_attestation(value: object) -> JsonObject:
    """Validate and sanitize a complete target-host ML attestation report."""

    try:
        expected_contract = _current_ml_environment_contract()
        report = expect_object(value, "Phase-0 ML environment attestation")
        expect_fields(
            report,
            {
                "attestation_id",
                "capabilities",
                "contracts",
                "environments",
                "failures",
                "host",
                "manifest_identity",
                "offline_environment",
                "schema_version",
                "status",
                "uv",
            },
            "Phase-0 ML environment attestation",
        )
        if (
            type(report["schema_version"]) is not int
            or report["schema_version"] != 1
            or report["status"] != "complete"
        ):
            raise ValueError("incomplete ML environment attestation")
        attestation_id = _require_id(
            report["attestation_id"],
            "environment attestation id",
        )
        manifest_identity = report["manifest_identity"]
        if (
            not _is_digest(manifest_identity, prefixed=True)
            or attestation_id != expected_contract["attestation_id"]
            or manifest_identity != expected_contract["manifest_identity"]
        ):
            raise ValueError("invalid environment manifest identity")
        if expect_list(report["failures"], "environment failures"):
            raise ValueError("environment attestation contains failures")

        host = expect_object(report["host"], "environment host")
        expect_fields(
            host,
            {"chip", "machine", "macos_major", "memory_gib", "status", "system"},
            "environment host",
        )
        expected_host = expect_object(
            expected_contract["host"],
            "expected environment host",
        )
        if (
            host["status"] != "complete"
            or host["system"] != expected_host["system"]
            or host["machine"] != expected_host["machine"]
            or host["chip"] != expected_host["chip"]
            or host["memory_gib"] != expected_host["memory_gib"]
            or _integer(host["macos_major"], minimum=1, maximum=99)
            < expected_host["minimum_macos_major"]
        ):
            raise ValueError("target host mismatch")

        offline = expect_object(
            report["offline_environment"],
            "environment offline policy",
        )
        expect_fields(offline, {"flags", "status"}, "environment offline policy")
        required_offline_flags = set(expected_contract["offline_flags"])
        if offline["status"] != "complete" or set(
            expect_list(offline["flags"], "offline flags")
        ) != required_offline_flags:
            raise ValueError("offline policy mismatch")

        environments = _exact_id_set(
            report["environments"],
            expected=REQUIRED_ENVIRONMENT_IDS,
        )
        expected_environments = {
            item["id"]: item
            for item in expect_list(
                expected_contract["environments"],
                "expected environments",
            )
        }
        environment_summaries: list[JsonObject] = []
        for identifier, environment in sorted(environments.items()):
            expect_fields(
                environment,
                {
                    "dependency_identity",
                    "distribution_identity",
                    "id",
                    "python",
                    "status",
                },
                f"environment {identifier}",
            )
            expected_environment = expected_environments[identifier]
            if (
                environment != expected_environment
                or not _is_digest(
                    environment["dependency_identity"],
                    prefixed=True,
                )
                or not _is_digest(
                    environment["distribution_identity"],
                    prefixed=True,
                )
            ):
                raise ValueError("environment identity mismatch")
            environment_summaries.append(
                {
                    "dependency_identity": environment["dependency_identity"],
                    "distribution_identity": environment["distribution_identity"],
                    "id": identifier,
                    "python": environment["python"],
                    "status": "complete",
                }
            )

        capabilities = _exact_id_set(
            report["capabilities"],
            expected=REQUIRED_CAPABILITY_IDS,
        )
        expected_capabilities = {
            item["id"]: item
            for item in expect_list(
                expected_contract["capabilities"],
                "expected capabilities",
            )
        }
        capability_summaries: list[JsonObject] = []
        for identifier, capability in sorted(capabilities.items()):
            expect_fields(
                capability,
                {"id", "model_identities", "runtime_identity", "status"},
                f"capability {identifier}",
            )
            models = expect_list(
                capability["model_identities"],
                f"capability {identifier} models",
            )
            if capability["status"] != "complete" or not models:
                raise ValueError("capability is incomplete")
            normalized_models = sorted(_identity(item) for item in models)
            runtime_identity = _identity(capability["runtime_identity"])
            if (
                len(normalized_models) != len(set(normalized_models))
                or normalized_models
                != expected_capabilities[identifier]["model_identities"]
                or runtime_identity
                != expected_capabilities[identifier]["runtime_identity"]
            ):
                raise ValueError("duplicate capability model identity")
            capability_summaries.append(
                {
                    "id": identifier,
                    "model_identities": normalized_models,
                    "runtime_identity": runtime_identity,
                    "status": "complete",
                }
            )

        contracts = expect_list(report["contracts"], "environment contracts")
        if not contracts:
            raise ValueError("environment contracts are missing")
        contract_summaries: list[JsonObject] = []
        observed_contract_ids: set[str] = set()
        for raw_contract in contracts:
            contract = expect_object(raw_contract, "environment contract")
            expected = {"id", "identity", "kind", "status"}
            optional = {"canonical_identity"}
            if set(contract) - expected - optional or expected - set(contract):
                raise ValueError("environment contract fields are invalid")
            identifier = _require_id(contract["id"], "environment contract id")
            if identifier in observed_contract_ids:
                raise ValueError("duplicate environment contract")
            observed_contract_ids.add(identifier)
            if contract["status"] != "complete" or not _is_digest(
                contract["identity"],
                prefixed=True,
            ):
                raise ValueError("environment contract is incomplete")
            summary: JsonObject = {
                "id": identifier,
                "identity": contract["identity"],
                "kind": _require_id(contract["kind"], "environment contract kind"),
                "status": "complete",
            }
            if "canonical_identity" in contract:
                if not _is_digest(contract["canonical_identity"], prefixed=True):
                    raise ValueError("invalid canonical contract identity")
                summary["canonical_identity"] = contract["canonical_identity"]
            contract_summaries.append(summary)
        if sorted(contract_summaries, key=lambda item: str(item["id"])) != sorted(
            expect_list(expected_contract["contracts"], "expected contracts"),
            key=lambda item: str(item["id"]),  # type: ignore[index]
        ):
            raise ValueError("environment contract identity mismatch")

        uv = expect_object(report["uv"], "environment uv")
        expect_fields(uv, {"identity", "status", "version"}, "environment uv")
        if (
            uv != expected_contract["uv"]
            or not _is_digest(uv["identity"], prefixed=True)
        ):
            raise ValueError("uv identity mismatch")
    except (
        BenchmarkDataError,
        KeyError,
        MlEnvironmentManifestError,
        OSError,
        TypeError,
        ValueError,
    ):
        _fail("environment_attestation_invalid")

    return {
        "attestation_id": attestation_id,
        "capabilities": capability_summaries,
        "contracts": sorted(contract_summaries, key=lambda item: str(item["id"])),
        "environments": environment_summaries,
        "host": {
            "chip": host["chip"],
            "machine": host["machine"],
            "macos_major": host["macos_major"],
            "memory_gib": host["memory_gib"],
            "status": "complete",
            "system": host["system"],
        },
        "manifest_identity": manifest_identity,
        "offline": True,
        "status": "complete",
        "uv": {
            "identity": uv["identity"],
            "status": "complete",
            "version": uv["version"],
        },
    }


def _validate_host_resources(value: object) -> JsonObject:
    host = expect_object(value, "full-ML host resources")
    expect_fields(
        host,
        {
            "schema_version",
            "provider_identity",
            "scope",
            "memory_accounting",
            "sample_interval_milliseconds",
            "raw_samples",
            "metal",
            "virtual_memory",
        },
        "full-ML host resources",
    )
    if (
        type(host["schema_version"]) is not int
        or host["schema_version"] != 1
        or host["provider_identity"] != HOST_RESOURCE_IDENTITY
        or host["scope"] != "system_wide"
        or host["memory_accounting"]
        != "metal_standalone_not_additive_with_process_rss"
    ):
        raise ValueError("host resource identity mismatch")
    provider_identity = _identity(host["provider_identity"])
    interval = _integer(
        host["sample_interval_milliseconds"],
        minimum=1,
        maximum=60_000,
    )
    raw_samples = expect_list(host["raw_samples"], "host resource raw samples")
    if len(raw_samples) < 2 or len(raw_samples) > 100_000:
        raise ValueError("host resource raw samples are missing or unbounded")
    normalized_samples: list[JsonObject] = []
    elapsed: list[int] = []
    in_use: list[int] = []
    allocated: list[int] = []
    recoveries: list[int] = []
    swapins: list[int] = []
    swapouts: list[int] = []
    page_sizes: list[int] = []
    for raw_sample in raw_samples:
        sample = expect_object(raw_sample, "host resource raw sample")
        expect_fields(
            sample,
            {"elapsed_nanoseconds", "metal", "virtual_memory"},
            "host resource raw sample",
        )
        metal = expect_object(sample["metal"], "host resource raw sample Metal")
        expect_fields(
            metal,
            {
                "in_use_system_memory_bytes",
                "alloc_system_memory_bytes",
                "recovery_count",
            },
            "host resource raw sample Metal",
        )
        vm = expect_object(
            sample["virtual_memory"],
            "host resource raw sample virtual memory",
        )
        expect_fields(
            vm,
            {"swapins_pages", "swapouts_pages", "page_size_bytes"},
            "host resource raw sample virtual memory",
        )
        elapsed.append(_integer(sample["elapsed_nanoseconds"]))
        in_use.append(_integer(metal["in_use_system_memory_bytes"]))
        allocated.append(_integer(metal["alloc_system_memory_bytes"]))
        recoveries.append(_integer(metal["recovery_count"]))
        swapins.append(_integer(vm["swapins_pages"]))
        swapouts.append(_integer(vm["swapouts_pages"]))
        page_size = _integer(vm["page_size_bytes"], minimum=4_096, maximum=1 << 30)
        if page_size & (page_size - 1):
            raise ValueError("host page size is not a power of two")
        page_sizes.append(page_size)
        normalized_samples.append(
            {
                "elapsed_nanoseconds": elapsed[-1],
                "metal": {
                    "alloc_system_memory_bytes": allocated[-1],
                    "in_use_system_memory_bytes": in_use[-1],
                    "recovery_count": recoveries[-1],
                },
                "virtual_memory": {
                    "page_size_bytes": page_size,
                    "swapins_pages": swapins[-1],
                    "swapouts_pages": swapouts[-1],
                },
            }
        )
    if (
        elapsed[0] != 0
        or any(current <= previous for previous, current in zip(elapsed, elapsed[1:]))
        or any(current < previous for previous, current in zip(recoveries, recoveries[1:]))
        or any(current < previous for previous, current in zip(swapins, swapins[1:]))
        or any(current < previous for previous, current in zip(swapouts, swapouts[1:]))
        or len(set(page_sizes)) != 1
    ):
        raise ValueError("host resource timeline is invalid")
    page_size = page_sizes[0]
    recovery_delta = recoveries[-1] - recoveries[0]
    swapins_delta_pages = swapins[-1] - swapins[0]
    swapouts_delta_pages = swapouts[-1] - swapouts[0]
    maximum = (1 << 63) - 1
    if (
        swapins_delta_pages > maximum // page_size
        or swapouts_delta_pages > maximum // page_size
    ):
        raise ValueError("host resource derived delta is unbounded")
    expected = {
        "metal_in_use": (in_use[0], max(in_use), max(in_use) - in_use[0]),
        "metal_alloc": (
            allocated[0],
            max(allocated),
            max(allocated) - allocated[0],
        ),
        "recovery": recovery_delta,
        "swapins_pages": swapins_delta_pages,
        "swapouts_pages": swapouts_delta_pages,
    }
    metal_summary = expect_object(host["metal"], "host resource Metal summary")
    expect_fields(
        metal_summary,
        {"in_use_system_memory", "alloc_system_memory", "recovery_delta"},
        "host resource Metal summary",
    )
    for key, source_key in (
        ("in_use_system_memory", "metal_in_use"),
        ("alloc_system_memory", "metal_alloc"),
    ):
        summary = expect_object(metal_summary[key], f"host resource {key}")
        expect_fields(
            summary,
            {"baseline_bytes", "peak_bytes", "increment_bytes"},
            f"host resource {key}",
        )
        observed_summary = (
            _integer(summary["baseline_bytes"]),
            _integer(summary["peak_bytes"]),
            _integer(summary["increment_bytes"]),
        )
        if observed_summary != expected[source_key]:
            raise ValueError("host Metal summary does not match raw samples")
    recovery_summary = _integer(metal_summary["recovery_delta"])
    if recovery_summary != expected["recovery"]:
        raise ValueError("host Metal recovery does not match raw samples")
    vm_summary = expect_object(host["virtual_memory"], "host virtual memory summary")
    expect_fields(
        vm_summary,
        {
            "swapins_delta_pages",
            "swapins_delta_bytes",
            "swapouts_delta_pages",
            "swapouts_delta_bytes",
            "page_size_bytes",
        },
        "host virtual memory summary",
    )
    normalized_vm_summary = {
        "page_size_bytes": _integer(
            vm_summary["page_size_bytes"], minimum=4_096, maximum=1 << 30
        ),
        "swapins_delta_pages": _integer(vm_summary["swapins_delta_pages"]),
        "swapins_delta_bytes": _integer(vm_summary["swapins_delta_bytes"]),
        "swapouts_delta_pages": _integer(vm_summary["swapouts_delta_pages"]),
        "swapouts_delta_bytes": _integer(vm_summary["swapouts_delta_bytes"]),
    }
    if (
        normalized_vm_summary["page_size_bytes"] != page_size
        or normalized_vm_summary["swapins_delta_pages"] != expected["swapins_pages"]
        or normalized_vm_summary["swapouts_delta_pages"] != expected["swapouts_pages"]
        or normalized_vm_summary["swapins_delta_bytes"]
        != expected["swapins_pages"] * page_size
        or normalized_vm_summary["swapouts_delta_bytes"]
        != expected["swapouts_pages"] * page_size
    ):
        raise ValueError("host virtual memory summary does not match raw samples")
    normalized_metal_summary: JsonObject = {
        "alloc_system_memory": {
            "baseline_bytes": expected["metal_alloc"][0],  # type: ignore[index]
            "increment_bytes": expected["metal_alloc"][2],  # type: ignore[index]
            "peak_bytes": expected["metal_alloc"][1],  # type: ignore[index]
        },
        "in_use_system_memory": {
            "baseline_bytes": expected["metal_in_use"][0],  # type: ignore[index]
            "increment_bytes": expected["metal_in_use"][2],  # type: ignore[index]
            "peak_bytes": expected["metal_in_use"][1],  # type: ignore[index]
        },
        "recovery_delta": recovery_delta,
    }
    return {
        "memory_accounting": host["memory_accounting"],
        "metal": normalized_metal_summary,
        "provider_identity": provider_identity,
        "raw_samples": normalized_samples,
        "sample_interval_milliseconds": interval,
        "schema_version": 1,
        "scope": "system_wide",
        "virtual_memory": normalized_vm_summary,
    }


def _validate_smoke_steps(value: object) -> None:
    specifications = (
        ("vision.image_embedding", {"dimensions": (1, 1_000_000), "vector_count": (1, 1)}),
        ("vision.text_embedding", {"dimensions": (1, 1_000_000), "vector_count": (1, 1)}),
        ("vision.rfdetr", {"detection_count": (0, 1_000_000)}),
        ("whisper.transcribe", {"segment_count": (0, 1_000_000)}),
        ("ocr.read", {"item_count": (0, 1_000_000)}),
        ("lighthouse.generation", {"generation_count": (1, 1)}),
        ("lighthouse.search", {"hit_count": (1, 1_000_000)}),
        ("qwen.judge", {"judgement_count": (1, 1)}),
    )
    steps = expect_list(value, "full-ML smoke steps")
    if len(steps) != len(specifications):
        raise ValueError("full-ML smoke step set mismatch")
    observed_dimensions: list[int] = []
    for raw_step, (expected_id, fields) in zip(steps, specifications):
        step = expect_object(raw_step, f"full-ML smoke step {expected_id}")
        expect_fields(step, {"id", "observations", "status"}, f"smoke step {expected_id}")
        if step["id"] != expected_id or step["status"] != "complete":
            raise ValueError("full-ML smoke step mismatch")
        observations = expect_object(
            step["observations"],
            f"full-ML smoke observations {expected_id}",
        )
        expect_fields(observations, set(fields), f"smoke observations {expected_id}")
        for field, (minimum, maximum) in fields.items():
            observed = _integer(
                observations[field],
                minimum=minimum,
                maximum=maximum,
            )
            if field == "dimensions":
                observed_dimensions.append(observed)
    if len(observed_dimensions) != 2 or len(set(observed_dimensions)) != 1:
        raise ValueError("vision embedding dimensions disagree")


def _selected_component_execution_ids(profile_id: str) -> tuple[str, ...]:
    profile = get_profile(profile_id)
    plan = profile.search_plan
    selected: list[str] = []
    if plan.text_search != "disabled":
        selected.extend(("text_vectors", "lexical_text"))
    if plan.visual_search != "disabled":
        selected.append("visual_dense")
    if plan.temporal_refinement:
        selected.append("temporal_refinement")
    if plan.lighthouse:
        selected.append("lighthouse")
    if plan.reranker == "qwen":
        selected.append("qwen_verification")
    if any(component_id not in COMPONENT_EXECUTION_IDS for component_id in selected):
        raise ValueError("frozen profile selects an unsupported execution component")
    return tuple(selected)


def _search_configuration_identity(profile_id: str) -> str:
    plan = get_profile(profile_id).search_plan
    digest = sha256(plan.canonical_json.encode("utf-8")).hexdigest()
    return f"evaluation-search-configuration@{plan.schema_version}:{digest}"


def _component_execution_counts(value: object, *, context: str) -> dict[str, int]:
    counts = expect_object(value, context)
    expect_fields(counts, set(COMPONENT_EXECUTION_IDS), context)
    return {
        component_id: _integer(counts[component_id])
        for component_id in COMPONENT_EXECUTION_IDS
    }


def _validate_component_execution(profile_id: str, value: object) -> JsonObject:
    trace = expect_object(value, f"profile {profile_id} component execution")
    expect_fields(
        trace,
        {
            "schema_version",
            "profile_identity",
            "search_configuration_identity",
            "invoked_component_ids",
            "component_input_counts",
            "component_output_counts",
            "component_evidence_counts",
        },
        f"profile {profile_id} component execution",
    )
    profile = get_profile(profile_id)
    if (
        type(trace["schema_version"]) is not int
        or trace["schema_version"] != COMPONENT_EXECUTION_SCHEMA_VERSION
        or _identity(trace["profile_identity"]) != profile.identity
        or _identity(trace["search_configuration_identity"])
        != _search_configuration_identity(profile_id)
    ):
        raise ValueError("component execution identity mismatch")
    invoked = tuple(
        _require_id(item, "component execution component id")
        for item in expect_list(
            trace["invoked_component_ids"],
            "component execution invoked components",
        )
    )
    expected_invoked = _selected_component_execution_ids(profile_id)
    if invoked != expected_invoked:
        raise ValueError("component execution invocation set mismatch")
    normalized_counts = {
        field: _component_execution_counts(
            trace[field],
            context=f"profile {profile_id} {field}",
        )
        for field in (
            "component_input_counts",
            "component_output_counts",
            "component_evidence_counts",
        )
    }
    selected = set(expected_invoked)
    for component_id in COMPONENT_EXECUTION_IDS:
        observed = tuple(
            counts[component_id] for counts in normalized_counts.values()
        )
        if component_id in selected:
            if any(count <= 0 for count in observed):
                raise ValueError("selected component execution is not substantive")
        elif any(count != 0 for count in observed):
            raise ValueError("unselected component execution leaked into profile")
    return {
        "component_evidence_counts": normalized_counts[
            "component_evidence_counts"
        ],
        "component_input_counts": normalized_counts["component_input_counts"],
        "component_output_counts": normalized_counts["component_output_counts"],
        "invoked_component_ids": list(expected_invoked),
        "profile_identity": profile.identity,
        "schema_version": COMPONENT_EXECUTION_SCHEMA_VERSION,
        "search_configuration_identity": _search_configuration_identity(profile_id),
    }


def validate_full_ml_smoke(
    value: object,
    *,
    code_sha: str,
    memory_limit_bytes: int,
) -> JsonObject:
    """Validate full product smoke and return its path-free evidence subset."""

    try:
        memory_limit = _integer(memory_limit_bytes, minimum=1)
        report = expect_object(value, "full-ML smoke report")
        if not isinstance(code_sha, str) or _CODE_SHA_RE.fullmatch(code_sha) is None:
            raise ValueError("invalid expected code SHA")
        expected_environment = _current_ml_environment_contract()
        expect_fields(
            report,
            {
                "code_sha_after",
                "code_sha_before",
                "environment_bindings",
                "ml_environment_attestation_id",
                "ml_environment_manifest_identity",
                "offline",
                "product_integration",
                "resources",
                "schema_version",
                "status",
                "steps",
                "toolchain_identity",
                "workspace_cleanup",
            },
            "full-ML smoke report",
        )
        if (
            type(report["schema_version"]) is not int
            or report["schema_version"] != FULL_ML_SMOKE_SCHEMA_VERSION
            or report["code_sha_before"] != code_sha
            or report["code_sha_after"] != code_sha
            or report["status"] != "ready"
            or report["offline"] is not True
            or report["workspace_cleanup"] != "complete"
            or not _is_digest(report["toolchain_identity"], prefixed=True)
            or report["ml_environment_attestation_id"]
            != expected_environment["attestation_id"]
            or report["ml_environment_manifest_identity"]
            != expected_environment["manifest_identity"]
            or report["environment_bindings"]
            != FULL_ML_SMOKE_ENVIRONMENT_BINDINGS
        ):
            raise ValueError("full-ML smoke is incomplete")
        _validate_smoke_steps(report["steps"])

        product = expect_object(report["product_integration"], "product receipt")
        expect_fields(
            product,
            {
                "status",
                "upload",
                "index",
                "search",
                "export",
                "generation_bound",
                "evidence_count",
                "profiles",
                "runtime_cleanup",
            },
            "product receipt",
        )
        if (
            product["status"] != "complete"
            or product["upload"] != "complete"
            or product["index"] != "complete"
            or product["search"] != "complete"
            or product["generation_bound"] is not True
            or product["runtime_cleanup"] != "complete"
        ):
            raise ValueError("product receipt is incomplete")
        profiles = expect_object(product["profiles"], "product profiles")
        if set(profiles) != set(ALL_PHASE0_PROFILE_IDS):
            raise ValueError("product profile set mismatch")
        normalized_profiles: JsonObject = {}
        total_evidence = 0
        for profile_id in REQUIRED_EXECUTED_PROFILE_IDS:
            profile = expect_object(profiles[profile_id], f"profile {profile_id}")
            expect_fields(
                profile,
                {
                    "close",
                    "component_execution",
                    "evidence_count",
                    "generation_bound",
                    "open",
                    "search",
                    "status",
                },
                f"profile {profile_id}",
            )
            component_execution = _validate_component_execution(
                profile_id,
                profile["component_execution"],
            )
            count = _integer(profile["evidence_count"], minimum=1)
            if (
                profile["status"] != "complete"
                or profile["open"] != "complete"
                or profile["search"] != "complete"
                or profile["close"] != "complete"
                or profile["generation_bound"] is not True
            ):
                raise ValueError("profile smoke is incomplete")
            total_evidence += count
            normalized_profiles[profile_id] = {
                "component_execution": component_execution,
                "evidence_count": count,
                "generation_bound": True,
                "status": "complete",
            }
        internvideo = expect_object(profiles["internvideo"], "InternVideo profile")
        expect_fields(
            internvideo,
            {"reason_code", "status"},
            "InternVideo profile",
        )
        if (
            internvideo["status"] != "not_configured"
            or internvideo["reason_code"] != "provider_not_configured"
        ):
            raise ValueError("InternVideo must be exactly not_configured")
        normalized_profiles["internvideo"] = {
            "reason_code": "provider_not_configured",
            "status": "not_configured",
        }
        if product["evidence_count"] != total_evidence:
            raise ValueError("product evidence count mismatch")

        export = expect_object(product["export"], "smoke export")
        expect_fields(
            export,
            {
                "byte_size",
                "container",
                "duration_seconds",
                "source_profile_id",
                "status",
            },
            "smoke export",
        )
        if (
            export["status"] != "complete"
            or export["container"] != "mp4"
            or export["source_profile_id"] != "lexical_qdrant"
            or _integer(export["byte_size"], minimum=1) < 1
            or _number(export["duration_seconds"], minimum=0.000_001) <= 0
        ):
            raise ValueError("smoke export is invalid")

        resources = expect_object(report["resources"], "full-ML smoke resources")
        expect_fields(
            resources,
            {
                "host_resources",
                "measurement_caveat",
                "oom",
                "peak_metal_bytes",
                "sampled_peak_process_tree_rss_bytes",
                "system_wide_pressure_deltas",
            },
            "full-ML smoke resources",
        )
        oom = expect_object(resources["oom"], "full-ML OOM status")
        expect_fields(oom, {"status"}, "full-ML OOM status")
        if oom["status"] != "not_observed":
            raise ValueError("Metal OOM was not disproven")
        process = expect_object(
            resources["sampled_peak_process_tree_rss_bytes"],
            "full-ML process-tree RSS",
        )
        expect_fields(
            process,
            {
                "baseline_bytes",
                "external_loopback_workers_included",
                "increment_bytes",
                "managed_worker_roles",
                "sample_count",
                "samples_bytes",
                "sampling_interval_ms",
                "scope",
                "status",
                "value",
            },
            "full-ML process-tree RSS",
        )
        samples = tuple(
            _integer(item)
            for item in expect_list(process["samples_bytes"], "process RSS samples")
        )
        if not samples or len(samples) > 250_000:
            raise ValueError("process RSS samples are missing or unbounded")
        roles = tuple(
            _require_id(item, "managed worker role")
            for item in expect_list(
                process["managed_worker_roles"],
                "managed worker roles",
            )
        )
        peak = max(samples)
        baseline = samples[0]
        if (
            process["status"] != "measured"
            or process["scope"] != "smoke_process_and_descendants"
            or process["external_loopback_workers_included"] is not True
            or roles != REQUIRED_MANAGED_WORKER_ROLES
            or process["sample_count"] != len(samples)
            or process["baseline_bytes"] != baseline
            or process["value"] != peak
            or process["increment_bytes"] != peak - baseline
            or peak > memory_limit
            or _integer(process["sampling_interval_ms"], minimum=1) != 50
        ):
            raise ValueError("process RSS measurement is invalid")

        host = _validate_host_resources(resources["host_resources"])
        caveat = expect_object(resources["measurement_caveat"], "measurement caveat")
        expect_fields(caveat, {"memory_accounting", "scope"}, "measurement caveat")
        if (
            caveat["memory_accounting"] != host["memory_accounting"]
            or caveat["scope"] != host["scope"]
        ):
            raise ValueError("measurement caveat mismatch")
        pressure = expect_object(
            resources["system_wide_pressure_deltas"],
            "system-wide pressure deltas",
        )
        expect_fields(
            pressure,
            {
                "metal_recovery_count",
                "swapins_bytes",
                "swapins_pages",
                "swapouts_bytes",
                "swapouts_pages",
            },
            "system-wide pressure deltas",
        )
        expected_pressure = {
            "metal_recovery_count": host["metal"]["recovery_delta"],  # type: ignore[index]
            "swapins_bytes": host["virtual_memory"]["swapins_delta_bytes"],  # type: ignore[index]
            "swapins_pages": host["virtual_memory"]["swapins_delta_pages"],  # type: ignore[index]
            "swapouts_bytes": host["virtual_memory"]["swapouts_delta_bytes"],  # type: ignore[index]
            "swapouts_pages": host["virtual_memory"]["swapouts_delta_pages"],  # type: ignore[index]
        }
        if pressure != expected_pressure or any(value != 0 for value in pressure.values()):
            raise ValueError("OOM recovery or sustained swap was observed")
        peak_metal = expect_object(resources["peak_metal_bytes"], "peak Metal memory")
        expect_fields(
            peak_metal,
            {"baseline_bytes", "increment_bytes", "scope", "status", "value"},
            "peak Metal memory",
        )
        host_in_use = host["metal"]["in_use_system_memory"]  # type: ignore[index]
        if (
            peak_metal["status"] != "measured"
            or peak_metal["scope"] != "system_wide"
            or peak_metal["baseline_bytes"] != host_in_use["baseline_bytes"]
            or peak_metal["value"] != host_in_use["peak_bytes"]
            or peak_metal["increment_bytes"] != host_in_use["increment_bytes"]
        ):
            raise ValueError("peak Metal summary mismatch")
    except (BenchmarkDataError, KeyError, OSError, TypeError, ValueError):
        _fail("full_ml_smoke_invalid")

    return {
        "environment_bindings": dict(FULL_ML_SMOKE_ENVIRONMENT_BINDINGS),
        "host_resources": host,
        "ml_environment_attestation_id": report[
            "ml_environment_attestation_id"
        ],
        "ml_environment_manifest_identity": report[
            "ml_environment_manifest_identity"
        ],
        "offline": True,
        "process_tree": {
            "baseline_bytes": baseline,
            "increment_bytes": peak - baseline,
            "managed_worker_roles": list(roles),
            "peak_bytes": peak,
            "sample_count": len(samples),
            "samples_bytes": list(samples),
            "sampling_interval_ms": 50,
            "scope": "smoke_process_and_descendants",
        },
        "profiles": normalized_profiles,
        "schema_version": FULL_ML_SMOKE_SCHEMA_VERSION,
        "status": "ready",
        "toolchain_identity": report["toolchain_identity"],
    }


def validate_rollback_proof(value: object, *, code_sha: str) -> JsonObject:
    """Validate a forced-failure, restart-observed generation rollback receipt."""

    try:
        if not isinstance(code_sha, str) or _CODE_SHA_RE.fullmatch(code_sha) is None:
            raise ValueError("invalid code SHA")
        proof = expect_object(value, "Phase-0 rollback proof")
        expect_fields(
            proof,
            {
                "schema_version",
                "proof_id",
                "code_sha",
                "forced_failure",
                "semantic_assertions",
                "status",
                "test_node_id",
                "test_result",
                "test_source_sha256",
            },
            "Phase-0 rollback proof",
        )
        if (
            type(proof["schema_version"]) is not int
            or proof["schema_version"] != PHASE0_ROLLBACK_PROOF_SCHEMA_VERSION
        ):
            raise ValueError("rollback proof schema mismatch")
        assertions = tuple(
            _identity(item)
            for item in expect_list(
                proof["semantic_assertions"],
                "rollback semantic assertions",
            )
        )
        expected_source = (
            Path(__file__).resolve().parents[4] / ROLLBACK_TEST_SOURCE
        )
        source_sha256 = sha256(
            _read_bounded_file(
                expected_source,
                4 * 1024**2,
                "rollback test source",
            )
        ).hexdigest()
        if (
            proof["status"] != "verified"
            or proof["code_sha"] != code_sha
            or proof["proof_id"] != ROLLBACK_PROOF_ID
            or proof["forced_failure"] != ROLLBACK_FORCED_FAILURE
            or proof["test_node_id"] != ROLLBACK_TEST_NODE_ID
            or proof["test_result"] != "passed"
            or proof["test_source_sha256"] != source_sha256
            or assertions != ROLLBACK_SEMANTIC_ASSERTIONS
        ):
            raise ValueError("rollback was not proved")
    except (BenchmarkDataError, KeyError, OSError, TypeError, ValueError):
        _fail("rollback_proof_invalid")

    return {
        "forced_failure": ROLLBACK_FORCED_FAILURE,
        "proof_id": ROLLBACK_PROOF_ID,
        "semantic_assertions": list(ROLLBACK_SEMANTIC_ASSERTIONS),
        "status": "verified",
        "test_node_id": ROLLBACK_TEST_NODE_ID,
        "test_result": "passed",
        "test_source_sha256": source_sha256,
    }


def _validate_product_dataset_binding(
    policy: FrozenMetricPolicy,
    dataset: BenchmarkDataset,
) -> JsonObject:
    """Bind the product fixture through policy data, never through a filename."""

    try:
        binding = getattr(policy.evaluation_data, "product_dataset")
        validate_frozen_metric_policy_product_dataset(policy, dataset)
        actual = {
            "dataset_id": dataset.dataset_id,
            "dataset_version": dataset.dataset_version,
            "dataset_revision": dataset_revision(dataset),
            "dataset_schema_version": dataset.schema_version,
            "case_count": len(dataset.cases),
            "source_group_count": len({case.split_group for case in dataset.cases}),
        }
        for field, observed in actual.items():
            if getattr(binding, field) != observed:
                raise ValueError("product dataset binding mismatch")
        if getattr(binding, "evidence_use") != "regression_only":
            raise ValueError("product dataset is not regression-only")
        if policy.evaluation_data.promotion_eligible is not False:
            raise ValueError("Phase-0 data cannot be promotion eligible")
        if policy.evaluation_data.promotion_evidence_status != "not_promotion_evidence":
            raise ValueError("Phase-0 data was labelled as promotion evidence")
    except (AttributeError, BenchmarkDataError, TypeError, ValueError):
        _fail("product_dataset_binding_invalid")
    return {
        **actual,
        "evidence_use": "regression_only",
        "promotion_eligible": False,
        "promotion_evidence_status": "not_promotion_evidence",
    }


def _validate_video_verifier(
    policy: FrozenMetricPolicy,
    dataset: VideoVerifierDataset,
    run: VideoVerifierRunManifest,
    *,
    code_sha: str,
    expected_model_identities: Sequence[str],
    expected_runtime_identity: str,
) -> JsonObject:
    try:
        validate_frozen_metric_policy_dataset(policy, dataset)
        revision = video_verifier_dataset_revision(dataset)
        if (
            run.code_sha != code_sha
            or run.run_status != "complete"
            or run.dataset_id != dataset.dataset_id
            or run.dataset_version != dataset.dataset_version
            or run.dataset_revision != revision
            or run.strict_no_fallback is not True
            or run.summary.infrastructure_error_count != 0
            or run.summary.case_count != len(dataset.cases)
            or run.verifier_identity.component_id != "video_verifier"
        ):
            raise ValueError("video verifier run mismatch")
        cases = {case.case_id: case for case in dataset.cases}
        if {attempt.case_id for attempt in run.attempts} != set(cases):
            raise ValueError("video verifier case set mismatch")
        for attempt in run.attempts:
            case = cases[attempt.case_id]
            if (
                attempt.status == "infrastructure_error"
                or attempt.prepared_input_sha256 != case.prepared_input_sha256
                or attempt.prepared_input_byte_size != case.prepared_input_byte_size
            ):
                raise ValueError("video verifier attempt mismatch")
            prediction = attempt.prediction
            if prediction is None:
                raise ValueError("video verifier prediction is missing")
            expected_facts = tuple(
                (item.fact_id, item.expected) for item in case.expected_facts
            )
            predicted_facts = tuple(
                (item.fact_id, item.expected) for item in prediction.facts
            )
            if tuple(item[0] for item in predicted_facts) != tuple(
                item[0] for item in expected_facts
            ):
                raise ValueError("video verifier prediction contract mismatch")
            expected_status = (
                "match"
                if predicted_facts == expected_facts
                and prediction.predicted_jersey == case.expected_jersey
                else "model_miss"
            )
            if attempt.status != expected_status:
                raise ValueError("video verifier result classification mismatch")
        verifier_identity_payload = parse_json_object(
            run.verifier_identity.identity,
            "video verifier identity",
        )
        required_identity_fields = {
            "mode",
            "contract",
            "model",
            "runtime_identity",
            "source_bundle_sha256",
            "prompt_protocol_sha256",
            "input_root_sha256",
            "fps",
            "max_tokens",
        }
        expect_fields(
            verifier_identity_payload,
            required_identity_fields,
            "video verifier identity",
        )
        if (
            verifier_identity_payload["mode"] != "isolated-worker"
            or verifier_identity_payload["contract"] != QWEN_WORKER_SCHEMA_VERSION
            or verifier_identity_payload["source_bundle_sha256"]
            != QWEN_SOURCE_BUNDLE_SHA256
            or verifier_identity_payload["prompt_protocol_sha256"]
            != QWEN_PROMPT_PROTOCOL_SHA256
            or verifier_identity_payload["runtime_identity"]
            != QWEN_INFERENCE_RUNTIME_IDENTITY
        ):
            raise ValueError("video verifier is not isolated")
        for field in (
            "source_bundle_sha256",
            "prompt_protocol_sha256",
            "input_root_sha256",
        ):
            if not _is_digest(verifier_identity_payload[field]):
                raise ValueError("video verifier source identity is invalid")
        for field in ("contract", "model", "runtime_identity"):
            _identity(verifier_identity_payload[field])
        if (
            verifier_identity_payload["model"] not in expected_model_identities
            or verifier_identity_payload["runtime_identity"]
            != expected_runtime_identity
        ):
            raise ValueError("video verifier environment identity mismatch")
        fps = _number(verifier_identity_payload["fps"], minimum=0.5)
        if fps > 8:
            raise ValueError("video verifier FPS is invalid")
        _integer(verifier_identity_payload["max_tokens"], minimum=64, maximum=1024)
    except (BenchmarkDataError, KeyError, TypeError, ValueError):
        _fail("video_verifier_evidence_invalid")
    return {
        "candidate_set_revision": run.candidate_set_revision,
        "dataset_id": dataset.dataset_id,
        "dataset_revision": revision,
        "dataset_version": dataset.dataset_version,
        "run_id": run.run_id,
        "run_status": "complete",
        "strict_no_fallback": True,
        "summary": {
            "case_count": run.summary.case_count,
            "infrastructure_error_count": 0,
            "match_count": run.summary.match_count,
            "model_miss_count": run.summary.model_miss_count,
        },
        "verifier_identity": "sha256:"
        + sha256(run.verifier_identity.identity.encode("utf-8")).hexdigest(),
    }


def _metric_map(run: BenchmarkRunManifest) -> dict[str, tuple[float, str | None]]:
    return {
        metric.name: (metric.value, metric.unit)
        for metric in (*run.quality_metrics, *run.system_metrics)
    }


def _policy_memory_limit(policy: FrozenMetricPolicy) -> int:
    matches = [
        guardrail
        for guardrail in policy.hard_guardrails.metrics
        if guardrail.guardrail_id == "peak_process_tree_rss_ceiling"
    ]
    if len(matches) != 1 or matches[0].absolute_threshold is None:
        _fail("metric_policy_memory_guardrail_invalid")
    threshold = matches[0].absolute_threshold
    if not math.isfinite(threshold) or threshold <= 0 or not threshold.is_integer():
        _fail("metric_policy_memory_guardrail_invalid")
    return int(threshold)


def _metric_observations(
    policy: FrozenMetricPolicy,
    run: BenchmarkRunManifest,
) -> JsonObject:
    available = _metric_map(run)
    observations: JsonObject = {}
    required_except_boundary = {
        metric.metric_name
        for metric in policy.primary_metrics
        if metric.metric_name
        not in {"infrastructure_error_count", "mean_boundary_error_seconds"}
    }
    missing = required_except_boundary - set(available)
    if missing:
        _fail("benchmark_primary_metrics_missing")
    expected_units = {
        "fraction": {"fraction", "ratio"},
        "count": {"count"},
        "seconds": {"seconds"},
        "milliseconds": {"milliseconds"},
        "bytes": {"bytes"},
    }
    for metric in policy.primary_metrics:
        if metric.metric_name == "infrastructure_error_count":
            observations[metric.metric_name] = {
                "status": "measured",
                "unit": metric.unit,
                "value": 0,
            }
            continue
        observed = available.get(metric.metric_name)
        if observed is None:
            observations[metric.metric_name] = {
                "reason_code": "no_matched_interval_for_boundary_measurement",
                "status": "not_observed",
                "unit": metric.unit,
            }
            continue
        value, raw_unit = observed
        if raw_unit not in expected_units[metric.unit]:
            _fail("benchmark_primary_metric_unit_mismatch")
        observations[metric.metric_name] = {
            "status": "measured",
            "unit": metric.unit,
            "value": value,
        }
    slice_metric_units = {
        "case_count": "count",
        "completed_case_count": "count",
        "error_count": "count",
        "candidate_recall_at_50": "fraction",
        "precision_at_5": "fraction",
        "recall_at_10": "fraction",
        "recall_at_20": "fraction",
        "ndcg_at_10": "fraction",
        "mean_boundary_error_seconds": "seconds",
        "hard_negative_hit_rate": "fraction",
        "p95_latency_ms": "milliseconds",
    }
    for critical_slice in policy.critical_slices:
        prefix = (
            f"slice.{critical_slice.dimension}.{critical_slice.value}."
        )
        slice_present = prefix + "case_count" in available
        for metric_name, unit in slice_metric_units.items():
            full_name = prefix + metric_name
            observed = available.get(full_name)
            if observed is None:
                observations[full_name] = {
                    "reason_code": (
                        "metric_not_applicable_to_slice_cases"
                        if slice_present
                        else "slice_absent_from_frozen_dataset"
                    ),
                    "status": "not_observed",
                    "unit": unit,
                }
                continue
            if observed[1] not in expected_units[unit]:
                _fail("benchmark_critical_slice_metric_unit_mismatch")
            observations[full_name] = {
                "status": "measured",
                "unit": unit,
                "value": observed[0],
            }
    return dict(sorted(observations.items()))


def _configured_profile_models(profile_id: str, run: BenchmarkRunManifest) -> None:
    models = {item.component_id: item.identity for item in run.model_identities}
    if profile_id != "lexical_qdrant" and models.get("visual_embedding") == "not-configured":
        _fail("benchmark_profile_not_executed")
    if profile_id in {"lighthouse", "qwen_verification"} and models.get(
        "lighthouse_model"
    ) == "not-configured":
        _fail("benchmark_profile_not_executed")
    if profile_id == "qwen_verification" and models.get("qwen_reranker") == "not-configured":
        _fail("benchmark_profile_not_executed")


def validate_benchmark_runs(
    policy: FrozenMetricPolicy,
    dataset: BenchmarkDataset,
    runs: Mapping[str, BenchmarkRunManifest],
    *,
    code_sha: str,
) -> tuple[list[JsonObject], list[JsonObject], list[JsonObject]]:
    """Validate five comparable, audited product runs and raw measurements."""

    if set(runs) != set(REQUIRED_EXECUTED_PROFILE_IDS):
        _fail("benchmark_profile_set_invalid")
    expected_revision = dataset_revision(dataset)
    memory_limit = _policy_memory_limit(policy)
    frozen_profiles = {item.profile_id: item for item in policy.profiles}
    summaries: list[JsonObject] = []
    measurements: list[JsonObject] = []
    errors: list[JsonObject] = []
    execution_mode: str | None = None
    hardware: object | None = None
    shared_component_identities: dict[tuple[str, str], str] = {}
    for profile_id in REQUIRED_EXECUTED_PROFILE_IDS:
        run = runs[profile_id]
        try:
            if (
                run.code_sha != code_sha
                or run.dataset_revision != expected_revision
                or run.run_status != "complete"
                or run.measurement_status != "complete"
                or run.measurement_evidence_status != "complete"
                or run.measurement_evidence is None
            ):
                raise ValueError("benchmark run is incomplete")
            profile = _validate_persisted_profile_identity_contract(run)
            if profile.profile_id != profile_id:
                raise ValueError("benchmark profile label mismatch")
            frozen = frozen_profiles[profile_id]
            if (
                profile.identity != frozen.profile_identity
                or profile.search_plan.identity != frozen.search_plan_identity
            ):
                raise ValueError("benchmark profile identity mismatch")
            audit_run_manifest(dataset, run)
        except (BenchmarkDataError, BenchmarkExecutionError, KeyError, TypeError, ValueError):
            _fail("benchmark_run_invalid")
        _configured_profile_models(profile_id, run)

        quality = {item.name: item.value for item in run.quality_metrics}
        if (
            quality.get("error_count") != 0
            or quality.get("infrastructure_failure_count") != 0
            or any(outcome.status != "complete" for outcome in run.case_outcomes)
        ):
            _fail("benchmark_infrastructure_error")
        rss = next(
            (
                item.value
                for item in run.system_metrics
                if item.name == "sampled_peak_process_tree_rss_bytes"
            ),
            None,
        )
        if rss is None or not math.isfinite(rss) or rss > memory_limit:
            _fail("benchmark_memory_guardrail_failed")
        protocol_prefix = policy.contracts.measurement_protocol_identity_prefix + ":"
        if (
            not run.measurement_protocol.identity.startswith(protocol_prefix)
            or not _is_digest(run.measurement_protocol.identity.removeprefix(protocol_prefix))
        ):
            _fail("benchmark_measurement_protocol_invalid")

        if execution_mode is None:
            execution_mode = run.execution_mode
            hardware = run.hardware
        elif run.execution_mode != execution_mode or run.hardware != hardware:
            _fail("benchmark_comparability_invalid")
        identities_by_role = {
            "model": {
                item.component_id: item.identity for item in run.model_identities
            },
            "index": {
                item.component_id: item.identity for item in run.index_identities
            },
            "config": {
                item.component_id: item.identity for item in run.config_identities
            },
        }
        for role, component_ids in (
            ("model", _SHARED_MODEL_COMPONENT_IDS),
            ("index", _SHARED_INDEX_COMPONENT_IDS),
            ("config", _SHARED_CONFIG_COMPONENT_IDS),
        ):
            for component_id in component_ids:
                value = identities_by_role[role].get(component_id)
                if value is None:
                    continue
                key = (role, component_id)
                expected = shared_component_identities.setdefault(key, value)
                if value != expected:
                    _fail("benchmark_comparability_invalid")
        metrics = _metric_observations(policy, run)
        model_miss_count = int(quality["model_miss_count"])
        summaries.append(
            {
                "execution_mode": run.execution_mode,
                "metrics": metrics,
                "profile_id": profile_id,
                "profile_identity": profile.identity,
                "run_id": run.run_id,
                "run_status": "complete",
                "search_plan_identity": profile.search_plan.identity,
            }
        )
        portable_run = run_to_dict(run)
        measurements.append(
            {
                "measurement_evidence": portable_run["measurement_evidence"],
                "measurement_protocol": portable_run["measurement_protocol"],
                "profile_id": profile_id,
                "run_id": run.run_id,
                "system_metrics": portable_run["system_metrics"],
            }
        )
        errors.append(
            {
                "category": "model_miss",
                "count": model_miss_count,
                "diagnostic_code": "relevant_interval_not_retrieved",
                "profile_id": profile_id,
                "source": "product_benchmark",
            }
        )
    return summaries, measurements, errors


def validate_baseline_batch(
    value: object,
    *,
    policy: FrozenMetricPolicy,
    dataset: BenchmarkDataset,
    environment: JsonObject,
    runs: Mapping[str, BenchmarkRunManifest],
    run_manifest_sha256s: Mapping[str, str],
    code_sha: str,
) -> JsonObject:
    """Bind all product runs to one attested ML environment and owner batch."""

    try:
        receipt = expect_object(value, "Phase-0 baseline batch")
        expect_fields(
            receipt,
            {
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
            },
            "Phase-0 baseline batch",
        )
        expected_policy_revision = frozen_metric_policy_revision(policy)
        expected_dataset_revision = dataset_revision(dataset)
        if (
            type(receipt["schema_version"]) is not int
            or receipt["schema_version"] != PHASE0_BASELINE_BATCH_SCHEMA_VERSION
            or receipt["status"] != "complete"
            or receipt["code_sha"] != code_sha
            or receipt["dataset_revision"] != expected_dataset_revision
            or receipt["policy_revision"] != expected_policy_revision
            or receipt["ml_environment_manifest_identity"]
            != environment["manifest_identity"]
            or receipt["ml_environment_attestation_id"]
            != environment["attestation_id"]
        ):
            raise ValueError("baseline batch identity mismatch")

        bindings = expect_object(
            receipt["environment_bindings"],
            "baseline batch environment bindings",
        )
        expect_fields(
            bindings,
            set(BASELINE_BATCH_ENVIRONMENT_BINDINGS),
            "baseline batch environment bindings",
        )
        if bindings != BASELINE_BATCH_ENVIRONMENT_BINDINGS:
            raise ValueError("baseline batch environment binding mismatch")

        lifecycle = expect_object(
            receipt["worker_lifecycle"],
            "baseline batch worker lifecycle",
        )
        expect_fields(
            lifecycle,
            {"cleanup_status", "retirement_status"},
            "baseline batch worker lifecycle",
        )
        if lifecycle != {
            "cleanup_status": "complete",
            "retirement_status": "complete",
        }:
            raise ValueError("baseline batch worker lifecycle incomplete")

        profile_runs = expect_list(
            receipt["profile_runs"],
            "baseline batch profile runs",
        )
        if len(profile_runs) != len(REQUIRED_EXECUTED_PROFILE_IDS):
            raise ValueError("baseline batch profile run set mismatch")
        normalized_runs: list[JsonObject] = []
        for expected_profile_id, raw_item in zip(
            REQUIRED_EXECUTED_PROFILE_IDS,
            profile_runs,
        ):
            item = expect_object(
                raw_item,
                f"baseline batch profile {expected_profile_id}",
            )
            expect_fields(
                item,
                {"profile_id", "run_id", "manifest_sha256"},
                f"baseline batch profile {expected_profile_id}",
            )
            manifest_sha256 = item["manifest_sha256"]
            run_id = _identity(item["run_id"])
            if (
                item["profile_id"] != expected_profile_id
                or set(runs) != set(REQUIRED_EXECUTED_PROFILE_IDS)
                or set(run_manifest_sha256s) != set(REQUIRED_EXECUTED_PROFILE_IDS)
                or run_id != runs[expected_profile_id].run_id
                or not _is_digest(manifest_sha256)
                or manifest_sha256
                != run_manifest_sha256s[expected_profile_id]
            ):
                raise ValueError("baseline batch profile run identity mismatch")
            normalized_runs.append(
                {
                    "manifest_sha256": manifest_sha256,
                    "profile_id": expected_profile_id,
                    "run_id": run_id,
                }
            )
    except (BenchmarkDataError, KeyError, TypeError, ValueError):
        _fail("baseline_batch_invalid")

    return {
        "code_sha": code_sha,
        "dataset_revision": expected_dataset_revision,
        "environment_bindings": dict(BASELINE_BATCH_ENVIRONMENT_BINDINGS),
        "ml_environment_attestation_id": environment["attestation_id"],
        "ml_environment_manifest_identity": environment["manifest_identity"],
        "policy_revision": expected_policy_revision,
        "profile_runs": normalized_runs,
        "status": "complete",
        "worker_lifecycle": {
            "cleanup_status": "complete",
            "retirement_status": "complete",
        },
    }


def _absolute_guardrail_observations(
    policy: FrozenMetricPolicy,
    profiles: Sequence[JsonObject],
) -> list[JsonObject]:
    result: list[JsonObject] = []
    for guardrail in policy.hard_guardrails.metrics:
        if guardrail.absolute_threshold is None:
            result.append(
                {
                    "guardrail_id": guardrail.guardrail_id,
                    "reason_code": "comparison_baseline_not_applicable_to_snapshot",
                    "status": "not_evaluable",
                }
            )
            continue
        for profile in profiles:
            metrics = profile["metrics"]
            if not isinstance(metrics, dict):
                _fail("internal_artifact_invalid")
            observation = metrics.get(guardrail.metric_name)
            if not isinstance(observation, dict) or observation.get("status") != "measured":
                result.append(
                    {
                        "guardrail_id": guardrail.guardrail_id,
                        "profile_id": profile["profile_id"],
                        "reason_code": "metric_not_observed",
                        "status": "not_evaluable",
                    }
                )
                continue
            actual = observation["value"]
            passed = (
                actual >= guardrail.absolute_threshold
                if guardrail.direction == "higher_is_better"
                else actual <= guardrail.absolute_threshold
            )
            result.append(
                {
                    "actual": actual,
                    "direction": guardrail.direction,
                    "guardrail_id": guardrail.guardrail_id,
                    "profile_id": profile["profile_id"],
                    "status": "pass" if passed else "fail",
                    "threshold": guardrail.absolute_threshold,
                }
            )
    return result


def _input_index_digest(code_sha: str, inputs: Mapping[str, str]) -> str:
    canonical: JsonObject = {
        "code_sha": code_sha,
        "evidence": [
            {"id": identifier, "sha256": inputs[identifier]}
            for identifier in sorted(inputs)
        ],
        "schema_version": PHASE0_EVIDENCE_SCHEMA_VERSION,
    }
    return "sha256:" + sha256(canonical_json_bytes(canonical)).hexdigest()


def build_phase0_evidence(
    *,
    policy: FrozenMetricPolicy,
    product_dataset: BenchmarkDataset,
    verifier_dataset: VideoVerifierDataset,
    environment_attestation: JsonObject,
    full_ml_smoke: JsonObject,
    baseline_batch: JsonObject,
    video_verifier_run: VideoVerifierRunManifest,
    benchmark_runs: Mapping[str, BenchmarkRunManifest],
    rollback_proof: JsonObject,
    code_sha: str,
    input_sha256s: Mapping[str, str],
) -> Phase0EvidenceArtifacts:
    """Build deterministic reports from already-loaded, exact evidence."""

    if not isinstance(code_sha, str) or _CODE_SHA_RE.fullmatch(code_sha) is None:
        _fail("code_identity_invalid")
    required_input_ids = {
        "frozen_metric_policy",
        "product_dataset",
        "video_verifier_dataset",
        "ml_environment_attestation",
        "full_ml_smoke",
        "baseline_batch",
        "video_verifier_run",
        "rollback_proof",
        *(f"benchmark_run.{profile_id}" for profile_id in REQUIRED_EXECUTED_PROFILE_IDS),
    }
    if set(input_sha256s) != required_input_ids or any(
        not _is_digest(value) for value in input_sha256s.values()
    ):
        _fail("input_evidence_index_invalid")

    product_binding = _validate_product_dataset_binding(policy, product_dataset)
    environment = validate_ml_environment_attestation(environment_attestation)
    memory_limit = _policy_memory_limit(policy)
    smoke = validate_full_ml_smoke(
        full_ml_smoke,
        code_sha=code_sha,
        memory_limit_bytes=memory_limit,
    )
    if (
        smoke["ml_environment_manifest_identity"]
        != environment["manifest_identity"]
        or smoke["ml_environment_attestation_id"]
        != environment["attestation_id"]
    ):
        _fail("full_ml_smoke_invalid")
    retained_component_execution = {
        profile_id: smoke["profiles"][profile_id]["component_execution"]  # type: ignore[index]
        for profile_id in REQUIRED_EXECUTED_PROFILE_IDS
    }
    qwen_capabilities = [
        item
        for item in environment["capabilities"]  # type: ignore[union-attr]
        if isinstance(item, dict) and item.get("id") == "qwen_verification"
    ]
    if len(qwen_capabilities) != 1:
        _fail("environment_attestation_invalid")
    qwen_capability = qwen_capabilities[0]
    verifier = _validate_video_verifier(
        policy,
        verifier_dataset,
        video_verifier_run,
        code_sha=code_sha,
        expected_model_identities=qwen_capability["model_identities"],  # type: ignore[arg-type]
        expected_runtime_identity=qwen_capability["runtime_identity"],  # type: ignore[arg-type]
    )
    rollback = validate_rollback_proof(rollback_proof, code_sha=code_sha)
    profile_summaries, benchmark_measurements, model_error_entries = (
        validate_benchmark_runs(
            policy,
            product_dataset,
            benchmark_runs,
            code_sha=code_sha,
        )
    )
    batch = validate_baseline_batch(
        baseline_batch,
        policy=policy,
        dataset=product_dataset,
        environment=environment,
        runs=benchmark_runs,
        run_manifest_sha256s={
            profile_id: input_sha256s[f"benchmark_run.{profile_id}"]
            for profile_id in REQUIRED_EXECUTED_PROFILE_IDS
        },
        code_sha=code_sha,
    )

    bundle_id = _input_index_digest(code_sha, input_sha256s)
    policy_revision = frozen_metric_policy_revision(policy)
    evidence_index = [
        {"id": identifier, "sha256": input_sha256s[identifier]}
        for identifier in sorted(input_sha256s)
    ]
    internvideo_policy = next(
        item for item in policy.profiles if item.profile_id == "internvideo"
    )
    all_profiles = [
        {
            "profile_id": item["profile_id"],
            "profile_identity": item["profile_identity"],
            "run_id": item["run_id"],
            "search_plan_identity": item["search_plan_identity"],
            "status": "executed",
        }
        for item in profile_summaries
    ] + [
        {
            "profile_id": "internvideo",
            "profile_identity": internvideo_policy.profile_identity,
            "reason_code": "provider_not_configured",
            "search_plan_identity": internvideo_policy.search_plan_identity,
            "status": "not_configured",
        }
    ]
    verifier_binding = policy.evaluation_data.verifier_dataset
    baseline_snapshot: JsonObject = {
        "bundle_id": bundle_id,
        "code_attestation": "clean_git_head_at_collection",
        "code_sha": code_sha,
        "datasets": {
            "product_regression": product_binding,
            "video_verifier_regression": {
                "case_count": verifier_binding.case_count,
                "dataset_id": verifier_binding.dataset_id,
                "dataset_revision": verifier_binding.dataset_revision,
                "dataset_schema_version": verifier_binding.dataset_schema_version,
                "dataset_version": verifier_binding.dataset_version,
                "evidence_use": "regression_only",
                "promotion_eligible": False,
                "source_group_count": verifier_binding.source_group_count,
            },
        },
        "environment": {
            "attestation_id": environment["attestation_id"],
            "manifest_identity": environment["manifest_identity"],
            "status": "complete",
        },
        "baseline_batch": batch,
        "evidence_index": evidence_index,
        "evidence_status": "valid",
        "full_ml_smoke": {
            "component_execution": retained_component_execution,
            "schema_version": smoke["schema_version"],
            "status": "ready",
            "toolchain_identity": smoke["toolchain_identity"],
        },
        "phase": 0,
        "policy": {
            "policy_id": policy.policy_id,
            "policy_revision": policy_revision,
            "policy_version": policy.policy_version,
        },
        "profiles": all_profiles,
        "promotion_eligible": False,
        "promotion_status": "not_promotion_evidence",
        "rollback": rollback,
        "schema_version": PHASE0_BASELINE_SNAPSHOT_SCHEMA_VERSION,
        "video_verifier": verifier,
    }
    raw_measurements: JsonObject = {
        "benchmark_runs": benchmark_measurements,
        "bundle_id": bundle_id,
        "code_sha": code_sha,
        "full_ml_smoke": {
            "host_resources": smoke["host_resources"],
            "process_tree": smoke["process_tree"],
        },
        "memory_accounting": "metal_standalone_not_additive_with_process_rss",
        "schema_version": PHASE0_RAW_MEASUREMENTS_SCHEMA_VERSION,
    }
    model_error_entries.append(
        {
            "category": "model_miss",
            "count": verifier["summary"]["model_miss_count"],  # type: ignore[index]
            "diagnostic_code": "strict_verifier_label_mismatch",
            "source": "video_verifier",
        }
    )
    error_ledger: JsonObject = {
        "bundle_id": bundle_id,
        "code_sha": code_sha,
        "entries": model_error_entries,
        "infrastructure_error_count": 0,
        "infrastructure_status": "clear",
        "model_miss_count_by_source": [
            {
                key: value
                for key, value in item.items()
                if key in {"source", "profile_id", "count"}
            }
            for item in model_error_entries
        ],
        "rollback_control": {
            "forced_failure_observed": True,
            "status": "verified",
        },
        "schema_version": PHASE0_ERROR_LEDGER_SCHEMA_VERSION,
    }
    sanitized_report: JsonObject = {
        "bundle_id": bundle_id,
        "code_sha": code_sha,
        "decision": {
            "baseline_evidence": "valid",
            "independent_holdout": "unavailable",
            "phase0_completion": "evidence_contract_satisfied",
            "promotion_eligible": False,
            "training_permission": "blocked_until_separate_phase_gate",
        },
        "environment": environment,
        "guardrail_observations": _absolute_guardrail_observations(
            policy,
            profile_summaries,
        ),
        "full_ml_smoke": {
            "component_execution": retained_component_execution,
            "schema_version": smoke["schema_version"],
            "status": "ready",
        },
        "profiles": profile_summaries,
        "reliability": {
            "full_ml_smoke": "ready",
            "infrastructure_error_count": 0,
            "internvideo": "not_configured",
            "metal_oom": "not_observed",
            "rollback": "verified",
            "sustained_swap": "not_observed",
        },
        "schema_version": PHASE0_SANITIZED_REPORT_SCHEMA_VERSION,
        "uncertainty": {
            "reason_code": "regression_seen_only_no_independent_holdout",
            "status": "insufficient_evidence_for_promotion",
        },
        "video_verifier": verifier,
    }
    return Phase0EvidenceArtifacts(
        baseline_snapshot=baseline_snapshot,
        raw_measurements=raw_measurements,
        sanitized_report=sanitized_report,
        error_ledger=error_ledger,
    )


def _load_json(path: Path, maximum: int, context: str) -> LoadedEvidence:
    raw = _read_bounded_file(Path(path), maximum, context)
    return LoadedEvidence(
        payload=parse_json_object(raw, context),
        sha256=sha256(raw).hexdigest(),
    )


def _attest_code_identity(
    code_sha: str,
    resolver: Callable[[], str],
) -> None:
    try:
        current = resolver()
    except Exception:
        _fail("code_identity_invalid")
    if current != code_sha:
        _fail("code_identity_invalid")


def collect_phase0_evidence(
    *,
    policy_path: Path,
    product_dataset_path: Path,
    verifier_dataset_path: Path,
    environment_attestation_path: Path,
    full_ml_smoke_path: Path,
    baseline_batch_path: Path,
    video_verifier_run_path: Path,
    benchmark_run_paths: Mapping[str, Path],
    rollback_proof_path: Path,
    output_dir: Path,
    code_sha: str,
    code_identity_resolver: Callable[[], str] | None = None,
) -> Phase0EvidenceArtifacts:
    """Validate one clean-checkout evidence set and publish it atomically."""

    if set(benchmark_run_paths) != set(REQUIRED_EXECUTED_PROFILE_IDS):
        _fail("benchmark_profile_set_invalid")
    if code_identity_resolver is None:
        from .cli import _current_code_sha

        code_identity_resolver = _current_code_sha
    _attest_code_identity(code_sha, code_identity_resolver)

    policy_loaded = _load_json(
        policy_path,
        MAX_PHASE0_EVIDENCE_BYTES,
        "frozen metric policy",
    )
    policy = frozen_metric_policy_from_dict(policy_loaded.payload)
    product_loaded = _load_json(
        product_dataset_path,
        MAX_DATASET_MANIFEST_BYTES,
        "Phase-0 product dataset",
    )
    product_dataset = dataset_from_dict(product_loaded.payload)
    verifier_dataset_loaded = _load_json(
        verifier_dataset_path,
        MAX_VIDEO_VERIFIER_MANIFEST_BYTES,
        "Phase-0 video verifier dataset",
    )
    verifier_dataset = video_verifier_dataset_from_json(
        canonical_json_bytes(verifier_dataset_loaded.payload)
    )
    environment_loaded = _load_json(
        environment_attestation_path,
        MAX_PHASE0_EVIDENCE_BYTES,
        "Phase-0 ML environment attestation",
    )
    smoke_loaded = _load_json(
        full_ml_smoke_path,
        MAX_PHASE0_EVIDENCE_BYTES,
        "Phase-0 full-ML smoke",
    )
    batch_loaded = _load_json(
        baseline_batch_path,
        MAX_PHASE0_EVIDENCE_BYTES,
        "Phase-0 baseline batch",
    )
    verifier_run_loaded = _load_json(
        video_verifier_run_path,
        MAX_VIDEO_VERIFIER_RUN_BYTES,
        "Phase-0 video verifier run",
    )
    video_verifier_run = video_verifier_run_from_dict(verifier_run_loaded.payload)
    rollback_loaded = _load_json(
        rollback_proof_path,
        MAX_PHASE0_EVIDENCE_BYTES,
        "Phase-0 rollback proof",
    )

    benchmark_runs: dict[str, BenchmarkRunManifest] = {}
    benchmark_loaded: dict[str, LoadedEvidence] = {}
    for profile_id in REQUIRED_EXECUTED_PROFILE_IDS:
        path = benchmark_run_paths[profile_id]
        loaded = _load_json(
            path,
            MAX_RUN_MANIFEST_BYTES,
            f"Phase-0 benchmark run {profile_id}",
        )
        benchmark_loaded[profile_id] = loaded
        benchmark_runs[profile_id] = run_from_dict(loaded.payload)

    input_sha256s = {
        "frozen_metric_policy": policy_loaded.sha256,
        "product_dataset": product_loaded.sha256,
        "video_verifier_dataset": verifier_dataset_loaded.sha256,
        "ml_environment_attestation": environment_loaded.sha256,
        "full_ml_smoke": smoke_loaded.sha256,
        "baseline_batch": batch_loaded.sha256,
        "video_verifier_run": verifier_run_loaded.sha256,
        "rollback_proof": rollback_loaded.sha256,
        **{
            f"benchmark_run.{profile_id}": benchmark_loaded[profile_id].sha256
            for profile_id in REQUIRED_EXECUTED_PROFILE_IDS
        },
    }
    artifacts = build_phase0_evidence(
        policy=policy,
        product_dataset=product_dataset,
        verifier_dataset=verifier_dataset,
        environment_attestation=environment_loaded.payload,
        full_ml_smoke=smoke_loaded.payload,
        baseline_batch=batch_loaded.payload,
        video_verifier_run=video_verifier_run,
        benchmark_runs=benchmark_runs,
        rollback_proof=rollback_loaded.payload,
        code_sha=code_sha,
        input_sha256s=input_sha256s,
    )
    # The last source check happens immediately before publication.  The output
    # directory may itself be destined for Git and therefore becomes untracked
    # only after this point.
    _attest_code_identity(code_sha, code_identity_resolver)
    write_phase0_evidence(output_dir, artifacts)
    return artifacts


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    if len(payload) > MAX_PHASE0_OUTPUT_BYTES:
        _fail("phase0_output_too_large")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def write_phase0_evidence(
    output_dir: Path,
    artifacts: Phase0EvidenceArtifacts,
) -> None:
    """Publish the complete four-file bundle without exposing a partial set."""

    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Phase-0 evidence output already exists")
    parent = target.parent
    if not parent.is_dir() or parent.is_symlink():
        raise FileNotFoundError("Phase-0 evidence output parent is unavailable")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=parent)
    )
    try:
        for filename, value in artifacts.files():
            _write_bytes_exclusive(temporary / filename, canonical_json_bytes(value))
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(temporary, target)
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


class _CliUsageError(ValueError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise _CliUsageError("invalid Phase-0 evidence arguments")


def _parser() -> _ArgumentParser:
    parser = _ArgumentParser(prog="phase0-evidence")
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--product-dataset", required=True, type=Path)
    parser.add_argument("--verifier-dataset", required=True, type=Path)
    parser.add_argument("--environment-attestation", required=True, type=Path)
    parser.add_argument("--full-ml-smoke", required=True, type=Path)
    parser.add_argument("--baseline-batch", required=True, type=Path)
    parser.add_argument("--video-verifier-run", required=True, type=Path)
    parser.add_argument("--benchmark-run", action="append", required=True)
    parser.add_argument("--rollback-proof", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--code-sha", required=True)
    return parser


def _benchmark_run_arguments(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if not isinstance(value, str) or "=" not in value:
            raise _CliUsageError("invalid benchmark run argument")
        profile_id, path = value.split("=", 1)
        if profile_id not in REQUIRED_EXECUTED_PROFILE_IDS or not path:
            raise _CliUsageError("invalid benchmark run argument")
        if profile_id in result:
            raise _CliUsageError("duplicate benchmark run argument")
        result[profile_id] = Path(path)
    if set(result) != set(REQUIRED_EXECUTED_PROFILE_IDS):
        raise _CliUsageError("incomplete benchmark run arguments")
    return result


def _write_cli_json(stream: object, value: JsonObject) -> None:
    stream.write(  # type: ignore[attr-defined]
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    try:
        arguments = _parser().parse_args(argv)
        runs = _benchmark_run_arguments(arguments.benchmark_run)
        artifacts = collect_phase0_evidence(
            policy_path=arguments.policy,
            product_dataset_path=arguments.product_dataset,
            verifier_dataset_path=arguments.verifier_dataset,
            environment_attestation_path=arguments.environment_attestation,
            full_ml_smoke_path=arguments.full_ml_smoke,
            baseline_batch_path=arguments.baseline_batch,
            video_verifier_run_path=arguments.video_verifier_run,
            benchmark_run_paths=runs,
            rollback_proof_path=arguments.rollback_proof,
            output_dir=arguments.output_dir,
            code_sha=arguments.code_sha,
        )
        _write_cli_json(
            sys.stdout,
            {
                "bundle_id": artifacts.baseline_snapshot["bundle_id"],
                "code_sha": arguments.code_sha,
                "status": "written",
            },
        )
        return 0
    except _CliUsageError:
        _write_cli_json(
            sys.stderr,
            {"error": "usage_error", "message": "invalid Phase-0 evidence arguments"},
        )
        return 2
    except FileNotFoundError:
        _write_cli_json(
            sys.stderr,
            {"error": "not_found", "message": "Phase-0 evidence input was not found"},
        )
        return 4
    except FileExistsError:
        _write_cli_json(
            sys.stderr,
            {"error": "conflict", "message": "Phase-0 evidence output already exists"},
        )
        return 5
    except (Phase0EvidenceError, BenchmarkDataError, BenchmarkExecutionError):
        _write_cli_json(
            sys.stderr,
            {"error": "invalid_evidence", "message": "Phase-0 evidence is invalid"},
        )
        return 3
    except OSError:
        _write_cli_json(
            sys.stderr,
            {"error": "io_error", "message": "Phase-0 evidence storage failed"},
        )
        return 7
    except KeyboardInterrupt:
        _write_cli_json(
            sys.stderr,
            {"error": "interrupted", "message": "Phase-0 evidence collection was interrupted"},
        )
        return 130
    except Exception:
        _write_cli_json(
            sys.stderr,
            {"error": "internal_error", "message": "Phase-0 evidence collection failed"},
        )
        return 70


__all__ = [
    "ALL_PHASE0_PROFILE_IDS",
    "BASELINE_BATCH_ENVIRONMENT_BINDINGS",
    "COMPONENT_EXECUTION_IDS",
    "COMPONENT_EXECUTION_SCHEMA_VERSION",
    "FULL_ML_SMOKE_ENVIRONMENT_BINDINGS",
    "FULL_ML_SMOKE_SCHEMA_VERSION",
    "PHASE0_BASELINE_BATCH_SCHEMA_VERSION",
    "PHASE0_BASELINE_SNAPSHOT_SCHEMA_VERSION",
    "PHASE0_ERROR_LEDGER_SCHEMA_VERSION",
    "PHASE0_EVIDENCE_SCHEMA_VERSION",
    "PHASE0_RAW_MEASUREMENTS_SCHEMA_VERSION",
    "PHASE0_ROLLBACK_PROOF_SCHEMA_VERSION",
    "PHASE0_SANITIZED_REPORT_SCHEMA_VERSION",
    "Phase0EvidenceArtifacts",
    "Phase0EvidenceError",
    "REQUIRED_EXECUTED_PROFILE_IDS",
    "build_phase0_evidence",
    "collect_phase0_evidence",
    "main",
    "validate_baseline_batch",
    "validate_benchmark_runs",
    "validate_full_ml_smoke",
    "validate_ml_environment_attestation",
    "validate_rollback_proof",
    "write_phase0_evidence",
]
