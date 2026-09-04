from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import subprocess
import sys
from typing import Literal


ML_ENVIRONMENT_MANIFEST_SCHEMA_VERSION = 1
ML_ENVIRONMENT_REPORT_SCHEMA_VERSION = 1
DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256 = (
    "ec4d3feb390d6c6bdffffe747491ad0f11ad83b954cc58cc9cb74698be3992ae"
)
MAX_ML_ENVIRONMENT_MANIFEST_BYTES = 1024 * 1024
MAX_ATTESTED_CONTRACT_BYTES = 64 * 1024 * 1024
MAX_PROBE_OUTPUT_BYTES = 2 * 1024 * 1024
PROBE_TIMEOUT_SECONDS = 15.0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PYTHON_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_DISTRIBUTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REQUIREMENT_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^ ;\\]+)(?:[ ;\\]|$)"
)
_CONTRACT_KINDS = frozenset(
    {
        "dependency_lock",
        "worker_manifest",
        "model_manifest",
        "model_source_manifest",
    }
)
_BOOTSTRAP_DISTRIBUTIONS = frozenset({"pip", "setuptools", "wheel"})
_SYSTEM_PROFILER = Path("/usr/sbin/system_profiler")


class MlEnvironmentManifestError(ValueError):
    """A bounded, path-free manifest failure suitable for CLI reporting."""

    def __init__(
        self,
        code: str,
        *,
        kind: Literal["drift", "infrastructure"] = "drift",
    ) -> None:
        self.code = code
        self.kind = kind
        super().__init__(code)


class _ProbeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HostSnapshot:
    system: str
    machine: str
    macos_version: str
    chip: str
    memory_gib: int


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    implementation: str
    python_version: str
    system: str
    machine: str
    macos_version: str
    distribution_identity: str


@dataclass(frozen=True, slots=True)
class UvSnapshot:
    version: str
    sha256: str


@dataclass(frozen=True, slots=True)
class _HostContract:
    system: str
    machine: str
    minimum_macos_major: int
    chip: str
    memory_gib: int


@dataclass(frozen=True, slots=True)
class _ContractFile:
    contract_id: str
    kind: str
    path: str
    sha256: str
    canonical_json_sha256: str | None


@dataclass(frozen=True, slots=True)
class _DistributionSource:
    kind: Literal["frozen_identity", "requirements_lock", "json_mapping"]
    sha256: str | None = None
    contract_id: str | None = None
    field: str | None = None
    direct_distributions: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class _EnvironmentContract:
    environment_id: str
    directory: str
    python: str
    dependency_contract_id: str
    distribution_source: _DistributionSource


@dataclass(frozen=True, slots=True)
class _CapabilityContract:
    capability_id: str
    environment_ids: tuple[str, ...]
    contract_ids: tuple[str, ...]
    runtime_identity: str
    model_identities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _UvContract:
    environment_id: str
    path: str
    version: str
    sha256: str


@dataclass(frozen=True, slots=True)
class MlEnvironmentManifest:
    schema_version: int
    attestation_id: str
    host: _HostContract
    offline_environment: Mapping[str, str]
    uv: _UvContract
    contracts: tuple[_ContractFile, ...]
    environments: tuple[_EnvironmentContract, ...]
    capabilities: tuple[_CapabilityContract, ...]
    raw_sha256: str


@dataclass(frozen=True, slots=True)
class _ContractObservation:
    status: Literal["complete", "failed"]
    failure_kind: Literal["drift", "infrastructure"] | None
    diagnostic: str | None
    raw: bytes | None
    payload: Mapping[str, object] | None


def canonical_json_sha256(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise MlEnvironmentManifestError("manifest_invalid") from error
    return sha256(encoded).hexdigest()


def _expect_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    optional: set[str] | None = None,
) -> None:
    optional_fields = optional or set()
    if set(value) - expected - optional_fields or expected - set(value):
        raise MlEnvironmentManifestError("manifest_invalid")


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise MlEnvironmentManifestError("manifest_invalid")
    return value


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise MlEnvironmentManifestError("manifest_invalid")
    return value


def _id(value: object) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise MlEnvironmentManifestError("manifest_invalid")
    return value


def _text(value: object, *, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
        or "\\" in value
    ):
        raise MlEnvironmentManifestError("manifest_invalid")
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MlEnvironmentManifestError("manifest_invalid")
    return value


def _relative_path(value: object) -> str:
    path = _text(value, maximum=512)
    parsed = PurePosixPath(path)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != path
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise MlEnvironmentManifestError("manifest_invalid")
    return path


def _parse_json(raw: bytes) -> dict[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise MlEnvironmentManifestError("manifest_invalid")
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise MlEnvironmentManifestError("manifest_invalid")

    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except MlEnvironmentManifestError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise MlEnvironmentManifestError("manifest_invalid") from error
    return _object(value)


def _read_bounded_regular_file(path: Path, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise MlEnvironmentManifestError(
            "manifest_unavailable",
            kind="infrastructure",
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise MlEnvironmentManifestError("manifest_invalid")
        if before.st_size < 2 or before.st_size > maximum:
            raise MlEnvironmentManifestError("manifest_invalid")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        raw = b"".join(chunks)
        if len(raw) > maximum:
            raise MlEnvironmentManifestError("manifest_invalid")
        fingerprint = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if fingerprint(before) != fingerprint(after) or len(raw) != before.st_size:
            raise MlEnvironmentManifestError("manifest_changed")
        return raw
    finally:
        os.close(descriptor)


def _parse_host(value: object) -> _HostContract:
    source = _object(value)
    _expect_fields(
        source,
        {"system", "machine", "minimum_macos_major", "chip", "memory_gib"},
    )
    minimum = source["minimum_macos_major"]
    memory = source["memory_gib"]
    if type(minimum) is not int or minimum < 14 or minimum > 99:
        raise MlEnvironmentManifestError("manifest_invalid")
    if type(memory) is not int or memory < 8 or memory > 1024:
        raise MlEnvironmentManifestError("manifest_invalid")
    return _HostContract(
        system=_text(source["system"], maximum=32),
        machine=_text(source["machine"], maximum=32),
        minimum_macos_major=minimum,
        chip=_text(source["chip"], maximum=128),
        memory_gib=memory,
    )


def _parse_offline_environment(value: object) -> Mapping[str, str]:
    source = _object(value)
    if not source or len(source) > 32:
        raise MlEnvironmentManifestError("manifest_invalid")
    result: dict[str, str] = {}
    for raw_name, raw_value in source.items():
        if _ID_RE.fullmatch(raw_name) is None:
            raise MlEnvironmentManifestError("manifest_invalid")
        result[raw_name] = _text(raw_value, maximum=32)
    return dict(sorted(result.items()))


def _parse_uv(value: object) -> _UvContract:
    source = _object(value)
    _expect_fields(source, {"environment_id", "path", "version", "sha256"})
    return _UvContract(
        environment_id=_id(source["environment_id"]),
        path=_relative_path(source["path"]),
        version=_text(source["version"], maximum=64),
        sha256=_digest(source["sha256"]),
    )


def _parse_contract(value: object) -> _ContractFile:
    source = _object(value)
    _expect_fields(
        source,
        {"id", "kind", "path", "sha256"},
        optional={"canonical_json_sha256"},
    )
    kind = _text(source["kind"], maximum=64)
    if kind not in _CONTRACT_KINDS:
        raise MlEnvironmentManifestError("manifest_invalid")
    canonical = source.get("canonical_json_sha256")
    if kind.endswith("manifest") and canonical is None:
        raise MlEnvironmentManifestError("manifest_invalid")
    if not kind.endswith("manifest") and canonical is not None:
        raise MlEnvironmentManifestError("manifest_invalid")
    return _ContractFile(
        contract_id=_id(source["id"]),
        kind=kind,
        path=_relative_path(source["path"]),
        sha256=_digest(source["sha256"]),
        canonical_json_sha256=None if canonical is None else _digest(canonical),
    )


def _parse_distribution_source(value: object) -> _DistributionSource:
    source = _object(value)
    kind = source.get("kind")
    if kind == "frozen_identity":
        _expect_fields(source, {"kind", "sha256"})
        return _DistributionSource(kind=kind, sha256=_digest(source["sha256"]))
    if kind == "requirements_lock":
        _expect_fields(
            source,
            {"kind", "contract_id"},
            optional={"direct_distributions"},
        )
        direct = source.get("direct_distributions")
        return _DistributionSource(
            kind=kind,
            contract_id=_id(source["contract_id"]),
            direct_distributions=(
                None
                if direct is None
                else _normalized_distribution_mapping(direct)
            ),
        )
    if kind == "json_mapping":
        _expect_fields(source, {"kind", "contract_id", "field"})
        return _DistributionSource(
            kind=kind,
            contract_id=_id(source["contract_id"]),
            field=_id(source["field"]),
        )
    raise MlEnvironmentManifestError("manifest_invalid")


def _parse_environment(value: object) -> _EnvironmentContract:
    source = _object(value)
    _expect_fields(
        source,
        {
            "id",
            "directory",
            "python",
            "dependency_contract_id",
            "distribution_source",
        },
    )
    python = source["python"]
    if not isinstance(python, str) or _PYTHON_RE.fullmatch(python) is None:
        raise MlEnvironmentManifestError("manifest_invalid")
    return _EnvironmentContract(
        environment_id=_id(source["id"]),
        directory=_relative_path(source["directory"]),
        python=python,
        dependency_contract_id=_id(source["dependency_contract_id"]),
        distribution_source=_parse_distribution_source(source["distribution_source"]),
    )


def _parse_capability(value: object) -> _CapabilityContract:
    source = _object(value)
    _expect_fields(
        source,
        {
            "id",
            "environment_ids",
            "contract_ids",
            "runtime_identity",
            "model_identities",
        },
    )
    environment_ids = tuple(_id(item) for item in _list(source["environment_ids"]))
    contract_ids = tuple(_id(item) for item in _list(source["contract_ids"]))
    model_identities = tuple(
        _text(item, maximum=1024) for item in _list(source["model_identities"])
    )
    if (
        not environment_ids
        or not contract_ids
        or not model_identities
        or len(environment_ids) > 8
        or len(contract_ids) > 16
        or len(model_identities) > 16
        or len(set(environment_ids)) != len(environment_ids)
        or len(set(contract_ids)) != len(contract_ids)
        or len(set(model_identities)) != len(model_identities)
    ):
        raise MlEnvironmentManifestError("manifest_invalid")
    return _CapabilityContract(
        capability_id=_id(source["id"]),
        environment_ids=environment_ids,
        contract_ids=contract_ids,
        runtime_identity=_text(source["runtime_identity"], maximum=4096),
        model_identities=model_identities,
    )


def load_ml_environment_manifest(
    path: Path,
    *,
    expected_sha256: str,
) -> MlEnvironmentManifest:
    expected = _digest(expected_sha256)
    raw = _read_bounded_regular_file(Path(path), MAX_ML_ENVIRONMENT_MANIFEST_BYTES)
    actual = sha256(raw).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise MlEnvironmentManifestError("manifest_identity_mismatch")
    source = _parse_json(raw)
    _expect_fields(
        source,
        {
            "schema_version",
            "attestation_id",
            "host",
            "offline_environment",
            "uv",
            "contracts",
            "environments",
            "capabilities",
        },
    )
    if source["schema_version"] != ML_ENVIRONMENT_MANIFEST_SCHEMA_VERSION:
        raise MlEnvironmentManifestError("manifest_invalid")
    contracts = tuple(_parse_contract(item) for item in _list(source["contracts"]))
    environments = tuple(
        _parse_environment(item) for item in _list(source["environments"])
    )
    capabilities = tuple(
        _parse_capability(item) for item in _list(source["capabilities"])
    )
    if not contracts or not environments or not capabilities:
        raise MlEnvironmentManifestError("manifest_invalid")
    contract_ids = tuple(item.contract_id for item in contracts)
    environment_ids = tuple(item.environment_id for item in environments)
    capability_ids = tuple(item.capability_id for item in capabilities)
    if (
        len(set(contract_ids)) != len(contract_ids)
        or len(set(environment_ids)) != len(environment_ids)
        or len(set(capability_ids)) != len(capability_ids)
    ):
        raise MlEnvironmentManifestError("manifest_invalid")
    contract_set = set(contract_ids)
    environment_set = set(environment_ids)
    uv = _parse_uv(source["uv"])
    if uv.environment_id not in environment_set:
        raise MlEnvironmentManifestError("manifest_invalid")
    for environment in environments:
        references = {environment.dependency_contract_id}
        if environment.distribution_source.contract_id is not None:
            references.add(environment.distribution_source.contract_id)
        if not references <= contract_set:
            raise MlEnvironmentManifestError("manifest_invalid")
    for capability in capabilities:
        if not set(capability.environment_ids) <= environment_set or not set(
            capability.contract_ids
        ) <= contract_set:
            raise MlEnvironmentManifestError("manifest_invalid")
    return MlEnvironmentManifest(
        schema_version=ML_ENVIRONMENT_MANIFEST_SCHEMA_VERSION,
        attestation_id=_id(source["attestation_id"]),
        host=_parse_host(source["host"]),
        offline_environment=_parse_offline_environment(source["offline_environment"]),
        uv=uv,
        contracts=contracts,
        environments=environments,
        capabilities=capabilities,
        raw_sha256=actual,
    )


def _project_path(project_root: Path, relative: str) -> Path:
    try:
        root = project_root.resolve(strict=True)
        candidate = root.joinpath(*PurePosixPath(relative).parts)
        resolved_parent = candidate.parent.resolve(strict=False)
        resolved_parent.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise _ProbeError("unsafe_project_path") from error
    return candidate


def _macos_major(value: str) -> int | None:
    try:
        result = int(value.split(".", 1)[0])
    except (AttributeError, TypeError, ValueError):
        return None
    return result if 1 <= result <= 99 else None


def _default_host_probe() -> HostSnapshot:
    try:
        completed = subprocess.run(
            [str(_SYSTEM_PROFILER), "SPHardwareDataType", "-json"],
            env={"LC_ALL": "C"},
            capture_output=True,
            check=False,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise _ProbeError("host_probe_failed") from error
    if (
        completed.returncode != 0
        or len(completed.stdout) > MAX_PROBE_OUTPUT_BYTES
        or completed.stderr
    ):
        raise _ProbeError("host_probe_failed")
    try:
        payload = json.loads(completed.stdout.decode("utf-8", errors="strict"))
        rows = payload["SPHardwareDataType"]
        hardware = rows[0]
        chip = hardware["chip_type"]
        memory = hardware["physical_memory"]
    except (KeyError, IndexError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise _ProbeError("host_probe_failed") from error
    match = re.fullmatch(r"([0-9]{1,4}) GB", memory) if isinstance(memory, str) else None
    if not isinstance(chip, str) or not chip or match is None:
        raise _ProbeError("host_probe_failed")
    return HostSnapshot(
        system=platform.system(),
        machine=platform.machine().casefold(),
        macos_version=platform.mac_ver()[0],
        chip=chip,
        memory_gib=int(match.group(1)),
    )


_RUNTIME_PROBE = r"""
import hashlib
import importlib.metadata
import json
import platform
import re

ignored = {"pip", "setuptools", "wheel"}
distributions = {}
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    if not isinstance(name, str) or not name:
        raise SystemExit(31)
    normalized = re.sub(r"[-_.]+", "-", name).casefold()
    if normalized in distributions:
        raise SystemExit(32)
    distributions[normalized] = distribution.version
for name in ignored:
    distributions.pop(name, None)
encoded = json.dumps(
    dict(sorted(distributions.items())),
    ensure_ascii=True,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
payload = {
    "distribution_identity": hashlib.sha256(encoded).hexdigest(),
    "implementation": platform.python_implementation(),
    "machine": platform.machine().casefold(),
    "macos_version": platform.mac_ver()[0],
    "python_version": platform.python_version(),
    "system": platform.system(),
}
print(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
"""


def _sanitized_subprocess_environment(
    offline_environment: Mapping[str, str],
) -> dict[str, str]:
    return {"LC_ALL": "C", **offline_environment}


def _default_runtime_probe(
    python: Path,
    _environment_id: str,
    *,
    offline_environment: Mapping[str, str],
) -> RuntimeSnapshot:
    try:
        completed = subprocess.run(
            [str(python), "-I", "-B", "-c", _RUNTIME_PROBE],
            env=_sanitized_subprocess_environment(offline_environment),
            capture_output=True,
            check=False,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise _ProbeError("environment_probe_failed") from error
    if (
        completed.returncode != 0
        or completed.stderr
        or not completed.stdout
        or len(completed.stdout) > MAX_PROBE_OUTPUT_BYTES
    ):
        raise _ProbeError("environment_probe_failed")
    try:
        payload = _parse_json(completed.stdout)
        _expect_fields(
            payload,
            {
                "implementation",
                "python_version",
                "system",
                "machine",
                "macos_version",
                "distribution_identity",
            },
        )
        return RuntimeSnapshot(
            implementation=_text(payload["implementation"], maximum=64),
            python_version=_text(payload["python_version"], maximum=32),
            system=_text(payload["system"], maximum=32),
            machine=_text(payload["machine"], maximum=32),
            macos_version=_text(payload["macos_version"], maximum=32),
            distribution_identity=_digest(payload["distribution_identity"]),
        )
    except MlEnvironmentManifestError as error:
        raise _ProbeError("environment_probe_failed") from error


def _default_uv_probe(
    executable: Path,
    *,
    offline_environment: Mapping[str, str],
) -> UvSnapshot:
    try:
        raw = _read_bounded_regular_file(executable, 128 * 1024 * 1024)
        completed = subprocess.run(
            [str(executable), "--version"],
            env=_sanitized_subprocess_environment(offline_environment),
            capture_output=True,
            check=False,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (MlEnvironmentManifestError, OSError, subprocess.SubprocessError) as error:
        raise _ProbeError("uv_probe_failed") from error
    if (
        completed.returncode != 0
        or completed.stderr
        or not completed.stdout
        or len(completed.stdout) > 4096
    ):
        raise _ProbeError("uv_probe_failed")
    try:
        output = completed.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeError as error:
        raise _ProbeError("uv_probe_failed") from error
    match = re.fullmatch(r"uv ([0-9]+\.[0-9]+\.[0-9]+)(?: \([^\r\n]{1,256}\))?", output)
    if match is None:
        raise _ProbeError("uv_probe_failed")
    return UvSnapshot(version=match.group(1), sha256=sha256(raw).hexdigest())


def _normalized_distribution_mapping(value: object) -> dict[str, str]:
    source = _object(value)
    if not source or len(source) > 4096:
        raise MlEnvironmentManifestError("manifest_invalid")
    result: dict[str, str] = {}
    for raw_name, raw_version in source.items():
        if _DISTRIBUTION_RE.fullmatch(raw_name) is None:
            raise MlEnvironmentManifestError("manifest_invalid")
        name = re.sub(r"[-_.]+", "-", raw_name).casefold()
        version = _text(raw_version, maximum=128)
        if name in result:
            raise MlEnvironmentManifestError("manifest_invalid")
        if name not in _BOOTSTRAP_DISTRIBUTIONS:
            result[name] = version
    if not result:
        raise MlEnvironmentManifestError("manifest_invalid")
    return dict(sorted(result.items()))


def _requirements_distribution_mapping(raw: bytes) -> dict[str, str]:
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeError as error:
        raise MlEnvironmentManifestError("manifest_invalid") from error
    result: dict[str, str] = {}
    for line in lines:
        match = _REQUIREMENT_RE.match(line)
        if match is None:
            continue
        name = re.sub(r"[-_.]+", "-", match.group(1)).casefold()
        version = match.group(2)
        if name in _BOOTSTRAP_DISTRIBUTIONS:
            continue
        if name in result and result[name] != version:
            raise MlEnvironmentManifestError("manifest_invalid")
        result[name] = version
    if not result:
        raise MlEnvironmentManifestError("manifest_invalid")
    return dict(sorted(result.items()))


def _failure(
    scope: str,
    identifier: str,
    kind: Literal["drift", "infrastructure"],
    code: str,
) -> dict[str, str]:
    return {"code": code, "id": identifier, "kind": kind, "scope": scope}


def _contract_observations(
    project_root: Path,
    manifest: MlEnvironmentManifest,
    failures: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, _ContractObservation]]:
    reports: list[dict[str, object]] = []
    observations: dict[str, _ContractObservation] = {}
    for contract in sorted(manifest.contracts, key=lambda item: item.contract_id):
        failure_kind: Literal["drift", "infrastructure"] | None = None
        diagnostic: str | None = None
        raw: bytes | None = None
        payload: Mapping[str, object] | None = None
        try:
            path = _project_path(project_root, contract.path)
            raw = _read_bounded_regular_file(path, MAX_ATTESTED_CONTRACT_BYTES)
        except (_ProbeError, MlEnvironmentManifestError):
            failure_kind = "infrastructure"
            diagnostic = "contract_unavailable"
        if raw is not None and not hmac.compare_digest(
            sha256(raw).hexdigest(), contract.sha256
        ):
            failure_kind = "drift"
            diagnostic = "contract_identity_mismatch"
        if raw is not None and contract.canonical_json_sha256 is not None:
            try:
                parsed = _parse_json(raw)
                actual_canonical = canonical_json_sha256(parsed)
            except MlEnvironmentManifestError:
                failure_kind = "drift"
                diagnostic = "contract_invalid"
            else:
                if not hmac.compare_digest(
                    actual_canonical,
                    contract.canonical_json_sha256,
                ):
                    failure_kind = "drift"
                    diagnostic = "contract_identity_mismatch"
                else:
                    payload = parsed
        if failure_kind is None:
            observations[contract.contract_id] = _ContractObservation(
                status="complete",
                failure_kind=None,
                diagnostic=None,
                raw=raw,
                payload=payload,
            )
            report: dict[str, object] = {
                "id": contract.contract_id,
                "identity": "sha256:" + contract.sha256,
                "kind": contract.kind,
                "status": "complete",
            }
            if contract.canonical_json_sha256 is not None:
                report["canonical_identity"] = (
                    "sha256:" + contract.canonical_json_sha256
                )
            reports.append(report)
        else:
            observations[contract.contract_id] = _ContractObservation(
                status="failed",
                failure_kind=failure_kind,
                diagnostic=diagnostic,
                raw=None,
                payload=None,
            )
            reports.append(
                {
                    "diagnostic": diagnostic,
                    "failure_kind": failure_kind,
                    "id": contract.contract_id,
                    "kind": contract.kind,
                    "status": "failed",
                }
            )
            failures.append(
                _failure(
                    "contract",
                    contract.contract_id,
                    failure_kind,
                    diagnostic or "contract_failed",
                )
            )
    return reports, observations


def _expected_distribution_identity(
    source: _DistributionSource,
    observations: Mapping[str, _ContractObservation],
) -> str:
    if source.kind == "frozen_identity":
        assert source.sha256 is not None
        return source.sha256
    assert source.contract_id is not None
    observation = observations[source.contract_id]
    if observation.status != "complete":
        raise _ProbeError("distribution_contract_failed")
    if source.kind == "requirements_lock":
        if observation.raw is None:
            raise _ProbeError("distribution_contract_failed")
        distributions = _requirements_distribution_mapping(observation.raw)
        for name, version in (source.direct_distributions or {}).items():
            if name in distributions and distributions[name] != version:
                raise _ProbeError("distribution_contract_failed")
            distributions[name] = version
        return canonical_json_sha256(dict(sorted(distributions.items())))
    if observation.payload is None or source.field is None:
        raise _ProbeError("distribution_contract_failed")
    try:
        selected = observation.payload[source.field]
    except KeyError as error:
        raise _ProbeError("distribution_contract_failed") from error
    return canonical_json_sha256(_normalized_distribution_mapping(selected))


def _environment_exists(path: Path) -> bool:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise _ProbeError("environment_unavailable") from error
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise _ProbeError("environment_unsafe")
    return True


def _python_executable(environment: Path) -> Path:
    candidate = environment / "bin" / "python"
    try:
        resolved = candidate.resolve(strict=True)
        value = resolved.stat()
    except (OSError, RuntimeError) as error:
        raise _ProbeError("python_unavailable") from error
    if not stat.S_ISREG(value.st_mode) or not os.access(resolved, os.X_OK):
        raise _ProbeError("python_unavailable")
    return candidate


def _attest_host(
    contract: _HostContract,
    probe: Callable[[], HostSnapshot],
    failures: list[dict[str, str]],
) -> dict[str, object]:
    try:
        observed = probe()
    except Exception:
        failures.append(
            _failure("host", "target-host", "infrastructure", "host_probe_failed")
        )
        return {
            "diagnostic": "host_probe_failed",
            "failure_kind": "infrastructure",
            "status": "failed",
        }
    major = _macos_major(observed.macos_version)
    if (
        observed.system != contract.system
        or observed.machine.casefold() != contract.machine.casefold()
        or major is None
        or major < contract.minimum_macos_major
        or observed.chip != contract.chip
        or observed.memory_gib != contract.memory_gib
    ):
        failures.append(
            _failure("host", "target-host", "drift", "host_contract_mismatch")
        )
        return {
            "diagnostic": "host_contract_mismatch",
            "failure_kind": "drift",
            "status": "failed",
        }
    return {
        "chip": observed.chip,
        "machine": observed.machine.casefold(),
        "macos_major": major,
        "memory_gib": observed.memory_gib,
        "status": "complete",
        "system": observed.system,
    }


def _attest_offline_environment(
    expected: Mapping[str, str],
    observed: Mapping[str, str],
    failures: list[dict[str, str]],
) -> dict[str, object]:
    flags = sorted(expected)
    if any(observed.get(name) != value for name, value in expected.items()):
        failures.append(
            _failure(
                "offline_environment",
                "offline-policy",
                "drift",
                "offline_policy_mismatch",
            )
        )
        return {
            "diagnostic": "offline_policy_mismatch",
            "failure_kind": "drift",
            "flags": flags,
            "status": "failed",
        }
    return {"flags": flags, "status": "complete"}


def _attest_environments(
    project_root: Path,
    manifest: MlEnvironmentManifest,
    observations: Mapping[str, _ContractObservation],
    runtime_probe: Callable[[Path, str], RuntimeSnapshot],
    failures: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    reports: list[dict[str, object]] = []
    by_id: dict[str, dict[str, object]] = {}
    for environment in sorted(
        manifest.environments,
        key=lambda item: item.environment_id,
    ):
        identifier = environment.environment_id
        try:
            directory = _project_path(project_root, environment.directory)
            present = _environment_exists(directory)
        except _ProbeError as error:
            report = {
                "diagnostic": str(error),
                "failure_kind": "infrastructure",
                "id": identifier,
                "status": "failed",
            }
            failures.append(
                _failure("environment", identifier, "infrastructure", str(error))
            )
            reports.append(report)
            by_id[identifier] = report
            continue
        if not present:
            report = {
                "diagnostic": "environment_missing",
                "id": identifier,
                "status": "not_configured",
            }
            reports.append(report)
            by_id[identifier] = report
            continue
        dependency = observations[environment.dependency_contract_id]
        source_contract_id = environment.distribution_source.contract_id
        source_observation = (
            observations[source_contract_id]
            if source_contract_id is not None
            else None
        )
        failed_contracts = [
            item
            for item in (dependency, source_observation)
            if item is not None and item.status == "failed"
        ]
        if failed_contracts:
            kind: Literal["drift", "infrastructure"] = (
                "infrastructure"
                if any(item.failure_kind == "infrastructure" for item in failed_contracts)
                else "drift"
            )
            report = {
                "diagnostic": "dependency_contract_failed",
                "failure_kind": kind,
                "id": identifier,
                "status": "failed",
            }
            failures.append(
                _failure("environment", identifier, kind, "dependency_contract_failed")
            )
            reports.append(report)
            by_id[identifier] = report
            continue
        try:
            python = _python_executable(directory)
        except _ProbeError as error:
            report = {
                "diagnostic": str(error),
                "failure_kind": "infrastructure",
                "id": identifier,
                "status": "failed",
            }
            failures.append(
                _failure("environment", identifier, "infrastructure", str(error))
            )
            reports.append(report)
            by_id[identifier] = report
            continue
        try:
            snapshot = runtime_probe(python, identifier)
        except Exception:
            report = {
                "diagnostic": "environment_probe_failed",
                "failure_kind": "infrastructure",
                "id": identifier,
                "status": "failed",
            }
            failures.append(
                _failure(
                    "environment",
                    identifier,
                    "infrastructure",
                    "environment_probe_failed",
                )
            )
            reports.append(report)
            by_id[identifier] = report
            continue
        if snapshot.python_version != environment.python:
            diagnostic = "python_version_mismatch"
        elif (
            snapshot.implementation != "CPython"
            or snapshot.system != manifest.host.system
            or snapshot.machine.casefold() != manifest.host.machine.casefold()
            or (_macos_major(snapshot.macos_version) or 0)
            < manifest.host.minimum_macos_major
        ):
            diagnostic = "runtime_platform_mismatch"
        else:
            try:
                expected_distribution = _expected_distribution_identity(
                    environment.distribution_source,
                    observations,
                )
            except (MlEnvironmentManifestError, _ProbeError):
                diagnostic = "distribution_contract_failed"
            else:
                diagnostic = (
                    None
                    if hmac.compare_digest(
                        snapshot.distribution_identity,
                        expected_distribution,
                    )
                    else "distribution_identity_mismatch"
                )
        if diagnostic is not None:
            report = {
                "diagnostic": diagnostic,
                "failure_kind": "drift",
                "id": identifier,
                "status": "failed",
            }
            failures.append(_failure("environment", identifier, "drift", diagnostic))
        else:
            report = {
                "dependency_identity": "sha256:" + next(
                    contract.sha256
                    for contract in manifest.contracts
                    if contract.contract_id == environment.dependency_contract_id
                ),
                "distribution_identity": "sha256:" + snapshot.distribution_identity,
                "id": identifier,
                "python": snapshot.python_version,
                "status": "complete",
            }
        reports.append(report)
        by_id[identifier] = report
    return reports, by_id


def _attest_uv(
    project_root: Path,
    manifest: MlEnvironmentManifest,
    environment_reports: Mapping[str, Mapping[str, object]],
    probe: Callable[[Path], UvSnapshot],
    failures: list[dict[str, str]],
) -> dict[str, object]:
    environment_status = environment_reports[manifest.uv.environment_id]["status"]
    if environment_status == "not_configured":
        return {"diagnostic": "environment_missing", "status": "not_configured"}
    try:
        executable = _project_path(project_root, manifest.uv.path)
        snapshot = probe(executable)
    except Exception:
        failures.append(_failure("uv", "uv", "infrastructure", "uv_probe_failed"))
        return {
            "diagnostic": "uv_probe_failed",
            "failure_kind": "infrastructure",
            "status": "failed",
        }
    if snapshot.version != manifest.uv.version or not hmac.compare_digest(
        snapshot.sha256,
        manifest.uv.sha256,
    ):
        failures.append(_failure("uv", "uv", "drift", "uv_identity_mismatch"))
        return {
            "diagnostic": "uv_identity_mismatch",
            "failure_kind": "drift",
            "status": "failed",
        }
    return {
        "identity": "sha256:" + snapshot.sha256,
        "status": "complete",
        "version": snapshot.version,
    }


def _attest_capabilities(
    manifest: MlEnvironmentManifest,
    environment_reports: Mapping[str, Mapping[str, object]],
    observations: Mapping[str, _ContractObservation],
    failures: list[dict[str, str]],
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for capability in sorted(
        manifest.capabilities,
        key=lambda item: item.capability_id,
    ):
        environment_prerequisites = [
            environment_reports[identifier] for identifier in capability.environment_ids
        ]
        contract_prerequisites = [
            observations[identifier] for identifier in capability.contract_ids
        ]
        failed_kinds = [
            str(item.get("failure_kind"))
            for item in environment_prerequisites
            if item["status"] == "failed"
        ] + [
            str(item.failure_kind)
            for item in contract_prerequisites
            if item.status == "failed"
        ]
        if failed_kinds:
            kind: Literal["drift", "infrastructure"] = (
                "infrastructure" if "infrastructure" in failed_kinds else "drift"
            )
            report: dict[str, object] = {
                "diagnostic": "prerequisite_failed",
                "failure_kind": kind,
                "id": capability.capability_id,
                "status": "failed",
            }
            failures.append(
                _failure(
                    "capability",
                    capability.capability_id,
                    kind,
                    "prerequisite_failed",
                )
            )
        elif any(
            item["status"] == "not_configured" for item in environment_prerequisites
        ):
            report = {
                "diagnostic": "environment_not_configured",
                "id": capability.capability_id,
                "status": "not_configured",
            }
        else:
            report = {
                "id": capability.capability_id,
                "model_identities": list(capability.model_identities),
                "runtime_identity": capability.runtime_identity,
                "status": "complete",
            }
        reports.append(report)
    return reports


def attest_ml_environment(
    project_root: Path,
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    environ: Mapping[str, str] | None = None,
    host_probe: Callable[[], HostSnapshot] = _default_host_probe,
    runtime_probe: Callable[[Path, str], RuntimeSnapshot] | None = None,
    uv_probe: Callable[[Path], UvSnapshot] | None = None,
) -> dict[str, object]:
    """Attest the Phase-0 host/runtime contracts without loading a model."""

    manifest = load_ml_environment_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
    )
    observed_environment = os.environ if environ is None else environ
    selected_runtime_probe = runtime_probe or (
        lambda python, environment_id: _default_runtime_probe(
            python,
            environment_id,
            offline_environment=manifest.offline_environment,
        )
    )
    selected_uv_probe = uv_probe or (
        lambda executable: _default_uv_probe(
            executable,
            offline_environment=manifest.offline_environment,
        )
    )
    failures: list[dict[str, str]] = []
    host = _attest_host(manifest.host, host_probe, failures)
    offline = _attest_offline_environment(
        manifest.offline_environment,
        observed_environment,
        failures,
    )
    contracts, observations = _contract_observations(
        Path(project_root),
        manifest,
        failures,
    )
    environments, environments_by_id = _attest_environments(
        Path(project_root),
        manifest,
        observations,
        selected_runtime_probe,
        failures,
    )
    uv = _attest_uv(
        Path(project_root),
        manifest,
        environments_by_id,
        selected_uv_probe,
        failures,
    )
    capabilities = _attest_capabilities(
        manifest,
        environments_by_id,
        observations,
        failures,
    )
    has_not_configured = any(
        item["status"] == "not_configured"
        for item in (*environments, *capabilities)
    ) or uv["status"] == "not_configured"
    status = "failed" if failures else "partial" if has_not_configured else "complete"
    return {
        "attestation_id": manifest.attestation_id,
        "capabilities": capabilities,
        "contracts": contracts,
        "environments": environments,
        "failures": sorted(
            failures,
            key=lambda item: (item["scope"], item["id"], item["code"]),
        ),
        "host": host,
        "manifest_identity": "sha256:" + manifest.raw_sha256,
        "offline_environment": offline,
        "schema_version": ML_ENVIRONMENT_REPORT_SCHEMA_VERSION,
        "status": status,
        "uv": uv,
    }


def _manifest_failure_report(error: MlEnvironmentManifestError) -> dict[str, object]:
    return {
        "attestation_id": "phase0-environment-unavailable",
        "capabilities": [],
        "contracts": [],
        "environments": [],
        "failures": [
            _failure("manifest", "phase0-environment", error.kind, error.code)
        ],
        "host": {"status": "not_checked"},
        "manifest_identity": None,
        "offline_environment": {"status": "not_checked"},
        "schema_version": ML_ENVIRONMENT_REPORT_SCHEMA_VERSION,
        "status": "failed",
        "uv": {"status": "not_checked"},
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ml-attest-offline",
        description="Read-only Phase-0 ML environment attestation.",
    )
    parser.parse_args(argv)
    root = _project_root()
    try:
        report = attest_ml_environment(
            root,
            manifest_path=root / "workers" / "ml-environment.lock.json",
            expected_manifest_sha256=DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256,
        )
    except MlEnvironmentManifestError as error:
        report = _manifest_failure_report(error)
    except Exception:
        report = _manifest_failure_report(
            MlEnvironmentManifestError(
                "attestation_failed",
                kind="infrastructure",
            )
        )
    try:
        output = json.dumps(
            report,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError):
        output = json.dumps(
            _manifest_failure_report(
                MlEnvironmentManifestError(
                    "report_serialization_failed",
                    kind="infrastructure",
                )
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    sys.stdout.write(output + "\n")
    return 2 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256",
    "HostSnapshot",
    "ML_ENVIRONMENT_MANIFEST_SCHEMA_VERSION",
    "ML_ENVIRONMENT_REPORT_SCHEMA_VERSION",
    "MlEnvironmentManifest",
    "MlEnvironmentManifestError",
    "RuntimeSnapshot",
    "UvSnapshot",
    "attest_ml_environment",
    "canonical_json_sha256",
    "load_ml_environment_manifest",
    "main",
]
