from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
from hashlib import sha256
import hmac
import os
from pathlib import Path
import stat
import uuid
from typing import Iterator

from .schema import (
    BenchmarkDataError,
    BenchmarkDataset,
    BenchmarkRunManifest,
    _require_id,
    _validate_utc_timestamp,
)
from .serialization import (
    canonical_json_bytes,
    dataset_from_dict,
    dataset_to_dict,
    expect_fields,
    expect_list,
    expect_object,
    parse_json_object,
    run_from_dict,
    run_to_dict,
)


REGISTRY_SCHEMA_VERSION = 1
MAX_DATASET_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_RUN_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_REGISTRY_BYTES = 16 * 1024 * 1024


class BenchmarkDurabilityError(OSError):
    """An atomic commit is visible, but its directory durability sync failed."""


@dataclass(frozen=True, slots=True)
class RunRegistryEntry:
    run_id: str
    created_at: str
    code_sha: str
    dataset_revision: str
    execution_mode: str
    manifest_sha256: str
    manifest_path: str


def write_dataset(path: Path, dataset: BenchmarkDataset) -> None:
    """Atomically create a portable dataset manifest without replacing a file."""
    payload = canonical_json_bytes(dataset_to_dict(dataset))
    _write_exclusive_bytes(Path(path), payload)


def load_dataset(path: Path) -> BenchmarkDataset:
    payload = _read_bounded_file(
        Path(path),
        MAX_DATASET_MANIFEST_BYTES,
        "dataset manifest",
    )
    return dataset_from_dict(parse_json_object(payload, "dataset manifest"))


class BenchmarkRunRegistry:
    """Append immutable run manifests to a small integrity-checked local registry."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.runs_path = self.root / "runs"
        self.registry_path = self.root / "registry.json"
        self.lock_path = self.root / ".registry.lock"

    def add(self, run: BenchmarkRunManifest) -> RunRegistryEntry:
        with self._lock():
            entries = self._read_entries_unlocked()
            run_path = self.runs_path / run.run_id
            if _path_exists(run_path) or any(
                entry.run_id == run.run_id for entry in entries
            ):
                raise FileExistsError(f"benchmark run {run.run_id!r} already exists")

            manifest_bytes = canonical_json_bytes(run_to_dict(run))
            entry = _entry_for_run(run, manifest_bytes)
            publication_error: BenchmarkDurabilityError | None = None
            try:
                self._publish_run_directory(run_path, manifest_bytes)
            except BenchmarkDurabilityError as exc:
                publication_error = exc
            try:
                self._write_entries((*entries, entry))
            except BenchmarkDurabilityError:
                raise
            except Exception:
                _remove_unregistered_run(run_path)
                raise
            if publication_error is not None:
                raise publication_error
            return entry

    def list(self) -> tuple[RunRegistryEntry, ...]:
        with self._lock():
            return self._read_entries_unlocked()

    def read(self, run_id: str) -> BenchmarkRunManifest:
        _require_id(run_id, "run_id")
        with self._lock():
            entries = self._read_entries_unlocked()
            entry = next((item for item in entries if item.run_id == run_id), None)
            if entry is None:
                raise KeyError(run_id)
            expected_path = f"runs/{run_id}/manifest.json"
            if entry.manifest_path != expected_path:
                raise BenchmarkDataError("registry manifest path is not canonical")
            manifest_path = self._validated_manifest_path(run_id)
            payload = _read_bounded_file(
                manifest_path,
                MAX_RUN_MANIFEST_BYTES,
                "run manifest",
            )
            actual_digest = sha256(payload).hexdigest()
            if not hmac.compare_digest(actual_digest, entry.manifest_sha256):
                raise BenchmarkDataError(
                    f"benchmark run {run_id!r} manifest digest does not match the registry"
                )
            run = run_from_dict(parse_json_object(payload, "run manifest"))
            if run.run_id != run_id:
                raise BenchmarkDataError("run manifest identity does not match its registry entry")
            registry_metadata = (
                entry.created_at,
                entry.code_sha,
                entry.dataset_revision,
                entry.execution_mode,
            )
            manifest_metadata = (
                run.created_at,
                run.code_sha,
                run.dataset_revision,
                run.execution_mode,
            )
            if registry_metadata != manifest_metadata:
                raise BenchmarkDataError(
                    "run registry metadata does not match the immutable manifest"
                )
            return run

    def rebuild(self) -> tuple[RunRegistryEntry, ...]:
        """Rebuild the derived registry from validated immutable run manifests."""
        with self._lock():
            self.runs_path.mkdir(parents=True, exist_ok=True)
            self._validate_layout()
            existing_entries: tuple[RunRegistryEntry, ...] = ()
            registry_is_valid = False
            if self.registry_path.exists():
                try:
                    existing_entries = self._read_entries_unlocked()
                except BenchmarkDataError:
                    pass
                else:
                    registry_is_valid = True
            existing_by_id = {entry.run_id: entry for entry in existing_entries}
            entries: list[RunRegistryEntry] = []
            for run_path in sorted(self.runs_path.iterdir(), key=lambda path: path.name):
                if run_path.name.startswith(".") and run_path.name.endswith(".tmp"):
                    continue
                _require_id(run_path.name, "run directory name")
                manifest_path = self._validated_manifest_path(run_path.name)
                payload = _read_bounded_file(
                    manifest_path,
                    MAX_RUN_MANIFEST_BYTES,
                    "run manifest",
                )
                run = run_from_dict(parse_json_object(payload, "run manifest"))
                if run.run_id != run_path.name:
                    raise BenchmarkDataError(
                        "run manifest identity does not match its directory"
                    )
                entry = _entry_for_run(run, payload)
                previous = existing_by_id.get(entry.run_id)
                if previous is not None and not hmac.compare_digest(
                    previous.manifest_sha256,
                    entry.manifest_sha256,
                ):
                    raise BenchmarkDataError(
                        f"immutable run manifest {entry.run_id!r} was modified"
                    )
                entries.append(entry)
            ordered = tuple(sorted(entries, key=lambda entry: entry.run_id))
            if registry_is_valid:
                recovered_ids = {entry.run_id for entry in ordered}
                missing_ids = sorted(set(existing_by_id) - recovered_ids)
                if missing_ids:
                    raise BenchmarkDataError(
                        "run registry references missing immutable runs: "
                        + ", ".join(missing_ids)
                    )
            self._write_entries(ordered)
            return ordered

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        self._validate_layout()
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.lock_path, flags, 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise BenchmarkDataError(
                    "benchmark registry lock must not be a symbolic link"
                ) from exc
            raise
        with os.fdopen(descriptor, "a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self._validate_layout()
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read_entries_unlocked(self) -> tuple[RunRegistryEntry, ...]:
        if not self.registry_path.exists():
            return ()
        value = parse_json_object(
            _read_bounded_file(
                self.registry_path,
                MAX_REGISTRY_BYTES,
                "run registry",
            ),
            "run registry",
        )
        expect_fields(value, {"schema_version", "runs"}, "run registry")
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != REGISTRY_SCHEMA_VERSION
        ):
            raise BenchmarkDataError(
                f"run registry schema_version must be {REGISTRY_SCHEMA_VERSION}"
            )
        entries = tuple(
            _entry_from_dict(expect_object(item, f"run registry runs[{index}]"))
            for index, item in enumerate(expect_list(value["runs"], "run registry runs"))
        )
        run_ids = tuple(entry.run_id for entry in entries)
        if len(run_ids) != len(set(run_ids)):
            raise BenchmarkDataError("run registry contains duplicate run ids")
        if run_ids != tuple(sorted(run_ids)):
            raise BenchmarkDataError("run registry entries are not in canonical order")
        return entries

    def _write_entries(self, entries: tuple[RunRegistryEntry, ...]) -> None:
        ordered = tuple(sorted(entries, key=lambda entry: entry.run_id))
        value = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "runs": [_entry_to_dict(entry) for entry in ordered],
        }
        _atomic_replace_bytes(self.registry_path, canonical_json_bytes(value))

    def _publish_run_directory(self, destination: Path, manifest_bytes: bytes) -> None:
        self.runs_path.mkdir(parents=True, exist_ok=True)
        self._validate_layout()
        staging = self.runs_path / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        staging.mkdir(mode=0o700)
        published = False
        try:
            _write_new_file(staging / "manifest.json", manifest_bytes)
            _fsync_directory(staging)
            os.replace(staging, destination)
            published = True
            try:
                _fsync_directory(self.runs_path)
            except OSError as exc:
                raise BenchmarkDurabilityError(
                    "run manifest was committed but its directory sync failed"
                ) from exc
        except Exception:
            if not published and staging.exists():
                manifest = staging / "manifest.json"
                if manifest.exists():
                    manifest.unlink()
                staging.rmdir()
            raise

    def _validate_layout(self) -> None:
        if self.root.is_symlink() or not self.root.is_dir():
            raise BenchmarkDataError(
                "benchmark registry root must be a real directory, not a symbolic link"
            )
        root = self.root.resolve(strict=True)
        _validate_internal_path(self.runs_path, root, expected="directory")
        _validate_internal_path(self.registry_path, root, expected="file")
        _validate_internal_path(self.lock_path, root, expected="file")

    def _validated_manifest_path(self, run_id: str) -> Path:
        run_path = self.runs_path / run_id
        root = self.root.resolve(strict=True)
        _validate_internal_path(run_path, root, expected="directory", required=True)
        manifest_path = run_path / "manifest.json"
        _validate_internal_path(manifest_path, root, expected="file", required=True)
        return manifest_path


def _entry_to_dict(entry: RunRegistryEntry) -> dict[str, object]:
    return {
        "run_id": entry.run_id,
        "created_at": entry.created_at,
        "code_sha": entry.code_sha,
        "dataset_revision": entry.dataset_revision,
        "execution_mode": entry.execution_mode,
        "manifest_sha256": entry.manifest_sha256,
        "manifest_path": entry.manifest_path,
    }


def _entry_for_run(run: BenchmarkRunManifest, payload: bytes) -> RunRegistryEntry:
    return RunRegistryEntry(
        run_id=run.run_id,
        created_at=run.created_at,
        code_sha=run.code_sha,
        dataset_revision=run.dataset_revision,
        execution_mode=run.execution_mode,
        manifest_sha256=sha256(payload).hexdigest(),
        manifest_path=f"runs/{run.run_id}/manifest.json",
    )


def _entry_from_dict(value: dict[str, object]) -> RunRegistryEntry:
    fields = {
        "run_id",
        "created_at",
        "code_sha",
        "dataset_revision",
        "execution_mode",
        "manifest_sha256",
        "manifest_path",
    }
    expect_fields(value, fields, "run registry entry")
    try:
        entry = RunRegistryEntry(**value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise BenchmarkDataError("run registry entry has invalid field types") from exc
    _require_id(entry.run_id, "registry run_id")
    if entry.execution_mode not in {"cold", "warm"}:
        raise BenchmarkDataError("registry execution_mode must be cold or warm")
    for field_name in ("code_sha", "dataset_revision", "manifest_sha256"):
        digest = getattr(entry, field_name)
        expected_length = (40, 64) if field_name == "code_sha" else (64,)
        if (
            not isinstance(digest, str)
            or len(digest) not in expected_length
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise BenchmarkDataError(f"registry {field_name} is invalid")
    if entry.manifest_path != f"runs/{entry.run_id}/manifest.json":
        raise BenchmarkDataError("registry manifest path is not canonical")
    _validate_utc_timestamp(entry.created_at)
    return entry


def _write_exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        _write_new_file(temporary, payload)
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite {path}") from None
        try:
            _fsync_directory(path.parent)
        except OSError as exc:
            raise BenchmarkDurabilityError(
                f"{path.name} was committed but its directory sync failed"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        _write_new_file(temporary, payload)
        os.replace(temporary, path)
        try:
            _fsync_directory(path.parent)
        except OSError as exc:
            raise BenchmarkDurabilityError(
                f"{path.name} was committed but its directory sync failed"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_new_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _remove_unregistered_run(path: Path) -> None:
    manifest = path / "manifest.json"
    if manifest.exists():
        manifest.unlink()
    if path.exists():
        path.rmdir()


def _validate_internal_path(
    path: Path,
    root: Path,
    *,
    expected: str,
    required: bool = False,
) -> None:
    if path.is_symlink():
        raise BenchmarkDataError(
            f"benchmark registry {path.name!r} must not be a symbolic link"
        )
    if not path.exists():
        if required:
            raise BenchmarkDataError(f"benchmark registry {path.name!r} is missing")
        return
    if expected == "directory" and not path.is_dir():
        raise BenchmarkDataError(f"benchmark registry {path.name!r} must be a directory")
    if expected == "file" and not path.is_file():
        raise BenchmarkDataError(f"benchmark registry {path.name!r} must be a regular file")
    if not path.resolve(strict=True).is_relative_to(root):
        raise BenchmarkDataError(f"benchmark registry {path.name!r} escapes its root")


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _read_bounded_file(path: Path, maximum_bytes: int, context: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise BenchmarkDataError(f"{context} must not be a symbolic link") from exc
        raise
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise BenchmarkDataError(f"{context} must be a regular file")
        if metadata.st_size > maximum_bytes:
            raise BenchmarkDataError(
                f"{context} exceeds the {maximum_bytes}-byte limit"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(maximum_bytes + 1)
    finally:
        os.close(descriptor)
    if len(payload) > maximum_bytes:
        raise BenchmarkDataError(f"{context} exceeds the {maximum_bytes}-byte limit")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
