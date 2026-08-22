from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from threading import Lock
from typing import Literal, Self

from videoscope.repository import Repository
from videoscope.runtime_lifecycle import ExclusiveRuntimeLock


_SNAPSHOT_PROTOCOL_VERSION = 1
_READ_CHUNK_BYTES = 1024 * 1024
_DESTINATION_DATABASE_NAME = "product.sqlite3"
_DEFAULT_MAX_ENTRIES = 3
_DEFAULT_MAX_FILE_BYTES = 16 * 1024 * 1024 * 1024
_DEFAULT_MAX_TOTAL_BYTES = 32 * 1024 * 1024 * 1024
_MAX_CLEANUP_ENTRIES = 32
_OPEN_CLEANUP_ATTEMPTS = 3
_SHA256_LENGTH = 64


class ProductSnapshotError(RuntimeError):
    """A product database cannot be opened as a verified private snapshot."""


class ProductSnapshotCleanupError(ProductSnapshotError):
    """A failed open still owns resources whose cleanup can be retried safely."""

    def __init__(
        self,
        primary_error: BaseException,
        cleanup: _OpenFailureCleanup,
    ) -> None:
        self._cleanup = cleanup
        super().__init__(
            f"{_snapshot_error(primary_error)}; product snapshot cleanup remains pending"
        )

    @property
    def cleanup_pending(self) -> bool:
        return not self._cleanup.is_closed

    def retry_cleanup(self) -> None:
        try:
            self._cleanup.close_with_retries(_OPEN_CLEANUP_ATTEMPTS)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise ProductSnapshotError(
                "product snapshot cleanup remains pending"
            ) from error


@dataclass(frozen=True, slots=True)
class ProductSnapshotLimits:
    max_entries: int = _DEFAULT_MAX_ENTRIES
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES
    max_total_bytes: int = _DEFAULT_MAX_TOTAL_BYTES

    def __post_init__(self) -> None:
        for name in ("max_entries", "max_file_bytes", "max_total_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("snapshot per-file byte limit exceeds the total byte limit")


@dataclass(frozen=True, slots=True)
class ProductSnapshotFileIdentity:
    role: Literal["database", "wal", "shm"]
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.role not in {"database", "wal", "shm"}:
            raise ValueError("product snapshot file role is invalid")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("product snapshot file size is invalid")
        if (
            type(self.sha256) is not str
            or len(self.sha256) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("product snapshot file digest must be lowercase SHA-256")

    @property
    def canonical_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ProductSnapshotIdentity:
    journal_mode: Literal["delete", "wal"]
    files: tuple[ProductSnapshotFileIdentity, ...]
    total_bytes: int
    protocol_version: int = _SNAPSHOT_PROTOCOL_VERSION
    snapshot_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.journal_mode not in {"delete", "wal"}:
            raise ValueError("product snapshot journal mode is invalid")
        if not isinstance(self.files, tuple) or any(
            not isinstance(item, ProductSnapshotFileIdentity) for item in self.files
        ):
            raise ValueError("product snapshot files are incomplete or unordered")
        roles = tuple(item.role for item in self.files)
        roles_are_valid = (
            roles == ("database",)
            if self.journal_mode == "delete"
            else roles in {("database",), ("database", "wal", "shm")}
        )
        if not roles_are_valid:
            raise ValueError("product snapshot files are incomplete or unordered")
        if (
            isinstance(self.total_bytes, bool)
            or not isinstance(self.total_bytes, int)
            or self.total_bytes != sum(item.size_bytes for item in self.files)
        ):
            raise ValueError("product snapshot total byte count is invalid")
        if self.protocol_version != _SNAPSHOT_PROTOCOL_VERSION:
            raise ValueError("product snapshot protocol version is unsupported")
        object.__setattr__(self, "snapshot_sha256", _canonical_sha256(self.content_dict))

    @property
    def content_dict(self) -> dict[str, object]:
        return {
            "files": [item.canonical_dict for item in self.files],
            "journal_mode": self.journal_mode,
            "protocol_version": self.protocol_version,
            "total_bytes": self.total_bytes,
        }

    @property
    def canonical_dict(self) -> dict[str, object]:
        return {
            **self.content_dict,
            "snapshot_sha256": self.snapshot_sha256,
        }


@dataclass(frozen=True, slots=True)
class _FileState:
    device: int
    inode: int
    mode: int
    links: int
    owner: int
    group: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _SourceRead:
    state: _FileState
    size_bytes: int
    sha256: str
    header: bytes


def _file_state(metadata: os.stat_result) -> _FileState:
    return _FileState(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        links=metadata.st_nlink,
        owner=metadata.st_uid,
        group=metadata.st_gid,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _stable_object_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
    )


def _object_binding_matches(
    observed: tuple[int, ...],
    expected: tuple[int, ...],
) -> bool:
    # Some filesystems vary a directory's link count with contained entries,
    # and an already-unlinked retained directory may report another value.
    # Device/inode/type+mode/owner remain the authoritative opened binding.
    return observed[:3] + observed[4:] == expected[:3] + expected[4:]


def _required_open_flags(*, directory: bool = False) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise ProductSnapshotError("platform does not support no-follow snapshot access")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    if directory:
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        if not directory_flag:
            raise ProductSnapshotError(
                "platform does not support descriptor-relative directory access"
            )
        flags |= directory_flag
    else:
        flags |= getattr(os, "O_NONBLOCK", 0)
    return flags


def _absolute_path(path: Path, *, label: str) -> Path:
    try:
        value = os.fspath(path)
        if not value or "\x00" in value:
            raise ValueError
        return Path(os.path.abspath(value))
    except (TypeError, ValueError, OSError) as error:
        raise ProductSnapshotError(f"{label} path is invalid") from error


def _open_directory_path(path: Path, *, label: str) -> tuple[Path, int]:
    absolute = _absolute_path(path, label=label)
    flags = _required_open_flags(directory=True)
    try:
        descriptor = os.open(absolute.anchor, flags)
    except OSError as error:
        raise ProductSnapshotError(f"{label} directory is unsafe or unavailable") from error
    try:
        for component in absolute.parts[1:]:
            child: int | None = None
            try:
                observed = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISDIR(observed.st_mode):
                    raise ProductSnapshotError(
                        f"{label} path contains an unsafe non-directory component"
                    )
                child = os.open(component, flags, dir_fd=descriptor)
                opened = os.fstat(child)
                current = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except ProductSnapshotError:
                if child is not None:
                    os.close(child)
                raise
            except OSError as error:
                if child is not None:
                    os.close(child)
                raise ProductSnapshotError(
                    f"{label} directory is unsafe or unavailable"
                ) from error
            if not (
                stat.S_ISDIR(opened.st_mode)
                and _stable_object_identity(observed)
                == _stable_object_identity(opened)
                == _stable_object_identity(current)
            ):
                os.close(child)
                raise ProductSnapshotError(f"{label} path changed while it was opened")
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ProductSnapshotError(f"{label} path is not a directory")
        return absolute, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _retained_directory_path(descriptor: int) -> Path:
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ProductSnapshotError("retained product directory is invalid")
        retained = Path(f"/.vol/{metadata.st_dev}/{metadata.st_ino}")
        verifier = os.open(retained, _required_open_flags(directory=True))
        try:
            current = os.fstat(verifier)
        finally:
            os.close(verifier)
    except ProductSnapshotError:
        raise
    except OSError as error:
        raise ProductSnapshotError(
            "stable retained product directory is unavailable"
        ) from error
    if _stable_object_identity(metadata) != _stable_object_identity(current):
        raise ProductSnapshotError("retained product directory identity changed")
    return retained


def _open_relative_directory(
    root_descriptor: int,
    components: tuple[str, ...],
    *,
    label: str,
) -> int:
    descriptor = os.dup(root_descriptor)
    flags = _required_open_flags(directory=True)
    try:
        for component in components:
            child: int | None = None
            try:
                observed = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISDIR(observed.st_mode):
                    raise ProductSnapshotError(f"{label} path is unsafe")
                child = os.open(component, flags, dir_fd=descriptor)
                opened = os.fstat(child)
                current = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except ProductSnapshotError:
                if child is not None:
                    os.close(child)
                raise
            except OSError as error:
                if child is not None:
                    os.close(child)
                raise ProductSnapshotError(f"{label} path is unsafe or unavailable") from error
            if not (
                stat.S_ISDIR(opened.st_mode)
                and _stable_object_identity(observed)
                == _stable_object_identity(opened)
                == _stable_object_identity(current)
            ):
                os.close(child)
                raise ProductSnapshotError(f"{label} path changed while it was opened")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _relative_contained_path(
    path: Path,
    root: Path,
    *,
    label: str,
    allow_root: bool,
) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ProductSnapshotError(f"{label} path must be contained in product data") from error
    if not allow_root and relative == Path("."):
        raise ProductSnapshotError(f"{label} path must name a contained file")
    if any(component in {"", ".", ".."} for component in relative.parts):
        raise ProductSnapshotError(f"{label} path is invalid")
    return relative


def _validate_lock_binding(
    runtime_lock: ExclusiveRuntimeLock,
    data_descriptor: int,
) -> None:
    lock_descriptor = getattr(runtime_lock, "_file_descriptor", None)
    if not isinstance(lock_descriptor, int):
        raise ProductSnapshotError("runtime lock ownership could not be verified")
    try:
        retained = os.fstat(lock_descriptor)
        current = os.stat(
            ExclusiveRuntimeLock.filename,
            dir_fd=data_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ProductSnapshotError("runtime lock ownership changed after acquisition") from error
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or _stable_object_identity(retained) != _stable_object_identity(current)
    ):
        raise ProductSnapshotError("runtime lock does not belong to product data")


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise ProductSnapshotError("product snapshot write made no progress")
        remaining = remaining[written:]


def _read_regular_source(
    parent_descriptor: int,
    name: str,
    *,
    max_bytes: int,
    remaining_total_bytes: int | None = None,
    output_parent_descriptor: int | None = None,
    output_name: str | None = None,
) -> _SourceRead:
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        raise ProductSnapshotError("product database entry name is invalid")
    try:
        observed_metadata = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ProductSnapshotError("product database file is unavailable") from error
    observed = _file_state(observed_metadata)
    if not stat.S_ISREG(observed.mode) or observed.links != 1:
        raise ProductSnapshotError(
            "product database files must be regular single-link files"
        )
    if observed.size > max_bytes:
        raise ProductSnapshotError("product database exceeds the per-file byte limit")
    if remaining_total_bytes is not None and observed.size > remaining_total_bytes:
        raise ProductSnapshotError("product snapshot exceeds the total byte limit")

    source_descriptor: int | None = None
    output_descriptor: int | None = None
    try:
        source_descriptor = os.open(
            name,
            _required_open_flags(),
            dir_fd=parent_descriptor,
        )
        opened = _file_state(os.fstat(source_descriptor))
        if opened != observed:
            raise ProductSnapshotError("product database changed while it was opened")
        if output_parent_descriptor is not None:
            if output_name is None:
                raise ProductSnapshotError("product snapshot output name is missing")
            output_descriptor = os.open(
                output_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=output_parent_descriptor,
            )
            os.fchmod(output_descriptor, 0o600)

        digest = hashlib.sha256()
        consumed = 0
        header = bytearray()
        while True:
            remaining_budget = max_bytes - consumed
            block = os.read(
                source_descriptor,
                min(_READ_CHUNK_BYTES, remaining_budget + 1),
            )
            if not block:
                break
            consumed += len(block)
            if consumed > max_bytes:
                raise ProductSnapshotError(
                    "product database exceeds the per-file byte limit"
                )
            digest.update(block)
            if len(header) < 100:
                header.extend(block[: 100 - len(header)])
            if output_descriptor is not None:
                _write_all(output_descriptor, block)

        after = _file_state(os.fstat(source_descriptor))
        current = _file_state(
            os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        )
        if opened != after or opened != current or consumed != opened.size:
            raise ProductSnapshotError("product database changed while it was read")
        if output_descriptor is not None:
            os.fsync(output_descriptor)
            os.fsync(output_parent_descriptor)
        return _SourceRead(
            state=opened,
            size_bytes=consumed,
            sha256=digest.hexdigest(),
            header=bytes(header),
        )
    except ProductSnapshotError:
        raise
    except OSError as error:
        raise ProductSnapshotError("product database could not be read or copied") from error
    finally:
        if output_descriptor is not None:
            os.close(output_descriptor)
        if source_descriptor is not None:
            os.close(source_descriptor)


def _journal_mode(database: _SourceRead) -> Literal["delete", "wal"]:
    header = database.header
    if len(header) < 100 or header[:16] != b"SQLite format 3\x00":
        raise ProductSnapshotError("product database is not a valid SQLite database")
    if header[18] not in {1, 2} or header[19] not in {1, 2}:
        raise ProductSnapshotError("product database has an invalid journal format")
    return "wal" if 2 in {header[18], header[19]} else "delete"


def _entry_exists(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ProductSnapshotError("product database sidecar could not be inspected") from error
    return True


def _create_owned_scratch(
    parent_descriptor: int,
    *,
    parent_path: Path,
) -> tuple[str, Path, int, tuple[int, ...]]:
    for _attempt in range(32):
        name = f".videoscope-product-{secrets.token_hex(16)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        except OSError as error:
            raise ProductSnapshotError("private product scratch could not be created") from error
        descriptor: int | None = None
        try:
            observed = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            descriptor = os.open(
                name,
                _required_open_flags(directory=True),
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(descriptor)
            current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            if not (
                stat.S_ISDIR(opened.st_mode)
                and opened.st_uid == os.geteuid()
                and stat.S_IMODE(opened.st_mode) == 0o700
                and _stable_object_identity(observed)
                == _stable_object_identity(opened)
                == _stable_object_identity(current)
            ):
                raise ProductSnapshotError("private product scratch is not owner-only")
            os.fsync(parent_descriptor)
            return (
                name,
                parent_path / name,
                descriptor,
                _stable_object_identity(opened),
            )
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.rmdir(name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except OSError:
                pass
            raise
    raise ProductSnapshotError("private product scratch name could not be allocated")


def _bounded_directory_names(
    descriptor: int,
    *,
    limit: int,
    list_error: str,
    overflow_error: str,
) -> tuple[str, ...]:
    names: list[str] = []
    try:
        with os.scandir(descriptor) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > limit:
                    raise ProductSnapshotError(overflow_error)
    except ProductSnapshotError:
        raise
    except OSError as error:
        raise ProductSnapshotError(list_error) from error
    return tuple(sorted(names))


def _clear_owned_directory(descriptor: int, *, budget: list[int]) -> None:
    remaining = _MAX_CLEANUP_ENTRIES - budget[0]
    names = _bounded_directory_names(
        descriptor,
        limit=remaining,
        list_error="owned product scratch could not be listed",
        overflow_error="owned product scratch cleanup exceeds its bound",
    )
    for name in names:
        if not name or name in {".", ".."} or "/" in name or "\x00" in name:
            raise ProductSnapshotError("owned product scratch contains an invalid entry")
        budget[0] += 1
        if budget[0] > _MAX_CLEANUP_ENTRIES:
            raise ProductSnapshotError("owned product scratch cleanup exceeds its bound")
        try:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(
                    name,
                    _required_open_flags(directory=True),
                    dir_fd=descriptor,
                )
                try:
                    opened = os.fstat(child)
                    current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if not (
                        _stable_object_identity(metadata)
                        == _stable_object_identity(opened)
                        == _stable_object_identity(current)
                    ):
                        raise ProductSnapshotError(
                            "owned product scratch changed during cleanup"
                        )
                    _clear_owned_directory(child, budget=budget)
                finally:
                    os.close(child)
                os.rmdir(name, dir_fd=descriptor)
            else:
                os.unlink(name, dir_fd=descriptor)
        except ProductSnapshotError:
            raise
        except OSError as error:
            raise ProductSnapshotError("owned product scratch could not be removed") from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise ProductSnapshotError("owned product scratch could not be synchronized") from error


def _remove_owned_scratch(
    parent_descriptor: int,
    scratch_descriptor: int,
    scratch_name: str,
    scratch_identity: tuple[int, ...],
    owned_entries: frozenset[str],
) -> None:
    try:
        retained = os.fstat(scratch_descriptor)
        current = os.stat(
            scratch_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        try:
            retained = os.fstat(scratch_descriptor)
        except OSError as error:
            raise ProductSnapshotError(
                "owned product scratch binding is unavailable"
            ) from error
        if not _object_binding_matches(
            _stable_object_identity(retained), scratch_identity
        ):
            raise ProductSnapshotError(
                "owned product scratch binding changed before removal"
            )
        try:
            os.fsync(parent_descriptor)
        except OSError as error:
            raise ProductSnapshotError(
                "owned product scratch root could not be removed"
            ) from error
        return
    except OSError as error:
        raise ProductSnapshotError("owned product scratch binding is unavailable") from error
    if not (
        _object_binding_matches(_stable_object_identity(retained), scratch_identity)
        and _object_binding_matches(_stable_object_identity(current), scratch_identity)
    ):
        raise ProductSnapshotError("owned product scratch binding changed before removal")
    current_entries = frozenset(
        _bounded_directory_names(
            scratch_descriptor,
            limit=_MAX_CLEANUP_ENTRIES,
            list_error="owned product scratch could not be listed",
            overflow_error="owned product scratch cleanup exceeds its bound",
        )
    )
    if not current_entries <= owned_entries:
        raise ProductSnapshotError(
            "product scratch contains caller-managed entries; close them first"
        )
    _clear_owned_directory(scratch_descriptor, budget=[0])
    try:
        retained_after = os.fstat(scratch_descriptor)
        current = os.stat(
            scratch_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ProductSnapshotError("owned product scratch binding is unavailable") from error
    if not (
        _stable_object_identity(retained_after) == scratch_identity
        and _stable_object_identity(current) == scratch_identity
    ):
        raise ProductSnapshotError("owned product scratch binding changed before removal")
    try:
        os.rmdir(scratch_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    except OSError as error:
        raise ProductSnapshotError("owned product scratch root could not be removed") from error


def _close_repository(repository: Repository) -> None:
    close = getattr(repository, "close", None)
    if close is not None:
        if not callable(close):
            raise ProductSnapshotError("read-only repository close contract is invalid")
        close()


def _snapshot_error(error: BaseException) -> ProductSnapshotError:
    if isinstance(error, ProductSnapshotError):
        return error
    message = str(error).strip() or "product snapshot operation failed"
    return ProductSnapshotError(message)


class _OpenFailureCleanup:
    """Retryable owner for resources acquired before snapshot construction failed."""

    def __init__(
        self,
        *,
        repository: Repository | None,
        runtime_lock: ExclusiveRuntimeLock,
        lock_acquired: bool,
        scratch_parent_descriptor: int | None,
        scratch_descriptor: int | None,
        scratch_name: str | None,
        scratch_identity: tuple[int, ...] | None,
        retained_descriptors: list[int],
    ) -> None:
        scratch_binding_values = (
            scratch_descriptor,
            scratch_name,
            scratch_identity,
        )
        if (
            any(value is not None for value in scratch_binding_values)
            and (
                scratch_parent_descriptor is None
                or not all(value is not None for value in scratch_binding_values)
            )
        ):
            raise ProductSnapshotError("product scratch cleanup state is incomplete")
        self._repository = repository
        self._runtime_lock = runtime_lock
        self._lock_acquired = lock_acquired
        self._scratch_parent_descriptor = scratch_parent_descriptor
        self._scratch_descriptor = scratch_descriptor
        self._scratch_name = scratch_name
        self._scratch_identity = scratch_identity
        self._repository_closed = repository is None
        self._scratch_removed = scratch_descriptor is None
        self._descriptors_to_close = list(retained_descriptors)
        if scratch_descriptor is not None:
            self._descriptors_to_close.insert(0, scratch_descriptor)
        if scratch_parent_descriptor is not None:
            insertion_index = 1 if scratch_descriptor is not None else 0
            self._descriptors_to_close.insert(insertion_index, scratch_parent_descriptor)
        self._descriptors_closed = not self._descriptors_to_close
        self._lock_released = not lock_acquired
        self._state_lock = Lock()

    @property
    def is_closed(self) -> bool:
        return (
            self._repository_closed
            and self._scratch_removed
            and self._descriptors_closed
            and self._lock_released
        )

    def _close_descriptors(self) -> None:
        while self._descriptors_to_close:
            descriptor = self._descriptors_to_close[0]
            try:
                os.close(descriptor)
            except OSError as error:
                raise ProductSnapshotError(
                    "product snapshot descriptors could not be closed"
                ) from error
            self._descriptors_to_close.pop(0)
        self._descriptors_closed = True

    def close(self) -> None:
        with self._state_lock:
            if self.is_closed:
                return
            if not self._repository_closed:
                assert self._repository is not None
                _close_repository(self._repository)
                self._repository_closed = True
            if not self._scratch_removed:
                assert self._scratch_parent_descriptor is not None
                assert self._scratch_descriptor is not None
                assert self._scratch_name is not None
                assert self._scratch_identity is not None
                _remove_owned_scratch(
                    self._scratch_parent_descriptor,
                    self._scratch_descriptor,
                    self._scratch_name,
                    self._scratch_identity,
                    frozenset(
                        {
                            _DESTINATION_DATABASE_NAME,
                            f"{_DESTINATION_DATABASE_NAME}-wal",
                            f"{_DESTINATION_DATABASE_NAME}-shm",
                        }
                    ),
                )
                self._scratch_removed = True
            if not self._descriptors_closed:
                self._close_descriptors()
            if not self._lock_released:
                self._runtime_lock.close()
                self._lock_released = True

    def close_with_retries(self, attempts: int) -> None:
        last_error: BaseException | None = None
        for _attempt in range(attempts):
            try:
                self.close()
                return
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                last_error = error
        assert last_error is not None
        raise _snapshot_error(last_error) from last_error


class ProductRuntimeSnapshot:
    """A lock-fenced, read-only repository backed by a private database copy.

    ``scratch_root`` is exclusively owned by this snapshot and is not a general
    benchmark workspace. Callers must put Qdrant, model, and other temporary
    resources elsewhere, or remove them before ``close``. Unexpected entries
    fail closed while retaining the product lock; a later ``close`` can retry.
    """

    def __init__(
        self,
        *,
        repository: Repository,
        media_root: Path,
        scratch_root: Path,
        identity: ProductSnapshotIdentity,
        runtime_lock: ExclusiveRuntimeLock,
        scratch_parent_descriptor: int,
        scratch_descriptor: int,
        scratch_name: str,
        scratch_identity: tuple[int, ...],
        data_descriptor: int,
        media_descriptor: int,
        retained_descriptors: tuple[int, ...],
    ) -> None:
        self.repository = repository
        self.media_root = media_root
        self.scratch_root = scratch_root
        self.identity = identity
        self._runtime_lock = runtime_lock
        self._scratch_parent_descriptor = scratch_parent_descriptor
        self._scratch_descriptor = scratch_descriptor
        self._scratch_name = scratch_name
        self._scratch_identity = scratch_identity
        self._data_descriptor = data_descriptor
        self._media_descriptor = media_descriptor
        self._data_identity = _stable_object_identity(os.fstat(data_descriptor))
        self._media_identity = _stable_object_identity(os.fstat(media_descriptor))
        self._owned_scratch_entries = frozenset(
            {
                "database": _DESTINATION_DATABASE_NAME,
                "wal": f"{_DESTINATION_DATABASE_NAME}-wal",
                "shm": f"{_DESTINATION_DATABASE_NAME}-shm",
            }[item.role]
            for item in identity.files
        )
        self._descriptors_to_close = [
            scratch_descriptor,
            scratch_parent_descriptor,
            *retained_descriptors,
        ]
        self._repository_closed = False
        self._scratch_removed = False
        self._descriptors_closed = False
        self._lock_released = False
        self._state_lock = Lock()

    @property
    def is_closed(self) -> bool:
        return self._lock_released

    def __enter__(self) -> Self:
        if self.is_closed:
            raise ProductSnapshotError("product snapshot is already closed")
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def _duplicate_bound_directory(
        self,
        descriptor: int,
        expected_identity: tuple[int, ...],
    ) -> int:
        with self._state_lock:
            if self._repository_closed or self._descriptors_closed:
                raise ProductSnapshotError("product snapshot directory is already closed")
            duplicate: int | None = None
            try:
                current = os.fstat(descriptor)
                duplicate = os.dup(descriptor)
                copied = os.fstat(duplicate)
            except OSError as error:
                if duplicate is not None:
                    os.close(duplicate)
                raise ProductSnapshotError(
                    "product snapshot directory capability is unavailable"
                ) from error
            if not (
                stat.S_ISDIR(current.st_mode)
                and _object_binding_matches(
                    _stable_object_identity(current), expected_identity
                )
                and _stable_object_identity(current)
                == _stable_object_identity(copied)
            ):
                os.close(duplicate)
                raise ProductSnapshotError(
                    "product snapshot directory capability changed"
                )
            return duplicate

    def duplicate_data_root_descriptor(self) -> int:
        """Return an owned fd bound to the exact data root that holds the lock."""
        return self._duplicate_bound_directory(
            self._data_descriptor,
            self._data_identity,
        )

    def duplicate_media_root_descriptor(self) -> int:
        """Return an owned fd bound to the exact media directory in the snapshot."""
        return self._duplicate_bound_directory(
            self._media_descriptor,
            self._media_identity,
        )

    def _close_retained_descriptors(self) -> None:
        while self._descriptors_to_close:
            descriptor = self._descriptors_to_close[0]
            try:
                os.close(descriptor)
            except OSError as error:
                raise ProductSnapshotError(
                    "product snapshot descriptors could not be closed"
                ) from error
            self._descriptors_to_close.pop(0)
        self._descriptors_closed = True

    def close(self) -> None:
        with self._state_lock:
            if self._lock_released:
                return
            try:
                if not self._repository_closed:
                    _close_repository(self.repository)
                    self._repository_closed = True
                if not self._scratch_removed:
                    _remove_owned_scratch(
                        self._scratch_parent_descriptor,
                        self._scratch_descriptor,
                        self._scratch_name,
                        self._scratch_identity,
                        self._owned_scratch_entries,
                    )
                    self._scratch_removed = True
                if not self._descriptors_closed:
                    self._close_retained_descriptors()
                self._runtime_lock.close()
                self._lock_released = True
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise _snapshot_error(error) from error


def _close_descriptors(descriptors: list[int]) -> None:
    while descriptors:
        descriptor = descriptors.pop()
        try:
            os.close(descriptor)
        except OSError:
            pass


def open_product_runtime_snapshot(
    *,
    data_dir: Path,
    database_path: Path,
    media_root: Path,
    scratch_parent: Path,
    limits: ProductSnapshotLimits | None = None,
) -> ProductRuntimeSnapshot:
    """Acquire product ownership and open only a verified private DB snapshot."""

    resolved_limits = limits or ProductSnapshotLimits()
    if not isinstance(resolved_limits, ProductSnapshotLimits):
        raise ValueError("product snapshot limits must be validated")

    data_path = _absolute_path(Path(data_dir), label="product data")
    database_absolute = _absolute_path(Path(database_path), label="product database")
    media_absolute = _absolute_path(Path(media_root), label="product media")
    scratch_parent_absolute = _absolute_path(
        Path(scratch_parent),
        label="benchmark scratch",
    )

    runtime_lock = ExclusiveRuntimeLock(data_path)
    lock_acquired = False
    descriptors: list[int] = []
    scratch_parent_descriptor: int | None = None
    scratch_descriptor: int | None = None
    scratch_name: str | None = None
    scratch_identity: tuple[int, ...] | None = None
    repository: Repository | None = None
    try:
        try:
            runtime_lock.acquire_existing()
            lock_acquired = True
        except Exception as error:
            raise _snapshot_error(error) from error

        opened_data_path, data_descriptor = _open_directory_path(
            data_path,
            label="product data",
        )
        descriptors.append(data_descriptor)
        _validate_lock_binding(runtime_lock, data_descriptor)

        database_relative = _relative_contained_path(
            database_absolute,
            opened_data_path,
            label="product database",
            allow_root=False,
        )
        media_relative = _relative_contained_path(
            media_absolute,
            opened_data_path,
            label="product media",
            allow_root=True,
        )
        if scratch_parent_absolute == opened_data_path or (
            scratch_parent_absolute.is_relative_to(opened_data_path)
        ):
            raise ProductSnapshotError(
                "benchmark scratch parent must be outside product data"
            )

        database_parent_descriptor = _open_relative_directory(
            data_descriptor,
            database_relative.parts[:-1],
            label="product database",
        )
        descriptors.append(database_parent_descriptor)
        media_descriptor = _open_relative_directory(
            data_descriptor,
            media_relative.parts,
            label="product media",
        )
        descriptors.append(media_descriptor)
        media_metadata = os.fstat(media_descriptor)
        if not stat.S_ISDIR(media_metadata.st_mode):
            raise ProductSnapshotError("product media root is unsafe")

        scratch_path, scratch_parent_descriptor = _open_directory_path(
            scratch_parent_absolute,
            label="benchmark scratch",
        )
        scratch_parent_metadata = os.fstat(scratch_parent_descriptor)
        if (
            scratch_parent_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(scratch_parent_metadata.st_mode) != 0o700
        ):
            raise ProductSnapshotError(
                "benchmark scratch parent must be an owner-only 0700 directory"
            )
        if _stable_object_identity(scratch_parent_metadata) == _stable_object_identity(
            os.fstat(data_descriptor)
        ):
            raise ProductSnapshotError(
                "benchmark scratch parent must be outside product data"
            )
        (
            scratch_name,
            scratch_root,
            scratch_descriptor,
            scratch_identity,
        ) = _create_owned_scratch(
            scratch_parent_descriptor,
            parent_path=scratch_path,
        )

        database_name = database_relative.parts[-1]
        source_parent_before = _file_state(os.fstat(database_parent_descriptor))
        first_reads: list[tuple[str, str, _SourceRead]] = []
        database_probe = _read_regular_source(
            database_parent_descriptor,
            database_name,
            max_bytes=resolved_limits.max_file_bytes,
            remaining_total_bytes=resolved_limits.max_total_bytes,
        )
        journal_mode = _journal_mode(database_probe)
        wal_exists = _entry_exists(
            database_parent_descriptor,
            f"{database_name}-wal",
        )
        shm_exists = _entry_exists(
            database_parent_descriptor,
            f"{database_name}-shm",
        )
        if journal_mode == "wal":
            if wal_exists != shm_exists:
                raise ProductSnapshotError(
                    "product database WAL and SHM sidecars must either both exist or both be absent"
                )
            required_entries = 3 if wal_exists else 1
            if resolved_limits.max_entries < required_entries:
                raise ProductSnapshotError("product snapshot exceeds the entry limit")
        else:
            if resolved_limits.max_entries < 1:
                raise ProductSnapshotError("product snapshot exceeds the entry limit")
            if _entry_exists(
                database_parent_descriptor,
                f"{database_name}-journal",
            ):
                raise ProductSnapshotError(
                    "DELETE-mode product database has an existing rollback journal"
                )
            if wal_exists or shm_exists:
                raise ProductSnapshotError(
                    "DELETE-mode product database has unexpected WAL sidecars"
                )

        database_read = _read_regular_source(
            database_parent_descriptor,
            database_name,
            max_bytes=resolved_limits.max_file_bytes,
            remaining_total_bytes=resolved_limits.max_total_bytes,
            output_parent_descriptor=scratch_descriptor,
            output_name=_DESTINATION_DATABASE_NAME,
        )
        if database_probe != database_read:
            raise ProductSnapshotError(
                "product database changed before snapshot copy"
            )
        first_reads.append(("database", database_name, database_read))
        copied_total_bytes = database_read.size_bytes
        if journal_mode == "wal" and wal_exists:
            source_names = (
                (
                    "wal",
                    f"{database_name}-wal",
                    f"{_DESTINATION_DATABASE_NAME}-wal",
                ),
                (
                    "shm",
                    f"{database_name}-shm",
                    f"{_DESTINATION_DATABASE_NAME}-shm",
                ),
            )
            for role, source_name, output_name in source_names:
                first_reads.append(
                    (
                        role,
                        source_name,
                        _read_regular_source(
                            database_parent_descriptor,
                            source_name,
                            max_bytes=resolved_limits.max_file_bytes,
                            remaining_total_bytes=(
                                resolved_limits.max_total_bytes - copied_total_bytes
                            ),
                            output_parent_descriptor=scratch_descriptor,
                            output_name=output_name,
                        ),
                    )
                )
                copied_total_bytes += first_reads[-1][2].size_bytes

        total_bytes = copied_total_bytes
        if total_bytes > resolved_limits.max_total_bytes:
            raise ProductSnapshotError("product snapshot exceeds the total byte limit")

        second_reads = [
            (
                role,
                source_name,
                _read_regular_source(
                    database_parent_descriptor,
                    source_name,
                    max_bytes=resolved_limits.max_file_bytes,
                ),
            )
            for role, source_name, _first in first_reads
        ]
        source_parent_after = _file_state(os.fstat(database_parent_descriptor))
        if first_reads != second_reads or source_parent_before != source_parent_after:
            raise ProductSnapshotError(
                "product database changed during snapshot creation"
            )

        destination_names = {
            "database": _DESTINATION_DATABASE_NAME,
            "wal": f"{_DESTINATION_DATABASE_NAME}-wal",
            "shm": f"{_DESTINATION_DATABASE_NAME}-shm",
        }
        for role, _source_name, source_read in first_reads:
            copied = _read_regular_source(
                scratch_descriptor,
                destination_names[role],
                max_bytes=resolved_limits.max_file_bytes,
            )
            if (
                copied.size_bytes != source_read.size_bytes
                or copied.sha256 != source_read.sha256
            ):
                raise ProductSnapshotError(
                    "private product database copy failed verification"
                )
        os.fsync(scratch_descriptor)

        identity = ProductSnapshotIdentity(
            journal_mode=journal_mode,
            files=tuple(
                ProductSnapshotFileIdentity(
                    role=role,  # type: ignore[arg-type]
                    size_bytes=source_read.size_bytes,
                    sha256=source_read.sha256,
                )
                for role, _source_name, source_read in first_reads
            ),
            total_bytes=total_bytes,
        )
        retained_scratch_root = _retained_directory_path(scratch_descriptor)
        repository = Repository.open_read_only(
            retained_scratch_root / _DESTINATION_DATABASE_NAME
        )

        retained = tuple(descriptors)
        descriptors.clear()
        assert scratch_parent_descriptor is not None
        assert scratch_descriptor is not None
        assert scratch_name is not None
        assert scratch_identity is not None
        snapshot = ProductRuntimeSnapshot(
            repository=repository,
            media_root=media_absolute,
            scratch_root=scratch_root,
            identity=identity,
            runtime_lock=runtime_lock,
            scratch_parent_descriptor=scratch_parent_descriptor,
            scratch_descriptor=scratch_descriptor,
            scratch_name=scratch_name,
            scratch_identity=scratch_identity,
            data_descriptor=data_descriptor,
            media_descriptor=media_descriptor,
            retained_descriptors=retained,
        )
        scratch_parent_descriptor = None
        scratch_descriptor = None
        lock_acquired = False
        repository = None
        return snapshot
    except BaseException as error:
        cleanup = _OpenFailureCleanup(
            repository=repository,
            runtime_lock=runtime_lock,
            lock_acquired=lock_acquired,
            scratch_parent_descriptor=scratch_parent_descriptor,
            scratch_descriptor=scratch_descriptor,
            scratch_name=scratch_name,
            scratch_identity=scratch_identity,
            retained_descriptors=descriptors,
        )
        try:
            cleanup.close_with_retries(_OPEN_CLEANUP_ATTEMPTS)
        except BaseException as cleanup_error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            pending = ProductSnapshotCleanupError(error, cleanup)
            raise pending from ExceptionGroup(
                "product snapshot open and cleanup both failed",
                [_snapshot_error(error), _snapshot_error(cleanup_error)],
            )
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise _snapshot_error(error) from error


__all__ = [
    "ProductRuntimeSnapshot",
    "ProductSnapshotCleanupError",
    "ProductSnapshotError",
    "ProductSnapshotFileIdentity",
    "ProductSnapshotIdentity",
    "ProductSnapshotLimits",
    "open_product_runtime_snapshot",
]
