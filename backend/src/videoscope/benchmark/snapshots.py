from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from importlib import resources
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from threading import Lock
from uuid import uuid4

from videoscope.model_manifest import (
    FASTEMBED_ALGORITHM_VERSION,
    FASTEMBED_REPOSITORY,
    FASTEMBED_RUNTIME_VERSION,
    MODEL_REVISIONS,
    TEXT_EMBEDDING_DIMENSIONS,
    TEXT_EMBEDDING_MODEL,
)


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_HF_BLOB_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_REQUIRED_FASTEMBED_FILES = frozenset(
    {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "onnx/model.onnx",
    }
)
_MAX_FASTEMBED_FILES = 64
_MAX_FASTEMBED_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
_MAX_FASTEMBED_FILE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_SYMLINK_TARGET_BYTES = 4096
_READ_CHUNK_BYTES = 1024 * 1024
_STAGING_CLEANUP_ATTEMPTS = 3
_REVIEWED_FASTEMBED_MANIFEST = (
    "fastembed-xenova-paraphrase-multilingual-mpnet-base-v2-"
    "e5d116277351513fd260955ece953ecddde7046e.json"
)


class SnapshotError(RuntimeError):
    """A source cannot be copied into a reproducible private benchmark snapshot."""


class SnapshotCleanupError(SnapshotError):
    """Snapshot creation failed and its private staging cleanup needs recovery."""

    def __init__(
        self,
        *,
        primary_error: Exception,
        cleanup_error: Exception,
        recovery: _StagingCleanupRecovery,
    ) -> None:
        super().__init__(
            "benchmark snapshot failed and private staging cleanup remains pending"
        )
        self.primary_error = primary_error
        self.cleanup_error = cleanup_error
        self._recovery = recovery

    @property
    def cleanup_pending(self) -> bool:
        return self._recovery.pending

    def retry_cleanup(self) -> bool:
        """Retry bounded descriptor-relative cleanup without exposing a local path."""
        if not self._recovery.pending:
            return False
        try:
            self._recovery.retry()
        except Exception as error:
            self.cleanup_error = error
            raise self from error
        return True


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_relative_path(value: object, *, max_depth: int = 16) -> str:
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        raise ValueError("snapshot relative path is invalid")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or parsed.as_posix() != value:
        raise ValueError("snapshot relative path is invalid")
    parts = parsed.parts
    if not parts or len(parts) > max_depth or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("snapshot relative path is invalid")
    return value


@dataclass(frozen=True, slots=True)
class FastEmbedFileManifestEntry:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _validated_relative_path(self.relative_path),
        )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
            or self.size_bytes > _MAX_FASTEMBED_FILE_BYTES
        ):
            raise ValueError("FastEmbed file size is invalid")
        if type(self.sha256) is not str or _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("FastEmbed file digest must be lowercase SHA-256")

    @property
    def canonical_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class FastEmbedSnapshotManifest:
    """Externally reviewed byte allowlist for the one pinned FastEmbed model."""

    model_name: str
    model_repository: str
    model_revision: str
    runtime_version: str
    algorithm_version: str
    dimensions: int
    files: tuple[FastEmbedFileManifestEntry, ...]
    _model_content_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        expected = (
            TEXT_EMBEDDING_MODEL,
            FASTEMBED_REPOSITORY,
            MODEL_REVISIONS[FASTEMBED_REPOSITORY],
            FASTEMBED_RUNTIME_VERSION,
            FASTEMBED_ALGORITHM_VERSION,
            TEXT_EMBEDDING_DIMENSIONS,
        )
        actual = (
            self.model_name,
            self.model_repository,
            self.model_revision,
            self.runtime_version,
            self.algorithm_version,
            self.dimensions,
        )
        if actual != expected:
            raise ValueError("FastEmbed manifest does not match the pinned model contract")
        if not isinstance(self.files, tuple) or not self.files:
            raise ValueError("FastEmbed manifest files must be a non-empty tuple")
        if len(self.files) > _MAX_FASTEMBED_FILES or any(
            not isinstance(item, FastEmbedFileManifestEntry) for item in self.files
        ):
            raise ValueError("FastEmbed manifest file allowlist is invalid")
        ordered = tuple(sorted(self.files, key=lambda item: item.relative_path))
        if ordered != self.files:
            raise ValueError("FastEmbed manifest files must be canonically ordered")
        paths = tuple(item.relative_path for item in ordered)
        if len(set(paths)) != len(paths):
            raise ValueError("FastEmbed manifest contains a duplicate path")
        missing = sorted(_REQUIRED_FASTEMBED_FILES - set(paths))
        if missing:
            raise ValueError(f"required FastEmbed file is missing: {missing[0]}")
        total = sum(item.size_bytes for item in ordered)
        if total > _MAX_FASTEMBED_TOTAL_BYTES:
            raise ValueError("FastEmbed manifest exceeds the byte limit")
        object.__setattr__(
            self,
            "_model_content_sha256",
            _canonical_sha256([item.canonical_dict for item in ordered]),
        )

    @property
    def model_content_sha256(self) -> str:
        return self._model_content_sha256

    @property
    def canonical_dict(self) -> dict[str, object]:
        return {
            "algorithm_version": self.algorithm_version,
            "dimensions": self.dimensions,
            "files": [item.canonical_dict for item in self.files],
            "model_name": self.model_name,
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "runtime_version": self.runtime_version,
        }


@dataclass(frozen=True, slots=True)
class FastEmbedSnapshotIdentity:
    model_name: str
    model_repository: str
    model_revision: str
    runtime_version: str
    algorithm_version: str
    dimensions: int
    model_content_sha256: str

    @property
    def canonical_dict(self) -> dict[str, object]:
        return {
            "algorithm_version": self.algorithm_version,
            "dimensions": self.dimensions,
            "model_content_sha256": self.model_content_sha256,
            "model_name": self.model_name,
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "runtime_version": self.runtime_version,
        }


@dataclass(frozen=True, slots=True)
class FastEmbedSnapshot:
    path: Path
    identity: FastEmbedSnapshotIdentity
    manifest: FastEmbedSnapshotManifest = field(repr=False)

    def create_embedding(self):  # type: ignore[no-untyped-def]
        from videoscope.search.embeddings import SemanticEmbedding

        verify_fastembed_snapshot(self)
        return SemanticEmbedding(
            model_name=self.identity.model_name,
            model_repository=self.identity.model_repository,
            model_revision=self.identity.model_revision,
            expected_runtime_version=self.identity.runtime_version,
            algorithm_version=self.identity.algorithm_version,
            dimensions=self.identity.dimensions,
            strict=True,
            specific_model_path=self.path,
            model_content_sha256=self.identity.model_content_sha256,
            model_verifier=lambda: verify_fastembed_snapshot(self),
        )


@dataclass(frozen=True, slots=True)
class QdrantSnapshotLimits:
    max_entries: int = 100_000
    max_depth: int = 32
    max_total_bytes: int = 64 * 1024 * 1024 * 1024
    max_file_bytes: int = 16 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("max_entries", "max_depth", "max_total_bytes", "max_file_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("Qdrant per-file byte limit exceeds the total byte limit")


@dataclass(frozen=True, slots=True)
class QdrantStorageSnapshot:
    path: Path
    snapshot_sha256: str
    entry_count: int
    total_bytes: int
    source_device: int
    source_inode: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).absolute())
        if (
            type(self.snapshot_sha256) is not str
            or _SHA256_PATTERN.fullmatch(self.snapshot_sha256) is None
        ):
            raise ValueError("Qdrant snapshot digest must be lowercase SHA-256")
        for name in ("entry_count", "total_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Qdrant snapshot {name} is invalid")
        for name in ("source_device", "source_inode"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Qdrant snapshot {name} is invalid")


@dataclass(frozen=True, slots=True)
class _FileState:
    device: int
    inode: int
    mode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _TreeScan:
    entries: tuple[dict[str, object], ...]
    states: tuple[tuple[str, _FileState], ...]
    total_bytes: int

    @property
    def digest(self) -> str:
        return _canonical_sha256(list(self.entries))


@dataclass(frozen=True, slots=True)
class _FastEmbedResolvedFileState:
    source_entry: _FileState
    symlink_target: str | None = None
    blob: _FileState | None = None
    blob_directory: _FileState | None = None


@dataclass(frozen=True, slots=True)
class _HuggingFaceRepository:
    path: Path
    file_descriptor: int
    initial_state: _FileState


def _file_state(metadata: os.stat_result) -> _FileState:
    return _FileState(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        links=metadata.st_nlink,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _required_open_flags(*, directory: bool = False) -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise SnapshotError("platform does not support no-follow snapshot reads")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if directory:
        flags |= os.O_DIRECTORY
    else:
        flags |= getattr(os, "O_NONBLOCK", 0)
    return flags


def _macos_file_id_path(
    absolute: Path,
) -> tuple[Path, tuple[str, ...], int, int] | None:
    parts = absolute.parts
    if len(parts) < 2 or parts[1] != ".vol":
        return None
    if (
        len(parts) < 4
        or not parts[2].isdigit()
        or not parts[3].isdigit()
        or str(int(parts[2])) != parts[2]
        or str(int(parts[3])) != parts[3]
        or any(part in {"", ".", ".."} for part in parts[4:])
    ):
        raise SnapshotError("macOS file-id directory path is invalid")
    device = int(parts[2])
    inode = int(parts[3])
    if device <= 0 or inode <= 0:
        raise SnapshotError("macOS file-id directory path is invalid")
    return Path(f"/.vol/{device}/{inode}"), parts[4:], device, inode


def _open_macos_file_id_root(path: Path, *, device: int, inode: int) -> int:
    def identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
        return metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_uid

    try:
        before = os.stat(path, follow_symlinks=False)
        file_descriptor = os.open(path, _required_open_flags(directory=True))
        opened = os.fstat(file_descriptor)
        after = os.stat(path, follow_symlinks=False)
    except OSError as error:
        if "file_descriptor" in locals():
            os.close(file_descriptor)
        raise SnapshotError("stable macOS directory capability is unavailable") from error
    if (
        not stat.S_ISDIR(before.st_mode)
        or identity(before) != identity(opened)
        or identity(opened) != identity(after)
        or (opened.st_dev, opened.st_ino) != (device, inode)
    ):
        os.close(file_descriptor)
        raise SnapshotError("stable macOS directory capability identity changed")
    return file_descriptor


def _open_directory_path(path: Path) -> tuple[Path, int]:
    absolute = Path(path).absolute()
    macos_file_id = _macos_file_id_path(absolute)
    if macos_file_id is not None:
        stable_root, relative_parts, device, inode = macos_file_id
        current_fd = _open_macos_file_id_root(
            stable_root,
            device=device,
            inode=inode,
        )
        try:
            for component in relative_parts:
                next_fd = os.open(
                    component,
                    _required_open_flags(directory=True),
                    dir_fd=current_fd,
                )
                os.close(current_fd)
                current_fd = next_fd
            return absolute, current_fd
        except OSError as error:
            os.close(current_fd)
            raise SnapshotError(
                "stable directory capability child is unsafe or missing"
            ) from error
    try:
        current_fd = os.open(absolute.anchor, _required_open_flags(directory=True))
    except OSError as error:
        raise SnapshotError("snapshot root could not be opened") from error
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(
                    component,
                    _required_open_flags(directory=True),
                    dir_fd=current_fd,
                )
            except OSError as error:
                raise SnapshotError("snapshot path contains an unsafe or missing component") from error
            os.close(current_fd)
            current_fd = next_fd
        metadata = os.fstat(current_fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise SnapshotError("snapshot root is not a directory")
        return absolute, current_fd
    except Exception:
        os.close(current_fd)
        raise


@dataclass(frozen=True, slots=True)
class _RetainedDirectoryIdentity:
    device: int
    inode: int
    mode: int
    owner: int


def _retained_directory_identity(metadata: os.stat_result) -> _RetainedDirectoryIdentity:
    if not stat.S_ISDIR(metadata.st_mode):
        raise SnapshotError("retained filesystem capability is not a directory")
    return _RetainedDirectoryIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        owner=metadata.st_uid,
    )


class RetainedDirectory:
    """Owned dirfd plus a verified macOS file-id path that survives renames."""

    def __init__(
        self,
        *,
        logical_path: Path,
        stable_path: Path,
        descriptor: int,
        identity: _RetainedDirectoryIdentity,
    ) -> None:
        self._lock = Lock()
        self.logical_path = logical_path
        self.stable_path = stable_path
        self._descriptor: int | None = descriptor
        self._identity = identity

    @classmethod
    def retain(cls, path: Path, directory_fd: int) -> RetainedDirectory:
        if isinstance(directory_fd, bool) or not isinstance(directory_fd, int):
            raise ValueError("retained directory descriptor must be an integer")
        try:
            retained_fd = os.dup(directory_fd)
        except OSError as error:
            raise SnapshotError("directory capability could not be retained") from error
        logical_fd: int | None = None
        stable_fd: int | None = None
        try:
            identity = _retained_directory_identity(os.fstat(retained_fd))
            logical_path, logical_fd = _open_directory_path(Path(path))
            if _retained_directory_identity(os.fstat(logical_fd)) != identity:
                raise SnapshotError("directory capability path binding changed")
            stable_candidate = Path(
                f"/.vol/{identity.device}/{identity.inode}"
            )
            stable_path, stable_fd = _open_directory_path(stable_candidate)
            if _retained_directory_identity(os.fstat(stable_fd)) != identity:
                raise SnapshotError("stable directory capability identity is unavailable")
            if _retained_directory_identity(os.fstat(retained_fd)) != identity:
                raise SnapshotError("retained directory capability changed")
            return cls(
                logical_path=logical_path,
                stable_path=stable_path,
                descriptor=retained_fd,
                identity=identity,
            )
        except Exception:
            os.close(retained_fd)
            raise
        finally:
            if stable_fd is not None:
                os.close(stable_fd)
            if logical_fd is not None:
                os.close(logical_fd)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._descriptor is None

    def child(self, relative_path: str) -> RetainedDirectoryPath:
        validated = _validated_relative_path(relative_path, max_depth=32)
        return RetainedDirectoryPath(
            root=self,
            parts=PurePosixPath(validated).parts,
        )

    def duplicate_descriptor(self) -> int:
        """Return a validated directory descriptor owned by the caller."""
        return self._open_descriptor()

    def read_regular_file(self, relative_path: str, *, max_bytes: int) -> bytes:
        content = _read_retained_regular_file(
            self,
            (),
            relative_path,
            max_bytes=max_bytes,
            optional=False,
        )
        assert content is not None
        return content

    def read_optional_regular_file(
        self,
        relative_path: str,
        *,
        max_bytes: int,
    ) -> bytes | None:
        return _read_retained_regular_file(
            self,
            (),
            relative_path,
            max_bytes=max_bytes,
            optional=True,
        )

    def _open_descriptor(self, parts: tuple[str, ...] = ()) -> int:
        with self._lock:
            descriptor = self._descriptor
            if descriptor is None:
                raise SnapshotError("retained directory capability is closed")
            try:
                if _retained_directory_identity(os.fstat(descriptor)) != self._identity:
                    raise SnapshotError("retained directory capability changed")
                opened = _open_relative_directory(descriptor, parts)
                stable_path = self.stable_path.joinpath(*parts)
                _stable_path, stable_fd = _open_directory_path(stable_path)
                try:
                    if _retained_directory_identity(os.fstat(stable_fd)) != (
                        _retained_directory_identity(os.fstat(opened))
                    ):
                        raise SnapshotError("stable directory capability child changed")
                finally:
                    os.close(stable_fd)
                if _retained_directory_identity(os.fstat(descriptor)) != self._identity:
                    raise SnapshotError("retained directory capability changed")
                return opened
            except Exception:
                if "opened" in locals():
                    os.close(opened)
                raise

    def close(self) -> bool:
        with self._lock:
            descriptor = self._descriptor
            if descriptor is None:
                return False
            try:
                os.close(descriptor)
            except OSError as error:
                raise SnapshotError(
                    "retained directory capability could not be closed"
                ) from error
            self._descriptor = None
            return True

    def __del__(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            descriptor = getattr(self, "_descriptor", None)
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                self._descriptor = None


@dataclass(frozen=True, slots=True)
class RetainedDirectoryPath:
    """A name-relative child view revalidated beneath one retained inode root."""
    root: RetainedDirectory = field(repr=False)
    parts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.root, RetainedDirectory) or not self.parts:
            raise ValueError("retained directory child is invalid")
        _validated_relative_path(PurePosixPath(*self.parts).as_posix(), max_depth=32)

    @property
    def stable_path(self) -> Path:
        return self.root.stable_path.joinpath(*self.parts)

    def child(self, relative_path: str) -> RetainedDirectoryPath:
        validated = _validated_relative_path(relative_path, max_depth=32)
        parts = (*self.parts, *PurePosixPath(validated).parts)
        _validated_relative_path(PurePosixPath(*parts).as_posix(), max_depth=32)
        return RetainedDirectoryPath(root=self.root, parts=parts)

    def duplicate_descriptor(self) -> int:
        """Return a validated directory descriptor owned by the caller."""
        return self._open_descriptor()

    def read_regular_file(self, relative_path: str, *, max_bytes: int) -> bytes:
        content = _read_retained_regular_file(
            self.root,
            self.parts,
            relative_path,
            max_bytes=max_bytes,
            optional=False,
        )
        assert content is not None
        return content

    def read_optional_regular_file(
        self,
        relative_path: str,
        *,
        max_bytes: int,
    ) -> bytes | None:
        return _read_retained_regular_file(
            self.root,
            self.parts,
            relative_path,
            max_bytes=max_bytes,
            optional=True,
        )

    def _open_descriptor(self) -> int:
        return self.root._open_descriptor(self.parts)


def _read_retained_regular_file(
    root: RetainedDirectory,
    prefix: tuple[str, ...],
    relative_path: str,
    *,
    max_bytes: int,
    optional: bool,
) -> bytes | None:
    validated = _validated_relative_path(relative_path, max_depth=32)
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes <= 0
    ):
        raise ValueError("retained file byte limit must be positive")
    parts = (*prefix, *PurePosixPath(validated).parts)
    if len(parts) > 32:
        raise ValueError("retained file path exceeds the depth limit")
    parent_fd = root._open_descriptor(parts[:-1])
    try:
        try:
            metadata = os.stat(
                parts[-1],
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            if optional:
                return None
            raise SnapshotError("retained regular file is missing")
        except OSError as error:
            raise SnapshotError("retained regular file could not be inspected") from error
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SnapshotError("retained file must be a regular single-link file")
        return _read_regular_content(
            parent_fd,
            parts[-1],
            max_bytes=max_bytes,
        )
    finally:
        os.close(parent_fd)


_DirectoryInput = Path | RetainedDirectory | RetainedDirectoryPath


def _acquire_directory(value: _DirectoryInput) -> tuple[Path, int]:
    if isinstance(value, RetainedDirectory):
        return value.stable_path, value._open_descriptor()
    if isinstance(value, RetainedDirectoryPath):
        return value.stable_path, value._open_descriptor()
    return _open_directory_path(Path(value))


def _open_relative_directory(root_fd: int, parts: tuple[str, ...]) -> int:
    current_fd = os.dup(root_fd)
    try:
        for component in parts:
            next_fd = os.open(
                component,
                _required_open_flags(directory=True),
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except OSError as error:
        os.close(current_fd)
        raise SnapshotError("snapshot source directory is unsafe or missing") from error


def _ensure_private_scratch(path: _DirectoryInput) -> tuple[Path, int]:
    absolute, scratch_fd = _acquire_directory(path)
    metadata = os.fstat(scratch_fd)
    if stat.S_IMODE(metadata.st_mode) != 0o700 or metadata.st_uid != os.getuid():
        os.close(scratch_fd)
        raise SnapshotError("benchmark scratch must be an owner-only 0700 directory")
    return absolute, scratch_fd


def _directory_is_at_or_beneath(ancestor_fd: int, candidate_fd: int) -> bool:
    ancestor = os.fstat(ancestor_fd)
    ancestor_identity = (ancestor.st_dev, ancestor.st_ino)
    current_fd = os.dup(candidate_fd)
    visited: set[tuple[int, int]] = set()
    try:
        for _ in range(256):
            current = os.fstat(current_fd)
            identity = (current.st_dev, current.st_ino)
            if identity == ancestor_identity:
                return True
            if identity in visited:
                raise SnapshotError("snapshot directory ancestry contains a cycle")
            visited.add(identity)
            try:
                parent_fd = os.open(
                    "..",
                    _required_open_flags(directory=True),
                    dir_fd=current_fd,
                )
            except OSError as error:
                raise SnapshotError("snapshot directory ancestry could not be verified") from error
            parent = os.fstat(parent_fd)
            if (parent.st_dev, parent.st_ino) == identity:
                os.close(parent_fd)
                return False
            os.close(current_fd)
            current_fd = parent_fd
        raise SnapshotError("snapshot directory ancestry exceeds the safety limit")
    finally:
        os.close(current_fd)


def _require_disjoint_roots(source_fd: int, scratch_fd: int) -> None:
    source_before = _file_state(os.fstat(source_fd))
    scratch_before = _file_state(os.fstat(scratch_fd))
    if _directory_is_at_or_beneath(source_fd, scratch_fd) or _directory_is_at_or_beneath(
        scratch_fd,
        source_fd,
    ):
        raise SnapshotError("snapshot source and scratch roots overlap")
    if (
        source_before != _file_state(os.fstat(source_fd))
        or scratch_before != _file_state(os.fstat(scratch_fd))
    ):
        raise SnapshotError("snapshot roots changed while checking overlap")


def _bounded_directory_names(
    directory_fd: int,
    *,
    remaining_entries: int,
    limit_error: str,
    read_error: str,
) -> tuple[str, ...]:
    if remaining_entries < 0:
        raise SnapshotError(limit_error)
    names: list[str] = []
    try:
        with os.scandir(directory_fd) as iterator:
            for _ in range(remaining_entries + 1):
                try:
                    entry = next(iterator)
                except StopIteration:
                    break
                names.append(entry.name)
    except OSError as error:
        raise SnapshotError(read_error) from error
    if len(names) > remaining_entries:
        raise SnapshotError(limit_error)
    return tuple(sorted(names))


def _write_all(file_descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(file_descriptor, view)
        if written <= 0:
            raise SnapshotError("snapshot output write made no progress")
        view = view[written:]


def _prepare_output_parent(root_fd: int, parts: tuple[str, ...]) -> int:
    current_fd = os.dup(root_fd)
    try:
        for component in parts:
            try:
                os.mkdir(component, mode=0o700, dir_fd=current_fd)
                os.fsync(current_fd)
            except FileExistsError:
                metadata = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise SnapshotError("snapshot output path collides with a non-directory")
            next_fd = os.open(
                component,
                _required_open_flags(directory=True),
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _read_regular_file(
    parent_fd: int,
    name: str,
    *,
    max_bytes: int,
    output_parent_fd: int | None = None,
    output_name: str | None = None,
) -> tuple[int, str, _FileState]:
    try:
        source_fd = os.open(name, _required_open_flags(), dir_fd=parent_fd)
    except OSError as error:
        raise SnapshotError("snapshot source file is unsafe or missing") from error
    output_fd: int | None = None
    digest = hashlib.sha256()
    try:
        before_metadata = os.fstat(source_fd)
        before = _file_state(before_metadata)
        if not stat.S_ISREG(before.mode) or before.links != 1:
            raise SnapshotError("snapshot source entries must be regular single-link files")
        if before.size > max_bytes:
            raise SnapshotError("snapshot source exceeds the per-file byte limit")
        if output_parent_fd is not None:
            if output_name is None:
                raise SnapshotError("snapshot output name is missing")
            output_fd = os.open(
                output_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=output_parent_fd,
            )
            os.fchmod(output_fd, 0o600)
        consumed = 0
        while True:
            block = os.read(source_fd, min(_READ_CHUNK_BYTES, max_bytes - consumed + 1))
            if not block:
                break
            consumed += len(block)
            if consumed > max_bytes:
                raise SnapshotError("snapshot source exceeds the per-file byte limit")
            digest.update(block)
            if output_fd is not None:
                _write_all(output_fd, block)
        after = _file_state(os.fstat(source_fd))
        current = _file_state(os.stat(name, dir_fd=parent_fd, follow_symlinks=False))
        if before != after or before != current or consumed != before.size:
            raise SnapshotError("snapshot source changed while it was read")
        if output_fd is not None:
            os.fsync(output_fd)
            os.fsync(output_parent_fd)
        return consumed, digest.hexdigest(), before
    except OSError as error:
        raise SnapshotError("snapshot file could not be read or copied") from error
    finally:
        if output_fd is not None:
            os.close(output_fd)
        os.close(source_fd)


def _read_regular_content(
    parent_fd: int,
    name: str,
    *,
    max_bytes: int,
) -> bytes:
    try:
        source_fd = os.open(name, _required_open_flags(), dir_fd=parent_fd)
    except OSError as error:
        raise SnapshotError("snapshot source file is unsafe or missing") from error
    try:
        before = _file_state(os.fstat(source_fd))
        if not stat.S_ISREG(before.mode) or before.links != 1:
            raise SnapshotError("snapshot source entries must be regular single-link files")
        if before.size > max_bytes:
            raise SnapshotError("snapshot source exceeds the per-file byte limit")
        content = bytearray()
        while True:
            block = os.read(source_fd, min(_READ_CHUNK_BYTES, max_bytes - len(content) + 1))
            if not block:
                break
            content.extend(block)
            if len(content) > max_bytes:
                raise SnapshotError("snapshot source exceeds the per-file byte limit")
        after = _file_state(os.fstat(source_fd))
        current = _file_state(os.stat(name, dir_fd=parent_fd, follow_symlinks=False))
        if before != after or before != current or len(content) != before.size:
            raise SnapshotError("snapshot source changed while it was read")
        return bytes(content)
    except OSError as error:
        raise SnapshotError("snapshot file could not be read") from error
    finally:
        os.close(source_fd)


def _expected_huggingface_repository_name(repository: str) -> str:
    parts = repository.split("/")
    if len(parts) != 2 or any(
        not part or part in {".", ".."} or "--" in part or "\\" in part
        for part in parts
    ):
        raise SnapshotError("FastEmbed repository identity is not cache-addressable")
    return f"models--{parts[0]}--{parts[1]}"


def _open_huggingface_repository(
    source_path: Path,
    source_fd: int,
    manifest: FastEmbedSnapshotManifest,
) -> _HuggingFaceRepository | None:
    """Recognize the one canonical HF cache layout without trusting symlinks."""
    repository_path = source_path.parent.parent
    if (
        source_path.name != manifest.model_revision
        or source_path.parent.name != "snapshots"
        or repository_path.name
        != _expected_huggingface_repository_name(manifest.model_repository)
    ):
        return None
    _opened_path, repository_fd = _open_directory_path(repository_path)
    try:
        initial_state = _file_state(os.fstat(repository_fd))
        snapshots_fd = _open_relative_directory(repository_fd, ("snapshots",))
        try:
            revision_fd = _open_relative_directory(
                snapshots_fd,
                (manifest.model_revision,),
            )
            try:
                if _file_state(os.fstat(revision_fd)) != _file_state(os.fstat(source_fd)):
                    raise SnapshotError(
                        "FastEmbed snapshot is not the pinned repository revision"
                    )
            finally:
                os.close(revision_fd)
        finally:
            os.close(snapshots_fd)
        return _HuggingFaceRepository(
            path=repository_path,
            file_descriptor=repository_fd,
            initial_state=initial_state,
        )
    except Exception:
        os.close(repository_fd)
        raise


def _canonical_huggingface_blob_name(
    raw_target: str,
    *,
    source_path: Path,
    relative_path: str,
    repository_path: Path,
) -> str:
    if (
        type(raw_target) is not str
        or not raw_target
        or "\x00" in raw_target
        or "\\" in raw_target
        or os.path.isabs(raw_target)
        or len(raw_target.encode("utf-8")) > _MAX_SYMLINK_TARGET_BYTES
        or os.path.normpath(raw_target) != raw_target
    ):
        raise SnapshotError("FastEmbed cache link target is unsafe")
    source_parent = source_path.joinpath(
        *PurePosixPath(relative_path).parts[:-1]
    )
    candidate = Path(
        os.path.normpath(os.path.join(os.fspath(source_parent), raw_target))
    )
    blob_root = repository_path / "blobs"
    if candidate.parent != blob_root or _HF_BLOB_PATTERN.fullmatch(candidate.name) is None:
        raise SnapshotError("FastEmbed cache link does not name a repository blob")
    if os.path.relpath(candidate, start=source_parent) != raw_target:
        raise SnapshotError("FastEmbed cache link target is not canonical")
    return candidate.name


def _read_huggingface_blob_link(
    source_parent_fd: int,
    source_name: str,
    *,
    source_path: Path,
    relative_path: str,
    repository: _HuggingFaceRepository,
    max_bytes: int,
    output_parent_fd: int | None,
    output_name: str | None,
) -> tuple[int, str, _FastEmbedResolvedFileState]:
    try:
        before = _file_state(
            os.stat(source_name, dir_fd=source_parent_fd, follow_symlinks=False)
        )
        if (
            not stat.S_ISLNK(before.mode)
            or before.links != 1
            or before.size > _MAX_SYMLINK_TARGET_BYTES
        ):
            raise SnapshotError("FastEmbed cache link is unsafe")
        target_before = os.readlink(source_name, dir_fd=source_parent_fd)
        if before != _file_state(
            os.stat(source_name, dir_fd=source_parent_fd, follow_symlinks=False)
        ):
            raise SnapshotError("FastEmbed cache link changed while it was inspected")
    except OSError as error:
        raise SnapshotError("FastEmbed cache link is unsafe or missing") from error
    blob_name = _canonical_huggingface_blob_name(
        target_before,
        source_path=source_path,
        relative_path=relative_path,
        repository_path=repository.path,
    )
    try:
        blob_directory_metadata = os.stat(
            "blobs",
            dir_fd=repository.file_descriptor,
            follow_symlinks=False,
        )
        blob_directory_before = _file_state(blob_directory_metadata)
        if not stat.S_ISDIR(blob_directory_metadata.st_mode):
            raise SnapshotError("FastEmbed repository blob root is unsafe")
        blob_directory_fd = os.open(
            "blobs",
            _required_open_flags(directory=True),
            dir_fd=repository.file_descriptor,
        )
    except OSError as error:
        raise SnapshotError("FastEmbed repository blob root is unsafe or missing") from error
    try:
        if _file_state(os.fstat(blob_directory_fd)) != blob_directory_before:
            raise SnapshotError("FastEmbed repository blob root changed while opening")
        size, digest, blob_state = _read_regular_file(
            blob_directory_fd,
            blob_name,
            max_bytes=max_bytes,
            output_parent_fd=output_parent_fd,
            output_name=output_name,
        )
        blob_directory_after = _file_state(os.fstat(blob_directory_fd))
        blob_directory_current = _file_state(
            os.stat(
                "blobs",
                dir_fd=repository.file_descriptor,
                follow_symlinks=False,
            )
        )
        if (
            blob_directory_before != blob_directory_after
            or blob_directory_before != blob_directory_current
        ):
            raise SnapshotError("FastEmbed repository blob root changed while reading")
    except OSError as error:
        raise SnapshotError("FastEmbed repository blob could not be read") from error
    finally:
        os.close(blob_directory_fd)
    try:
        target_after = os.readlink(source_name, dir_fd=source_parent_fd)
        after = _file_state(
            os.stat(source_name, dir_fd=source_parent_fd, follow_symlinks=False)
        )
    except OSError as error:
        raise SnapshotError("FastEmbed cache link changed while reading its blob") from error
    if before != after or target_before != target_after:
        raise SnapshotError("FastEmbed cache link changed while reading its blob")
    return (
        size,
        digest,
        _FastEmbedResolvedFileState(
            source_entry=before,
            symlink_target=target_before,
            blob=blob_state,
            blob_directory=blob_directory_before,
        ),
    )


def _read_fastembed_source_file(
    source_parent_fd: int,
    source_name: str,
    *,
    source_path: Path | None,
    relative_path: str,
    repository: _HuggingFaceRepository | None,
    output_parent_fd: int | None,
    output_name: str | None,
) -> tuple[int, str, _FastEmbedResolvedFileState]:
    try:
        metadata = os.stat(
            source_name,
            dir_fd=source_parent_fd,
            follow_symlinks=False,
        )
    except OSError as error:
        raise SnapshotError("FastEmbed source file is unsafe or missing") from error
    if stat.S_ISREG(metadata.st_mode):
        size, digest, state = _read_regular_file(
            source_parent_fd,
            source_name,
            max_bytes=_MAX_FASTEMBED_FILE_BYTES,
            output_parent_fd=output_parent_fd,
            output_name=output_name,
        )
        return size, digest, _FastEmbedResolvedFileState(source_entry=state)
    if stat.S_ISLNK(metadata.st_mode) and source_path is not None and repository is not None:
        return _read_huggingface_blob_link(
            source_parent_fd,
            source_name,
            source_path=source_path,
            relative_path=relative_path,
            repository=repository,
            max_bytes=_MAX_FASTEMBED_FILE_BYTES,
            output_parent_fd=output_parent_fd,
            output_name=output_name,
        )
    raise SnapshotError(
        "FastEmbed source entries must be regular files or pinned repository blob links"
    )


def _scan_allowlisted_files(
    source_fd: int,
    manifest: FastEmbedSnapshotManifest,
    *,
    output_fd: int | None = None,
    source_path: Path | None = None,
    repository: _HuggingFaceRepository | None = None,
) -> tuple[tuple[str, int, str, _FastEmbedResolvedFileState], ...]:
    records: list[tuple[str, int, str, _FastEmbedResolvedFileState]] = []
    total = 0
    for entry in manifest.files:
        parts = PurePosixPath(entry.relative_path).parts
        source_parent = _open_relative_directory(source_fd, parts[:-1])
        output_parent: int | None = None
        try:
            if output_fd is not None:
                output_parent = _prepare_output_parent(output_fd, parts[:-1])
            size, digest, state = _read_fastembed_source_file(
                source_parent,
                parts[-1],
                source_path=source_path,
                relative_path=entry.relative_path,
                repository=repository,
                output_parent_fd=output_parent,
                output_name=parts[-1] if output_parent is not None else None,
            )
        finally:
            if output_parent is not None:
                os.close(output_parent)
            os.close(source_parent)
        if size != entry.size_bytes or digest != entry.sha256:
            raise SnapshotError("FastEmbed source does not match the reviewed manifest")
        total += size
        if total > _MAX_FASTEMBED_TOTAL_BYTES:
            raise SnapshotError("FastEmbed source exceeds the total byte limit")
        records.append((entry.relative_path, size, digest, state))
    return tuple(records)


@dataclass(slots=True)
class _OwnedStaging:
    name: str
    max_entries: int
    created: bool = False
    device: int | None = None
    inode: int | None = None
    directory_removed: bool = False
    parent_synced: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or self.name in {".", ".."}
            or "/" in self.name
            or "\\" in self.name
            or "\x00" in self.name
        ):
            raise ValueError("owned staging name is invalid")
        if (
            isinstance(self.max_entries, bool)
            or not isinstance(self.max_entries, int)
            or self.max_entries <= 0
        ):
            raise ValueError("owned staging entry limit must be positive")

    def bind(self, staging_fd: int) -> None:
        metadata = os.fstat(staging_fd)
        identity = (metadata.st_dev, metadata.st_ino)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise SnapshotError("owned snapshot staging directory is unsafe")
        if self.device is not None and identity != (self.device, self.inode):
            raise SnapshotError("owned snapshot staging identity changed")
        self.device, self.inode = identity


def _remove_owned_staging(scratch_fd: int, staging: _OwnedStaging) -> None:
    if staging.parent_synced:
        return

    budget_entries = 0

    def clear(directory_fd: int) -> None:
        nonlocal budget_entries
        names = _bounded_directory_names(
            directory_fd,
            remaining_entries=staging.max_entries - budget_entries,
            limit_error="incomplete snapshot exceeds the cleanup entry limit",
            read_error="failed to inspect an incomplete benchmark snapshot",
        )
        budget_entries += len(names)
        for name in names:
            try:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    child_fd = os.open(
                        name,
                        _required_open_flags(directory=True),
                        dir_fd=directory_fd,
                    )
                    try:
                        clear(child_fd)
                    finally:
                        os.close(child_fd)
                    os.rmdir(name, dir_fd=directory_fd)
                else:
                    os.unlink(name, dir_fd=directory_fd)
            except OSError as error:
                raise SnapshotError(
                    "failed to remove an incomplete benchmark snapshot"
                ) from error
        try:
            os.fsync(directory_fd)
        except OSError as error:
            raise SnapshotError(
                "failed to synchronize incomplete snapshot cleanup"
            ) from error

    if not staging.directory_removed:
        try:
            staging_fd = os.open(
                staging.name,
                _required_open_flags(directory=True),
                dir_fd=scratch_fd,
            )
        except FileNotFoundError:
            staging.directory_removed = True
        except OSError as error:
            raise SnapshotError(
                "failed to remove an incomplete benchmark snapshot"
            ) from error
        else:
            try:
                staging.bind(staging_fd)
                clear(staging_fd)
                current = os.stat(
                    staging.name,
                    dir_fd=scratch_fd,
                    follow_symlinks=False,
                )
                if (current.st_dev, current.st_ino) != (staging.device, staging.inode):
                    raise SnapshotError("owned snapshot staging identity changed")
                os.rmdir(staging.name, dir_fd=scratch_fd)
                staging.directory_removed = True
            except OSError as error:
                raise SnapshotError(
                    "failed to remove an incomplete benchmark snapshot"
                ) from error
            finally:
                os.close(staging_fd)

    try:
        os.fsync(scratch_fd)
    except OSError as error:
        raise SnapshotError("failed to synchronize incomplete snapshot cleanup") from error
    staging.parent_synced = True
    staging.created = False


class _StagingCleanupRecovery:
    def __init__(self, scratch_fd: int, staging: _OwnedStaging) -> None:
        self._scratch_fd: int | None = scratch_fd
        self._staging = staging

    @property
    def pending(self) -> bool:
        return self._scratch_fd is not None

    def retry(self) -> None:
        scratch_fd = self._scratch_fd
        if scratch_fd is None:
            return
        last_error: Exception | None = None
        for _ in range(_STAGING_CLEANUP_ATTEMPTS):
            try:
                _remove_owned_staging(scratch_fd, self._staging)
            except Exception as error:
                last_error = error
            else:
                self._scratch_fd = None
                try:
                    os.close(scratch_fd)
                except OSError:
                    pass
                return
        assert last_error is not None
        raise last_error

    def __del__(self) -> None:
        scratch_fd = self._scratch_fd
        if scratch_fd is not None:
            try:
                os.close(scratch_fd)
            except OSError:
                pass
            self._scratch_fd = None


def _cleanup_staging_after_failure(
    scratch_fd: int,
    staging: _OwnedStaging,
    primary_error: Exception,
) -> SnapshotCleanupError | None:
    cleanup_error: Exception | None = None
    for _ in range(_STAGING_CLEANUP_ATTEMPTS):
        try:
            _remove_owned_staging(scratch_fd, staging)
        except Exception as error:
            cleanup_error = error
        else:
            return None
    assert cleanup_error is not None
    return SnapshotCleanupError(
        primary_error=primary_error,
        cleanup_error=cleanup_error,
        recovery=_StagingCleanupRecovery(scratch_fd, staging),
    )


def _publish_staging(
    scratch_fd: int,
    staging: _OwnedStaging,
    final_name: str,
) -> None:
    try:
        os.rename(
            staging.name,
            final_name,
            src_dir_fd=scratch_fd,
            dst_dir_fd=scratch_fd,
        )
    except OSError as error:
        raise SnapshotError("benchmark snapshot could not be atomically published") from error
    staging.name = final_name
    try:
        os.fsync(scratch_fd)
    except OSError as error:
        raise SnapshotError("benchmark snapshot could not be atomically published") from error


def _fastembed_output_entry_limit(manifest: FastEmbedSnapshotManifest) -> int:
    directories = {
        PurePosixPath(*PurePosixPath(entry.relative_path).parts[:depth]).as_posix()
        for entry in manifest.files
        for depth in range(1, len(PurePosixPath(entry.relative_path).parts))
    }
    return len(manifest.files) + len(directories)


def materialize_fastembed_snapshot(
    source_root: _DirectoryInput,
    scratch_root: _DirectoryInput,
    manifest: FastEmbedSnapshotManifest,
) -> FastEmbedSnapshot:
    if not isinstance(manifest, FastEmbedSnapshotManifest):
        raise ValueError("FastEmbed snapshot manifest must be validated")
    source_path, source_fd = _acquire_directory(source_root)
    scratch_fd: int | None = None
    repository: _HuggingFaceRepository | None = None
    staging_name = f".fastembed-{uuid4().hex}.partial"
    final_name = f"fastembed-{manifest.model_content_sha256[:16]}-{uuid4().hex}"
    staging = _OwnedStaging(
        name=staging_name,
        max_entries=_fastembed_output_entry_limit(manifest),
    )
    try:
        scratch_path, scratch_fd = _ensure_private_scratch(scratch_root)
        _require_disjoint_roots(source_fd, scratch_fd)
        repository = _open_huggingface_repository(source_path, source_fd, manifest)
        source_before = _file_state(os.fstat(source_fd))
        os.mkdir(staging.name, mode=0o700, dir_fd=scratch_fd)
        staging.created = True
        staging_fd = os.open(
            staging.name,
            _required_open_flags(directory=True),
            dir_fd=scratch_fd,
        )
        try:
            os.fchmod(staging_fd, 0o700)
            staging.bind(staging_fd)
            first = _scan_allowlisted_files(
                source_fd,
                manifest,
                output_fd=staging_fd,
                source_path=source_path,
                repository=repository,
            )
            try:
                second = _scan_allowlisted_files(
                    source_fd,
                    manifest,
                    source_path=source_path,
                    repository=repository,
                )
            except SnapshotError as error:
                raise SnapshotError(
                    "FastEmbed source changed during snapshot creation"
                ) from error
            source_after = _file_state(os.fstat(source_fd))
            repository_changed = repository is not None and (
                repository.initial_state
                != _file_state(os.fstat(repository.file_descriptor))
            )
            if first != second or source_before != source_after or repository_changed:
                raise SnapshotError("FastEmbed source changed during snapshot creation")
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        _publish_staging(scratch_fd, staging, final_name)
        staging.created = False
    except Exception as primary_error:
        if staging.created and scratch_fd is not None:
            cleanup_pending = _cleanup_staging_after_failure(
                scratch_fd,
                staging,
                primary_error,
            )
            if cleanup_pending is not None:
                scratch_fd = None
                raise cleanup_pending from primary_error
        raise
    finally:
        if repository is not None:
            os.close(repository.file_descriptor)
        if scratch_fd is not None:
            os.close(scratch_fd)
        os.close(source_fd)
    return FastEmbedSnapshot(
        path=scratch_path / final_name,
        identity=FastEmbedSnapshotIdentity(
            model_name=manifest.model_name,
            model_repository=manifest.model_repository,
            model_revision=manifest.model_revision,
            runtime_version=manifest.runtime_version,
            algorithm_version=manifest.algorithm_version,
            dimensions=manifest.dimensions,
            model_content_sha256=manifest.model_content_sha256,
        ),
        manifest=manifest,
    )


@dataclass(slots=True)
class _TraversalBudget:
    limits: QdrantSnapshotLimits
    entries: int = 0
    bytes: int = 0

    @property
    def remaining_entries(self) -> int:
        return self.limits.max_entries - self.entries

    def add_entry(self) -> None:
        self.entries += 1
        if self.entries > self.limits.max_entries:
            raise SnapshotError("Qdrant snapshot exceeds the entry limit")

    def add_bytes(self, amount: int) -> None:
        self.bytes += amount
        if self.bytes > self.limits.max_total_bytes:
            raise SnapshotError("Qdrant snapshot exceeds the total byte limit")


def _scan_storage_tree(
    source_fd: int,
    limits: QdrantSnapshotLimits,
    *,
    output_fd: int | None = None,
) -> _TreeScan:
    budget = _TraversalBudget(limits)
    entries: list[dict[str, object]] = []
    states: list[tuple[str, _FileState]] = []

    def visit(
        current_source_fd: int,
        current_output_fd: int | None,
        prefix: tuple[str, ...],
    ) -> None:
        names = _bounded_directory_names(
            current_source_fd,
            remaining_entries=budget.remaining_entries,
            limit_error="Qdrant snapshot exceeds the entry limit",
            read_error="Qdrant snapshot directory could not be listed",
        )
        for name in names:
            if not name or name in {".", ".."} or "/" in name or "\x00" in name:
                raise SnapshotError("Qdrant snapshot contains an invalid entry name")
            relative_parts = (*prefix, name)
            depth = len(relative_parts)
            if depth > limits.max_depth:
                raise SnapshotError("Qdrant snapshot exceeds the depth limit")
            relative = PurePosixPath(*relative_parts).as_posix()
            budget.add_entry()
            try:
                metadata = os.stat(name, dir_fd=current_source_fd, follow_symlinks=False)
            except OSError as error:
                raise SnapshotError("Qdrant snapshot entry could not be inspected") from error
            initial_state = _file_state(metadata)
            if stat.S_ISDIR(metadata.st_mode):
                entries.append({"path": relative, "type": "directory"})
                states.append((relative, initial_state))
                try:
                    child_source_fd = os.open(
                        name,
                        _required_open_flags(directory=True),
                        dir_fd=current_source_fd,
                    )
                except OSError as error:
                    raise SnapshotError("Qdrant snapshot contains an unsafe directory") from error
                child_output_fd: int | None = None
                try:
                    if current_output_fd is not None:
                        os.mkdir(name, mode=0o700, dir_fd=current_output_fd)
                        os.fsync(current_output_fd)
                        child_output_fd = os.open(
                            name,
                            _required_open_flags(directory=True),
                            dir_fd=current_output_fd,
                        )
                        os.fchmod(child_output_fd, 0o700)
                    visit(child_source_fd, child_output_fd, relative_parts)
                    after = _file_state(os.fstat(child_source_fd))
                    current = _file_state(
                        os.stat(name, dir_fd=current_source_fd, follow_symlinks=False)
                    )
                    if after != initial_state or current != initial_state:
                        raise SnapshotError("Qdrant source changed during snapshot creation")
                    if child_output_fd is not None:
                        os.fsync(child_output_fd)
                finally:
                    if child_output_fd is not None:
                        os.close(child_output_fd)
                    os.close(child_source_fd)
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise SnapshotError(
                        "Qdrant snapshot entries must be regular single-link files"
                    )
                remaining_bytes = limits.max_total_bytes - budget.bytes
                if metadata.st_size > remaining_bytes:
                    raise SnapshotError("Qdrant snapshot exceeds the total byte limit")
                size, digest, state = _read_regular_file(
                    current_source_fd,
                    name,
                    max_bytes=min(limits.max_file_bytes, remaining_bytes),
                    output_parent_fd=current_output_fd,
                    output_name=name if current_output_fd is not None else None,
                )
                budget.add_bytes(size)
                entries.append(
                    {
                        "path": relative,
                        "sha256": digest,
                        "size_bytes": size,
                        "type": "file",
                    }
                )
                states.append((relative, state))
            else:
                raise SnapshotError(
                    "Qdrant snapshot entries must be directories or regular single-link files"
                )

    visit(source_fd, output_fd, ())
    return _TreeScan(
        entries=tuple(entries),
        states=tuple(states),
        total_bytes=budget.bytes,
    )


def snapshot_qdrant_storage(
    source_root: _DirectoryInput,
    scratch_root: _DirectoryInput,
    *,
    limits: QdrantSnapshotLimits | None = None,
) -> QdrantStorageSnapshot:
    resolved_limits = limits or QdrantSnapshotLimits()
    if not isinstance(resolved_limits, QdrantSnapshotLimits):
        raise ValueError("Qdrant snapshot limits must be validated")
    _source_path, source_fd = _acquire_directory(source_root)
    scratch_fd: int | None = None
    staging_name = f".qdrant-{uuid4().hex}.partial"
    staging = _OwnedStaging(
        name=staging_name,
        max_entries=resolved_limits.max_entries,
    )
    try:
        scratch_path, scratch_fd = _ensure_private_scratch(scratch_root)
        _require_disjoint_roots(source_fd, scratch_fd)
        source_before = _file_state(os.fstat(source_fd))
        os.mkdir(staging.name, mode=0o700, dir_fd=scratch_fd)
        staging.created = True
        staging_fd = os.open(
            staging.name,
            _required_open_flags(directory=True),
            dir_fd=scratch_fd,
        )
        try:
            os.fchmod(staging_fd, 0o700)
            staging.bind(staging_fd)
            first = _scan_storage_tree(
                source_fd,
                resolved_limits,
                output_fd=staging_fd,
            )
            second = _scan_storage_tree(source_fd, resolved_limits)
            source_after = _file_state(os.fstat(source_fd))
            if first != second or source_before != source_after:
                raise SnapshotError("Qdrant source changed during snapshot creation")
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        final_name = f"qdrant-{first.digest[:16]}-{uuid4().hex}"
        _publish_staging(scratch_fd, staging, final_name)
        staging.created = False
    except Exception as primary_error:
        if staging.created and scratch_fd is not None:
            cleanup_pending = _cleanup_staging_after_failure(
                scratch_fd,
                staging,
                primary_error,
            )
            if cleanup_pending is not None:
                scratch_fd = None
                raise cleanup_pending from primary_error
        raise
    finally:
        if scratch_fd is not None:
            os.close(scratch_fd)
        os.close(source_fd)
    return QdrantStorageSnapshot(
        path=scratch_path / final_name,
        snapshot_sha256=first.digest,
        entry_count=len(first.entries),
        total_bytes=first.total_bytes,
        source_device=source_before.device,
        source_inode=source_before.inode,
    )


def verify_qdrant_storage_snapshot(
    snapshot: QdrantStorageSnapshot,
    *,
    limits: QdrantSnapshotLimits | None = None,
) -> None:
    if not isinstance(snapshot, QdrantStorageSnapshot):
        raise ValueError("Qdrant storage snapshot must be validated")
    resolved_limits = limits or QdrantSnapshotLimits()
    _path, snapshot_fd = _open_directory_path(snapshot.path)
    try:
        metadata = os.fstat(snapshot_fd)
        if stat.S_IMODE(metadata.st_mode) != 0o700 or metadata.st_uid != os.getuid():
            raise SnapshotError("Qdrant snapshot root is not private")
        scan = _scan_storage_tree(snapshot_fd, resolved_limits)
    finally:
        os.close(snapshot_fd)
    if (
        scan.digest != snapshot.snapshot_sha256
        or len(scan.entries) != snapshot.entry_count
        or scan.total_bytes != snapshot.total_bytes
    ):
        raise SnapshotError("Qdrant snapshot digest does not match its attestation")
    entry_types = {str(entry["path"]): str(entry["type"]) for entry in scan.entries}
    if any(
        stat.S_IMODE(state.mode) != (0o600 if entry_types[path] == "file" else 0o700)
        for path, state in scan.states
    ):
        raise SnapshotError("Qdrant snapshot permissions are not private")


def verify_fastembed_snapshot(snapshot: FastEmbedSnapshot) -> None:
    if not isinstance(snapshot, FastEmbedSnapshot):
        raise ValueError("FastEmbed snapshot must be validated")
    expected_identity = FastEmbedSnapshotIdentity(
        model_name=snapshot.manifest.model_name,
        model_repository=snapshot.manifest.model_repository,
        model_revision=snapshot.manifest.model_revision,
        runtime_version=snapshot.manifest.runtime_version,
        algorithm_version=snapshot.manifest.algorithm_version,
        dimensions=snapshot.manifest.dimensions,
        model_content_sha256=snapshot.manifest.model_content_sha256,
    )
    if snapshot.identity != expected_identity:
        raise SnapshotError("FastEmbed snapshot identity does not match its manifest")
    _path, snapshot_fd = _open_directory_path(snapshot.path)
    try:
        metadata = os.fstat(snapshot_fd)
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise SnapshotError("FastEmbed snapshot root is not private")
        _scan_allowlisted_files(snapshot_fd, snapshot.manifest)
        tree = _scan_storage_tree(
            snapshot_fd,
            QdrantSnapshotLimits(
                max_entries=_MAX_FASTEMBED_FILES * 2,
                max_depth=16,
                max_total_bytes=_MAX_FASTEMBED_TOTAL_BYTES,
                max_file_bytes=_MAX_FASTEMBED_FILE_BYTES,
            ),
        )
    finally:
        os.close(snapshot_fd)
    expected_files = {item.relative_path for item in snapshot.manifest.files}
    expected_directories = {
        PurePosixPath(*PurePosixPath(path).parts[:depth]).as_posix()
        for path in expected_files
        for depth in range(1, len(PurePosixPath(path).parts))
    }
    actual_files = {
        str(entry["path"]) for entry in tree.entries if entry["type"] == "file"
    }
    actual_directories = {
        str(entry["path"]) for entry in tree.entries if entry["type"] == "directory"
    }
    if actual_files != expected_files or actual_directories != expected_directories:
        raise SnapshotError("FastEmbed snapshot contains content outside its manifest")
    modes = {path: state.mode for path, state in tree.states}
    if any(stat.S_IMODE(modes[path]) != 0o600 for path in actual_files) or any(
        stat.S_IMODE(modes[path]) != 0o700 for path in actual_directories
    ):
        raise SnapshotError("FastEmbed snapshot permissions are not private")


def _decode_fastembed_snapshot_manifest(raw: bytes) -> FastEmbedSnapshotManifest:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise SnapshotError("FastEmbed manifest is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {
        "algorithm_version",
        "dimensions",
        "files",
        "model_name",
        "model_repository",
        "model_revision",
        "runtime_version",
    }:
        raise SnapshotError("FastEmbed manifest has unsupported fields")
    raw_files = payload.pop("files")
    if not isinstance(raw_files, list):
        raise SnapshotError("FastEmbed manifest files must be a list")
    try:
        files = tuple(
            FastEmbedFileManifestEntry(**item)
            for item in raw_files
            if isinstance(item, dict)
        )
        if len(files) != len(raw_files):
            raise ValueError("invalid FastEmbed manifest file entry")
        return FastEmbedSnapshotManifest(files=files, **payload)
    except (TypeError, ValueError) as error:
        raise SnapshotError("FastEmbed manifest contract is invalid") from error


def load_fastembed_snapshot_manifest(path: Path) -> FastEmbedSnapshotManifest:
    """Load a manifest file without following any path component."""
    absolute = Path(path).absolute()
    parent_path = absolute.parent
    _parent, parent_fd = _open_directory_path(parent_path)
    try:
        raw = _read_regular_content(
            parent_fd,
            absolute.name,
            max_bytes=_MAX_MANIFEST_BYTES,
        )
    finally:
        os.close(parent_fd)
    return _decode_fastembed_snapshot_manifest(raw)


def load_reviewed_fastembed_snapshot_manifest() -> FastEmbedSnapshotManifest:
    """Load VideoScope's checked-in, reviewable manifest for the pinned model."""
    resource = resources.files("videoscope.benchmark").joinpath(
        "manifests",
        _REVIEWED_FASTEMBED_MANIFEST,
    )
    try:
        resource_path = Path(os.fspath(resource))
    except TypeError as error:
        raise SnapshotError("reviewed FastEmbed manifest is not a filesystem resource") from error
    return load_fastembed_snapshot_manifest(resource_path)
