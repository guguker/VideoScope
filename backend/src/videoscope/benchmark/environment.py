from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from threading import Lock
from types import MappingProxyType
from typing import Literal, Self

from videoscope.config import AppSettings
from videoscope.indexing_attestation import attest_indexing_toolchain
from videoscope.media.ffmpeg import FFmpeg
from videoscope.model_manifest import model_identity, model_revision
from videoscope.providers.lighthouse_worker import LighthouseWorkerClient
from videoscope.providers.qwen_video import QwenVideoReranker
from videoscope.providers.qwen_worker import QwenWorkerClient
from videoscope.providers.vision_worker_client import VisionWorkerClient
from videoscope.providers.whisper import snapshot_whisper_prompt_from_content
from videoscope.repository import Repository
from videoscope.runtime import (
    create_indexing_specifications_from_prompt_snapshot,
    create_vision_worker_specification,
)
from videoscope.search.service import SearchService
from videoscope.search.temporal_refinement import TemporalRefiner
from videoscope.search.text_matching import normalize_text
from videoscope.search.vector_index import QdrantVectorIndex
from videoscope.search.visual_index import SiglipVisualIndex

from .adapter import (
    BENCHMARK_PRODUCT_ENVIRONMENT_COMPONENT_ID,
    ProductBenchmarkSearchAdapter,
)
from .catalog import LocalAssetResolver
from .product_runtime import (
    ProductRuntimeSnapshot,
    ProductSnapshotCleanupError,
    open_product_runtime_snapshot,
)
from .profiles import FROZEN_PROFILES, BenchmarkProfile
from .schema import ComponentIdentity
from .snapshots import (
    RetainedDirectory,
    load_reviewed_fastembed_snapshot_manifest,
    materialize_fastembed_snapshot,
    snapshot_qdrant_storage,
    verify_fastembed_snapshot,
)


_ENVIRONMENT_PROTOCOL_VERSION = 2
_SUPPORTED_EXECUTION_MODE = "warm"
_MAX_GLOSSARY_BYTES = 1024 * 1024
_MAX_SCRATCH_ENTRIES = 200_000
_MAX_SCRATCH_DEPTH = 64
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


class BenchmarkEnvironmentError(RuntimeError):
    """The product benchmark cannot be opened or closed without ambiguity."""


class BenchmarkEnvironmentCleanupError(BenchmarkEnvironmentError):
    """Setup failed and ownership remains available for an explicit close retry."""

    def __init__(
        self,
        message: str,
        *,
        environment: ProductBenchmarkEnvironment,
    ) -> None:
        super().__init__(message)
        self.environment = environment


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class BenchmarkEnvironmentIdentity:
    profile_id: str
    profile_identity: str
    execution_mode: Literal["warm"]
    product_snapshot_sha256: str
    fastembed_model_content_sha256: str
    qdrant_snapshot_sha256: str
    qdrant_attestation_sha256: str
    indexing_specification_hashes: tuple[tuple[str, str], ...]
    glossary_sha256: str
    semantic_text_min_score: float
    visual_min_score: float
    worker_input_root_identities: tuple[tuple[str, str], ...]
    protocol_version: int = _ENVIRONMENT_PROTOCOL_VERSION
    environment_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        profile = FROZEN_PROFILES.get(self.profile_id)
        if profile is None or self.profile_identity != profile.identity:
            raise ValueError("benchmark environment profile identity is invalid")
        if self.execution_mode != _SUPPORTED_EXECUTION_MODE:
            raise ValueError("benchmark environment execution mode is invalid")
        if self.protocol_version != _ENVIRONMENT_PROTOCOL_VERSION:
            raise ValueError("benchmark environment protocol version is unsupported")
        for name in (
            "product_snapshot_sha256",
            "fastembed_model_content_sha256",
            "qdrant_snapshot_sha256",
            "qdrant_attestation_sha256",
            "glossary_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"benchmark environment {name} is invalid")
        expected_names = ("objects", "ocr", "scenes", "speech", "text_vectors")
        if tuple(name for name, _digest in self.indexing_specification_hashes) != (
            expected_names
        ) or any(
            not _is_sha256(digest)
            for _name, digest in self.indexing_specification_hashes
        ):
            raise ValueError("benchmark environment indexing identities are invalid")
        for value in (self.semantic_text_min_score, self.visual_min_score):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("benchmark environment score threshold is invalid")
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError("benchmark environment score threshold is invalid")
        roles = tuple(role for role, _identity in self.worker_input_root_identities)
        if (
            roles != tuple(sorted(roles))
            or len(roles) != len(set(roles))
            or any(role not in {"qwen", "vision"} for role in roles)
            or any(
                not identity.startswith("sha256:")
                or not _is_sha256(identity.removeprefix("sha256:"))
                for _role, identity in self.worker_input_root_identities
            )
        ):
            raise ValueError("benchmark worker input root identities are invalid")
        object.__setattr__(self, "environment_sha256", _canonical_digest(self.content_dict))

    @property
    def content_dict(self) -> dict[str, object]:
        return {
            "execution_mode": self.execution_mode,
            "fastembed_model_content_sha256": self.fastembed_model_content_sha256,
            "glossary_sha256": self.glossary_sha256,
            "indexing_specification_hashes": {
                name: digest for name, digest in self.indexing_specification_hashes
            },
            "product_snapshot_sha256": self.product_snapshot_sha256,
            "profile_id": self.profile_id,
            "profile_identity": self.profile_identity,
            "protocol_version": self.protocol_version,
            "qdrant_attestation_sha256": self.qdrant_attestation_sha256,
            "qdrant_snapshot_sha256": self.qdrant_snapshot_sha256,
            "semantic_text_min_score": float(self.semantic_text_min_score),
            "visual_min_score": float(self.visual_min_score),
            "worker_input_root_identities": {
                role: identity
                for role, identity in self.worker_input_root_identities
            },
        }

    @property
    def canonical_dict(self) -> dict[str, object]:
        return {
            **self.content_dict,
            "environment_sha256": self.environment_sha256,
        }

    @property
    def identity(self) -> str:
        return (
            f"benchmark-product-environment@{self.protocol_version}:"
            f"{self.environment_sha256}"
        )


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    mode: int
    links: int
    owner: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _ProfileProviderWiring:
    visual_search: object | None = None
    temporal_refiner: object | None = None
    moment_search: object | None = None
    evaluation_rerankers: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    worker_input_root_identities: tuple[tuple[str, str], ...] = ()
    worker_source_probes: tuple[tuple[str, Callable[[Path], object]], ...] = ()


def _file_identity(value: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=value.st_dev,
        inode=value.st_ino,
        mode=value.st_mode,
        links=value.st_nlink,
        owner=value.st_uid,
        size=value.st_size,
        modified_ns=value.st_mtime_ns,
        changed_ns=value.st_ctime_ns,
    )


def _directory_object_identity(value: _FileIdentity) -> tuple[int, int, int, int]:
    """Fields that identify a directory while its owned contents legitimately change."""
    return value.device, value.inode, value.mode, value.owner


def _absolute_path(value: Path, *, label: str) -> Path:
    try:
        raw = os.fspath(value)
        if not raw or "\x00" in raw:
            raise ValueError
        return Path(os.path.abspath(raw))
    except (TypeError, ValueError, OSError) as error:
        raise BenchmarkEnvironmentError(f"{label} path is invalid") from error


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise BenchmarkEnvironmentError(
            "platform does not support no-follow benchmark scratch access"
        )
    return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


def _open_directory_path(path: Path, *, label: str) -> tuple[Path, int]:
    absolute = _absolute_path(path, label=label)
    flags = _directory_flags()
    descriptor: int | None = None
    try:
        descriptor = os.open(absolute.anchor, flags)
        for component in absolute.parts[1:]:
            observed = os.stat(
                component,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            child = os.open(component, flags, dir_fd=descriptor)
            opened = os.fstat(child)
            current = os.stat(
                component,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if not (
                stat.S_ISDIR(observed.st_mode)
                and _file_identity(observed) == _file_identity(opened)
                and _file_identity(opened) == _file_identity(current)
            ):
                os.close(child)
                raise BenchmarkEnvironmentError(f"{label} path changed while opening")
            os.close(descriptor)
            descriptor = child
        return absolute, descriptor
    except BenchmarkEnvironmentError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise BenchmarkEnvironmentError(
            f"{label} directory is unsafe or unavailable"
        ) from error


class _PrivateScratch:
    def __init__(
        self,
        *,
        path: Path,
        parent_descriptor: int,
        root_descriptor: int,
        name: str,
        identity: _FileIdentity,
    ) -> None:
        self.path = path
        self._parent_descriptor = parent_descriptor
        self._root_descriptor = root_descriptor
        self._name = name
        self._identity = identity
        self._root_unlinked = False
        self._parent_synced = False
        self._descriptors = [root_descriptor, parent_descriptor]

    @classmethod
    def create(cls, parent: Path) -> _PrivateScratch:
        parent_path, parent_descriptor = _open_directory_path(
            parent,
            label="benchmark scratch parent",
        )
        metadata = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            os.close(parent_descriptor)
            raise BenchmarkEnvironmentError(
                "benchmark scratch parent must be an owner-only 0700 directory"
            )
        name: str | None = None
        root_descriptor: int | None = None
        try:
            for _attempt in range(64):
                candidate = f"benchmark-environment-{secrets.token_hex(16)}"
                try:
                    os.mkdir(candidate, mode=0o700, dir_fd=parent_descriptor)
                except FileExistsError:
                    continue
                name = candidate
                break
            if name is None:
                raise BenchmarkEnvironmentError(
                    "private benchmark scratch name could not be allocated"
                )
            root_descriptor = os.open(
                name,
                _directory_flags(),
                dir_fd=parent_descriptor,
            )
            os.fchmod(root_descriptor, 0o700)
            os.fsync(parent_descriptor)
            identity = _file_identity(os.fstat(root_descriptor))
            current = _file_identity(
                os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            )
            if (
                identity != current
                or not stat.S_ISDIR(identity.mode)
                or identity.owner != os.geteuid()
                or stat.S_IMODE(identity.mode) != 0o700
            ):
                raise BenchmarkEnvironmentError(
                    "private benchmark scratch identity is invalid"
                )
            return cls(
                path=parent_path / name,
                parent_descriptor=parent_descriptor,
                root_descriptor=root_descriptor,
                name=name,
                identity=identity,
            )
        except BaseException:
            if root_descriptor is not None:
                os.close(root_descriptor)
            if name is not None:
                try:
                    os.rmdir(name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
                except OSError:
                    pass
            os.close(parent_descriptor)
            raise

    def _validate_root_binding(self) -> None:
        opened = _file_identity(os.fstat(self._root_descriptor))
        try:
            current = _file_identity(
                os.stat(
                    self._name,
                    dir_fd=self._parent_descriptor,
                    follow_symlinks=False,
                )
            )
        except OSError as error:
            raise BenchmarkEnvironmentError(
                "private benchmark scratch binding is unavailable"
            ) from error
        expected = _directory_object_identity(self._identity)
        if (
            _directory_object_identity(opened) != expected
            or _directory_object_identity(current) != expected
        ):
            raise BenchmarkEnvironmentError(
                "private benchmark scratch binding changed"
            )

    @staticmethod
    def _clear_directory(
        descriptor: int,
        *,
        depth: int,
        budget: list[int],
    ) -> None:
        if depth > _MAX_SCRATCH_DEPTH:
            raise BenchmarkEnvironmentError(
                "private benchmark scratch exceeds the cleanup depth limit"
            )
        remaining = _MAX_SCRATCH_ENTRIES - budget[0]
        names: list[str] = []
        try:
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    if len(names) >= remaining:
                        raise BenchmarkEnvironmentError(
                            "private benchmark scratch exceeds the cleanup entry limit"
                        )
                    names.append(entry.name)
        except BenchmarkEnvironmentError:
            raise
        except OSError as error:
            raise BenchmarkEnvironmentError(
                "private benchmark scratch could not be listed"
            ) from error
        for name in names:
            budget[0] += 1
            if not name or name in {".", ".."} or "/" in name or "\x00" in name:
                raise BenchmarkEnvironmentError(
                    "private benchmark scratch contains an invalid entry"
                )
            try:
                observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise BenchmarkEnvironmentError(
                    "private benchmark scratch entry could not be inspected"
                ) from error
            if stat.S_ISDIR(observed.st_mode):
                try:
                    child = os.open(name, _directory_flags(), dir_fd=descriptor)
                except OSError as error:
                    raise BenchmarkEnvironmentError(
                        "private benchmark scratch directory is unsafe"
                    ) from error
                try:
                    opened = os.fstat(child)
                    current = os.stat(
                        name,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                    if not (
                        _file_identity(observed) == _file_identity(opened)
                        and _file_identity(opened) == _file_identity(current)
                    ):
                        raise BenchmarkEnvironmentError(
                            "private benchmark scratch directory changed"
                        )
                    _PrivateScratch._clear_directory(
                        child,
                        depth=depth + 1,
                        budget=budget,
                    )
                    current = os.stat(
                        name,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                    if _file_identity(os.fstat(child)) != _file_identity(current):
                        raise BenchmarkEnvironmentError(
                            "private benchmark scratch directory changed"
                        )
                finally:
                    os.close(child)
                try:
                    os.rmdir(name, dir_fd=descriptor)
                    os.fsync(descriptor)
                except OSError as error:
                    raise BenchmarkEnvironmentError(
                        "private benchmark scratch directory could not be removed"
                    ) from error
            elif stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
                try:
                    current = os.stat(
                        name,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                    if _file_identity(current) != _file_identity(observed):
                        raise BenchmarkEnvironmentError(
                            "private benchmark scratch file changed"
                        )
                    os.unlink(name, dir_fd=descriptor)
                    os.fsync(descriptor)
                except OSError as error:
                    raise BenchmarkEnvironmentError(
                        "private benchmark scratch file could not be removed"
                    ) from error
            else:
                raise BenchmarkEnvironmentError(
                    "private benchmark scratch contains an unsupported entry"
                )

    def remove(self) -> None:
        if not self._root_unlinked:
            self._validate_root_binding()
            self._clear_directory(
                self._root_descriptor,
                depth=0,
                budget=[0],
            )
            self._validate_root_binding()
            try:
                os.rmdir(self._name, dir_fd=self._parent_descriptor)
            except OSError as error:
                raise BenchmarkEnvironmentError(
                    "private benchmark scratch root could not be removed"
                ) from error
            self._root_unlinked = True
        if not self._parent_synced:
            try:
                os.fsync(self._parent_descriptor)
            except OSError as error:
                raise BenchmarkEnvironmentError(
                    "private benchmark scratch removal could not be synchronized"
                ) from error
            self._parent_synced = True
        while self._descriptors:
            descriptor = self._descriptors[0]
            try:
                os.close(descriptor)
            except OSError as error:
                raise BenchmarkEnvironmentError(
                    "private benchmark scratch descriptors could not be closed"
                ) from error
            self._descriptors.pop(0)


class _FrozenSearchLexicon:
    def __init__(
        self,
        entries: Mapping[str, list[str]],
        *,
        state: Literal["invalid", "missing", "ready"],
    ) -> None:
        self._entries = MappingProxyType(
            {key: tuple(values) for key, values in entries.items()}
        )
        self.state = state
        self.sha256 = _canonical_digest(
            {
                "entries": [
                    [key, list(values)]
                    for key, values in self._entries.items()
                ],
                "state": self.state,
            }
        )

    def read(self) -> dict[str, list[str]]:
        return {key: list(values) for key, values in self._entries.items()}

    def expand(self, query: str) -> list[str]:
        normalized_query = normalize_text(query)
        output = [query.strip()]
        for canonical, aliases in self._entries.items():
            variants = [canonical, *aliases]
            normalized_variants = [normalize_text(value) for value in variants]
            if any(value and value in normalized_query for value in normalized_variants):
                output.extend(variants)
        return list(dict.fromkeys(value for value in output if value))

    def replace(self, _entries: dict[str, list[str]]) -> None:
        raise RuntimeError("benchmark glossary snapshot is read-only")


def _load_frozen_glossary(
    data_root: RetainedDirectory,
) -> tuple[_FrozenSearchLexicon, bytes | None]:
    try:
        raw = data_root.read_optional_regular_file(
            "search-glossary.json",
            max_bytes=_MAX_GLOSSARY_BYTES,
        )
        if raw is None:
            return _FrozenSearchLexicon({}, state="missing"), None
    except (ValueError, OSError, RuntimeError) as error:
        raise BenchmarkEnvironmentError(
            "benchmark glossary is unsafe or unavailable"
        ) from error
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError):
        return _FrozenSearchLexicon({}, state="invalid"), raw
    if not isinstance(payload, dict):
        return _FrozenSearchLexicon({}, state="invalid"), raw
    entries: dict[str, list[str]] = {}
    for term, aliases in payload.items():
        if type(term) is not str or not term.strip() or not isinstance(aliases, list):
            continue
        entries[term] = [
            str(alias)
            for alias in aliases
            if str(alias).strip()
        ]
    return _FrozenSearchLexicon(entries, state="ready"), raw


def _indexing_hashes(specifications: object) -> tuple[tuple[str, str], ...]:
    hashes: list[tuple[str, str]] = []
    for name in ("objects", "ocr", "scenes", "speech", "text_vectors"):
        digest = getattr(getattr(specifications, name, None), "specification_hash", None)
        if not _is_sha256(digest):
            raise BenchmarkEnvironmentError(
                "indexing specification identity is unavailable"
            )
        hashes.append((name, digest))
    return tuple(hashes)


def _validate_qdrant_attestation(
    vector_index: object,
    *,
    embedding: object,
    fastembed_snapshot_identity: object,
    qdrant_snapshot_sha256: str,
    fastembed_model_content_sha256: str,
) -> str:
    if getattr(embedding, "strict_no_fallback", None) is not True:
        raise BenchmarkEnvironmentError(
            "benchmark embedding does not attest strict no-fallback mode"
        )
    embedding_identity = getattr(embedding, "benchmark_identity", None)
    if callable(embedding_identity):
        embedding_identity = embedding_identity()
    expected_embedding_identity = {
        "embedding_identity": getattr(embedding, "identity", None),
        "model_name": getattr(fastembed_snapshot_identity, "model_name", None),
        "model_repository": getattr(
            fastembed_snapshot_identity,
            "model_repository",
            None,
        ),
        "model_revision": getattr(
            fastembed_snapshot_identity,
            "model_revision",
            None,
        ),
        "runtime_version": getattr(
            fastembed_snapshot_identity,
            "runtime_version",
            None,
        ),
        "algorithm_version": getattr(
            fastembed_snapshot_identity,
            "algorithm_version",
            None,
        ),
        "dimensions": getattr(fastembed_snapshot_identity, "dimensions", None),
        "model_content_sha256": fastembed_model_content_sha256,
    }
    if (
        not isinstance(embedding_identity, Mapping)
        or set(embedding_identity) != set(expected_embedding_identity)
        or dict(embedding_identity) != expected_embedding_identity
    ):
        raise BenchmarkEnvironmentError("benchmark embedding attestation is invalid")
    try:
        attestation = vector_index.benchmark_attestation
    except Exception as error:
        raise BenchmarkEnvironmentError(
            "Qdrant benchmark attestation is unavailable"
        ) from error
    if not isinstance(attestation, Mapping):
        raise BenchmarkEnvironmentError("Qdrant benchmark attestation is invalid")
    index = attestation.get("index")
    attested_embedding = attestation.get("embedding")
    expected_index_hash = getattr(
        getattr(vector_index, "index_specification", None),
        "specification_hash",
        None,
    )
    if (
        set(attestation) != {
            "schema_version",
            "provider",
            "strict_no_fallback",
            "embedding",
            "index",
        }
        or attestation.get("schema_version") != 1
        or attestation.get("provider") != "qdrant"
        or attestation.get("strict_no_fallback") is not True
        or not isinstance(attested_embedding, Mapping)
        or dict(attested_embedding) != dict(embedding_identity)
        or not isinstance(index, Mapping)
        or set(index) != {
            "index_specification_hash",
            "collection_name",
            "snapshot_sha256",
        }
        or index.get("snapshot_sha256") != qdrant_snapshot_sha256
        or index.get("index_specification_hash") != expected_index_hash
        or not _is_sha256(expected_index_hash)
        or type(index.get("collection_name")) is not str
        or not str(index.get("collection_name")).strip()
    ):
        raise BenchmarkEnvironmentError("Qdrant benchmark attestation is invalid")
    try:
        canonical = json.loads(
            json.dumps(
                attestation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise BenchmarkEnvironmentError("Qdrant benchmark attestation is invalid") from error
    return _canonical_digest(canonical)


class ProductBenchmarkEnvironment:
    """Owned, read-only product resources for one lexical benchmark process."""

    def __init__(
        self,
        *,
        product_snapshot: ProductRuntimeSnapshot,
        scratch: _PrivateScratch | None = None,
    ) -> None:
        self.repository = product_snapshot.repository
        self._product_snapshot = product_snapshot
        self._scratch = scratch
        self._data_root: RetainedDirectory | None = None
        self._media_root: RetainedDirectory | None = None
        self._scratch_root: RetainedDirectory | None = None
        self._borrowed_descriptors: list[int] = []
        self._fastembed_snapshot: object | None = None
        self._vector_index: object | None = None
        self._search_adapter: ProductBenchmarkSearchAdapter | None = None
        self._asset_resolver: LocalAssetResolver | None = None
        self._identity: BenchmarkEnvironmentIdentity | None = None
        self._worker_source_probes: tuple[
            tuple[str, Callable[[Path], object]], ...
        ] = ()
        self._adapter_closed = False
        self._vector_index_closed = False
        self._scratch_root_closed = True
        self._media_root_closed = True
        self._data_root_closed = True
        self._scratch_removed = scratch is None
        self._product_closed = False
        self._state_lock = Lock()

    @property
    def scratch_root(self) -> Path:
        if self._scratch is None:
            raise BenchmarkEnvironmentError("private benchmark scratch is unavailable")
        return self._scratch.path

    def _attach_scratch(self, scratch: _PrivateScratch) -> None:
        if self._scratch is not None or not self._scratch_removed:
            raise BenchmarkEnvironmentError("private benchmark scratch is already attached")
        self._scratch = scratch
        self._scratch_removed = False

    def _attach_worker_source_probes(
        self,
        probes: tuple[tuple[str, Callable[[Path], object]], ...],
    ) -> None:
        roles = tuple(role for role, _probe in probes)
        if (
            roles != tuple(sorted(roles))
            or len(roles) != len(set(roles))
            or any(role not in {"qwen", "vision"} for role in roles)
            or any(not callable(probe) for _role, probe in probes)
        ):
            raise BenchmarkEnvironmentError(
                "benchmark worker source probe contract is invalid"
            )
        self._worker_source_probes = probes

    def probe_worker_sources(self) -> tuple[str, ...]:
        """Prove configured workers can read this environment's private spool."""

        if not self._worker_source_probes:
            return ()
        if self._scratch is None or self._scratch_removed:
            raise BenchmarkEnvironmentError("private benchmark scratch is unavailable")
        marker = self._scratch.path / "worker-source-probe.bin"
        body = b"videoscope-worker-source-probe-v1\n"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                marker,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            written = os.write(descriptor, body)
            if written != len(body):
                raise OSError("short worker probe write")
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            completed: list[str] = []
            for role, probe in self._worker_source_probes:
                probe(marker)
                completed.append(role)
            return tuple(completed)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            raise BenchmarkEnvironmentError(
                "benchmark worker source boundary probe failed"
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                marker.unlink(missing_ok=True)
            except OSError as error:
                raise BenchmarkEnvironmentError(
                    "benchmark worker source probe cleanup failed"
                ) from error

    def _retain_product_directory(
        self,
        *,
        path: Path,
        duplicate: Callable[[], int],
        role: Literal["data", "media"],
    ) -> RetainedDirectory:
        descriptor = duplicate()
        if isinstance(descriptor, bool) or not isinstance(descriptor, int):
            raise BenchmarkEnvironmentError(
                f"product benchmark {role} directory capability is invalid"
            )
        self._borrowed_descriptors.append(descriptor)
        capability = RetainedDirectory.retain(path, descriptor)
        if role == "data":
            if self._data_root is not None or not self._data_root_closed:
                capability.close()
                raise BenchmarkEnvironmentError(
                    "product benchmark data directory is already retained"
                )
            self._data_root = capability
            self._data_root_closed = False
        else:
            if self._media_root is not None or not self._media_root_closed:
                capability.close()
                raise BenchmarkEnvironmentError(
                    "product benchmark media directory is already retained"
                )
            self._media_root = capability
            self._media_root_closed = False
        self._close_borrowed_descriptors()
        return capability

    def _attach_scratch_root(self, capability: RetainedDirectory) -> None:
        if self._scratch_root is not None or not self._scratch_root_closed:
            capability.close()
            raise BenchmarkEnvironmentError(
                "private benchmark scratch capability is already retained"
            )
        self._scratch_root = capability
        self._scratch_root_closed = False

    def _close_borrowed_descriptors(self) -> None:
        while self._borrowed_descriptors:
            descriptor = self._borrowed_descriptors[0]
            try:
                os.close(descriptor)
            except OSError as error:
                raise BenchmarkEnvironmentError(
                    "borrowed benchmark directory capability could not be closed"
                ) from error
            self._borrowed_descriptors.pop(0)

    @staticmethod
    def _close_retained_directory(
        capability: RetainedDirectory | None,
        *,
        label: str,
    ) -> None:
        if capability is None:
            return
        try:
            capability.close()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            raise BenchmarkEnvironmentError(
                f"{label} directory capability could not be closed"
            ) from error
        if not capability.closed:
            raise BenchmarkEnvironmentError(
                f"{label} directory capability close is uncertain"
            )

    @property
    def search_adapter(self) -> ProductBenchmarkSearchAdapter:
        if self._search_adapter is None:
            raise BenchmarkEnvironmentError("benchmark search adapter is unavailable")
        return self._search_adapter

    @property
    def asset_resolver(self) -> LocalAssetResolver:
        if self._asset_resolver is None:
            raise BenchmarkEnvironmentError("benchmark asset resolver is unavailable")
        return self._asset_resolver

    @property
    def identity(self) -> BenchmarkEnvironmentIdentity:
        if self._identity is None:
            raise BenchmarkEnvironmentError("benchmark environment identity is unavailable")
        return self._identity

    @property
    def is_closed(self) -> bool:
        return self._product_closed

    def __enter__(self) -> Self:
        if self.is_closed:
            raise BenchmarkEnvironmentError("benchmark environment is already closed")
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        with self._state_lock:
            if self._product_closed:
                return
            if not self._adapter_closed:
                if self._search_adapter is not None:
                    try:
                        self._search_adapter.close()
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except BaseException as error:
                        raise BenchmarkEnvironmentError(
                            "benchmark search adapter could not be closed"
                        ) from error
                self._adapter_closed = True
            verification_failure: BenchmarkEnvironmentError | None = None
            if self._fastembed_snapshot is not None:
                try:
                    verify_fastembed_snapshot(self._fastembed_snapshot)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as error:
                    verification_failure = BenchmarkEnvironmentError(
                        "FastEmbed benchmark snapshot failed final verification"
                    )
                    verification_failure.__cause__ = error
                    verification_failure.__suppress_context__ = True
            try:
                if not self._vector_index_closed:
                    if self._vector_index is not None:
                        close = getattr(self._vector_index, "close", None)
                        if not callable(close):
                            raise BenchmarkEnvironmentError(
                                "Qdrant benchmark close contract is invalid"
                            )
                        try:
                            close()
                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except BaseException as error:
                            raise BenchmarkEnvironmentError(
                                "Qdrant benchmark snapshot could not be closed"
                            ) from error
                    self._vector_index_closed = True
                self._close_borrowed_descriptors()
                if not self._scratch_root_closed:
                    self._close_retained_directory(
                        self._scratch_root,
                        label="private benchmark scratch",
                    )
                    self._scratch_root_closed = True
                if not self._media_root_closed:
                    self._close_retained_directory(
                        self._media_root,
                        label="product benchmark media",
                    )
                    self._media_root_closed = True
                if not self._data_root_closed:
                    self._close_retained_directory(
                        self._data_root,
                        label="product benchmark data",
                    )
                    self._data_root_closed = True
                if not self._scratch_removed:
                    if self._scratch is None:
                        raise BenchmarkEnvironmentError(
                            "private benchmark scratch cleanup contract is invalid"
                        )
                    self._scratch.remove()
                    self._scratch_removed = True
                try:
                    self._product_snapshot.close()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as error:
                    raise BenchmarkEnvironmentError(
                        "product benchmark snapshot could not release ownership"
                    ) from error
                self._product_closed = True
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as cleanup_error:
                if verification_failure is not None:
                    if isinstance(cleanup_error, Exception):
                        cause: BaseException = ExceptionGroup(
                            "benchmark attestation and cleanup both failed",
                            [verification_failure, cleanup_error],
                        )
                    else:
                        cause = cleanup_error
                    raise BenchmarkEnvironmentError(
                        "benchmark final verification failed and cleanup requires retry"
                    ) from cause
                raise
            if verification_failure is not None:
                raise verification_failure


def _profile_contract(profile_id: str, execution_mode: str) -> BenchmarkProfile:
    profile = FROZEN_PROFILES.get(profile_id)
    if profile is None or execution_mode != _SUPPORTED_EXECUTION_MODE:
        raise BenchmarkEnvironmentError(
            "product benchmark environment supports only frozen profiles in warm mode"
        )
    return profile


def _attested_benchmark_ffmpeg(
    *,
    ffmpeg_binary: Path | None = None,
    ffprobe_binary: Path | None = None,
) -> tuple[FFmpeg, str] | None:
    """Return an exact read-only media runtime or decline optional capabilities."""
    if (ffmpeg_binary is None) != (ffprobe_binary is None):
        return None
    attestation_arguments: dict[str, object] = {}
    if ffmpeg_binary is not None and ffprobe_binary is not None:
        attestation_arguments = {
            "ffmpeg_binary": ffmpeg_binary,
            "ffprobe_binary": ffprobe_binary,
        }
    try:
        toolchain = attest_indexing_toolchain(**attestation_arguments)
        identity = toolchain.verify_current()
        ffmpeg = toolchain.create_ffmpeg()
    except Exception:
        return None
    if (
        not isinstance(ffmpeg, FFmpeg)
        or type(identity) is not str
        or not identity.startswith("sha256:")
        or not _is_sha256(identity.removeprefix("sha256:"))
    ):
        return None
    return ffmpeg, identity


def _profile_provider_wiring(
    *,
    profile: BenchmarkProfile,
    settings: AppSettings,
    repository: Repository,
    data_root: RetainedDirectory,
    media_root: RetainedDirectory,
    scratch_path: Path,
    scratch_parent: Path,
) -> _ProfileProviderWiring:
    """Wire only frozen-plan readers; never build or activate an artifact."""
    plan = profile.search_plan
    vision_client: VisionWorkerClient | None = None
    qwen_client: QwenWorkerClient | None = None
    visual_search: SiglipVisualIndex | None = None
    if plan.visual_search == "dense_siglip":
        if settings.vision_worker_endpoint:
            vision_client = VisionWorkerClient(
                endpoint=settings.vision_worker_endpoint,
                api_key=settings.vision_worker_api_key or "",
                input_root=scratch_parent,
                specification=create_vision_worker_specification(settings),
                timeout=settings.vision_worker_timeout,
            )
        visual_search = SiglipVisualIndex(
            data_root.child("visual-index").stable_path,
            model_name=settings.siglip_model,
            model_revision=model_revision(settings.siglip_model),
            batch_size=settings.siglip_batch_size,
            sample_step=settings.visual_index_step,
            max_width=settings.visual_index_max_width,
            inference_client=vision_client,
        )

    moment_search: LighthouseWorkerClient | None = None
    if plan.lighthouse and settings.lighthouse_endpoint:
        moment_search = LighthouseWorkerClient(
            endpoint=settings.lighthouse_endpoint,
            api_key=settings.lighthouse_api_key or "",
            input_root=media_root.stable_path,
            cache_dir=data_root.child("cache").stable_path,
            timeout=settings.lighthouse_timeout,
        )

    needs_media_runtime = (
        plan.temporal_refinement and vision_client is not None
    ) or (
        plan.reranker == "qwen"
        and settings.qwen_video_endpoint is not None
        and settings.qwen_video_model is not None
    )
    media_runtime = (
        _attested_benchmark_ffmpeg(
            ffmpeg_binary=getattr(settings, "ffmpeg_binary", None),
            ffprobe_binary=getattr(settings, "ffprobe_binary", None),
        )
        if needs_media_runtime
        else None
    )
    temporal_refiner: TemporalRefiner | None = None
    if (
        plan.temporal_refinement
        and vision_client is not None
        and visual_search is not None
        and media_runtime is not None
    ):
        ffmpeg, ffmpeg_identity = media_runtime
        worker_identity = vision_client.identity.get("siglip_specification_hash")
        scorer_identity = (
            f"{visual_search.model_identity}#{visual_search.specification_identity}"
        )
        runtime_identity = (
            f"vision-worker:{worker_identity}"
            if type(worker_identity) is str and worker_identity
            else None
        )
        if runtime_identity is not None:
            temporal_refiner = TemporalRefiner(
                repository=repository,
                extractor=ffmpeg,
                scorer=visual_search,
                temp_dir=scratch_path / "temporal-refinement",
                top_candidates=settings.temporal_refinement_candidates,
                sample_step=settings.temporal_refinement_step,
                min_score=settings.visual_min_score,
                ffmpeg_identity=ffmpeg_identity,
                scorer_identity=scorer_identity,
                runtime_identity=runtime_identity,
            )

    rerankers: dict[str, object] = {}
    if (
        plan.reranker == "qwen"
        and settings.qwen_video_endpoint is not None
        and settings.qwen_video_model is not None
        and media_runtime is not None
    ):
        ffmpeg, _ffmpeg_identity = media_runtime
        revision = model_revision(settings.qwen_video_model)
        qwen_client = QwenWorkerClient(
            endpoint=settings.qwen_video_endpoint,
            api_key=settings.qwen_video_api_key or "",
            input_root=scratch_parent,
            expected_model_identity=model_identity(
                settings.qwen_video_model,
                revision,
            ),
            timeout=settings.qwen_video_timeout,
        )
        rerankers["qwen"] = QwenVideoReranker(
            model_name=settings.qwen_video_model,
            model_revision=revision,
            repository=repository,
            extractor=ffmpeg,
            temp_dir=scratch_path / "qwen-inputs",
            cache_dir=scratch_path / "qwen-cache",
            top_candidates=plan.reranker_candidate_limit,
            context_seconds=settings.qwen_video_context_seconds,
            min_clip_seconds=settings.qwen_video_min_clip_seconds,
            max_clip_seconds=settings.qwen_video_max_clip_seconds,
            frame_count=settings.qwen_video_frame_count,
            video_fps=settings.qwen_video_fps,
            inference_client=qwen_client,
            allow_in_process=False,
        )

    # InternVideo is intentionally not benchmark-wired until it has the same
    # local, source-bound attestation contract as the other frozen providers.
    root_identities: list[tuple[str, str]] = []
    probes: list[tuple[str, Callable[[Path], object]]] = []
    if qwen_client is not None:
        qwen_root_identity = getattr(qwen_client, "input_root_sha256", None)
        qwen_probe = getattr(qwen_client, "probe_source", None)
        if not _is_sha256(qwen_root_identity) or not callable(qwen_probe):
            raise BenchmarkEnvironmentError(
                "Qwen benchmark worker source boundary is unavailable"
            )
        root_identities.append(("qwen", f"sha256:{qwen_root_identity}"))
        probes.append(("qwen", qwen_probe))
    if vision_client is not None:
        vision_root_identity = getattr(vision_client, "input_root_identity", None)
        vision_probe = getattr(vision_client, "probe_source", None)
        if (
            type(vision_root_identity) is not str
            or not vision_root_identity.startswith("sha256:")
            or not _is_sha256(vision_root_identity.removeprefix("sha256:"))
            or not callable(vision_probe)
        ):
            raise BenchmarkEnvironmentError(
                "Vision benchmark worker source boundary is unavailable"
            )
        root_identities.append(("vision", vision_root_identity))
        probes.append(("vision", vision_probe))
    return _ProfileProviderWiring(
        visual_search=visual_search,
        temporal_refiner=temporal_refiner,
        moment_search=moment_search,
        evaluation_rerankers=MappingProxyType(rerankers),
        worker_input_root_identities=tuple(sorted(root_identities)),
        worker_source_probes=tuple(sorted(probes, key=lambda item: item[0])),
    )


def open_product_benchmark_environment(
    settings: AppSettings,
    scratch_parent: Path,
    *,
    profile_id: str = "lexical_qdrant",
    execution_mode: Literal["cold", "warm"] = _SUPPORTED_EXECUTION_MODE,
) -> ProductBenchmarkEnvironment:
    """Open the first concrete benchmark profile without touching product state."""

    if not isinstance(settings, AppSettings):
        raise ValueError("benchmark environment settings must be a validated AppSettings")
    profile = _profile_contract(
        profile_id,
        execution_mode,
    )
    data_dir = _absolute_path(settings.data_dir, label="product data")
    # Freeze every derived product path against one canonical data root. This
    # keeps validated relative settings independent from later cwd changes.
    resolved_settings = settings.model_copy(update={"data_dir": data_dir})
    database_path = _absolute_path(
        resolved_settings.database_path,
        label="product database",
    )
    media_root = _absolute_path(resolved_settings.media_dir, label="product media")
    scratch_parent_absolute = _absolute_path(
        Path(scratch_parent),
        label="benchmark scratch parent",
    )

    try:
        product_snapshot = open_product_runtime_snapshot(
            data_dir=data_dir,
            database_path=database_path,
            media_root=media_root,
            scratch_parent=scratch_parent_absolute,
        )
    except ProductSnapshotCleanupError:
        # The product snapshot layer deliberately exposes its retained owner and
        # retry method. Preserve that exact recovery contract for the caller.
        raise
    scratch: _PrivateScratch | None = None
    environment = ProductBenchmarkEnvironment(product_snapshot=product_snapshot)
    try:
        duplicate_data_root = getattr(
            product_snapshot,
            "duplicate_data_root_descriptor",
            None,
        )
        duplicate_media_root = getattr(
            product_snapshot,
            "duplicate_media_root_descriptor",
            None,
        )
        if not callable(duplicate_data_root) or not callable(duplicate_media_root):
            raise BenchmarkEnvironmentError(
                "product snapshot does not expose retained data and media roots"
            )
        data_root = environment._retain_product_directory(
            path=data_dir,
            duplicate=duplicate_data_root,
            role="data",
        )
        media_access_root = environment._retain_product_directory(
            path=media_root,
            duplicate=duplicate_media_root,
            role="media",
        )

        scratch = _PrivateScratch.create(scratch_parent_absolute)
        environment._attach_scratch(scratch)
        scratch_root = RetainedDirectory.retain(
            scratch.path,
            scratch._root_descriptor,
        )
        environment._attach_scratch_root(scratch_root)

        manifest = load_reviewed_fastembed_snapshot_manifest()
        manifest_model = getattr(manifest, "model_name", None)
        manifest_dimensions = getattr(manifest, "dimensions", None)
        if (
            manifest_model != resolved_settings.text_embedding_model
            or manifest_dimensions != resolved_settings.text_embedding_dimensions
        ):
            raise BenchmarkEnvironmentError(
                "reviewed FastEmbed manifest differs from validated settings"
            )
        fastembed_snapshot = materialize_fastembed_snapshot(
            data_root.child("models/fastembed"),
            scratch_root,
            manifest,
        )
        environment._fastembed_snapshot = fastembed_snapshot
        embedding = fastembed_snapshot.create_embedding()
        ensure_embedding_ready = getattr(embedding, "ensure_ready", None)
        if not callable(ensure_embedding_ready) or ensure_embedding_ready() is not True:
            raise BenchmarkEnvironmentError(
                "strict FastEmbed benchmark model could not be warmed"
            )
        fastembed_digest = getattr(
            getattr(fastembed_snapshot, "identity", None),
            "model_content_sha256",
            None,
        )
        if (
            not _is_sha256(fastembed_digest)
            or fastembed_digest != getattr(manifest, "model_content_sha256", None)
        ):
            raise BenchmarkEnvironmentError(
                "FastEmbed snapshot identity is unavailable"
            )

        qdrant_snapshot = snapshot_qdrant_storage(
            data_root.child("qdrant/text"),
            scratch_root,
        )
        qdrant_digest = getattr(qdrant_snapshot, "snapshot_sha256", None)
        if not _is_sha256(qdrant_digest):
            raise BenchmarkEnvironmentError("Qdrant snapshot identity is unavailable")
        vector_index = QdrantVectorIndex.open_existing_snapshot(
            qdrant_snapshot,
            embedding=embedding,
        )
        environment._vector_index = vector_index
        attestation_digest = _validate_qdrant_attestation(
            vector_index,
            embedding=embedding,
            fastembed_snapshot_identity=fastembed_snapshot.identity,
            qdrant_snapshot_sha256=qdrant_digest,
            fastembed_model_content_sha256=fastembed_digest,
        )

        glossary_before, glossary_content_before = _load_frozen_glossary(data_root)
        prompt_before = snapshot_whisper_prompt_from_content(
            resolved_settings.whisper_initial_prompt,
            glossary_content_before,
            glossary_state=(
                "missing" if glossary_content_before is None else "ready"
            ),
        )
        if prompt_before.glossary_state != glossary_before.state:
            raise BenchmarkEnvironmentError(
                "benchmark glossary and indexing prompt states differ"
            )
        specifications = create_indexing_specifications_from_prompt_snapshot(
            resolved_settings,
            prompt_before,
        )
        glossary_after, glossary_content_after = _load_frozen_glossary(data_root)
        prompt_after = snapshot_whisper_prompt_from_content(
            resolved_settings.whisper_initial_prompt,
            glossary_content_after,
            glossary_state=(
                "missing" if glossary_content_after is None else "ready"
            ),
        )
        if (
            glossary_content_before != glossary_content_after
            or glossary_before.sha256 != glossary_after.sha256
            or glossary_before.read() != glossary_after.read()
            or prompt_before != prompt_after
        ):
            raise BenchmarkEnvironmentError(
                "benchmark glossary changed while indexing identities were captured"
            )
        indexing_hashes = _indexing_hashes(specifications)
        provider_wiring = _profile_provider_wiring(
            profile=profile,
            settings=resolved_settings,
            repository=product_snapshot.repository,
            data_root=data_root,
            media_root=media_access_root,
            scratch_path=scratch.path,
            scratch_parent=scratch_parent_absolute,
        )
        environment._attach_worker_source_probes(
            provider_wiring.worker_source_probes
        )
        search = SearchService(
            product_snapshot.repository,
            vector_index,
            moment_search=provider_wiring.moment_search,
            visual_search=provider_wiring.visual_search,
            lexicon=glossary_before,
            temporal_refiner=provider_wiring.temporal_refiner,
            candidate_reranker=None,
            evaluation_rerankers=provider_wiring.evaluation_rerankers,
            semantic_text_min_score=resolved_settings.semantic_text_min_score,
            visual_min_score=resolved_settings.visual_min_score,
            specification_resolver=lambda resolved=specifications: resolved,
            media_root=media_root,
            media_access_root=media_access_root.stable_path,
        )
        environment._asset_resolver = LocalAssetResolver(product_snapshot.repository)
        product_digest = getattr(product_snapshot.identity, "snapshot_sha256", None)
        if not _is_sha256(product_digest):
            raise BenchmarkEnvironmentError("product snapshot identity is unavailable")
        environment._identity = BenchmarkEnvironmentIdentity(
            profile_id=profile.profile_id,
            profile_identity=profile.identity,
            execution_mode="warm",
            product_snapshot_sha256=product_digest,
            fastembed_model_content_sha256=fastembed_digest,
            qdrant_snapshot_sha256=qdrant_digest,
            qdrant_attestation_sha256=attestation_digest,
            indexing_specification_hashes=indexing_hashes,
            glossary_sha256=glossary_before.sha256,
            semantic_text_min_score=resolved_settings.semantic_text_min_score,
            visual_min_score=resolved_settings.visual_min_score,
            worker_input_root_identities=(
                provider_wiring.worker_input_root_identities
            ),
        )
        environment._search_adapter = ProductBenchmarkSearchAdapter(
            search,
            environment_identity=ComponentIdentity(
                BENCHMARK_PRODUCT_ENVIRONMENT_COMPONENT_ID,
                environment._identity.identity,
            ),
        )
        return environment
    except BaseException as setup_error:
        try:
            environment.close()
        except BaseException as cleanup_error:
            if isinstance(setup_error, (KeyboardInterrupt, SystemExit)):
                raise setup_error from cleanup_error
            if environment.is_closed:
                if isinstance(setup_error, Exception) and isinstance(
                    cleanup_error,
                    Exception,
                ):
                    completed_cleanup_cause: BaseException = ExceptionGroup(
                        f"benchmark environment setup failed: {setup_error}",
                        [setup_error, cleanup_error],
                    )
                else:
                    completed_cleanup_cause = cleanup_error
                raise BenchmarkEnvironmentError(
                    "benchmark environment setup failed and final attestation rejected "
                    "the fully cleaned snapshot"
                ) from completed_cleanup_cause
            if isinstance(setup_error, Exception) and isinstance(cleanup_error, Exception):
                combined: BaseException = ExceptionGroup(
                    f"benchmark environment setup failed: {setup_error}",
                    [setup_error, cleanup_error],
                )
            else:
                combined = cleanup_error
            raise BenchmarkEnvironmentCleanupError(
                "benchmark environment setup failed and cleanup requires retry",
                environment=environment,
            ) from combined
        raise


__all__ = [
    "BenchmarkEnvironmentCleanupError",
    "BenchmarkEnvironmentError",
    "BenchmarkEnvironmentIdentity",
    "ProductBenchmarkEnvironment",
    "open_product_benchmark_environment",
]
