from __future__ import annotations

from dataclasses import dataclass
import hmac
from itertools import islice
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence, runtime_checkable

from .schema import BenchmarkAsset, BenchmarkDataError, BenchmarkDataset, _require_id
from .serialization import dataset_from_dict, dataset_revision, dataset_to_dict
from .storage import load_dataset, write_dataset


_REVISION_RE = re.compile(r"^[0-9a-f]{64}$")
_DURATION_ABSOLUTE_TOLERANCE_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class ImportedDataset:
    dataset: BenchmarkDataset
    revision: str
    path: Path


class DatasetCatalog:
    """Import validated manifests into an immutable revision-addressed catalog."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def import_file(self, source: Path) -> ImportedDataset:
        # Validation deliberately happens before the catalog root is created.
        dataset = load_dataset(Path(source))
        normalized = dataset_from_dict(dataset_to_dict(dataset, canonical=True))
        revision = dataset_revision(normalized)
        target = self._path(normalized.dataset_id, revision)
        self._validate_parent_layout(normalized.dataset_id)
        try:
            write_dataset(target, normalized)
        except FileExistsError:
            existing = self.load(normalized.dataset_id, revision)
            if dataset_revision(existing) != revision:
                raise BenchmarkDataError(
                    "catalog entry does not match its canonical dataset revision"
                )
            normalized = existing
        self._validate_parent_layout(normalized.dataset_id)
        self._validate_target(target)
        return ImportedDataset(normalized, revision, target)

    def load(self, dataset_id: str, revision: str) -> BenchmarkDataset:
        target = self._path(dataset_id, revision)
        self._validate_parent_layout(dataset_id)
        self._validate_target(target)
        dataset = load_dataset(target)
        if dataset.dataset_id != dataset_id:
            raise BenchmarkDataError("catalog dataset identity does not match its path")
        if dataset_revision(dataset) != revision:
            raise BenchmarkDataError("catalog dataset revision does not match its path")
        return dataset

    def _path(self, dataset_id: str, revision: str) -> Path:
        _require_id(dataset_id, "dataset_id")
        if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
            raise BenchmarkDataError(
                "dataset revision must be a lowercase 64-character SHA-256"
            )
        return self.root / dataset_id / f"{revision}.json"

    def _validate_parent_layout(self, dataset_id: str) -> None:
        if self.root.is_symlink():
            raise BenchmarkDataError("dataset catalog root must not be a symbolic link")
        if self.root.exists() and not self.root.is_dir():
            raise BenchmarkDataError("dataset catalog root must be a directory")
        dataset_root = self.root / dataset_id
        if dataset_root.is_symlink():
            raise BenchmarkDataError("dataset catalog entry must not be a symbolic link")
        if dataset_root.exists() and not dataset_root.is_dir():
            raise BenchmarkDataError("dataset catalog entry must be a directory")
        if self.root.exists() and dataset_root.exists():
            root = self.root.resolve(strict=True)
            if not dataset_root.resolve(strict=True).is_relative_to(root):
                raise BenchmarkDataError("dataset catalog entry escapes its root")

    def _validate_target(self, target: Path) -> None:
        if target.is_symlink():
            raise BenchmarkDataError("dataset catalog manifest must not be a symbolic link")
        if not target.exists():
            raise FileNotFoundError(target)
        if not target.is_file():
            raise BenchmarkDataError("dataset catalog manifest must be a regular file")
        root = self.root.resolve(strict=True)
        if not target.resolve(strict=True).is_relative_to(root):
            raise BenchmarkDataError("dataset catalog manifest escapes its root")


@runtime_checkable
class RepositoryAsset(Protocol):
    """Read-only projection needed from the future repository Asset model."""

    id: str
    sha256: str
    byte_size: int
    duration_seconds: float
    video_id: str


class RepositoryAssetLookup(Protocol):
    def find_assets_by_sha256(self, digest: str) -> Sequence[RepositoryAsset]: ...


@dataclass(frozen=True, slots=True)
class ResolvedAsset:
    asset_id: str
    repository_asset_id: str
    video_id: str
    sha256: str
    byte_size: int
    duration_seconds: float


class AssetResolutionError(RuntimeError):
    """A sanitized local binding failure safe to persist in a run manifest."""

    def __init__(self, code: str) -> None:
        _require_id(code, "asset resolution diagnostic code")
        self.code = code
        super().__init__(f"benchmark asset resolution failed ({code})")


class LocalAssetResolver:
    """Bind portable aliases to private repository assets by content identity.

    Portable ``asset_id`` values are dataset aliases and intentionally need not
    equal the repository's authoritative ``sha256:<digest>`` asset id.  When a
    digest is attached to multiple local video rows, the alias must be bound
    explicitly unless all returned rows identify the same video.
    """

    def __init__(
        self,
        repository: RepositoryAssetLookup,
        *,
        bindings: Mapping[str, str] | None = None,
    ) -> None:
        self._repository = repository
        copied_bindings = dict(bindings or {})
        for asset_id, video_id in copied_bindings.items():
            _require_id(asset_id, "asset binding alias")
            _require_id(video_id, "asset binding video_id")
        self._bindings: Mapping[str, str] = MappingProxyType(copied_bindings)

    def resolve(self, portable: BenchmarkAsset) -> ResolvedAsset:
        if not isinstance(portable, BenchmarkAsset):
            raise AssetResolutionError("asset_contract_invalid")
        explicit_video_id = self._bindings.get(portable.asset_id)
        candidate_limit = 1 if explicit_video_id is not None else 2
        try:
            bounded_lookup = getattr(
                self._repository,
                "find_assets_by_sha256_bounded",
                None,
            )
            if callable(bounded_lookup):
                candidates_value = bounded_lookup(
                    portable.sha256,
                    limit=candidate_limit,
                    video_id=explicit_video_id,
                )
                enumeration_limit = candidate_limit
            else:
                candidates_value = self._repository.find_assets_by_sha256(
                    portable.sha256
                )
                enumeration_limit = 32 if explicit_video_id is not None else 2
            candidates = tuple(
                islice(iter(candidates_value), enumeration_limit + 1)
            )
        except Exception as exc:
            raise AssetResolutionError("asset_lookup_failed") from exc
        if len(candidates) > enumeration_limit:
            raise AssetResolutionError("asset_lookup_overflow")
        if not candidates:
            raise AssetResolutionError("asset_missing")

        if explicit_video_id is not None:
            matching_candidates = tuple(
                item
                for item in candidates
                if _candidate_video_id(item) == explicit_video_id
            )
            if not matching_candidates:
                raise AssetResolutionError("asset_binding_missing")
            validated = tuple(
                self._validate_candidate(portable, item)
                for item in matching_candidates
            )
            return validated[0]

        validated: list[ResolvedAsset] = []
        failures: list[AssetResolutionError] = []
        for item in candidates:
            try:
                validated.append(self._validate_candidate(portable, item))
            except AssetResolutionError as exc:
                failures.append(exc)
        if not validated:
            if failures:
                raise AssetResolutionError(failures[0].code)
            raise AssetResolutionError("asset_missing")
        by_video_id = {candidate.video_id: candidate for candidate in validated}
        if len(by_video_id) != 1:
            raise AssetResolutionError("asset_binding_ambiguous")
        return next(iter(by_video_id.values()))

    @staticmethod
    def _validate_candidate(
        portable: BenchmarkAsset,
        candidate: RepositoryAsset,
    ) -> ResolvedAsset:
        try:
            repository_id = candidate.id
            digest = candidate.sha256
            byte_size = candidate.byte_size
            duration = candidate.duration_seconds
            video_id = candidate.video_id
        except Exception as exc:
            raise AssetResolutionError("asset_record_invalid") from exc

        expected_repository_id = f"sha256:{portable.sha256}"
        if repository_id != expected_repository_id:
            raise AssetResolutionError("asset_content_id_mismatch")
        if (
            not isinstance(digest, str)
            or not hmac.compare_digest(digest, portable.sha256)
        ):
            raise AssetResolutionError("asset_sha_mismatch")
        if type(byte_size) is not int or byte_size != portable.byte_size:
            raise AssetResolutionError("asset_size_mismatch")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise AssetResolutionError("asset_duration_mismatch")
        duration_value = float(duration)
        if not math.isfinite(duration_value) or not math.isclose(
            duration_value,
            portable.duration_seconds,
            rel_tol=1e-6,
            abs_tol=_DURATION_ABSOLUTE_TOLERANCE_SECONDS,
        ):
            raise AssetResolutionError("asset_duration_mismatch")
        try:
            _require_id(video_id, "repository asset video_id")
        except BenchmarkDataError as exc:
            raise AssetResolutionError("asset_record_invalid") from exc
        return ResolvedAsset(
            asset_id=portable.asset_id,
            repository_asset_id=repository_id,
            video_id=video_id,
            sha256=digest,
            byte_size=byte_size,
            duration_seconds=duration_value,
        )


def _candidate_video_id(candidate: RepositoryAsset) -> str | None:
    try:
        value = candidate.video_id
    except Exception:
        return None
    return value if isinstance(value, str) else None
