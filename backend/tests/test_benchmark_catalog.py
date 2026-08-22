from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import pytest

from videoscope.benchmark import (
    AssetProvenance,
    AssetResolutionError,
    BenchmarkAsset,
    BenchmarkDataError,
    BenchmarkDataset,
    BenchmarkInterval,
    DatasetCatalog,
    LocalAssetResolver,
    QueryCase,
    dataset_revision,
    write_dataset,
)


def _dataset() -> BenchmarkDataset:
    asset = BenchmarkAsset(
        asset_id="asset-a",
        sha256="a" * 64,
        byte_size=1_024,
        duration_seconds=30.0,
        provenance=AssetProvenance(
            source="Public fixture",
            source_uri="https://example.test/asset-a.mp4",
            license_id="CC-BY-4.0",
        ),
    )
    return BenchmarkDataset(
        schema_version=1,
        dataset_id="portable-core",
        dataset_version="1.0.0",
        description="Portable fixture",
        assets=(asset,),
        cases=(
            QueryCase(
                case_id="case-a",
                query="a made basket",
                asset_ids=(asset.asset_id,),
                domain="basketball",
                modalities=("sports", "visual"),
                label_quality="gold",
                split_group="match-a",
                relevant_intervals=(
                    BenchmarkInterval(asset.asset_id, 10.0, 12.0),
                ),
            ),
        ),
    )


def test_dataset_catalog_validates_and_imports_by_canonical_revision(tmp_path) -> None:
    source = tmp_path / "source.json"
    dataset = _dataset()
    write_dataset(source, dataset)
    catalog = DatasetCatalog(tmp_path / "catalog")

    imported = catalog.import_file(source)

    assert imported.dataset == dataset
    assert imported.revision == dataset_revision(dataset)
    assert imported.path == (
        tmp_path
        / "catalog"
        / dataset.dataset_id
        / f"{dataset_revision(dataset)}.json"
    )
    assert catalog.load(dataset.dataset_id, imported.revision) == dataset


def test_dataset_catalog_import_is_idempotent_for_reordered_equivalent_data(
    tmp_path,
) -> None:
    source = tmp_path / "source.json"
    reordered_source = tmp_path / "reordered.json"
    dataset = _dataset()
    write_dataset(source, dataset)
    reordered_case = replace(
        dataset.cases[0],
        modalities=tuple(reversed(dataset.cases[0].modalities)),
    )
    reordered = replace(dataset, cases=(reordered_case,))
    write_dataset(reordered_source, reordered)
    catalog = DatasetCatalog(tmp_path / "catalog")

    first = catalog.import_file(source)
    second = catalog.import_file(reordered_source)

    assert first.revision == second.revision
    assert first.path == second.path
    assert second.dataset == first.dataset


def test_dataset_catalog_concurrent_import_is_single_immutable_revision(tmp_path) -> None:
    source = tmp_path / "source.json"
    dataset = _dataset()
    write_dataset(source, dataset)
    catalog = DatasetCatalog(tmp_path / "catalog")

    with ThreadPoolExecutor(max_workers=8) as executor:
        imported = tuple(executor.map(lambda _: catalog.import_file(source), range(16)))

    assert {item.revision for item in imported} == {dataset_revision(dataset)}
    assert {item.path for item in imported} == {imported[0].path}
    assert tuple(imported[0].path.parent.iterdir()) == (imported[0].path,)


def test_dataset_catalog_refuses_invalid_source_without_partial_import(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{broken", encoding="utf-8")
    catalog_root = tmp_path / "catalog"

    with pytest.raises(BenchmarkDataError):
        DatasetCatalog(catalog_root).import_file(source)

    assert not catalog_root.exists()


@pytest.mark.parametrize("symlink_level", ["root", "dataset"])
def test_dataset_catalog_rejects_directory_symlink_escape(
    tmp_path,
    symlink_level: str,
) -> None:
    source = tmp_path / "source.json"
    write_dataset(source, _dataset())
    outside = tmp_path / "outside"
    outside.mkdir()
    catalog_root = tmp_path / "catalog"
    if symlink_level == "root":
        catalog_root.symlink_to(outside, target_is_directory=True)
    else:
        catalog_root.mkdir()
        (catalog_root / _dataset().dataset_id).symlink_to(
            outside,
            target_is_directory=True,
        )

    with pytest.raises(BenchmarkDataError, match="symbolic link"):
        DatasetCatalog(catalog_root).import_file(source)

    assert tuple(outside.iterdir()) == ()


def test_dataset_catalog_rejects_manifest_symlink_on_load(tmp_path) -> None:
    dataset = _dataset()
    revision = dataset_revision(dataset)
    outside = tmp_path / "outside.json"
    write_dataset(outside, dataset)
    target = tmp_path / "catalog" / dataset.dataset_id / f"{revision}.json"
    target.parent.mkdir(parents=True)
    target.symlink_to(outside)

    with pytest.raises(BenchmarkDataError, match="symbolic link"):
        DatasetCatalog(tmp_path / "catalog").load(dataset.dataset_id, revision)


def test_dataset_catalog_rejects_tampered_revision_address(tmp_path) -> None:
    dataset = _dataset()
    wrong_revision = "b" * 64
    target = tmp_path / "catalog" / dataset.dataset_id / f"{wrong_revision}.json"
    write_dataset(target, dataset)

    with pytest.raises(BenchmarkDataError, match="revision"):
        DatasetCatalog(tmp_path / "catalog").load(dataset.dataset_id, wrong_revision)


@dataclass(frozen=True)
class FakeRepositoryAsset:
    id: str
    sha256: str
    byte_size: int
    duration_seconds: float
    video_id: str


class FakeAssetRepository:
    def __init__(self, *assets: FakeRepositoryAsset) -> None:
        self.assets = assets
        self.failure: Exception | None = None

    def find_assets_by_sha256(self, digest: str) -> tuple[FakeRepositoryAsset, ...]:
        del digest
        if self.failure is not None:
            raise self.failure
        return self.assets


def _repository_asset() -> FakeRepositoryAsset:
    asset = _dataset().assets[0]
    return FakeRepositoryAsset(
        id=f"sha256:{asset.sha256}",
        sha256=asset.sha256,
        byte_size=asset.byte_size,
        duration_seconds=asset.duration_seconds,
        video_id="video-local-a",
    )


def test_local_asset_resolver_maps_portable_identity_to_repository_asset() -> None:
    portable = _dataset().assets[0]

    resolved = LocalAssetResolver(FakeAssetRepository(_repository_asset())).resolve(
        portable
    )

    assert resolved.asset_id == portable.asset_id
    assert resolved.video_id == "video-local-a"
    assert resolved.sha256 == portable.sha256
    assert resolved.byte_size == portable.byte_size
    assert resolved.duration_seconds == portable.duration_seconds


@pytest.mark.parametrize(
    ("repository_asset", "code"),
    [
        (None, "asset_missing"),
        (replace(_repository_asset(), id="sha256:" + "b" * 64), "asset_content_id_mismatch"),
        (replace(_repository_asset(), sha256="b" * 64), "asset_sha_mismatch"),
        (replace(_repository_asset(), byte_size=2_048), "asset_size_mismatch"),
        (
            replace(_repository_asset(), duration_seconds=29.0),
            "asset_duration_mismatch",
        ),
    ],
)
def test_local_asset_resolver_fails_closed_on_identity_mismatch(
    repository_asset,
    code: str,
) -> None:  # type: ignore[no-untyped-def]
    repository = (
        FakeAssetRepository()
        if repository_asset is None
        else FakeAssetRepository(repository_asset)
    )
    with pytest.raises(AssetResolutionError) as caught:
        LocalAssetResolver(repository).resolve(
            _dataset().assets[0]
        )

    assert caught.value.code == code
    assert "private" not in str(caught.value)


def test_local_asset_resolver_sanitizes_repository_failure() -> None:
    repository = FakeAssetRepository(_repository_asset())
    repository.failure = RuntimeError("private database at /Users/person/videoscope.sqlite3")

    with pytest.raises(AssetResolutionError) as caught:
        LocalAssetResolver(repository).resolve(_dataset().assets[0])

    assert caught.value.code == "asset_lookup_failed"
    assert "/Users/person" not in str(caught.value)


def test_local_asset_resolver_bounds_untrusted_candidate_enumeration() -> None:
    class OverflowingRepository:
        def __init__(self) -> None:
            self.consumed = 0

        def find_assets_by_sha256(self, _digest: str):  # type: ignore[no-untyped-def]
            for index in range(1_000_000):
                self.consumed += 1
                yield replace(
                    _repository_asset(),
                    video_id=f"video-{index}",
                )

    repository = OverflowingRepository()

    with pytest.raises(AssetResolutionError) as caught:
        LocalAssetResolver(repository).resolve(_dataset().assets[0])  # type: ignore[arg-type]

    assert caught.value.code == "asset_lookup_overflow"
    assert repository.consumed == 3


def test_local_asset_resolver_rejects_ambiguous_content_bindings() -> None:
    first = _repository_asset()
    second = replace(first, video_id="video-local-b")

    with pytest.raises(AssetResolutionError) as caught:
        LocalAssetResolver(FakeAssetRepository(first, second)).resolve(
            _dataset().assets[0]
        )

    assert caught.value.code == "asset_binding_ambiguous"


def test_local_asset_resolver_accepts_explicit_portable_alias_binding() -> None:
    first = _repository_asset()
    second = replace(first, video_id="video-local-b")

    resolved = LocalAssetResolver(
        FakeAssetRepository(first, second),
        bindings={"asset-a": "video-local-b"},
    ).resolve(_dataset().assets[0])

    assert resolved.asset_id == "asset-a"
    assert resolved.repository_asset_id == "sha256:" + "a" * 64
    assert resolved.video_id == "video-local-b"


def test_local_asset_resolver_deduplicates_equivalent_repository_rows() -> None:
    asset = _repository_asset()

    resolved = LocalAssetResolver(FakeAssetRepository(asset, asset)).resolve(
        _dataset().assets[0]
    )

    assert resolved.video_id == "video-local-a"
