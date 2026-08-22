from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
import stat

import pytest

from videoscope.artifacts import AssetIdentityError
from videoscope import repository as repository_module
from videoscope.repository import LATEST_SCHEMA_VERSION, Repository


def _retained_directory_path(directory: Path) -> tuple[Path, int]:
    descriptor = os.open(
        directory,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    metadata = os.fstat(descriptor)
    retained = Path(f"/.vol/{metadata.st_dev}/{metadata.st_ino}")
    try:
        verifier = os.open(
            retained,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError:
        os.close(descriptor)
        pytest.skip("retained volume paths are unavailable on this platform")
    else:
        os.close(verifier)
    return retained, descriptor


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int, int, str | None]]:
    snapshot: dict[str, tuple[int, int, int, str | None]] = {}
    for path in sorted((root, *root.rglob("*"))):
        metadata = path.stat(follow_symlinks=False)
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if stat.S_ISREG(metadata.st_mode)
            else None
        )
        snapshot[str(path.relative_to(root))] = (
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
            digest,
        )
    return snapshot


def _tree_layout(root: Path) -> dict[str, int]:
    return {
        str(path.relative_to(root)): path.stat(follow_symlinks=False).st_mode
        for path in sorted((root, *root.rglob("*")))
    }


def _create_repository_with_media(
    root: Path,
    *,
    persist_asset: bool = True,
) -> tuple[Path, Path, bytes]:
    data_dir = root / "data"
    media_root = data_dir / "media"
    media_root.mkdir(parents=True)
    content = b"benchmark-media-content"
    media_path = media_root / "video-1.mp4"
    media_path.write_bytes(content)
    database_path = data_dir / "videoscope.sqlite3"
    repository = Repository(database_path)
    repository.initialize()
    video_payload = {
        "video_id": "video-1",
        "original_name": "match.mp4",
        "stored_name": media_path.name,
        "media_path": str(media_path),
        "size_bytes": len(content),
    }
    if persist_asset:
        repository.create_video_with_asset(
            **video_payload,
            source_sha256=hashlib.sha256(content).hexdigest(),
        )
    else:
        repository.create_video(**video_payload)
    return database_path, media_root, content


def _copy_database_as_delete_journal(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(source_path)
    destination = sqlite3.connect(target_path)
    try:
        source.backup(destination)
        assert destination.execute("PRAGMA journal_mode = DELETE").fetchone()[0] == "delete"
    finally:
        destination.close()
        source.close()


def _copy_database_as_clean_closed_wal(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(source_path)
    destination = sqlite3.connect(target_path)
    try:
        source.backup(destination)
        assert destination.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
        destination.execute(
            "UPDATE videos SET display_name = ? WHERE id = ?",
            ("clean-wal-visible", "video-1"),
        )
        destination.commit()
        assert Path(f"{target_path}-wal").is_file()
        assert Path(f"{target_path}-shm").is_file()
    finally:
        destination.close()
        source.close()
    assert target_path.read_bytes()[18:20] == b"\x02\x02"
    assert Path(f"{target_path}-wal").exists() is False
    assert Path(f"{target_path}-shm").exists() is False


def test_read_only_repository_never_creates_a_missing_database_or_parent(
    tmp_path,
) -> None:
    database_path = tmp_path / "missing" / "videoscope.sqlite3"

    with pytest.raises(RuntimeError, match="unavailable"):
        Repository.open_read_only(database_path)

    assert database_path.parent.exists() is False
    assert database_path.exists() is False
    assert Path(f"{database_path}-wal").exists() is False
    assert Path(f"{database_path}-shm").exists() is False


def test_read_only_repository_accepts_clean_closed_wal_without_sidecars(
    tmp_path,
) -> None:
    source_database, _media_root, _content = _create_repository_with_media(tmp_path)
    database_path = tmp_path / "clean-wal" / "videoscope.sqlite3"
    _copy_database_as_clean_closed_wal(source_database, database_path)
    wal_path = Path(f"{database_path}-wal")
    shm_path = Path(f"{database_path}-shm")
    assert wal_path.exists() is False
    assert shm_path.exists() is False
    assert database_path.read_bytes()[18:20] == b"\x02\x02"
    before = _tree_snapshot(database_path.parent)

    repository = Repository.open_read_only(database_path)

    video = repository.get_video("video-1")
    assert video is not None
    assert video.display_name == "clean-wal-visible"
    assert _tree_snapshot(database_path.parent) == before
    assert wal_path.exists() is False
    assert shm_path.exists() is False


@pytest.mark.parametrize("present_sidecar", ["wal", "shm"])
def test_read_only_repository_rejects_a_half_present_wal_pair(
    tmp_path,
    present_sidecar: str,
) -> None:
    source_database, _media_root, _content = _create_repository_with_media(tmp_path)
    database_path = tmp_path / f"half-{present_sidecar}" / "videoscope.sqlite3"
    _copy_database_as_clean_closed_wal(source_database, database_path)
    wal_path = Path(f"{database_path}-wal")
    shm_path = Path(f"{database_path}-shm")
    assert database_path.read_bytes()[18:20] == b"\x02\x02"
    assert wal_path.exists() is False
    assert shm_path.exists() is False
    Path(f"{database_path}-{present_sidecar}").write_bytes(b"half-present")
    before = _tree_snapshot(database_path.parent)

    with pytest.raises(RuntimeError, match="WAL and SHM.*both"):
        Repository.open_read_only(database_path)

    assert _tree_snapshot(database_path.parent) == before


def test_read_only_repository_rejects_delete_mode_rollback_journal(
    tmp_path,
) -> None:
    source_database, _media_root, _content = _create_repository_with_media(tmp_path)
    database_path = tmp_path / "snapshot" / "videoscope.sqlite3"
    _copy_database_as_delete_journal(source_database, database_path)
    journal_path = Path(f"{database_path}-journal")
    journal_path.write_bytes(b"possibly-hot-rollback-journal")
    before = _tree_snapshot(database_path.parent)

    with pytest.raises(RuntimeError, match="rollback journal"):
        Repository.open_read_only(database_path)

    assert _tree_snapshot(database_path.parent) == before


@pytest.mark.parametrize("unexpected_sidecar", ["wal", "shm"])
def test_read_only_repository_rejects_delete_mode_wal_sidecars(
    tmp_path,
    unexpected_sidecar: str,
) -> None:
    source_database, _media_root, _content = _create_repository_with_media(tmp_path)
    database_path = tmp_path / f"delete-{unexpected_sidecar}" / "videoscope.sqlite3"
    _copy_database_as_delete_journal(source_database, database_path)
    Path(f"{database_path}-{unexpected_sidecar}").write_bytes(b"unexpected")
    before = _tree_snapshot(database_path.parent)

    with pytest.raises(RuntimeError, match="unexpected WAL"):
        Repository.open_read_only(database_path)

    assert _tree_snapshot(database_path.parent) == before


def test_read_only_header_validation_detects_one_same_size_change(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_database, _media_root, _content = _create_repository_with_media(tmp_path)
    database_path = tmp_path / "snapshot" / "videoscope.sqlite3"
    _copy_database_as_delete_journal(source_database, database_path)
    native_read = repository_module.os.read
    changed = False

    def mutate_after_header(file_descriptor: int, byte_count: int) -> bytes:
        nonlocal changed
        content = native_read(file_descriptor, byte_count)
        if not changed:
            changed = True
            with database_path.open("r+b") as database:
                database.seek(72)
                original = database.read(1)
                database.seek(72)
                database.write(bytes([original[0] ^ 1]))
                database.flush()
                repository_module.os.fsync(database.fileno())
        return content

    monkeypatch.setattr(repository_module.os, "read", mutate_after_header)

    with pytest.raises(RuntimeError, match="changed while opening"):
        repository_module._read_stable_sqlite_header(database_path)


def test_read_only_repository_rejects_a_multi_link_database(tmp_path) -> None:
    source_database, _media_root, _content = _create_repository_with_media(tmp_path)
    database_path = tmp_path / "snapshot" / "videoscope.sqlite3"
    _copy_database_as_delete_journal(source_database, database_path)
    linked_database = tmp_path / "linked.sqlite3"
    linked_database.hardlink_to(database_path)

    with pytest.raises(RuntimeError, match="managed regular file"):
        Repository.open_read_only(linked_database)


@pytest.mark.parametrize("unsafe_kind", ["missing", "symlink", "hardlink"])
def test_read_only_repository_requires_a_managed_existing_wal_shm(
    tmp_path,
    unsafe_kind: str,
) -> None:
    source_database, _media_root, _content = _create_repository_with_media(tmp_path)
    database_path = tmp_path / f"wal-{unsafe_kind}.sqlite3"
    source = sqlite3.connect(source_database)
    destination = sqlite3.connect(database_path)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    wal_path = Path(f"{database_path}-wal")
    wal_path.write_bytes(b"")
    shm_path = Path(f"{database_path}-shm")
    outside = tmp_path / f"outside-{unsafe_kind}.shm"
    if unsafe_kind == "symlink":
        outside.write_bytes(b"outside")
        shm_path.symlink_to(outside)
    elif unsafe_kind == "hardlink":
        outside.write_bytes(b"outside")
        shm_path.hardlink_to(outside)

    expected = (
        r"WAL and SHM.*both"
        if unsafe_kind == "missing"
        else r"SHM.*managed regular file"
    )
    with pytest.raises(RuntimeError, match=expected):
        Repository.open_read_only(database_path)

    assert wal_path.read_bytes() == b""
    if outside.exists():
        assert outside.read_bytes() == b"outside"


def test_read_only_repository_preserves_the_product_tree_and_enforces_sqlite_writes(
    tmp_path,
) -> None:
    source_database, media_root, _content = _create_repository_with_media(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    database_path = snapshot_dir / "videoscope.sqlite3"
    _copy_database_as_delete_journal(source_database, database_path)
    assert Path(f"{database_path}-wal").exists() is False
    assert Path(f"{database_path}-shm").exists() is False
    database_before = _tree_snapshot(snapshot_dir)
    media_before = _tree_snapshot(media_root)

    repository = Repository.open_read_only(database_path)
    with repository._connect() as connection:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert repository.schema_version() == LATEST_SCHEMA_VERSION
    assert repository.verify_existing_asset_identity(
        "video-1",
        media_root=media_root,
    ).sha256 == hashlib.sha256(b"benchmark-media-content").hexdigest()
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        repository.create_video(
            video_id="forbidden",
            original_name="forbidden.mp4",
            stored_name="forbidden.mp4",
            media_path=str(media_root / "forbidden.mp4"),
            size_bytes=1,
        )
    with pytest.raises(RuntimeError, match="read-only"):
        repository.initialize()

    # The source database above is fixture scaffolding and SQLite may checkpoint
    # its WAL after the backup connection is released.  The read-only target and
    # the managed media are the resources whose immutability is under test.
    assert _tree_snapshot(snapshot_dir) == database_before
    assert _tree_snapshot(media_root) == media_before
    assert Path(f"{database_path}-wal").exists() is False
    assert Path(f"{database_path}-shm").exists() is False


def test_read_only_repository_on_wal_preserves_durable_files_and_tree_layout(
    tmp_path,
) -> None:
    database_path, _media_root, _content = _create_repository_with_media(tmp_path)
    wal_path = Path(f"{database_path}-wal")
    keeper = sqlite3.connect(database_path)
    try:
        assert keeper.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
        keeper.execute("UPDATE videos SET id = id WHERE id = ?", ("video-1",))
        keeper.commit()
        assert wal_path.exists()
        layout_before = _tree_layout(tmp_path)
        database_before = database_path.read_bytes()
        wal_before = wal_path.read_bytes()

        repository = Repository.open_read_only(database_path)
        assert repository.get_video("video-1") is not None

        assert _tree_layout(tmp_path) == layout_before
        assert database_path.read_bytes() == database_before
        assert wal_path.read_bytes() == wal_before
        # SQLite is allowed to update lock bytes in an already-existing -shm file.
    finally:
        keeper.close()


@pytest.mark.parametrize(
    ("version", "message"),
    [
        (LATEST_SCHEMA_VERSION - 1, "older"),
        (LATEST_SCHEMA_VERSION + 1, "newer"),
    ],
)
def test_read_only_repository_requires_the_exact_schema_version(
    tmp_path,
    version: int,
    message: str,
) -> None:
    database_path, _media_root, _content = _create_repository_with_media(tmp_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(f"PRAGMA user_version = {version}")

    with pytest.raises(RuntimeError, match=message):
        Repository.open_read_only(database_path)


def test_read_only_repository_rejects_a_false_latest_schema_marker(tmp_path) -> None:
    database_path = tmp_path / "not-videoscope.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(f"PRAGMA user_version = {LATEST_SCHEMA_VERSION}")

    with pytest.raises(RuntimeError, match="incomplete or corrupt"):
        Repository.open_read_only(database_path)


def test_read_only_repository_rejects_a_missing_safety_trigger(tmp_path) -> None:
    database_path, _media_root, _content = _create_repository_with_media(tmp_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TRIGGER assets_immutable_update")

    with pytest.raises(RuntimeError, match="incomplete or corrupt"):
        Repository.open_read_only(database_path)


def test_verify_existing_asset_identity_requires_a_persisted_asset(tmp_path) -> None:
    source_database, media_root, _content = _create_repository_with_media(
        tmp_path,
        persist_asset=False,
    )
    database_path = tmp_path / "snapshot" / "videoscope.sqlite3"
    _copy_database_as_delete_journal(source_database, database_path)
    database_before = database_path.read_bytes()
    media_before = (media_root / "video-1.mp4").read_bytes()
    repository = Repository.open_read_only(database_path)

    with pytest.raises(AssetIdentityError, match="unavailable"):
        repository.verify_existing_asset_identity("video-1", media_root=media_root)

    assert database_path.read_bytes() == database_before
    assert (media_root / "video-1.mp4").read_bytes() == media_before
    assert repository.get_video_asset("video-1") is None


def test_existing_asset_verifier_resolves_a_managed_relative_path_without_cwd(
    tmp_path,
    monkeypatch,
) -> None:
    product_root = tmp_path / "product"
    media_root = product_root / "data" / "media"
    media_root.mkdir(parents=True)
    content = b"portable-managed-media"
    source = media_root / "video-1.mp4"
    source.write_bytes(content)
    repository = Repository(product_root / "data" / "videoscope.sqlite3")
    repository.initialize()
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="match.mp4",
        stored_name=source.name,
        media_path="data/media/video-1.mp4",
        size_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )
    elsewhere = tmp_path / "unrelated-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    asset = repository.verify_existing_asset_identity(
        "video-1",
        media_root=media_root,
    )

    assert asset.sha256 == hashlib.sha256(content).hexdigest()


def test_existing_asset_verifier_rejects_a_deceptive_relative_parent(
    tmp_path,
) -> None:
    media_root = tmp_path / "data" / "media"
    media_root.mkdir(parents=True)
    content = b"portable-managed-media"
    source = media_root / "video-1.mp4"
    source.write_bytes(content)
    repository = Repository(tmp_path / "data" / "videoscope.sqlite3")
    repository.initialize()
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="match.mp4",
        stored_name=source.name,
        media_path="outside/media/video-1.mp4",
        size_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )

    with pytest.raises(AssetIdentityError, match="managed regular file"):
        repository.verify_existing_asset_identity(
            "video-1",
            media_root=media_root,
        )


def test_verify_existing_asset_identity_rehashes_and_rejects_changed_media(
    tmp_path,
) -> None:
    database_path, media_root, content = _create_repository_with_media(tmp_path)
    media_path = media_root / "video-1.mp4"
    media_path.write_bytes(b"x" * len(content))
    repository = Repository(database_path)

    with pytest.raises(AssetIdentityError, match="does not match"):
        repository.verify_existing_asset_identity("video-1", media_root=media_root)


def test_verify_existing_asset_identity_rejects_a_symlinked_caller_root(
    tmp_path,
) -> None:
    database_path, media_root, _content = _create_repository_with_media(tmp_path)
    linked_root = tmp_path / "linked-media"
    linked_root.symlink_to(media_root, target_is_directory=True)
    repository = Repository(database_path)

    with pytest.raises(AssetIdentityError, match="managed regular file"):
        repository.verify_existing_asset_identity("video-1", media_root=linked_root)


def test_verify_existing_asset_identity_rejects_a_symlinked_media_ancestor(
    tmp_path,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    content = b"benchmark-media-content"
    media_path = media_root / "video-1.mp4"
    media_path.write_bytes(content)
    linked_root = tmp_path / "linked-media"
    linked_root.symlink_to(media_root, target_is_directory=True)
    database_path = tmp_path / "videoscope.sqlite3"
    repository = Repository(database_path)
    repository.initialize()
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="match.mp4",
        stored_name=media_path.name,
        media_path=str(linked_root / media_path.name),
        size_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )
    with pytest.raises(AssetIdentityError, match="managed regular file"):
        repository.verify_existing_asset_identity("video-1", media_root=media_root)


def test_regular_repository_remains_writable_after_read_only_support(tmp_path) -> None:
    database_path = tmp_path / "videoscope.sqlite3"
    repository = Repository(database_path)
    repository.initialize()

    created = repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "video-1.mp4"),
        size_bytes=1,
    )

    assert created.id == "video-1"


def test_read_only_repository_binds_a_retained_database_after_parent_replacement(
    tmp_path: Path,
) -> None:
    source_database, _media_root, _content = _create_repository_with_media(tmp_path)
    lexical_root = tmp_path / "private-snapshot"
    original_database = lexical_root / "videoscope.sqlite3"
    _copy_database_as_delete_journal(source_database, original_database)
    with sqlite3.connect(original_database) as connection:
        connection.execute(
            "UPDATE videos SET display_name = ? WHERE id = ?",
            ("retained-original", "video-1"),
        )
    retained_root, descriptor = _retained_directory_path(lexical_root)
    moved_root = tmp_path / "moved-private-snapshot"
    lexical_root.rename(moved_root)
    replacement_database = lexical_root / "videoscope.sqlite3"
    _copy_database_as_delete_journal(source_database, replacement_database)
    with sqlite3.connect(replacement_database) as connection:
        connection.execute(
            "UPDATE videos SET display_name = ? WHERE id = ?",
            ("lexical-replacement", "video-1"),
        )

    try:
        repository = Repository.open_read_only(retained_root / "videoscope.sqlite3")
        video = repository.get_video("video-1")
    finally:
        os.close(descriptor)

    assert video is not None
    assert video.display_name == "retained-original"


def test_existing_asset_verifier_reads_retained_media_not_lexical_replacement(
    tmp_path: Path,
) -> None:
    source_database, media_root, content = _create_repository_with_media(tmp_path)
    snapshot_database = tmp_path / "repository-snapshot" / "videoscope.sqlite3"
    _copy_database_as_delete_journal(source_database, snapshot_database)
    retained_media, descriptor = _retained_directory_path(media_root)
    moved_data = tmp_path / "moved-data"
    media_root.parent.rename(moved_data)
    media_root.mkdir(parents=True)
    replacement = b"x" * len(content)
    (media_root / "video-1.mp4").write_bytes(replacement)
    repository = Repository.open_read_only(snapshot_database)

    try:
        asset = repository.verify_existing_asset_identity(
            "video-1",
            media_root=media_root,
            media_access_root=retained_media,
        )
        with pytest.raises(AssetIdentityError, match="does not match"):
            repository.verify_existing_asset_identity(
                "video-1",
                media_root=media_root,
            )
    finally:
        os.close(descriptor)

    assert asset.sha256 == hashlib.sha256(content).hexdigest()
    assert (media_root / "video-1.mp4").read_bytes() == replacement
