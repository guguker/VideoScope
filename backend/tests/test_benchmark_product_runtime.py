from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat

import pytest

from videoscope.benchmark import product_runtime as product_runtime_module
from videoscope.benchmark.product_runtime import (
    ProductRuntimeSnapshot,
    ProductSnapshotCleanupError,
    ProductSnapshotError,
    ProductSnapshotFileIdentity,
    ProductSnapshotIdentity,
    ProductSnapshotLimits,
    open_product_runtime_snapshot,
)
from videoscope.repository import Repository
from videoscope.runtime_lifecycle import ExclusiveRuntimeLock


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


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def _persist_runtime_lock(data_dir: Path) -> None:
    lock = ExclusiveRuntimeLock(data_dir)
    lock.acquire()
    lock.close()


def _create_product_tree(
    root: Path,
    *,
    wal: bool,
) -> tuple[Path, Path, sqlite3.Connection | None]:
    data_dir = root / "data"
    media_root = data_dir / "media"
    media_root.mkdir(parents=True)
    media_path = media_root / "video-1.mp4"
    media_path.write_bytes(b"benchmark-product-media")
    database_path = data_dir / "videoscope.sqlite3"
    repository = Repository(database_path)
    repository.initialize()
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="fixture.mp4",
        stored_name=media_path.name,
        media_path=str(media_path),
        size_bytes=media_path.stat().st_size,
        source_sha256=hashlib.sha256(media_path.read_bytes()).hexdigest(),
    )

    keeper: sqlite3.Connection | None = None
    if wal:
        keeper = sqlite3.connect(database_path)
        keeper.execute("PRAGMA wal_autocheckpoint = 0")
        assert keeper.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
        keeper.execute(
            "UPDATE videos SET display_name = ? WHERE id = ?",
            ("wal-visible", "video-1"),
        )
        keeper.commit()
        assert Path(f"{database_path}-wal").is_file()
        assert Path(f"{database_path}-shm").is_file()
    else:
        delete_copy = data_dir / "delete-copy.sqlite3"
        source = sqlite3.connect(database_path)
        destination = sqlite3.connect(delete_copy)
        try:
            source.backup(destination)
            assert destination.execute("PRAGMA journal_mode = DELETE").fetchone()[0] == "delete"
        finally:
            destination.close()
            source.close()
        os.replace(delete_copy, database_path)
        Path(f"{database_path}-wal").unlink(missing_ok=True)
        Path(f"{database_path}-shm").unlink(missing_ok=True)
        assert Path(f"{database_path}-wal").exists() is False
        assert Path(f"{database_path}-shm").exists() is False

    _persist_runtime_lock(data_dir)
    return data_dir, media_root, keeper


def _open(
    data_dir: Path,
    media_root: Path,
    scratch_parent: Path,
    *,
    limits: ProductSnapshotLimits | None = None,
) -> ProductRuntimeSnapshot:
    return open_product_runtime_snapshot(
        data_dir=data_dir,
        database_path=data_dir / "videoscope.sqlite3",
        media_root=media_root,
        scratch_parent=scratch_parent,
        limits=limits,
    )


def _transition_to_clean_closed_wal(data_dir: Path) -> None:
    database_path = data_dir / "videoscope.sqlite3"
    writer = sqlite3.connect(database_path)
    try:
        assert writer.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
        writer.execute(
            "UPDATE videos SET display_name = ? WHERE id = ?",
            ("clean-wal-visible", "video-1"),
        )
        writer.commit()
        assert Path(f"{database_path}-wal").is_file()
        assert Path(f"{database_path}-shm").is_file()
    finally:
        writer.close()
    assert database_path.read_bytes()[18:20] == b"\x02\x02"
    assert Path(f"{database_path}-wal").exists() is False
    assert Path(f"{database_path}-shm").exists() is False


def test_wal_product_snapshot_preserves_full_source_tree_and_copies_all_sidecars(
    tmp_path: Path,
) -> None:
    data_dir, media_root, keeper = _create_product_tree(tmp_path, wal=True)
    assert keeper is not None
    scratch_parent = _private_directory(tmp_path / "scratch")
    source_before = _tree_snapshot(data_dir)
    source_hashes = {
        role: hashlib.sha256(path.read_bytes()).hexdigest()
        for role, path in {
            "database": data_dir / "videoscope.sqlite3",
            "wal": data_dir / "videoscope.sqlite3-wal",
            "shm": data_dir / "videoscope.sqlite3-shm",
        }.items()
    }
    try:
        snapshot = _open(data_dir, media_root, scratch_parent)
        assert snapshot.repository.is_read_only is True
        assert snapshot.media_root == media_root.absolute()
        video = snapshot.repository.get_video("video-1")
        assert video is not None
        assert video.display_name == "wal-visible"
        assert stat.S_IMODE(snapshot.scratch_root.stat().st_mode) == 0o700
        assert snapshot.scratch_root.parent == scratch_parent.absolute()
        assert {item.role for item in snapshot.identity.files} == {
            "database",
            "wal",
            "shm",
        }
        assert {
            item.role: item.sha256 for item in snapshot.identity.files
        } == source_hashes
        encoded = json.dumps(snapshot.identity.canonical_dict, sort_keys=True)
        assert str(tmp_path) not in encoded
        assert snapshot.identity.snapshot_sha256 == hashlib.sha256(
            json.dumps(
                snapshot.identity.content_dict,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        assert _tree_snapshot(data_dir) == source_before

        scratch_root = snapshot.scratch_root
        snapshot.close()
        assert scratch_root.exists() is False
        assert list(scratch_parent.iterdir()) == []
        assert _tree_snapshot(data_dir) == source_before
    finally:
        keeper.close()


def test_clean_closed_wal_snapshot_uses_coherent_main_database_only(
    tmp_path: Path,
) -> None:
    data_dir, media_root, keeper = _create_product_tree(tmp_path, wal=False)
    assert keeper is None
    _transition_to_clean_closed_wal(data_dir)
    database_path = data_dir / "videoscope.sqlite3"
    assert database_path.read_bytes()[18:20] == b"\x02\x02"
    assert Path(f"{database_path}-wal").exists() is False
    assert Path(f"{database_path}-shm").exists() is False
    scratch_parent = _private_directory(tmp_path / "scratch")
    source_before = _tree_snapshot(data_dir)

    with _open(data_dir, media_root, scratch_parent) as snapshot:
        assert snapshot.identity.journal_mode == "wal"
        assert [item.role for item in snapshot.identity.files] == ["database"]
        video = snapshot.repository.get_video("video-1")
        assert video is not None
        assert video.display_name == "clean-wal-visible"
        assert _tree_snapshot(data_dir) == source_before

    assert list(scratch_parent.iterdir()) == []
    assert _tree_snapshot(data_dir) == source_before


@pytest.mark.parametrize("present_sidecar", ["wal", "shm"])
def test_product_snapshot_rejects_a_half_present_wal_pair(
    tmp_path: Path,
    present_sidecar: str,
) -> None:
    data_dir, media_root, keeper = _create_product_tree(tmp_path, wal=False)
    assert keeper is None
    _transition_to_clean_closed_wal(data_dir)
    database_path = data_dir / "videoscope.sqlite3"
    Path(f"{database_path}-{present_sidecar}").write_bytes(b"half-present")
    scratch_parent = _private_directory(tmp_path / "scratch")
    source_before = _tree_snapshot(data_dir)

    with pytest.raises(ProductSnapshotError, match="WAL and SHM.*both"):
        _open(data_dir, media_root, scratch_parent)

    assert list(scratch_parent.iterdir()) == []
    assert _tree_snapshot(data_dir) == source_before


def test_delete_journal_snapshot_does_not_create_source_sidecars(tmp_path: Path) -> None:
    data_dir, media_root, keeper = _create_product_tree(tmp_path, wal=False)
    assert keeper is None
    scratch_parent = _private_directory(tmp_path / "scratch")
    database_path = data_dir / "videoscope.sqlite3"
    source_before = _tree_snapshot(data_dir)

    with _open(data_dir, media_root, scratch_parent) as snapshot:
        assert snapshot.repository.get_video("video-1") is not None
        assert [item.role for item in snapshot.identity.files] == ["database"]
        assert Path(f"{database_path}-wal").exists() is False
        assert Path(f"{database_path}-shm").exists() is False
        assert _tree_snapshot(data_dir) == source_before

    assert Path(f"{database_path}-wal").exists() is False
    assert Path(f"{database_path}-shm").exists() is False
    assert _tree_snapshot(data_dir) == source_before


def test_product_snapshot_requires_existing_unowned_runtime_lock(tmp_path: Path) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    lock_path = data_dir / ExclusiveRuntimeLock.filename
    lock_path.unlink()

    with pytest.raises(ProductSnapshotError, match="lock"):
        _open(data_dir, media_root, scratch_parent)

    assert list(scratch_parent.iterdir()) == []
    assert lock_path.exists() is False


def test_product_snapshot_rejects_busy_and_concurrent_runtimes(tmp_path: Path) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    owner = ExclusiveRuntimeLock(data_dir)
    owner.acquire_existing()
    try:
        with pytest.raises(ProductSnapshotError, match="already owns"):
            _open(data_dir, media_root, scratch_parent)
        assert list(scratch_parent.iterdir()) == []
    finally:
        owner.close()

    first = _open(data_dir, media_root, scratch_parent)
    try:
        with pytest.raises(ProductSnapshotError, match="already owns"):
            _open(data_dir, media_root, scratch_parent)
        assert list(scratch_parent.iterdir()) == [first.scratch_root]
    finally:
        first.close()
    second = _open(data_dir, media_root, scratch_parent)
    second.close()


@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink", "fifo"])
def test_product_snapshot_rejects_unsafe_database_files(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    database_path = data_dir / "videoscope.sqlite3"
    original = database_path.read_bytes()
    database_path.unlink()
    outside = tmp_path / "outside.sqlite3"
    if unsafe_kind == "symlink":
        outside.write_bytes(original)
        database_path.symlink_to(outside)
    elif unsafe_kind == "hardlink":
        outside.write_bytes(original)
        database_path.hardlink_to(outside)
    else:
        os.mkfifo(database_path)

    with pytest.raises(ProductSnapshotError, match="regular single-link"):
        _open(data_dir, media_root, scratch_parent)

    assert list(scratch_parent.iterdir()) == []
    if outside.exists():
        assert outside.read_bytes() == original


@pytest.mark.parametrize(
    ("sidecar", "unsafe_kind"),
    [("wal", "symlink"), ("shm", "hardlink"), ("wal", "fifo")],
)
def test_product_snapshot_rejects_unsafe_wal_sidecars(
    tmp_path: Path,
    sidecar: str,
    unsafe_kind: str,
) -> None:
    data_dir, media_root, keeper = _create_product_tree(tmp_path, wal=True)
    assert keeper is not None
    scratch_parent = _private_directory(tmp_path / "scratch")
    target = data_dir / f"videoscope.sqlite3-{sidecar}"
    original = target.read_bytes()
    target.unlink()
    outside = tmp_path / f"outside-{sidecar}"
    if unsafe_kind == "symlink":
        outside.write_bytes(original)
        target.symlink_to(outside)
    elif unsafe_kind == "hardlink":
        outside.write_bytes(original)
        target.hardlink_to(outside)
    else:
        os.mkfifo(target)
    try:
        with pytest.raises(ProductSnapshotError, match="regular single-link"):
            _open(data_dir, media_root, scratch_parent)
        assert list(scratch_parent.iterdir()) == []
    finally:
        keeper.close()


def test_product_snapshot_detects_same_size_source_mutation_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    database_path = data_dir / "videoscope.sqlite3"
    native_read = product_runtime_module._read_regular_source
    calls = 0

    def mutate_after_copy(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        result = native_read(*args, **kwargs)
        calls += 1
        if calls == 1:
            content = database_path.read_bytes()
            database_path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
        return result

    monkeypatch.setattr(
        product_runtime_module,
        "_read_regular_source",
        mutate_after_copy,
    )

    with pytest.raises(ProductSnapshotError, match="changed"):
        _open(data_dir, media_root, scratch_parent)

    assert list(scratch_parent.iterdir()) == []
    retry = ExclusiveRuntimeLock(data_dir)
    retry.acquire_existing()
    retry.close()


def test_product_snapshot_enforces_file_total_and_entry_limits(tmp_path: Path) -> None:
    data_dir, media_root, keeper = _create_product_tree(tmp_path, wal=True)
    assert keeper is not None
    scratch_parent = _private_directory(tmp_path / "scratch")
    database_size = (data_dir / "videoscope.sqlite3").stat().st_size
    source_sizes = [
        (data_dir / name).stat().st_size
        for name in (
            "videoscope.sqlite3",
            "videoscope.sqlite3-wal",
            "videoscope.sqlite3-shm",
        )
    ]
    try:
        with pytest.raises(ProductSnapshotError, match="per-file"):
            _open(
                data_dir,
                media_root,
                scratch_parent,
                limits=ProductSnapshotLimits(
                    max_entries=3,
                    max_file_bytes=database_size - 1,
                    max_total_bytes=database_size * 4,
                ),
            )
        with pytest.raises(ProductSnapshotError, match="total"):
            _open(
                data_dir,
                media_root,
                scratch_parent,
                    limits=ProductSnapshotLimits(
                        max_entries=3,
                        max_file_bytes=max(source_sizes),
                        max_total_bytes=sum(source_sizes) - 1,
                    ),
            )
        with pytest.raises(ProductSnapshotError, match="entry"):
            _open(
                data_dir,
                media_root,
                scratch_parent,
                    limits=ProductSnapshotLimits(
                        max_entries=2,
                        max_file_bytes=max(source_sizes),
                        max_total_bytes=sum(source_sizes),
                    ),
            )
    finally:
        keeper.close()
    assert list(scratch_parent.iterdir()) == []


def test_product_snapshot_rejects_symlink_or_product_owned_scratch_parent(
    tmp_path: Path,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    outside = _private_directory(tmp_path / "outside-scratch")
    linked = tmp_path / "linked-scratch"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ProductSnapshotError, match="scratch"):
        _open(data_dir, media_root, linked)
    assert list(outside.iterdir()) == []

    inside = _private_directory(data_dir / "scratch")
    source_before = _tree_snapshot(data_dir)
    with pytest.raises(ProductSnapshotError, match="outside"):
        _open(data_dir, media_root, inside)
    assert _tree_snapshot(data_dir) == source_before


def test_product_snapshot_rejects_uncontained_or_symlinked_product_paths(
    tmp_path: Path,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    outside_db = tmp_path / "outside.sqlite3"
    outside_db.write_bytes((data_dir / "videoscope.sqlite3").read_bytes())

    with pytest.raises(ProductSnapshotError, match="contained"):
        open_product_runtime_snapshot(
            data_dir=data_dir,
            database_path=outside_db,
            media_root=media_root,
            scratch_parent=scratch_parent,
        )

    outside_media = tmp_path / "outside-media"
    outside_media.mkdir()
    with pytest.raises(ProductSnapshotError, match="contained"):
        open_product_runtime_snapshot(
            data_dir=data_dir,
            database_path=data_dir / "videoscope.sqlite3",
            media_root=outside_media,
            scratch_parent=scratch_parent,
        )

    linked_media = data_dir / "linked-media"
    linked_media.symlink_to(media_root, target_is_directory=True)
    with pytest.raises(ProductSnapshotError, match="unsafe"):
        open_product_runtime_snapshot(
            data_dir=data_dir,
            database_path=data_dir / "videoscope.sqlite3",
            media_root=linked_media,
            scratch_parent=scratch_parent,
        )
    assert list(scratch_parent.iterdir()) == []


def test_open_failure_removes_owned_scratch_before_releasing_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    events: list[str] = []
    native_remove = product_runtime_module._remove_owned_scratch
    native_close = ExclusiveRuntimeLock.close

    def rejecting_open(_path: Path) -> Repository:
        raise RuntimeError("deliberate schema failure")

    def recording_remove(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append("scratch")
        return native_remove(*args, **kwargs)

    def recording_close(lock: ExclusiveRuntimeLock) -> None:
        events.append("lock")
        native_close(lock)

    monkeypatch.setattr(Repository, "open_read_only", rejecting_open)
    monkeypatch.setattr(
        product_runtime_module,
        "_remove_owned_scratch",
        recording_remove,
    )
    monkeypatch.setattr(ExclusiveRuntimeLock, "close", recording_close)

    with pytest.raises(ProductSnapshotError, match="schema failure"):
        _open(data_dir, media_root, scratch_parent)

    assert events == ["scratch", "lock"]
    assert list(scratch_parent.iterdir()) == []
    retry = ExclusiveRuntimeLock(data_dir)
    retry.acquire_existing()
    retry.close()


def test_product_snapshot_close_is_idempotent_and_context_errors_still_cleanup(
    tmp_path: Path,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    snapshot = _open(data_dir, media_root, scratch_parent)
    scratch_root = snapshot.scratch_root
    snapshot.close()
    snapshot.close()
    assert scratch_root.exists() is False

    with pytest.raises(ValueError, match="body failed"):
        with _open(data_dir, media_root, scratch_parent):
            raise ValueError("body failed")
    assert list(scratch_parent.iterdir()) == []


def test_repository_close_failure_keeps_product_lock_and_close_is_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    snapshot = _open(data_dir, media_root, scratch_parent)
    native_close_repository = product_runtime_module._close_repository
    attempts = 0

    def fail_once(repository: Repository) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("repository close failed")
        native_close_repository(repository)

    monkeypatch.setattr(product_runtime_module, "_close_repository", fail_once)

    with pytest.raises(ProductSnapshotError, match="repository close failed"):
        snapshot.close()

    assert snapshot.is_closed is False
    assert snapshot.scratch_root.exists() is True
    with pytest.raises(RuntimeError, match="already owns"):
        ExclusiveRuntimeLock(data_dir).acquire_existing()

    snapshot.close()
    assert attempts == 2
    assert snapshot.is_closed is True
    assert snapshot.scratch_root.exists() is False
    retry = ExclusiveRuntimeLock(data_dir)
    retry.acquire_existing()
    retry.close()


def test_scratch_removal_failure_keeps_product_lock_and_resumes_after_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    snapshot = _open(data_dir, media_root, scratch_parent)
    native_close_repository = product_runtime_module._close_repository
    native_remove = product_runtime_module._remove_owned_scratch
    events: list[str] = []
    remove_attempts = 0

    def recording_close(repository: Repository) -> None:
        events.append("repository")
        native_close_repository(repository)

    def fail_remove_once(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal remove_attempts
        remove_attempts += 1
        events.append("scratch")
        if remove_attempts == 1:
            raise RuntimeError("scratch removal failed")
        return native_remove(*args, **kwargs)

    monkeypatch.setattr(product_runtime_module, "_close_repository", recording_close)
    monkeypatch.setattr(product_runtime_module, "_remove_owned_scratch", fail_remove_once)

    with pytest.raises(ProductSnapshotError, match="scratch removal failed"):
        snapshot.close()

    assert events == ["repository", "scratch"]
    assert snapshot.is_closed is False
    with pytest.raises(RuntimeError, match="already owns"):
        ExclusiveRuntimeLock(data_dir).acquire_existing()

    snapshot.close()
    assert events == ["repository", "scratch", "scratch"]
    assert snapshot.is_closed is True
    retry = ExclusiveRuntimeLock(data_dir)
    retry.acquire_existing()
    retry.close()


def test_parent_fsync_failure_after_scratch_unlink_retries_only_durability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    snapshot = _open(data_dir, media_root, scratch_parent)
    scratch_root = snapshot.scratch_root
    parent_descriptor = snapshot._scratch_parent_descriptor
    native_fsync = os.fsync
    failed = False

    def fail_parent_once_after_unlink(descriptor: int) -> None:
        nonlocal failed
        if (
            descriptor == parent_descriptor
            and not scratch_root.exists()
            and not failed
        ):
            failed = True
            raise OSError("parent durability sync failed")
        native_fsync(descriptor)

    monkeypatch.setattr(product_runtime_module.os, "fsync", fail_parent_once_after_unlink)

    with pytest.raises(ProductSnapshotError, match="root could not be removed"):
        snapshot.close()

    assert failed is True
    assert scratch_root.exists() is False
    assert snapshot.is_closed is False
    with pytest.raises(RuntimeError, match="already owns"):
        ExclusiveRuntimeLock(data_dir).acquire_existing()

    snapshot.close()
    assert snapshot.is_closed is True
    retry = ExclusiveRuntimeLock(data_dir)
    retry.acquire_existing()
    retry.close()


def test_open_failure_retries_transient_scratch_cleanup_before_unlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    native_unlink = os.unlink
    attempts = 0

    def rejecting_open(_path: Path) -> Repository:
        raise RuntimeError("deliberate schema failure")

    def fail_database_unlink_once(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempts
        if path == "product.sqlite3":
            attempts += 1
            if attempts == 1:
                raise OSError("transient unlink failure")
        return native_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Repository, "open_read_only", rejecting_open)
    monkeypatch.setattr(product_runtime_module.os, "unlink", fail_database_unlink_once)

    with pytest.raises(ProductSnapshotError, match="deliberate schema failure"):
        _open(data_dir, media_root, scratch_parent)

    assert attempts == 2
    assert list(scratch_parent.iterdir()) == []
    retry = ExclusiveRuntimeLock(data_dir)
    retry.acquire_existing()
    retry.close()


def test_persistent_open_cleanup_failure_exposes_retry_owner_without_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    native_unlink = os.unlink
    cleanup_blocked = True

    def rejecting_open(_path: Path) -> Repository:
        raise RuntimeError("deliberate schema failure")

    def fail_database_unlink(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if cleanup_blocked and path == "product.sqlite3":
            raise OSError("persistent unlink failure")
        return native_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Repository, "open_read_only", rejecting_open)
    monkeypatch.setattr(product_runtime_module.os, "unlink", fail_database_unlink)

    with pytest.raises(ProductSnapshotCleanupError) as captured:
        _open(data_dir, media_root, scratch_parent)

    error = captured.value
    assert "deliberate schema failure" in str(error)
    assert str(tmp_path) not in str(error)
    assert error.cleanup_pending is True
    assert len(list(scratch_parent.iterdir())) == 1
    with pytest.raises(RuntimeError, match="already owns"):
        ExclusiveRuntimeLock(data_dir).acquire_existing()

    cleanup_blocked = False
    error.retry_cleanup()
    assert error.cleanup_pending is False
    assert list(scratch_parent.iterdir()) == []
    retry = ExclusiveRuntimeLock(data_dir)
    retry.acquire_existing()
    retry.close()


def test_cleanup_directory_scan_stops_at_the_declared_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumed = 0

    class _Entry:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Scan:
        def __enter__(self) -> _Scan:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self) -> _Scan:
            return self

        def __next__(self) -> _Entry:
            nonlocal consumed
            consumed += 1
            return _Entry(f"entry-{consumed}")

    monkeypatch.setattr(product_runtime_module.os, "scandir", lambda _fd: _Scan())

    with pytest.raises(ProductSnapshotError, match="bound"):
        product_runtime_module._bounded_directory_names(
            123,
            limit=1,
            list_error="list failed",
            overflow_error="cleanup exceeds bound",
        )

    assert consumed == 2


def test_lock_close_failure_is_not_reported_closed_and_only_retries_lock_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    snapshot = _open(data_dir, media_root, scratch_parent)
    native_remove = product_runtime_module._remove_owned_scratch
    native_lock_close = ExclusiveRuntimeLock.close
    events: list[str] = []
    lock_attempts = 0

    def recording_remove(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append("scratch")
        return native_remove(*args, **kwargs)

    def fail_lock_once(lock: ExclusiveRuntimeLock) -> None:
        nonlocal lock_attempts
        lock_attempts += 1
        events.append("lock")
        if lock_attempts == 1:
            raise RuntimeError("lock close failed")
        native_lock_close(lock)

    monkeypatch.setattr(product_runtime_module, "_remove_owned_scratch", recording_remove)
    monkeypatch.setattr(ExclusiveRuntimeLock, "close", fail_lock_once)

    with pytest.raises(ProductSnapshotError, match="lock close failed"):
        snapshot.close()

    assert events == ["scratch", "lock"]
    assert snapshot.is_closed is False
    assert snapshot.scratch_root.exists() is False
    with pytest.raises(RuntimeError, match="already owns"):
        ExclusiveRuntimeLock(data_dir).acquire_existing()

    snapshot.close()
    assert events == ["scratch", "lock", "lock"]
    assert snapshot.is_closed is True
    snapshot.close()
    assert events == ["scratch", "lock", "lock"]


def test_late_descriptor_close_failure_retries_exact_remaining_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    snapshot = _open(data_dir, media_root, scratch_parent)
    target_descriptor = snapshot._descriptors_to_close[-1]
    native_close = os.close
    attempts = 0

    def fail_target_once(descriptor: int) -> None:
        nonlocal attempts
        if descriptor == target_descriptor:
            attempts += 1
            if attempts == 1:
                raise OSError("late retained descriptor close failed")
        native_close(descriptor)

    monkeypatch.setattr(product_runtime_module.os, "close", fail_target_once)

    with pytest.raises(ProductSnapshotError, match="descriptors"):
        snapshot.close()

    assert snapshot.is_closed is False
    assert snapshot._descriptors_to_close == [target_descriptor]
    with pytest.raises(RuntimeError, match="already owns"):
        ExclusiveRuntimeLock(data_dir).acquire_existing()

    snapshot.close()
    assert attempts == 2
    assert snapshot._descriptors_to_close == []
    assert snapshot.is_closed is True
    retry = ExclusiveRuntimeLock(data_dir)
    retry.acquire_existing()
    retry.close()


def test_caller_managed_scratch_entries_must_close_before_product_snapshot(
    tmp_path: Path,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    snapshot = _open(data_dir, media_root, scratch_parent)
    semantic_workspace = snapshot.scratch_root / "qdrant-snapshot"
    semantic_workspace.mkdir()
    semantic_file = semantic_workspace / "state.bin"
    semantic_file.write_bytes(b"caller-owned")

    with pytest.raises(ProductSnapshotError, match="caller-managed"):
        snapshot.close()

    assert snapshot.is_closed is False
    assert semantic_file.read_bytes() == b"caller-owned"
    with pytest.raises(RuntimeError, match="already owns"):
        ExclusiveRuntimeLock(data_dir).acquire_existing()

    semantic_file.unlink()
    semantic_workspace.rmdir()
    snapshot.close()
    assert snapshot.is_closed is True
    assert snapshot.scratch_root.exists() is False
    retry = ExclusiveRuntimeLock(data_dir)
    retry.acquire_existing()
    retry.close()


def test_product_snapshot_limits_validate_positive_bounded_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        ProductSnapshotLimits(max_entries=0)
    with pytest.raises(ValueError, match="per-file"):
        ProductSnapshotLimits(max_file_bytes=2, max_total_bytes=1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"role": "journal"}, "role"),
        ({"size_bytes": -1}, "size"),
        ({"sha256": "A" * 64}, "digest"),
    ],
)
def test_product_snapshot_file_identity_rejects_noncanonical_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "role": "database",
        "size_bytes": 1,
        "sha256": "a" * 64,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ProductSnapshotFileIdentity(**values)  # type: ignore[arg-type]


def test_product_snapshot_identity_rejects_invalid_mode_files_total_and_protocol() -> None:
    database = ProductSnapshotFileIdentity(
        role="database",
        size_bytes=1,
        sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="journal mode"):
        ProductSnapshotIdentity(
            journal_mode="memory",  # type: ignore[arg-type]
            files=(database,),
            total_bytes=1,
        )
    clean_wal = ProductSnapshotIdentity(
        journal_mode="wal",
        files=(database,),
        total_bytes=1,
    )
    assert clean_wal.journal_mode == "wal"
    assert [item.role for item in clean_wal.files] == ["database"]
    with pytest.raises(ValueError, match="incomplete"):
        ProductSnapshotIdentity(
            journal_mode="delete",
            files=(
                database,
                ProductSnapshotFileIdentity(
                    role="wal",
                    size_bytes=0,
                    sha256="b" * 64,
                ),
            ),
            total_bytes=1,
        )
    with pytest.raises(ValueError, match="total"):
        ProductSnapshotIdentity(
            journal_mode="delete",
            files=(database,),
            total_bytes=2,
        )
    with pytest.raises(ValueError, match="protocol"):
        ProductSnapshotIdentity(
            journal_mode="delete",
            files=(database,),
            total_bytes=1,
            protocol_version=2,
        )


def test_delete_snapshot_rejects_unexpected_wal_sidecars(tmp_path: Path) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    (data_dir / "videoscope.sqlite3-wal").write_bytes(b"stale")

    with pytest.raises(ProductSnapshotError, match="unexpected WAL"):
        _open(data_dir, media_root, scratch_parent)

    assert list(scratch_parent.iterdir()) == []


def test_delete_snapshot_rejects_existing_rollback_journal_before_copying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    journal_path = data_dir / "videoscope.sqlite3-journal"
    journal_path.write_bytes(b"possibly-hot-rollback-journal")
    source_before = _tree_snapshot(data_dir)
    native_read = product_runtime_module._read_regular_source
    copied_outputs: list[str] = []

    def record_copy(*args, **kwargs):  # type: ignore[no-untyped-def]
        output_name = kwargs.get("output_name")
        if output_name is not None:
            copied_outputs.append(str(output_name))
        return native_read(*args, **kwargs)

    monkeypatch.setattr(product_runtime_module, "_read_regular_source", record_copy)

    with pytest.raises(ProductSnapshotError, match="rollback journal"):
        _open(data_dir, media_root, scratch_parent)

    assert copied_outputs == []
    assert list(scratch_parent.iterdir()) == []
    assert _tree_snapshot(data_dir) == source_before


def test_product_snapshot_rejects_invalid_database_and_nonprivate_scratch(
    tmp_path: Path,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir(mode=0o755)
    scratch_parent.chmod(0o755)

    with pytest.raises(ProductSnapshotError, match="0700"):
        _open(data_dir, media_root, scratch_parent)

    scratch_parent.chmod(0o700)
    (data_dir / "videoscope.sqlite3").write_bytes(b"not-sqlite".ljust(100, b"!"))
    with pytest.raises(ProductSnapshotError, match="valid SQLite"):
        _open(data_dir, media_root, scratch_parent)


def test_product_snapshot_directory_capabilities_survive_lexical_root_replacement(
    tmp_path: Path,
) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    snapshot = _open(data_dir, media_root, scratch_parent)
    moved_data = tmp_path / "moved-data"
    data_dir.rename(moved_data)
    replacement_media = data_dir / "media"
    replacement_media.mkdir(parents=True)
    (replacement_media / "video-1.mp4").write_bytes(b"replacement-media")

    data_descriptor = snapshot.duplicate_data_root_descriptor()
    media_descriptor = snapshot.duplicate_media_root_descriptor()
    try:
        assert os.fstat(data_descriptor).st_ino == moved_data.stat().st_ino
        source_descriptor = os.open(
            "video-1.mp4",
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=media_descriptor,
        )
        try:
            assert os.read(source_descriptor, 1024) == b"benchmark-product-media"
        finally:
            os.close(source_descriptor)
    finally:
        os.close(media_descriptor)
        os.close(data_descriptor)

    snapshot.close()
    assert snapshot.is_closed is True


def test_closed_product_snapshot_cannot_be_reentered(tmp_path: Path) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")
    snapshot = _open(data_dir, media_root, scratch_parent)
    snapshot.close()

    with pytest.raises(ProductSnapshotError, match="already closed"):
        snapshot.__enter__()


def test_open_requires_validated_snapshot_limits(tmp_path: Path) -> None:
    data_dir, media_root, _keeper = _create_product_tree(tmp_path, wal=False)
    scratch_parent = _private_directory(tmp_path / "scratch")

    with pytest.raises(ValueError, match="validated"):
        open_product_runtime_snapshot(
            data_dir=data_dir,
            database_path=data_dir / "videoscope.sqlite3",
            media_root=media_root,
            scratch_parent=scratch_parent,
            limits=object(),  # type: ignore[arg-type]
        )
