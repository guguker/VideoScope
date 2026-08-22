from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
from threading import Barrier, Lock
from time import sleep
from types import SimpleNamespace

import pytest

import videoscope.repository as repository_module
from videoscope.artifacts import AssetIdentityError, StageKind, StageSpecification
from videoscope.repository import Repository


def _create_legacy_video(
    repository: Repository,
    media_root: Path,
    *,
    video_id: str = "video-1",
    content: bytes = b"same video bytes",
) -> Path:
    media_root.mkdir(parents=True, exist_ok=True)
    media_path = media_root / f"{video_id}.mp4"
    media_path.write_bytes(content)
    repository.create_video(
        video_id=video_id,
        original_name=f"{video_id}.mp4",
        stored_name=media_path.name,
        media_path=str(media_path),
        size_bytes=len(content),
    )
    return media_path


def _speech_specification() -> StageSpecification:
    return StageSpecification(
        kind=StageKind.SPEECH,
        schema_version=1,
        implementation_revision="speech-v1",
        model_identity="whisper@revision",
    )


def test_create_video_with_asset_links_normalized_identity_atomically(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    content = b"uploaded bytes"
    digest = hashlib.sha256(content).hexdigest()

    video = repository.create_video_with_asset(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "media" / "video-1.mp4"),
        size_bytes=len(content),
        source_sha256=digest,
    )
    asset = repository.get_video_asset(video.id)

    assert video.asset_id == f"sha256:{digest}"
    assert asset is not None
    assert asset.asset_id == video.asset_id
    assert asset.sha256 == digest
    assert asset.size_bytes == len(content)


def test_create_video_with_asset_has_no_fallible_post_commit_read(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()

    def reject_post_commit_read(video_id: str):
        del video_id
        raise AssertionError("asset transaction must return its committed record directly")

    monkeypatch.setattr(repository, "get_video", reject_post_commit_read)

    video = repository.create_video_with_asset(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "video-1.mp4"),
        size_bytes=10,
        source_sha256="a" * 64,
    )

    assert video.id == "video-1"
    assert video.asset_id == "sha256:" + "a" * 64


def test_duplicate_content_shares_one_asset_between_video_rows(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    content = b"duplicate video"
    digest = hashlib.sha256(content).hexdigest()

    first = repository.create_video_with_asset(
        video_id="video-1",
        original_name="first.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "media" / "video-1.mp4"),
        size_bytes=len(content),
        source_sha256=digest,
    )
    second = repository.create_video_with_asset(
        video_id="video-2",
        original_name="second.mp4",
        stored_name="video-2.mp4",
        media_path=str(tmp_path / "media" / "video-2.mp4"),
        size_bytes=len(content),
        source_sha256=digest,
    )

    assert first.asset_id == second.asset_id
    with repository._connect() as connection:
        count = connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert count == 1


def test_repository_finds_benchmark_ready_asset_bindings_by_digest(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    digest = "a" * 64
    for index, duration in ((1, 12.5), (2, 13.0)):
        repository.create_video_with_asset(
            video_id=f"video-{index}",
            original_name=f"video-{index}.mp4",
            stored_name=f"video-{index}.mp4",
            media_path=str(tmp_path / f"video-{index}.mp4"),
            size_bytes=10,
            source_sha256=digest,
        )
        repository.update_video(f"video-{index}", duration=duration)
    repository.create_video_with_asset(
        video_id="video-not-ready",
        original_name="not-ready.mp4",
        stored_name="not-ready.mp4",
        media_path=str(tmp_path / "not-ready.mp4"),
        size_bytes=10,
        source_sha256=digest,
    )

    matches = repository.find_assets_by_sha256(digest)

    assert [match.video_id for match in matches] == ["video-1", "video-2"]
    assert {match.id for match in matches} == {f"sha256:{digest}"}
    assert {match.sha256 for match in matches} == {digest}
    assert {match.byte_size for match in matches} == {10}
    assert [match.duration_seconds for match in matches] == [12.5, 13.0]


def test_repository_asset_lookup_fails_closed_on_corrupt_video_link(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    digest = "a" * 64
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="video-1.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "video-1.mp4"),
        size_bytes=10,
        source_sha256=digest,
    )
    repository.update_video("video-1", duration=12.5)
    with repository._connect() as connection:
        connection.execute(
            "UPDATE videos SET size_bytes = ? WHERE id = ?",
            (11, "video-1"),
        )

    with pytest.raises(AssetIdentityError, match="corrupt"):
        repository.find_assets_by_sha256(digest)


def test_repository_exposes_a_bounded_asset_lookup_for_benchmark_preflight(
    tmp_path,
) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    digest = "a" * 64
    for index in range(5):
        repository.create_video_with_asset(
            video_id=f"video-{index}",
            original_name=f"video-{index}.mp4",
            stored_name=f"video-{index}.mp4",
            media_path=str(tmp_path / f"video-{index}.mp4"),
            size_bytes=10,
            source_sha256=digest,
        )
        repository.update_video(f"video-{index}", duration=12.5)

    bounded = repository.find_assets_by_sha256_bounded(digest, limit=2)
    explicit = repository.find_assets_by_sha256_bounded(
        digest,
        limit=1,
        video_id="video-3",
    )

    assert [item.video_id for item in bounded] == ["video-0", "video-1", "video-2"]
    assert [item.video_id for item in explicit] == ["video-3"]


def test_asset_collision_rolls_back_video_and_preserves_original_asset(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    digest = "a" * 64
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="first.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "video-1.mp4"),
        size_bytes=10,
        source_sha256=digest,
    )

    with pytest.raises(AssetIdentityError, match="size"):
        repository.create_video_with_asset(
            video_id="video-2",
            original_name="second.mp4",
            stored_name="video-2.mp4",
            media_path=str(tmp_path / "video-2.mp4"),
            size_bytes=11,
            source_sha256=digest,
        )

    assert repository.get_video("video-2") is None
    assert repository.get_video_asset("video-1").size_bytes == 10  # type: ignore[union-attr]


def test_legacy_asset_identity_hashes_managed_media_once(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    media_root = tmp_path / "media"
    content = b"legacy video bytes"
    media_path = _create_legacy_video(repository, media_root, content=content)
    before = media_path.read_bytes()
    video_before = repository.get_video("video-1")

    first = repository.ensure_asset_identity("video-1", media_root=media_root)
    real_open = os.open

    def reject_second_read(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("persisted asset identity must not re-read media")

    monkeypatch.setattr(repository_module.os, "open", reject_second_read)
    second = repository.ensure_asset_identity("video-1", media_root=media_root)
    monkeypatch.setattr(repository_module.os, "open", real_open)

    assert first == second
    assert first.sha256 == hashlib.sha256(content).hexdigest()
    assert media_path.read_bytes() == before
    assert repository.get_video("video-1").updated_at == video_before.updated_at  # type: ignore[union-attr]


def test_concurrent_legacy_asset_identity_hashes_same_video_once(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    media_root = tmp_path / "media"
    _create_legacy_video(repository, media_root, content=b"large legacy content")
    original_hash = repository_module._hash_managed_regular_file
    starting_line = Barrier(4)
    count_lock = Lock()
    hash_count = 0

    def tracked_hash(path, *, expected_size):  # type: ignore[no-untyped-def]
        nonlocal hash_count
        with count_lock:
            hash_count += 1
        sleep(0.05)
        return original_hash(path, expected_size=expected_size)

    monkeypatch.setattr(repository_module, "_hash_managed_regular_file", tracked_hash)

    def ensure():  # type: ignore[no-untyped-def]
        starting_line.wait()
        concurrent_repository = Repository(repository.database_path)
        return concurrent_repository.ensure_asset_identity("video-1", media_root=media_root)

    with ThreadPoolExecutor(max_workers=4) as executor:
        assets = list(executor.map(lambda _: ensure(), range(4)))

    assert hash_count == 1
    assert len({asset.asset_id for asset in assets}) == 1


@pytest.mark.parametrize("unsafe_kind", ["outside", "symlink", "directory"])
def test_legacy_asset_identity_rejects_unmanaged_or_non_regular_media(
    tmp_path,
    unsafe_kind: str,
) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    media_root = tmp_path / "media"
    media_root.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    if unsafe_kind == "outside":
        media_path = outside
    elif unsafe_kind == "symlink":
        media_path = media_root / "video-1.mp4"
        media_path.symlink_to(outside)
    else:
        media_path = media_root / "video-1.mp4"
        media_path.mkdir()
    repository.create_video(
        video_id="video-1",
        original_name="video-1.mp4",
        stored_name="video-1.mp4",
        media_path=str(media_path),
        size_bytes=len(b"outside"),
    )

    with pytest.raises(AssetIdentityError, match="managed regular file"):
        repository.ensure_asset_identity("video-1", media_root=media_root)

    assert repository.get_video_asset("video-1") is None


def test_legacy_asset_identity_rejects_file_changed_during_hash(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    media_root = tmp_path / "media"
    _create_legacy_video(repository, media_root)
    real_fstat = os.fstat
    calls = 0

    def changing_fstat(file_descriptor: int):
        nonlocal calls
        current = real_fstat(file_descriptor)
        calls += 1
        if calls == 2:
            return SimpleNamespace(
                st_dev=current.st_dev,
                st_ino=current.st_ino,
                st_mode=current.st_mode,
                st_size=current.st_size,
                st_mtime_ns=current.st_mtime_ns,
                st_ctime_ns=current.st_ctime_ns + 1,
            )
        return current

    monkeypatch.setattr(repository_module.os, "fstat", changing_fstat)

    with pytest.raises(AssetIdentityError, match="changed while hashing"):
        repository.ensure_asset_identity("video-1", media_root=media_root)

    assert repository.get_video_asset("video-1") is None


def test_legacy_asset_identity_rejects_video_row_replaced_after_hash(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    media_root = tmp_path / "media"
    _create_legacy_video(repository, media_root, content=b"old bytes")
    original_hash = repository_module._hash_managed_regular_file

    def replace_video_after_hash(path, *, expected_size):  # type: ignore[no-untyped-def]
        digest = original_hash(path, expected_size=expected_size)
        with repository._connect() as connection:
            connection.execute("DELETE FROM videos WHERE id = ?", ("video-1",))
        repository.create_video(
            video_id="video-1",
            original_name="replacement.mp4",
            stored_name="video-1.mp4",
            media_path=str(path),
            size_bytes=expected_size,
        )
        return digest

    monkeypatch.setattr(
        repository_module,
        "_hash_managed_regular_file",
        replace_video_after_hash,
    )

    with pytest.raises(AssetIdentityError, match="video record changed"):
        repository.ensure_asset_identity("video-1", media_root=media_root)

    assert repository.get_video_asset("video-1") is None


def test_video_asset_link_and_asset_rows_are_immutable_and_fail_closed(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    first_digest = "a" * 64
    second_digest = "b" * 64
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="first.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "video-1.mp4"),
        size_bytes=10,
        source_sha256=first_digest,
    )
    repository.create_video_with_asset(
        video_id="video-2",
        original_name="second.mp4",
        stored_name="video-2.mp4",
        media_path=str(tmp_path / "video-2.mp4"),
        size_bytes=10,
        source_sha256=second_digest,
    )
    second_asset = repository.get_video_asset("video-2")
    assert second_asset is not None

    with repository._connect() as connection:
        with pytest.raises(repository_module.sqlite3.IntegrityError):
            connection.execute(
                "UPDATE videos SET asset_id = ? WHERE id = ?",
                (second_asset.asset_id, "video-1"),
            )
        with pytest.raises(repository_module.sqlite3.IntegrityError):
            connection.execute(
                "UPDATE assets SET size_bytes = ? WHERE asset_id = ?",
                (11, second_asset.asset_id),
            )

    assert repository.get_video_asset("video-1").sha256 == first_digest  # type: ignore[union-attr]


def test_corrupt_asset_link_fails_closed_for_reads_and_stage_creation(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    digest = "a" * 64
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "video-1.mp4"),
        size_bytes=10,
        source_sha256=digest,
    )
    with repository._connect() as connection:
        connection.execute("DROP TRIGGER assets_immutable_update")
        connection.execute(
            "UPDATE assets SET created_at = ? WHERE asset_id = ?",
            ("not-a-timestamp", f"sha256:{digest}"),
        )

    with pytest.raises(AssetIdentityError, match="corrupt"):
        repository.get_video_asset("video-1")
    with pytest.raises(AssetIdentityError, match="corrupt"):
        repository.create_stage_run(
            video_id="video-1",
            specification=_speech_specification(),
            run_id="speech-run-1",
        )


def test_stage_run_resolves_source_hash_from_persisted_asset_link(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    digest = "a" * 64
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "video-1.mp4"),
        size_bytes=10,
        source_sha256=digest,
    )

    stage_run = repository.create_stage_run(
        video_id="video-1",
        specification=_speech_specification(),
        run_id="speech-run-1",
    )

    assert stage_run.source_sha256 == digest


def test_stage_run_requires_authoritative_asset_identity(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="legacy.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "video-1.mp4"),
        size_bytes=10,
    )

    with pytest.raises(AssetIdentityError, match="identity is unavailable"):
        repository.create_stage_run(
            video_id="video-1",
            specification=_speech_specification(),
            run_id="speech-run-1",
        )
