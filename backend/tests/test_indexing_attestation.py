from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from videoscope.indexing_attestation import (
    MAX_ATTESTED_EXECUTABLE_BYTES,
    IndexingToolchainAttestationError,
    attest_indexing_toolchain,
)


_LOCKED_VERSIONS = {
    "numpy": "2.5.1",
    "opencv-python": "4.14.0.94",
    "qdrant-client": "1.19.0",
    "scenedetect": "0.6.7",
    "videoscope-backend": "0.1.0",
}


def _write_tool(path: Path, output: str, *, exit_code: int = 0, delay: float = 0) -> Path:
    delay_command = f"/bin/sleep {delay}\n" if delay else ""
    path.write_text(
        "#!/bin/sh\n"
        f"{delay_command}"
        f"printf '%s' '{output}'\n"
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    path.chmod(0o555)
    return path


def _write_dynamic_version_tool(path: Path, version_file: Path) -> Path:
    path.write_text(
        "#!/bin/sh\n"
        f"/bin/cat '{version_file.as_posix()}'\n",
        encoding="utf-8",
    )
    path.chmod(0o555)
    return path


def _write_lock(path: Path, versions: dict[str, str] | None = None) -> tuple[Path, str]:
    selected = versions or _LOCKED_VERSIONS
    payload = "version = 1\n" + "".join(
        "\n[[package]]\n"
        f'name = "{name}"\n'
        f'version = "{version}"\n'
        for name, version in sorted(selected.items())
    )
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o444)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return path, f"sha256:{digest}"


def _resolver(versions: dict[str, str]):
    def resolve(name: str) -> str:
        if name not in versions:
            raise LookupError(name)
        return versions[name]

    return resolve


def _attest(
    tmp_path: Path,
    *,
    ffmpeg: Path | None = None,
    ffprobe: Path | None = None,
    lock_path: Path | None = None,
    reviewed_lock_sha256: str | None = None,
    versions: dict[str, str] | None = None,
    version_timeout: float = 5.0,
    python_runtime_identity=None,
):
    selected_ffmpeg = ffmpeg or _write_tool(tmp_path / "ffmpeg", "ffmpeg version 1\n")
    selected_ffprobe = ffprobe or _write_tool(tmp_path / "ffprobe", "ffprobe version 1\n")
    if lock_path is None:
        selected_lock, selected_digest = _write_lock(tmp_path / "uv.lock")
    else:
        selected_lock = lock_path
        assert reviewed_lock_sha256 is not None
        selected_digest = reviewed_lock_sha256
    selected_versions = dict(versions or _LOCKED_VERSIONS)
    return attest_indexing_toolchain(
        ffmpeg_binary=selected_ffmpeg,
        ffprobe_binary=selected_ffprobe,
        lock_path=selected_lock,
        reviewed_lock_sha256=selected_digest,
        distribution_version=_resolver(selected_versions),
        version_timeout=version_timeout,
        **(
            {"python_runtime_identity": python_runtime_identity}
            if python_runtime_identity is not None
            else {}
        ),
    )


def test_attestation_is_pathless_deterministic_and_builds_exact_ffmpeg(tmp_path) -> None:
    first_dir = tmp_path / "private-one"
    second_dir = tmp_path / "private-two"
    first_dir.mkdir()
    second_dir.mkdir()

    first = _attest(first_dir)
    second = _attest(second_dir)

    assert first.identity == second.identity
    assert first.identity.startswith("sha256:")
    assert json.loads(first.canonical_json)["execution_policy"] == {
        "subprocess_environment": "empty-v1"
    }
    assert str(first_dir) not in first.canonical_json
    assert str(second_dir) not in second.canonical_json
    assert "health" not in first.canonical_json
    runner = first.create_ffmpeg()
    assert runner.ffmpeg_binary == str((first_dir / "ffmpeg").resolve())
    assert runner.ffprobe_binary == str((first_dir / "ffprobe").resolve())
    assert first.verify_current() == first.identity


def test_python_runtime_platform_is_pathless_normalized_and_identity_bound(
    tmp_path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _attest(
        first_dir,
        python_runtime_identity=lambda: {
            "implementation": "CPython",
            "version": "3.12.10",
            "system": "Darwin",
            "machine": "ARM64",
        },
    )
    second = _attest(
        second_dir,
        python_runtime_identity=lambda: {
            "implementation": "cpython",
            "version": "3.12.11",
            "system": "darwin",
            "machine": "aarch64",
        },
    )

    payload = json.loads(first.canonical_json)
    assert payload["python_runtime"] == {
        "implementation": "cpython",
        "machine": "aarch64",
        "system": "darwin",
        "version": "3.12.10",
    }
    assert first.identity != second.identity
    assert str(tmp_path) not in first.canonical_json


def test_python_runtime_platform_drift_is_rechecked_and_fails_closed(tmp_path) -> None:
    current = {
        "implementation": "cpython",
        "version": "3.12.10",
        "system": "linux",
        "machine": "x86_64",
    }
    attested = _attest(
        tmp_path,
        python_runtime_identity=lambda: dict(current),
    )
    current["version"] = "3.12.11"

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        attested.verify_current()

    assert raised.value.code == "python_runtime_drift"
    assert str(tmp_path) not in str(raised.value)


def test_command_symlinks_are_resolved_once_to_regular_targets(tmp_path) -> None:
    first_target = _write_tool(tmp_path / "ffmpeg-real", "ffmpeg version 1\n")
    second_target = _write_tool(tmp_path / "ffmpeg-other", "ffmpeg version 2\n")
    alias = tmp_path / "ffmpeg"
    alias.symlink_to(first_target.name)

    attested = _attest(tmp_path, ffmpeg=alias)
    alias.unlink()
    alias.symlink_to(second_target.name)

    assert attested.ffmpeg_binary == str(first_target.resolve())
    assert attested.verify_current() == attested.identity


def test_path_replacement_with_identical_bytes_is_detected_by_private_seal(tmp_path) -> None:
    ffmpeg = _write_tool(tmp_path / "ffmpeg", "ffmpeg version 1\n")
    attested = _attest(tmp_path, ffmpeg=ffmpeg)
    original = ffmpeg.read_bytes()
    ffmpeg.unlink()
    ffmpeg.write_bytes(original)
    ffmpeg.chmod(0o555)

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        attested.verify_current()

    assert raised.value.code == "executable_drift"
    assert str(tmp_path) not in str(raised.value)


def test_verify_rejects_replacement_before_executing_its_version_command(tmp_path) -> None:
    ffmpeg = _write_tool(tmp_path / "ffmpeg", "ffmpeg version 1\n")
    attested = _attest(tmp_path, ffmpeg=ffmpeg)
    marker = tmp_path / "replacement-was-executed"
    ffmpeg.unlink()
    ffmpeg.write_text(
        "#!/bin/sh\n"
        f"/usr/bin/touch '{marker.as_posix()}'\n"
        "printf 'ffmpeg version hostile\\n'\n",
        encoding="utf-8",
    )
    ffmpeg.chmod(0o555)

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        attested.verify_current()

    assert raised.value.code == "executable_drift"
    assert not marker.exists()


def test_path_replacement_during_version_command_is_rejected(tmp_path) -> None:
    ffmpeg = tmp_path / "ffmpeg"
    replacement = (
        "#!/bin/sh\n"
        "/bin/echo ffmpeg-version-replacement\n"
    )
    ffmpeg.write_text(
        "#!/bin/sh\n"
        f"/bin/rm '{ffmpeg.as_posix()}'\n"
        f"printf '%s' '{replacement}' > '{ffmpeg.as_posix()}'\n"
        f"/bin/chmod 555 '{ffmpeg.as_posix()}'\n"
        "printf 'ffmpeg version original\\n'\n",
        encoding="utf-8",
    )
    ffmpeg.chmod(0o555)

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(tmp_path, ffmpeg=ffmpeg)

    assert raised.value.code == "executable_not_stable"


def test_hardlinked_executable_is_rejected(tmp_path) -> None:
    source = _write_tool(tmp_path / "ffmpeg-real", "ffmpeg version 1\n")
    hardlink = tmp_path / "ffmpeg"
    os.link(source, hardlink)

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(tmp_path, ffmpeg=hardlink)

    assert raised.value.code == "executable_not_stable"


def test_special_executable_is_rejected_without_blocking(tmp_path) -> None:
    fifo = tmp_path / "ffmpeg"
    os.mkfifo(fifo)

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(tmp_path, ffmpeg=fifo)

    assert raised.value.code == "executable_not_regular"


def test_oversized_sparse_executable_is_rejected_before_read(tmp_path) -> None:
    ffmpeg = tmp_path / "ffmpeg"
    with ffmpeg.open("wb") as handle:
        handle.truncate(MAX_ATTESTED_EXECUTABLE_BYTES + 1)
    ffmpeg.chmod(0o555)

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(tmp_path, ffmpeg=ffmpeg)

    assert raised.value.code == "executable_too_large"


def test_nonzero_version_command_fails_closed_with_sanitized_error(tmp_path) -> None:
    ffmpeg = _write_tool(
        tmp_path / "ffmpeg",
        "/Users/private/token=super-secret\n",
        exit_code=7,
    )

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(tmp_path, ffmpeg=ffmpeg)

    assert raised.value.code == "version_command_failed"
    assert "super-secret" not in str(raised.value)
    assert str(tmp_path) not in str(raised.value)


def test_version_command_timeout_is_bounded_and_fails_closed(tmp_path) -> None:
    ffmpeg = _write_tool(
        tmp_path / "ffmpeg",
        "ffmpeg version too late\n",
        delay=1.0,
    )

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(tmp_path, ffmpeg=ffmpeg, version_timeout=0.05)

    assert raised.value.code == "version_command_timeout"


def test_oversized_version_output_is_bounded_and_rejected(tmp_path) -> None:
    ffmpeg = _write_tool(tmp_path / "ffmpeg", "x" * 70_000)

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(tmp_path, ffmpeg=ffmpeg)

    assert raised.value.code == "version_output_too_large"


def test_empty_version_output_is_rejected(tmp_path) -> None:
    ffmpeg = _write_tool(tmp_path / "ffmpeg", "")

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(tmp_path, ffmpeg=ffmpeg)

    assert raised.value.code == "version_output_invalid"


def test_non_executable_regular_file_is_rejected(tmp_path) -> None:
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_text("not executable", encoding="utf-8")
    ffmpeg.chmod(0o444)

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(tmp_path, ffmpeg=ffmpeg)

    assert raised.value.code == "executable_not_executable"


def test_missing_bare_command_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    lock_path, digest = _write_lock(tmp_path / "uv.lock")

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        attest_indexing_toolchain(
            ffmpeg_binary="not-installed",
            ffprobe_binary="also-not-installed",
            lock_path=lock_path,
            reviewed_lock_sha256=digest,
            distribution_version=_resolver(dict(_LOCKED_VERSIONS)),
        )

    assert raised.value.code == "executable_not_found"


def test_version_output_drift_is_detected_even_when_executable_is_unchanged(tmp_path) -> None:
    version_file = tmp_path / "ffmpeg.version"
    version_file.write_text("ffmpeg version 1\n", encoding="utf-8")
    ffmpeg = _write_dynamic_version_tool(tmp_path / "ffmpeg", version_file)
    attested = _attest(tmp_path, ffmpeg=ffmpeg)

    version_file.write_text("ffmpeg version 2\n", encoding="utf-8")

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        attested.verify_current()

    assert raised.value.code == "executable_drift"


def test_distribution_drift_is_detected_by_verify_current(tmp_path) -> None:
    versions = dict(_LOCKED_VERSIONS)
    lock_path, digest = _write_lock(tmp_path / "uv.lock")
    ffmpeg = _write_tool(tmp_path / "ffmpeg", "ffmpeg version 1\n")
    ffprobe = _write_tool(tmp_path / "ffprobe", "ffprobe version 1\n")

    def resolve(name: str) -> str:
        return versions[name]

    attested = attest_indexing_toolchain(
        ffmpeg_binary=ffmpeg,
        ffprobe_binary=ffprobe,
        lock_path=lock_path,
        reviewed_lock_sha256=digest,
        distribution_version=resolve,
    )
    versions["numpy"] = "999.0"

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        attested.verify_current()

    assert raised.value.code == "distribution_version_mismatch"


def test_missing_distribution_fails_closed(tmp_path) -> None:
    versions = dict(_LOCKED_VERSIONS)
    versions.pop("qdrant-client")

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(tmp_path, versions=versions)

    assert raised.value.code == "distribution_unavailable"


def test_reviewed_dev_and_bootstrap_extras_do_not_change_required_identity(
    tmp_path,
) -> None:
    first_dir = tmp_path / "base"
    second_dir = tmp_path / "dev"
    first_dir.mkdir()
    second_dir.mkdir()
    base = _attest(first_dir)
    installed = {
        **_LOCKED_VERSIONS,
        "pip": "25.0.1",
        "pytest": "8.4.2",
        "torch": "2.13.0",
        "uv": "0.12.3",
        "unowned-debug-helper": "1.0",
    }
    requested: list[str] = []

    def resolve(name: str) -> str:
        requested.append(name)
        return installed[name]

    lock_path, digest = _write_lock(second_dir / "uv.lock")
    dev = attest_indexing_toolchain(
        ffmpeg_binary=_write_tool(second_dir / "ffmpeg", "ffmpeg version 1\n"),
        ffprobe_binary=_write_tool(second_dir / "ffprobe", "ffprobe version 1\n"),
        lock_path=lock_path,
        reviewed_lock_sha256=digest,
        distribution_version=resolve,
    )

    assert dev.identity == base.identity
    assert requested == sorted(_LOCKED_VERSIONS)
    assert "pytest" not in dev.canonical_json
    assert "unowned-debug-helper" not in dev.canonical_json


def test_unowned_lookalike_cannot_substitute_for_exact_distribution(tmp_path) -> None:
    installed = dict(_LOCKED_VERSIONS)
    installed.pop("opencv-python")
    installed["opencv-contrib-python"] = "4.14.0.94"

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(tmp_path, versions=installed)

    assert raised.value.code == "distribution_unavailable"


def test_lock_version_drift_is_rejected_before_executable_execution(tmp_path) -> None:
    versions = dict(_LOCKED_VERSIONS)
    versions["numpy"] = "2.5.2"
    lock_path, digest = _write_lock(tmp_path / "uv.lock", versions)
    marker = tmp_path / "executed"
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_text(
        "#!/bin/sh\n"
        f"/usr/bin/touch '{marker.as_posix()}'\n"
        "printf 'ffmpeg version 1\\n'\n",
        encoding="utf-8",
    )
    ffmpeg.chmod(0o555)

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(
            tmp_path,
            ffmpeg=ffmpeg,
            lock_path=lock_path,
            reviewed_lock_sha256=digest,
        )

    assert raised.value.code == "distribution_version_mismatch"
    assert not marker.exists()


def test_reviewed_lock_content_and_inode_drift_are_detected(tmp_path) -> None:
    lock_path, digest = _write_lock(tmp_path / "uv.lock")
    attested = _attest(
        tmp_path,
        lock_path=lock_path,
        reviewed_lock_sha256=digest,
    )
    original = lock_path.read_bytes()
    lock_path.chmod(0o644)
    lock_path.unlink()
    lock_path.write_bytes(original)
    lock_path.chmod(0o444)

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        attested.verify_current()

    assert raised.value.code == "lock_drift"


def test_unreviewed_lock_digest_is_rejected(tmp_path) -> None:
    lock_path, _digest = _write_lock(tmp_path / "uv.lock")

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(
            tmp_path,
            lock_path=lock_path,
            reviewed_lock_sha256="sha256:" + "0" * 64,
        )

    assert raised.value.code == "lock_identity_mismatch"


def test_exactly_reviewed_but_malformed_lock_is_rejected(tmp_path) -> None:
    payload = b"version = [definitely not TOML"
    lock_path = tmp_path / "uv.lock"
    lock_path.write_bytes(payload)
    lock_path.chmod(0o444)

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(
            tmp_path,
            lock_path=lock_path,
            reviewed_lock_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        )

    assert raised.value.code == "lock_invalid"


def test_reviewed_lock_missing_required_distribution_is_rejected(tmp_path) -> None:
    versions = dict(_LOCKED_VERSIONS)
    versions.pop("scenedetect")
    lock_path, digest = _write_lock(tmp_path / "uv.lock", versions)

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(
            tmp_path,
            lock_path=lock_path,
            reviewed_lock_sha256=digest,
        )

    assert raised.value.code == "lock_invalid"


def test_lock_symlink_is_rejected_even_without_a_hardlink(tmp_path) -> None:
    original, digest = _write_lock(tmp_path / "reviewed.lock")
    symlink = tmp_path / "symlink.lock"
    symlink.symlink_to(original.name)

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(
            tmp_path,
            lock_path=symlink,
            reviewed_lock_sha256=digest,
        )

    assert raised.value.code == "lock_not_stable"


def test_lock_hardlink_is_rejected(tmp_path) -> None:
    original, digest = _write_lock(tmp_path / "reviewed.lock")
    hardlink = tmp_path / "hardlink.lock"
    os.link(original, hardlink)

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(
            tmp_path,
            lock_path=hardlink,
            reviewed_lock_sha256=digest,
        )

    assert raised.value.code == "lock_not_stable"


def test_bare_commands_are_resolved_to_exact_absolute_paths(tmp_path, monkeypatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_tool(bin_dir / "ffmpeg", "ffmpeg version 1\n")
    _write_tool(bin_dir / "ffprobe", "ffprobe version 1\n")
    monkeypatch.setenv("PATH", str(bin_dir))
    lock_path, digest = _write_lock(tmp_path / "uv.lock")

    attested = attest_indexing_toolchain(
        ffmpeg_binary="ffmpeg",
        ffprobe_binary="ffprobe",
        lock_path=lock_path,
        reviewed_lock_sha256=digest,
        distribution_version=_resolver(dict(_LOCKED_VERSIONS)),
    )

    assert Path(attested.ffmpeg_binary).is_absolute()
    assert Path(attested.ffprobe_binary).is_absolute()
    assert shutil.which("ffmpeg") != attested.identity


def test_version_command_has_no_inherited_environment_or_shell_proxy(
    tmp_path,
    monkeypatch,
) -> None:
    marker = tmp_path / "must-not-exist"
    hostile_dir = tmp_path / "bin;touch must-not-exist"
    hostile_dir.mkdir()
    ffmpeg = hostile_dir / "ffmpeg"
    ffmpeg.write_text(
        "#!/bin/sh\n"
        "if [ -n \"${VIDEOSCOPE_ATTESTATION_SECRET-}\" ]; then\n"
        f"  /usr/bin/touch '{marker.as_posix()}'\n"
        "fi\n"
        "printf 'ffmpeg version clean\\n'\n",
        encoding="utf-8",
    )
    ffmpeg.chmod(0o555)
    monkeypatch.setenv("VIDEOSCOPE_ATTESTATION_SECRET", "leak-me")

    attested = _attest(tmp_path, ffmpeg=ffmpeg)

    assert attested.verify_current() == attested.identity
    assert not marker.exists()


def test_repository_default_lock_digest_and_versions_are_reviewed(tmp_path) -> None:
    ffmpeg = _write_tool(tmp_path / "ffmpeg", "ffmpeg version 1\n")
    ffprobe = _write_tool(tmp_path / "ffprobe", "ffprobe version 1\n")

    attested = attest_indexing_toolchain(
        ffmpeg_binary=ffmpeg,
        ffprobe_binary=ffprobe,
        distribution_version=_resolver(dict(_LOCKED_VERSIONS)),
    )

    assert attested.distribution_versions == _LOCKED_VERSIONS


def test_ffmpeg_factory_rejects_non_absolute_attested_paths() -> None:
    from videoscope.media.ffmpeg import FFmpeg

    with pytest.raises(ValueError, match="must be absolute"):
        FFmpeg.from_attested_paths(Path("ffmpeg"), Path("ffprobe"))


@pytest.mark.parametrize(
    ("reviewed_digest", "code"),
    [
        ("", "invalid_configuration"),
        ("0" * 64, "invalid_configuration"),
        ("sha256:" + "A" * 64, "invalid_configuration"),
    ],
)
def test_invalid_reviewed_lock_identity_is_rejected(
    tmp_path,
    reviewed_digest: str,
    code: str,
) -> None:
    lock_path, _ = _write_lock(tmp_path / "uv.lock")

    with pytest.raises(IndexingToolchainAttestationError) as raised:
        _attest(
            tmp_path,
            lock_path=lock_path,
            reviewed_lock_sha256=reviewed_digest,
        )

    assert raised.value.code == code


@pytest.mark.parametrize("timeout", [True, 0, 30.1, "5"])
def test_invalid_version_timeout_is_rejected(tmp_path, timeout: object) -> None:
    with pytest.raises(IndexingToolchainAttestationError) as raised:
        attest_indexing_toolchain(
            lock_path=tmp_path / "unused.lock",
            distribution_version=_resolver(dict(_LOCKED_VERSIONS)),
            version_timeout=timeout,  # type: ignore[arg-type]
        )

    assert raised.value.code == "invalid_configuration"


def test_non_callable_distribution_resolver_is_rejected(tmp_path) -> None:
    with pytest.raises(IndexingToolchainAttestationError) as raised:
        attest_indexing_toolchain(
            lock_path=tmp_path / "unused.lock",
            distribution_version=None,  # type: ignore[arg-type]
        )

    assert raised.value.code == "invalid_configuration"


def test_non_path_lock_configuration_is_rejected_without_leaking_type_error() -> None:
    with pytest.raises(IndexingToolchainAttestationError) as raised:
        attest_indexing_toolchain(
            lock_path=object(),  # type: ignore[arg-type]
            distribution_version=_resolver(dict(_LOCKED_VERSIONS)),
        )

    assert raised.value.code == "invalid_configuration"
