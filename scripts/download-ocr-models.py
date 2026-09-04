#!/usr/bin/env python3
"""Acquire the reviewed OCR model bytes from immutable Hugging Face revisions."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = ROOT / "workers" / "ocr" / "model-sources.lock.json"
DEFAULT_ARTIFACT_MANIFEST = ROOT / "workers" / "ocr" / "model-artifacts.lock.json"
DEFAULT_DESTINATION = Path.home() / ".paddlex" / "official_models"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
Downloader = Callable[[str, str, Sequence[str], Path], None]

_HF_DOWNLOAD_ENVIRONMENT = {
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_XET": "1",
    "NO_PROXY": "*",
    "no_proxy": "*",
}
_HF_DOWNLOAD_PROGRAM = """\
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download

repository, revision, raw_destination, *includes = sys.argv[1:]
destination = Path(raw_destination)
destination.mkdir(parents=True, exist_ok=False, mode=0o700)
cache = destination / ".hub-cache"
try:
    for name in includes:
        downloaded = hf_hub_download(
            repo_id=repository,
            filename=name,
            revision=revision,
            cache_dir=str(cache),
            endpoint="https://huggingface.co",
            token=False,
            local_files_only=False,
        )
        shutil.copyfile(downloaded, destination / name)
finally:
    shutil.rmtree(cache, ignore_errors=True)
"""


def _load_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit("OCR model contract is unreadable or invalid") from error
    if not isinstance(payload, dict):
        raise SystemExit("OCR model contract must be a JSON object")
    return payload


def _safe_name(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise SystemExit(f"OCR {label} is invalid")
    return value


def _artifact_record(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"name", "sha256", "size"}:
        raise SystemExit("OCR artifact record is invalid")
    name = _safe_name(value.get("name"), label="artifact name")
    digest = value.get("sha256")
    size = value.get("size")
    if (
        type(digest) is not str
        or _SHA256_RE.fullmatch(digest) is None
        or type(size) is not int
        or size <= 0
    ):
        raise SystemExit(f"OCR artifact contract is invalid for {name}")
    return {"name": name, "sha256": digest, "size": size}


def _load_contracts(
    source_manifest_path: Path,
    artifact_manifest_path: Path,
) -> tuple[tuple[dict[str, object], dict[str, object]], ...]:
    sources = _load_object(source_manifest_path)
    artifacts = _load_object(artifact_manifest_path)
    if (
        set(sources) != {"models", "provider", "schema_version"}
        or sources.get("schema_version") != 1
        or sources.get("provider") != "huggingface_hub"
        or not isinstance(sources.get("models"), list)
        or set(artifacts) != {"engine", "models", "profile", "schema_version"}
        or artifacts.get("schema_version") != 1
        or artifacts.get("engine") != "transformers"
        or not isinstance(artifacts.get("models"), list)
    ):
        raise SystemExit("OCR model manifests use an unsupported contract")

    source_by_directory: dict[str, dict[str, object]] = {}
    for raw in sources["models"]:
        if not isinstance(raw, dict) or set(raw) != {
            "directory",
            "license",
            "license_evidence",
            "repository",
            "revision",
            "role",
        }:
            raise SystemExit("OCR source record is invalid")
        directory = _safe_name(raw.get("directory"), label="source directory")
        role = _safe_name(raw.get("role"), label="source role")
        repository = raw.get("repository")
        revision = raw.get("revision")
        evidence = _artifact_record(raw.get("license_evidence"))
        if (
            directory in source_by_directory
            or raw.get("license") != "apache-2.0"
            or type(repository) is not str
            or repository.count("/") != 1
            or any(part in {"", ".", ".."} for part in repository.split("/"))
            or type(revision) is not str
            or _REVISION_RE.fullmatch(revision) is None
            or evidence["name"] != "README.md"
        ):
            raise SystemExit(f"OCR source contract is invalid for {directory}")
        source_by_directory[directory] = {
            **raw,
            "directory": directory,
            "license_evidence": evidence,
            "role": role,
        }

    paired: list[tuple[dict[str, object], dict[str, object]]] = []
    seen: set[str] = set()
    for raw in artifacts["models"]:
        if not isinstance(raw, dict) or set(raw) != {
            "artifacts",
            "directory",
            "role",
        }:
            raise SystemExit("OCR artifact model record is invalid")
        directory = _safe_name(raw.get("directory"), label="model directory")
        role = _safe_name(raw.get("role"), label="model role")
        records = raw.get("artifacts")
        source = source_by_directory.get(directory)
        if (
            directory in seen
            or source is None
            or source["role"] != role
            or not isinstance(records, list)
            or not records
        ):
            raise SystemExit(f"OCR source/artifact contract mismatch for {directory}")
        normalized = [_artifact_record(record) for record in records]
        names = [str(record["name"]) for record in normalized]
        if len(names) != len(set(names)) or "README.md" in names:
            raise SystemExit(f"OCR artifact names are invalid for {directory}")
        paired.append((source, {**raw, "artifacts": normalized, "directory": directory}))
        seen.add(directory)
    if seen != set(source_by_directory):
        raise SystemExit("OCR source manifest does not match artifact manifest")
    return tuple(paired)


def _digest_regular_file(path: Path, expected_size: int) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SystemExit("OCR artifact is missing or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SystemExit("OCR artifact is not a private regular file")
        if metadata.st_size != expected_size:
            raise SystemExit("OCR artifact size does not match the reviewed contract")
        digest = sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, expected_size - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > expected_size:
                raise SystemExit("OCR artifact size does not match the reviewed contract")
            digest.update(chunk)
        if total != expected_size:
            raise SystemExit("OCR artifact size does not match the reviewed contract")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _verify_model(directory: Path, source: dict[str, object], model: dict[str, object]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise SystemExit("OCR model directory is missing or unsafe")
    records = [source["license_evidence"], *model["artifacts"]]
    for record in records:
        assert isinstance(record, dict)
        path = directory / str(record["name"])
        actual = _digest_regular_file(path, int(record["size"]))
        if actual != record["sha256"]:
            raise SystemExit("OCR artifact SHA-256 does not match the reviewed contract")


def _hf_download(
    repository: str,
    revision: str,
    includes: Sequence[str],
    destination: Path,
) -> None:
    command = [
        sys.executable,
        "-I",
        "-c",
        _HF_DOWNLOAD_PROGRAM,
        repository,
        revision,
        str(destination),
        *includes,
    ]
    completed = subprocess.run(
        command,
        check=False,
        env=dict(_HF_DOWNLOAD_ENVIRONMENT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise SystemExit("pinned OCR model download failed")


def install_ocr_models(
    destination: Path,
    *,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    artifact_manifest_path: Path = DEFAULT_ARTIFACT_MANIFEST,
    downloader: Downloader = _hf_download,
) -> tuple[str, ...]:
    contracts = _load_contracts(source_manifest_path, artifact_manifest_path)
    destination = Path(destination).absolute()
    if destination.is_symlink():
        raise SystemExit("OCR model root must not be a symlink")
    try:
        destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    except FileExistsError:
        if destination.is_symlink() or not destination.is_dir():
            raise SystemExit("OCR model root is not a safe directory") from None
    else:
        os.chmod(destination, 0o700)

    missing: list[tuple[dict[str, object], dict[str, object]]] = []
    for source, model in contracts:
        target = destination / str(model["directory"])
        if target.exists() or target.is_symlink():
            try:
                _verify_model(target, source, model)
            except SystemExit as error:
                raise SystemExit(
                    "existing OCR model is invalid; refusing to replace user state"
                ) from error
        else:
            missing.append((source, model))

    if not missing:
        return tuple(str(model["directory"]) for _, model in contracts)

    staging_root = Path(
        tempfile.mkdtemp(prefix=".ocr-download-", dir=destination)
    )
    os.chmod(staging_root, 0o700)
    try:
        prepared: list[tuple[Path, dict[str, object], dict[str, object]]] = []
        for source, model in missing:
            staging = staging_root / str(model["directory"])
            includes = sorted(
                {
                    str(source["license_evidence"]["name"]),
                    *(str(record["name"]) for record in model["artifacts"]),
                }
            )
            downloader(
                str(source["repository"]),
                str(source["revision"]),
                includes,
                staging,
            )
            _verify_model(staging, source, model)
            os.chmod(staging, 0o700)
            for name in includes:
                os.chmod(staging / name, 0o400)
            prepared.append((staging, source, model))

        for staging, source, model in prepared:
            target = destination / str(model["directory"])
            if target.exists() or target.is_symlink():
                raise SystemExit("OCR model appeared during publish; refusing to replace it")
            staging.rename(target)
            _verify_model(target, source, model)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return tuple(str(model["directory"]) for _, model in contracts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--artifact-manifest", type=Path, default=DEFAULT_ARTIFACT_MANIFEST)
    args = parser.parse_args()
    installed = install_ocr_models(
        args.destination,
        source_manifest_path=args.source_manifest,
        artifact_manifest_path=args.artifact_manifest,
    )
    print("Verified OCR models: " + ", ".join(installed))


if __name__ == "__main__":
    main()
