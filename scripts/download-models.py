import argparse
from collections.abc import Callable
from hashlib import sha256
import os
from pathlib import Path
import shutil
import stat
import tempfile

from huggingface_hub import snapshot_download
from videoscope.model_manifest import (
    FASTEMBED_MODEL,
    MODEL_REVISIONS,
    QWEN_VIDEO_MODEL,
    SIGLIP_224_MODEL,
    SIGLIP_384_MODEL,
    WHISPER_MODEL,
)
from videoscope.providers.vision_worker_contract import RFDETR_SMALL_CHECKPOINT_SHA256


MODEL_PROFILES = {
    "base": (),
    "vision": (
        (SIGLIP_224_MODEL, MODEL_REVISIONS[SIGLIP_224_MODEL]),
        (SIGLIP_384_MODEL, MODEL_REVISIONS[SIGLIP_384_MODEL]),
    ),
    "whisper": (
        (WHISPER_MODEL, MODEL_REVISIONS[WHISPER_MODEL]),
    ),
    "video": (
        (QWEN_VIDEO_MODEL, MODEL_REVISIONS[QWEN_VIDEO_MODEL]),
    ),
}

FASTEMBED_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "onnx/model.onnx",
)


def file_sha256(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("model checkpoint is not a single-link regular file")
        digest = sha256()
        total = 0
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(1024 * 1024):
                total += len(chunk)
                digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise ValueError("model checkpoint changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def install_rfdetr_checkpoint(
    checkpoint: Path,
    *,
    downloader: Callable[[str], None],
    expected_sha256: str,
) -> None:
    """Publish a downloaded checkpoint only after stable SHA-256 verification."""
    checkpoint = Path(os.path.abspath(checkpoint))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and not checkpoint.is_symlink():
        try:
            if file_sha256(checkpoint) == expected_sha256:
                print(f"Verified existing RF-DETR checkpoint: {checkpoint}")
                return
        except (OSError, ValueError):
            pass

    staging_root = Path(
        tempfile.mkdtemp(prefix=".rfdetr-download-", dir=checkpoint.parent)
    )
    staging = staging_root / checkpoint.name
    try:
        downloader(str(staging))
        try:
            digest = file_sha256(staging)
        except (OSError, ValueError) as error:
            raise SystemExit(
                "Downloaded RF-DETR checkpoint failed the pinned SHA-256 check"
            ) from error
        if digest != expected_sha256:
            raise SystemExit(
                "Downloaded RF-DETR checkpoint failed the pinned SHA-256 check"
            )
        os.chmod(staging, 0o600, follow_symlinks=False)
        staging_descriptor = os.open(
            staging,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(staging_descriptor)
        finally:
            os.close(staging_descriptor)
        staging.replace(checkpoint)
        directory_descriptor = os.open(
            checkpoint.parent,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pinned VideoScope model snapshots")
    parser.add_argument(
        "--profile",
        action="append",
        choices=tuple(MODEL_PROFILES),
        required=True,
        help="Profile to download; repeat the option for multiple profiles.",
    )
    return parser.parse_args()


def main() -> None:
    profiles = list(dict.fromkeys(parse_args().profile))
    for profile in profiles:
        for model, revision in MODEL_PROFILES[profile]:
            print(f"Downloading {model}@{revision}...")
            snapshot_download(model, revision=revision)

    if "base" in profiles:
        fastembed_model = FASTEMBED_MODEL
        fastembed_revision = MODEL_REVISIONS[fastembed_model]
        print(f"Downloading multilingual MPNet ONNX@{fastembed_revision}...")
        snapshot_download(
            fastembed_model,
            revision=fastembed_revision,
            allow_patterns=list(FASTEMBED_FILES),
            cache_dir=Path("data/models/fastembed"),
        )
    if "vision" in profiles:
        from rfdetr.assets.model_weights import download_pretrain_weights

        checkpoint = Path("data/models/rfdetr/rf-detr-small.pth").resolve()
        print(f"Downloading pinned RF-DETR Small checkpoint to {checkpoint}...")
        install_rfdetr_checkpoint(
            checkpoint,
            downloader=download_pretrain_weights,
            expected_sha256=RFDETR_SMALL_CHECKPOINT_SHA256,
        )
    print(f"VideoScope model profiles are ready: {', '.join(profiles)}.")


if __name__ == "__main__":
    main()
