from __future__ import annotations

from pathlib import Path


WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
SIGLIP_224_MODEL = "google/siglip2-base-patch16-224"
SIGLIP_384_MODEL = "google/siglip2-base-patch16-384"
QWEN_VIDEO_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
TEXT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TEXT_EMBEDDING_DIMENSIONS = 768
FASTEMBED_REPOSITORY = "xenova/paraphrase-multilingual-mpnet-base-v2"
FASTEMBED_RUNTIME_VERSION = "0.8.0"
FASTEMBED_ALGORITHM_VERSION = "mean-pooling-v1"
# Backwards-compatible name used by the downloader.
FASTEMBED_MODEL = FASTEMBED_REPOSITORY
LIGHTHOUSE_CHECKPOINT_SHA256 = (
    "42798d352dde089a835cb4995eb9c8084a2e97337166abbbb09c12856aec2c55"
)
LIGHTHOUSE_CLIP_CHECKPOINT_SHA256 = (
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
)
LIGHTHOUSE_SOURCE_REVISION = "d095eaa552cecef240897a8b750306b3b2a08740"
LIGHTHOUSE_CLIP_REVISION = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"

MODEL_REVISIONS: dict[str, str] = {
    WHISPER_MODEL: "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
    SIGLIP_224_MODEL: "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    SIGLIP_384_MODEL: "f775b65a79762255128c981547af89addcfe0f88",
    QWEN_VIDEO_MODEL: "938d8919941c6e7efd3c7150eff7fe9d12afa631",
    FASTEMBED_MODEL: "e5d116277351513fd260955ece953ecddde7046e",
}


def model_revision(model_name: str | None) -> str | None:
    return MODEL_REVISIONS.get(model_name or "")


def model_identity(model_name: str, revision: str | None) -> str:
    """Return the identity that must key every artifact derived from a model."""
    return f"{model_name}@{revision}" if revision else model_name


def fastembed_snapshot(model_name: str) -> tuple[str | None, str | None]:
    """Map FastEmbed's public model name to its immutable ONNX repository."""
    if model_name != TEXT_EMBEDDING_MODEL:
        return None, None
    return FASTEMBED_REPOSITORY, MODEL_REVISIONS[FASTEMBED_REPOSITORY]


def fastembed_cache_snapshot_path(
    cache_dir: Path,
    model_name: str,
) -> Path | None:
    """Resolve the one immutable Hugging Face cache snapshot used by FastEmbed.

    This function is deliberately lexical: callers must still open and attest the
    returned directory without following unsafe path components. It never invokes
    a hub client and therefore cannot download or advance a mutable revision.
    """
    repository, revision = fastembed_snapshot(model_name)
    if repository is None or revision is None:
        return None
    organization, name = repository.split("/", maxsplit=1)
    return (
        Path(cache_dir).absolute()
        / f"models--{organization}--{name}"
        / "snapshots"
        / revision
    )
