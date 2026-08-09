import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from videoscope.model_manifest import (
    FASTEMBED_MODEL,
    MODEL_REVISIONS,
    QWEN_VIDEO_MODEL,
    SIGLIP_224_MODEL,
    SIGLIP_384_MODEL,
    WHISPER_MODEL,
)


MODEL_PROFILES = {
    "ml": (
        (WHISPER_MODEL, MODEL_REVISIONS[WHISPER_MODEL]),
        (SIGLIP_224_MODEL, MODEL_REVISIONS[SIGLIP_224_MODEL]),
        (SIGLIP_384_MODEL, MODEL_REVISIONS[SIGLIP_384_MODEL]),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pinned VideoScope model snapshots")
    parser.add_argument(
        "--profile",
        action="append",
        choices=("ml", "video"),
        help="Profile to download; repeat for both. Defaults to both profiles.",
    )
    return parser.parse_args()


def main() -> None:
    requested = parse_args().profile or ["ml", "video"]
    profiles = list(dict.fromkeys(requested))
    for profile in profiles:
        for model, revision in MODEL_PROFILES[profile]:
            print(f"Downloading {model}@{revision}...")
            snapshot_download(model, revision=revision)

    if "ml" in profiles:
        fastembed_model = FASTEMBED_MODEL
        fastembed_revision = MODEL_REVISIONS[fastembed_model]
        print(f"Downloading multilingual MPNet ONNX@{fastembed_revision}...")
        snapshot_download(
            fastembed_model,
            revision=fastembed_revision,
            allow_patterns=list(FASTEMBED_FILES),
            cache_dir=Path("data/models/fastembed"),
        )
        print("Downloading local Roboflow RF-DETR Small...")
        rfdetr_dir = Path("data/models/rfdetr").resolve()
        rfdetr_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("RF_HOME", str(rfdetr_dir))
        from rfdetr import RFDETRSmall

        RFDETRSmall()
    print(f"VideoScope model profiles are ready: {', '.join(profiles)}.")


if __name__ == "__main__":
    main()
