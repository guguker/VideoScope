import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = (
    "mlx-community/whisper-large-v3-turbo",
    "google/siglip2-base-patch16-224",
    "google/siglip2-base-patch16-384",
    "mlx-community/Qwen3.5-9B-MLX-4bit",
)

FASTEMBED_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "onnx/model.onnx",
)


def main() -> None:
    for model in MODELS:
        print(f"Downloading {model}...")
        snapshot_download(model)
    print("Downloading multilingual MPNet ONNX...")
    snapshot_download(
        "xenova/paraphrase-multilingual-mpnet-base-v2",
        allow_patterns=list(FASTEMBED_FILES),
        cache_dir=Path("data/models/fastembed"),
    )
    print("Downloading local Roboflow RF-DETR Small...")
    rfdetr_dir = Path("data/models/rfdetr").resolve()
    rfdetr_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RF_HOME", str(rfdetr_dir))
    from rfdetr import RFDETRSmall

    RFDETRSmall()
    print("VideoScope models are ready.")


if __name__ == "__main__":
    main()
