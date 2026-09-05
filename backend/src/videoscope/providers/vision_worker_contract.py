from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


VISION_WORKER_SCHEMA_VERSION = "vision-worker-v1"
VISION_WORKER_LOCK_SHA256 = (
    "261be2a4694b09e0b0e4a9b1919feb05a6d5fa334f46bfb2b7d261cb5e3487c6"
)
SIGLIP_COMPUTE_BACKEND = "mps"
SIGLIP_COMPUTE_DTYPE = "float32"
RFDETR_COMPUTE_BACKEND = "mps"
RFDETR_COMPUTE_DTYPE = "float32"
VISION_WORKER_MINIMUM_MACOS_MAJOR = 14
VISION_WORKER_RUNTIME_IDENTITY = (
    "videoscope-vision-worker-v1|python==3.12.13|"
    "platform==aarch64-apple-darwin|"
    f"macos-major>={VISION_WORKER_MINIMUM_MACOS_MAJOR}|"
    f"siglip-backend=={SIGLIP_COMPUTE_BACKEND}|"
    f"siglip-dtype=={SIGLIP_COMPUTE_DTYPE}|"
    f"rfdetr-backend=={RFDETR_COMPUTE_BACKEND}|"
    f"rfdetr-dtype=={RFDETR_COMPUTE_DTYPE}|"
    "mps-residency==exclusive-vision-backbone-v1|"
    "detector-release==stage-bound-v1|"
    f"lock-sha256:{VISION_WORKER_LOCK_SHA256}"
)
VISION_WORKER_PYTHON_VERSION = (3, 12, 13)
VISION_WORKER_CORE_DISTRIBUTIONS = {
    "fastapi": "0.141.1",
    "huggingface-hub": "1.27.0",
    "numpy": "2.5.1",
    "opencv-python": "4.14.0.94",
    "pillow": "12.3.0",
    "pydantic": "2.13.4",
    "pydantic-settings": "2.15.0",
    "rfdetr": "1.7.1",
    "supervision": "0.28.0",
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "transformers": "5.14.1",
    "uvicorn": "0.52.1",
}

MAX_REQUEST_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_IMAGES = 32
MAX_TEXTS = 64
MAX_TEXT_CHARS = 500
SIGLIP_MAX_TEXT_TOKENS = 64
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_DIMENSION = 8192
MAX_IMAGE_PIXELS = 40_000_000
MAX_BATCH_IMAGE_BYTES = 128 * 1024 * 1024
MAX_BATCH_IMAGE_PIXELS = 160_000_000
MAX_DETECTIONS = 500
MAX_EMBEDDING_DIMENSIONS = 4096
MAX_PROBE_BYTES = 4096

SIGLIP_PREPROCESSING_REVISION = "siglip2-auto-processor-rgb-normalized-v1"
SIGLIP_TOKENIZER_REVISION = "siglip2-auto-tokenizer-max-length-64-truncation-v1"
RFDETR_ADAPTER_REVISION = "rfdetr-coco-rgb-clipped-center-box-v3"
RFDETR_SMALL_CHECKPOINT_SHA256 = (
    "d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33"
)
REVIEWED_SIGLIP_PROFILES: dict[tuple[str, str], tuple[int, float, float]] = {
    (
        "google/siglip2-base-patch16-224",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    ): (768, 112.66890226978423, -16.771724700927734),
    (
        "google/siglip2-base-patch16-384",
        "f775b65a79762255128c981547af89addcfe0f88",
    ): (768, 112.84601055046839, -16.77224349975586),
}
SIGLIP_ARTIFACT_MANIFEST: dict[
    tuple[str, str],
    tuple[tuple[str, str, int], ...],
] = {
    (
        "google/siglip2-base-patch16-224",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    ): (
        ("config.json", "c8cd2a20e58a738f44f267ae19f9568ff1095698", 253),
        (
            "model.safetensors",
            "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
            1_500_800_904,
        ),
        (
            "preprocessor_config.json",
            "2e52d8e8492b5c496ae04c37bfa09760469fb18b",
            394,
        ),
        (
            "tokenizer.json",
            "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
            34_363_039,
        ),
        (
            "tokenizer_config.json",
            "d97c5412159422c3b56fbc99076b1dcc25dd7856",
            47_164,
        ),
        (
            "tokenizer.model",
            "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
            4_241_003,
        ),
        (
            "special_tokens_map.json",
            "8d6368f7e735fbe4781bf6e956b7c6ad0586df80",
            636,
        ),
    ),
    (
        "google/siglip2-base-patch16-384",
        "f775b65a79762255128c981547af89addcfe0f88",
    ): (
        ("config.json", "a0b7bd2d2dba42cf27078f5597f4701ec30d5ffb", 276),
        (
            "model.safetensors",
            "ed72c0ace85020ae610fc817c2538b9cae5a477b012a50859c60af5b3ad30857",
            1_501_968_264,
        ),
        (
            "preprocessor_config.json",
            "e9e084ab5a0d74573432f1dcf11c1bdd8d9b3655",
            394,
        ),
        (
            "tokenizer.json",
            "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
            34_363_039,
        ),
        (
            "tokenizer_config.json",
            "d97c5412159422c3b56fbc99076b1dcc25dd7856",
            47_164,
        ),
        (
            "tokenizer.model",
            "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
            4_241_003,
        ),
        (
            "special_tokens_map.json",
            "8d6368f7e735fbe4781bf6e956b7c6ad0586df80",
            636,
        ),
    ),
}

_REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_ITEM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROOT_IDENTITY_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HF_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_HF_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_RELATIVE_PATH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
_LABEL_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,120}$")
_DETECTOR_IDS = frozenset(
    {"rfdetr-nano", "rfdetr-small", "rfdetr-medium", "rfdetr-large"}
)


def validate_relative_image_path(value: str) -> str:
    if (
        "\\" in value
        or "\x00" in value
        or _SAFE_RELATIVE_PATH_RE.fullmatch(value) is None
    ):
        raise ValueError("relative_path must be a safe POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("relative_path must stay inside the worker input root")
    return path.as_posix()


def worker_input_root_identity_from_metadata(metadata: os.stat_result) -> str:
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("Vision worker input root must be a directory")
    payload = {
        "device": metadata.st_dev,
        "group": metadata.st_gid,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "owner": metadata.st_uid,
        "protocol": "videoscope-worker-input-root@1",
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{sha256(canonical).hexdigest()}"


def worker_input_root_identity(path: Path) -> str:
    """Return a path-free identity for one no-follow directory capability."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise ValueError("Vision worker requires no-follow directory access")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | directory | nofollow
    absolute = Path(os.path.abspath(path))
    descriptor: int | None = None
    try:
        descriptor = os.open(absolute.anchor, flags)
        for component in absolute.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return worker_input_root_identity_from_metadata(os.fstat(descriptor))
    except OSError as error:
        raise ValueError("Vision worker input root must be a safe directory") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class VisionWorkerSpecification:
    """Every model and preprocessing input that can change worker output."""

    siglip_model: str
    siglip_revision: str
    embedding_dimensions: int
    detector_model_id: str
    detector_checkpoint_sha256: str
    siglip_logit_scale: float
    siglip_logit_bias: float
    minimum_confidence: float
    runtime_identity: str = VISION_WORKER_RUNTIME_IDENTITY
    siglip_preprocessing_revision: str = SIGLIP_PREPROCESSING_REVISION
    siglip_tokenizer_revision: str = SIGLIP_TOKENIZER_REVISION
    detector_adapter_revision: str = RFDETR_ADAPTER_REVISION

    def __post_init__(self) -> None:
        if _HF_REPOSITORY_RE.fullmatch(self.siglip_model) is None:
            raise ValueError("SigLIP model must be an exact Hugging Face repository ID")
        if _HF_REVISION_RE.fullmatch(self.siglip_revision) is None:
            raise ValueError("SigLIP revision must be a lowercase 40-character commit")
        if (
            type(self.embedding_dimensions) is not int
            or not 1 <= self.embedding_dimensions <= MAX_EMBEDDING_DIMENSIONS
        ):
            raise ValueError("SigLIP embedding dimensions are invalid")
        if self.detector_model_id not in _DETECTOR_IDS:
            raise ValueError("unsupported local RF-DETR model")
        if _SHA256_RE.fullmatch(self.detector_checkpoint_sha256) is None:
            raise ValueError("RF-DETR checkpoint SHA-256 is invalid")
        if (
            isinstance(self.siglip_logit_scale, bool)
            or not math.isfinite(float(self.siglip_logit_scale))
            or not 0 < float(self.siglip_logit_scale) <= 1_000
            or isinstance(self.siglip_logit_bias, bool)
            or not math.isfinite(float(self.siglip_logit_bias))
            or abs(float(self.siglip_logit_bias)) > 1_000
        ):
            raise ValueError("SigLIP calibration values are invalid")
        if (
            isinstance(self.minimum_confidence, bool)
            or not math.isfinite(float(self.minimum_confidence))
            or not 0 <= float(self.minimum_confidence) <= 1
        ):
            raise ValueError("RF-DETR confidence threshold is invalid")
        if self.runtime_identity != VISION_WORKER_RUNTIME_IDENTITY:
            raise ValueError("vision worker runtime identity is not supported")
        if self.siglip_preprocessing_revision != SIGLIP_PREPROCESSING_REVISION:
            raise ValueError("SigLIP preprocessing revision is not implemented")
        if self.siglip_tokenizer_revision != SIGLIP_TOKENIZER_REVISION:
            raise ValueError("SigLIP tokenizer revision is not implemented")
        if self.detector_adapter_revision != RFDETR_ADAPTER_REVISION:
            raise ValueError("RF-DETR adapter revision is not implemented")

    @property
    def siglip_model_identity(self) -> str:
        return f"{self.siglip_model}@{self.siglip_revision}"

    @property
    def detector_model_identity(self) -> str:
        return (
            f"roboflow/{self.detector_model_id}"
            f"@sha256:{self.detector_checkpoint_sha256}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "detector": self.detector_projection(),
            "siglip": self.siglip_projection(),
            "worker_contract": VISION_WORKER_SCHEMA_VERSION,
        }

    def siglip_projection(self) -> dict[str, object]:
        artifacts = SIGLIP_ARTIFACT_MANIFEST.get(
            (self.siglip_model, self.siglip_revision),
            (),
        )
        return {
            "artifact_manifest": [
                {"filename": filename, "oid": oid, "size_bytes": size_bytes}
                for filename, oid, size_bytes in artifacts
            ],
            "compute_backend": SIGLIP_COMPUTE_BACKEND,
            "compute_dtype": SIGLIP_COMPUTE_DTYPE,
            "embedding_dimensions": self.embedding_dimensions,
            "logit_bias": float(self.siglip_logit_bias),
            "logit_scale": float(self.siglip_logit_scale),
            "max_text_tokens": SIGLIP_MAX_TEXT_TOKENS,
            "model": self.siglip_model,
            "preprocessing_revision": self.siglip_preprocessing_revision,
            "revision": self.siglip_revision,
            "runtime_identity": self.runtime_identity,
            "tokenizer_revision": self.siglip_tokenizer_revision,
            "worker_contract": VISION_WORKER_SCHEMA_VERSION,
        }

    def detector_projection(self) -> dict[str, object]:
        return {
            "adapter_revision": self.detector_adapter_revision,
            "checkpoint_sha256": self.detector_checkpoint_sha256,
            "compute_backend": RFDETR_COMPUTE_BACKEND,
            "compute_dtype": RFDETR_COMPUTE_DTYPE,
            "minimum_confidence": float(self.minimum_confidence),
            "model_id": self.detector_model_id,
            "runtime_identity": self.runtime_identity,
            "worker_contract": VISION_WORKER_SCHEMA_VERSION,
        }

    @staticmethod
    def _projection_identity(projection: dict[str, object]) -> str:
        canonical = json.dumps(
            projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def siglip_identity(self) -> str:
        return self._projection_identity(self.siglip_projection())

    @property
    def detector_identity(self) -> str:
        return self._projection_identity(self.detector_projection())

    @property
    def identity(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(canonical.encode("utf-8")).hexdigest()


class _ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class VisionIdentity(_ContractModel):
    schema_version: Literal[VISION_WORKER_SCHEMA_VERSION]
    runtime_identity: Literal[VISION_WORKER_RUNTIME_IDENTITY]
    specification_hash: str = Field(pattern=_SHA256_RE.pattern)
    siglip_specification_hash: str = Field(pattern=_SHA256_RE.pattern)
    detector_specification_hash: str = Field(pattern=_SHA256_RE.pattern)
    siglip_model_identity: str = Field(min_length=1, max_length=200)
    detector_model_identity: str = Field(min_length=1, max_length=200)
    input_root_identity: str = Field(pattern=_ROOT_IDENTITY_RE.pattern)


class VisionRequestIdentity(VisionIdentity):
    request_id: str = Field(pattern=_REQUEST_ID_RE.pattern)


class VisionImageItem(_ContractModel):
    item_id: str = Field(pattern=_ITEM_ID_RE.pattern)
    relative_path: str = Field(min_length=1, max_length=240)
    expected_sha256: str = Field(pattern=_SHA256_RE.pattern)
    expected_size_bytes: int = Field(gt=0, le=MAX_IMAGE_BYTES)

    @field_validator("relative_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_relative_image_path(value)


class VisionTextItem(_ContractModel):
    item_id: str = Field(pattern=_ITEM_ID_RE.pattern)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class VisionEmbedImagesRequest(VisionRequestIdentity):
    items: list[VisionImageItem] = Field(min_length=1, max_length=MAX_IMAGES)

    @field_validator("items")
    @classmethod
    def validate_unique_items(
        cls,
        value: list[VisionImageItem],
    ) -> list[VisionImageItem]:
        if len({item.item_id for item in value}) != len(value):
            raise ValueError("image item IDs must be unique")
        if len({item.relative_path for item in value}) != len(value):
            raise ValueError("image paths must be unique")
        return value


class VisionEmbedTextsRequest(VisionRequestIdentity):
    items: list[VisionTextItem] = Field(min_length=1, max_length=MAX_TEXTS)

    @field_validator("items")
    @classmethod
    def validate_unique_items(
        cls,
        value: list[VisionTextItem],
    ) -> list[VisionTextItem]:
        if len({item.item_id for item in value}) != len(value):
            raise ValueError("text item IDs must be unique")
        return value


class VisionDetectRequest(VisionRequestIdentity):
    source: VisionImageItem
    minimum_confidence: float = Field(ge=0, le=1)


class VisionReleaseDetectorRequest(VisionRequestIdentity):
    pass


class VisionSourceProbeRequest(VisionRequestIdentity):
    relative_path: str = Field(min_length=1, max_length=240)
    expected_sha256: str = Field(pattern=_SHA256_RE.pattern)
    expected_size_bytes: int = Field(gt=0, le=MAX_PROBE_BYTES)

    @field_validator("relative_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_relative_image_path(value)


class VisionSourceProbeResponse(VisionRequestIdentity):
    source_sha256: str = Field(pattern=_SHA256_RE.pattern)
    source_size_bytes: int = Field(gt=0, le=MAX_PROBE_BYTES)


class VisionVectorItem(_ContractModel):
    item_id: str = Field(pattern=_ITEM_ID_RE.pattern)
    vector: list[float] = Field(min_length=1, max_length=MAX_EMBEDDING_DIMENSIONS)


class VisionEmbeddingResponse(VisionRequestIdentity):
    embedding_dimensions: int = Field(ge=1, le=MAX_EMBEDDING_DIMENSIONS)
    items: list[VisionVectorItem] = Field(min_length=1, max_length=MAX_TEXTS)

    @model_validator(mode="after")
    def validate_vectors(self) -> Self:
        if len({item.item_id for item in self.items}) != len(self.items):
            raise ValueError("embedding response item IDs must be unique")
        if any(len(item.vector) != self.embedding_dimensions for item in self.items):
            raise ValueError("embedding vector dimensions do not match")
        for item in self.items:
            squared_norm = sum(value * value for value in item.vector)
            if not math.isclose(squared_norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError("embedding vectors must be L2-normalized")
        return self


class VisionDetectionPayload(_ContractModel):
    label: str = Field(min_length=1, max_length=120)
    confidence: float = Field(ge=0, le=1)
    x: float = Field(ge=0, le=MAX_IMAGE_DIMENSION)
    y: float = Field(ge=0, le=MAX_IMAGE_DIMENSION)
    width: float = Field(gt=0, le=MAX_IMAGE_DIMENSION)
    height: float = Field(gt=0, le=MAX_IMAGE_DIMENSION)

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if _LABEL_RE.fullmatch(value) is None:
            raise ValueError("detection label contains control characters")
        return value


class VisionDetectionResponse(VisionRequestIdentity):
    source_id: str = Field(pattern=_ITEM_ID_RE.pattern)
    image_width: int = Field(gt=0, le=MAX_IMAGE_DIMENSION)
    image_height: int = Field(gt=0, le=MAX_IMAGE_DIMENSION)
    detections: list[VisionDetectionPayload] = Field(max_length=MAX_DETECTIONS)

    @model_validator(mode="after")
    def validate_boxes(self) -> Self:
        for detection in self.detections:
            if (
                detection.x - detection.width / 2 < 0
                or detection.x + detection.width / 2 > self.image_width
                or detection.y - detection.height / 2 < 0
                or detection.y + detection.height / 2 > self.image_height
            ):
                raise ValueError("detection box is outside the source image")
        return self


class VisionReleaseDetectorResponse(VisionRequestIdentity):
    released: Literal[True]
    siglip_loaded: bool
    detector_loaded: Literal[False]


class VisionHealthResponse(VisionIdentity):
    status: Literal["ok", "unavailable"]
    siglip_loaded: bool
    detector_loaded: bool
    operations: list[
        Literal[
            "probe",
            "embed_images",
            "embed_texts",
            "detect",
            "release_detector",
        ]
    ] = Field(min_length=5, max_length=5)
    embedding_dimensions: int = Field(ge=1, le=MAX_EMBEDDING_DIMENSIONS)
    max_images: int = Field(ge=1, le=MAX_IMAGES)
    max_texts: int = Field(ge=1, le=MAX_TEXTS)
    max_image_bytes: int = Field(gt=0, le=MAX_IMAGE_BYTES)
    max_image_dimension: int = Field(gt=0, le=MAX_IMAGE_DIMENSION)
    max_image_pixels: int = Field(gt=0, le=MAX_IMAGE_PIXELS)
    max_batch_image_bytes: int = Field(gt=0, le=MAX_BATCH_IMAGE_BYTES)
    max_batch_image_pixels: int = Field(gt=0, le=MAX_BATCH_IMAGE_PIXELS)
    max_detections: int = Field(ge=1, le=MAX_DETECTIONS)
    max_concurrency: int = Field(ge=1, le=4)

    @field_validator("operations")
    @classmethod
    def validate_operations(
        cls,
        value: list[
            Literal[
                "probe",
                "embed_images",
                "embed_texts",
                "detect",
                "release_detector",
            ]
        ],
    ) -> list[
        Literal[
            "probe",
            "embed_images",
            "embed_texts",
            "detect",
            "release_detector",
        ]
    ]:
        if value != [
            "probe",
            "embed_images",
            "embed_texts",
            "detect",
            "release_detector",
        ]:
            raise ValueError("vision worker operations do not match the contract")
        return value


def identity_fields(
    specification: VisionWorkerSpecification,
    *,
    input_root_identity: str,
) -> dict[str, str]:
    if _ROOT_IDENTITY_RE.fullmatch(input_root_identity) is None:
        raise ValueError("Vision worker input root identity is invalid")
    return {
        "schema_version": VISION_WORKER_SCHEMA_VERSION,
        "runtime_identity": specification.runtime_identity,
        "specification_hash": specification.identity,
        "siglip_specification_hash": specification.siglip_identity,
        "detector_specification_hash": specification.detector_identity,
        "siglip_model_identity": specification.siglip_model_identity,
        "detector_model_identity": specification.detector_model_identity,
        "input_root_identity": input_root_identity,
    }


def siglip_profile_is_reviewed(specification: VisionWorkerSpecification) -> bool:
    reviewed = REVIEWED_SIGLIP_PROFILES.get(
        (specification.siglip_model, specification.siglip_revision)
    )
    if reviewed is None:
        return False
    dimensions, logit_scale, logit_bias = reviewed
    return (
        specification.embedding_dimensions == dimensions
        and math.isclose(
            float(specification.siglip_logit_scale),
            logit_scale,
            rel_tol=0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(specification.siglip_logit_bias),
            logit_bias,
            rel_tol=0,
            abs_tol=1e-12,
        )
    )
