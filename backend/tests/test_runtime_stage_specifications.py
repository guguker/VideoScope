from __future__ import annotations

from videoscope.artifacts import StageKind
from videoscope.config import AppSettings
from videoscope.model_manifest import SIGLIP_384_MODEL
from videoscope.runtime import create_indexing_specifications
from videoscope.providers.vision_worker_contract import VISION_WORKER_SCHEMA_VERSION
from videoscope.providers.whisper_worker import (
    WHISPER_DEPENDENCY_IDENTITY,
    WHISPER_WORKER_SCHEMA_VERSION,
)
import videoscope.runtime as runtime_module


def test_runtime_builds_explicit_complete_stage_specifications(tmp_path) -> None:
    secret = "a" * 40
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        roboflow_api_key=secret,
        whisper_language="ru",
    )

    specifications = create_indexing_specifications(settings)

    assert [item.kind for item in specifications.segment_specifications] == [
        StageKind.SCENES,
        StageKind.SPEECH,
        StageKind.OCR,
        StageKind.OBJECTS,
    ]
    assert specifications.text_vectors.kind is StageKind.TEXT_VECTORS
    for specification in (*specifications.segment_specifications, specifications.text_vectors):
        assert specification.schema_version >= 1
        assert specification.implementation_revision.startswith("videoscope.")
        assert "object at 0x" not in specification.canonical_json
        assert str(tmp_path) not in specification.canonical_json
        assert secret not in specification.canonical_json


def test_scene_configuration_invalidates_scene_sampled_downstream_specs(
    tmp_path,
) -> None:
    baseline = create_indexing_specifications(AppSettings(data_dir=tmp_path / "one"))
    changed = create_indexing_specifications(
        AppSettings(
            data_dir=tmp_path / "two",
            scene_threshold=4.5,
        )
    )

    assert baseline.scenes.specification_hash != changed.scenes.specification_hash
    assert baseline.speech.specification_hash == changed.speech.specification_hash
    assert baseline.ocr.specification_hash != changed.ocr.specification_hash
    assert baseline.objects.specification_hash != changed.objects.specification_hash
    assert baseline.text_vectors.specification_hash != changed.text_vectors.specification_hash


def test_glossary_change_invalidates_only_speech_and_text_vector_contracts(
    tmp_path,
) -> None:
    settings = AppSettings(data_dir=tmp_path / "data")
    settings.ensure_directories()
    settings.glossary_path.write_text('{"Мозгов":["Mozgov"]}', encoding="utf-8")
    baseline = create_indexing_specifications(settings)
    settings.glossary_path.write_text(
        '{"Мозгов":["Mozgov","Мозгова"]}',
        encoding="utf-8",
    )
    changed = create_indexing_specifications(settings)

    assert baseline.scenes.specification_hash == changed.scenes.specification_hash
    assert baseline.speech.specification_hash != changed.speech.specification_hash
    assert baseline.ocr.specification_hash == changed.ocr.specification_hash
    assert baseline.objects.specification_hash == changed.objects.specification_hash
    assert baseline.text_vectors.specification_hash != changed.text_vectors.specification_hash


def test_runtime_specifications_include_pinned_model_and_output_contract_identity(
    tmp_path,
) -> None:
    specifications = create_indexing_specifications(
        AppSettings(data_dir=tmp_path / "data")
    )

    assert "@" in (specifications.speech.model_identity or "")
    assert specifications.ocr.model_identity
    assert specifications.objects.model_identity
    assert "dimensions" in specifications.text_vectors.parameters
    assert "segment_schema" in specifications.scenes.parameters


def test_indexing_run_plan_shares_one_bounded_whisper_prompt_snapshot(
    tmp_path,
) -> None:
    settings = AppSettings(
        data_dir=tmp_path / "data",
        whisper_worker_endpoint="http://127.0.0.1:8784",
        whisper_worker_api_key="w" * 32,
    )
    settings.ensure_directories()
    settings.glossary_path.write_text(
        '{"Мозгов":["Mozgov"]}',
        encoding="utf-8",
    )

    plan = runtime_module.create_indexing_run_plan(settings)
    speech = plan.specifications.speech

    assert speech.parameters["effective_prompt_sha256"] == (
        plan.whisper_prompt_snapshot.effective_prompt_sha256
    )
    assert speech.parameters["glossary_state"] == (
        plan.whisper_prompt_snapshot.glossary_state
    )
    assert plan.whisper_prompt_snapshot.glossary_state == "ready"
    assert "Мозгов" in (plan.whisper_prompt_snapshot.effective_prompt or "")
    assert "Мозгов" not in speech.canonical_json
    assert str(settings.glossary_path) not in speech.canonical_json


def test_worker_stage_specs_bump_contracts_without_persisting_secrets_or_paths(
    tmp_path,
) -> None:
    vision_secret = "v" * 32
    whisper_secret = "w" * 32
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "private-data",
        vision_worker_endpoint="http://127.0.0.1:8783",
        vision_worker_api_key=vision_secret,
        whisper_worker_endpoint="http://127.0.0.1:8784",
        whisper_worker_api_key=whisper_secret,
    )
    settings.ensure_directories()

    specifications = create_indexing_specifications(settings)

    assert specifications.speech.schema_version == 2
    assert specifications.speech.implementation_revision.endswith(".v2")
    assert specifications.objects.schema_version == 2
    assert specifications.objects.implementation_revision.endswith(".v2")
    assert WHISPER_WORKER_SCHEMA_VERSION in specifications.speech.canonical_json
    assert WHISPER_DEPENDENCY_IDENTITY in specifications.speech.canonical_json
    assert VISION_WORKER_SCHEMA_VERSION in specifications.objects.canonical_json
    for specification in (
        specifications.speech,
        specifications.objects,
        specifications.text_vectors,
    ):
        assert vision_secret not in specification.canonical_json
        assert whisper_secret not in specification.canonical_json
        assert "127.0.0.1" not in specification.canonical_json
        assert str(tmp_path) not in specification.canonical_json


def _vision_identities(settings: AppSettings) -> tuple[str, str]:
    settings.ensure_directories()
    client = runtime_module.create_vision_worker_client(settings)
    assert client is not None
    visual = runtime_module.create_visual_index(
        settings,
        inference_client=client,
    )
    objects = create_indexing_specifications(settings).objects
    return visual.specification.identity, objects.specification_hash


def test_siglip_detector_and_endpoint_identity_dimensions_are_isolated(
    tmp_path,
) -> None:
    common = {
        "vision_worker_endpoint": "http://127.0.0.1:8783",
        "vision_worker_api_key": "v" * 32,
    }
    baseline = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "baseline",
        **common,
    )
    detector_changed = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "detector",
        vision_detector_checkpoint_sha256="a" * 64,
        **common,
    )
    siglip_changed = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "siglip",
        siglip_model=SIGLIP_384_MODEL,
        **common,
    )
    endpoint_changed = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "endpoint",
        vision_worker_endpoint="http://127.0.0.1:9783",
        vision_worker_api_key="x" * 32,
    )

    baseline_visual, baseline_objects = _vision_identities(baseline)
    detector_visual, detector_objects = _vision_identities(detector_changed)
    siglip_visual, siglip_objects = _vision_identities(siglip_changed)
    endpoint_visual, endpoint_objects = _vision_identities(endpoint_changed)

    assert detector_visual == baseline_visual
    assert detector_objects != baseline_objects
    assert siglip_visual != baseline_visual
    assert siglip_objects == baseline_objects
    assert endpoint_visual == baseline_visual
    assert endpoint_objects == baseline_objects
