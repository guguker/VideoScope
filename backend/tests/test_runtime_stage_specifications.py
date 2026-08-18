from __future__ import annotations

from videoscope.artifacts import StageKind
from videoscope.config import AppSettings
from videoscope.runtime import create_indexing_specifications


def test_runtime_builds_explicit_complete_stage_specifications(tmp_path) -> None:
    secret = "a" * 40
    settings = AppSettings(
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
