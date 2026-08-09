import re

from videoscope.model_manifest import (
    FASTEMBED_REPOSITORY,
    MODEL_REVISIONS,
    TEXT_EMBEDDING_MODEL,
    fastembed_snapshot,
    model_identity,
    model_revision,
)


def test_model_manifest_uses_immutable_hugging_face_revisions() -> None:
    assert MODEL_REVISIONS
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in MODEL_REVISIONS.values())
    assert all(model_revision(model) == revision for model, revision in MODEL_REVISIONS.items())
    assert model_revision("custom/local-model") is None


def test_model_identity_includes_the_immutable_revision() -> None:
    revision = "a" * 40

    assert model_identity("organization/model", revision) == f"organization/model@{revision}"
    assert model_identity("./local-model", None) == "./local-model"


def test_fastembed_runtime_uses_the_downloaded_pinned_snapshot() -> None:
    repository, revision = fastembed_snapshot(TEXT_EMBEDDING_MODEL)

    assert repository == FASTEMBED_REPOSITORY
    assert revision == MODEL_REVISIONS[FASTEMBED_REPOSITORY]
    assert fastembed_snapshot("custom/model") == (None, None)
