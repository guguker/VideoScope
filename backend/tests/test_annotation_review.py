import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videoscope.annotation_review.schema import BatchManifest, batch_revision
from videoscope.annotation_review.server import create_app
from videoscope.annotation_review import server


ORIGIN = "http://127.0.0.1:8766"


@pytest.fixture
def batch(tmp_path):
    root = tmp_path / "batch"
    (root / "clips").mkdir(parents=True)
    (root / "posters").mkdir()
    clip = b"synthetic-test-only-media"
    (root / "clips/e01.mp4").write_bytes(clip)
    (root / "posters/e01.jpg").write_bytes(b"synthetic-poster")
    manifest = {
        "schema_version": 1, "batch_id": "uba-pilot-01", "title": "Проверка UBA",
        "created_at": "2026-09-11T12:00:00Z", "code_sha": "a" * 40,
        "purpose": "annotation_pilot", "training_allowed": False,
        "promotion_allowed": False,
        "sources": [{"source_id": "uba-01", "sha256": "b" * 64,
                     "byte_size": 100000, "duration_seconds": 7200.0,
                     "source_group": "game-01", "usage": "development_review",
                     "review_allowed": True, "training_rights": "unknown"}],
        "examples": [{"example_id": "e01", "source_id": "uba-01",
                      "source_start_seconds": 50.0, "source_end_seconds": 62.0,
                      "clip_duration_seconds": 12.0,
                      "prepared_input_sha256": hashlib.sha256(clip).hexdigest(),
                      "prepared_input_byte_size": len(clip),
                      "clip_path": "clips/e01.mp4", "poster_path": "posters/e01.jpg",
                      "selection_method": "uniform_timeline",
                      "selection_notes": "Unlabelled pilot candidate"}],
    }
    write_manifest(root, manifest)
    return root, manifest


def write_manifest(root, manifest):
    (root / "batch.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


def client_for(root):
    return TestClient(create_app(root, port=8766), base_url=ORIGIN)


def annotation(manifest, **changes):
    result = {
        "batch_revision": batch_revision(manifest), "example_id": "e01",
        "expected_revision": 0, "shot_type": "three", "outcome": "made",
        "presentation": "live", "boundary_status": "complete",
        "start_seconds": 2.0, "end_seconds": 10.0, "notes": "Видно попадание",
    }
    result.update(changes)
    return result


def save(client, payload):
    return client.post("/api/annotations", json=payload, headers={"origin": ORIGIN})


def test_batch_revision_uses_canonical_input_without_newline(batch):
    _, manifest = batch
    raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert batch_revision(manifest) == hashlib.sha256(raw).hexdigest()
    assert batch_revision(dict(reversed(list(manifest.items())))) == batch_revision(manifest)


@pytest.mark.parametrize("field,value", [
    ("training_allowed", True), ("promotion_allowed", True),
    ("schema_version", 2), ("unexpected", "field"), ("batch_id", "../secret"),
])
def test_manifest_rejects_unknown_or_unsafe_batch_contract(batch, field, value):
    _, manifest = batch
    manifest[field] = value
    with pytest.raises(ValueError):
        BatchManifest.model_validate(manifest)


@pytest.mark.parametrize("field,value", [
    ("source_end_seconds", float("inf")), ("source_start_seconds", -1.0),
    ("source_end_seconds", 49.0), ("clip_duration_seconds", 61.0),
    ("source_id", "unknown"), ("prepared_input_byte_size", 0),
    ("clip_path", "../outside.mp4"), ("poster_path", "/outside.jpg"),
    ("source_start_seconds", True),
])
def test_manifest_rejects_invalid_examples(batch, field, value):
    _, manifest = batch
    manifest["examples"][0][field] = value
    with pytest.raises(ValueError):
        BatchManifest.model_validate(manifest)


@pytest.mark.parametrize("field,value", [
    ("usage", "promotion_holdout"), ("review_allowed", False),
    ("training_rights", "automatically_allowed"), ("duration_seconds", 40.0),
])
def test_manifest_rejects_holdout_and_unknown_source_scope(batch, field, value):
    _, manifest = batch
    manifest["sources"][0][field] = value
    with pytest.raises(ValueError):
        BatchManifest.model_validate(manifest)


def test_manifest_rejects_duplicates_and_overlarge_batch(batch):
    _, manifest = batch
    duplicate = copy.deepcopy(manifest)
    duplicate["examples"] *= 2
    with pytest.raises(ValueError):
        BatchManifest.model_validate(duplicate)
    duplicate = copy.deepcopy(manifest)
    duplicate["sources"] *= 2
    with pytest.raises(ValueError):
        BatchManifest.model_validate(duplicate)
    manifest["examples"] *= 101
    with pytest.raises(ValueError):
        BatchManifest.model_validate(manifest)


def test_review_only_exposes_aliases_and_attested_prepared_media(batch):
    root, manifest = batch
    client = client_for(root)
    review = client.get("/api/review")
    assert review.status_code == 200
    result = review.json()
    assert result["batch_revision"] == batch_revision(manifest)
    assert result["annotations"] == {}
    assert result["examples"][0]["source_alias"] == "uba-01"
    assert str(root) not in review.text
    assert "selection_notes" not in review.text
    assert "sha256" not in review.text
    media = client.get("/media/e01", headers={"Range": "bytes=0-8"})
    assert media.status_code == 206
    assert media.content == b"synthetic"
    assert client.get("/posters/e01").content == b"synthetic-poster"
    assert client.get("/media/missing").status_code == 404
    assert client.get("/api/health").json()["status"] == "ready"
    assert not (root / "annotations").exists()


def test_human_review_roundtrip_durable_revision_history_and_export(batch):
    root, manifest = batch
    original = (root / "batch.json").read_bytes()
    client = client_for(root)
    saved = save(client, annotation(manifest))
    assert saved.status_code == 200
    assert saved.json()["annotation"]["revision"] == 1
    assert saved.json()["reviewed_count"] == 1
    assert saved.json()["total_count"] == 1
    assert save(client, annotation(manifest)).status_code == 409
    second = annotation(manifest, expected_revision=1, shot_type="unclear", outcome="unclear",
                        start_seconds=None, end_seconds=None, boundary_status="unclear")
    assert save(client, second).json()["annotation"]["revision"] == 2
    assert save(client, {**second, "expected_revision": 2}).json()["reviewed_count"] == 0
    restarted = client_for(root)
    assert restarted.get("/api/review").json()["annotations"]["e01"]["shot_type"] == "unclear"
    exported = restarted.get("/api/export")
    assert exported.status_code == 200
    assert "attachment" in exported.headers["content-disposition"]
    data = exported.json()
    assert data["batch_revision"] == batch_revision(manifest)
    assert len(data["records"]) == 3
    for record in data["records"]:
        assert record["label_status"] == ("human_reviewed" if record["revision"] == 1 else "draft")
        assert record["destination"] == "annotation_inbox"
        assert record["gold"] is False
        assert record["source_sha256"] == "b" * 64
        assert record["prepared_input_sha256"] == manifest["examples"][0]["prepared_input_sha256"]
    assert (root / "batch.json").read_bytes() == original


@pytest.mark.parametrize("changes", [
    {"batch_revision": "c" * 64}, {"example_id": "unknown"}, {"expected_revision": -1},
    {"start_seconds": -1.0}, {"end_seconds": 13.0}, {"start_seconds": 11.0},
    {"shot_type": "made_three"}, {"gold": True}, {"notes": "x" * 2001},
    {"start_seconds": None}, {"expected_revision": True},
])
def test_annotation_rejects_invalid_cross_batch_or_unsafe_fields(batch, changes):
    root, manifest = batch
    response = save(client_for(root), annotation(manifest, **changes))
    assert response.status_code in {404, 409, 422}
    assert not list(root.glob("annotations/*/*.json"))


@pytest.mark.parametrize("headers", [
    {}, {"origin": "https://evil.example"}, {"origin": "null"},
    {"origin": "http://localhost:8766"}, {"origin": "http://127.0.0.1:8767"},
])
def test_write_rejects_missing_and_cross_origin(batch, headers):
    root, manifest = batch
    assert client_for(root).post("/api/annotations", json=annotation(manifest), headers=headers).status_code == 403


def test_security_headers_host_json_body_bound_and_static_allowlist(batch):
    root, manifest = batch
    client = client_for(root)
    assert client.get("/api/review", headers={"host": "evil.example"}).status_code == 400
    response = client.get("/api/review")
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert client.get("/batch.json").status_code == 404
    assert client.get("/annotations/.lock").status_code == 404
    assert client.get("/docs").status_code == 404
    assert client.post("/api/annotations", content=json.dumps(annotation(manifest)),
                       headers={"origin": ORIGIN, "content-type": "text/plain"}).status_code == 415
    assert client.post("/api/annotations", content="x" * 20000,
                       headers={"origin": ORIGIN, "content-type": "application/json", "content-length": "1"}).status_code == 413


@pytest.mark.parametrize("target", ["clips/e01.mp4", "posters/e01.jpg", "batch.json"])
def test_rejects_symlink_assets_at_startup(batch, tmp_path, target):
    root, _ = batch
    original = root / target
    external = tmp_path / "external"
    original.rename(external)
    original.symlink_to(external)
    with pytest.raises(ValueError):
        create_app(root)


def test_startup_rejects_hash_mismatch_and_replaced_media_fails_closed(batch):
    root, _ = batch
    client = client_for(root)
    (root / "clips/e01.mp4").write_bytes(b"tampered")
    assert client.get("/media/e01").status_code == 409
    with pytest.raises(ValueError):
        create_app(root)


def test_manifest_mutation_invalidates_existing_server(batch):
    root, manifest = batch
    client = client_for(root)
    manifest["title"] = "Changed"
    write_manifest(root, manifest)
    assert client.get("/api/review").status_code == 409


def test_corrupt_history_rejected_on_restart(batch):
    root, manifest = batch
    client = client_for(root)
    assert save(client, annotation(manifest)).status_code == 200
    record = next(root.glob("annotations/*/*.json"))
    data = json.loads(record.read_text())
    data["batch_revision"] = "d" * 64
    record.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        create_app(root)


def test_concurrent_servers_cannot_overwrite_same_revision(batch):
    root, manifest = batch
    clients = [client_for(root), client_for(root)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda c: save(c, annotation(manifest)), clients))
    assert sorted(response.status_code for response in responses) == [200, 409]
    assert len(list(root.glob("annotations/*/*.json"))) == 1


def test_annotation_symlink_directory_is_not_followed(batch, tmp_path):
    root, manifest = batch
    client = client_for(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "annotations").symlink_to(outside, target_is_directory=True)
    assert save(client, annotation(manifest)).status_code == 409
    assert list(outside.iterdir()) == []


def test_forced_publish_failure_retains_previous_review_and_allows_retry(batch, monkeypatch):
    root, manifest = batch
    client = client_for(root)
    assert save(client, annotation(manifest)).status_code == 200
    first_path = next(root.glob("annotations/*/*.json"))
    first_bytes = first_path.read_bytes()
    link = server.os.link

    def fail_link(*args, **kwargs):
        raise OSError("forced disk publication failure")

    monkeypatch.setattr(server.os, "link", fail_link)
    second = annotation(manifest, expected_revision=1, outcome="miss")
    failed = save(client, second)
    assert failed.status_code == 409
    assert "forced disk" not in failed.text
    assert first_path.read_bytes() == first_bytes
    assert len(list(root.glob("annotations/*/*.json"))) == 1
    assert not list(root.glob("annotations/*/.pending-*"))
    restarted = client_for(root)
    assert restarted.get("/api/review").json()["annotations"]["e01"]["revision"] == 1
    monkeypatch.setattr(server.os, "link", link)
    assert save(restarted, second).json()["annotation"]["revision"] == 2


def test_null_answers_are_preserved_without_invented_labels(batch):
    root, manifest = batch
    payload = annotation(manifest, shot_type=None, outcome=None, presentation=None,
                         boundary_status=None, start_seconds=None, end_seconds=None, notes="")
    response = save(client_for(root), payload)
    assert response.status_code == 200
    assert response.json()["annotation"]["shot_type"] is None
    assert response.json()["reviewed_count"] == 0
    client = client_for(root)
    assert client.get("/api/export").json()["records"][0]["label_status"] == "draft"
    completed = save(client, annotation(manifest, expected_revision=1, shot_type="unclear",
        outcome="unclear", presentation="unclear", boundary_status="unclear"))
    assert completed.json()["reviewed_count"] == 1
    assert client.get("/api/export").json()["records"][1]["label_status"] == "human_reviewed"


def test_four_answers_without_boundaries_remain_draft(batch):
    root, manifest = batch
    client = client_for(root)
    response = save(client, annotation(manifest, start_seconds=None, end_seconds=None))
    assert response.json()["reviewed_count"] == 0
    assert client.get("/api/export").json()["records"][0]["label_status"] == "draft"


@pytest.mark.parametrize("content", [
    '{"example_id":"e01","example_id":"different"}',
    '{"start_seconds":NaN}', '{"start_seconds":Infinity}', "broken json",
])
def test_duplicate_keys_and_nonfinite_json_rejected_before_validation(batch, content):
    root, _ = batch
    response = client_for(root).post("/api/annotations", content=content,
        headers={"origin": ORIGIN, "content-type": "application/json"})
    assert response.status_code == 422


@pytest.mark.parametrize("corruption", ["unknown_example", "gap", "invalid_schema", "symlink_lock"])
def test_history_corruption_fails_closed(batch, tmp_path, corruption):
    root, manifest = batch
    client = client_for(root)
    assert save(client, annotation(manifest)).status_code == 200
    record = next(root.glob("annotations/*/*.json"))
    if corruption == "unknown_example":
        (root / "annotations/unknown").mkdir()
    elif corruption == "gap":
        record.rename(record.with_name("000002.json"))
    elif corruption == "invalid_schema":
        record.write_text('{"invalid":true}')
    else:
        lock = root / "annotations/.lock"
        lock.unlink()
        lock.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError):
        create_app(root)


def test_orphaned_pending_file_is_inert_after_crash(batch):
    root, manifest = batch
    client = client_for(root)
    assert save(client, annotation(manifest)).status_code == 200
    pending = root / "annotations/e01/.pending-crash"
    pending.write_bytes(b"incomplete annotation")
    restarted = client_for(root)
    assert len(restarted.get("/api/export").json()["records"]) == 1
    assert save(restarted, annotation(manifest, expected_revision=1)).status_code == 200
    assert pending.read_bytes() == b"incomplete annotation"


def test_static_resources_are_local_allowlisted_and_receive_csp(batch, tmp_path, monkeypatch):
    root, _ = batch
    web = tmp_path / "web"
    web.mkdir()
    for filename in ["index.html", "app.js", "style.css"]:
        (web / filename).write_text("synthetic asset")
    monkeypatch.setattr(server, "WEB_ROOT", web)
    client = client_for(root)
    for resource in ["/", "/index.html", "/app.js", "/style.css"]:
        response = client.get(resource)
        assert response.status_code == 200
        assert response.text == "synthetic asset"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_cli_binds_only_loopback_and_rejects_invalid_port(batch, monkeypatch):
    root, _ = batch
    import uvicorn

    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.update(kwargs))
    server.main(["--batch", str(root), "--port", "8766"])
    assert captured["host"] == "127.0.0.1"
    assert captured["access_log"] is False
    with pytest.raises(ValueError):
        create_app(root, port=80)
