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
        "schema_version": 2,
        "batch_revision": batch_revision(manifest), "example_id": "e01",
        "expected_revision": 0, "shot_type": "three", "outcome": "made",
        "presentation": "live", "boundary_status": "complete",
        "scoring_decision": "counted", "play_context": "in_play",
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


def legacy_record(manifest, **changes):
    """A historical v1 record, constructed independently of the current writer."""
    request = annotation(manifest)
    record = {key: value for key, value in request.items() if key not in {
        "expected_revision", "scoring_decision", "play_context"}}
    record.update({
        "schema_version": 1, "batch_id": manifest["batch_id"], "revision": 1,
        "created_at": "2026-09-11T12:05:00+00:00", "reviewer": "local_owner",
        "label_status": "human_reviewed", "destination": "annotation_inbox",
        "gold": False, "training_allowed": False, "promotion_allowed": False,
        "source_id": "uba-01", "source_sha256": "b" * 64,
        "source_start_seconds": 50.0, "source_end_seconds": 62.0,
        "prepared_input_sha256": manifest["examples"][0]["prepared_input_sha256"],
    })
    record.update(changes)
    return record


def write_legacy_record(root, record):
    directory = root / "annotations/e01"
    directory.mkdir(parents=True)
    path = directory / "000001.json"
    raw = json.dumps(record, ensure_ascii=False, indent=3).encode() + b"\n\n"
    path.write_bytes(raw)
    return path, raw


def test_v1_history_is_readable_needs_review_and_not_rewritten_when_v2_appends(batch):
    root, manifest = batch
    old = legacy_record(manifest)
    path, old_bytes = write_legacy_record(root, old)
    client = client_for(root)
    review = client.get("/api/review").json()
    assert review["annotation_schema_version"] == 3
    assert client.get("/api/health").json()["annotation_schema_version"] == 3
    latest = review["annotations"]["e01"]
    assert latest["schema_version"] == 1
    assert latest["needs_review"] is True
    assert latest["scoring_decision"] is None
    assert latest["play_context"] is None
    assert latest["shot_type"] == "three"
    assert latest["revision"] == 1
    assert client.get("/api/export").json()["records"] == [old]
    saved = save(client, annotation(manifest, expected_revision=1,
        scoring_decision="not_counted", play_context="after_whistle"))
    assert saved.status_code == 200
    assert saved.json()["annotation"]["schema_version"] == 2
    assert saved.json()["annotation"]["needs_review"] is False
    assert saved.json()["reviewed_count"] == 1
    restarted = client_for(root)
    exported = restarted.get("/api/export").json()
    assert exported["schema_version"] == 3
    assert exported["records"][0] == old
    assert exported["records"][0]["label_status"] == "human_reviewed"
    assert exported["records"][1]["schema_version"] == 2
    assert exported["records"][1]["scoring_decision"] == "not_counted"
    assert all(record["gold"] is False for record in exported["records"])
    assert path.read_bytes() == old_bytes
    assert save(restarted, annotation(manifest, expected_revision=1)).status_code == 409


def test_legacy_draft_remains_draft_without_guessed_new_fields(batch):
    root, manifest = batch
    old = legacy_record(manifest, outcome=None, label_status="draft")
    path, old_bytes = write_legacy_record(root, old)
    client = client_for(root)
    assert client.get("/api/export").json()["records"] == [old]
    latest = client.get("/api/review").json()["annotations"]["e01"]
    assert latest["needs_review"] is True
    assert latest["outcome"] is None
    assert latest["scoring_decision"] is None
    assert latest["play_context"] is None
    assert path.read_bytes() == old_bytes


@pytest.mark.parametrize("schema_version", [None, 1, 4, True, "2", 2.0])
def test_stale_form_requests_fail_with_reload_message_and_preserve_history(batch, schema_version):
    root, manifest = batch
    old = legacy_record(manifest)
    path, old_bytes = write_legacy_record(root, old)
    request = annotation(manifest, schema_version=schema_version, expected_revision=1)
    if schema_version is None:
        request.pop("schema_version")
    response = save(client_for(root), request)
    assert response.status_code == 409
    assert "reload" in response.json()["detail"].lower()
    assert path.read_bytes() == old_bytes
    assert len(list(root.glob("annotations/*/*.json"))) == 1


@pytest.mark.parametrize("field", ["scoring_decision", "play_context"])
@pytest.mark.parametrize("omit", [False, True])
def test_missing_new_answers_stay_null_draft_until_explicitly_answered(batch, field, omit):
    root, manifest = batch
    client = client_for(root)
    request = annotation(manifest, **{field: None})
    if omit:
        request.pop(field)
    response = save(client, request)
    assert response.status_code == 200
    assert response.json()["annotation"][field] is None
    assert response.json()["annotation"]["needs_review"] is True
    assert response.json()["reviewed_count"] == 0
    assert client.get("/api/export").json()["records"][0]["label_status"] == "draft"
    response = save(client, annotation(manifest, expected_revision=1, **{field: "unclear"}))
    assert response.json()["reviewed_count"] == 1
    assert response.json()["annotation"]["needs_review"] is False


@pytest.mark.parametrize("outcome,scoring,context", [
    ("made", "not_counted", "after_whistle"),
    ("miss", "not_counted", "foul_on_shot"),
    ("made", "counted", "foul_on_shot"),
    ("miss", "counted", "in_play"),  # Points awarded without a physical make.
    ("unclear", "unclear", "unclear"),
    ("made", "not_counted", "other_dead_ball"),
])
def test_visible_outcome_and_official_points_remain_independent(batch, outcome, scoring, context):
    root, manifest = batch
    client = client_for(root)
    response = save(client, annotation(manifest, outcome=outcome,
        scoring_decision=scoring, play_context=context))
    assert response.status_code == 200
    record = client_for(root).get("/api/export").json()["records"][0]
    assert (record["outcome"], record["scoring_decision"], record["play_context"]) == (
        outcome, scoring, context)
    assert record["schema_version"] == 2


@pytest.mark.parametrize("context", ["after_whistle", "other_dead_ball"])
def test_entirely_new_dead_ball_shot_cannot_have_counted_points(batch, context):
    root, manifest = batch
    response = save(client_for(root), annotation(manifest, play_context=context))
    assert response.status_code == 422
    assert not list(root.glob("annotations/*/*.json"))


@pytest.mark.parametrize("changes", [
    {"scoring_decision": "automatically_counted"}, {"play_context": "no_foul_seen"},
    {"scoring_decision": True}, {"play_context": False},
])
def test_new_annotation_fields_reject_unknown_or_coerced_labels(batch, changes):
    root, manifest = batch
    assert save(client_for(root), annotation(manifest, **changes)).status_code == 422
    assert not list(root.glob("annotations/*/*.json"))


def test_annotation_schema_export_matches_current_request_model():
    from videoscope.annotation_review.schema import AnnotationRequest

    schema_path = Path(__file__).resolve().parents[2] / "docs/benchmarks/phase1/annotation-request.schema.json"
    assert json.loads(schema_path.read_text()) == {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **AnnotationRequest.model_json_schema(),
    }


@pytest.mark.parametrize("version", [0, 3, True, 1.0, "1"])
def test_history_never_guesses_unknown_or_coerced_schema_version(batch, version):
    root, manifest = batch
    path, old_bytes = write_legacy_record(root, legacy_record(manifest, schema_version=version))
    with pytest.raises(ValueError):
        create_app(root)
    assert path.read_bytes() == old_bytes


def test_concurrent_v2_append_to_legacy_history_preserves_old_bytes(batch):
    root, manifest = batch
    path, old_bytes = write_legacy_record(root, legacy_record(manifest))
    clients = [client_for(root), client_for(root)]
    request = annotation(manifest, expected_revision=1)
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda c: save(c, request), clients))
    assert sorted(response.status_code for response in responses) == [200, 409]
    records = client_for(root).get("/api/export").json()["records"]
    assert [record["schema_version"] for record in records] == [1, 2]
    assert [record["revision"] for record in records] == [1, 2]
    assert path.read_bytes() == old_bytes


def event_annotation(manifest, event_id='primary', **changes):
    return annotation(manifest, schema_version=3, event_id=event_id, **changes)


def test_multiple_events_keep_separate_boundaries_history_and_legacy_answer(batch):
    root, manifest = batch
    client = client_for(root)
    assert save(client, annotation(manifest, outcome='miss', scoring_decision='not_applicable',
                                   play_context='foul_on_shot', end_seconds=5.844)).status_code == 200
    original = (root/'annotations/e01/000001.json').read_bytes()
    second = 'event-' + 'a'*32
    payload = event_annotation(manifest, second, start_seconds=5.844, end_seconds=11.0,
                               scoring_decision='not_counted', play_context='after_whistle')
    response = save(client, payload)
    assert response.status_code == 200, response.text
    assert response.json()['annotation']['event_id'] == second
    restarted = client_for(root)
    events = restarted.get('/api/review').json()['events']['e01']
    assert [item['event_id'] for item in events] == ['primary', second]
    assert [(item['start_seconds'], item['end_seconds']) for item in events] == [(2.0, 5.844), (5.844, 11.0)]
    assert save(restarted, {**payload, 'expected_revision':1, 'start_seconds':6.0}).status_code == 200
    assert (root/'annotations/e01/000001.json').read_bytes() == original
    exported = restarted.get('/api/export').json()
    assert exported['schema_version'] == 3
    assert [(item.get('event_id','primary'), item['revision']) for item in exported['records']] == [('primary',1),(second,1),(second,2)]
    assert all(not item['gold'] and not item['training_allowed'] for item in exported['records'])
    # An old open form still saves only the original event, without touching the second.
    assert save(restarted, annotation(manifest, expected_revision=1)).status_code == 200
    assert restarted.get('/api/review').json()['events']['e01'][1]['start_seconds'] == 6.0


def test_two_concurrent_event_creations_are_independent_but_same_event_conflicts(batch):
    root, manifest = batch
    clients = [client_for(root), client_for(root)]
    payloads = [event_annotation(manifest, 'event-'+character*32) for character in ('a','b')]
    with ThreadPoolExecutor(2) as pool:
        responses = list(pool.map(lambda pair: save(*pair), zip(clients,payloads)))
    assert [r.status_code for r in responses] == [200,200]
    with ThreadPoolExecutor(2) as pool:
        responses = list(pool.map(lambda client: save(client,{**payloads[0],'expected_revision':1}),clients))
    assert sorted(r.status_code for r in responses) == [200,409]
    assert len(clients[0].get('/api/review').json()['events']['e01']) == 2


@pytest.mark.parametrize('event_id', ['../outside','events','primary/other','event-'+'g'*32,''])
def test_event_identity_rejects_paths_and_noncanonical_names(batch,event_id):
    root,manifest=batch
    assert save(client_for(root),event_annotation(manifest,event_id)).status_code == 422


def test_failed_second_event_publish_keeps_first_and_retries_without_overwrite(batch,monkeypatch):
    root,manifest=batch
    client=client_for(root)
    assert save(client,annotation(manifest)).status_code==200
    before=(root/'annotations/e01/000001.json').read_bytes()
    link=server.os.link
    def fail(*args,**kwargs):raise OSError('injected publication failure')
    monkeypatch.setattr(server.os,'link',fail)
    payload=event_annotation(manifest,'event-'+'c'*32)
    assert save(client,payload).status_code==409
    assert (root/'annotations/e01/000001.json').read_bytes()==before
    monkeypatch.setattr(server.os,'link',link)
    restarted=client_for(root)
    assert save(restarted,payload).status_code==200
    assert len(restarted.get('/api/review').json()['events']['e01'])==2


def test_extra_event_limit_reserves_primary_and_can_still_edit(batch):
    root,manifest=batch
    client=client_for(root)
    for index in range(31):
        assert save(client,event_annotation(manifest,f'event-{index:032x}')).status_code==200
    assert save(client,event_annotation(manifest,'event-'+'f'*32)).status_code==422
    assert save(client,event_annotation(manifest)).status_code==200
    assert save(client,event_annotation(manifest,'event-'+'0'*32,expected_revision=1)).status_code==200


def test_event_record_cannot_be_moved_into_another_events_history(batch):
    root,manifest=batch
    client=client_for(root)
    first,second='event-'+'a'*32,'event-'+'b'*32
    assert save(client,event_annotation(manifest,first)).status_code==200
    (root/f'annotations/e01/events/{first}').rename(root/f'annotations/e01/events/{second}')
    with pytest.raises(server.ReviewConflict):client_for(root)


def test_event_display_order_survives_restart_and_edits(batch):
    root,manifest=batch
    client=client_for(root)
    first,second='event-'+'f'*32,'event-'+'a'*32
    for event_id in (first,second):
        assert save(client,event_annotation(manifest,event_id)).status_code==200
    assert save(client,event_annotation(manifest,first,expected_revision=1)).status_code==200
    assert [event['event_id'] for event in client_for(root).get('/api/review').json()['events']['e01']] == [first,second]
