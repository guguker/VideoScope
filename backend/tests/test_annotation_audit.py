from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from videoscope.annotation_review.audit import AuditError, run_audit


@pytest.fixture
def fake_probe(tmp_path: Path) -> Path:
    executable = tmp_path / 'ffprobe'
    executable.write_text('''#!/usr/bin/env python3
import json, sys, time
from pathlib import Path
if '-version' in sys.argv:
    print('ffprobe fixture 1.0')
    raise SystemExit(0)
source = Path(sys.argv[-1])
if source.name.startswith('fail'):
    print('private path error', file=sys.stderr)
    raise SystemExit(2)
if source.name.startswith('slow'):
    time.sleep(2)
if source.name.startswith('mutate'):
    source.write_bytes(source.read_bytes() + b'changed')
if source.name.startswith('malformed'):
    print('{bad json')
    raise SystemExit(0)
if source.name.startswith('nan'):
    duration = 'nan'
else:
    duration = '12.5'
print(json.dumps({'format': {'duration': duration, 'format_name': 'mov,mp4'},
    'streams': [{'index': 0, 'codec_type': 'video', 'codec_name': 'h264',
    'width': 1920, 'height': 1080, 'avg_frame_rate': '60/1'},
    {'index': 1, 'codec_type': 'audio', 'codec_name': 'aac', 'sample_rate': '44100',
    'channels': 2}]}))
''')
    executable.chmod(0o700)
    return executable


def sources(tmp_path: Path, names: tuple[str, ...] = ('Match [abcdefghijk].mp4',)) -> Path:
    source_dir = tmp_path / 'sources'
    source_dir.mkdir()
    for name in names:
        (source_dir / name).write_bytes(b'synthetic source fixture')
    return source_dir


def test_metadata_audit_records_identity_duplicates_and_unknown_rights(tmp_path: Path, fake_probe: Path):
    source_dir = sources(tmp_path, ('Match [abcdefghijk].mp4', 'different-name.mp4'))
    before = {p.name: p.stat() for p in source_dir.iterdir()}
    output = tmp_path / 'audit'
    result = run_audit(source_dir, output, ffprobe=fake_probe)
    stored = json.loads((output / 'inventory.json').read_text())
    rights = json.loads((output / 'source-rights-ledger.json').read_text())
    assert result == stored
    assert stored['summary']['file_count'] == 2
    assert stored['summary']['exact_duplicate_group_count'] == 1
    assert stored['summary']['all_sources_stable'] is True
    named = next(row for row in stored['sources'] if row['filename'].startswith('Match'))
    unnamed = next(row for row in stored['sources'] if row['filename'].startswith('different'))
    assert named['source_sha256'] == hashlib.sha256(b'synthetic source fixture').hexdigest()
    assert named['duration_seconds'] == 12.5
    assert named['video_streams'][0]['avg_frame_rate'] == '60/1'
    assert named['origin']['confidence'] == 'inferred_from_filename'
    assert named['origin']['externally_verified'] is False
    assert unnamed['origin']['video_id'] is None
    assert named['readability_check']['full_stream_decode'] == 'not_run'
    assert stored['runtime']['ffprobe']['sha256'] == hashlib.sha256(fake_probe.read_bytes()).hexdigest()
    assert stored['runtime']['audit_implementation_sha256']
    assert all(not row['training_allowed'] and not row['publication_allowed'] for row in rights['sources'])
    assert all(row['rights_status'] == 'unknown' for row in rights['sources'])
    assert stored['scope']['content_decoded'] is False
    for item in source_dir.iterdir():
        assert item.stat().st_size == before[item.name].st_size
        assert item.stat().st_mtime_ns == before[item.name].st_mtime_ns


def test_refuses_to_overwrite_existing_user_output(tmp_path: Path, fake_probe: Path):
    source_dir = sources(tmp_path)
    output = tmp_path / 'audit'
    output.mkdir()
    sentinel = output / 'inventory.json'
    sentinel.write_text('user content')
    with pytest.raises(AuditError, match='output_exists'):
        run_audit(source_dir, output, ffprobe=fake_probe)
    assert sentinel.read_text() == 'user content'


@pytest.mark.parametrize('kind', ['source_file', 'source_directory', 'output_parent', 'inside_source'])
def test_rejects_symlinks_and_output_inside_source(tmp_path: Path, fake_probe: Path, kind: str):
    source_dir = sources(tmp_path)
    output = tmp_path / 'audit'
    if kind == 'source_file':
        (source_dir / 'link.mp4').symlink_to(fake_probe)
    elif kind == 'source_directory':
        link = tmp_path / 'source-link'
        link.symlink_to(source_dir, target_is_directory=True)
        source_dir = link
    elif kind == 'output_parent':
        link = tmp_path / 'output-link'
        link.symlink_to(tmp_path, target_is_directory=True)
        output = link / 'audit'
    else:
        output = source_dir / 'audit'
    with pytest.raises(AuditError):
        run_audit(source_dir, output, ffprobe=fake_probe)
    assert not output.exists()


@pytest.mark.parametrize(('name', 'code'), [
    ('fail.mp4', 'ffprobe_failed'), ('mutate.mp4', 'source_changed'),
    ('malformed.mp4', 'ffprobe_invalid_json'), ('nan.mp4', 'invalid_duration'),
])
def test_failures_cannot_produce_success_inventory(tmp_path: Path, fake_probe: Path, name: str, code: str):
    source_dir = sources(tmp_path, (name,))
    output = tmp_path / 'audit'
    with pytest.raises(AuditError, match=code):
        run_audit(source_dir, output, ffprobe=fake_probe)
    assert not (output / 'inventory.json').exists()
    failure = json.loads((output / 'failure.json').read_text())
    assert failure['error_code'] == code
    assert 'private path error' not in json.dumps(failure)


def test_probe_timeout_is_bounded_and_audited(tmp_path: Path, fake_probe: Path):
    source_dir = sources(tmp_path, ('slow.mp4',))
    output = tmp_path / 'audit'
    with pytest.raises(AuditError, match='ffprobe_timeout'):
        run_audit(source_dir, output, ffprobe=fake_probe, timeout_seconds=0.5)
    assert not (output / 'inventory.json').exists()


def test_empty_input_is_explicit_failure(tmp_path: Path, fake_probe: Path):
    source_dir = sources(tmp_path, ())
    with pytest.raises(AuditError, match='no_source_media'):
        run_audit(source_dir, tmp_path / 'audit', ffprobe=fake_probe)


def test_cli_roundtrip_and_fail_closed_json(tmp_path: Path, fake_probe: Path, capsys):
    from videoscope.annotation_review.audit import main

    source_dir = sources(tmp_path)
    output = tmp_path / 'cli-audit'
    arguments = ['--source-dir', str(source_dir), '--output', str(output), '--ffprobe', str(fake_probe)]
    assert main(arguments) == 0
    success = json.loads(capsys.readouterr().out)
    assert success['summary']['file_count'] == 1
    before = (output / 'inventory.json').read_bytes()
    assert main(arguments) == 1
    failure = json.loads(capsys.readouterr().out)
    assert failure == {'status': 'failed', 'error_class': 'infrastructure', 'error_code': 'output_exists'}
    assert (output / 'inventory.json').read_bytes() == before


@pytest.mark.parametrize('timeout', [0, float('nan'), float('inf'), 301])
def test_rejects_unbounded_timeout(tmp_path: Path, fake_probe: Path, timeout: float):
    source_dir = sources(tmp_path)
    with pytest.raises(AuditError, match='invalid_timeout'):
        run_audit(source_dir, tmp_path / 'audit', ffprobe=fake_probe, timeout_seconds=timeout)
