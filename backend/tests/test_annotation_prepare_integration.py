from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from videoscope.annotation_review import prepare
from videoscope.annotation_review.schema import BatchManifest


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def proposal_fixture(tmp_path: Path, monkeypatch):
    root = tmp_path.resolve()
    source = root / 'source.mp4'
    source.write_bytes(b'original source bytes')
    reserved = root / 'reserved.mp4'
    reserved.write_bytes(b'reserved unseen source')
    audit = root / 'audit'; audit.mkdir()
    inventory = {'sources': [
        {'source_id': 'youtube:development', 'source_relpath': source.name, 'bytes': source.stat().st_size,
         'source_sha256': prepare.digest(source), 'duration_seconds': 12.0},
        {'source_id': 'youtube:reserved', 'source_relpath': reserved.name, 'bytes': reserved.stat().st_size,
         'source_sha256': prepare.digest(reserved), 'duration_seconds': 12.0},
    ]}
    roles = {'sources': [
        {'source_id': 'youtube:development', 'role': 'development_review',
         'content_decode_allowed_current_slice': True, 'training_allowed': False, 'promotion_eligible': False},
        {'source_id': 'youtube:reserved', 'role': 'reserve_uninspected',
         'content_decode_allowed_current_slice': False, 'training_allowed': False, 'promotion_eligible': False},
    ]}
    dump(audit / 'inventory.json', inventory); dump(audit / 'source-roles.json', roles)
    ffmpeg = root / 'fake-ffmpeg'
    ffmpeg.write_text('''#!/usr/bin/env python3
import json, sys
from pathlib import Path
args = sys.argv[1:]
target = Path(args[-1])
if '%06d.jpg' in str(target):
    for i in range(3):
        target.with_name(f'{i+1:06d}.jpg').write_bytes(f'frame-{i}'.encode())
        print(f'[Parsed_showinfo] n: {i} pts: {i*4} pts_time:{i*4} duration:1', file=sys.stderr)
elif '-f' in args and args[args.index('-f')+1] == 'null':
    pass
elif '-frames:v' in args:
    target.write_bytes(b'poster')
else:
    duration = float(args[args.index('-t')+1])
    target.write_text(json.dumps({'duration': duration, 'source': args[args.index('-i')+1]}))
'''); ffmpeg.chmod(0o700)
    ffprobe = root / 'fake-ffprobe'
    ffprobe.write_text('''#!/usr/bin/env python3
import json, sys
from pathlib import Path
clip = json.loads(Path(sys.argv[-1]).read_text())
print(json.dumps({'format': {'duration': str(clip['duration'])},
    'streams': [{'codec_type': 'video', 'codec_name': 'h264', 'width': 32, 'height': 32}]}))
'''); ffprobe.chmod(0o700)
    real_check_output = prepare.subprocess.check_output
    def check_output(command, **kwargs):
        if command[:3] == ['git', 'rev-parse', 'HEAD']:
            return 'a' * 40 + '\n'
        return real_check_output(command, **kwargs)
    monkeypatch.setattr(prepare.subprocess, 'check_output', check_output)
    calls = {'images': [], 'texts': []}
    class FakeRuntime:
        def __init__(self, **kwargs):
            self.available = True
        def embed_texts(self, texts):
            calls['texts'].append(texts)
            return [[float(i == j) for j in range(5)] for i in range(len(texts))]
        def embed_images(self, paths):
            calls['images'].append(paths)
            return [[1.0, 0.2, 0.3, 0.4, 0.5] for _ in paths]
    monkeypatch.setitem(sys.modules, 'videoscope.providers.vision_worker',
                        SimpleNamespace(LocalVisionWorkerRuntime=FakeRuntime))
    monkeypatch.setitem(sys.modules, 'torch',
                        SimpleNamespace(mps=SimpleNamespace(current_allocated_memory=lambda: 0)))
    args = SimpleNamespace(root=root, audit=audit, work=root/'work', output=root/'batch',
                           ffmpeg=str(ffmpeg), ffprobe=str(ffprobe), detector=root/'unused.pth',
                           batch_id='synthetic-review-v1')
    return args, source, reserved, calls


def test_sample_score_package_preserves_user_state_and_produces_no_labels(proposal_fixture):
    args, source, reserved, calls = proposal_fixture
    before = (source.read_bytes(), reserved.read_bytes(), source.stat().st_mtime_ns, reserved.stat().st_mtime_ns)
    prepare.sample(args)
    sampled = json.loads((args.work/'sampling.json').read_text())
    assert len(sampled['sources']) == 1
    assert sampled['sources'][0]['source_id'] == 'youtube:development'
    assert sampled['sources'][0]['times'] == [0.0, 4.0, 8.0]
    prepare.score(args)
    scored = json.loads((args.work/'scores.json').read_text())
    assert scored['labels_are_gold'] is False
    assert scored['model_revision'] == prepare.MODEL_REVISION
    assert scored['sampling_sha256'] == prepare.digest(args.work/'sampling.json')
    assert len(calls['images']) == 1
    prepare.package(args)
    batch = BatchManifest.model_validate_json((args.output/'batch.json').read_bytes())
    assert batch.training_allowed is False and batch.promotion_allowed is False
    assert len(batch.examples) == 1
    example = batch.examples[0]
    clip = args.output / example.clip_path
    assert example.prepared_input_sha256 == prepare.digest(clip)
    assert example.prepared_input_byte_size == clip.stat().st_size
    assert example.clip_duration_seconds == 12.0
    assert example.source_start_seconds == 0.0 and example.source_end_seconds == 12.0
    receipt = json.loads((args.output/'preparation-receipt.json').read_text())
    assert receipt['label_count'] == 0 and receipt['training'] is False
    assert receipt['all_previews_full_decode'] == 'passed'
    assert receipt['sources_unchanged'] is True
    assert not list(args.output.glob('*annotation*'))
    assert before == (source.read_bytes(), reserved.read_bytes(), source.stat().st_mtime_ns, reserved.stat().st_mtime_ns)
    command = receipt['clip_commands'][0]
    assert command[command.index('-i') + 1] == str(source)
    assert '-n' in command and '-y' not in command


def test_sampling_rejects_same_size_source_mutation_before_decode(proposal_fixture):
    args, source, _, calls = proposal_fixture
    source.write_bytes(b'X' * source.stat().st_size)
    with pytest.raises(ValueError, match='source hash changed'):
        prepare.sample(args)
    assert not (args.work/'sampling.json').exists()
    assert not calls['images']


@pytest.mark.parametrize('drift', ['roles', 'inventory'])
def test_score_rejects_audit_identity_drift_before_inference(proposal_fixture, drift):
    args, _, _, calls = proposal_fixture
    prepare.sample(args)
    path = args.audit / ('source-roles.json' if drift == 'roles' else 'inventory.json')
    value = json.loads(path.read_text())
    value['revision_note'] = 'changed after sampling'
    dump(path, value)
    with pytest.raises(ValueError):
        prepare.score(args)
    assert not calls['images'] and not calls['texts']
    assert not (args.work/'scores.json').exists()


@pytest.mark.parametrize('tamper', ['reserved_source', 'unknown_source', 'alias', 'source_hash', 'duplicate_source'])
def test_score_rejects_sampling_membership_drift_before_inference(proposal_fixture, tamper):
    args, _, _, calls = proposal_fixture
    prepare.sample(args)
    path = args.work/'sampling.json'
    sampling = json.loads(path.read_text())
    row = sampling['sources'][0]
    if tamper == 'reserved_source': row['source_id'] = 'youtube:reserved'
    if tamper == 'unknown_source': row['source_id'] = 'youtube:unknown'
    if tamper == 'alias': row['alias'] = 'uba-review-99'
    if tamper == 'source_hash': row['source_sha256'] = '0' * 64
    if tamper == 'duplicate_source': sampling['sources'].append(copy.deepcopy(row))
    dump(path, sampling)
    with pytest.raises(ValueError):
        prepare.score(args)
    assert not calls['images'] and not calls['texts']
    assert not (args.work/'scores.json').exists()


def test_score_rejects_changed_sampled_frame_bytes(proposal_fixture):
    args, _, _, calls = proposal_fixture
    prepare.sample(args)
    sampled = json.loads((args.work/'sampling.json').read_text())
    path = args.work / sampled['sources'][0]['frame_paths'][0]
    path.write_bytes(b'changed sample pixels')
    with pytest.raises(ValueError):
        prepare.score(args)
    assert not calls['images']
    assert not (args.work/'scores.json').exists()


def test_score_rejects_external_frame_path(proposal_fixture):
    args, source, _, calls = proposal_fixture
    prepare.sample(args)
    path = args.work/'sampling.json'
    sampled = json.loads(path.read_text())
    sampled['sources'][0]['frame_paths'][0] = str(source)
    dump(path, sampled)
    with pytest.raises(ValueError):
        prepare.score(args)
    assert not calls['images']


@pytest.mark.parametrize('drift', ['sampling', 'roles', 'inventory', 'source'])
def test_package_rejects_changed_provenance_or_source(proposal_fixture, drift):
    args, source, _, _ = proposal_fixture
    prepare.sample(args); prepare.score(args)
    if drift == 'source':
        source.write_bytes(b'X' * source.stat().st_size)
    else:
        path = args.work/'sampling.json' if drift == 'sampling' else args.audit / ('source-roles.json' if drift == 'roles' else 'inventory.json')
        value = json.loads(path.read_text()); value['tampered'] = True; dump(path, value)
    with pytest.raises(ValueError):
        prepare.package(args)
    assert not (args.output/'batch.json').exists()


@pytest.mark.parametrize('drift', ['duplicate_source', 'extra_source', 'times'])
def test_package_rejects_scores_disconnected_from_sampled_sources(proposal_fixture, drift):
    args, _, _, _ = proposal_fixture
    prepare.sample(args); prepare.score(args)
    path = args.work/'scores.json'
    scored = json.loads(path.read_text())
    if drift == 'duplicate_source': scored['sources'].append(copy.deepcopy(scored['sources'][0]))
    elif drift == 'extra_source':
        row = copy.deepcopy(scored['sources'][0]); row['source_id'] = 'youtube:reserved'; scored['sources'].append(row)
    else: scored['sources'][0]['times'][0] = 0.1
    dump(path, scored)
    with pytest.raises(ValueError):
        prepare.package(args)
    assert not (args.output/'batch.json').exists()


def test_real_ffmpeg_synthetic_preview_roundtrip(proposal_fixture):
    args, source, _, _ = proposal_fixture
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        pytest.skip('FFmpeg is required for the local media integration check')
    source.unlink()
    subprocess.run([ffmpeg, '-v', 'error', '-n', '-f', 'lavfi', '-i',
                    'color=c=blue:s=32x32:r=5:d=8', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)],
                   check=True, capture_output=True, timeout=30)
    inventory_path = args.audit/'inventory.json'
    inventory = json.loads(inventory_path.read_text())
    inventory['sources'][0].update(bytes=source.stat().st_size, source_sha256=prepare.digest(source), duration_seconds=8.0)
    dump(inventory_path, inventory)
    args.ffmpeg, args.ffprobe = ffmpeg, ffprobe
    original_sha = prepare.digest(source)
    prepare.sample(args); prepare.score(args); prepare.package(args)
    batch = BatchManifest.model_validate_json((args.output/'batch.json').read_bytes())
    assert len(batch.examples) == 1
    example = batch.examples[0]
    assert example.clip_duration_seconds == pytest.approx(8.0, abs=0.15)
    assert prepare.digest(args.output/example.clip_path) == example.prepared_input_sha256
    assert (args.output/example.poster_path).stat().st_size > 0
    assert prepare.digest(source) == original_sha


def test_package_failure_preserves_existing_batch(proposal_fixture):
    args, _, _, _ = proposal_fixture
    prepare.sample(args); prepare.score(args); prepare.package(args)
    paths = [p for p in args.output.rglob('*') if p.is_file()]
    before = {p: p.read_bytes() for p in paths}
    with pytest.raises(FileExistsError):
        prepare.package(args)
    assert before == {p: p.read_bytes() for p in paths}


def test_package_source_mutation_during_export_never_publishes_batch(proposal_fixture, monkeypatch):
    args, source, _, _ = proposal_fixture
    prepare.sample(args); prepare.score(args)
    real_run = prepare.subprocess.run
    def mutate_after_export(command, **kwargs):
        result = real_run(command, **kwargs)
        if '-t' in command and str(source) in command:
            source.write_bytes(b'X' * source.stat().st_size)
        return result
    monkeypatch.setattr(prepare.subprocess, 'run', mutate_after_export)
    with pytest.raises(ValueError, match='source changed during preparation'):
        prepare.package(args)
    assert not (args.output/'batch.json').exists()
    assert not (args.output/'preparation-receipt.json').exists()


def test_export_duration_drift_never_publishes_batch(proposal_fixture, monkeypatch):
    args, _, _, _ = proposal_fixture
    prepare.sample(args); prepare.score(args)
    real_check_output = prepare.subprocess.check_output
    def wrong_duration(command, **kwargs):
        if command[0] == args.ffprobe:
            return json.dumps({'format': {'duration': '14'}, 'streams': []}).encode()
        return real_check_output(command, **kwargs)
    monkeypatch.setattr(prepare.subprocess, 'check_output', wrong_duration)
    with pytest.raises(ValueError, match='clip duration drift'):
        prepare.package(args)
    assert not (args.output/'batch.json').exists()
    assert not (args.output/'preparation-receipt.json').exists()


@pytest.mark.parametrize('must_exist', [True, False])
@pytest.mark.parametrize('placement', ['itself', 'ancestor'])
def test_plain_directory_rejects_symlinks_before_resolution(tmp_path: Path, must_exist, placement):
    root = tmp_path.resolve()
    target = root / 'target'; target.mkdir()
    (target / 'child').mkdir()
    link = root / 'link'; link.symlink_to(target, target_is_directory=True)
    path = link if placement == 'itself' else link / 'child' / ('existing' if must_exist else 'future')
    if must_exist and placement == 'ancestor':
        (target/'child'/'existing').mkdir()
    with pytest.raises(ValueError):
        prepare.require_plain_directory(path, must_exist=must_exist)


def test_plain_directory_requires_existing_directory_when_requested(tmp_path: Path):
    root = tmp_path.resolve()
    prepare.require_plain_directory(root, must_exist=True)
    prepare.require_plain_directory(root/'future'/'nested', must_exist=False)
    with pytest.raises(ValueError):
        prepare.require_plain_directory(root/'absent', must_exist=True)
    file = root/'file'; file.write_text('user file')
    with pytest.raises(ValueError):
        prepare.require_plain_directory(file, must_exist=True)
    assert file.read_text() == 'user file'


def test_sample_rejects_symlink_work_ancestor_before_decoding(proposal_fixture):
    args, source, _, _ = proposal_fixture
    target = args.root/'another'; target.mkdir()
    link = args.root/'linked'; link.symlink_to(target, target_is_directory=True)
    args.work = link/'new-work'
    original = source.read_bytes()
    with pytest.raises(ValueError):
        prepare.sample(args)
    assert list(target.iterdir()) == []
    assert source.read_bytes() == original


def test_validate_sampling_rejects_symlink_work_itself(proposal_fixture):
    args, _, _, _ = proposal_fixture
    prepare.sample(args)
    sampling = json.loads((args.work/'sampling.json').read_text())
    target = args.root/'moved-work'
    args.work.rename(target)
    args.work.symlink_to(target, target_is_directory=True)
    before = {p: p.read_bytes() for p in target.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        prepare.validate_sampling(args, sampling)
    assert before == {p: p.read_bytes() for p in before}


def test_package_rejects_symlink_output_ancestor_before_export(proposal_fixture):
    args, source, _, _ = proposal_fixture
    prepare.sample(args); prepare.score(args)
    target = args.root/'another'; target.mkdir()
    link = args.root/'linked'; link.symlink_to(target, target_is_directory=True)
    args.output = link/'new-batch'
    original = source.read_bytes()
    with pytest.raises(ValueError):
        prepare.package(args)
    assert list(target.iterdir()) == []
    assert source.read_bytes() == original


def test_plain_directory_cannot_hide_symlink_before_parent_component(tmp_path: Path):
    root = tmp_path.resolve()
    other = root/'other'; other.mkdir()
    child = other/'child'; child.mkdir()
    link = root/'link'; link.symlink_to(child, target_is_directory=True)
    # Lexical normalization points to root/new; the filesystem would follow link
    # and then '..', reaching root/other/new instead.
    path = link/'..'/'new'
    with pytest.raises(ValueError):
        prepare.require_plain_directory(path, must_exist=False)
    assert not (other/'new').exists()
