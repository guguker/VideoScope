"""Read-only source inventory for private annotation preparation.

This command probes headers and reads bytes for SHA-256. It does not decode frames,
import media into VideoScope, infer labels, grant rights or assign dataset splits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
from typing import Any

BLOCK_BYTES = 8 * 1024 * 1024
MAX_PROBE_OUTPUT_BYTES = 4 * 1024 * 1024


class AuditError(ValueError):
    """An infrastructure or data-contract failure, never a model miss."""


def _time() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    for current in (absolute, *absolute.parents):
        if current.is_symlink():
            raise AuditError('symlink_path_not_allowed')
    return absolute


def _identity(value: os.stat_result) -> dict[str, int]:
    return {
        'device': value.st_dev, 'inode': value.st_ino, 'bytes': value.st_size,
        'mtime_ns': value.st_mtime_ns, 'ctime_ns': value.st_ctime_ns,
    }


def _source_identity(path: Path) -> dict[str, int]:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise AuditError('source_not_regular_file')
    return _identity(value)


def _hash_stable(path: Path, expected: dict[str, int]) -> str:
    digest = hashlib.sha256()
    size = 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as source:
        if _identity(os.fstat(source.fileno())) != expected:
            raise AuditError('source_changed')
        while block := source.read(BLOCK_BYTES):
            digest.update(block)
            size += len(block)
        if _identity(os.fstat(source.fileno())) != expected:
            raise AuditError('source_changed')
    if _source_identity(path) != expected or size != expected['bytes']:
        raise AuditError('source_changed')
    return digest.hexdigest()


def _execute(command: list[str], timeout: float) -> str:
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AuditError('ffprobe_timeout') from exc
    except OSError as exc:
        raise AuditError('ffprobe_unavailable') from exc
    if result.returncode:
        raise AuditError('ffprobe_failed')
    if len(result.stdout) > MAX_PROBE_OUTPUT_BYTES:
        raise AuditError('ffprobe_output_too_large')
    try:
        return result.stdout.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise AuditError('ffprobe_invalid_encoding') from exc


def _probe(path: Path, executable: Path, timeout: float) -> tuple[dict[str, Any], list[str]]:
    command = [str(executable), '-v', 'error', '-protocol_whitelist', 'file,pipe',
               '-show_format', '-show_streams', '-of', 'json', str(path)]
    try:
        value = json.loads(_execute(command, timeout))
    except json.JSONDecodeError as exc:
        raise AuditError('ffprobe_invalid_json') from exc
    if not isinstance(value, dict) or not isinstance(value.get('format'), dict):
        raise AuditError('ffprobe_invalid_metadata')
    try:
        duration = float(value['format']['duration'])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError('invalid_duration') from exc
    if not math.isfinite(duration) or duration <= 0:
        raise AuditError('invalid_duration')
    streams = value.get('streams')
    if not isinstance(streams, list) or not all(isinstance(s, dict) for s in streams):
        raise AuditError('ffprobe_invalid_streams')
    if not any(s.get('codec_type') == 'video' for s in streams):
        raise AuditError('source_has_no_video')
    return value, command


def _origin(filename: str) -> dict[str, Any]:
    found = re.search(r'\[([A-Za-z0-9_-]{11})\]', filename)
    identifier = found.group(1) if found else None
    return {
        'platform': 'youtube' if identifier else None,
        'video_id': identifier,
        'candidate_url': f'https://www.youtube.com/watch?v={identifier}' if identifier else None,
        'confidence': 'inferred_from_filename' if identifier else 'unknown',
        'externally_verified': False,
    }


def _write_new(path: Path, value: dict[str, Any]) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    path.chmod(0o600)


def run_audit(
    source_dir: Path,
    output: Path,
    *,
    ffprobe: Path | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Create a new private audit directory, refusing all existing output state."""
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 300:
        raise AuditError('invalid_timeout')
    source_dir, output = _safe_path(source_dir), _safe_path(output)
    if output.exists():
        raise AuditError('output_exists')
    if not source_dir.is_dir() or not output.parent.is_dir():
        raise AuditError('directory_missing')
    if output.is_relative_to(source_dir):
        raise AuditError('output_inside_source_directory')
    sources = sorted(p for p in source_dir.iterdir() if p.suffix.lower() == '.mp4')
    if not sources:
        raise AuditError('no_source_media')
    identities = {p: _source_identity(_safe_path(p)) for p in sources}
    directory_identity = _identity(source_dir.stat())
    executable_name = str(ffprobe) if ffprobe is not None else shutil.which('ffprobe')
    if not executable_name:
        raise AuditError('ffprobe_unavailable')
    # Homebrew executable links are resolved once; media/output links are forbidden.
    executable = Path(executable_name).resolve(strict=True)
    executable_identity = _source_identity(executable)
    executable_sha = _hash_stable(executable, executable_identity)
    output.mkdir(mode=0o700)
    try:
        probe_version = _execute([str(executable), '-version'], timeout_seconds)
        rows: list[dict[str, Any]] = []
        raw_probes: list[dict[str, Any]] = []
        duplicates: dict[str, list[str]] = defaultdict(list)
        for index, path in enumerate(sources):
            before = identities[path]
            if _source_identity(path) != before:
                raise AuditError('source_changed')
            probe, command = _probe(path, executable, timeout_seconds)
            if _source_identity(path) != before:
                raise AuditError('source_changed')
            digest = _hash_stable(path, before)
            identifier = f'source-{index + 1:04d}'
            origin = _origin(path.name)
            stream_keys = ('index', 'codec_name', 'profile', 'width', 'height', 'pix_fmt',
                           'avg_frame_rate', 'r_frame_rate', 'time_base', 'duration', 'nb_frames',
                           'sample_rate', 'channels', 'channel_layout', 'bit_rate')
            selected = lambda kind: [{key: stream[key] for key in stream_keys if key in stream}
                                    for stream in probe['streams'] if stream.get('codec_type') == kind]
            row = {
                'source_id': identifier, 'source_sha256': digest, 'filename': path.name,
                'source_relpath': path.name, 'bytes': before['bytes'],
                'duration_seconds': float(probe['format']['duration']),
                'format_name': probe['format'].get('format_name'),
                'video_streams': selected('video'), 'audio_streams': selected('audio'),
                'subtitle_stream_count': len(selected('subtitle')),
                'stable_source': {'before': before, 'after': _source_identity(path),
                                  'metadata_and_hash_stats_identical': True},
                'origin': origin, 'role': 'unassigned_metadata_only',
                'grouping': {'source_group_id': f'sha256:{digest}',
                             'origin_group_suggestion': f"youtube:{origin['video_id']}" if origin['video_id'] else None,
                             'confirmed_game_group': None, 'cross_source_replay_groups': 'unknown'},
                'readability_check': {'metadata_probe': 'passed', 'all_bytes_read_for_sha256': 'passed',
                                      'full_stream_decode': 'not_run'},
                'training_allowed': False, 'promotion_eligible': False,
                'issues': ['rights_unknown', 'game_and_replay_groups_unverified',
                           'full_stream_decode_not_performed', 'coverage_not_observed'],
            }
            rows.append(row)
            duplicates[digest].append(identifier)
            raw_probes.append({'source_id': identifier, 'command': command, 'metadata': probe})
        if _identity(source_dir.stat()) != directory_identity:
            raise AuditError('source_directory_changed')
        for path, before in identities.items():
            if _source_identity(path) != before:
                raise AuditError('source_changed')
        if _hash_stable(executable, executable_identity) != executable_sha:
            raise AuditError('ffprobe_changed')
        exact_duplicates = [group for group in duplicates.values() if len(group) > 1]
        inventory = {
            'schema': 'videoscope-private-source-inventory-v1', 'created_at': _time(),
            'status': 'metadata_and_sha_audit_complete_group_and_rights_audit_incomplete',
            'scope': {'source_directory': str(source_dir), 'source_glob': '*.mp4 (case insensitive)',
                      'content_decoded': False, 'external_requests': False, 'training': False,
                      'production_ingest_or_index': False, 'source_media_modified': False},
            'runtime': {'python_version': platform.python_version(), 'platform': platform.platform(),
                        'ffprobe': {'path': str(executable), 'sha256': executable_sha, 'version': probe_version},
                        'audit_implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'protocol': {'hash_algorithm': 'sha256', 'block_bytes': BLOCK_BYTES, 'concurrent_readers': 1,
                         'ffprobe_timeout_seconds': timeout_seconds, 'network_protocols_allowed': False},
            'summary': {'file_count': len(rows), 'total_bytes': sum(r['bytes'] for r in rows),
                        'total_duration_seconds': sum(r['duration_seconds'] for r in rows),
                        'exact_duplicate_groups': exact_duplicates,
                        'exact_duplicate_group_count': len(exact_duplicates), 'all_sources_stable': True},
            'sources': rows, 'phase1_exit_gate': 'not_met',
            'limitations': ['SHA equality proves exact duplicates; inequality does not exclude another encoding or replay.',
                            'Metadata and complete byte reads do not prove every frame decodes.',
                            'This inventory is not a dataset split, rights grant, gold annotation or promotion evidence.'],
        }
        ledger = {
            'schema': 'videoscope-private-source-rights-ledger-v1', 'created_at': _time(),
            'sources': [{'source_id': row['source_id'], 'source_sha256': row['source_sha256'],
                         'origin': row['origin'], 'rights_status': 'unknown', 'license_id': None,
                         'license_evidence': [], 'owner_consent_evidence': [],
                         'training_allowed': False, 'external_upload_allowed': False,
                         'publication_allowed': False, 'privacy_class': 'private_local_user_media'} for row in rows],
        }
        _write_new(output / 'metadata-probes.json', {'sources': raw_probes})
        _write_new(output / 'source-rights-ledger.json', ledger)
        _write_new(output / 'inventory.json', inventory)
        return inventory
    except (AuditError, OSError) as exc:
        code = str(exc) if isinstance(exc, AuditError) else 'audit_io_failure'
        _write_new(output / 'failure.json', {'status': 'failed', 'error_class': 'infrastructure',
                                           'error_code': code, 'created_at': _time()})
        if isinstance(exc, AuditError):
            raise
        raise AuditError(code) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New directory beneath an existing parent')
    parser.add_argument('--ffprobe', type=Path)
    parser.add_argument('--timeout-seconds', type=float, default=30.0)
    arguments = parser.parse_args(argv)
    try:
        result = run_audit(arguments.source_dir, arguments.output, ffprobe=arguments.ffprobe,
                           timeout_seconds=arguments.timeout_seconds)
    except (AuditError, OSError) as exc:
        code = str(exc) if isinstance(exc, AuditError) else 'audit_io_failure'
        print(json.dumps({'status': 'failed', 'error_class': 'infrastructure', 'error_code': code}))
        return 1
    print(json.dumps({'status': result['status'], 'summary': result['summary']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
