"""Offline, development-only proposal sampling and human-review clip preparation.

This CLI never writes labels, opens a product index, or treats proposals as gold.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


QUERIES = (
    ('three', 'A basketball player shoots a three point jump shot from beyond the three point line toward the basket.'),
    ('two', 'A basketball player drives to the hoop and shoots a layup close to the basket.'),
    ('free_throw', 'A basketball player takes a free throw while other players line up on both sides of the lane.'),
    ('replay', 'A close-up television replay of a basketball player shooting the ball in slow motion.'),
    ('non_game', 'A basketball broadcast shows a timeout, a coach, bench, interview or advertisement instead of live play.'),
)
MODEL_ID = 'google/siglip2-base-patch16-224'
MODEL_REVISION = '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as stream:
        os.chmod(path, 0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def finite(value: float) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def require_plain_directory(path: Path, *, must_exist: bool = True) -> None:
    if '..' in Path(path).parts:
        raise ValueError('parent traversal is not an explicit review directory')
    absolute = Path(os.path.abspath(path))
    if any(p.is_symlink() for p in (absolute, *absolute.parents)):
        raise ValueError('directory or ancestor is a symlink')
    if (absolute.exists() and not absolute.is_dir()) or (must_exist and not absolute.is_dir()):
        raise ValueError('expected a plain directory')


def selected_sources(inventory: dict, roles: dict, root: Path) -> list[dict]:
    root = root.resolve(strict=True)
    source_rows = inventory['sources']
    by_id = {r['source_id']: r for r in source_rows}
    role_rows = roles['sources']
    if len(by_id) != len(source_rows) or len({r['source_id'] for r in role_rows}) != len(role_rows):
        raise ValueError('duplicate source identity')
    result = []
    for role in role_rows:
        if role.get('content_decode_allowed_current_slice') is not True:
            continue
        if (role.get('role') != 'development_review' or
                role.get('training_allowed') is not False or
                role.get('promotion_eligible') is not False):
            raise ValueError('only explicit development_review sources may be decoded')
        source = by_id[role['source_id']]
        relative = Path(source['source_relpath'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('source path escapes allowed root')
        path = root / relative
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('source symlinks are not allowed')
        if not path.is_file() or path.stat().st_size != source['bytes']:
            raise ValueError('source is absent or size changed')
        result.append({**source, 'path': path, 'alias': f'uba-review-{len(result)+1:02d}'})
    if not result:
        raise ValueError('no sources explicitly authorized for review')
    return result


def parse_sample_times(log: str) -> list[float]:
    times = [float(t) for t in re.findall(r'\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:([\d.eE+-]+)', log)]
    if not times or any(not finite(t) or t < 0 for t in times):
        raise ValueError('no valid decoded sample timestamps')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('sample timestamps must increase')
    return times


def select_windows(times: list[float], scores: list[list[float]], duration: float) -> list[dict]:
    if not finite(duration) or duration <= 0 or len(times) != len(scores) or not times:
        raise ValueError('invalid sampling dimensions or duration')
    if any(not finite(t) or not 0 <= t < duration for t in times) or any(b <= a for a,b in zip(times, times[1:])):
        raise ValueError('invalid sampling times')
    if any(len(row) != len(QUERIES) or any(not finite(v) for v in row) for row in scores):
        raise ValueError('invalid proposal scores')
    selected: list[dict] = []

    def add(center: float, method: str, note: str) -> bool:
        if any(abs(center - row['center']) < 30 for row in selected):
            return False
        start = max(0.0, min(center - 6.0, duration - 18.0))
        selected.append({'center': center, 'start': start, 'end': min(duration, start + 18.0),
                         'selection_method': method, 'selection_notes': note})
        return True

    for n in range(4):
        add(duration * (n + 0.5) / 4, 'timeline_uniform', 'Systematic full-timeline control; no event label.')
    for k, (name, _) in enumerate(QUERIES):
        remaining = 2 if k < 3 else 1
        for i in sorted(range(len(times)), key=lambda i: (-scores[i][k], times[i])):
            if add(times[i], 'siglip_proposal', f'Weak retrieval prompt: {name}; not a human label.'):
                remaining -= 1
                if not remaining:
                    break
    return selected


def clip_command(binary: str, source: Path, target: Path, start: float, duration: float) -> list[str]:
    if source.resolve() == target.resolve() or not finite(start) or start < 0 or not finite(duration) or not 0 < duration <= 60:
        raise ValueError('invalid clip request')
    return [binary, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-threads', '2',
            '-ss', f'{start:.6f}', '-protocol_whitelist', 'file,pipe', '-i', str(source),
            '-t', f'{duration:.6f}', '-map', '0:v:0', '-map', '0:a:0?', '-c:v', 'libx264',
            '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2',
            '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', str(target)]


def _fingerprint(path: Path) -> tuple:
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def verify_source(source: dict) -> tuple:
    before = _fingerprint(source['path'])
    if digest(source['path']) != source['source_sha256'] or before != _fingerprint(source['path']):
        raise ValueError('source hash changed since inventory')
    return before


def sources_for(args) -> list[dict]:
    inventory = json.loads((args.audit / 'inventory.json').read_text())
    roles = json.loads((args.audit / 'source-roles.json').read_text())
    return selected_sources(inventory, roles, args.root)


def identity(args) -> dict:
    return {'prepare_sha256': digest(Path(__file__)),
            'inventory_sha256': digest(args.audit / 'inventory.json'),
            'roles_sha256': digest(args.audit / 'source-roles.json'),
            'code_sha': subprocess.check_output(['git','rev-parse','HEAD'], cwd=args.root, text=True).strip()}


def sample(args) -> None:
    require_plain_directory(args.work, must_exist=False)
    sources = sources_for(args)
    args.work.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {'schema_version': 1, 'identity': identity(args), 'sampling':
               {'method': 'keyframes_minimum_4s_actual_pts', 'minimum_step_seconds': 4,
                'frame_width': 384, 'is_retrieval_benchmark': False}, 'sources': []}
    for source in sources:
        before = verify_source(source)
        frames = args.work / source['alias']; frames.mkdir(mode=0o700)
        cmd = [args.ffmpeg, '-hide_banner','-nostdin','-n','-threads','2',
               '-skip_frame','nokey','-protocol_whitelist','file,pipe','-i',str(source['path']),
               '-an','-vf',"select='isnan(prev_selected_t)+gte(t-prev_selected_t,4)',showinfo,scale=384:-2",
               '-filter_threads','1','-fps_mode','vfr','-q:v','3',str(frames / '%06d.jpg')]
        print(f"Sampling {source['alias']}", flush=True)
        with (frames / 'decode.log').open('x') as log:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=log, check=True, timeout=1800)
        times = parse_sample_times((frames / 'decode.log').read_text())
        images = sorted(frames.glob('*.jpg'))
        if len(images) != len(times) or before != _fingerprint(source['path']):
            raise ValueError('sample count mismatch or source changed')
        receipt['sources'].append({'source_id': source['source_id'], 'alias': source['alias'],
            'source_sha256': source['source_sha256'], 'duration_seconds': source['duration_seconds'],
            'times': times, 'frame_paths': [str(p.relative_to(args.work)) for p in images],
            'frame_sha256': [digest(p) for p in images],
            'command': cmd, 'source_fingerprint': before})
        print(f"Sampled {len(images)} frames from {source['alias']}", flush=True)
    write_json(args.work / 'sampling.json', receipt)


def validate_sampling(args, sampling: dict) -> list[dict]:
    require_plain_directory(args.work)
    sources = sources_for(args)
    expected = identity(args)
    if any(sampling['identity'][key] != expected[key] for key in ('inventory_sha256', 'roles_sha256')):
        raise ValueError('sampling source authorization or inventory changed')
    rows = sampling['sources']
    if len(rows) != len(sources) or len({r['source_id'] for r in rows}) != len(rows):
        raise ValueError('sampling source membership changed')
    by_id = {r['source_id']: r for r in rows}
    if set(by_id) != {s['source_id'] for s in sources}:
        raise ValueError('sampling contains an unauthorized source')
    for source in sources:
        verify_source(source)
        row = by_id[source['source_id']]
        if any(row[k] != source[k] for k in ('alias', 'source_sha256', 'duration_seconds')):
            raise ValueError('sampling source identity changed')
        times = row['times']
        if (not times or len(times) != len(row['frame_paths']) or len(times) != len(row['frame_sha256']) or
            any(not finite(t) or not 0 <= t < source['duration_seconds'] for t in times) or
            any(b <= a for a,b in zip(times,times[1:]))):
            raise ValueError('invalid sample timeline')
        for i, (relative, sha) in enumerate(zip(row['frame_paths'], row['frame_sha256']), 1):
            if relative != f"{source['alias']}/{i:06d}.jpg":
                raise ValueError('unexpected frame path')
            path = args.work / relative
            if any(p.is_symlink() for p in (path, path.parent)) or not path.is_file() or digest(path) != sha:
                raise ValueError('sample frame changed or is not contained')
    return sources


def validate_scores(args, scored: dict, sampling: dict) -> None:
    expected = identity(args)
    if (scored['sampling_sha256'] != digest(args.work / 'sampling.json') or
        any(scored['identity'][key] != expected[key] for key in ('inventory_sha256','roles_sha256'))):
        raise ValueError('proposal provenance changed')
    rows = scored['sources']; samples = {r['source_id']:r for r in sampling['sources']}
    if (len(rows) != len(samples) or len({r['source_id'] for r in rows}) != len(rows) or
        {r['source_id'] for r in rows} != set(samples)):
        raise ValueError('proposal source membership changed')
    for row in rows:
        sample_row = samples[row['source_id']]
        if row['alias'] != sample_row['alias'] or row['times'] != sample_row['times']:
            raise ValueError('proposal timeline or alias changed')
        select_windows(row['times'], row['scores'], sample_row['duration_seconds'])


def score(args) -> None:
    sampling = json.loads((args.work / 'sampling.json').read_text())
    validate_sampling(args, sampling)
    # Reuse the production, pinned, offline-only preprocessing and embedding runtime.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from videoscope.providers.vision_worker import LocalVisionWorkerRuntime
    from videoscope.providers.vision_worker_contract import (
        VisionWorkerSpecification, REVIEWED_SIGLIP_PROFILES, RFDETR_SMALL_CHECKPOINT_SHA256,
    )
    import numpy as np
    import torch
    dims, scale, bias = REVIEWED_SIGLIP_PROFILES[(MODEL_ID, MODEL_REVISION)]
    spec = VisionWorkerSpecification(siglip_model=MODEL_ID, siglip_revision=MODEL_REVISION,
        embedding_dimensions=dims, detector_model_id='rfdetr-small',
        detector_checkpoint_sha256=RFDETR_SMALL_CHECKPOINT_SHA256,
        siglip_logit_scale=scale, siglip_logit_bias=bias, minimum_confidence=0.3)
    runtime = LocalVisionWorkerRuntime(specification=spec, detector_checkpoint=args.detector)
    if not runtime.available:
        raise RuntimeError('Pinned offline Vision runtime unavailable; no download or fallback')
    started = time.monotonic()
    text = np.asarray(runtime.embed_texts(tuple(q for _,q in QUERIES)), dtype=np.float32)
    results = []
    for source in sampling['sources']:
        paths = [(args.work / p).resolve(strict=True) for p in source['frame_paths']]
        if any(not p.is_relative_to(args.work.resolve()) for p in paths):
            raise ValueError('sample path escaped work directory')
        rows = []
        for n in range(0, len(paths), 32):
            embeddings = np.asarray(runtime.embed_images(tuple(paths[n:n+32])), dtype=np.float32)
            rows.extend((embeddings @ text.T).tolist())
            if n % 320 == 0:
                print(f"Scored {source['alias']} {min(n+32,len(paths))}/{len(paths)}", flush=True)
        results.append({'source_id': source['source_id'], 'alias': source['alias'],
            'times': source['times'], 'scores': rows,
            'windows': select_windows(source['times'], rows, source['duration_seconds'])})
    write_json(args.work / 'scores.json', {'schema_version':1, 'identity':identity(args),
        'sampling_sha256':digest(args.work / 'sampling.json'), 'model_id':MODEL_ID,
        'model_revision':MODEL_REVISION, 'preprocessing_revision':spec.siglip_preprocessing_revision,
        'runtime_identity':spec.runtime_identity, 'prompts':list(QUERIES),
        'elapsed_seconds':time.monotonic()-started, 'current_mps_allocated_bytes':torch.mps.current_allocated_memory(),
        'labels_are_gold':False, 'sources':results})


def package(args) -> None:
    require_plain_directory(args.output, must_exist=False)
    scored = json.loads((args.work / 'scores.json').read_text())
    sampling = json.loads((args.work / 'sampling.json').read_text())
    sources = validate_sampling(args, sampling)
    validate_scores(args, scored, sampling)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (args.output / 'clips').mkdir(mode=0o700); (args.output / 'posters').mkdir(mode=0o700)
    examples = []; source_rows = []; commands = []
    scored_by_id = {r['source_id']:r for r in scored['sources']}
    for source in sources:
        before = verify_source(source)
        raw = scored_by_id[source['source_id']]
        # Recompute deterministic selection instead of trusting edited proposed windows.
        windows = select_windows(raw['times'], raw['scores'], source['duration_seconds'])
        source_rows.append({'source_id':source['alias'], 'sha256':source['source_sha256'],
            'byte_size':source['bytes'], 'duration_seconds':source['duration_seconds'],
            'source_group':source['alias'], 'usage':'development_review',
            'review_allowed':True, 'training_rights':'unknown'})
        for window in windows:
            example_id = f'uba-{len(examples)+1:03d}'
            clip = args.output / 'clips' / f'{example_id}.mp4'
            poster = args.output / 'posters' / f'{example_id}.jpg'
            cmd = clip_command(args.ffmpeg, source['path'], clip, window['start'], window['end']-window['start'])
            print(f'Preparing {example_id}', flush=True)
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            commands.append(cmd)
            probe = json.loads(subprocess.check_output([args.ffprobe,'-v','error','-show_format','-show_streams','-of','json',str(clip)], timeout=30))
            clip_duration = float(probe['format']['duration'])
            if abs(clip_duration-(window['end']-window['start'])) > 0.15:
                raise ValueError('exported clip duration drift')
            subprocess.run([args.ffmpeg,'-v','error','-nostdin','-n','-ss','6','-i',str(clip),
                '-frames:v','1','-vf','scale=640:-2',str(poster)], check=True, capture_output=True, timeout=30)
            # Full-decode check of every user-facing preview, not just container metadata.
            subprocess.run([args.ffmpeg,'-v','error','-xerror','-threads','2','-i',str(clip),
                '-f','null','-'], check=True, capture_output=True, timeout=60)
            examples.append({'example_id':example_id,'source_id':source['alias'],
                'source_start_seconds':window['start'],'source_end_seconds':window['end'],
                'clip_duration_seconds':clip_duration,'prepared_input_sha256':digest(clip),
                'prepared_input_byte_size':clip.stat().st_size,'clip_path':str(clip.relative_to(args.output)),
                'poster_path':str(poster.relative_to(args.output)),
                'selection_method':window['selection_method'],'selection_notes':window['selection_notes']})
        if before != _fingerprint(source['path']):
            raise ValueError('source changed during preparation')
    batch = {'schema_version':1,'batch_id':args.batch_id,'title':'UBA · первая проверка эпизодов',
        'created_at':datetime.now(timezone.utc).isoformat(),'code_sha':identity(args)['code_sha'],
        'purpose':'annotation_pilot','training_allowed':False,'promotion_allowed':False,
        'sources':source_rows,'examples':examples}
    from .schema import BatchManifest
    BatchManifest.model_validate(batch)
    write_json(args.output / 'preparation-receipt.json', {'identity':identity(args),
        'scores_sha256':digest(args.work / 'scores.json'),'sampling_sha256':digest(args.work / 'sampling.json'),
        'clip_commands':commands,'ffmpeg_sha256':digest(Path(args.ffmpeg)),
        'ffprobe_sha256':digest(Path(args.ffprobe)),'all_previews_full_decode':'passed',
        'sources_unchanged':True,'label_count':0,'training':False,'external_requests':False})
    write_json(args.output / 'batch.json', batch)
    print(f'Ready: {len(examples)} unlabelled review clips at {args.output}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('sample','score','package'))
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--batch-id', default='uba-pilot-v1')
    parser.add_argument('--ffmpeg', default='/opt/homebrew/bin/ffmpeg')
    parser.add_argument('--ffprobe', default='/opt/homebrew/bin/ffprobe')
    parser.add_argument('--detector', type=Path, default=Path('data/models/rfdetr/rf-detr-small.pth'))
    args = parser.parse_args()
    if args.command == 'package' and args.output is None:
        parser.error('--output is required for package')
    {'sample':sample, 'score':score, 'package':package}[args.command](args)


if __name__ == '__main__':
    main()
