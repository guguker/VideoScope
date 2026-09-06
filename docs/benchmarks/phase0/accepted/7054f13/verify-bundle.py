"""Replay accepted evidence through the unchanged clean-checkout collector; no ML."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[4]
CODE_SHA = '7054f137f92c18676461242a18c8fc2619898b68'
BUNDLE_ID = 'sha256:5abee44ff9d4d85bd2c78de7b89e66668c4a1170aab9861eb5c56537b95111c1'


def bound_path(binding):
    relative = Path(binding['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = (REPOSITORY / relative).resolve(strict=True)
    assert path.is_relative_to(REPOSITORY)
    raw = path.read_bytes()
    assert len(raw) == binding['byte_size'], str(relative)
    assert sha256(raw).hexdigest() == binding['sha256'], str(relative)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', required=True, type=Path,
                        help='Clean detached 7054f13 checkout with installed base environment')
    args = parser.parse_args()
    checkout = args.checkout.resolve(strict=True)
    index = json.loads((ROOT / 'evidence-index.json').read_bytes())
    assert index['code_sha'] == CODE_SHA and index['bundle_id'] == BUNDLE_ID
    assert index['collector_exit_code'] == 0 and index['collector_status'] == 'written'
    inputs = {key: bound_path(value) for key, value in index['inputs'].items()}
    artifacts = {key: bound_path(value) for key, value in index['artifacts'].items()}
    for value in index['supporting_evidence'].values():
        bound_path(value)
    assert set(artifacts) == {'baseline-snapshot.json', 'raw-measurements.json',
                              'sanitized-report.json', 'error-ledger.json'}
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout,
                                   text=True).strip() == CODE_SHA
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=checkout)
    options = {
        'frozen_metric_policy': '--policy', 'product_dataset': '--product-dataset',
        'video_verifier_dataset': '--verifier-dataset',
        'ml_environment_attestation': '--environment-attestation',
        'full_ml_smoke': '--full-ml-smoke', 'baseline_batch': '--baseline-batch',
        'video_verifier_run': '--video-verifier-run', 'rollback_proof': '--rollback-proof',
    }
    command = [str(checkout / '.venv/bin/python'), '-I',
               str(checkout / 'scripts/phase0-evidence.py'), '--code-sha', CODE_SHA]
    for identity, option in options.items():
        command.extend([option, str(inputs[identity])])
    profiles = ('lexical_qdrant', 'dense_siglip', 'temporal_refinement',
                'lighthouse', 'qwen_verification')
    for profile in profiles:
        command.extend(['--benchmark-run', profile + '=' + str(inputs['benchmark_run.' + profile])])
    with tempfile.TemporaryDirectory(prefix='videoscope-phase0-bundle-replay-') as temporary:
        destination = Path(temporary) / 'bundle'
        result = subprocess.run(command + ['--output-dir', str(destination)],
                                cwd=checkout, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {
            'status': 'written', 'bundle_id': BUNDLE_ID, 'code_sha': CODE_SHA,
        }
        assert {path.name for path in destination.iterdir()} == set(artifacts)
        for name, original in artifacts.items():
            assert (destination / name).read_bytes() == original.read_bytes(), name
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=checkout)
    print(json.dumps({'status': 'verified', 'bundle_id': BUNDLE_ID, 'code_sha': CODE_SHA,
                      'exact_inputs': len(inputs), 'byte_identical_artifacts': len(artifacts),
                      'supporting_bindings': len(index['supporting_evidence']),
                      'model_inference_started': False}, sort_keys=True))


if __name__ == '__main__':
    main()
