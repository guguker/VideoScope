# Phase-0 product-retrieval regression fixture

`seed-v1.json` is the portable product-level projection of the ten prepared
clips already identified by `../video-verifier/seed-v1.json`. It contains only
content identities, public-safe provenance, queries, intervals and slice labels.
It deliberately contains neither media, local paths nor private speech text.

The frozen identity is:

- dataset: `video-verifier-product-regression@1.0.0`;
- schema: `BenchmarkDataset@1` with `CriticalSliceLabels@1`;
- revision: `8cda261148ce3f459e08c3762017cec86970ab7c85d36eaa77ef9f832a2ae6a1`;
- scope: 10 prepared assets, 5 retrieval cases and 2 whole-source groups;
- evidence use: `regression_only`; promotion eligibility: `false`.

All related windows remain in the same `split_group`. Every critical case is
labelled `regression_seen`. The positive ranges are exact prepared-window
boundaries inherited from the verifier fixture; they are not newly adjudicated
event boundaries and must not be presented as independent promotion evidence.

## Private local binding

Create a local manifest outside version control that binds every public alias to
the exact prepared clip already present on the machine:

```json
{
  "schema_version": 1,
  "dataset_revision": "8cda261148ce3f459e08c3762017cec86970ab7c85d36eaa77ef9f832a2ae6a1",
  "inputs": [
    {
      "asset_id": "panel-light-top-gesture-negative",
      "path": "/absolute/private/location/panel-light-top-gesture-negative.mp4"
    }
  ]
}
```

The real manifest must contain exactly all ten aliases. Do not commit it. The
provisioner rejects a symbolic-link manifest, relative or missing paths, alias or
revision drift, changed bytes, a duration difference greater than 50 ms, and a
non-empty or unsafe destination.

## Managed five-profile baseline batch

The public command owns the complete Phase-0 lifecycle: prevalidation, byte-exact
staging, five local worker processes, production indexing, retirement of the
ingestion-only Vision and Whisper processes, five warm measured benchmark runs,
manifest audit and worker cleanup. It neither downloads nor trains anything and
does not accept caller-supplied PIDs as trusted measurement bindings.

First create a private worker-launch manifest outside version control. Every
path must be absolute and already exist. The five Python executables must point
lexically to this checkout's exact manifest directories; an otherwise compatible
substitute virtual environment is rejected. `ffmpeg` and `ffprobe` must be
executable bindings with exactly those names and resolve into one canonical
directory. Homebrew aliases are resolved once; that canonical directory becomes
every worker's entire `PATH`, so an ambient shell `PATH` cannot select another
media toolchain. The OCR child also receives an explicit HOME/TMPDIR/cache/PATH
allowlist, and its copied worker bundle and frames stay under the disposable
data root.

```json
{
  "schema_version": 1,
  "executables": {
    "vision": "/absolute/path/to/VideoScope/.venv-vision/bin/python",
    "whisper": "/absolute/path/to/VideoScope/.venv-whisper/bin/python",
    "lighthouse": "/absolute/path/to/VideoScope/.venv-lighthouse/bin/python",
    "qwen": "/absolute/path/to/VideoScope/.venv-qwen/bin/python",
    "ocr": "/absolute/path/to/VideoScope/.venv-ocr/bin/python"
  },
  "hf_home": "/absolute/reviewed/offline-hf-cache",
  "ocr_model_root": "/absolute/reviewed/offline-ocr-models",
  "ffmpeg_binary": "/opt/homebrew/bin/ffmpeg",
  "ffprobe_binary": "/opt/homebrew/bin/ffprobe"
}
```

The data root must be nonexistent or empty. The registry must be nonexistent or
empty, while the scratch parent must already be an empty owner-only (`0700`)
directory. Data, scratch and registry must be disposable locations outside both
the checkout and the user's home. The reviewed model, Hugging Face and OCR roots
must be read-only inputs outside the checkout. All six roots must be mutually
disjoint. The Git worktree must be clean, because every run is bound to one exact
commit SHA. The owner process must itself be the same checkout's
`.venv/bin/python`. Before staging, the command requires the pinned
`workers/ml-environment.lock.json` attestation to be fully complete; a partial
report, missing environment, distribution drift or host mismatch is fatal.

```bash
cd backend
env \
  HF_DATASETS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  TRANSFORMERS_OFFLINE=1 \
  UV_OFFLINE=1 \
  ../.venv/bin/python -I -m videoscope.benchmark.regression_fixture batch \
  --dataset ../docs/benchmarks/product-retrieval/seed-v1.json \
  --bindings /absolute/private/product-regression-bindings.json \
  --policy ../docs/benchmarks/policies/phase0-regression-v1.json \
  --data-root /private/tmp/videoscope-phase0-product \
  --models-root /absolute/reviewed/videoscope-models \
  --scratch-parent /private/tmp/videoscope-phase0-scratch \
  --registry /private/tmp/videoscope-phase0-registry \
  --worker-launch /absolute/private/phase0-worker-launch.json \
  --run-id-prefix phase0-baseline-20260904
```

All configuration and source checks finish before either the output roots are
created or private media is copied. Success writes these path-free evidence
artifacts:

- `benchmark-bindings.json` and `provision-receipt.json` in the data root;
- five complete run manifests in the registry, in the fixed order
  `lexical_qdrant`, `dense_siglip`, `temporal_refinement`, `lighthouse`, then
  `qwen_verification`;
- `phase0-baseline-receipt.json` in the data root, with exactly the code,
  dataset, policy and pinned ML-environment identities; the fixed path-free
  owner/worker-to-environment mapping; the ordered five
  `{profile_id, run_id, manifest_sha256}` bindings; and complete retirement and
  cleanup status. The referenced run manifests carry the frozen profile, model,
  measurement and threshold evidence.

The process-tree measurement covers only the owner process tree plus the still
live benchmark workers (`vision`, `lighthouse`, `qwen`). The ingestion-only
`vision_index` and `whisper` processes are stopped and removed from measurement
bindings before the first preflight. Every run must be warm, raw-measurement
complete and audited before the final receipt is published. Any failure leaves
the fixture non-promotable and still attempts owned-worker cleanup.

The command emits one path-free JSON object. Exit status is `0` on complete
success, `2` for usage/fixture/configuration failures, `8` for managed-worker or
benchmark execution failures, `9` for measurement failure, `70` for a sanitized
unexpected failure, and `130` for interruption.

## Validation

```bash
.venv/bin/pytest -q \
  backend/tests/test_benchmark_regression_fixture.py \
  backend/tests/test_benchmark_managed_workers.py
.venv/bin/python -I -m videoscope.benchmark.metric_policy \
  validate \
  --policy docs/benchmarks/policies/phase0-regression-v1.json \
  --product-dataset docs/benchmarks/product-retrieval/seed-v1.json \
  --verifier-dataset docs/benchmarks/video-verifier/seed-v1.json
```

These ten already-seen clips remain regression evidence after a successful full
batch. They cannot satisfy a promotion gate or substitute for the independent
grouped holdout required by the ML-autonomy plan.
