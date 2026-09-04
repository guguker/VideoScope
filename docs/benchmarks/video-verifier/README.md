# Video verifier regression seed

`seed-v1.json` is the first strict, content-addressed input set for the direct
video verifier. It contains ten manually reviewed cases from two private assets:
four basketball fact cases and six generic visual/OOD cases. Source files and
prepared MP4 files are intentionally absent from Git.

`exploratory-ab-2026-08-23.json` preserves the sanitized 9B/2B/27B observations,
including runtime/model revisions, prepared-input hashes, inference-only latency,
Metal memory and both 27B OOM attempts. Its status is explicitly
`exploratory_confounded_not_for_promotion`: the temporary runner is hashed but was
not committed when executed, and only two previously seen positives were scored.
Completed outputs use `facts_subset`: the retained summary omitted `shot_attempt`
and other judgement fields, so the evidence does not silently turn them into nulls.

Every case is marked `regression_seen`. These videos were already inspected while
developing the system, so this dataset must never be reported as validation,
promotion holdout, or evidence of generalization. Whole source assets are bound to
one split group by the schema. Partial interval overlap is rejected; an exact source
interval may repeat only for a protocol A/B with distinct preparation protocols and
identical labels.

Overlap is checked on declared source intervals before protocol-specific timestamp
quantization. Because an entire source asset is locked to one split, a codec-level
boundary-frame overlap cannot cross splits; prepared-input hashes remain the exact
inference identity. Any future metric that treats boundary frames as independent
samples must additionally validate the prepared media timeline.

Canonical `seed-v1` revision:
`2758ce6fd86c4ad921ae29bff83126a31a5ddbfabdca79748c58fa3d42f1763b`.

## Privacy and portability

- Public aliases replace local video IDs and filenames.
- No local paths, transcript text, personal names, frames, or raw/free-form model
  output text are committed; exploratory evidence contains only sanitized normalized
  summaries.
- Source and prepared-input SHA-256 values identify the owner's local files; they
  do not grant permission to redistribute those files.
- `LicenseRef-All-Rights-Reserved` records the media rights status. The workspace
  owner's publication request covers this metadata/evaluation use; it does not
  grant media rights or permit redistribution of source or prepared media.

## Verifying and best-effort regenerating prepared inputs

Resolve each source by its SHA-256, then run the pinned command template from the
manifest with the case's source interval. The implementation rounds the start and
duration to three decimals, uses FFmpeg 8.1.2/Homebrew revision 1, libx264
`veryfast`/CRF 20, optional source audio encoded as AAC, and `+faststart`.

The prepared-input byte size and SHA-256 are authoritative for verifying an existing
local input. The command is sufficient for best-effort regeneration on the captured
host, but it is not a cross-host byte-reproducibility claim: the current identity
does not attest FFmpeg's dynamic codec dependency closure, CPU dispatch, or thread
scheduling. A non-matching output is a different benchmark input and must receive a
new protocol/dataset version. A future cross-host protocol must also pin those
dependencies and execution parameters, or use a redistributable prepared fixture.

Validate and calculate the canonical dataset revision with the Python contract:

```bash
.venv/bin/python -I - <<'PY'
from pathlib import Path

from videoscope.benchmark import (
    video_verifier_dataset_from_json,
    video_verifier_dataset_revision,
)

path = Path("docs/benchmarks/video-verifier/seed-v1.json")
dataset = video_verifier_dataset_from_json(path.read_bytes())
print(video_verifier_dataset_revision(dataset))
PY
```

## Strict direct runner

The committed direct runner executes the frozen prepared inputs without entering
the product search pipeline and without calling the fallback-capable reranker
method. A local candidate manifest binds exactly one proposal to every frozen
case; ranks must be unique and contiguous, and its `dataset_revision` must equal
the canonical dataset revision:

```json
{
  "schema_version": 1,
  "dataset_revision": "<64-character dataset SHA-256>",
  "candidates": [
    {
      "case_id": "sports-made-three-blue-15",
      "candidate_id": "proposal-blue-15",
      "prepared_input_path": "/absolute/private/path/blue-15.mp4",
      "proposal_rank": 1,
      "proposal_score": 0.91
    }
  ]
}
```

Repeat the candidate object for every case in the dataset.

The path is a local binding only. Before and after inference the runner verifies
the prepared file against the case byte size and SHA-256. The output contains
the content identity, candidate identity, normalized typed facts, latency and a
derived binding revision, but never the local path, query/description, free-form
model evidence, bearer token or raw exception.

With the isolated Qwen worker already running, invoke it as follows (the exact
model identity must match the worker health contract):

```bash
export VIDEOSCOPE_VERIFIER_BENCHMARK_KEY='<same bearer token as the worker>'
.venv/bin/python -I \
  -m videoscope.benchmark.video_verifier_runner run \
  --dataset docs/benchmarks/video-verifier/seed-v1.json \
  --candidates /absolute/private/path/candidates.json \
  --output /absolute/private/path/run-v1.json \
  --run-id qwen-9b-regression-v1 \
  --code-sha "$(git rev-parse HEAD)" \
  --endpoint http://127.0.0.1:8781 \
  --input-root /absolute/path/to/VIDEOSCOPE_DATA_DIR/tmp \
  --model-identity \
    mlx-community/Qwen3.5-9B-MLX-4bit@938d8919941c6e7efd3c7150eff7fe9d12afa631 \
  --api-key-env VIDEOSCOPE_VERIFIER_BENCHMARK_KEY
```

Publication is create-once: an existing result is never replaced. A valid but
wrong typed prediction is `model_miss`; an unavailable worker, unsupported
input/prompt pair, corrupt or changed prepared input, invalid response contract,
or inference exception is `infrastructure_error`. Any infrastructure error makes
the run `infrastructure_failed` and the CLI exits with code 8 after preserving
the sanitized attempt records.

The versioned `qwen-worker-v4` strict contract supports `basketball_facts` on a
native MP4, and `generic_visual` on either a prepared JPEG storyboard or a native
MP4. Native generic requests bind the exact frozen query and FPS in the typed
request; every request also binds the case's frozen prepared-input SHA-256 and
byte size. The worker performs inference only on a private read-only copy made
from a retained `O_NOFOLLOW` descriptor and fails closed on source namespace
drift. Storyboard requests bind the same query and forbid FPS. Therefore all ten
native inputs in `seed-v1.json` execute through the direct worker boundary. An
unsupported prompt/input combination is still an infrastructure error and is
never silently converted into a model miss or an untracked derived input.

The next meaningful milestone is a separate set of new whole videos for
train/validation/promotion. The current ten cases are a regression harness only.
