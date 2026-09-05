# Benchmark core

The benchmark package is the local, portable layer between the legacy
single-interval evaluator and a future experiment platform. It includes a safe
inspection/comparison CLI plus one fail-closed product execution path. It does
not expose an API/UI, download models, initialise product state, or build an
index during a benchmark.

## Portable dataset and local binding

`DatasetCatalog` validates a dataset manifest before creating catalog state,
computes its order-independent revision, and imports it at
`<dataset_id>/<revision>.json` without overwriting an existing revision.
Catalog roots, dataset directories, and manifests reject symbolic-link escapes.

A manifest `asset_id` is a stable dataset alias. It is deliberately different
from the repository Asset id, whose authoritative form is
`sha256:<content-digest>`. `LocalAssetResolver` looks up repository rows by the
manifest SHA-256, then verifies the content id, byte size, and duration. If the
same bytes belong to multiple local video rows, the resolver fails with
`asset_binding_ambiguous` unless the caller supplies an explicit alias-to-video
binding. No local path or private video id is written into the portable dataset.

The repository dependency is a small read-only Protocol. The concrete
`Repository.find_assets_by_sha256` bridge can satisfy it without coupling the
benchmark schema to SQLite records.

Dataset manifests are capped at 128 assets, 16 GiB per asset, and 128 GiB of
aggregate declared media. A pinned search session applies aggregate caps of
200,000 text-vector points, 200,000 segments, 64 MiB of segment text, and 64 MiB
of segment metadata. Crossing a cap fails preflight/execution; it never silently
truncates the evaluated corpus.

## Frozen profiles and search adapter

The approved profile-schema-v2 profiles are a cumulative product ablation
followed by one mutually exclusive reranker fork:

| Profile | Exact search plan | Required capabilities |
| --- | --- | --- |
| `lexical_qdrant` | lexical plus semantic speech/OCR/object search | text vectors |
| `dense_siglip` | previous plan plus dense SigLIP visual search | previous plus dense visual index |
| `temporal_refinement` | previous plan plus temporal refinement | previous plus temporal refiner |
| `lighthouse` | previous plan plus Lighthouse | previous plus Lighthouse |
| `qwen_verification` | Lighthouse plan, then Qwen over the top 12 base candidates | Lighthouse base plus Qwen |
| `internvideo` | Lighthouse plan, then InternVideo over the top 4 base candidates | Lighthouse base plus InternVideo |

Qwen and InternVideo are never chained. Every profile has result limit 50. Text
weights remain fixed in every profile (`speech=1.0`, `ocr=0.88`,
`objects=0.92`); dense visual has weight `1.08`, and Lighthouse has weight
`0.78` when present. The `all_candidates` trigger means unconditional reranking
of every candidate inside the frozen reranker cap: 12 for Qwen and 4 for
InternVideo, not all 50 final result slots. Modalities, weights, refinement
flags, reranker trigger/candidate limit, and result limit are part of canonical
`EvaluationSearchPlan` JSON and its SHA-256 identity. Provider input, frame, and
token settings remain part of the pinned runtime/model identity. Input tuple
order cannot change the plan identity. A profile identity is also content-bound:
its canonical payload includes the profile id and schema, the sorted required
capabilities, and the exact search-plan identity. Changing capability
requirements without changing the profile identity is therefore impossible.

`BenchmarkSearchAdapter` is a session factory, not a stateless collection of
callbacks. After local assets are resolved, the runner calls exactly one
`open_session(profile, resolved_assets)`. That session must pin the assets and
exact model/index/config generations for the whole run, then expose
`identities`, `lifecycle_identity`, `capability_state`, `search`, and `close`.
The concrete `ProductBenchmarkSearchAdapter` now translates every frozen plan
exactly into `SearchService.open_pinned_evaluation`, binds portable aliases to
verified local assets, and translates local hits back to those aliases. The
product session pins source fingerprints, relevant artifact generations and
provider/configuration attestations, checks them during the run, and performs a
full final verification on close. Missing, failed, stale, not-configured,
incomplete, invalid, or exceptional dependencies become infrastructure errors;
they are never scored as model misses. Session close is mandatory even after an
error, and a close failure prevents publication.

The default lifecycle honestly supports only warm execution with preserved
process caches. Cold execution requires a separate externally attested
lifecycle and is rejected without one. The product environment accepts every
frozen profile and builds read-only wiring for dense SigLIP, temporal refinement,
Lighthouse, and Qwen from the selected settings and existing generation state;
it never indexes, activates, or falls back to another provider. InternVideo is
the explicit exception: the profile is accepted, but the provider remains
machine-readably `not_configured` until it has the same local, source-bound
attestation contract.

## Read-only benchmark CLI

The stdlib CLI exposes validation, import, registry inspection, audit,
comparison, and one read-only product benchmark. It is available as
`python -m videoscope.benchmark` and, after package installation, as
`videoscope-benchmark`:

```text
python -m videoscope.benchmark dataset validate <file>
python -m videoscope.benchmark dataset import <file> --catalog <root>
python -m videoscope.benchmark runs list --registry <root>
python -m videoscope.benchmark runs show <run_id> --registry <root>
python -m videoscope.benchmark runs audit <run_id> --registry <root> --dataset <file>
python -m videoscope.benchmark compare <baseline> <candidate> \
  --registry <root> --policy <policy.json> --dataset <file>
python -m videoscope.benchmark run --dataset <file> --registry <existing-root> \
  --data-dir <product-data> --scratch-parent <existing-0700-directory> \
  --run-id <id> [--bindings <alias-to-video.json>] \
  [--profile <frozen-profile-id>] [--execution-mode warm] [--preflight]
```

Successful commands write one compact, key-sorted JSON object to stdout. Errors
write a generic JSON object to stderr without a traceback, input path, or raw
storage/provider exception. Stable exit codes are `2` for usage, `3` for invalid
or corrupt data, `4` for missing data, `5` for an existing destination, `6` for
an audit failure, `7` for storage I/O, `8` for product execution/preflight,
`9` for unavailable or failed measurement/preflight measurement, `70` for an
unexpected internal failure, `130` for an interrupt, and `143` for termination.
Registry reads and benchmark runs require an already initialised registry; they
never initialise one.

Dataset and policy inputs must be bounded regular files. Symbolic links and
special files such as FIFOs are rejected without following or waiting on them.

Promotion policy JSON is strict: duplicate or unknown fields, non-finite values,
unsupported versions, symbolic links, oversized files, and more than 256
guardrails are rejected. Every guardrail declares its complete decision gate:

```json
{
  "schema_version": 2,
  "policy_id": "retrieval-core",
  "minimum_completed_cases": 20,
  "require_no_errors": true,
  "allowed_differences": {
    "allow_code_sha_difference": false,
    "allow_benchmark_profile_difference": true,
    "allow_benchmark_search_plan_difference": true,
    "model_component_ids": [],
    "index_component_ids": [],
    "config_component_ids": []
  },
  "guardrails": [
    {
      "metric_name": "recall_at_5",
      "direction": "higher_is_better",
      "allowed_regression": 0.0,
      "absolute_threshold": 0.8
    }
  ]
}
```

Schema v2 requires the complete `allowed_differences` object. Code SHA, profile,
and search-plan changes each have a dedicated boolean; model, index, and config
identity changes require exact component-id allowlists. Missing fields,
duplicates, unknown fields, and reserved benchmark component ids in the generic
config allowlist are rejected. An empty/default object therefore means that all
execution identities must match exactly.

The executable path is deliberately narrow in a different sense: all six frozen
profiles are accepted, but only in `warm` mode and only against existing product
state. It opens the product database, FastEmbed model, Qdrant state, applicable
visual/Lighthouse generations, and configured loopback providers through
retained read-only boundaries. It never calls normal mutable runtime
construction, model download, schema initialisation, migration, or indexing. If
`--bindings` is supplied, it must be a bounded JSON object whose keys exactly
equal all dataset aliases and whose values are explicit local video ids;
otherwise content identity resolution must be unambiguous.

`--preflight` resolves every asset, opens the selected profile's pinned session,
captures model, index, configuration, lifecycle, and environment identities,
and returns the complete per-asset capability matrix. Every required capability
must be `complete` for status `ready`; `not_configured`, missing, stale, failed,
queued, or running states remain visible and are never converted to misses. It
performs no search and does not change registry bytes. A real run is first
prepared in memory, then the complete environment is closed and finally
attested, and only then is the manifest published. Partial, failed, or cancelled
runs, failed configured measurements, cleanup failures, and final attestation
failures are never published.

The CLI measurement boundary is independently fail-closed. The lexical profile
can bind its local process tree. Profiles using an already running Vision,
Lighthouse, Qwen, or InternVideo endpoint currently return measurement status
`not_configured` with reason
`external_worker_process_binding_unavailable`, because a health response does
not prove the endpoint PID, start token, or executable identity. Such a
preflight exits `9`, even when the provider capability matrix itself is ready;
an environment/capability failure with a ready measurement exits `8`.

Both preflight and execution require an exact clean Git `HEAD`. Any tracked
change or relevant untracked source, worker, script, configuration, or lock file
fails closed. The same identity is checked again after environment cleanup and
before success/publication, so code drift during a run cannot be recorded under
the earlier commit. The manifest also records a deterministic snapshot of the
current OS, architecture, processor, physical memory, and accelerator class.

## Read-only product and model snapshots

`Repository.open_read_only` opens only an existing database at the exact current
schema with SQLite query-only behavior; it never creates or migrates a database.
`open_product_runtime_snapshot` first acquires the existing exclusive product
lock, verifies a stable no-follow database/WAL/SHM view, copies it into private
scratch, records a content identity without source paths, and opens only that
copy. Source media stays outside the snapshot and is verified against the
portable binding.

FastEmbed and Qdrant have separate bounded, no-follow private snapshot paths.
The FastEmbed copy is restricted to the reviewed byte manifest for the pinned
model, and Qdrant is opened as an existing read-only snapshot whose generation
payloads are validated for benchmark use. The FastEmbed path additionally
rejects files outside its exact allowlist. Source/scratch overlap,
symbolic-link escapes, unstable inputs, and incomplete cleanup fail closed.
The CLI composes these owners in a fixed order and does not publish until every
owner reports a complete close.

## Run semantics and auditability

The runner handles cases with zero, one, or several relevant intervals and
optional hard negatives. Quality metrics use only completed positive cases as
the Recall/MRR/IoU denominator. Zero-interval cases contribute to the explicit
negative false-positive metric. Result-level false positives, hard-negative
hits, per-case latency, and domain/modality/label-quality/split-group slices are
recorded separately.

Each complete case persists at most the profile result limit of ranked portable
evidence: rank, asset alias, interval, and normalized score. It never persists a
local video id, media path, or provider exception. Provider iterables are read
only through `limit + 1`, exact duplicate hits are rejected, and failed outcomes
contain no partial evidence. `audit_run_manifest` requires the exact current
profile, search-plan, and methodology identities, then recomputes per-case and
aggregate quality metrics from the frozen dataset, ranked evidence, and
persisted per-case latency. For a measured v2 run it also reconstructs every
system aggregate from the persisted portable raw measurement evidence.

Run schema v2 includes dataset revision, code SHA, model/index/config
identities, hardware, cold/warm mode, outcomes, ranked evidence, and explicit
run start/finish/manifest timestamps. `run_status` is derived from outcomes as
`complete`, `partial`, `failed`, or `cancelled`.

`quality_metrics` contains the outcome-recomputable aggregates. The existing
`mean_latency_ms` compatibility aggregate remains there because it is exactly
recomputed from persisted complete-case latency. `system_metrics` is a separate
namespace; metric names cannot overlap the two namespaces. `measurement_status`
is `not_measured`, `complete`, or `failed`, and measured runs require their own
start/finish timestamps inside the run interval. Unmeasured runs use the
explicit `not-measured@1` sentinel, contain no system metrics or measurement
timestamps, and therefore do not pretend that process-tree memory or storage
was measured.

The managed measurement implementation is all-or-nothing. On Darwin it uses the
native `libproc` provider, not `ps`, to sample the identity-pinned recursive
process tree at 50 ms. It persists the bounded raw RSS byte samples from which
sample count, baseline, peak, and increment are recomputed, plus bounded
descriptor-relative storage snapshots before and after the run. Declared active
immutable roots must remain unchanged; only declared benchmark scratch may grow.
An unavailable provider, unbound external worker, identity drift, storage drift,
or cleanup failure produces a failed measurement with no partial system metrics.
Metal is explicitly `unavailable` in this benchmark-run evidence contract.

`BenchmarkRunRegistry` publishes v2 atomically in an immutable run directory
and refuses duplicate run ids. The loader still accepts the exact strict v1
JSON shape. It represents that value in memory as v2 with the
`legacy-unmeasured@1` sentinel and conservative equal start/finish/creation
timestamps; reads and registry rebuilds never rewrite the immutable v1 bytes.
All new writes are v2. Legacy runs lack the frozen search-plan and measurement
contract, so they cannot pass the current audit or promotion comparison.

The search timer starts immediately before `session.search` and stops only
after bounded `limit + 1` materialization and portable hit translation. Asset
binding, capability preflight, identity capture, scoring, audit work, registry
I/O, and session open/close are outside that boundary. Capability/binding
failures therefore have zero search latency rather than a misleading preflight
duration. The managed measurement surrounds the pinned search session and
finishes before that session is closed. Its protocol identity, raw RSS samples,
and storage snapshots are stored in the run manifest; audit recomputes the
system metrics and rejects missing or inconsistent evidence. Promotion policy
still deliberately rejects observed system-metric guardrails in the current
comparison layer, so resource acceptance remains a separate Phase 0 gate rather
than an implicit quality-promotion check.

## Offline full-ML smoke

`scripts/full-ml-smoke.py` is the separate Phase 0 product-path proof. It requires
an explicit existing mode-0700 disposable root outside the checkout and home
directory, an explicit non-overlapping read-only model root outside the
checkout, offline flags, exact manifest-bound owner/worker Python environments,
and an explicitly attested FFmpeg/FFprobe pair. Worker `PATH`, `HOME`, and
`TMPDIR` are deterministic. It self-starts Vision, Whisper, Lighthouse, and Qwen as authenticated random-port
loopback descendants; OCR is a bounded child process. It never connects to
prestarted workers, downloads artifacts, or writes to project `data/`.

Against a synthetic local MP4, the smoke performs real SigLIP image/text
embedding, RF-DETR detection, Whisper transcription, OCR, Lighthouse
generation/search, and Qwen judging. It then uses production components for
upload, durable indexing, generation-pinned search through
`lexical_qdrant`, `dense_siglip`, `temporal_refinement`, `lighthouse`, and
`qwen_verification`, inspectable evidence, and a real MP4 export verified by
FFprobe. Each executed profile must report successful open/search/close,
`generation_bound=true`, and non-empty evidence. `internvideo` must instead
report `not_configured/provider_not_configured`; it is not silently substituted.

The sanitized schema-v2 receipt binds the same pinned ML-environment manifest
and path-free role mapping as the standalone attestation and benchmark batch. It
also binds each profile to an identity-owned trace of the exact search
components that ran and contains bounded raw process-tree RSS samples
including every managed loopback worker, plus native host-resource raw samples,
Metal in-use/allocated peaks and recovery delta, and VM swap deltas. The Metal
and VM measurements are `system_wide` and explicitly non-additive with
process-tree RSS; they are not per-worker attribution. OOM is reported separately
from configuration, infrastructure, and contract failures. Exact provisioning
and invocation are documented in
[`ml-environment-attestation.md`](ml-environment-attestation.md). This smoke has
not yet been executed as target-M4 promotion evidence, so it does not by itself
complete Phase 0.

## Comparison and promotion guardrails

`PromotionPolicy` is immutable and declares a minimum completed-case count, an
infrastructure-error policy, the exact allowed execution-identity differences,
and per-metric direction, allowed regression, and absolute threshold. Nothing
is implicitly treated as an ablation: code, profile, search plan, and individual
model/index/config components may differ only when policy schema v2 names that
difference exactly.

Comparison first evaluates deterministic compatibility. Different dataset
revisions, hardware, cold/warm modes, measurement contracts/statuses,
methodology or lifecycle identities, case identities, exact completed-case
cohorts, and undeclared execution identities return `incomparable` with no
guardrail checks. Failed measurements, legacy measurement contracts, and
cancelled/failed runs are also incomparable. This result is deterministic and
does not attempt to audit one supplied dataset against already-incompatible run
revisions.

Every compatible comparison requires a `BenchmarkDataset`; the public
`compare_runs` entry point and CLI audit the portable ranked evidence and
recompute quality metrics for both baseline and candidate before evaluating any
guardrail. It also recomputes system aggregates for measured v2 runs from their
raw evidence. Forged or internally inconsistent aggregates therefore fail audit
instead of becoming promotion-eligible. Observed system-metric guardrails still
fail closed under the current comparison policy and produce an audit error;
quality guardrails remain available. Machine-readable successful comparisons
include every declared gate and sort checks deterministically; incomparable
results may omit guardrail checks.

The statuses are `eligible`, `reject`, `insufficient_evidence`, and
`incomparable`. `eligible` means only that the declared deterministic guardrails
passed. This core does not run a significance test and always emits
`statistical_significance: not_assessed`; a sample below the policy minimum can
never be promoted.
