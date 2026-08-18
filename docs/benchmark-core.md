# Executable benchmark core

The benchmark package is the local, portable layer between the legacy
single-interval evaluator and a future experiment platform. It does not expose
an API/UI, download models, or run against a user's library by itself.

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

## Frozen profiles and search adapter

The approved v1 profiles are a cumulative product ablation followed by one
mutually exclusive reranker fork:

| Profile | Exact search plan | Required capabilities |
| --- | --- | --- |
| `lexical_qdrant` | lexical plus semantic speech/OCR/object search | text vectors |
| `dense_siglip` | previous plan plus dense SigLIP visual search | previous plus dense visual index |
| `temporal_refinement` | previous plan plus temporal refinement | previous plus temporal refiner |
| `lighthouse` | previous plan plus Lighthouse | previous plus Lighthouse |
| `qwen_verification` | Lighthouse plan, then Qwen over the top 12 base candidates | Lighthouse base plus Qwen |
| `internvideo` | Lighthouse plan, then InternVideo over the top 4 base candidates | Lighthouse base plus InternVideo |

Qwen and InternVideo are never chained. Every profile has result limit 20. Text
weights remain fixed in every profile (`speech=1.0`, `ocr=0.88`,
`objects=0.92`); dense visual has weight `1.08`, and Lighthouse has weight
`0.78` when present. The `all_candidates` trigger means unconditional reranking
of every candidate inside the frozen reranker cap: 12 for Qwen and 4 for
InternVideo, not all 20 final result slots. Modalities, weights, refinement
flags, reranker trigger/candidate limit, and result limit are part of canonical
`EvaluationSearchPlan` JSON and its SHA-256 identity. Provider input, frame, and
token settings remain part of the pinned runtime/model identity. Input tuple
order cannot change the plan identity.

`BenchmarkSearchAdapter` is a session factory, not a stateless collection of
callbacks. After local assets are resolved, the runner calls exactly one
`open_session(profile, resolved_assets)`. That session must pin the assets and
exact model/index/config generations for the whole run, then expose
`identities`, `capability_state`, `search`, and `close`. A future concrete
adapter must call the fail-closed evaluation boundary and translate local hits
back to portable aliases inside `search` or its bounded iterable. Missing,
failed, stale, not-configured, incomplete, invalid, or exceptional dependencies
become infrastructure errors; they are never scored as model misses. Session
close is mandatory even after an error, and a close failure prevents
publication.

## Safe foundation CLI

The stdlib CLI exposes validation, import, registry inspection, audit, and
comparison without loading a model or touching a user's media library. It is
available as `python -m videoscope.benchmark` and, after package installation,
as `videoscope-benchmark`:

```text
python -m videoscope.benchmark dataset validate <file>
python -m videoscope.benchmark dataset import <file> --catalog <root>
python -m videoscope.benchmark runs list --registry <root>
python -m videoscope.benchmark runs show <run_id> --registry <root>
python -m videoscope.benchmark runs audit <run_id> --registry <root> --dataset <file>
python -m videoscope.benchmark compare <baseline> <candidate> \
  --registry <root> --policy <policy.json>
```

Successful commands write one compact, key-sorted JSON object to stdout. Errors
write a generic JSON object to stderr without a traceback, input path, or raw
storage/provider exception. Stable exit codes are `2` for usage, `3` for invalid
or corrupt data, `4` for missing data, `5` for an existing destination, `6` for
an audit failure, `7` for storage I/O, and `70` for an unexpected internal
failure. Registry reads require an existing registry and do not initialise one;
`dataset import` is the only state-creating command in this CLI.

Dataset and policy inputs must be bounded regular files. Symbolic links and
special files such as FIFOs are rejected without following or waiting on them.

Promotion policy JSON is strict: duplicate or unknown fields, non-finite values,
unsupported versions, symbolic links, oversized files, and more than 256
guardrails are rejected. Every guardrail declares its complete decision gate:

```json
{
  "schema_version": 1,
  "policy_id": "retrieval-core",
  "minimum_completed_cases": 20,
  "require_no_errors": true,
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

There is deliberately no `run` command yet. A concrete adapter must first bind
the frozen profiles to `SearchService.search_for_evaluation`, active capability
identities, and explicit local asset bindings. A command that silently used
normal search or guessed those bindings would produce plausible but
untrustworthy runs.

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
persisted per-case latency. It does not claim to validate future process
measurements.

Run schema v2 includes dataset revision, code SHA, model/index/config
identities, hardware, cold/warm mode, outcomes, ranked evidence, and explicit
run start/finish/manifest timestamps. `run_status` is derived from outcomes as
`complete`, `partial`, `failed`, or `cancelled`.

`quality_metrics` contains the outcome-recomputable aggregates. The existing
`mean_latency_ms` compatibility aggregate remains there because it is exactly
recomputed from persisted complete-case latency. `system_metrics` is a separate
namespace for a future managed measurement protocol; metric names cannot
overlap the two namespaces. `measurement_status` is `not_measured`, `complete`,
or `failed`, and measured runs require their own start/finish timestamps inside
the run interval. Foundation runs use the explicit `not-measured@1` sentinel,
contain no system metrics or measurement timestamps, and therefore do not
pretend that process-tree memory or storage was measured.

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
duration. This increment records retrieval latency and the hardware identity,
but it does **not** yet sample peak memory or index/storage growth. Those system
metrics require a concrete local runner adapter with explicit measurement
boundaries (process tree, cache policy, and before/after artifact roots). They
remain a follow-up alongside the concrete search adapter/run command and must be
added before claiming the full latency/memory/storage benchmark promised by
`system-rebuild.md`.

## Comparison and promotion guardrails

`PromotionPolicy` is immutable and declares a minimum completed-case count, an
infrastructure-error policy, and per-metric direction, allowed regression, and
absolute threshold. A guardrail may address either quality or system metrics.
Comparison is allowed only for the same dataset revision, methodology,
hardware, cold/warm mode, case identities, measurement protocol, and
measurement status. Failed measurements, legacy measurement contracts, and
cancelled/failed runs are incomparable. Search-plan identities may differ: that
declared difference is the ablation being compared. For comparable runs,
machine-readable output includes every declared gate and sorts checks
deterministically; incomparable results may omit guardrail checks.

The statuses are `eligible`, `reject`, `insufficient_evidence`, and
`incomparable`. `eligible` means only that the declared deterministic guardrails
passed. This core does not run a significance test and always emits
`statistical_significance: not_assessed`; a sample below the policy minimum can
never be promoted.
