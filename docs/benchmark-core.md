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

The approved versioned profiles are:

1. `lexical_qdrant`
2. `dense_siglip`
3. `temporal_refinement`
4. `lighthouse`
5. `qwen_verification`
6. `internvideo`

`BenchmarkSearchAdapter` is the explicit boundary to product search. A concrete
adapter must snapshot exact model/index/config identities, report each required
capability state, call the fail-closed evaluation search path, and translate
local results back to portable asset aliases. Missing, failed, stale,
not-configured, incomplete, or exceptional dependencies become infrastructure
errors; they are never scored as model misses.

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
hits, latency, and domain/modality/label-quality/split-group slices are recorded
separately.

Each complete case persists at most the profile result limit of ranked portable
evidence: rank, asset alias, interval, and normalized score. It never persists a
local video id, media path, or provider exception. Provider iterables are read
only through `limit + 1`, exact duplicate hits are rejected, and failed outcomes
contain no partial evidence. `audit_run_manifest` recomputes per-case and
aggregate metrics from the frozen dataset and ranked evidence.

The finished `BenchmarkRunManifest` includes dataset revision, code SHA,
model/index/config identities, hardware, cold/warm mode, metrics, outcomes, and
ranked evidence. `BenchmarkRunRegistry` publishes it atomically in an immutable
run directory and refuses duplicate run ids.

This increment measures retrieval latency and records the hardware identity, but
it does **not** yet sample peak memory or index/storage growth. Those system
metrics require a concrete local runner adapter with explicit measurement
boundaries (process tree, cache policy, and before/after artifact roots). They
remain a follow-up alongside the concrete search adapter/run command and must be
added before claiming the full latency/memory/storage benchmark promised by
`system-rebuild.md`.

## Comparison and promotion guardrails

`PromotionPolicy` is immutable and declares a minimum completed-case count, an
infrastructure-error policy, and per-metric direction, allowed regression, and
absolute threshold. Comparison is allowed only for the same dataset revision,
methodology, hardware, cold/warm mode, and case identities. For comparable runs,
machine-readable output includes every declared gate and sorts checks
deterministically; incomparable results may omit guardrail checks.

The statuses are `eligible`, `reject`, `insufficient_evidence`, and
`incomparable`. `eligible` means only that the declared deterministic guardrails
passed. This core does not run a significance test and always emits
`statistical_significance: not_assessed`; a sample below the policy minimum can
never be promoted.
