# VideoScope system rebuild

Status: approved implementation instruction.

Decision owner: repository owner.

Implementation branch: `codex/system-rebuild`.

## Objective

Turn VideoScope from a collection of working retrieval providers into a
reproducible local research system whose source assets, processing stages,
derived artifacts, active index generations, jobs and evaluation runs have
explicit identities and lifecycle rules.

The existing product surface remains valuable and must be preserved:

- React library, player, search results and clip workflow;
- strict local FastAPI boundary;
- provider adapters and explainable fusion;
- source videos and local user data;
- current search/export behavior unless a versioned migration deliberately
  changes it.

This is a replacement of the internal orchestration and evidence core, not a
rewrite of the whole application.

## Non-goals

- macOS `.app`, notarization, App Store packaging or auto-update;
- visual redesign, cover suggestions or decorative product features;
- adding Kimi or another large model before a benchmark demonstrates a gap;
- training a bespoke large VLM;
- deleting local source videos or evaluation labels during migration;
- exposing the primary application beyond loopback.

## Invariants

1. `data/` is user state. A migration may add metadata or rebuild derived data,
   but must not silently delete source media, labels, glossary entries or valid
   exports.
2. A new derived generation is built beside the active generation. The active
   pointer changes only after validation and an atomic commit.
3. Search never consumes an artifact whose source hash, stage specification,
   provider/model identity or schema is unknown or incompatible.
4. `ready` means that the source video is usable; every optional capability has
   its own `complete`, `failed`, `stale` or `not_configured` state.
5. Evaluation fails closed when the selected variant requires an unavailable,
   failed, stale or incomplete capability. Infrastructure failure is never a
   model miss.
6. Model and ranking changes are promoted only against a frozen, versioned
   benchmark and declared quality/system guardrails.
7. Every persisted identifier, path and manifest is validated as untrusted
   input when read. Public API responses never expose internal paths, secrets or
   raw provider exceptions.
8. Tests are written before behavior changes. Each production failure becomes a
   regression test.

## Target core model

### Asset

An immutable identity for a source video:

- stable asset id;
- SHA-256 and byte size;
- media location kept internal;
- probe metadata;
- provenance/licensing metadata when used by a portable benchmark.

The existing video id remains the product-facing library identity. An asset id
is content identity and must not depend on a local SQLite row id.

### Stage specification

A canonical, hashable description of work:

- stage kind and schema version;
- implementation revision;
- model repository and pinned revision/checksum;
- preprocessing and sampling parameters;
- dependency/runtime identity relevant to output compatibility.

Examples include `speech`, `ocr`, `objects`, `visual_dense`, `lighthouse` and
future sports tracking stages.

### Stage run

One attempt to apply a stage specification to an asset:

- queued/running/complete/failed/cancelled state;
- timestamps and sanitized diagnostic code;
- source asset hash and specification hash;
- output generation id;
- retry lineage.

### Artifact and generation

Artifacts are immutable outputs such as transcript segments, dense frame
metadata, embeddings or tracking observations. A generation groups a complete,
validated set of artifacts and has an atomic active pointer. The previous active
generation is the rollback target until retention policy removes it.

### Job

Durable orchestration for indexing, reindexing, evaluation and export:

- idempotency key and deduplication;
- progress and current stage;
- cancel/retry semantics;
- bounded concurrency;
- durable recovery after restart.

### Evaluation run

An immutable record linked to:

- dataset revision;
- code commit;
- runtime/search configuration;
- exact active artifact generations;
- model identities;
- hardware profile;
- cold/warm execution mode;
- metrics and per-case outcomes.

## First safety-critical increment: visual indexing

The audit found two incompatible writers for one SigLIP index: normal
ingest/reindex wrote scene thumbnails, while `scripts/index-visual.py` wrote a
dense fixed-step index. Both used only the model identity as the current marker,
so a normal reindex could replace a dense sports index with a sparse scene index
without being detected. The replacement contract below is now implemented.

The replacement contract is:

1. A shared `VisualIndexSpecification` defines model identity, sampling strategy,
   step, maximum frame width, preprocessing/schema revision and extractor
   identity.
2. Normal ingest, API reindex and the maintenance script call the same source-
   based dense indexing path.
3. The complete specification hash is stored in each generation manifest.
4. Legacy model-only and scene-only artifacts are stale, never current.
5. Building happens in a unique generation directory and validates vector shape,
   finite values, metadata ordering, source identity and expected specification.
6. Activation is atomic and does not remove the previous generation on failure.
7. Evaluation runtime identity includes the complete visual specification and
   active generation.

Required regressions:

- upload and API reindex use the configured dense step;
- maintenance backfill and runtime produce compatible specifications;
- changing step, frame width, model revision or schema makes the old generation
  stale;
- failed extraction/embedding/persistence preserves the previous active index;
- a scene-only legacy index cannot pass dense readiness;
- interrupted or malformed manifests fail closed.

## Capability provenance

Per-video readiness is represented independently for segment stages, text
vectors, dense visual indexing and Lighthouse. Schema v5 introduced
source/specification-linked generations for scenes, speech, OCR and objects;
schema v6 adds immutable text-vector generations with exact upstream lineage,
verified Qdrant manifests and an atomic SQLite active pointer. Schema v7 adds
global generation tombstones and leased GC jobs; schema v8 adds monotonic fenced
total attempts, a bounded window of 256 immutable recent audit rows,
indefinitely recoverable transient retries with capped backoff and quarantine for
corrupt recovery metadata. SigLIP and Lighthouse retain their own validated
immutable manifests. A unified probe-stage readiness record remains follow-up
work.

Search requirements are derived from the actual query plan. Evaluation variants
declare stronger deterministic requirements. Missing speech/OCR/object artifacts
must therefore be reported as unavailable infrastructure when the variant needs
them, rather than counted as a successful empty retrieval.

The first migration derives only facts that can be proven from existing data.
Unknown legacy provenance becomes `stale` or `unknown`; it is never guessed as
complete.

Schema v5 implements this rule for SQLite segment evidence, and schema v6 applies
the same build-before-swap rule to semantic text vectors. Legacy segment rows
stay present with no generation identity and are deliberately excluded from
search. Reindex publishes immutable per-stage generations and swaps each active
pointer only after the full candidate set (and, for scenes, its generation
directory) is ready. Failed or unavailable optional providers therefore preserve
the last proven active generation and produce explicit `failed` or
`not_configured` runs. Startup does not enqueue already-ready legacy videos or
rebuild their text evidence automatically; migration requires an explicit
per-video reindex so model work and user-visible changes remain intentional.

The API acquires an exclusive data-directory lock before migration or recovery.
Indexer and GC share one storage gate: reservation, Qdrant build and SQLite
commit/failure cleanup cannot overlap expiry or deletion. GC deletes only an
uncommitted exact generation, verifies a zero remaining point count and records
every leased attempt. Stale leases and transient storage failures return to the
bounded-backoff queue; permanent contract failures are terminal. Corrupt build
or job metadata is quarantined without exposing it as a trusted deletion target.

## Environment isolation

The target development/runtime profiles are:

- base API/search environment;
- vision worker for Torch, SigLIP and RF-DETR;
- speech worker for MLX Whisper;
- video worker for MLX/Qwen;
- OCR worker;
- Lighthouse worker or maintained minimal service.

Qwen, OCR, Lighthouse, Vision and Whisper now use isolated worker boundaries.
Workers use bounded local contracts, pinned dependencies and explicit
health/capability responses. A clean base sync removes stale in-process ML
packages, while optional workers are reinstalled independently. Existing
`data/` is not recreated as part of dependency migration; artifacts with old
provider identities remain inert until explicit reindex.

## Benchmark subsystem

The approved scope sits between a standalone evaluator and a full experiment
platform:

- portable, versioned dataset/query manifests;
- local/private asset resolution by stable asset id and checksum;
- schema validation and import tooling;
- immutable run directories plus a small local run registry;
- CLI runner and machine-readable comparison output;
- summary API/UI integration only after the core is stable;
- no multi-user hosted experiment service.

Cases support zero, one or multiple relevant intervals, hard negatives, domain
and modality slices, gold/silver provenance and complete-video split groups.
The current core records auditable retrieval quality, per-case latency, bounded
raw process-tree RSS samples, and contained artifact/storage snapshots. The
warm, read-only product runner accepts every frozen profile, emits a complete
per-asset capability matrix during preflight, and binds profile identities to
the selected plan. Existing external workers are not measurable merely because
an HTTP health probe succeeds: without PID/start-token/executable binding their
profiles fail measurement with
`external_worker_process_binding_unavailable`. InternVideo is accepted as a
profile but remains explicitly `not_configured`. Dataset and promotion
thresholds are frozen before model comparison.

The separate offline full-ML smoke self-starts Vision, Whisper, Lighthouse and
Qwen as measured descendants, runs OCR as a child, and exercises the production
path from synthetic upload and durable indexing through five generation-bound
search profiles, inspectable evidence and an FFprobe-verified MP4 export. Its
receipt includes raw process-tree RSS plus system-wide Metal and VM samples;
system-wide Metal is standalone and non-additive with process RSS, not a
per-worker attribution. The smoke implementation is present, but Phase 0 remains
open until the exact clean-checkout artifacts and target M4 Pro produce and pass
the retained evidence gate.

## Model direction

The target basketball system is a specialized event engine, not another giant
VLM:

- scoreboard OCR with temporal voting;
- player/ball/rim/backboard detector;
- player and ball tracking;
- court/hoop geometry and ball-through-rim observations;
- event state machine;
- a small learned fusion/ranking component only after enough labelled data
  exists.

Existing general retrieval remains the candidate generator and fallback during
the transition. Qwen remains an observable-facts verifier until ablation proves
another role. InternVideo remains optional. No component is promoted merely
because it is installed.

Approved ablations run an identical frozen benchmark through explicit profiles:

1. lexical + Qdrant baseline;
2. dense SigLIP;
3. temporal refinement;
4. Lighthouse;
5. Qwen verification;
6. optional InternVideo;
7. specialized event engine stages as they become available.

## Reliability scope

Crash-safe text-vector ownership, recovery and GC are implemented. The durable
video-index job slice now persists plans, progress, cancellation, retry and
deduplication; its deterministic browser smoke test crosses the real Vite proxy,
FastAPI, SQLite dispatcher and React polling path without loading production
models. Remaining reliability work is narrower:

- video and export deletion with validated cascade plans;
- free-space checks, quotas, retention and temporary-file scavenging;
- expose recovery quarantine/degraded state and audit/storage growth in health metrics;
- background evaluation/export and bounded heavy reranking;
- generated OpenAPI-to-TypeScript contracts plus runtime response validation;
- frontend coverage gates focused on critical flows;
- backup/restore verification for the numbered SQLite migrations;
- local serving of the built frontend without macOS application packaging.

## GitHub scope

- close issue 1 with factual before/after evidence;
- triage dependency updates in compatible groups and regenerate locks;
- require CI/hygiene checks on the default branch;
- enable vulnerability alerts/security updates;
- update maintenance documentation after the completed history rewrite;
- create a first tagged release only after the new core has a migration and
  rollback note.

Design and nonessential product features remain last priority.

## Implementation order and gates

1. **Architecture contract** — this document, schemas and regression fixtures.
2. **Visual generation** — one dense path, specification identity, atomic
   activation and rollback tests.
3. **Stage provenance** — persisted capability states and fail-closed evaluation.
4. **Environment workers** — reproducible isolated profiles and smoke contracts.
5. **Benchmark registry** — portable manifests, immutable runs and baseline.
6. **Ablations** — evidence-based provider retention decisions.
7. **Sports engine** — detector/tracker/OCR/rules, then learned fusion if justified.
8. **Lifecycle/jobs/contracts/E2E** — text-vector crash recovery, durable video
   jobs, atomic release publication and one deterministic real-stack browser
   path are implemented; retention, generated contracts and broader integration
   coverage remain.
9. **GitHub/release** — protected, documented release baseline.

No phase may claim completion only because unit tests are green. Relevant data
migrations, fail-closed behavior, rollback and a proportional real-hardware smoke
test are part of the gate.

## Iteration compact: first increment

- Goal: make dense visual retrieval reproducible across upload, reindex and
  backfill.
- Who cares: local user and benchmark owner.
- Decision owner: repository owner.
- User action changed: upload/reindex no longer changes the visual sampling
  profile.
- Success metric: identical specification identity and dense sampling semantics
  on all entry points.
- Guardrails: existing search/API behavior, source videos and active working index
  remain available after a failed rebuild.
- Unacceptable mistakes: silent sparse replacement, activation of partial data,
  deletion of source media, treating legacy provenance as current.
- Baseline before this increment: the scene writer and manual dense writer shared
  one model-only marker.
- First experiment: specification and generation persistence using fake
  extractors and embeddings, followed by one deterministic demo-video integration
  smoke.
- Rollback: keep the previous generation active and allow the current read path
  during a bounded migration window.
