# Data and API contracts

## HTTP boundary

FastAPI schema at `/api/openapi.json` is the canonical HTTP contract. JSON field
names use `snake_case`. Request bodies reject unknown fields, unsafe video IDs,
non-finite numbers, inverted time ranges and empty explicit video scopes. Internal
filesystem fields such as stored filenames and media paths are never returned.

The React client mirrors these DTOs in `frontend/src/types.ts`. Any contract change
must update the Pydantic model, its API test and the frontend type/test in the same
change. OpenAPI-to-TypeScript generation remains a tracked follow-up; TypeScript
assertions alone do not validate untrusted JSON at runtime.

Errors use FastAPI's `{ "detail": ... }` envelope. `detail` may be a string or an
array of validation issues, so the client normalizes it before displaying a
message. Video, thumbnail and export endpoints return binary responses instead of
JSON. All referenced files are resolved and checked against their allowed data
directory before they are served.

## Local persistence

`data/` is runtime state and is never committed:

| Path | Role | Authority |
| --- | --- | --- |
| `videoscope.sqlite3` | video records, processing state and temporal evidence | source of truth |
| `media/` | uploaded source videos with generated storage names | referenced by SQLite |
| `thumbnails/` | generated scene thumbnails | derived |
| `clips/` | exported MP4 montages | derived |
| `qdrant/` | text/vector retrieval index | rebuildable index |
| `visual-index/` | SigLIP arrays and JSON metadata | rebuildable index |
| `cache/` | Lighthouse/Qwen/provider caches | derived, untrusted input on read |
| `search-glossary.json` | user-maintained search aliases | local user data |
| `evaluation/cases.json` | local developer evaluation cases | local user data |
| `evaluation/latest-report.json` | latest generated evaluation result | derived |
| `.videoscope-runtime.lock` | persistent process-ownership inode guarded by `flock` | coordination state; never delete as cleanup |

SQLite schema v9 stores derived evidence, durable video-index jobs and recovery
state. Migration v5
introduced source/specification-linked scene, speech, OCR and object generations;
migration v6 added verified per-video text-vector generations; v7 introduced
global generation tombstones and leased crash-safe GC jobs; v8 adds indefinitely
recoverable transient retries with capped backoff, a monotonic total-attempt fence,
a bounded window of 256 immutable recent audit rows, and quarantine for corrupt
build/job metadata. Migration v9 adds canonical `VideoIndexPlanSnapshot` JSON,
source/plan/idempotency identities, queued/running/terminal states, retry
lineage, cancellation timestamps, secret execution-token fences, job-linked
StageRuns and immutable external-index descriptors. A stage run, source SHA-256 and
canonical specification hash identify each generation; separate SQLite pointers
select the active segment, vector, SigLIP and Lighthouse generations atomically.
Job-owned segment/vector/external outputs are staged without changing those
pointers. One successful job-completion transaction selects a coherent release;
cancel/fail/recovery never publishes a partial release. Reindexing never deletes
the previous active generation. That transaction requires an exact terminal
`StageRun` receipt for all seven entries of the persisted plan; missing,
duplicate, extra, cancelled or still-running receipts fail closed. Optional
stages may finish as `failed` or `not_configured` without being mistaken for an
unexecuted stage. Scene images follow the
same rule under `thumbnails/<video-id>/generations/<generation-id>/` and are
published before the database pointer changes.

Rows migrated from older databases retain `generation_id = NULL`. They remain
stored for recovery but are not trusted by lexical search, semantic indexing or
thumbnail fallback. The application does not silently rebuild a ready library at
startup: each legacy ready video needs an explicit Reindex action before its
evidence becomes searchable again. Pre-v9 queued/processing rows are adopted only
under the exclusive data lock after their current Asset and a newly persisted
plan have been verified. Text-vector points are written into a new immutable
Qdrant generation, followed by a manifest sentinel and an exact read-back
validation. Only then may one SQLite transaction activate that generation and
complete an unowned legacy stage, or stage it for a durable job's final release.
A failed build preserves the previous active generation.
Legacy unversioned Qdrant points and markers remain inert and untrusted until an
explicit reindex publishes a verified generation.

GC never targets a committed generation. At startup the API acquires the data-dir
lock before SQLite initialization and terminalizes reservations left by the
previous owner. During normal operation one shared storage gate makes Qdrant
build/commit mutually exclusive with lease recovery and exact generation delete;
deletion is complete only after an exact zero-count read-back. Transient storage
failures stay retryable with backoff up to 300 seconds, while permanent contract
failures and quarantined metadata fail closed. The lock file is intentionally
persistent: ownership is the live `flock`, not file presence.

Legacy evaluation cases contain local video IDs, so they are not a built-in
portable dataset. The benchmark subsystem instead accepts versioned manifests
whose stable aliases and source checksums are explicitly resolved to local Asset
records; paths and private repository IDs never enter the portable manifest.
Legacy cases are accepted only for existing, fully processed videos and their
labelled intervals must fit inside the recorded video duration. The same
references are checked again before every run; an empty case set is not a
successful benchmark.

Generated reports include separate cases, runtime, schema and methodology
revisions. The runtime fingerprint covers the active retrieval components, pinned
model identities, provider availability, search thresholds/configuration and the
current glossary without exposing their raw values. A saved report becomes stale
when any of those inputs changes. Recall@K counts a result only when IoU is at
least `0.3`; failed searches are reported separately and excluded from aggregates.

Dense visual generations include the isolated SigLIP inference projection in
their canonical specification. The RF-DETR projection is separate and therefore
cannot silently invalidate or bless SigLIP vectors. Speech generations include
the exact model/runtime/dependency identity and SHA-256 of the bounded effective
prompt. Existing generations created by the former in-process providers remain
stored but are incompatible and inert until an explicit reindex publishes the
new specification.

## Trust boundaries

The primary application is local-only: configuration accepts only loopback hosts,
and requests with an untrusted `Host` or browser `Origin` are rejected. Upload
basenames and extensions are checked before storage; FFprobe performs the real
container validation asynchronously during ingest. File extensions alone are not
a content authenticity check.

An optional InternVideo deployment is a separate network service and must be
treated as a private authenticated boundary. Frames and bearer credentials must
not be sent over public plaintext HTTP.

Local Vision and Whisper workers are separate authenticated loopback boundaries.
They receive only relative paths below fixed roots, re-open every path component
without following symlinks, infer over private bounded copies, and bind every
response to exact model/runtime/lock identities. They never initialize the
Repository, acquire the application data lock, or access Qdrant.
