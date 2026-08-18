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

SQLite schema v6 stores derived evidence as immutable generations. Migration v5
introduced source/specification-linked scene, speech, OCR and object generations;
migration v6 adds verified per-video text-vector generations. A stage run, source
SHA-256 and canonical specification hash identify each generation; separate
SQLite pointers select the active segment and vector generations atomically.
Reindexing never deletes the previous active generation. Scene images follow the
same rule under `thumbnails/<video-id>/generations/<generation-id>/` and are
published before the database pointer changes.

Rows migrated from older databases retain `generation_id = NULL`. They remain
stored for recovery but are not trusted by lexical search, semantic indexing or
thumbnail fallback. The application does not silently rebuild a ready library at
startup: each legacy video needs an explicit Reindex action before its evidence
becomes searchable again. Text-vector points are written into a new immutable
Qdrant generation, followed by a manifest sentinel and an exact read-back
validation. Only then may one SQLite transaction activate that generation and
complete its stage run. A failed build preserves the previous active generation.
Legacy unversioned Qdrant points and markers remain inert and untrusted until an
explicit reindex publishes a verified generation.

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

## Trust boundaries

The primary application is local-only: configuration accepts only loopback hosts,
and requests with an untrusted `Host` or browser `Origin` are rejected. Upload
basenames and extensions are checked before storage; FFprobe performs the real
container validation asynchronously during ingest. File extensions alone are not
a content authenticity check.

An optional InternVideo deployment is a separate network service and must be
treated as a private authenticated boundary. Frames and bearer credentials must
not be sent over public plaintext HTTP.
