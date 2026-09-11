# Phase 1: private UBA annotation pilot

This bounded slice prepares local clips for explicit owner review. It does not
train a model, publish a dataset, modify the production library or complete
Phase 1. The accepted [Phase 0 baseline](../phase0/accepted/7054f13/README.md)
and its rollback evidence remain unchanged. The strategic gate is defined in
[the ML autonomy plan](../../ml-autonomy-plan.md).

The owner's 2026-09-11 delivery preference extends the next tool goal to a
comfortable full-match annotation workbench with resumable progress, events,
possessions, participants and spatial observations. The pilot below records
what already exists; its 24 clips are not a limit on future annotation volume.
The current goal and its acceptance checks live under “Текущая цель Фазы 1” in
the plan. This documentation update does not implement that workbench or change
source roles. The [public dataset/model survey](external-candidates-2026-09-11.md)
records possible external components, including defensive interactions, without
importing media or training models.

## Authorization and observed inventory

The owner authorized auditing the eight downloaded UBA files and preparing a
first private review batch. NBA remains deferred. Local review authorization
does not establish training or publication rights for the broadcasts.

The metadata and complete-byte SHA-256 audit observed eight stable MP4 files:
47,170,278,639 bytes and approximately 19.092 hours in total. Each reports
1920×1080 H.264 video at 60 FPS, with AAC 44.1 kHz stereo audio. There are zero
exact-byte duplicate groups. Different hashes do not establish independent
games or exclude another encoding and replay of the same event. Full decoding
of the complete source streams was not performed.

The current preparation roles are deliberately narrower than a dataset split:

| Role | Sources | Content access in this slice |
| --- | ---: | --- |
| `development_review` | 2 | Local sampling and prepared previews |
| Suspected existing regression source | 1 | Excluded from preparation pending identity review |
| `reserve_uninspected` | 5 | Metadata/hash audit only; no frames selected or model inference |

The suspected regression source requires review against an already studied
game; a duration similarity is not proof of identity. Reserved sources are
**not a sealed holdout**. Rights remain `unknown`, replay/game grouping remains
provisional, and all sources retain `training_allowed:false` and
`promotion_eligible:false` in the role contract.

## Pilot contract

The first batch contains 24 unlabelled context clips from the two development
sources: eight timeline controls and sixteen weak SigLIP proposals. The selector does
not invent duplicate windows to fill a quota; the completed batch manifest is
the authority for its actual count. The five proposal prompts cover three-point
attempts, close-range attempts, free throws, replays and non-game content.
Neither a prompt nor a similarity score establishes the event or its outcome.

Sampling uses keyframes separated by at least four seconds, records their
actual decoded timestamps, and retains their hashes. Scoring reuses the local
production `LocalVisionWorkerRuntime` and its pinned SigLIP preprocessing with
`google/siglip2-base-patch16-224` revision
`75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. It opens no production index and runs
no Qwen verification. Sparse keyframe selection is an annotation aid; it does
not measure full-timeline candidate recall or event quality.

Each prepared preview retains the original 1080p frame geometry and frame rate,
uses H.264/AAC, and includes approximately 18 seconds of context. There is no
spatial crop. Every exported preview is separately probed and fully decoded;
its byte size and SHA-256 bind it to the source interval. The source files are
read-only throughout preparation. The longer review context is not a change to
the frozen Phase 0 protocol or the future ranker prediction contract.

The review form asks explicitly for shot type, visible outcome, scoring decision,
play context, live/replay status and boundary completeness. Corrections use **clip-relative seconds**, bounded by
the actual prepared duration. Unclear answers are valid; an ambiguous episode
does not acquire an invented label. The proposed selection category is omitted
from the review API to avoid presenting the model's guess as a human answer.

Saving appends a revision with source and prepared-input provenance. Six
explicit answers and both clip-relative boundaries are required for status
`human_reviewed`; a saved nullable field remains a `draft` and is excluded from
the completed-review count. Explicit `unclear` is a valid completed answer.
Both statuses remain in `annotation_inbox`, with `gold:false`,
`training_allowed:false` and `promotion_allowed:false`. Explicit answers still
require the later annotation policy, critical-conflict adjudication and dataset
sealing before they can support training or promotion. The server creates no
labels before a person saves a review. Implicit clicks are not labels.

### Annotation version 2: visible outcome and awarded points

The owner requested this correction after encountering a missed shot with a foul
followed by a teammate's new shot after the whistle. One annotation describes
one event within its selected boundaries; different shots must not be merged
into one outcome. If the event cannot be isolated confidently, retain uncertainty
and explain the ambiguity in the note. Version 2 saved one selected event per
preview; version 3 below removes that limitation.

`outcome` records what happened physically, not an official field-goal statistic.
The independent `scoring_decision` is `counted`, `not_counted`, `not_applicable`
or `unclear`. `play_context` is `in_play`, `foul_on_shot`, `after_whistle`,
`other_dead_ball`, `not_applicable` or `unclear`. Here `after_whistle` means a
**new** shot after play stopped, not continuation of a shot with a foul. A
continued shot with a foul may count. A new dead-ball shot cannot be marked
`counted`; the form and API reject that contradiction without changing answers.
Visible outcome and points remain independent, including cases such as awarded
points without a visible make. Replay presentation is independent of play context.
These distinctions follow [FIBA rules, articles 10.3–10.4](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-v10a.pdf).

Version-2 requests and saved records use annotation `schema_version:2`; the immutable
batch manifest stays at version 1. Old version-1 revisions remain byte-identical
on disk and unchanged in the version-2 export history. Their old answers remain
available, while the two new fields appear blank and require explicit review.
Old `human_reviewed` status describes the old protocol; it does not complete
version 2. Missing values are not inferred from outcomes, comments or whistles.
An outdated form cannot save through the new API without the version-2 request
contract. Answers remain private and do not become gold or training data.

Version 2 validation passed 2,678 backend tests and 66 frontend tests, including
92 backend review tests and 21 frontend review tests; the frontend build passed.
The synthetic browser check now seeds a version-1 answer and verifies blank new
fields, byte-identical old history, explicit version-2 correction, a counted shot
with a foul, a visible make after stoppage without awarded points, contradiction
rejection, restart and mixed-version export. The real batch's 50 existing
manifest/media files were hash-checked unchanged; no test answers were written
to it. The local server advertises annotation version 2 and retains the original
batch revision. Full-ML smoke and model inference were not run for this form
change; Phase 0 artifacts and rollback remain unchanged.

### Annotation version 3: several events in the same clip

The owner requested a usable way to retain both the original missed foul shot
and a teammate's later dead-ball make. The current form has a selector for each
shot, **Save shot**, **Save and next**, and **Add another shot in this clip**.
Adding a shot creates a separate blank draft; it does not save, copy labels or
replace the first shot. The proposed start is the previous selected end when
there is room in the clip. The owner checks both boundaries before saving.
Switching shots restores their individual answers and seeks to their start;
an accidentally added unsaved shot can be removed. Drafts remain in the current
tab during navigation; only explicitly saved answers survive closing the tab.

New requests and records use `schema_version:3` and a stable `event_id` bound to
the same immutable source/example. The original event is `primary`; historical
v1/v2 files remain byte-identical. Old open version-2 forms may still save only
`primary`, so a pending old-form answer cannot overwrite an additional event.
All six human labels keep version-2 semantics; no new semantic answers are
inferred. A v1 answer still needs the two missing version-2 fields.

Additional events have independent optimistic revisions and atomic append-only
histories under `annotations/<example>/events/<event-id>/`; primary revisions
retain their existing path. At most 31 additional events are allowed per clip.
Display order follows first save, so editing or restarting does not reorder
saved shots. The version-3 export retains every original v1/v2/v3 record and its
source/clip provenance. All related events inherit the same source/replay split
constraints. Neither preview preparation nor production indexing is rerun.

The version-3 request schema is
[event-annotation-request.schema.json](event-annotation-request.schema.json).
Regression checks exercise two shots, separate boundaries, independent edits,
mixed legacy history, concurrent creation, failed publication/retry and restart.
The browser smoke also verifies both shots remain separately selectable after
restart and exports both histories. This verifies the annotation workflow;
it does not establish complete event coverage or model quality.

Validation: 2,689 backend tests, including 103 review tests, and 68 frontend
tests passed; the frontend build passed. The synthetic browser smoke saved and
independently edited a second event after restart, retained the first event and
legacy revision bytes, and exported all six revisions. Source media, preparation
receipts and the Phase 0 baseline/rollback are unchanged. No ML inference or
training is part of this update.

## Private artifacts and replay

Private files live under `data/ml/reviews/uba-pilot-v1/` and are excluded from
Git. The retained local layout is:

```text
audit/                 inventory, raw probes, rights ledger, reviewed source roles
preparation-v2/        sampled frames, sampling receipt, weak scores
preparation-committed-replay/  independent replay from the committed implementation
batch-v1/              immutable batch.json, receipt, clips/, posters/
  annotations/         append-only human review revisions, created on first save
    <example>/events/  separate revision histories for additional shots
```

Only schemas, code, synthetic tests and this aggregate report belong in Git.
Private filenames, source URLs/game identifiers, clips, frames, labels and
model scores are not included here. The two exported schemas are
[review-batch.schema.json](review-batch.schema.json) and
[annotation-request.schema.json](annotation-request.schema.json).
They are deterministic Pydantic JSON Schema exports. Cross-record source
membership, interval consistency, finite numbers and file attestation are also
enforced by the Python validators/server; structural JSON Schema validation
alone is insufficient to approve a batch.

Run commands below from the repository root. Audit and preparation outputs must
be **new directories**. Pick a new suffix for another replay; never clear an
existing batch or its annotations. The local FFmpeg/FFprobe tools and reviewed
Python environments must already be installed.

```bash
.venv/bin/python -m videoscope.annotation_review.audit \
  --source-dir 'Данные матчи' \
  --output data/ml/reviews/uba-pilot-v1/audit-replay-01
```

The audit does not assign preparation access. `source-roles.json` is a separate,
reviewed authorization artifact bound to the inventory. The following commands
use the existing audited source set and its explicit development roles. A fresh
audit requires a separately checked role artifact before decoding any source.

```bash
.venv/bin/python -m videoscope.annotation_review.prepare sample \
  --audit data/ml/reviews/uba-pilot-v1/audit \
  --root . \
  --work data/ml/reviews/uba-pilot-v1/preparation-replay-01

PYTHONPATH=backend/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-vision/bin/python -m videoscope.annotation_review.prepare score \
  --audit data/ml/reviews/uba-pilot-v1/audit \
  --root . \
  --work data/ml/reviews/uba-pilot-v1/preparation-replay-01

.venv/bin/python -m videoscope.annotation_review.prepare package \
  --audit data/ml/reviews/uba-pilot-v1/audit \
  --root . \
  --work data/ml/reviews/uba-pilot-v1/preparation-replay-01 \
  --output data/ml/reviews/uba-pilot-v1/batch-replay-01 \
  --batch-id uba-pilot-replay-01

.venv/bin/python -m videoscope.annotation_review.server \
  --batch data/ml/reviews/uba-pilot-v1/batch-v1 --port 8766
```

The initial private audit records source paths relative to the **repository**,
including the `Данные матчи/` prefix, hence `--root .` above. The reusable audit
CLI records paths relative to its **source directory**. When preparing from a
fresh `audit-replay-01` inventory and its separately reviewed roles, use
`--audit .../audit-replay-01 --root 'Данные матчи'` and new work/output directories.
These two path contracts must not be interchanged.

Open `http://127.0.0.1:8766` locally. The review server binds only loopback and
serves an explicit static/media allowlist. It does not construct the production
Runtime, start model workers, ingest videos or open production databases.
Writes require matching Host/Origin and bounded JSON. Prepared media are
attested at startup, and changed paths/content fail closed.

`batch_revision` is the lowercase SHA-256 of the original manifest serialized
as UTF-8 JSON with `sort_keys=True`, `separators=(',', ':')`,
`ensure_ascii=False`, `allow_nan=False`, **without a trailing newline**.
Annotations bind this revision and use expected-revision concurrency checks.
The export endpoint returns complete private review history and provenance.

## Validation and stopping condition

The [pilot receipt](pilot-receipt.json) records the completed 2026-09-11 handoff
from implementation `96a1d6fb6c1ab9c2af1e062cfb379eb52875b296`: 24 previews,
432 seconds of context and 404,914,368 prepared bytes. All previews passed full
decoding, format/hash checks and HTTP range playback checks. All eight original
source hashes remained unchanged. The real batch contained zero annotations at
handoff; browser write/restart checks used only synthetic media.

Replaying the committed sampler and scorer reproduced all 2,620 frame hashes,
timestamps and 24 selected intervals, with a maximum absolute score difference
of zero against the predeclared tolerance of `1e-6`. This is a reproducibility
check, not evidence of shot recognition quality.

The full backend suite passed 2,647 tests; the 120 annotation tests passed with
90.03% combined coverage with branch measurement enabled. All 60 frontend tests,
the frontend build and the real Chromium review smoke passed. The private error
ledger retains two resolved infrastructure failures: unavailable sandbox Metal
access before inference, and a coverage-file collision after an earlier passing
test run. Neither is a model miss; semantic model quality remains unevaluated.
This pilot does not supply a new full-ML memory or promotion attestation.

Narrow backend checks, including concurrent writers and a forced failed
annotation publication that retains the previous revision:

```bash
.venv/bin/pytest backend/tests/test_annotation_audit.py \
  backend/tests/test_annotation_prepare.py \
  backend/tests/test_annotation_prepare_integration.py \
  backend/tests/test_annotation_review.py

COVERAGE_FILE=.coverage-annotation-review .venv/bin/pytest \
  backend/tests/test_annotation_audit.py backend/tests/test_annotation_prepare.py \
  backend/tests/test_annotation_prepare_integration.py backend/tests/test_annotation_review.py \
  --cov=videoscope.annotation_review \
  --cov-branch --cov-report=term-missing --cov-fail-under=80

pnpm --dir frontend test
.venv/bin/pytest backend/tests
pnpm --dir frontend build
node scripts/smoke-annotation-review.mjs
git diff --check
```

The browser check must verify actual local preview playback/seeking, explicit
answers, boundary correction, save acknowledgement, reload persistence and
export. Write/reload smoke checks use a **disposable synthetic batch**, keeping
the real owner-review batch unlabelled. Human content review is not replaced
by this browser smoke. The receipt above records this handoff's actual checks;
future replays must retain their own results.

This slice stops when the local batch is technically validated, the owner can
review and save it, and the private provenance/annotation history are retained.
It does not assert model accuracy, candidate recall, a sealed split or the
Phase 1 exit gate. Rights, independent groups, confirmed labels, required strata
and untouched evaluation sources remain prerequisites for that gate. No model
training or active-artifact switch occurs in this slice.
