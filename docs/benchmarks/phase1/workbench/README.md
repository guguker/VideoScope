# Full-match annotation workbench

This is the Phase 1 local annotation tool described by the
[approved design](../../../superpowers/specs/2026-09-11-annotation-workbench-design.md)
and [implementation plan](../../../superpowers/plans/2026-09-11-annotation-workbench.md).
Implementation verification is in progress; no accepted smoke receipt is claimed
by this document yet. The tool creates annotation-inbox records, not gold labels,
training jobs or production indexes.

## Owner workflow

1. Open the local workbench and select a match available for review. Existing
   pilot answers appear at their original full-match times.
2. Mark an attack with “Добавить владение”, or start with “Добавить событие”.
   Set start/end using playback controls and answer only what is observable.
3. Add teams and players once per match, then select them on later events.
   Preserve an unreadable number as unknown; `0` and `00` are different values.
4. For another shot, pass, foul or substitution, add a separate event. Link events
   to their possession when useful. A last passer is not automatically an assist.
5. Pause the video and choose “Добавить кадр с точками” to mark visible players
   near their feet and the ball at its visible center. Coordinates describe the
   image; they do not claim a calibrated court location or inferred trajectory.
6. Drafts save automatically. Use “Подтвердить разметку” after reviewing the
   required facts. A failed save remains visible and preserves the entered data.
7. Return later to continue, or download a backup and restore it using the backup
   controls. Export contains annotations/provenance and no source video.

After preparation, double-click
[`open-annotation-workbench.command`](../../../../scripts/open-annotation-workbench.command)
to reopen the workbench after restarting the Mac. It checks the local workspace,
starts a server if needed and opens the browser. Keep its terminal window running
while annotating; closing it never removes saved answers.

The known library has two content-authorized development sources, five reserved
sources and one suspected regression source. The latter six are listed as metadata
and remain unavailable for playback under the current source-role ledger. The
tool does not impose a fixed clip or annotation quota. Source allocation is
separate from building a convenient full-match editor.

## Local preparation and launch

Run from the repository root with the installed base Python environment. The
project root is explicit: the original UBA inventory uses paths relative to that
root, while a separately generated audit may use paths relative to its source
directory. Never infer the media root from a downloaded manifest.

```sh
PYTHONPATH=backend/src .venv/bin/python -m videoscope.annotation_workbench.prepare \
  --audit data/ml/reviews/uba-pilot-v1/audit \
  --project-root "$PWD" \
  --legacy-batch data/ml/reviews/uba-pilot-v1/batch-v1 \
  --output data/ml/reviews/uba-pilot-v1/workbench-v1

PYTHONPATH=backend/src .venv/bin/python -m videoscope.annotation_workbench.server \
  --workspace data/ml/reviews/uba-pilot-v1/workbench-v1 --port 8767
```

Preparation targets a new directory and never replaces the pilot. Source bindings
are read-only and kept server-side. Public identities use source SHA, groups and
rights rather than local paths. Legacy event aliases resolve by source SHA and
duration; original JSON revision bytes and clip-relative coordinates are retained.

The server binds only to `127.0.0.1`, checks Host and Origin and serves local
scripts/media. It creates no production Runtime or provider. Changing or replacing
an attested file, corrupting a record, or submitting a stale write fails closed.

## Verification commands

All write/restart/restore tests use synthetic media and temporary workspaces.

```sh
PYTHONPATH=backend/src .venv/bin/pytest backend/tests/test_annotation_workbench_storage.py \
  backend/tests/test_annotation_workbench_server.py \
  backend/tests/test_annotation_workbench_prepare.py

cd frontend
node node_modules/vitest/vitest.mjs run src/lib/annotationWorkbench.test.ts
node node_modules/typescript/bin/tsc -b
node node_modules/vite/bin/vite.js build
cd ..
node scripts/smoke-annotation-workbench.mjs --require-clean
```

The accepted run must include source Range playback after the one-hour mark,
multiple events, teams/players, possessions, spatial observations, draft recovery,
server restart, immutable legacy preservation and portable backup restore. Full
backend/frontend suites and the legacy browser smoke also remain required for
handoff. Actual counts, serving SHA and receipt location will be recorded after
those checks finish.

## Phase and rollback

The [Phase 0 baseline](../../phase0/accepted/7054f13/README.md) is unchanged.
The pilot remains a separate usable path. Workbench histories and their backups
are user state and must be retained during rollback; do not delete a workspace to
return to the old UI. No trained model or active retrieval artifact is changed.
Rights, group isolation, annotation adjudication, coverage and dataset sealing
still determine the Phase 1 exit gate.
