# Annotation Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a usable durable full-match annotation tool preserving the owner's existing labels.

**Architecture:** Add an isolated annotation_workbench package, source-bound SQLite revision storage and a loopback API. A separate static web application edits the shared versioned contract. The pilot server and production runtime remain independent.

**Tech Stack:** Python 3.12, existing FastAPI/Pydantic/SQLite, browser ES modules, Vitest/jsdom and Playwright; existing FFmpeg for synthetic tests only.

**Spec:** `docs/superpowers/specs/2026-09-11-annotation-workbench-design.md`

## Global Constraints

- No training, new model inference, external download/upload or production import/indexing.
- Existing user media, pilot manifests and raw v1/v2/v3 answers must remain unchanged.
- Only source roles already authorizing review may serve content; all eight may appear as metadata.
- All source/entity times use complete-video seconds. All point x/y use intrinsic video normalized coordinates.
- Source SHA/group/rights and annotation history survive restart, export and restore; no answer automatically becomes gold.
- API names, payloads and enums are the shared contract in the spec; changes require updating that spec and both consumers.
- Work in `/private/tmp/videoscope-workbench-20260911`; do not touch original .gitignore/downloader or the live pilot server.
- Use existing dependencies; no network installs. Run Python with `PYTHONPATH=backend/src` from repo root.

## Task 1: Durable data contract, preparation and loopback API

**Files:**
- Create `backend/src/videoscope/annotation_workbench/{__init__,schema,storage,prepare,server}.py`.
- Split source validation, legacy import, transfer and streaming into
  `source_library.py`, `legacy.py`, `transfer.py`, `streaming.py` when needed to
  keep responsibilities focused; their public contract remains the Store/API above.
- Create `backend/tests/test_annotation_workbench_{storage,server,prepare}.py`.
- Create JSON schema exports under `docs/benchmarks/phase1/workbench/`.
- Read/reuse pilot helpers without changing legacy completion/version behavior.

**Interfaces:**
- `prepare_workspace(*, audit_dir: Path, project_root: Path, legacy_batch: Path, output: Path, code_sha: str) -> dict` builds a new contained workspace; reject pre-existing output except explicit idempotent legacy sync.
- `WorkbenchStore(root: Path)` exposes `workspace()`, `source(source_id)`, `save(payload)`, `progress(payload)`, `history(record_id)`, `export_lines()`, `restore_lines(lines)` and `media(source_id)`.
- `create_app(workspace_dir: Path, *, port: int=8767) -> FastAPI`; CLI `python -m videoscope.annotation_workbench.server --workspace DIR --port PORT`.
- Preparation CLI: `python -m videoscope.annotation_workbench.prepare --audit DIR --project-root DIR --legacy-batch DIR --output DIR`.
- Store private manifest resolver paths separately from public revision identity. Export/restore identities do not depend on filesystem paths or code SHA.

- [ ] Write storage behavior tests against actual temporary SQLite and source fixtures. Literal event example for testing:

```python
shot = {"event_type":"shot", "start_seconds":12.0, "end_seconds":17.0,
        "shot_type":"two", "outcome":"miss", "scoring_decision":"not_applicable",
        "play_context":"foul_on_shot", "presentation":"live", "boundary_status":"complete"}
saved = store.save({"schema_version":1, "workspace_revision":store.workspace()["workspace_revision"],
                   "source_id":"source-a", "record_id":"event-" + "a" * 32,
                   "expected_revision":0, "kind":"event", "status":"human_reviewed",
                   "archived":False, "data":shot})
assert saved["revision"] == 1
assert WorkbenchStore(root).source("source-a")["records"][0]["data"]["end_seconds"] == 17.0
```

Test missing end as saved draft, `0` vs `00`, cross-source player rejection, shrinking
possession past event rejection, counted post-whistle rejection, stale revision
does not append, exact raw legacy bytes and absolute interval conversion, restore
valid tail vs divergent/history/trailer rejection without partial writes.

- [ ] Run narrow tests and record the expected failing behavior before implementation:

```sh
PYTHONPATH=backend/src .venv/bin/pytest backend/tests/test_annotation_workbench_storage.py -q
```

- [ ] Implement strict discriminated record schemas, immutable revision writes and
  transactional projections/reference checks; private attested manifest and read-only
  legacy import. Use source provenance from inventory/roles/rights; hash only as
  required to verify authorized source bindings, never decode reserve. Backend must
  support partial drafts without silently removing fields. Stream backup with
  per-line bounds, total-size bounds and complete trailer validation.
- [ ] Add route tests using real `TestClient(create_app(root, port=8767))` with
  `base_url='http://127.0.0.1:8767'`, Origin on writes. Assert 206 response bytes for
  `Range: bytes=0-15`, forbidden source returns no media, unknown Host/Origin denied,
  malformed/oversized JSON/backup rejected, changed source identity fails closed,
  no local paths in API/export. Implement server and CLI to pass them.
- [ ] Run all three new files plus old annotation tests, inspect every failure,
  export strict JSON schemas, self-review and commit only owned files.

## Task 2: Complete full-match editing interface

**Files:**
- Create `backend/src/videoscope/annotation_workbench/web/{index.html,style.css,app.js,state.js,geometry.js}`.
- Create `frontend/src/lib/annotationWorkbench.test.ts`.

**Interfaces:**
- Consume the exact API/record contract from the spec. No source paths in the UI.
- `geometry.js`: export `pointFromClient({clientX,clientY,rect,videoWidth,videoHeight})`
  returning normalized `{x,y}` or null outside letterboxed video.
- `state.js`: export state/save queue helpers as needed; actual app uses the same
  code. Expose no test-only production hooks. DOM tests import actual modules and
  supply a complete API fixture at the network boundary.

- [ ] Write geometry and real DOM editing tests before code. Independent literal
  expectation for a 16:9 video inside an 800×600 box:

```ts
expect(pointFromClient({clientX:400,clientY:75,
  rect:{left:0,top:0,width:800,height:600},videoWidth:1920,videoHeight:1080}))
  .toEqual({x:0.5,y:0})
expect(pointFromClient({clientX:400,clientY:50,
  rect:{left:0,top:0,width:800,height:600},videoWidth:1920,videoHeight:1080}))
  .toBeNull()
```

Test independent events retain answers, new player preserves `00`, draft restores
after reload, rejected/stale save retains edited data, autosave never confirms
review, points disappear on time change, keyboard shortcuts ignore inputs.

- [ ] Run and record expected failures:

```sh
cd frontend
node node_modules/vitest/vitest.mjs run src/lib/annotationWorkbench.test.ts
```

- [ ] Implement the connected Russian interface from the spec: match/record
  navigation, full time controls, contextual editors, reusable teams/players,
  possessions and multiple events, visible draft/review state, selectable point
  overlay, autosave recovery, history inspection and backup/restore. Keep save
  failures actionable and never overwrite a conflicting newer answer automatically.
  Use semantic labels for browser smoke, and no inline scripts/external resources.
- [ ] Run focused tests, full frontend suite and build, then self-review/commit
  only owned files. Report stable accessible labels needed by the E2E worker.

## Task 3: End-to-end proof, documentation and safe local handoff

**Files:**
- Create `scripts/smoke-annotation-workbench.mjs`.
- Create `scripts/open-annotation-workbench.command` for reopening the prepared
  local workspace after a Mac restart, without manual terminal commands.
- Create `docs/benchmarks/phase1/workbench/README.md` and sanitized smoke receipt.
- Update `AGENTS.md`, `docs/ml-autonomy-plan.md`, `docs/benchmarks/phase1/README.md`
  with measured current capabilities and remaining Phase 1 gates.

**Interfaces:**
- Use the real Task 1 CLI/API and Task 2 DOM. Reuse installed Playwright and the
  legacy smoke's child-process/FFmpeg patterns. Own temporary workspaces only.
- All production worktree code is committed before the accepted smoke receipt;
  final documentation commit may record that exact serving code SHA.

- [ ] Build a generated low-resolution H.264/AAC source longer than one hour
  (low FPS/static pattern), a separate reserved source, and a legacy fixture with
  byte-exact saved answers; use no user content. Start the actual server.
- [ ] Exercise actual browser playback and seek; create team/player `00`, possession,
  shot and pass, substitution and player/ball points. Save another event, reload
  with a draft, restart server, amend first event and assert other answers retained.
  Download backup, restore to a second workspace with identical source hashes at
  different paths, compare data and assert no external requests/page errors.
  Example public assertions:

```js
assert.equal(exportedPlayer.data.jersey_number, '00')
assert.equal(restoredShot.data.start_seconds, 3605)
assert.equal(await hashFile(legacyPath), legacyHashBefore)
assert.deepEqual(externalRequests, [])
```

- [ ] Run focused suites, relevant full backend/frontend suites, frontend build and
  old/new real browser smoke. Inspect synthetic screenshots at desktop and narrow
  widths, fix regressions through the owning worker and repeat covering checks.

```sh
PYTHONPATH=backend/src .venv/bin/pytest backend/tests
cd frontend && node node_modules/vitest/vitest.mjs run
node node_modules/typescript/bin/tsc -b && node node_modules/vite/bin/vite.js build
cd ..
node scripts/smoke-annotation-review.mjs
node scripts/smoke-annotation-workbench.mjs --require-clean
```

- [ ] Obtain task and final reviews against exact committed diffs, address material
  findings, and verify original user changes/pilot raw histories preserved.
- [ ] Fast-forward the original feature checkout only when safe; initialize a new
  private workspace beside the pilot using real audit/roles and latest legacy
  answers. Keep the pilot reachable, start the workbench on 8767, verify metadata
  and authorized playback through local HTTP without external capture. Record
  truthful receipt and give owner link plus first actions. No training or whole
  Phase 1 completion claim.
