# Full-match annotation workbench

Owner approval: 2026-09-11, “супер, начинаем, что делать?”, following the Phase 1
goal in `docs/ml-autonomy-plan.md`. Deliver the connected local annotation workflow.

## Outcome and scope

Open an authorized full match, resume progress, create possessions and events,
identify teams/players, place observations on the actual video frame, save drafts
and reviewed answers, close/restart, correct, export and restore. Existing pilot
answers are carried over without modifying their original bytes. This improves
the correction/evidence part of query → interval → evidence → clip and prepares
training data; it performs no training, new ML inference or production import.

The workbench uses the eight-source inventory as a library. Only the two sources
currently authorized for content access can play. Other source metadata is listed
with an explanatory state, without loading media/posters. No role-changing API
is part of this delivery. The UI supports future additional authorized sources;
the current role restriction is a data boundary, not an annotation quota.

## Architecture

Add `videoscope.annotation_workbench` alongside the pilot tool. Keep all pilot
modules and its live server compatible. A separate workspace directory owns an
immutable private `workspace.json`, an isolated SQLite database and imported raw
legacy histories. Source video paths are read-only server-side bindings, never
accepted in browser payloads or exposed in responses/exports. No production
Runtime, provider or index is constructed.

Legacy pilot source IDs are aliases: resolve them to catalog sources by exact
SHA plus compatible duration. Namespace imported entity IDs by batch revision,
example ID and original event ID. Add clip source offset to event boundaries but
retain exact original JSON bytes, raw field values and old completion status.
Missing new optional fields never invalidate a previously reviewed answer.

SQLite stores immutable per-entity revisions and a latest projection. Transactions
enforce expected_revision, provenance, source membership and references. Deletes
are explicit archived revisions; earlier revisions remain. Progress is distinct
from label completion. Database corruption, changed media and stale writes fail
closed with a comprehensible error. Use bounded request sizes and streamed backup
processing so full-match annotation does not require loading an unlimited history
into browser or server memory.

## Shared API contract

All endpoints are loopback-only at the configured port (default 8767), validate
Host/Origin, return no internal filesystem paths, and prohibit external requests.
JSON writes require `application/json`; backup upload uses `application/x-ndjson`.
`schema_version` is 1 for this separate workbench contract (pilot remains v3).
`GET /api/health` returns `{status:"ready",workspace_id,schema_version:1}`.

`GET /api/workspace` returns:

```json
{"schema_version":1,"workspace_id":"uba-workbench-v1","workspace_revision":"sha256hex","title":"UBA · Разметка матчей","sources":[{"source_id":"uba-01","title":"Матч 01","sha256":"sha256hex","duration_seconds":7200,"review_allowed":true,"role":"development_review","media_url":"/media/uba-01"}],"progress":null}
```

Unavailable sources have `media_url:null`. Titles are stable aliases; user may
name teams separately. Public source entries also contain `source_group` and
`training_rights` when known; rights remain unknown by default.

`GET /api/sources/{source_id}` returns `{source, records, progress}`. `records` is
the latest non-archived projection by default, including saved drafts, each shaped:

```json
{"record_id":"event-uuidhex","revision":1,"source_id":"uba-01","source_sha256":"sha256hex","kind":"event","status":"draft","archived":false,"data":{},"created_at":"ISO8601","updated_at":"ISO8601","origin":"human"}
```

`POST /api/records` body is `{schema_version:1,workspace_revision,source_id,
record_id,expected_revision,kind,status,archived:false,data}`; returns the saved
record. IDs are ASCII `team|player|possession|event|frame` + hyphen + 32 hex UUID;
legacy IDs use deterministic UUIDs. Kind/source cannot change after creation.
`status` is `draft|human_reviewed`; server enforces completeness for reviewed
records. Archiving is a revision and cannot break live references. Stale or
conflicting writes return 409 without mutating history. Invalid labels/references
return 422. Missing resources return 404. Data requests for reserved sources fail.

`GET /api/records/{record_id}/history` returns source-bound revisions for inspection.
`POST /api/progress` body is `{schema_version:1,workspace_revision,source_id,
position_seconds,selected_record_id:null,playback_rate:1}`. It preserves the last
position and selection across restarts; never creates a reviewed label.
`GET /media/{source_id}` and HEAD support HTTP Range using an attested open file;
identity/role checks apply to every request, including changed symlinks/files.

`GET /api/export` streams an NDJSON attachment: a versioned header, immutable
record revisions and raw legacy histories, then a trailer with counts and hash.
Public source identities/groups/rights are included, local paths excluded.
`POST /api/restore` accepts that backup for the same source/manifest identities.
Fully validate into contained temporary storage before an atomic transaction;
duplicate identical revisions are idempotent, divergent revisions or incomplete
backups reject the entire restore. Restore never rewrites active data or grants
gold/training rights. Return counts added/unchanged. Portability means the backup
can restore into a new initialized workspace bound to the same source hashes,
even when local root paths differ. Do not require the original absolute path.

## Record data

All user text is bounded and escaped in UI. Unknown is null or an explicit
visibility value; no model-derived/default answer becomes a human fact.
Record references must exist in the same source, have the expected kind and be
non-archived. Source times are finite seconds from start of the complete video.

- `team`: `{name, color,evidence_seconds:null}` strings, max 80/40; reviewed requires a nonempty name.
- `player`: `{team_id:null, jersey_number:null, number_status:"not_reviewed",
  evidence_seconds:null,notes:""}`. Number is a string of 1–3 decimal digits, preserving `0` vs `00`.
  Status is `not_reviewed|readable|unreadable|offscreen|unclear`; readable requires
  a number, other states require null. No roster-name identity is inferred.
- `possession`: `{start_seconds:null,end_seconds:null,team_id:null,
  attack_direction:"unclear",notes:""}`. Direction `left|right|unclear` is image
  direction for the annotated camera context, not calibrated court orientation.
  Reviewed requires both ordered boundaries and explicit team or notes explaining
  uncertainty. Drafts may retain incomplete boundaries.
- `event`: `{event_type:"shot",start_seconds:null,end_seconds:null,
  possession_id:null,actor_id:null,receiver_id:null,passer_id:null,last_pass_seconds:null,incoming_player_id:null,
  outgoing_player_id:null,shot_type:null,outcome:null,scoring_decision:null,
  play_context:null,presentation:null,boundary_status:null,
  defensive_action:null,notes:""}`. Event type is
  `shot|pass|rebound|turnover|foul|substitution|screen|defense|other|unclear`.
  Existing six shot answer enums and dead-ball consistency rule are retained.
  Reviewed shots require those six answers and both boundaries; participant and
  tactical fields stay optional. Reviewed other events require boundaries,
  presentation and boundary_status, and an explanation for `other|unclear`.
  Defensive action is nullable or `on_ball|help|switch|trap|zone|man_to_man|unclear`.
  Substitution entry/exit players are explicitly observed, not inferred from
  a player leaving the video frame. Last passer is `actor_id` on a pass event;
  on a shot `actor_id` is shooter and `passer_id` is the last observed passer;
  `last_pass_seconds` is optional evidence time. A separate pass event can capture
  more detail through the shared possession/context, with no automatic assist.
  If possession_id is set, event boundaries must lie inside that possession's
  known boundaries; editing a parent must not invalidate child events.
- `frame`: `{timestamp_seconds,event_id:null,possession_id:null,notes:"",points:[]}`.
  Each point: `{point_id:"uuidhex",entity:"player",player_id:null,x:null,y:null,
  visibility:"visible",anchor:"floor_contact",notes:""}`. Entity `player|ball`;
  visibility `visible|occluded|offscreen|unclear`; visible requires x,y in [0,1],
  offscreen requires null x,y. Anchor `floor_contact|image_center`; ball uses
  image_center. Coordinates are normalized to intrinsic video bounds, not the
  surrounding letterbox. Unique point IDs and no duplicate known player per
  frame; max 32 points/frame, no assumption all ten players are visible.
  Reviewed frame requires at least one point and explicit visibility. Camera
  calibration, interpolated trajectories and tactical inference are not produced.

## UI and durable editing

New separate `web/index.html`, CSS and ES modules; same local visual language.
Left: match list, playback, exact seek hh:mm:ss, speed, frame/short step controls,
timeline/list with filter for event/possession/frame and review state. Right:
contextual editor with clear primary actions. Persistent quick-start explains
the first three actions. Full-match time is used consistently everywhere.

Team/player management is reusable within the match. Editors expose only relevant
shot/interaction/substitution fields; help is collapsed. Create several events
in a possession without losing earlier answers. Completed annotations can be
amended without re-entering unchanged fields. Legacy annotations link to their
source-time event and original clip context.

Pause at a frame, choose/add a player, click near the feet to record position;
ball is a separate marker. Markers scale with the video; selecting a marker
permits correction/removal before save, with an explicit archived/update history
after save. Offer offscreen/unclear instead of invented points. Changing playback
time hides stale markers; selecting a frame record seeks to its timestamp.

Debounced autosave persists drafts to the server; all pending draft payloads also
persist locally for recovery during offline failures. A saved badge only follows
server acknowledgement. Serialize writes per entity, preserve unsent changes on
409/network error, and recover them on reload rather than replacing newer server
answers. Progress saves on pause/seek and periodically while playing. Warn before
leaving with data not acknowledged/persisted. Explicit “Подтвердить разметку”
marks complete answers; autosave cannot do so. Show current draft/review status.

Backup download and restore are available in the UI with a concrete summary.
The first ready screen selects the most recent saved event/progress rather than
an arbitrary empty start. Keyboard shortcuts must not fire inside inputs.

## Validation and handoff

Use failing behavior tests before implementation. Verify real SQLite restart,
stale writes, foreign-source references, reverse invalidation, drafts, backup
roundtrip/atomic rejection, media role/identity/range and raw legacy preservation.
Frontend tests exercise actual modules/DOM and failed-save recovery.

Browser smoke uses disposable generated video, including a timeline longer than
one hour, a legacy pilot fixture and reserved source. Create team/player `00`,
possession, shot, pass, substitution and spatial frame; modify drafts, reload and
restart server; assert saved values, no lost first event, export/restore and
unchanged original raw legacy bytes. Verify video playback/seek/206, precise
coordinates under letterboxing, no external requests and no console exceptions.
Run relevant full backend/frontend suites and build. Perform synthetic visual QA,
then initialize a new private real workspace and verify source/label preservation
without fabricating user annotations or inspecting reserved source content.

Do not declare Phase 1 sealed dataset complete. Preserve Phase 0 baseline/rollback.
Final delivery includes committed code/spec, replay commands, sanitized receipt,
working local link, short owner instructions and explicit unverified ML claims.
