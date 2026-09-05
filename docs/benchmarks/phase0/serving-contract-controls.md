# Phase 0: serving contract controls

Status: infrastructure correction in progress; Phase 0 remains open.

The path under test is `index → query → interval → evidence → clip`. These
controls use only the existing ten prepared `regression_seen` inputs or generated
synthetic media. They do not add data, train weights, change quality thresholds,
or provide promotion evidence. The frozen policy remains
`e387059314c5c7372520f971c815d03f3cfd181f92135e08e8fdea0bd5340eef`.

## Evidence at 08d23aa

Clean `08d23aa95a09410eb8d6d1c05a07fe507e9f756c` installed and attested all six
environments offline on the target M4 Pro/24 GiB. Its clean backend suite passed
2,298 tests, and the forced publication-failure rollback proof passed. Two full
smokes completed all eight component steps and the five required product
profiles, including MP4 export and both Qwen calls. Both were rejected by the
unchanged resource gate: swapin was 1,092 and 148 pages; swapout and Metal
recovery were zero. Exact raw receipts and their hashes are retained in the
[rejection index](negative-controls/08d23aa-rejections.json). The first is a
first process run in newly installed environments, not a proven host-cold run.
System-wide swapins cannot be attributed to Qwen from these observations.

The [actual batch rejection](negative-controls/08d23aa-product-batch-rejection.json)
exposed a deterministic fixture identity bug. Prefixing a full source SHA with
`reg_` produced 68 characters; both serving indexes allow at most 64. All ten
dense and Lighthouse generations therefore failed before profile evaluation.
The correction uses the full 64-character SHA as the video ID and preserves the
full source identity and existing serving validators. Seven new tests reproduced
the failure and passed after correction; 118 relevant tests passed.

Five object stages also failed. A separate isolated RF-DETR diagnostic proved a
different cause: finite predicted boxes extended slightly beyond a 960×540
frame. One box had `y_min=-0.257652998`; another had `x_max=962.16271973`.
`VisionDetectionResponse` rejected the response as outside the source image.
The same worker completed two other frames and a positive control. The worker
was released, terminated and reaped. The portable [detector diagnostic](negative-controls/08d23aa-detector-bounds.json)
retains response status, checkpoint/runtime identity and numeric counterexamples.
Private diagnostic media and model text are not repository artifacts. The
correction clips valid ordered boxes to the visible frame and discards empty
intersections; malformed coordinates and confidence still fail. Adapter revision
`rfdetr-coco-rgb-clipped-center-box-v3` invalidates incompatible old detector
generations without changing SigLIP identity. The 130 relevant tests passed.

The [direct verifier run](negative-controls/08d23aa-direct-verifier-run.json)
attempted all ten cases. It reported two matches, seven model misses and one
infrastructure error: the first generic case produced invalid JSON. The run is
rejected; these counts are not accepted baseline metrics. In particular, the
raw generation contract and parser normalization required further checking before
interpreting all reported model misses. A separate single-request [generation
diagnostic](negative-controls/08d23aa-qwen-truncation.json), with the same model,
FPS, token limit, temperature and offline inputs, confirmed `finish_reason=length`
and 320 generated tokens at the unchanged 320-token cap. The response had an
unterminated JSON string and containers; its private capture is hash-bound in the
receipt. The worker was terminated and reaped after that one attempt.

An independent parser control also demonstrated that missing fields and invalid
types could normalize into an all-null judgement and then count as a model miss
or a valid abstention match. Explicit schema-valid nulls remain legitimate; missing
or malformed fields must be classified as infrastructure errors before scoring.
The old direct-run counts cannot be repaired from the retained typed receipt and
remain rejected.

## Bounded hypotheses and stopping condition

- Use valid deterministic fixture IDs before starting expensive inference.
- Normalize finite, ordered, intersecting RF-DETR boxes to image boundaries;
  preserve classes, confidence and the detector threshold. Reject malformed or
  non-finite coordinates, and discard boxes with no visible area. Version the
  adapter identity; retain the response contract's strict bounds.
- Use a fresh request-local JSON-schema grammar from the already pinned
  `mlx-vlm==0.6.7` / `llguidance==1.7.6`, with unchanged FPS/token limits,
  temperature and model weights. Require every prompt-specific field, exact
  types, finite bounded numbers and evidence of at most 240 characters. Reject
  unknown or duplicate keys and incomplete generation. The schema, constrained
  decoder and strict parser must be bound into the prompt protocol identity.
  Basketball jersey values follow the already frozen direct prediction contract
  (`0`, `00`, `1–99`, or null); the generic prompt retains its one-to-three-digit
  format. This prevents a schema-valid answer from failing the downstream
  prediction contract without changing labels or benchmark scoring.
  Keep malformed or incomplete output as infrastructure failure.
  Do not repair arbitrary model text into a scored answer or tune semantic
  predictions against these seen labels.

Minimum useful effect is zero execution/contract errors on the complete frozen
path while preserving existing negative validation and fallback behavior.
The output-contract change passed 191 relevant tests, including worker HTTP to
direct-run classification controls; the combined backend suite passed 2,403
tests with two existing deprecation warnings. The installed pinned llguidance
also accepted both final schemas with the hash-verified cached tokenizer and
its EOS, without importing MLX or loading weights. Prompt protocol SHA is
`0519f326e5fbd47784d8cfa3d0fe67b8e6893fbca227b0bdb1a84f5782045d54`.
Required verification is the narrow regression tests, the full backend suite,
and a new committed clean-checkout run on the exact six target environments.
All five measured profiles, ten direct verifier attempts, unchanged resource
guardrails, portable raw measurements and forced generation rollback must pass
the Phase-0 collector before the phase can be marked complete.

The next measured smoke also holds an exclusive lease on agent helper processes:
no parallel shell/Python readers, tests or file scans until its owned workers are
cleaned up. The previous repeat ran alongside an agent's receipt analysis. That
does not establish attribution for its system-wide swapins, but leaves a
controllable source of host activity. User applications remain untouched;
background pressure is neither subtracted nor exempted from the gate.

## Clean ea10ad0 control

Clean `ea10ad0577484c74696e432bc09ef55a73721811` installed and attested all six
environments offline on the same M4 Pro. Its 2,403 backend tests and forced
generation rollback proof passed. With the exclusive helper-process lease, the
[full smoke](negative-controls/ea10ad0-first-smoke.json) completed all eight steps,
five required profiles and export. The unchanged gate still [rejected it](negative-controls/ea10ad0-rejection.json):
315 swapin pages (5,160,960 bytes), zero swapout and Metal recovery, no observed
OOM, peak process-tree RSS 10,265,329,664 bytes and peak system-wide Metal
11,165,057,024 bytes. Neither attribution nor a host-cold state is established.

The [full batch diagnostic](negative-controls/ea10ad0-product-batch-rejection.json)
confirmed ten ready videos, ten completed jobs and all 70 stage attempts complete.
Four profiles persisted audited manifests. Each recorded zero execution failures;
their P@5 was 0.24 and hard-negative hit rate 1.0. A zero model-miss query count
only means relevant intervals were present somewhere in the returned results.
It does not hide false positives or demonstrate useful precision.

The fifth profile, Qwen, failed the search service's pinned-candidate check. The
reranker can refine a confirmed generic event within the inspected source context,
while the service used the mutable output interval as candidate identity. A
synthetic no-ML control reproduced this mismatch. Three logged occurrences do not
constitute a complete case error count, and no Qwen profile or batch receipt was
published. The correction must preserve legitimate refinement while independently
binding every result to one original proposal and retaining its source evidence.

The separate [ten-case direct verifier run](negative-controls/ea10ad0-direct-verifier-run.json)
completed with zero infrastructure errors, three matches and seven model misses.
Its exact typed receipt is retained without private paths or raw generation text.
The worker and runner were cleaned up. This run was diagnostic while independent
agent code review continued; its latency is not an exclusive-host measurement.
These controls do not complete Phase 0: the corrected five-profile batch and the
resource gate must still pass together at a committed clean revision.

The candidate correction uses detached inputs and per-call proposal tokens.
Validation checks one-to-one membership, source video, exact source evidence
types/content and preservation of modality tags. Qwen refinement is checked
against its pre-call source context and corroborating evidence independently of
proposal identity. Tokens are removed before returning results; stable evidence
IDs encode the exact original float intervals. In-place provider failures retain
the original fused candidates and untouched tail. The change passed 146 focused
tests, including 35 independent adversarial controls, then all 2,441 backend tests
with the same two deprecation warnings. A read-only audit also found all 56
retained result intervals from the four prior profiles inside their source
durations; it did not infer missing Qwen outputs.
