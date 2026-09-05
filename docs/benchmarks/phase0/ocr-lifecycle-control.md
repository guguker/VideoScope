# Phase 0: bounded OCR lifetime

Status: implementation control; Phase 0 remains open.

## Frozen experiment

The user-facing path under test is `index → query → evidence → clip` on the
target Apple M4 Pro with 24 GiB unified memory. This slice changes only the
lifetime of the existing isolated OCR child. OCR weights, preprocessing,
predictions, search profiles, frozen datasets and metric thresholds are unchanged.
No training, dataset import, source reindexing or external inference is involved.
Only generated synthetic media is used by the constellation control. Existing
ten-case datasets remain `regression_seen`, never promotion evidence.

Baseline: clean `be2a4d235e304d5d3b4e35275aff10ff061a89b7`, proof14. Its full
smoke was functionally ready but rejected: 350 swapin pages, 54,232 swapout
pages (888,537,088 bytes), zero Metal recovery, no observed OOM, and
11,679,711,232-byte peak process-tree RSS. Host samples show swapout increasing
between 95.467 and 101.102 seconds. The old receipt has neither stage timestamps
nor per-PID RSS; it cannot attribute that burst to a particular worker. Lighthouse
uses CPU and must not be described as a Metal-resident model.

Hypothesis: OCR children retained after ingestion create avoidable memory
pressure during the later Qwen stage. `build_runtime` constructs an OCR reader
but the previous shutdown did not close it. The smoke also kept its standalone
OCR reader through product indexing and all query profiles. Dropping the local
runtime variable did not remove the coordinator/queue/indexer references.

Primary resource metric and minimum useful effect: all OCR child processes must
have confirmed exit at their stage boundary, and the complete unchanged smoke
must record zero swapins, swapouts and Metal recovery, no OOM and peak process
RSS at most 16 GiB. All five required profiles and eight component steps must
still execute. A repeated pressure breach rejects this hypothesis as a sufficient
fix; it does not permit weakening the gate or removing verifier coverage.

## Production change

The Indexer releases OCR in `finally` after frame reads, including cancellation
and failure, before publishing its segment generation. An unconfirmed release
is an infrastructure failure and aborts the attempt, preserving the previously
active release. Ordinary OCR inference failures remain optional-stage failures.
The next OCR stage may start a freshly attested child from the same local weights.

The reader retains process ownership until exit is confirmed. Failed retirement
keeps the handle and private bundle, blocks protocol reuse, and can be retried.
Runtime shutdown additionally owns and closes its exact reader after dispatcher
and GC stop, before releasing storage ownership. The standalone smoke uses the
same production release method; its final cleanup remains a safety net.

## Verification and decision

Required commands from the repository root:

```bash
./.venv/bin/pytest backend/tests/test_paddle_ocr_attestation.py \
  backend/tests/test_runtime_lifecycle.py backend/tests/test_runtime_workers.py \
  backend/tests/test_indexer.py backend/tests/test_indexer_durable_jobs.py \
  backend/tests/test_indexer_generations_integration.py \
  backend/tests/test_indexer_workers.py backend/tests/test_full_ml_smoke.py
./.venv/bin/pytest backend/tests
```

Then record the real synthetic constellation control, commit the coherent
implementation, create a new clean detached checkout, and run `make
ml-attest-offline` and `make full-ml-smoke` with the documented explicit roots.
Proof14 stays as negative evidence. The full Phase-0 evidence collector and
forced generation rollback are still required after this resource gate.

Initial diagnostic observation: after one synthetic OCR read the child had
5,590,155,264-byte RSS. With a second reader, their observed RSS values were
4,963,729,408 and 5,600,690,176 bytes. That control stopped at a rejected duplicate
Vision input path before Qwen; all children were shut down. These point samples
establish the cost of retained OCR, not a passing full-smoke measurement.

The corrected synthetic control explicitly retired each OCR reader before
starting the next. OCR PIDs disappeared at 25.078 and 28.381 seconds, after
observed RSS of 5,594,398,720 and 5,604,442,112 bytes. SigLIP plus retained CPU
Lighthouse then completed both Qwen storyboard and native-video requests.
Measured peak process-tree RSS was 9,988,177,920 bytes; system-wide Metal in-use
peak was 11,315,838,976 bytes. Swapout and Metal recovery were zero; swapin was
16 pages (262,144 bytes). This is a diagnostic control, still not an accepted
zero-swap smoke or proof of attribution to a single worker.

Verification: 226 focused tests passed; the complete backend suite passed
2,267 tests with two existing deprecation warnings. Measured line coverage of
the three changed production modules was 87% (Indexer 86%, OCR 85%, Runtime
89%). The development test interpreter was Python 3.12.14; it is not substituted
for the separately required clean-checkout Python 3.12.13 attestation.
Independent review found no blocking lifecycle/rollback defect.

Decision: retain this bounded ownership fix and validate it in a new clean
checkout. The next experiment is the complete unchanged-coverage smoke with
stage-scoped OCR lifetime. Phase 0 remains open; forced generation rollback
and the final evidence bundle are not yet demonstrated for this revision.
