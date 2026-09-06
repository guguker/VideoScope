# Phase 0: bounded OCR lifetime

Status: historical implementation control. Phase 0 subsequently completed at
clean `7054f13`; see the [accepted baseline](accepted/7054f13/README.md). The
rejections and decisions below describe the earlier OCR/lifecycle iteration and
remain unchanged as evidence. The accepted baseline provides no model-promotion
or training authorization.

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

## Clean-checkout result

Clean commit `8d473c655dd1b4bac87d777f7704121eec81b6a5` installed all six
environments offline from the existing cache. Environment attestation completed
on the target M4 Pro, with Python 3.12.13 for base/vision/whisper/OCR/Qwen and
3.11.14 for Lighthouse. The clean backend suite passed 2,267 tests with the two
existing warnings. The forced pre-commit publication failure verified that the
prior generation remained active and searchable after restart, without source
changes or reindexing.

The full smoke passed the product path and five profiles but failed its final
standalone Qwen request. Its compact infrastructure error did not preserve the
resource receipt. Private diagnostic repetitions of the same workload isolated
HTTP 409, `input changed during inference`; OOM remained unknown. An additional
instrumented run found only the `product` ancestor changing size/mtime/ctime;
the source, root, immediate parent and private materialized input fingerprints
were unchanged. This is consistent with deferred removal of SQLite sidecars,
not modified source media.

[Portable negative diagnostic](negative-controls/ocr-lifetime-8d473c6.json)
records the measurements and stage observations, explicitly as diagnostic-only
evidence. Standalone OCR used 5,626,511,360 bytes at 22.488 seconds and its PID
was absent by 22.565 seconds. No OCR child remained at product completion.
Peak process-tree RSS was 10,944,692,224 bytes; swapin was 11,730,944 bytes and
swapout 342,097,920 bytes. Swapouts occurred during the product stage, before
the final Qwen failure. Stage and host clocks have different origins and there
are no subprofile timestamps; these measurements do not attribute the burst
exactly to a model. They reject OCR retirement as a sufficient memory fix.

Decision: retain the verified OCR ownership correction; keep Phase 0 open.
Next bounded infrastructure controls are deterministic SQLite connection
closure (the strict input lease must remain unchanged), terminal release of
FastEmbed sessions owned by closed runtimes, and durable sanitized negative
smoke receipts. The FastEmbed hypothesis is that live sessions retained through
stale runtime references add avoidable memory at the first Qwen request. Its
minimum useful effect is confirmed release after in-flight operations finish,
unchanged profiles/predictions, and the same complete zero-swap smoke gate.
Frozen data, metric policy, model revisions and the 16 GiB budget are unchanged.

The SQLite control reproduced the exact 409 without ML: collecting an open
connection after its context exited removed WAL/SHM during fake inference while
the source hash and metadata remained unchanged. Closing the connection after
commit/rollback made that test pass, without changing the Qwen lease.
Commit `f1db3c5` passed 240 repository/generation/worker tests.

Commit `45110d0` makes embedding close terminal, waits for in-flight loading or
encoding, and retains snapshots on a bounded busy-close failure for retry.
Closed writers and benchmark environments release their owned FastEmbed model
even when a stale runtime reference remains. Failed initialization also clears
completed traceback frames retaining the session before removing its snapshot.
The 141 focused tests passed; actual RSS/swap benefit still requires measurement.

Negative smoke receipts preserve independently validated measurements and
cleanup results on typed errors; the 148 smoke/evidence tests passed. Ready
schema 2 and acceptance checks are unchanged. The combined backend suite passed
2,298 tests with two existing warnings. The frozen policy validation still
resolves to `e387059314c5c7372520f971c815d03f3cfd181f92135e08e8fdea0bd5340eef`.
The next check is a new clean six-environment install and complete target smoke;
these unit/integration results do not close Phase 0.
