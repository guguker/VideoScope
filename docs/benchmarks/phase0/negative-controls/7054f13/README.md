# Clean 7054f13 diagnostic controls

**Phase 0 remains open.** These are retained diagnostic and rejected resource
controls, not an accepted baseline bundle. The serving and measurement source
was clean `7054f137f92c18676461242a18c8fc2619898b68` on the M4 Pro/24 GiB.
All six environments were freshly installed offline and attested. The clean
backend suite passed 2,527 tests with two existing warnings in 86.61 seconds.
Forced generation rollback retained the previous searchable release after
restart, with unchanged source identity and no source reindexing.

The [control record](control-record.json) binds exact retained bytes and
separates resource failures, the invalid diagnostic driver attempt and model
quality. The five-profile batch and ten-case direct verifier previously retained
at [1b11a54](../1b11a54/README.md) are not measurements of this revision and cannot
be mixed into a `7054f13` bundle. Preparing their same-revision replacements is
still required. Neither diagnostic below used those user clips; the full smoke
used its existing synthetic fixture and the second control ran no inference.

## Complete diagnostic smoke, failed resource gate

The [native receipt](full-ml-smoke-diagnostic.json) completed all eight component
steps, all five product profiles, both Qwen checks and MP4 export. InternVideo
was `not_configured`; before/after source SHA matched. Cleanup completed and the
disposable smoke root was empty. The optional wrapper and
[timeline](full-ml-timeline.json) both completed without diagnostic errors.

| Observation | Exact result |
| --- | ---: |
| Host measurement window | 280.401613750 s |
| Enclosing launcher | 281.920 s |
| Process-tree RSS peak | 10,575,396,864 B |
| System-wide Metal in-use peak | 11,141,300,224 B |
| Swap-in | 176 pages / 2,883,584 B |
| Swap-out / Metal recovery | 0 / 0 |
| OOM | not observed |

RSS and Metal are separate and must not be added. The unchanged collector's
`validate_full_ml_smoke` rejects this receipt with `full_ml_smoke_invalid`.
The full four-file collector was not run with mismatched older benchmark inputs;
no accepted bundle was created.

The [derived analysis](diagnostic-analysis.json) preserves all 18 swap-counter
observation windows, overlapping stages and adjacent measured worker RSS. All
4,808 timeline totals exactly match the formal RSS vector; exclusive role sums
and lifetime identities are valid. Seventy events form 35 completed stages.

| Overlapping leaf stage | Swap-in pages |
| --- | ---: |
| RF-DETR | 45 |
| OCR read | 12 |
| Lighthouse generation | 12 |
| Product setup / indexing | 4 / 8 |
| Temporal refinement | 4 |
| Qwen profile / final direct smoke call | 20 / 71 |

These are temporal associations, not per-process swap attribution. RF-DETR
observations occurred with about 1.37 GiB total owned RSS, before Qwen. The RSS
peak occurred during product indexing at 184.629 seconds: the `owner` bucket
held 7,428,308,992 B across three processes and Lighthouse held 1,602,486,272 B.
The owner includes unmanaged descendants such as OCR/media processes; this
does not identify tensor ownership.

The first 171.244 seconds preceded product integration. OCR read took 48.076 s,
Lighthouse generation 35.404 s, Whisper health 24.719 s, and Vision image and
RF-DETR calls about 20.7 s each. The old 103-second smoke had no stage timings
and a different preceding run order, so neither a causal latency regression nor
diagnostic callback overhead can be inferred from that comparison. Snapshot
acquisition timing excludes callback and scheduling time. Host observations
also have acquisition uncertainty; boundary and uncovered-margin attribution
must remain explicit.

## Fixed control without model inference

The [corrected control](no-ml-control.json) ran the same committed native host
and process samplers, at their existing 250/50 ms cadences, for a predeclared
120 seconds. It loaded no models and started no managed workers. Every one of
2,088 RSS snapshots contained only its owner process, with a stable identity;
the peak was 69,533,696 B. Both samplers finished and stopped.

The system-wide counter still increased by **4 pages / 65,536 B**, with zero
swap-out and recovery. The observed host interval was
`(5.916153250, 6.171945625]` seconds; elapsed time after baseline was
120.006116208 seconds. This establishes that an increment can occur without
model inference in the controlled process tree. It does not identify the
causing process, establish an absence of every possible model elsewhere on the
Mac, or permit subtraction of these pages from the full smoke.

The first ad hoc no-ML driver called a nonexistent `HostResourceSampler.close`
after `finish` and failed before publication. Its
[invalid record](no-ml-control-invalid.json), original
[plan](no-ml-control-invalid-plan.json) and
[driver](no-ml-control-invalid-driver.py) remain available. The empty output
contains no measurements: its counts are unavailable, not zero. The
[corrected driver](run-no-ml-control-v2.py) uses the existing `finish` lifecycle,
stops both samplers and preserves an original failure. Three synthetic-provider
tests passed, including secondary cleanup failure; they are not native evidence.

## Replay and next control

From the repository root, with the attested backend environment:

```sh
./.venv/bin/python -I \
  docs/benchmarks/phase0/negative-controls/7054f13/verify-control.py

TEST_NO_ML_DRIVER="$PWD/docs/benchmarks/phase0/negative-controls/7054f13/run-no-ml-control-v2.py" \
  ./.venv/bin/pytest \
  docs/benchmarks/phase0/negative-controls/7054f13/test_no_ml_driver.py \
  docs/benchmarks/phase0/negative-controls/7054f13/test_no_ml_driver_lifecycle.py -q
```

The replay verifies hashes, environment, rollback, raw counter consistency,
timeline/RSS agreement and the expected rejection by the unchanged smoke gate.
It runs no inference and creates no accepted bundle. The tests use synthetic
providers and a controlled clock, not another 120-second host measurement.

After resumption on September 6, the [separate host read](host-state-after-resumption.json)
still reported the August 18 boot and 1,798.75 MiB of used swap. This observation
was outside the measured windows and cannot correct their counts. No owner
application was closed or altered.

The next environmental control is one unchanged normal full-ML smoke after an
owner-provided fresh macOS session. A reboot tests a host-state hypothesis; it
does not promise zero swap or excuse a failing result. Complete the same-SHA
benchmark inputs independently, retain every control and keep the zero-swap
gate unchanged. A baseline can be accepted only when all required inputs at one
clean SHA pass the full collector. No training, data import/upload, promotion or
later phase has started.
