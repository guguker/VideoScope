# Accepted Phase 0 baseline — 7054f13

**Phase 0 is complete on the target Apple M4 Pro / 24 GiB.** On September 6,
2026, the unchanged collector accepted all required evidence at clean serving
revision `7054f137f92c18676461242a18c8fc2619898b68`. The final evidence/docs commit
retains the measurements; it does not change that serving revision.

Bundle ID:
`sha256:5abee44ff9d4d85bd2c78de7b89e66668c4a1170aab9861eb5c56537b95111c1`.

The four original collector outputs share that ID:

- [Baseline snapshot](baseline-snapshot.json): frozen code, policy, datasets,
  environments, actual profile execution, batch lifecycle and rollback.
- [Portable raw measurements](raw-measurements.json): product measurements and
  complete accepted smoke observations.
- [Sanitized report](sanitized-report.json): frozen quality/system metrics,
  critical slices, uncertainty and guardrail outcomes.
- [Error ledger](error-ledger.json): model misses separated from infrastructure
  errors and the deliberate rollback failure.

The [evidence index](evidence-index.json) binds all 13 exact collector inputs,
all four outputs and supporting checks by SHA-256 and byte size. Paths are
repository-relative. `inputs/` contains the native smoke receipt, attestation,
five profile manifests, batch receipt, strict verifier and rollback proof. The
policy and two already committed datasets are referenced at their existing paths.

## Accepted control

An owner-confirmed reboot changed the macOS boot identity from `1787056703`
(August 18) to `1788693664` (September 6). It also cleared the former temporary
proof checkout. A new detached checkout of the same `7054f13` was created from
local Git. Six independent environments were installed using the committed
Make targets, offline UV cache and pinned managed Python versions. Existing
approved model files were copied into a new owned model root and verified;
source media and user state were retained.

The [clean-install receipt](clean-install-receipt.json) records the commands and
log hashes. [Fresh environment attestation](inputs/ml-environment.json) is
byte-identical to the attestation used by the existing same-SHA benchmark runs.
Its manifest identity is
`sha256:f4d551a169e981af2279647e900cf7f87b8c174548cd0356c8df5e6cee82c9ac`.

One normal `make full-ml-smoke` then ran with unchanged coverage, models, serving
code and resource gate. The [predeclared plan](fresh-boot-control-plan.json),
[launch receipt](fresh-boot-control-launch-result.json), exact
[native receipt](inputs/full-ml-smoke.json) and
[independent audit](fresh-smoke-independent-audit.json) retain the result.

| Gate or observation | Result |
| --- | --- |
| Component steps | All 8 complete |
| Product profiles | All 5 actually invoked and complete |
| Qwen coverage | Product verification and direct smoke judge complete |
| InternVideo | `not_configured` / `provider_not_configured` |
| Product path | Upload, durable index, search, evidence, FFprobe-verified MP4 |
| Code SHA before/after | Exact clean `7054f13` |
| Cleanup | Complete; root empty; no matching ML workers afterward |
| Swap-in / swap-out | **0 / 0 pages and bytes** |
| Metal recovery / OOM | **0 / not observed** |
| Native host window | 305.153625375 s |
| Enclosing launcher | 310.443847208 s |
| Raw host / RSS samples | 1,099 / 4,853 |
| Process-tree peak RSS | 8,253,521,920 B |
| System-wide Metal in-use peak | 11,172,855,808 B |
| System-wide Metal alloc peak | 13,395,492,864 B |

Every raw host sample has absolute swap and recovery counters equal to zero.
RSS and system-wide Metal are separate, non-additive measurements. One passing
window does not guarantee future workloads or attribute earlier swap events.
The [176-page diagnostic control](../../negative-controls/7054f13/README.md),
separate four-page no-ML control and older controls remain unchanged. No
background counts were subtracted and no inference stage was removed.

## Quality, fallback and rollback

All five same-SHA benchmark profiles completed with zero query infrastructure
errors and confirmed worker retirement. The strict direct verifier attempted
all ten authorized prepared cases: **3 matches, 7 model misses, 0 infrastructure
errors**. Its source results were retained without rerunning or replacing them
following the fresh-boot smoke.

This is an accepted regression baseline, not model promotion. The frozen report
retains 15 failed absolute quality guardrails. P@5 and hard-negative guards fail
for every profile; boundary guards fail for the first four, and Qwen fails the
candidate-recall floor. Four critical slices are represented, seven absent,
and every slice lacks sufficient independent support. The existing product and
verifier datasets remain `regression_seen` / regression-only. No independent
holdout, generalization gain or causal latency improvement is claimed.

The universal lexical/Qdrant path supplies the smoke's exported MP4. Missing
optional providers and rejected learned refinements retain the fallback path;
these cases are covered by the backend tests. The
[forced rollback proof](inputs/phase0-rollback-proof.json) injects
`injected_pre_commit_publication_failure`, restarts readers, and verifies that
the previous seven-stage release remains active and searchable, the candidate
remains inactive, source bytes are unchanged and no source reindexing is needed.

After the accepted smoke, the new clean checkout ran the entire backend suite
from its root: `./.venv/bin/pytest backend/tests` — **2,527 passed, 2 warnings,
44.62 s**, including the forced rollback test. The
[test receipt](backend-tests-receipt.json) records the exact command and private
log hash. The [whole-objective audit](phase0-completion-audit.json) independently
maps 13 requirements to 38 source/test bindings and rebuilds all four bundle
payloads. It was captured before final repository retention and commit; the
replay below verifies the retained copies.

## Reproduce the retained report without inference

Use a clean detached `7054f13` checkout with its pinned base environment. From
this repository, run:

```sh
./.venv/bin/python -I \
  docs/benchmarks/phase0/accepted/7054f13/verify-bundle.py \
  --checkout /absolute/path/to/clean-7054f13-checkout
```

The [replay script](verify-bundle.py) verifies every retained hash, calls the
unchanged committed `scripts/phase0-evidence.py` from that clean checkout, and
requires byte-identical outputs for all four files. It creates a disposable
output directory and removes only that owned directory. It starts no models.
The collector still validates its own clean Git identity before publication;
no substitute identity resolver or bypass is used.

For a fresh capture, the committed setup commands are `make install-backend`,
`make install-vision`, `make install-whisper`, `make install-video`,
`make install-lighthouse`, `make install-ocr` and `make ml-attest-offline`.
Use `UV_OFFLINE=1` and `UV_PYTHON_DOWNLOADS=never` with already approved local
models and the pinned cache. The [capture wrapper](capture-normal-smoke.py)
records the exact normal smoke orchestration used here; copy it beside the
owned `checkout/`, `models/`, `private/` and empty mode-0700 `smoke-root/`
directories. Its `--preflight-only` mode starts no inference. The recorded
attestation must exist as `private/ml-environment.json` before capture.

Product batch and strict direct verifier commands remain the committed CLIs in
[product retrieval](../../../product-retrieval/README.md) and
[video verifier](../../../video-verifier/README.md); the complete collector
interface is documented [here](../../README.md). Reproduction retains the same
frozen input bindings, model revisions, protocol, and resource gates.

## Decision

Accept this evidence bundle as the frozen Phase 0 baseline. Preserve the
last-known-good user path and all negative controls. No training, model
promotion, external data import/upload or later phase was performed. A Phase 1
read-only data/provenance audit is the next proposed bounded goal and requires
separate owner authorization.
