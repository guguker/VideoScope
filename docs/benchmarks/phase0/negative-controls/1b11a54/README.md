# Clean 1b11a54 rejected baseline control

Phase 0 remains open. These files retain exact observations from the clean
`1b11a5420e2fc66417f314c03192c1a853f8c843` checkout on the target M4 Pro/24 GiB.
They are **not an accepted Phase-0 bundle** and are not promotion evidence.
Only the previously frozen `regression_seen` clips and synthetic smoke media
were used. No weights were trained, external data imported or user state changed.

[rejection.json](rejection.json) binds the twelve exact source receipts by byte
size and SHA-256. It records the actual collector exit code 3, no bundle output,
the independently checked contracts and six observed swapin events. The separate
[diagnostic error ledger](error-ledger-diagnostic.json) distinguishes the resource
failure from product/direct execution errors and model-quality failures.

- [Environment](ml-environment.json): six freshly installed, attested offline
  environments. The clean backend suite passed 2,441 tests with two warnings.
- [Rollback](phase0-rollback-proof.json): injected publication failure retained
  the prior searchable release after restart without source changes or reindexing.
- [Batch receipt](phase0-baseline-receipt.json): all five profile manifests below,
  exact hashes, completed worker retirement and cleanup.
- [lexical_qdrant](lexical_qdrant.json), [dense_siglip](dense_siglip.json),
  [temporal_refinement](temporal_refinement.json), [lighthouse](lighthouse.json)
  and [qwen_verification](qwen_verification.json): complete raw measurements,
  every frozen metric/slice and portable result intervals; zero execution errors.
- [Direct verifier](video-verifier-run.json): all ten typed predictions and
  latencies; three matches, seven model misses, zero infrastructure errors.
  [Launch receipt](direct-verifier-launch-result.json) confirms cleanup.
- [First full smoke](full-ml-smoke-first.json): eight completed component steps,
  five profiles, export, InternVideo `not_configured`, matching before/after SHA
  and complete cleanup. Rejected for 446 swapin pages; swapout/recovery were zero.
- [Predeclared run order](run-order.json): batch, direct verifier, then smoke;
  no simultaneous agent helper processes during measured runs. Host-cold state
  and ownership of the system-wide swapins are not established.

The exact raw JSON files contain stable IDs, hashes and numeric/typed
observations. They contain no private paths, queries, source media, credentials
or raw generated text. Source clips and private logs remain outside Git.
The full quality interpretation is in the
[serving control report](../../serving-contract-controls.md#clean-1b11a54-control).

To reproduce the rejection without inference, run the unchanged collector from
a clean checkout of the SHA above, with its installed backend environment.
Point `phase0_control_receipts` to this retained directory outside that checkout
and `phase0_rejected_output` to a new, nonexistent output directory:

```sh
.venv/bin/python -I scripts/phase0-evidence.py \
  --policy docs/benchmarks/policies/phase0-regression-v1.json \
  --product-dataset docs/benchmarks/product-retrieval/seed-v1.json \
  --verifier-dataset docs/benchmarks/video-verifier/seed-v1.json \
  --environment-attestation "$phase0_control_receipts/ml-environment.json" \
  --full-ml-smoke "$phase0_control_receipts/full-ml-smoke-first.json" \
  --baseline-batch "$phase0_control_receipts/phase0-baseline-receipt.json" \
  --video-verifier-run "$phase0_control_receipts/video-verifier-run.json" \
  --benchmark-run lexical_qdrant="$phase0_control_receipts/lexical_qdrant.json" \
  --benchmark-run dense_siglip="$phase0_control_receipts/dense_siglip.json" \
  --benchmark-run temporal_refinement="$phase0_control_receipts/temporal_refinement.json" \
  --benchmark-run lighthouse="$phase0_control_receipts/lighthouse.json" \
  --benchmark-run qwen_verification="$phase0_control_receipts/qwen_verification.json" \
  --rollback-proof "$phase0_control_receipts/phase0-rollback-proof.json" \
  --output-dir "$phase0_rejected_output" \
  --code-sha 1b11a5420e2fc66417f314c03192c1a853f8c843
```

Expected: exit code `3`, `invalid_evidence`, and no output bundle. An accepted
baseline requires an unchanged full smoke passing the strict resource gate and
successful collection of all four shared-ID artifacts. The next bounded control
is one run after the owner confirms a quiet host window. Preserve every rejected
attempt and do not change the gate or infer that time elapsed means host readiness.

The [repository replay receipt](repository-replay.json) records a fresh invocation
using these twelve copied files: byte hashes and local links passed, the unchanged
collector again returned `3`, and no output directory was created.
