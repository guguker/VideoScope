# Frozen Phase-0 metric policy

`phase0-regression-v1.json` is the machine-readable quality contract frozen
before further ML work. It binds the product and verifier regression fixtures by
their canonical SHA-256 revisions and records the exact benchmark methodology, schema,
measurement-protocol, profile, and search-plan revisions against which a
baseline may be described.

The data status is intentionally fail-closed: all ten verifier cases and the
five product cases are frozen `regression_seen`, and
`promotion_eligible` is `false`. This artifact is suitable for regression and
Phase-0 completeness checks only. It is not a promotion policy and cannot turn
the existing seen fixtures into holdout evidence. The separate benchmark
`PromotionPolicy` remains the deterministic run-comparison contract; a future
promotion decision additionally requires a new independent sealed holdout and
the uncertainty evidence declared here.

The policy freezes:

- proposal, component, product, reliability, and system primary metrics;
- one-to-one maximum-cardinality matching for overlapping gold intervals, so
  one returned interval can never inflate recall or boundary coverage for two
  distinct events;
- basketball, capture-condition, distribution-shift, OOD, and universal
  critical slices;
- absolute quality/resource gates and maximum slice regressions;
- the `+0.03` nDCG@10 or Recall@10 minimum useful effect versus a
  frozen-feature control;
- a 95% whole-source-group bootstrap interval that excludes zero, with at
  least 20 completed gold cases and at least two source groups per critical
  slice; unavailable uncertainty yields `insufficient_evidence`.

Validate the artifact, its live code contracts, and its exact dataset binding
without accessing product or user data:

```bash
PYTHONPATH=backend/src .venv/bin/python \
  -m videoscope.benchmark.metric_policy validate \
  --policy docs/benchmarks/policies/phase0-regression-v1.json \
  --verifier-dataset docs/benchmarks/video-verifier/seed-v1.json \
  --product-dataset docs/benchmarks/product-retrieval/seed-v1.json
```

Success writes one compact JSON object containing only status, policy id,
policy revision, dataset revision, and `promotion_eligible: false`. Inputs are
bounded regular files; duplicate/unknown fields, non-finite numbers, symlinks,
dataset drift, and live methodology/schema/profile drift fail validation.

Changing a metric, threshold, slice, evidence rule, dataset binding, or runtime
contract requires an explicit policy/schema revision and review. Do not edit a
policy after observing candidate results; add a new version instead.
