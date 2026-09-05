# VideoScope project instructions

## Project mission

VideoScope is a local-first application for multimodal natural-language search of
precise moments in long user videos and for assembling those moments into MP4
clips. The product remains universal: basketball is the first demanding domain
profile, not a permanent product boundary.

Preserve the complete user path:

```text
local video → indexing → natural-language query → ranked precise interval
→ inspectable evidence → preview/correction → MP4 export
```

Do not optimize a component metric at the expense of this path.

## Mandatory ML operating contract

Before planning, changing, evaluating, or documenting any ML-related behavior,
read these files in order:

1. `docs/ml-autonomy-plan.md` — primary ML strategy, phase gates and stop rules;
2. `docs/system-rebuild.md` — artifact, benchmark, job and rollback invariants;
3. `docs/local-model-strategy.md` — current hardware-tested model decisions;
4. `docs/architecture.md` and `docs/user-journeys.md` — system and product contract;
5. `docs/basketball-model-upgrade.md` when the work touches the sports profile.

Treat `docs/ml-autonomy-plan.md` as the strategic source of truth for ML work.
Do not silently weaken its constraints or skip an exit gate. A direct user request
may revise the plan; when that happens, update the document in the same change so
future work does not inherit a stale decision.

## Goal discipline

- Work toward one numbered phase or one bounded vertical slice at a time.
- At the start of a goal, name the phase, current evidence, exact deliverables,
  validation commands and verifiable stopping condition.
- Determine progress from repository artifacts and fresh checks, not chat memory.
- Do not advance because time passed or code exists; advance only when the current
  phase's exit gate is demonstrated.
- If a gate cannot yet be satisfied, record the blocker and leave the system on the
  last-known-good path. Do not disguise incomplete work as a completed phase.
- Never start model training merely because an active goal says “ML autonomy”. The
  data, benchmark, provenance and candidate-recall prerequisites still apply.

## Required loop for every ML iteration

1. State which part of `query → interval → evidence → clip` should improve.
2. Inspect Git state, active artifacts, runtime identity, dataset revision and the
   frozen baseline. Preserve unrelated user changes.
3. Form one falsifiable hypothesis and change one principal component.
4. Freeze prediction contract, primary metric, critical slices, guardrails and
   minimum useful effect before implementation or training.
5. Verify rights/consent, provenance, source/replay groups and split isolation.
6. Add or update schemas, golden fixtures and tests before behavioral code.
7. Keep train and serve preprocessing in one versioned implementation.
8. Run the cheapest meaningful control before a larger backbone or adapter.
9. Record code SHA, dataset/split revision, base-model revision, dependencies,
   config, seed, hardware, preprocessing hash, artifact hashes and raw metrics.
10. Evaluate proposal, component and end-to-end product behavior on frozen data;
    report every critical slice, uncertainty, latency, memory and infrastructure
    failure separately.
11. Perform ablation and error analysis. Reject a candidate that fails any hard
    gate; do not tune against the final holdout.
12. Use `offline → shadow → canary → explicit promotion`; verify rollback before
    canary and retain the previous artifact and compatible feature generation.
13. Put clicks and model-generated labels into an annotation inbox only. Gold
    labels require human confirmation, with adjudication for critical conflicts.
14. End with a short decision log: evidence, decision, remaining risk and the next
    highest-value bounded experiment.

## Non-negotiable boundaries

- The universal retrieval path and upload/search/preview/export workflow must work
  when every learned domain component is disabled or unavailable.
- User source media, labels, indexes and exports are user state. Never silently
  delete, replace, upload or repurpose them as training data.
- Production inference is offline by default. External services and cloud training
  require explicit authorization for the exact data and destination.
- Split by whole source/game/event/replay groups. Related windows, crops, audio,
  OCR and embeddings stay in the same split. Any leakage invalidates the holdout.
- Existing studied videos and the current ten cases remain `regression_seen`, never
  promotion evidence.
- Teacher, heuristic and pseudo labels are not gold and cannot evaluate a model
  trained from those labels.
- Learned scores do not replace evidence, calibrated uncertainty or honest
  `insufficient_evidence`/abstention.
- Build derived generations beside active data and publish atomically only after
  validation. No candidate may require source reindexing to roll back.
- Pin external model/dataset revisions and verify licenses, provenance and hashes.
  Prefer `safetensors`; do not enable arbitrary remote code or unknown pickle.
- Reject any production artifact that causes Metal OOM, sustained swap growth,
  train/serve mismatch, unbounded latency or degradation beyond frozen guardrails.
- Do not train a foundation model from scratch. The first owned model is a small
  query-conditioned temporal ranker on frozen features. X-CLIP is an optional
  challenger, RF-DETR a separate perception component, and Qwen QLoRA a later
  option only after a verifier-specific bottleneck is proven.
- Basketball weights initially activate only through the basketball domain pack.
  They may not replace universal ranking without a separate multi-domain holdout.
- Automatic retraining may build and evaluate a candidate, but changing `active`
  requires explicit owner approval until the project separately approves L4.

## Current next phase

The default next objective is Phase 0 of `docs/ml-autonomy-plan.md`: establish a
reproducible and trustworthy baseline before training. Phase 0 remains open:
the clean `1b11a54` control completed all five profiles and direct verifier, but
its full smoke failed the unchanged zero-swap gate. Exact evidence is retained in
`docs/benchmarks/phase0/negative-controls/1b11a54/README.md`; no accepted baseline
bundle, training or later phase is active. The active scope is Phase 0 only;
a Phase 1 read-only dataset audit requires a separately authorized goal.
No external or user data may be imported,
uploaded or trained on merely because it is listed as a candidate.

When Phase 0 is demonstrably complete, update both the plan's status and this
section before proposing Phase 1 as the next active goal.

## Verification and handoff

- Test in proportion to the change; for documentation-only changes, validate
  links, formatting and consistency with existing contracts.
- For code or ML changes, run the narrow tests first and the relevant full suite
  before declaring a gate satisfied.
- Full-ML promotion evidence must come from the target M4 Pro and the exact
  artifact/runtime intended for serving.
- Report what changed, what was actually verified, what was not run, current phase
  status, rollback state and the next bounded action.
