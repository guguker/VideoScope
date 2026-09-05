# Phase 0 smoke timeline diagnostic

The `1b11a54` owner-ready control still observed 116 swapin pages. All occurred
before the late Metal rise, and 64 occurred within the first 5.085 seconds.
The previous aggregate RSS samples had no timestamps or role partition, and the
smoke receipt had no stage boundaries. These observations do not establish a
Lighthouse leak or identify which process caused a system-wide swapin.

The bounded hypothesis is that a shared monotonic timeline can localize the
remaining events to startup, a component call or a product phase, while exposing
the resident RSS of the owned workers. This is a measurement change only:
inference order, model residency, prompts, frozen metrics, sample cadence,
measurement window and zero-swap acceptance gate remain unchanged.

## Run one diagnostic control

Use the same clean checkout, six installed environments, reviewed offline model
roots and explicit environment variables as `make full-ml-smoke`. Select a new
timeline file outside the checkout, home, smoke workspace and all model/cache
roots. Its existing immediate parent must be an owner-only `0700` directory;
symlinks in any ancestor are rejected.

```sh
export VIDEOSCOPE_FULL_ML_SMOKE_TIMELINE="$phase0_private/full-ml-timeline.json"
make full-ml-smoke-diagnostic \
  > "$phase0_private/full-ml-smoke-diagnostic.json" \
  2> "$phase0_private/full-ml-smoke-diagnostic.stderr.log"
```

The caller must choose new stdout/stderr destinations as well. The runner
reserves its timeline with `O_EXCL` before inference, writes mode `0600`, and
never replaces an existing file. The optional target shares the ordinary
smoke's environment recipe and invokes the same native main. It captures and
replays the native schema-2 stdout without changing its bytes. The separate
timeline binds those bytes by `smoke_stdout_sha256` and byte size, and records
the native exit code. There is no model download, dataset import or training.

## Read the timeline

The sidecar contains fixed stage IDs for worker startup/readiness, each direct
component call and release, product indexing, all five profiles, runtime close,
export and cleanup. `begin`, `end` and `error` describe the observed operation
boundary. They contain no query, file path, model text or exception message.

Each process observation reuses the already validated native snapshot at the
existing RSS cadence. It adds the snapshot's start/end time, exclusive role RSS,
process counts and hashed lifetime/executable identities. No second native
snapshot or sampling thread is introduced. The `owner` bucket includes the
backend and unmanaged descendants, including transient OCR/media processes;
it does not assert that every byte belongs to backend Python. The managed
worker subtrees are counted once each. These identities prove process lifetime,
not which model is loaded or which process caused swap.

The following offsets share one monotonic coordinate system:

```text
host sample position = host_sampler_started_elapsed_ns
                     + raw_host_sample.elapsed_nanoseconds
```

Compare that position with event timestamps and process snapshot ranges. Do not
align RSS by sample index or nominal cadence. A swap counter increment lies
between host observations; timestamps do not establish the exact fault instant.
Temporal association alone does not prove causality. Host swap/recovery totals
remain intact, and Metal is not added to process RSS.

## Failure and acceptance

Diagnostic event failures latch a path-free failure flag. They cannot skip
mandatory cleanup or discard ownership of an already started worker. Process
observation failures retain the existing fail-closed measurement behavior.
The wrapper preserves an original nonzero smoke exit; otherwise diagnostic
failure returns nonzero and marks the sidecar failed. A sidecar write failure
can leave a reserved empty or partial file, which is not a valid diagnostic.

A native `ready` receipt can therefore coexist with a failed wrapper/sidecar;
check both outcomes before interpreting a diagnostic. The sidecar is explicitly
`diagnostic_only_not_gate_evidence` and is never substituted for the formal raw
measurements or an accepted Phase-0 bundle. Preserve every control. Use the
new temporal evidence to choose one bounded follow-up, without subtracting
background pressure, changing coverage, or repeating smoke blindly.

## Validation before the clean control

The focused controls cover shared clock alignment, exclusive role accounting,
no additional native snapshots, create-once sidecar output, byte-identical native
stdout, isolated Python entrypoints and failure ownership. Adversarial event
failures cannot skip cleanup or lose a started worker. The full fake-product
path produces the same receipt and operation order with diagnostics enabled or
disabled, including five profiles and export. An independent AST comparison
confirmed that trace wrappers preserve the product operations.

From the repository root, `./.venv/bin/pytest backend/tests` passed **2,527 tests**
with two existing deprecation warnings in 38.38 seconds. `git diff --check` passed.
The real target-host diagnostic control remains pending; these tests alone do
not establish model residency, eliminate swap or satisfy the Phase-0 exit gate.
