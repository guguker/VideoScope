# Phase 0 ML environment attestation

`make ml-attest-offline` is the read-only, fail-closed environment gate for the
Phase 0 benchmark. It emits exactly one sanitized JSON object and does not
download models or packages, import an inference runtime, start a worker, or
write under `data/`.

The command starts from an empty process environment and supplies only the
offline policy recorded in `workers/ml-environment.lock.json`. It checks:

- Darwin/arm64, Apple M4 Pro, 24 GB unified memory and macOS 14 or newer;
- the raw SHA-256 identity of every dependency lock and the raw plus canonical
  JSON identity of the committed Qwen and OCR manifests;
- CPython `3.12.13` in `.venv`, `.venv-vision`, `.venv-whisper`, `.venv-ocr`
  and `.venv-qwen`, and CPython `3.11.14` in `.venv-lighthouse`;
- the normalized installed-distribution identity of each present environment;
- `uv==0.12.3` and the SHA-256 identity of its executable; and
- the immutable runtime and model identities attached to each declared
  capability.

The base distribution identity was produced from the committed `backend/uv.lock`
resolution for the `dev` extra on CPython 3.12.13/Darwin/arm64. It is not a
snapshot of the currently installed `.venv`. Worker identities come from their
hash-pinned requirements locks; Qwen uses the exact distribution map in its
composite runtime manifest. Lighthouse's two direct URL distributions are
explicitly frozen as `clip==1.0` and `lighthouse==0.1` in addition to its
requirements lock. Bootstrap packages `pip`, `setuptools`, and `wheel` are not
part of distribution identity because they are not runtime capability inputs.

## Status contract

Exit code `0` means `complete` or `partial`. Exit code `2` means `failed`.

- `complete`: every declared environment and contract matches.
- `partial`: no drift or infrastructure error exists, but at least one optional
  environment and its capability are `not_configured`.
- `failed`: at least one `drift` or `infrastructure` failure exists.
- `not_configured`: an environment directory is absent. This is intentionally
  distinct from an existing but broken environment.
- `drift`: observed host, Python, package set, lock, manifest, offline policy or
  `uv` identity differs from the frozen contract.
- `infrastructure`: an expected object exists but cannot be safely read or
  executed, or a bounded probe fails.

Reports contain only stable IDs, public model/runtime identities, versions,
hashes and diagnostic codes. They never contain local paths, environment values,
tokens, serial numbers, UUIDs, device numbers or inode numbers. Probe stderr and
exception text are deliberately discarded.

The attestation proves the environment and committed manifests, not serving
readiness. It does not hash multi-gigabyte cached model bytes or run inference.
Each provider's existing startup attestation remains responsible for verifying
the local model artifacts before the capability can serve requests. Therefore a
`complete` capability here is necessary but not sufficient evidence for an
end-to-end benchmark run.

## Clean rebuild gate

An installed uv-managed interpreter by itself is only a provisioning
prerequisite; it is not an attested environment. From a clean checkout with
`uv==0.12.3`, FFmpeg and FFprobe already on `PATH`, create the six backend and
worker environments in this order. Phase 0 deliberately uses the backend-only
target so an incomplete JavaScript package cache cannot invalidate an otherwise
reproducible offline ML environment; the normal `make install` still installs
the complete product and requires Node.js 22 plus pnpm 11.

```sh
uv --version
export UV_OFFLINE=1
make install-backend
make install-vision
make install-whisper
make install-video
make install-lighthouse

# Artifact acquisition is a separate, explicitly networked operation.
unset UV_OFFLINE

# Explicit networked artifact acquisition; never part of an offline proof.
videoscope_hf_home=/absolute/path/to/new-reviewed-hf-cache
HF_HOME="$videoscope_hf_home" make models-base
HF_HOME="$videoscope_hf_home" make models-vision
HF_HOME="$videoscope_hf_home" make models-whisper
HF_HOME="$videoscope_hf_home" make models-video
make models-lighthouse

videoscope_ocr_models=/absolute/path/to/new-reviewed-ocr-model-root
.venv/bin/python -I scripts/download-ocr-models.py \
  --destination "$videoscope_ocr_models"

# Return to the offline gate before creating the final worker environment.
export UV_OFFLINE=1
VIDEOSCOPE_OCR_MODEL_ROOT="$videoscope_ocr_models" make install-ocr

make ml-attest-offline
```

Do not weaken a Python or dependency pin to make this gate pass. Recreate the
drifted environment from its committed lock instead. Every isolated-worker
install refuses to reuse an existing directory, so the commands above describe
a clean installation, not an in-place repair. The attestation JSON is the
evidence that all six environments satisfy the manifest; repository unit tests
or the presence of environment directories are not substitutes.

## Offline full-ML product smoke

The attestation does not execute a model. The next Phase 0 gate is
`scripts/full-ml-smoke.py`, exposed by `make full-ml-smoke`. The Make target has
no install, download, cache, model-root, disposable-root, worker-Python or media
executable defaults: every state-bearing input below is mandatory. The script
re-runs the pinned ML-environment attestation, requires the owner and every role
to use the exact manifest venv, and attests the explicitly selected FFmpeg pair;
each worker then attests its own artifact boundary during startup.

The FastEmbed, RF-DETR and Lighthouse files used by the smoke must be copied or
otherwise provisioned into one existing absolute, non-symlink model root outside
the checkout. It uses the same relative layout as `data/models/`. The pinned
SigLIP, Whisper and Qwen snapshots live in the explicitly selected `HF_HOME`;
OCR uses its separate explicit model root. One clean way to create the direct
model root after the networked acquisition above is:

```sh
videoscope_models_root="$(mktemp -d /private/tmp/videoscope-models.XXXXXX)"
cp -R data/models/. "$videoscope_models_root/"
```

Run the proof only after network access is disabled. The disposable root must be
an existing owner-controlled mode-0700 directory outside both the checkout and
the user's home directory. The five worker interpreters must be the exact
distinct paths from this checkout's environment manifest. FFmpeg and FFprobe
must be absolute bindings named `ffmpeg` and `ffprobe` that resolve to regular
executables in one canonical directory. Homebrew aliases are accepted, resolved
once to their Cellar targets, and the attested canonical directory becomes the
entire worker `PATH`:

```sh
videoscope_checkout="$(pwd -P)"
videoscope_smoke_root="$(mktemp -d /private/tmp/videoscope-full-ml.XXXXXX)"
chmod 700 "$videoscope_smoke_root"

HF_HOME="$videoscope_hf_home" \
VIDEOSCOPE_OCR_MODEL_ROOT="$videoscope_ocr_models" \
VIDEOSCOPE_FULL_ML_SMOKE_ROOT="$videoscope_smoke_root" \
VIDEOSCOPE_FULL_ML_SMOKE_MODELS_ROOT="$videoscope_models_root" \
VIDEOSCOPE_FULL_ML_SMOKE_VISION_PYTHON="$videoscope_checkout/.venv-vision/bin/python" \
VIDEOSCOPE_FULL_ML_SMOKE_WHISPER_PYTHON="$videoscope_checkout/.venv-whisper/bin/python" \
VIDEOSCOPE_FULL_ML_SMOKE_OCR_PYTHON="$videoscope_checkout/.venv-ocr/bin/python" \
VIDEOSCOPE_FULL_ML_SMOKE_LIGHTHOUSE_PYTHON="$videoscope_checkout/.venv-lighthouse/bin/python" \
VIDEOSCOPE_FULL_ML_SMOKE_QWEN_PYTHON="$videoscope_checkout/.venv-qwen/bin/python" \
VIDEOSCOPE_FULL_ML_SMOKE_FFMPEG_BINARY=/opt/homebrew/bin/ffmpeg \
VIDEOSCOPE_FULL_ML_SMOKE_FFPROBE_BINARY=/opt/homebrew/bin/ffprobe \
make full-ml-smoke
```

The target passes matching explicit data, Vision-input, Whisper-input and
Whisper-work roots, enforces Hugging Face/Transformers/Paddle offline flags, and
invokes both `--root` and `--models-root`; the script performs no downloads and
does not touch project `data/`. It self-starts authenticated random-port Vision,
Whisper, Lighthouse and Qwen workers as measured subprocess descendants and
runs OCR as a child. Both OCR instances receive an explicit allowlisted
environment; their HOME, cache, TMPDIR and copied control/frame bundle remain
inside separate disposable roots and never inherit the caller's ambient paths.
Typed configuration, infrastructure, OOM and contract failures also write one
`status: failed` JSON to stdout, so redirecting the command preserves a negative
artifact. Stderr retains the compact error classifier and the exit code remains
nonzero. Failure diagnostics contain available validated raw measurements,
completed steps, identities and cleanup outcomes; missing measurement parts
remain unavailable. An ambiguous worker failure keeps OOM `unknown`. These
diagnostics never satisfy the successful-smoke evidence validator. Interruptions
and unexpected internal errors still use only the compact stderr error contract.
The successful schema-v2 receipt must contain:

- the exact ML-environment manifest/attestation identities and the path-free
  `owner/base` plus worker-role environment mapping;
- real Vision image/text embeddings and RF-DETR, Whisper, OCR, Lighthouse and
  Qwen calls;
- `upload → index → query → ranked interval → evidence → MP4 export`, including
  an FFprobe-verified positive-duration MP4;
- complete, generation-bound open/search/close receipts with non-empty evidence
  for `lexical_qdrant`, `dense_siglip`, `temporal_refinement`, `lighthouse` and
  `qwen_verification`, plus an identity-bound execution trace proving the exact
  cumulative search components that actually ran for each profile;
- `internvideo: not_configured/provider_not_configured`, never a fallback;
- bounded raw recursive process-tree RSS samples that include all four loopback
  workers, and native host-resource raw samples for Metal and VM pressure; and
- `oom: not_observed`, complete runtime/workspace cleanup, and zero swap growth
  for Phase 0 acceptance.

Metal in-use/allocated peaks, recovery and VM swap are system-wide observations.
Their receipt says
`metal_standalone_not_additive_with_process_rss`: they cannot be added to RSS or
attributed to one worker. Exit codes are `0` for a ready receipt, `2` for invalid
configuration, `3` for infrastructure, `4` for a contract violation, `6` for
observed OOM, `70` for an unexpected internal failure and `130` for interruption.
Failures are sanitized and do not include paths or tokens.

The command contract and its lightweight tests exist, but no successful target
M4 Pro receipt has yet been recorded. Phase 0 therefore remains open until this
exact clean-checkout run and the separate benchmark/rollback evidence gate pass.
