# Attested OCR worker

This worker is intentionally limited to CPython 3.12.13 on Apple Silicon macOS.
`requirements.lock` pins every runtime distribution and accepted wheel by SHA-256.
Refresh it only as an explicit dependency review:

```sh
make lock-ocr
make lock-ocr-check
```

Both targets are gated by the actual build interpreter and host: exactly
CPython 3.12.13 on Apple Silicon Darwin with macOS 14 or newer. They deliberately
use that interpreter's concrete platform tags instead of a generic
`--python-platform` projection. `lock-ocr-check` resolves to a temporary file
and requires byte-for-byte equality with the committed lock.

`model-artifacts.lock.json` pins the runtime files for the detection and
recognition models. The worker never downloads a missing or changed model. By
default it verifies the files below `~/.paddlex/official_models`; set
`VIDEOSCOPE_OCR_MODEL_ROOT` when the reviewed files live elsewhere.

`model-sources.lock.json` binds both model directories to immutable Hugging Face
repository revisions, Apache-2.0 license evidence, and the exact runtime bytes in
`model-artifacts.lock.json`. Run `make models-ocr` as the explicit networked
acquisition step before `make install-ocr`. The downloader requests only reviewed
filenames, validates every size and SHA-256 in private staging, and publishes a
model directory only after verification. It accepts an already-correct directory
but does not overwrite an existing invalid directory or other user state.

Given those pre-provisioned model bytes, `make install-ocr` creates a fresh
hash-locked dependency environment and runs the attestation-only startup path.
It rejects an unsupported host before creating the environment or invoking pip.
The fresh `uv pip sync` environment must contain exactly the distributions in
`requirements.lock`; an injected bootstrap `pip` distribution is rejected like
any other unreviewed package.
The worker will not become ready when the Python patch version, platform,
dependency lock, installed versions, worker script, or model artifacts differ
from the reviewed identities.

At runtime the worker re-reads every reviewed model artifact through a stable,
no-follow file descriptor and copies the verified bytes into a new mode-0700
worker-private directory. PaddleOCR receives only those private directory paths.
The shared model cache is therefore outside the model-loader race boundary: a
cache mutation during Paddle initialization cannot become the loaded model.
Private copies are checked again after initialization and removed when startup
fails or the worker exits. This boundary does not defend against root, debugger
attachment, or another process with the same effective user deliberately
changing the worker's own mode-0700 directory.

Frame requests are pathless. The application reads the source frame once via a
stable descriptor, rejects links and special files, checks the exact compressed
byte count, SHA-256, image dimensions, pixel count, and decoded-memory bound,
then writes a mode-0400 per-request copy below the private worker bundle. The
worker independently verifies that contract, unlinks the copy, decodes from the
verified bytes, and passes the decoded array—not a filesystem path—to PaddleOCR.

## Remaining installed-package attestation blocker

`requirements.lock` cryptographically pins accepted wheel *archives* during the
fresh `pip --require-hashes --only-binary` install, and startup rejects any
unexpected distribution name or version. It does not contain a reviewed mapping
from each accepted wheel hash to every installed file and file digest. The local
`.dist-info/RECORD` files cannot supply that trust anchor because an attacker can
modify a package and its RECORD together; accepting RECORD would be TOFU.
Consequently the worker does not claim cryptographic installed-file ownership or
complete detection of injected `.pth`/unowned `site-packages` files. Closing that
gap offline requires reviewed wheel payloads (or a generated, reviewed
per-platform installed-files manifest) whose identity is added to the public
worker attestation. No identity is synthesized from the current machine.
