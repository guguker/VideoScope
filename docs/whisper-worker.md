# Whisper worker

MLX Whisper runs outside the FastAPI process in a dedicated, hash-locked Python
3.12.13 environment on Apple Silicon with macOS 14 or newer. Intel Mac and Linux
are not part of this worker's supported runtime identity. The backend owns media
identity, stage runs and immutable segment generations; the worker only
transcribes one verified media snapshot.

## Install and configure

```bash
make install-whisper
make models-whisper
```

Installation performs an exact `uv pip sync --require-hashes` from
`workers/whisper/requirements.lock`. Model download separately fetches the exact
Hugging Face revision. Neither command changes the backend `.venv`.

Generate a dedicated token and set:

```dotenv
VIDEOSCOPE_WHISPER_WORKER_ENDPOINT=http://127.0.0.1:8784
VIDEOSCOPE_WHISPER_WORKER_API_KEY=<dedicated token>
VIDEOSCOPE_WHISPER_WORKER_PORT=8784
VIDEOSCOPE_WHISPER_WORKER_INPUT_ROOT=./data/media
VIDEOSCOPE_WHISPER_WORKER_WORK_ROOT=./data/tmp/whisper-worker
WHISPER_MODEL=mlx-community/whisper-large-v3-turbo
VIDEOSCOPE_WHISPER_WORKER_MODEL_REVISION=a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb
```

Start `make whisper-worker` in its own terminal. Without the endpoint pair the
speech stage is `not configured`; there is no automatic in-process fallback.

## One prompt snapshot per run

At the start of indexing, the backend reads the bounded glossary through an
all-component `O_NOFOLLOW` descriptor walk and creates one immutable
`WhisperPromptSnapshot`. Its digest is used both in the speech
`StageSpecification` and in the worker request. Immediately before activation,
the Indexer resolves the current specification again. If the glossary or prompt
changed during inference, the new output is rejected and the previous speech
generation remains active.

This preserves the product rule that glossary changes apply on the next
indexing run without requiring an application restart, while preventing a single
run from mixing two glossary versions.

## Security and failure behavior

- The endpoint is exactly `http://127.0.0.1:<port>` with bearer authentication.
- Health and transcription are bound to exact model, runtime and dependency-lock
  identities; the model is resolved from a pinned local snapshot only.
- The worker accepts only relative regular media files below `data/media` and
  rejects symlinks in every path component.
- It streams the source into a size-bounded private copy, verifies SHA-256 and
  file identity before inference, checks the source again afterward, and removes
  the private copy on every exit path.
- Input bytes/duration, output text, segment count, word count, timestamps and
  response size are bounded; non-finite or overlapping timelines fail closed.
- Only one inference operation is admitted at a time; excess work receives a
  retryable capacity response rather than an unbounded queue.
- A configured but unavailable worker produces an explicit failed stage and does
  not erase the previous verified generation.

This is process and dependency isolation, not an OS sandbox. Stronger
confidentiality requires a dedicated user or read-only mounts around the worker.
