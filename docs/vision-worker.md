# Vision worker

SigLIP 2 and local RF-DETR run outside the FastAPI process in a dedicated,
hash-locked Python 3.12.13 environment on Apple Silicon with macOS 14 or newer.
Intel Mac and Linux are not part of this worker's supported runtime identity.
The backend owns SQLite, source media and generation activation; the worker only
performs bounded inference over approved files below `data/`.

## Install and configure

```bash
make install-vision
make models-vision
```

Installation performs an exact `uv pip sync --require-hashes` from
`workers/vision/requirements.lock`. Model download is a separate explicit step:
it fetches pinned SigLIP snapshots and RF-DETR Small, then verifies the RF-DETR
SHA-256. Neither command changes the backend `.venv`.

Generate a dedicated token, for example with `openssl rand -hex 32`, and set:

```dotenv
VIDEOSCOPE_VISION_WORKER_ENDPOINT=http://127.0.0.1:8783
VIDEOSCOPE_VISION_WORKER_API_KEY=<dedicated token>
VIDEOSCOPE_VISION_WORKER_PORT=8783
VIDEOSCOPE_VISION_WORKER_INPUT_ROOT=./data
# Set only when INPUT_ROOT is the direct parent of the product data directory.
# This keeps access limited to product/{visual-index,thumbnails,tmp}.
# VIDEOSCOPE_VISION_WORKER_PRODUCT_DATA_SUBDIRECTORY=product
VIDEOSCOPE_VISION_WORKER_RFDETR_CHECKPOINT=./data/models/rfdetr/rf-detr-small.pth
VIDEOSCOPE_VISION_DETECTOR_MODEL_ID=rfdetr-small
VIDEOSCOPE_VISION_DETECTOR_CHECKPOINT_SHA256=d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33
VIDEOSCOPE_SIGLIP_MODEL=google/siglip2-base-patch16-224
```

Start `make vision-worker` in its own terminal. The API may start without it;
visual and local-object stages then report `not configured` instead of falling
back to in-process ML. Hosted Roboflow is an explicit alternative and cannot be
enabled together with the local worker.

## Contract and artifact identity

Every health and inference response is bound to the exact worker lock, Python
runtime, SigLIP repository revision, preprocessing/tokenizer policy, embedding
dimension and RF-DETR checkpoint hash. Both models are fixed to MPS with
`float32`; the backend, dtype and device checks are part of their projection
identities, so a CPU fallback cannot silently reuse an MPS generation. SigLIP
and detector projections have separate hashes, so changing an object detector
does not invalidate dense visual vectors. A model, lock, compute or
preprocessing change does invalidate the corresponding derived generation.

Only one vision backbone may remain resident on MPS. Loading SigLIP releases
RF-DETR first and loading RF-DETR releases SigLIP first. In addition, the
Indexer explicitly closes the RF-DETR ingestion stage before later text-vector
and dense-visual work, while the full-ML probe closes it before starting
Whisper. That authenticated lifecycle call is idempotent and fail-closed: if the
worker cannot confirm detector release, the indexing attempt fails while the
previously active generations remain untouched.

`SiglipVisualIndex` remains the generation owner. It builds in a private
directory, validates finite fixed-shape vectors and metadata, then atomically
switches `active.json`. A worker error or malformed response never overwrites the
previous active generation.

The default 224 model and reviewed 384 quality profile are supported. To switch
profiles, stop the API, update `VIDEOSCOPE_SIGLIP_MODEL`, restart the worker and
run `make index-visual-quality`. The maintenance command takes the same
exclusive data lock as the API and intentionally fails if `make dev` is running.

## Security and failure behavior

- The endpoint is exactly `http://127.0.0.1:<port>` with bearer authentication.
- Proxy environment, redirects, untrusted `Host`, oversized bodies and extra
  fields are rejected.
- Input paths are relative and limited to `visual-index`, `thumbnails` and
  `tmp`. When a shared parent root is explicitly required, the worker accepts
  only those same three derived directories below the configured product-data
  subdirectory; sibling `media` and prefix collisions remain inaccessible. All
  path components reject symlinks and special files.
- The worker hashes and copies each image into a bounded private snapshot,
  performs inference only on that copy, then rechecks the original file.
- Image bytes, pixels, batch sizes, response bytes and ML concurrency are capped.
- RF-DETR cannot auto-download or replace a checkpoint during model load.
- A configured but unavailable worker stays attached to the Indexer. The stage
  fails explicitly and can be retried after the worker recovers.

This is process and dependency isolation, not an OS sandbox. Stronger
confidentiality requires a dedicated user or read-only mounts around the worker.
