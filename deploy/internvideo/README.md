# InternVideo 2.5 GPU endpoint

This service implements the `/rerank` contract used by VideoScope. Run it in a separate CUDA 12.1 environment; the 8B model is not intended for the local Apple Silicon process.

VideoScope prefers the local Qwen3.5 9B verifier when it is ready. This endpoint is an optional fallback for the best fused candidates; the main search pipeline works without either heavy reranker.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install flash-attn --no-build-isolation
INTERNVIDEO_API_KEY=change-me .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8780
```

Configure the VideoScope machine:

```dotenv
INTERNVIDEO_ENDPOINT=http://gpu-host:8780
INTERNVIDEO_API_KEY=change-me
```

The endpoint receives up to eight downscaled JPEG frames from each of at most four fused candidates. It never receives the complete source video.
