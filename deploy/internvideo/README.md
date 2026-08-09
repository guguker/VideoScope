# InternVideo 2.5 GPU endpoint

This optional service implements the `/rerank` contract used by VideoScope. Run it in a separate CUDA 12.1 environment; the 8B model is not intended for the local Apple Silicon process. VideoScope continues to work when this service is unavailable.

## Security boundary

Treat the endpoint as an internal GPU service, not as a public API:

- keep it on the same host or a trusted private network;
- use HTTPS whenever the bearer token crosses a network boundary;
- terminate TLS in a private reverse proxy, or configure Uvicorn with `--ssl-keyfile` and `--ssl-certfile`;
- add network ACLs, request-rate limiting, and timeouts at the proxy;
- do not expose a plain-HTTP `0.0.0.0:8780` listener to the internet.

`INTERNVIDEO_API_KEY` is mandatory and must contain at least 32 characters. `/rerank` rejects a missing or invalid bearer token before reading or validating its JSON body. `/health` is intentionally unauthenticated for private-network health checks and returns only model/load metadata.

The model uses `trust_remote_code=True`, so `INTERNVIDEO_REVISION` is also mandatory. Set it to a reviewed, full 40-character commit SHA from the model repository; branches and movable tags such as `main` are rejected. The same revision is passed to both tokenizer and model loading.

## Install and run

Create a strong token and store it in the secret manager used by both machines:

```bash
openssl rand -hex 32
```

Then install the pinned runtime. FlashAttention needs Torch to be present before its build, so it remains a separate step:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-flash.txt --no-build-isolation
```

For a TLS reverse proxy on the same machine, bind Uvicorn only to loopback:

```bash
export INTERNVIDEO_API_KEY='<secret-manager-value>'
export INTERNVIDEO_REVISION='<reviewed-40-character-commit-sha>'
.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8780 --proxy-headers
```

Configure VideoScope with the HTTPS address exposed by that proxy:

```dotenv
INTERNVIDEO_ENDPOINT=https://gpu-host.example/rerank
INTERNVIDEO_API_KEY=<same-secret-manager-value>
```

## HTTP contract and limits

`POST /rerank` requires `Authorization: Bearer <INTERNVIDEO_API_KEY>` and accepts strict JSON: unknown fields, non-finite numbers, invalid intervals, duplicate candidate IDs, unsafe video IDs, and frame timestamps outside their interval are rejected.

The service enforces these limits before GPU inference:

| Limit | Value |
| --- | ---: |
| Total request body | 64 MiB |
| Candidates | 4 |
| JPEG frames per candidate | 8 |
| Base64 characters per frame | 2,000,000 |
| Decoded image dimensions | 4,194,304 pixels |

Only JPEG frames from the best fused candidates are transmitted; the complete source video is never sent. Model/runtime failures return a generic `503` response, while diagnostic details stay in server logs.

Example request shape:

```json
{
  "model": "OpenGVLab/InternVideo2_5_Chat_8B",
  "task": "temporal_relevance",
  "query": "three-point shot",
  "candidates": [
    {
      "id": "video-1:12.000:18.000",
      "video_id": "video-1",
      "start": 12.0,
      "end": 18.0,
      "frames": [
        {"timestamp": 14.0, "jpeg_base64": "<base64-jpeg>"}
      ]
    }
  ]
}
```

Successful response:

```json
{
  "scores": [
    {
      "id": "video-1:12.000:18.000",
      "score": 0.82,
      "reason": "Visible shot attempt and made basket"
    }
  ]
}
```
