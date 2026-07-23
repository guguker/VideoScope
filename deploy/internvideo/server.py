from __future__ import annotations

import base64
from io import BytesIO
import json
import os
import re
from threading import Lock

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


MODEL_NAME = os.getenv("INTERNVIDEO_MODEL", "OpenGVLab/InternVideo2_5_Chat_8B")
API_KEY = os.getenv("INTERNVIDEO_API_KEY")


class FramePayload(BaseModel):
    timestamp: float = Field(ge=0)
    jpeg_base64: str = Field(min_length=4, max_length=3_000_000)


class CandidatePayload(BaseModel):
    id: str = Field(min_length=1, max_length=160)
    video_id: str = Field(min_length=1, max_length=64)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    frames: list[FramePayload] = Field(min_length=2, max_length=16)


class RerankRequest(BaseModel):
    model: str
    task: str
    query: str = Field(min_length=1, max_length=500)
    candidates: list[CandidatePayload] = Field(min_length=1, max_length=8)


class ModelRuntime:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.transform = None
        self.lock = Lock()

    def load(self):  # type: ignore[no-untyped-def]
        if self.model is not None:
            return self.model, self.tokenizer, self.transform
        import torch
        import torchvision.transforms as transforms
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("InternVideo 2.5 endpoint requires a CUDA GPU")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            use_flash_attn=True,
        ).eval().cuda()
        self.transform = transforms.Compose([
            transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        return self.model, self.tokenizer, self.transform

    def score(self, query: str, candidate: CandidatePayload) -> tuple[float, str]:
        import torch
        from PIL import Image

        model, tokenizer, transform = self.load()
        images = []
        for frame in candidate.frames:
            try:
                raw = base64.b64decode(frame.jpeg_base64, validate=True)
                image = Image.open(BytesIO(raw)).convert("RGB")
            except Exception as error:
                raise ValueError("candidate contains an invalid JPEG frame") from error
            images.append(transform(image))
        pixel_values = torch.stack(images).to(device=model.device, dtype=torch.bfloat16)
        frame_prefix = "".join(f"Frame{index + 1}: <image>\n" for index in range(len(images)))
        question = (
            frame_prefix
            + "Determine how well this video interval matches the search query: "
            + json.dumps(query, ensure_ascii=False)
            + '. Return only JSON in the form {"score": 0.0, "reason": "short evidence"}. '
            + "Score 1 means an exact visible temporal match and 0 means unrelated."
        )
        generation_config = {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 96,
            "top_p": 0.1,
            "num_beams": 1,
        }
        with self.lock, torch.inference_mode():
            response = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=[1] * len(images),
            )
        match = re.search(r"\{.*?\}", str(response), flags=re.DOTALL)
        if not match:
            raise ValueError("model did not return a JSON score")
        payload = json.loads(match.group(0))
        score = max(0.0, min(1.0, float(payload["score"])))
        return score, str(payload.get("reason") or "InternVideo visual match")[:300]


runtime = ModelRuntime()
app = FastAPI(title="VideoScope InternVideo 2.5 endpoint", version="0.1.0")


def authorize(authorization: str | None) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model": MODEL_NAME, "loaded": runtime.model is not None}


@app.post("/rerank")
def rerank(request: RerankRequest, authorization: str | None = Header(default=None)) -> dict[str, object]:
    authorize(authorization)
    if request.model != MODEL_NAME or request.task != "temporal_relevance":
        raise HTTPException(status_code=422, detail="Unsupported model or task")
    scores = []
    for candidate in request.candidates:
        try:
            score, reason = runtime.score(request.query, candidate)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        scores.append({"id": candidate.id, "score": score, "reason": reason})
    return {"scores": scores}
