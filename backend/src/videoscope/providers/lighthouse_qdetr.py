from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


class QDDETRPredictor:
    """Изолированный от зависимостей путь вывода CLIP/QD-DETR для Lighthouse.

    Публичный модуль ``models`` библиотеки Lighthouse импортирует все поддерживаемые
    аудио- и видеокодировщики. Этот адаптер сохраняет официальный код модели QD-DETR
    и формат контрольной точки, загружая только те компоненты CLIP, которые использует VideoScope.
    """

    _image_size = 224
    _maximum_frames = 75
    _batch_size = 32

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cpu",
        feature_name: str = "clip",
    ) -> None:
        if feature_name != "clip":
            raise ValueError("VideoScope's Lighthouse adapter supports feature_name='clip' only")

        import clip
        import torch
        from lighthouse.common.qd_detr import build_model

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        options = checkpoint["opt"]
        options.device = device

        model, _ = build_model(options)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()

        clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        clip_model.eval()

        self._torch = torch
        self._clip = clip
        self._model = model
        self._clip_model = clip_model
        self._device = device
        self._clip_length = float(options.clip_length)
        self._moment_count = 10

    @staticmethod
    def _resize_and_crop(frame: np.ndarray, size: int) -> np.ndarray:
        import cv2

        height, width = frame.shape[:2]
        scale = size / min(height, width)
        resized_width = max(size, round(width * scale))
        resized_height = max(size, round(height * scale))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        left = (resized_width - size) // 2
        top = (resized_height - size) // 2
        return resized[top : top + size, left : left + size]

    def _sample_frames(self, source: str) -> Any:
        import cv2

        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Lighthouse could not open video: {source}")

        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
        if duration <= 0:
            capture.release()
            raise RuntimeError("Lighthouse could not determine video duration")

        sample_count = min(self._maximum_frames, max(1, math.ceil(duration / self._clip_length)))
        frames: list[np.ndarray] = []
        try:
            for index in range(sample_count):
                timestamp = min(duration - 0.001, (index + 0.5) * self._clip_length)
                capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000)
                ok, frame = capture.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                crop = self._resize_and_crop(rgb, self._image_size)
                frames.append(np.transpose(crop, (2, 0, 1)))
        finally:
            capture.release()

        if not frames:
            raise RuntimeError("Lighthouse could not decode any video frames")
        return self._torch.from_numpy(np.stack(frames)).float()

    def _encode_images(self, frames: Any) -> Any:
        torch = self._torch
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        frames = (frames / 255.0 - mean) / std

        encoded = []
        with torch.inference_mode():
            for offset in range(0, len(frames), self._batch_size):
                batch = frames[offset : offset + self._batch_size].to(self._device)
                encoded.append(self._clip_model.encode_image(batch).float())
        return torch.cat(encoded, dim=0)

    def _encode_text(self, query: str) -> tuple[Any, Any]:
        torch = self._torch
        tokens = self._clip.tokenize([query]).to(self._device)
        model = self._clip_model
        dtype = model.visual.conv1.weight.dtype

        with torch.inference_mode():
            features = model.token_embedding(tokens).type(dtype)
            features = features + model.positional_embedding.type(dtype)
            features = features.permute(1, 0, 2)
            features = model.transformer(features)
            features = features.permute(1, 0, 2)
            features = model.ln_final(features).float()
            features = torch.nn.functional.normalize(features, dim=-1, eps=1e-5)
        mask = (tokens != 0).float().to(self._device)
        return features, mask

    def encode_video(self, source: str) -> dict[str, Any]:
        torch = self._torch
        video_features = self._encode_images(self._sample_frames(source))
        frame_count = len(video_features)
        starts = torch.arange(frame_count, device=self._device, dtype=torch.float32) / frame_count
        temporal_features = torch.stack((starts, starts + 1.0 / frame_count), dim=1)
        timestamped = torch.cat((video_features, temporal_features), dim=1).unsqueeze(0)
        mask = torch.ones(1, frame_count, device=self._device)
        return {"video_feats": timestamped, "video_mask": mask, "audio_feats": None}

    def predict(self, query: str, inputs: dict[str, Any]) -> dict[str, list[Any]]:
        torch = self._torch
        query_features, query_mask = self._encode_text(query)
        model_inputs = {
            "src_vid": inputs["video_feats"].to(self._device),
            "src_vid_mask": inputs["video_mask"].to(self._device),
            "src_txt": query_features,
            "src_txt_mask": query_mask,
            "src_aud": None,
        }
        with torch.inference_mode():
            output = self._model(**model_inputs)

        probabilities = torch.nn.functional.softmax(output["pred_logits"], -1).squeeze(0).cpu()
        scores = probabilities[:, 0]
        spans = output["pred_spans"].squeeze(0).cpu()
        centers, widths = spans.unbind(-1)
        duration = model_inputs["src_vid"].shape[1] * self._clip_length
        starts = (centers - 0.5 * widths) * duration
        ends = (centers + 0.5 * widths) * duration
        windows = torch.stack((starts.clamp(0, duration), ends.clamp(0, duration), scores), dim=1)
        ranked = sorted(windows.tolist(), key=lambda item: item[2], reverse=True)

        video_mask = model_inputs["src_vid_mask"].bool()
        saliency = output["saliency_scores"][video_mask].detach().cpu().tolist()
        return {
            "pred_relevant_windows": [
                [round(float(value), 4) for value in window]
                for window in ranked[: self._moment_count]
            ],
            "pred_saliency_scores": [float(value) for value in saliency],
        }
