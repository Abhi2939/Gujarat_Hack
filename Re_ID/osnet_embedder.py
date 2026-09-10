from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger("netra.l5.embedder")


class EmbeddingBackend(Protocol):
    def embed(self, crop: np.ndarray) -> np.ndarray:
        ...

    @property
    def embedding_dim(self) -> int:
        ...


class OSNetBackend:
    def __init__(self, model_name: str = "osnet_x1_0", weights_path: str | None = None,
                 input_height: int = 256, input_width: int = 256, device: str = "cpu"):
        import torch
        import torchreid

        self.device = device
        self.input_height = input_height
        self.input_width = input_width

        self.model = torchreid.models.build_model(
            name=model_name, num_classes=1, loss="softmax", pretrained=False,
        )
        if weights_path:
            torchreid.utils.load_pretrained_weights(self.model, weights_path)
        else:
            logger.warning(
                "OSNetBackend built with NO weights_path — running on ImageNet-init "
                "weights only. Embeddings will not be meaningful for Re-ID matching "
                "until a real VeRi-776 checkpoint is loaded. See module docstring."
            )

        self.model.to(device)
        self.model.eval()
        self._torch = torch

        self._imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        return 512

    def _preprocess(self, crop: np.ndarray):
        import cv2
        resized = cv2.resize(crop, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self._imagenet_mean) / self._imagenet_std
        chw = normalized.transpose(2, 0, 1)
        return self._torch.from_numpy(chw).unsqueeze(0).float().to(self.device)

    def embed(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        tensor = self._preprocess(crop)
        with self._torch.no_grad():
            feat = self.model(tensor)
        vec = feat.squeeze(0).cpu().numpy().astype(np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class ReIDEmbedder:
    def __init__(self, backend: EmbeddingBackend):
        self.backend = backend

    def embed(self, vehicle_crop: np.ndarray) -> np.ndarray:
        return self.backend.embed(vehicle_crop)