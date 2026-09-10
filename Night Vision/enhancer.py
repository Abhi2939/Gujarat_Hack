from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger("netra.night_vision.enhancer")


class NightVisionBackend(Protocol):
    def enhance(self, image: np.ndarray) -> np.ndarray:
        ...


class GammaCorrectionBackend:
    def __init__(self, gamma: float = 2.2):
        self.gamma = gamma
        inv_gamma = 1.0 / gamma
        self._lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        return cv2.LUT(image, self._lut)


class ZeroDCEPPBackend:
    def __init__(self, checkpoint_path: str, device: str = "cpu", scale_factor: float = 1.0):
        import torch
        from .zero_dce_model import EnhanceNetNoPool

        self.device = device
        self.model = EnhanceNetNoPool(scale_factor=scale_factor)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self._torch = torch

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = self._torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with self._torch.no_grad():
            enhanced = self.model(tensor)

        enhanced_np = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced_np = np.clip(enhanced_np, 0.0, 1.0)
        return cv2.cvtColor((enhanced_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def enhance_if_needed(image: np.ndarray, backend: NightVisionBackend,
                       brightness_threshold: float = 60.0) -> tuple[np.ndarray, bool]:
    from .brightness_detector import is_low_light

    if not is_low_light(image, threshold=brightness_threshold):
        return image, False
    return backend.enhance(image), True