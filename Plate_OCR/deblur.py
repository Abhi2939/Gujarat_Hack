from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger("netra.l3.deblur")


class DeblurBackend(Protocol):
    def deblur(self, image: np.ndarray) -> np.ndarray:
        ...


class UnsharpMaskBackend:
    def __init__(self, sigma: float = 1.0, strength: float = 1.5):
        self.sigma = sigma
        self.strength = strength

    def deblur(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        blurred = cv2.GaussianBlur(image, (0, 0), self.sigma)
        sharpened = cv2.addWeighted(image, 1.0 + self.strength, blurred, -self.strength, 0)
        return sharpened


class NAFNetBackend:
    def __init__(self, checkpoint_path: str | None = None):
        raise NotImplementedError(
            "NAFNetBackend is not usable in this environment: basicsr fails to import "
            "(torchvision.transforms.functional_tensor was removed upstream), and NAFNet's "
            "pretrained checkpoint is Google-Drive-hosted regardless. See this module's "
            "docstring. Use UnsharpMaskBackend, or fix basicsr's import and supply a real "
            "checkpoint_path yourself."
        )

    def deblur(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError