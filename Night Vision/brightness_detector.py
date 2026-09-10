from __future__ import annotations

import cv2
import numpy as np


def brightness_score(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(gray.mean())


def is_low_light(image: np.ndarray, threshold: float = 60.0) -> bool:
    return brightness_score(image) < threshold