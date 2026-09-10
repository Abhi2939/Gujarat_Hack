from __future__ import annotations

import cv2
import numpy as np


def blur_score(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_blurry(image: np.ndarray, threshold: float = 60.0) -> bool:
    return blur_score(image) < threshold