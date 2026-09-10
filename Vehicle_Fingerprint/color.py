from __future__ import annotations

import numpy as np
import cv2

_REFERENCE_HSV: dict[str, tuple[int, int, int]] = {
    "black":  (0, 0, 25),
    "white":  (0, 0, 235),
    "gray":   (0, 0, 130),
    "silver": (0, 20, 190),
    "red":    (2, 200, 180),
    "blue":   (110, 200, 180),
    "green":  (60, 180, 150),
    "yellow": (28, 200, 200),
}


def _crop_interior(vehicle_crop: np.ndarray, margin_frac: float = 0.2) -> np.ndarray:
    h, w = vehicle_crop.shape[:2]
    my, mx = int(h * margin_frac), int(w * margin_frac)
    if h - 2 * my <= 0 or w - 2 * mx <= 0:
        return vehicle_crop
    return vehicle_crop[my:h - my, mx:w - mx]


def _nearest_bucket(hsv_pixel: np.ndarray) -> str:
    h, s, v = hsv_pixel

    if s < 40:
        if v < 60:
            return "black"
        if v > 200:
            return "white"
        if v > 150:
            return "silver"
        return "gray"

    chromatic = {k: v for k, v in _REFERENCE_HSV.items() if k not in ("black", "white", "gray", "silver")}
    best_name, best_dist = None, float("inf")
    for name, (rh, _, _) in chromatic.items():
        dist = min(abs(int(h) - rh), 180 - abs(int(h) - rh))
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name


def estimate_color(vehicle_crop: np.ndarray, k: int = 3) -> tuple[str, float]:
    if vehicle_crop is None or vehicle_crop.size == 0:
        return "unknown", 0.0

    interior = _crop_interior(vehicle_crop)
    hsv = cv2.cvtColor(interior, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    if pixels.shape[0] < k:
        return "unknown", 0.0

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, attempts=3,
                                     flags=cv2.KMEANS_PP_CENTERS)

    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k)
    dominant_idx = int(np.argmax(counts))
    dominant_center = centers[dominant_idx]
    confidence = float(counts[dominant_idx]) / float(len(labels))

    bucket = _nearest_bucket(dominant_center)
    return bucket, confidence