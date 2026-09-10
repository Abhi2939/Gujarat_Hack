"""
Scene-cut detection — supports ADR-007.
"""

from __future__ import annotations

import cv2
import numpy as np


class SceneCutDetector:
    def __init__(self, correlation_floor: float = 0.55, min_frames_between_cuts: int = 5):
        self.correlation_floor = correlation_floor
        self.min_frames_between_cuts = min_frames_between_cuts
        self._prev_hist: np.ndarray | None = None
        self._frames_since_cut = min_frames_between_cuts

    def _histogram(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def update(self, frame: np.ndarray) -> bool:
        hist = self._histogram(frame)
        self._frames_since_cut += 1

        is_cut = False
        if self._prev_hist is not None:
            correlation = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL)
            if correlation < self.correlation_floor and self._frames_since_cut >= self.min_frames_between_cuts:
                is_cut = True
                self._frames_since_cut = 0

        self._prev_hist = hist
        return is_cut

    def reset(self) -> None:
        self._prev_hist = None
        self._frames_since_cut = self.min_frames_between_cuts