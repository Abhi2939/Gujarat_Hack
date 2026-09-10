"""OCR primitives for a detected licence-plate crop.

The detector lives in :mod:`plate_detector`; keeping OCR independent makes it
possible to unit test and to substitute an Indian-fine-tuned Paddle model.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .deblur import DeblurBackend


class OCRBackend(Protocol):
    def read_text(self, image: np.ndarray) -> tuple[str, float]:
        ...


def _looks_multiline(plate_crop: np.ndarray, height_width_ratio_floor: float) -> bool:
    height, width = plate_crop.shape[:2]
    return width > 0 and height / width >= height_width_ratio_floor


def _split_two_line(plate_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = plate_crop.shape[0]
    midpoint = max(1, height // 2)
    return plate_crop[:midpoint, :], plate_crop[midpoint:, :]


class PaddleOCRBackend:
    """Small adapter around PaddleOCR, imported only when it is actually used."""

    def __init__(self, lang: str = "en", **paddle_kwargs):
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(lang=lang, **paddle_kwargs)

    def read_text(self, image: np.ndarray) -> tuple[str, float]:
        result = self._ocr.predict(image)
        texts: list[str] = []
        confidences: list[float] = []
        # PaddleOCR v3 returns a list of result dictionaries.  The fallback
        # supports the v2 ``ocr`` response shape used by older deployments.
        for item in result or []:
            if isinstance(item, dict):
                item_texts = item.get("rec_texts", [])
                item_scores = item.get("rec_scores", [])
                texts.extend(str(text) for text in item_texts)
                confidences.extend(float(score) for score in item_scores)
            elif isinstance(item, (list, tuple)):
                for line in item:
                    if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                        texts.append(str(line[1][0]))
                        confidences.append(float(line[1][1]))
        return "".join(texts).replace(" ", "").upper(), (
            float(sum(confidences) / len(confidences)) if confidences else 0.0
        )


class PlateReader:
    def __init__(self, backend: OCRBackend, height_width_ratio_floor: float = 0.55,
                 deblur_backend: DeblurBackend | None = None,
                 retry_confidence_threshold: float = 0.5, blur_score_threshold: float = 60.0):
        self.backend = backend
        self.height_width_ratio_floor = height_width_ratio_floor
        self.deblur_backend = deblur_backend
        self.retry_confidence_threshold = retry_confidence_threshold
        self.blur_score_threshold = blur_score_threshold

    def _read_once(self, plate_crop: np.ndarray) -> tuple[str, float, bool]:
        if _looks_multiline(plate_crop, self.height_width_ratio_floor):
            top, bottom = _split_two_line(plate_crop)
            top_text, top_conf = self.backend.read_text(top)
            bottom_text, bottom_conf = self.backend.read_text(bottom)
            texts = [t for t in (top_text, bottom_text) if t]
            confs = [c for t, c in ((top_text, top_conf), (bottom_text, bottom_conf)) if t]
            combined_text = "".join(texts)
            mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
            return combined_text, mean_conf, True

        text, conf = self.backend.read_text(plate_crop)
        return text, conf, False

    def read(self, plate_crop: np.ndarray) -> tuple[str, float, bool]:
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0, False

        result = self._read_once(plate_crop)

        if self.deblur_backend is None:
            return result

        _, confidence, _ = result
        if confidence >= self.retry_confidence_threshold:
            return result

        from .blur_detection import is_blurry
        if not is_blurry(plate_crop, threshold=self.blur_score_threshold):
            return result

        deblurred_crop = self.deblur_backend.deblur(plate_crop)
        retry_result = self._read_once(deblurred_crop)

        return retry_result if retry_result[1] > result[1] else result
