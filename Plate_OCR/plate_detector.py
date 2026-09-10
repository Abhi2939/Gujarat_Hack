"""YOLO plate detector and coordinate-safe crop extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlateDetection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float


class YOLOPlateDetector:
    def __init__(self, weights: str, conf_threshold: float = 0.35,
                 imgsz: int = 640, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device

    def detect(self, vehicle_crop: np.ndarray) -> list[PlateDetection]:
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []
        result = self.model.predict(vehicle_crop, conf=self.conf_threshold,
                                    imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return [PlateDetection(tuple(float(v) for v in box), float(score))
                for box, score in zip(boxes, scores)]
