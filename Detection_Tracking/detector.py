from __future__ import annotations

import logging

import numpy as np
import supervision as sv
from ultralytics import YOLO

from common.config import DetectorConfig

logger = logging.getLogger("netra.l2.detector")


class VehicleDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        if cfg.device:
            self.model.to(cfg.device)
        self._allowed_ids = set(cfg.allowed_classes.keys())
        logger.info(
            "VehicleDetector ready: weights=%s device=%s classes=%s",
            cfg.weights, cfg.device, list(cfg.allowed_classes.values()),
        )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        predict_kwargs = dict(
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            device=self.cfg.device,
            classes=list(self._allowed_ids),
            verbose=False,
        )
        if self.cfg.half_precision:
            predict_kwargs["half"] = True

        results = self.model.predict(frame, **predict_kwargs)
        return sv.Detections.from_ultralytics(results[0])

    def class_name(self, class_id: int) -> str:
        return self.cfg.allowed_classes.get(class_id, f"class_{class_id}")