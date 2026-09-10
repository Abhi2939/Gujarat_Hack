"""Step 3: associate the best plate read with each tracked vehicle."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from common.schemas import PlateRead, TrackedDetection
from .plate_detector import PlateDetection
from .plate_ocr import PlateReader


class PlateDetector(Protocol):
    def detect(self, vehicle_crop: np.ndarray) -> list[PlateDetection]:
        ...


def _clip_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, max(x1 + 1, int(x2))), min(height, max(y1 + 1, int(y2)))
    return x1, y1, x2, y2


class PlatePipeline:
    def __init__(self, detector: PlateDetector, reader: PlateReader,
                 run_every_n_track_frames: int = 3, minimum_text_confidence: float = 0.0):
        self.detector = detector
        self.reader = reader
        self.run_every_n_track_frames = max(1, run_every_n_track_frames)
        self.minimum_text_confidence = minimum_text_confidence
        self._seen: dict[tuple[str, int, int], int] = {}

    def process(self, frame: np.ndarray, tracked: list[TrackedDetection]) -> list[PlateRead]:
        height, width = frame.shape[:2]
        reads: list[PlateRead] = []
        for track in tracked:
            if track.class_name not in {"car", "motorcycle", "bus", "truck"}:
                continue
            key = (track.camera_id, track.track_session_id, track.track_id)
            seen = self._seen.get(key, 0)
            self._seen[key] = seen + 1
            if seen % self.run_every_n_track_frames:
                continue
            vx1, vy1, vx2, vy2 = _clip_box(track.bbox_xyxy, width, height)
            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            detections = self.detector.detect(vehicle_crop)
            if not detections:
                continue
            best = max(detections, key=lambda item: item.confidence)
            px1, py1, px2, py2 = _clip_box(best.bbox_xyxy, vx2 - vx1, vy2 - vy1)
            text, confidence, multiline = self.reader.read(vehicle_crop[py1:py2, px1:px2])
            if not text or confidence < self.minimum_text_confidence:
                continue
            reads.append(PlateRead(
                camera_id=track.camera_id, frame_index=track.frame_index,
                track_id=track.track_id, track_session_id=track.track_session_id,
                vehicle_class_name=track.class_name, plate_text=text,
                plate_text_confidence=confidence, plate_detection_confidence=best.confidence,
                plate_bbox_xyxy=(vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2),
                was_multiline=multiline,
            ))
        return reads
