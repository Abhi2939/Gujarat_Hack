from __future__ import annotations

import logging

import numpy as np

from .color import estimate_color
from .body_type import estimate_body_type
from common.schemas import TrackedDetection, VehicleAttributes

logger = logging.getLogger("netra.l4.fingerprint_pipeline")

VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}


def _clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    return x1, y1, x2, y2


class FingerprintPipeline:
    def __init__(self, run_every_n_track_frames: int = 1):
        self.run_every_n_track_frames = run_every_n_track_frames
        self._frames_seen_per_track: dict[tuple[str, int, int], int] = {}

    def _should_run(self, track_key: tuple[str, int, int]) -> bool:
        count = self._frames_seen_per_track.get(track_key, 0)
        self._frames_seen_per_track[track_key] = count + 1
        return count % self.run_every_n_track_frames == 0

    def process(self, frame: np.ndarray, tracked: list[TrackedDetection]) -> list[VehicleAttributes]:
        height, width = frame.shape[:2]
        out: list[VehicleAttributes] = []

        for t in tracked:
            if t.class_name not in VEHICLE_CLASS_NAMES:
                continue

            track_key = (t.camera_id, t.track_session_id, t.track_id)
            if not self._should_run(track_key):
                continue

            x1, y1, x2, y2 = _clip_box(*t.bbox_xyxy, width=width, height=height)
            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue

            color_name, color_conf = estimate_color(vehicle_crop)
            body_type, body_conf = estimate_body_type(t.class_name, t.bbox_xyxy)

            out.append(VehicleAttributes(
                camera_id=t.camera_id,
                frame_index=t.frame_index,
                track_id=t.track_id,
                track_session_id=t.track_session_id,
                vehicle_class_name=t.class_name,
                color_name=color_name,
                color_confidence=color_conf,
                body_type=body_type,
                body_type_confidence=body_conf,
                vehicle_bbox_xyxy=t.bbox_xyxy,
            ))

        return out