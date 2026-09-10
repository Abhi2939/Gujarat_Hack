from __future__ import annotations

import logging

import numpy as np
import supervision as sv

from common.config import TrackerConfig, SceneCutConfig
from .scene_cut import SceneCutDetector
from common.schemas import TrackedDetection

logger = logging.getLogger("netra.l2.tracker")


class VehicleTracker:
    def __init__(self, camera_id: str, tracker_cfg: TrackerConfig, scene_cut_cfg: SceneCutConfig,
                 class_name_lookup: dict[int, str]):
        self.camera_id = camera_id
        self.tracker_cfg = tracker_cfg
        self.class_name_lookup = class_name_lookup

        self.scene_cut_cfg = scene_cut_cfg
        self._scene_cut = SceneCutDetector(
            correlation_floor=scene_cut_cfg.hist_correlation_floor,
            min_frames_between_cuts=scene_cut_cfg.min_frames_between_cuts,
        ) if scene_cut_cfg.enabled else None

        self._tracker = self._new_bytetrack()
        self.track_session_id = 0
        self._frame_index = 0

    def _new_bytetrack(self) -> sv.ByteTrack:
        return sv.ByteTrack(
            track_activation_threshold=self.tracker_cfg.track_activation_threshold,
            lost_track_buffer=self.tracker_cfg.lost_track_buffer,
            minimum_matching_threshold=self.tracker_cfg.minimum_matching_threshold,
            minimum_consecutive_frames=self.tracker_cfg.minimum_consecutive_frames,
            frame_rate=self.tracker_cfg.frame_rate,
        )

    def reset(self, reason: str) -> None:
        self._tracker = self._new_bytetrack()
        self.track_session_id += 1
        logger.info("Tracker reset on camera=%s reason=%s new_session=%d",
                     self.camera_id, reason, self.track_session_id)

    def update(self, frame: np.ndarray, detections: sv.Detections) -> list[TrackedDetection]:
        self._frame_index += 1

        if self._scene_cut is not None and self._scene_cut.update(frame):
            self.reset(reason="scene_cut")

        tracked: sv.Detections = self._tracker.update_with_detections(detections)

        out: list[TrackedDetection] = []
        if tracked.tracker_id is None:
            return out

        for i in range(len(tracked)):
            class_id = int(tracked.class_id[i])
            out.append(TrackedDetection(
                camera_id=self.camera_id,
                frame_index=self._frame_index,
                track_id=int(tracked.tracker_id[i]),
                track_session_id=self.track_session_id,
                class_id=class_id,
                class_name=self.class_name_lookup.get(class_id, f"class_{class_id}"),
                confidence=float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
                bbox_xyxy=tuple(float(v) for v in tracked.xyxy[i]),
            ))
        return out