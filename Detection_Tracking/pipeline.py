from __future__ import annotations

import argparse
import logging
import sys

import cv2
import supervision as sv

from common.config import PipelineConfig
from .detector import VehicleDetector
from .tracker import VehicleTracker
from common.schemas import TrackedDetection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("netra.l2.pipeline")


class PerceptionPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.detector = VehicleDetector(cfg.detector)
        self.tracker = VehicleTracker(
            camera_id=cfg.camera_id,
            tracker_cfg=cfg.tracker,
            scene_cut_cfg=cfg.scene_cut,
            class_name_lookup=cfg.detector.allowed_classes,
        )
        self._raw_frame_count = 0

    def on_reconnect(self) -> None:
        self.tracker.reset(reason="reconnect")

    def process_frame(self, frame) -> list[TrackedDetection] | None:
        self._raw_frame_count += 1
        if self._raw_frame_count % self.cfg.process_every_n_frames != 0:
            return None

        detections = self.detector.detect(frame)
        return self.tracker.update(frame, detections)


def _annotate(frame, tracked: list[TrackedDetection]):
    if not tracked:
        return frame
    import numpy as np
    xyxy = [t.bbox_xyxy for t in tracked]
    class_ids = [t.class_id for t in tracked]
    tracker_ids = [t.track_id for t in tracked]
    confidences = [t.confidence for t in tracked]

    detections = sv.Detections(
        xyxy=np.array(xyxy, dtype=float),
        class_id=np.array(class_ids, dtype=int),
        tracker_id=np.array(tracker_ids, dtype=int),
        confidence=np.array(confidences, dtype=float),
    )
    labels = [f"#{t.track_id} {t.class_name} {t.confidence:.2f}" for t in tracked]

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
    return frame


def run_on_video(source: str, camera_id: str, output_path: str | None, process_every_n: int, device: str):
    cfg = PipelineConfig(camera_id=camera_id)
    cfg.process_every_n_frames = process_every_n
    cfg.detector.device = device

    pipeline = PerceptionPipeline(cfg)

    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        logger.error("Could not open video source: %s", source)
        sys.exit(1)

    writer = None
    if output_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    last_tracked: list[TrackedDetection] = []
    total_events = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        result = pipeline.process_frame(frame)
        if result is not None:
            last_tracked = result
            total_events += len(result)
            if result:
                logger.info("frame=%d tracks=%s", frame_idx,
                             [(t.track_id, t.class_name, round(t.confidence, 2)) for t in result])

        if writer is not None:
            writer.write(_annotate(frame, last_tracked))

    cap.release()
    if writer is not None:
        writer.release()
    logger.info("Done. frames=%d tracked_events=%d output=%s", frame_idx, total_events, output_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NETRA L2 Step 1-2 demo runner (detection + tracking)")
    p.add_argument("--source", required=True, help="Video file path, or '0' for webcam")
    p.add_argument("--camera-id", default="cam-00")
    p.add_argument("--output", default=None, help="Optional annotated .mp4 output path")
    p.add_argument("--every-n", type=int, default=1, help="Process every Nth frame (CPU-only boxes)")
    p.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 | mps")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_on_video(args.source, args.camera_id, args.output, args.every_n, args.device)