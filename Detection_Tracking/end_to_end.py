"""Runnable L2 orchestration from one decoded frame to durable events.

This module deliberately receives its optional ML components by dependency
injection.  A deployment can therefore start with detection/tracking only,
then enable plate, Re-ID, and persistence after its approved weights and
services are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from common.schemas import PlateRead, ReIDEmbedding, TrackedDetection, VehicleAttributes, WatchlistAlert


class TrackingStage(Protocol):
    def process_frame(self, frame: np.ndarray) -> list[TrackedDetection] | None:
        ...

    def on_reconnect(self) -> None:
        ...


class TrackStage(Protocol):
    def process(self, frame: np.ndarray, tracked: list[TrackedDetection]):
        ...


@dataclass
class FrameResult:
    tracked: list[TrackedDetection] = field(default_factory=list)
    plate_reads: list[PlateRead] = field(default_factory=list)
    attributes: list[VehicleAttributes] = field(default_factory=list)
    reid_embeddings: list[ReIDEmbedding] = field(default_factory=list)
    alerts: list[WatchlistAlert] = field(default_factory=list)
    processed: bool = True


class EndToEndPerceptionPipeline:
    def __init__(self, tracking: TrackingStage, *, plate_pipeline: TrackStage | None = None,
                 fingerprint_pipeline: TrackStage | None = None,
                 reid_pipeline: TrackStage | None = None, storage_pipeline=None,
                 alerting_pipeline=None):
        self.tracking = tracking
        self.plate_pipeline = plate_pipeline
        self.fingerprint_pipeline = fingerprint_pipeline
        self.reid_pipeline = reid_pipeline
        self.storage_pipeline = storage_pipeline
        self.alerting_pipeline = alerting_pipeline

    def on_reconnect(self) -> None:
        self.tracking.on_reconnect()

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        tracked = self.tracking.process_frame(frame)
        if tracked is None:  # frame intentionally skipped by the rate limiter
            return FrameResult(processed=False)

        result = FrameResult(tracked=tracked)
        if self.plate_pipeline:
            result.plate_reads = self.plate_pipeline.process(frame, tracked)
        if self.fingerprint_pipeline:
            result.attributes = self.fingerprint_pipeline.process(frame, tracked)
        if self.reid_pipeline:
            result.reid_embeddings = self.reid_pipeline.process(frame, tracked)
        if self.storage_pipeline:
            self.storage_pipeline.write_frame(
                tracked=result.tracked, plate_reads=result.plate_reads,
                vehicle_attributes=result.attributes, reid_embeddings=result.reid_embeddings,
            )
        if self.alerting_pipeline and result.plate_reads:
            result.alerts = self.alerting_pipeline.check_plate_reads(result.plate_reads)
        return result
