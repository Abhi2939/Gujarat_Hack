from __future__ import annotations

import logging

from . import db_writer, event_bus
from common.schemas import TrackedDetection, PlateRead, VehicleAttributes, ReIDEmbedding

logger = logging.getLogger("netra.l6.storage_pipeline")


class StoragePipeline:
    def __init__(self, pg_conn, redis_client):
        self.pg_conn = pg_conn
        self.redis_client = redis_client

    def write_tracked(self, events: list[TrackedDetection]) -> None:
        db_writer.insert_tracked_detections(self.pg_conn, events)
        event_bus.publish_tracked_detections(self.redis_client, events)

    def write_plate_reads(self, events: list[PlateRead]) -> None:
        db_writer.insert_plate_reads(self.pg_conn, events)
        event_bus.publish_plate_reads(self.redis_client, events)

    def write_vehicle_attributes(self, events: list[VehicleAttributes]) -> None:
        db_writer.insert_vehicle_attributes(self.pg_conn, events)
        event_bus.publish_vehicle_attributes(self.redis_client, events)

    def write_reid_embeddings(self, events: list[ReIDEmbedding]) -> None:
        db_writer.insert_reid_embeddings(self.pg_conn, events)
        event_bus.publish_reid_embeddings(self.redis_client, events)

    def write_frame(self, tracked=None, plate_reads=None, vehicle_attributes=None,
                     reid_embeddings=None) -> None:
        if tracked:
            self.write_tracked(tracked)
        if plate_reads:
            self.write_plate_reads(plate_reads)
        if vehicle_attributes:
            self.write_vehicle_attributes(vehicle_attributes)
        if reid_embeddings:
            self.write_reid_embeddings(reid_embeddings)