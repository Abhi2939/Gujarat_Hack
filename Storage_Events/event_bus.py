from __future__ import annotations

import json
import logging

import redis

from common.schemas import TrackedDetection, PlateRead, VehicleAttributes, ReIDEmbedding

logger = logging.getLogger("netra.l6.event_bus")

MAXLEN_APPROX = 100_000


def connect(host: str = "localhost", port: int = 6379, db: int = 0) -> redis.Redis:
    return redis.Redis(host=host, port=port, db=db, decode_responses=True)


def _publish(r: redis.Redis, event) -> str:
    stream = event.event_type
    payload = {k: json.dumps(v) for k, v in event.to_dict().items()}
    return r.xadd(stream, payload, maxlen=MAXLEN_APPROX, approximate=True)


def publish_tracked_detections(r: redis.Redis, events: list[TrackedDetection]) -> list[str]:
    return [_publish(r, e) for e in events]


def publish_plate_reads(r: redis.Redis, events: list[PlateRead]) -> list[str]:
    return [_publish(r, e) for e in events]


def publish_vehicle_attributes(r: redis.Redis, events: list[VehicleAttributes]) -> list[str]:
    return [_publish(r, e) for e in events]


def publish_reid_embeddings(r: redis.Redis, events: list[ReIDEmbedding]) -> list[str]:
    return [_publish(r, e) for e in events]


def read_recent(r: redis.Redis, event_type: str, count: int = 10) -> list[tuple[str, dict]]:
    entries = r.xrevrange(event_type, count=count)
    decoded = []
    for entry_id, fields in entries:
        decoded.append((entry_id, {k: json.loads(v) for k, v in fields.items()}))
    return decoded