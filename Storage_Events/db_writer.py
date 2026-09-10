from __future__ import annotations

import logging
from datetime import datetime, timezone

import psycopg2.extras

from common.schemas import TrackedDetection, PlateRead, VehicleAttributes, ReIDEmbedding

logger = logging.getLogger("netra.l6.writer")


def _ts(event) -> datetime:
    return datetime.fromtimestamp(event.timestamp, tz=timezone.utc)


def insert_tracked_detections(conn, events: list[TrackedDetection]) -> int:
    if not events:
        return 0
    rows = [
        (e.camera_id, e.track_session_id, e.track_id, e.frame_index,
         e.class_name, e.confidence, list(e.bbox_xyxy), _ts(e))
        for e in events
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO vehicle_tracks "
            "(camera_id, track_session_id, track_id, frame_index, class_name, "
            " confidence, bbox_xyxy, observed_at) VALUES %s",
            rows,
        )
    conn.commit()
    return len(rows)


def insert_plate_reads(conn, events: list[PlateRead]) -> int:
    if not events:
        return 0
    rows = [
        (e.camera_id, e.track_session_id, e.track_id, e.frame_index,
         e.vehicle_class_name, e.plate_text, e.plate_text_confidence,
         e.plate_detection_confidence, list(e.plate_bbox_xyxy), e.was_multiline, _ts(e))
        for e in events
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO plate_reads "
            "(camera_id, track_session_id, track_id, frame_index, vehicle_class_name, "
            " plate_text, plate_text_confidence, plate_detection_confidence, "
            " plate_bbox_xyxy, was_multiline, observed_at) VALUES %s",
            rows,
        )
    conn.commit()
    return len(rows)


def insert_vehicle_attributes(conn, events: list[VehicleAttributes]) -> int:
    if not events:
        return 0
    rows = [
        (e.camera_id, e.track_session_id, e.track_id, e.frame_index,
         e.vehicle_class_name, e.color_name, e.color_confidence,
         e.body_type, e.body_type_confidence, list(e.vehicle_bbox_xyxy), _ts(e))
        for e in events
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO vehicle_attributes "
            "(camera_id, track_session_id, track_id, frame_index, vehicle_class_name, "
            " color_name, color_confidence, body_type, body_type_confidence, "
            " vehicle_bbox_xyxy, observed_at) VALUES %s",
            rows,
        )
    conn.commit()
    return len(rows)


def insert_reid_embeddings(conn, events: list[ReIDEmbedding]) -> int:
    if not events:
        return 0
    rows = [
        (e.camera_id, e.track_session_id, e.track_id, e.frame_index,
         e.vehicle_class_name, [float(v) for v in e.embedding],
         [float(v) for v in e.vehicle_bbox_xyxy], _ts(e))
        for e in events
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO reid_embeddings "
            "(camera_id, track_session_id, track_id, frame_index, vehicle_class_name, "
            " embedding, vehicle_bbox_xyxy, observed_at) VALUES %s",
            rows,
            template="(%s, %s, %s, %s, %s, %s::vector, %s, %s)",
        )
    conn.commit()
    return len(rows)


def query_plates_fuzzy(conn, plate_text: str, max_distance: int = 2, limit: int = 20):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;")
        cur.execute(
            "SELECT id, camera_id, track_session_id, track_id, plate_text, observed_at, "
            "       levenshtein(plate_text, %s) AS distance "
            "FROM plate_reads "
            "WHERE plate_text %% %s "
            "  AND levenshtein(plate_text, %s) <= %s "
            "ORDER BY distance ASC, observed_at DESC "
            "LIMIT %s;",
            (plate_text, plate_text, plate_text, max_distance, limit),
        )
        return cur.fetchall()


def query_reid_nearest(conn, embedding: tuple[float, ...], exclude_camera_id: str | None = None,
                        limit: int = 5):
    embedding_literal = "[" + ",".join(str(float(v)) for v in embedding) + "]"
    camera_filter = "AND camera_id != %s" if exclude_camera_id else ""

    params: list = [embedding_literal]
    if exclude_camera_id:
        params.append(exclude_camera_id)
    params.append(embedding_literal)
    params.append(limit)

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT camera_id, track_session_id, track_id, observed_at, "
            f"       embedding <=> %s::vector AS cosine_distance "
            f"FROM reid_embeddings "
            f"WHERE TRUE {camera_filter} "
            f"ORDER BY embedding <=> %s::vector ASC "
            f"LIMIT %s;",
            params,
        )
        return cur.fetchall()
