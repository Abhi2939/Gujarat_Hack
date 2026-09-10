from __future__ import annotations

from Storage_Events import db_writer
from common.schemas import VehicleMatch


def correlate_plate(conn, plate_text: str, exclude_camera_id: str | None = None,
                     max_fuzzy_distance: int = 2, limit: int = 20) -> list[VehicleMatch]:
    rows = db_writer.query_plates_fuzzy(conn, plate_text, max_distance=max_fuzzy_distance, limit=limit)

    out: list[VehicleMatch] = []
    for row_id, camera_id, track_session_id, track_id, matched_text, observed_at, distance in rows:
        if exclude_camera_id is not None and camera_id == exclude_camera_id:
            continue
        is_exact = distance == 0
        out.append(VehicleMatch(
            tier=1 if is_exact else 2,
            tier_name="exact_plate" if is_exact else "fuzzy_plate",
            matched_camera_id=camera_id,
            matched_track_session_id=track_session_id,
            matched_track_id=track_id,
            confidence=1.0 if is_exact else max(0.0, 1.0 - distance / (max_fuzzy_distance + 1)),
            requires_review=False,
            detail=f"plate='{matched_text}' distance={distance}",
        ))
    return out


def correlate_reid(conn, embedding: tuple[float, ...], exclude_camera_id: str | None = None,
                    min_similarity: float = 0.75, limit: int = 5) -> list[VehicleMatch]:
    rows = db_writer.query_reid_nearest(conn, embedding, exclude_camera_id=exclude_camera_id, limit=limit)

    out: list[VehicleMatch] = []
    for camera_id, track_session_id, track_id, observed_at, cosine_distance in rows:
        similarity = 1.0 - cosine_distance
        if similarity < min_similarity:
            continue
        out.append(VehicleMatch(
            tier=3,
            tier_name="reid",
            matched_camera_id=camera_id,
            matched_track_session_id=track_session_id,
            matched_track_id=track_id,
            confidence=similarity,
            requires_review=True,
            detail=f"cosine_distance={cosine_distance:.4f}",
        ))
    return out


def correlate_attributes(conn, color_name: str, body_type: str, exclude_camera_id: str | None = None,
                          within_minutes: int = 60, limit: int = 20) -> list[VehicleMatch]:
    camera_filter = "AND camera_id != %s" if exclude_camera_id else ""
    params = [color_name, body_type, within_minutes]
    if exclude_camera_id:
        params.append(exclude_camera_id)
    params.append(limit)

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT camera_id, track_session_id, track_id, observed_at "
            f"FROM vehicle_attributes "
            f"WHERE color_name = %s AND body_type = %s "
            f"  AND observed_at >= now() - (%s || ' minutes')::interval "
            f"  {camera_filter} "
            f"ORDER BY observed_at DESC "
            f"LIMIT %s;",
            params,
        )
        rows = cur.fetchall()

    return [
        VehicleMatch(
            tier=4,
            tier_name="attributes",
            matched_camera_id=camera_id,
            matched_track_session_id=track_session_id,
            matched_track_id=track_id,
            confidence=0.2,
            requires_review=True,
            detail=f"color={color_name} body_type={body_type}",
        )
        for camera_id, track_session_id, track_id, observed_at in rows
    ]