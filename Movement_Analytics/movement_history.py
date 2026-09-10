from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sighting:
    camera_id: str
    track_session_id: int
    track_id: int
    observed_at: object
    plate_text_confidence: float


def get_movement_history(conn, plate_text: str, limit: int = 200) -> list[Sighting]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT camera_id, track_session_id, track_id, observed_at, plate_text_confidence "
            "FROM plate_reads "
            "WHERE plate_text = %s "
            "ORDER BY observed_at ASC "
            "LIMIT %s;",
            (plate_text, limit),
        )
        rows = cur.fetchall()
    return [Sighting(*row) for row in rows]


def get_camera_sequence(conn, plate_text: str, limit: int = 200) -> list[str]:
    return [s.camera_id for s in get_movement_history(conn, plate_text, limit=limit)]


def get_all_plate_sequences(conn, min_sightings: int = 2, limit_plates: int = 10_000) -> dict[str, list[str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT plate_text, camera_id, observed_at "
            "FROM plate_reads "
            "WHERE plate_text != '' "
            "ORDER BY plate_text, observed_at ASC "
            "LIMIT %s;",
            (limit_plates * 50,),
        )
        rows = cur.fetchall()

    sequences: dict[str, list[str]] = {}
    for plate_text, camera_id, observed_at in rows:
        sequences.setdefault(plate_text, []).append(camera_id)

    return {p: seq for p, seq in sequences.items() if len(seq) >= min_sightings}