from __future__ import annotations

import logging

logger = logging.getLogger("netra.phase4.watchlist")

WATCHLIST_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS watchlist_plates (
    plate_text  TEXT PRIMARY KEY,
    reason      TEXT NOT NULL DEFAULT '',
    added_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_watchlist_plates_trgm
    ON watchlist_plates USING gin (plate_text gin_trgm_ops);
"""


def init_watchlist_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(WATCHLIST_SCHEMA_SQL)
    conn.commit()


def add_plate(conn, plate_text: str, reason: str = "") -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO watchlist_plates (plate_text, reason) VALUES (%s, %s) "
            "ON CONFLICT (plate_text) DO UPDATE SET reason = EXCLUDED.reason;",
            (plate_text, reason),
        )
    conn.commit()


def remove_plate(conn, plate_text: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM watchlist_plates WHERE plate_text = %s;", (plate_text,))
    conn.commit()


def list_plates(conn) -> list[tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT plate_text, reason FROM watchlist_plates ORDER BY added_at;")
        return cur.fetchall()


def check_plate(conn, plate_text: str, max_fuzzy_distance: int = 2) -> list[tuple[str, str, int]]:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;")
        cur.execute(
            "SELECT plate_text, reason, levenshtein(plate_text, %s) AS distance "
            "FROM watchlist_plates "
            "WHERE plate_text %% %s "
            "  AND levenshtein(plate_text, %s) <= %s "
            "ORDER BY distance ASC "
            "LIMIT 5;",
            (plate_text, plate_text, plate_text, max_fuzzy_distance),
        )
        return cur.fetchall()