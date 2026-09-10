from __future__ import annotations

import logging
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

logger = logging.getLogger("netra.l6.db")

REID_EMBEDDING_DIM = 512

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS vehicle_tracks (
    id              BIGSERIAL,
    camera_id       TEXT        NOT NULL,
    track_session_id INTEGER    NOT NULL,
    track_id        INTEGER     NOT NULL,
    frame_index     INTEGER     NOT NULL,
    class_name      TEXT        NOT NULL,
    confidence      REAL        NOT NULL,
    bbox_xyxy       REAL[4]     NOT NULL,
    observed_at     TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_vehicle_tracks_track
    ON vehicle_tracks (camera_id, track_session_id, track_id, observed_at DESC);

CREATE TABLE IF NOT EXISTS plate_reads (
    id                  BIGSERIAL,
    camera_id           TEXT        NOT NULL,
    track_session_id    INTEGER     NOT NULL,
    track_id            INTEGER     NOT NULL,
    frame_index         INTEGER     NOT NULL,
    vehicle_class_name  TEXT        NOT NULL,
    plate_text          TEXT        NOT NULL,
    plate_text_confidence     REAL  NOT NULL,
    plate_detection_confidence REAL NOT NULL,
    plate_bbox_xyxy     REAL[4]     NOT NULL,
    was_multiline       BOOLEAN     NOT NULL,
    observed_at         TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, observed_at)
);
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_plate_reads_text_trgm
    ON plate_reads USING gin (plate_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_plate_reads_track
    ON plate_reads (camera_id, track_session_id, track_id, observed_at DESC);

CREATE TABLE IF NOT EXISTS vehicle_attributes (
    id                  BIGSERIAL,
    camera_id           TEXT        NOT NULL,
    track_session_id    INTEGER     NOT NULL,
    track_id            INTEGER     NOT NULL,
    frame_index         INTEGER     NOT NULL,
    vehicle_class_name  TEXT        NOT NULL,
    color_name          TEXT        NOT NULL,
    color_confidence    REAL        NOT NULL,
    body_type           TEXT        NOT NULL,
    body_type_confidence REAL       NOT NULL,
    vehicle_bbox_xyxy   REAL[4]     NOT NULL,
    observed_at         TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_vehicle_attributes_track
    ON vehicle_attributes (camera_id, track_session_id, track_id, observed_at DESC);

CREATE TABLE IF NOT EXISTS reid_embeddings (
    id                  BIGSERIAL,
    camera_id           TEXT        NOT NULL,
    track_session_id    INTEGER     NOT NULL,
    track_id            INTEGER     NOT NULL,
    frame_index         INTEGER     NOT NULL,
    vehicle_class_name  TEXT        NOT NULL,
    embedding           VECTOR({dim}) NOT NULL,
    vehicle_bbox_xyxy   REAL[4]     NOT NULL,
    observed_at         TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_reid_embeddings_track
    ON reid_embeddings (camera_id, track_session_id, track_id, observed_at DESC);
""".format(dim=REID_EMBEDDING_DIM)

_HYPERTABLE_TABLES = ["vehicle_tracks", "plate_reads", "vehicle_attributes", "reid_embeddings"]


def _timescaledb_available(cur) -> bool:
    cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb';")
    return cur.fetchone() is not None


def _try_create_hypertables(conn) -> bool:
    with conn.cursor() as cur:
        if not _timescaledb_available(cur):
            logger.info("timescaledb extension not available — using plain Postgres tables "
                        "(fully correct, just no automatic time-partitioning).")
            return False

        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        for table in _HYPERTABLE_TABLES:
            cur.execute(
                "SELECT create_hypertable(%s, 'observed_at', if_not_exists => TRUE, "
                "migrate_data => TRUE);", (table,)
            )
        conn.commit()
        logger.info("timescaledb available — converted %s to hypertables.", _HYPERTABLE_TABLES)
        return True


def connect(dsn: str):
    conn = psycopg2.connect(dsn)
    psycopg2.extras.register_uuid()
    return conn


def init_schema(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()
    return _try_create_hypertables(conn)


@contextmanager
def cursor(conn):
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()