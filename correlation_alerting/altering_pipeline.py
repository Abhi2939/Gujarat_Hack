from __future__ import annotations

import logging
from datetime import datetime, timezone

from . import watchlist
from Storage_Events import event_bus
from common.schemas import PlateRead, WatchlistAlert

logger = logging.getLogger("netra.phase4.alerting")

ALERTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS watchlist_alerts (
    id                      BIGSERIAL PRIMARY KEY,
    camera_id               TEXT        NOT NULL,
    track_session_id        INTEGER     NOT NULL,
    track_id                INTEGER     NOT NULL,
    observed_plate_text     TEXT        NOT NULL,
    matched_watchlist_plate TEXT        NOT NULL,
    watchlist_reason        TEXT        NOT NULL,
    tier                    INTEGER     NOT NULL,
    tier_name               TEXT        NOT NULL,
    confidence               REAL       NOT NULL,
    requires_review          BOOLEAN    NOT NULL,
    observed_at             TIMESTAMPTZ NOT NULL
);
"""


def init_alerts_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(ALERTS_SCHEMA_SQL)
    conn.commit()


def _insert_alert(conn, alert: WatchlistAlert) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO watchlist_alerts "
            "(camera_id, track_session_id, track_id, observed_plate_text, "
            " matched_watchlist_plate, watchlist_reason, tier, tier_name, "
            " confidence, requires_review, observed_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);",
            (alert.camera_id, alert.track_session_id, alert.track_id,
             alert.observed_plate_text, alert.matched_watchlist_plate,
             alert.watchlist_reason, alert.tier, alert.tier_name,
             alert.confidence, alert.requires_review,
             datetime.fromtimestamp(alert.timestamp, tz=timezone.utc)),
        )
    conn.commit()


class AlertingPipeline:
    def __init__(self, pg_conn, redis_client, max_fuzzy_distance: int = 2):
        self.pg_conn = pg_conn
        self.redis_client = redis_client
        self.max_fuzzy_distance = max_fuzzy_distance
        watchlist.init_watchlist_schema(pg_conn)
        init_alerts_schema(pg_conn)

    def check_plate_read(self, plate_read: PlateRead) -> WatchlistAlert | None:
        if not plate_read.plate_text:
            return None

        hits = watchlist.check_plate(self.pg_conn, plate_read.plate_text,
                                      max_fuzzy_distance=self.max_fuzzy_distance)
        if not hits:
            return None

        matched_plate, reason, distance = hits[0]
        is_exact = distance == 0

        alert = WatchlistAlert(
            camera_id=plate_read.camera_id,
            track_session_id=plate_read.track_session_id,
            track_id=plate_read.track_id,
            observed_plate_text=plate_read.plate_text,
            matched_watchlist_plate=matched_plate,
            watchlist_reason=reason,
            tier=1 if is_exact else 2,
            tier_name="exact_plate" if is_exact else "fuzzy_plate",
            confidence=1.0 if is_exact else max(0.0, 1.0 - distance / (self.max_fuzzy_distance + 1)),
            requires_review=False,
        )

        _insert_alert(self.pg_conn, alert)
        event_bus.publish_watchlist_alert(self.redis_client, alert)
        logger.warning("WATCHLIST ALERT: plate=%s matched=%s tier=%s camera=%s",
                        plate_read.plate_text, matched_plate, alert.tier_name, plate_read.camera_id)
        return alert

    def check_plate_reads(self, plate_reads: list[PlateRead]) -> list[WatchlistAlert]:
        alerts = []
        for pr in plate_reads:
            alert = self.check_plate_read(pr)
            if alert:
                alerts.append(alert)
        return alerts