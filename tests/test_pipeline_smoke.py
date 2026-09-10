from Storage_Events import event_bus
from Storage_Events.db_writer import query_plates_fuzzy
from common.schemas import WatchlistAlert


class FakeRedis:
    def __init__(self):
        self.calls = []

    def xadd(self, stream, payload, **kwargs):
        self.calls.append((stream, payload, kwargs))
        return "1-0"


class Cursor:
    def execute(self, query, params):
        self.query, self.params = query, params

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Connection:
    def __init__(self):
        self.cur = Cursor()

    def cursor(self):
        return self.cur


def test_watchlist_alert_is_published_to_its_own_stream():
    redis = FakeRedis()
    event_bus.publish_watchlist_alert(redis, WatchlistAlert(camera_id="cam"))
    assert redis.calls[0][0] == "watchlist.alert"


def test_fuzzy_plate_query_returns_track_identity_columns():
    conn = Connection()
    query_plates_fuzzy(conn, "GJ01AB1234")
    assert "track_session_id, track_id" in conn.cur.query
