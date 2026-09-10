import numpy as np

from Detection_Tracking.end_to_end import EndToEndPerceptionPipeline
from common.schemas import TrackedDetection


class Tracking:
    def process_frame(self, frame):
        return [TrackedDetection(camera_id="cam", track_id=1, class_name="car")]

    def on_reconnect(self):
        self.reconnected = True


class Stage:
    def __init__(self, value):
        self.value = value

    def process(self, frame, tracked):
        return self.value


class Storage:
    def __init__(self):
        self.written = None

    def write_frame(self, **kwargs):
        self.written = kwargs


def test_orchestrator_runs_and_persists_each_enabled_stage():
    storage = Storage()
    pipeline = EndToEndPerceptionPipeline(
        Tracking(), fingerprint_pipeline=Stage(["attributes"]),
        reid_pipeline=Stage(["embedding"]), storage_pipeline=storage,
    )
    result = pipeline.process_frame(np.zeros((5, 5, 3), dtype=np.uint8))
    assert result.processed and result.attributes == ["attributes"]
    assert storage.written["reid_embeddings"] == ["embedding"]
