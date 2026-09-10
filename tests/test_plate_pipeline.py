import numpy as np

from Plate_OCR.plate_detector import PlateDetection
from Plate_OCR.plate_pipeline import PlatePipeline
from common.schemas import TrackedDetection


class FakeDetector:
    def detect(self, crop):
        return [PlateDetection((2, 3, 8, 7), 0.9)]


class FakeReader:
    def read(self, crop):
        return "GJ01AB1234", 0.95, False


def test_plate_pipeline_associates_global_plate_coordinates():
    track = TrackedDetection(camera_id="cam-a", frame_index=4, track_id=9,
                             class_name="car", bbox_xyxy=(10, 20, 50, 60))
    result = PlatePipeline(FakeDetector(), FakeReader()).process(
        np.zeros((100, 100, 3), dtype=np.uint8), [track]
    )
    assert len(result) == 1
    assert result[0].plate_text == "GJ01AB1234"
    assert result[0].plate_bbox_xyxy == (12, 23, 18, 27)
