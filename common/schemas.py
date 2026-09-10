from dataclasses import dataclass, asdict
from time import time

@dataclass
class TrackedDetection:
    event_type: str = "vehicle.tracked"
    camera_id: str = ""
    frame_index: int = 0
    track_id: int = -1
    track_session_id: int = 0
    class_id: int = -1
    class_name: str = ""
    confidence: float = 0.0
    bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlateRead:
    event_type: str = "plate.read"
    camera_id: str = ""
    frame_index: int = 0
    track_id: int = -1
    track_session_id: int = 0
    vehicle_class_name: str = ""
    plate_text: str = ""
    plate_text_confidence: float = 0.0
    plate_detection_confidence: float = 0.0
    plate_bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    was_multiline: bool = False
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VehicleAttributes:

    event_type: str = "vehicle.attributes"
    camera_id: str = ""
    frame_index: int = 0
    track_id: int = -1
    track_session_id: int = 0
    vehicle_class_name: str = ""     
    color_name: str = ""             
    color_confidence: float = 0.0    
    body_type: str = ""             
    body_type_confidence: float = 0.0
    vehicle_bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReIDEmbedding:
    """Step 5 — one OSNet appearance embedding for one tracked vehicle,
    in one frame.

    This is the raw output stored to pgvector (Section 6) — matching
    logic (cosine similarity, threshold, "possible match for review" vs
    an alert) lives in Phase 4 / Section 3's tier 3, not here. Per
    ML_FLOW.md's confidence caveat: a Re-ID match is inherently less
    reliable than a plate match, and should never itself trigger an
    automatic alert.
    """

    event_type: str = "vehicle.reid_embedding"
    camera_id: str = ""
    frame_index: int = 0
    track_id: int = -1
    track_session_id: int = 0
    vehicle_class_name: str = ""
    embedding: tuple[float, ...] = ()   # L2-normalized, dim depends on backend (512 for OSNet x1_0)
    vehicle_bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time()

    def to_dict(self) -> dict:
        return asdict(self)