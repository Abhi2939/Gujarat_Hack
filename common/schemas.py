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

@dataclass
class VehicleMatch:
    """One candidate match from Phase 4's tiered correlation search
    (ML_FLOW.md Section 3): exact plate / fuzzy plate / Re-ID / attributes
    only. `requires_review` is set by tier, not left to the caller to
    decide — tiers 1-2 (plate-based) are confident enough to drive an
    automatic alert; tiers 3-4 (Re-ID, attributes) are NOT, per the doc's
    explicit confidence caveat on Re-ID, and the same logic extends to
    the even-weaker attribute-only tier.
    """

    tier: int = 0
    tier_name: str = ""
    matched_camera_id: str = ""
    matched_track_session_id: int = 0
    matched_track_id: int = -1
    confidence: float = 0.0
    requires_review: bool = True
    detail: str = ""


@dataclass
class WatchlistAlert:
    """A plate read matched a watchlist entry. Only tiers 1-2 (exact/
    fuzzy plate) produce these — see VehicleMatch's docstring for why
    Re-ID/attribute matches don't directly become alerts in this scope."""

    event_type: str = "watchlist.alert"
    camera_id: str = ""
    track_session_id: int = 0
    track_id: int = -1
    observed_plate_text: str = ""
    matched_watchlist_plate: str = ""
    watchlist_reason: str = ""
    tier: int = 0
    tier_name: str = ""
    confidence: float = 0.0
    requires_review: bool = False
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time()

    def to_dict(self) -> dict:
        return asdict(self)