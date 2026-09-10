from dataclasses import dataclass, field


COCO_VEHICLE_CLASSES: dict[int, str] = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class DetectorConfig:
    weights: str = "yolov8n.pt"
    device: str = "cpu"  # "cpu" | "cuda" | "cuda:0" | "mps"
    imgsz: int = 640
    conf_threshold: float = 0.35
    iou_threshold: float = 0.5
    allowed_classes: dict[int, str] = field(default_factory=lambda: dict(COCO_VEHICLE_CLASSES))
    half_precision: bool = False


@dataclass
class TrackerConfig:
    track_activation_threshold: float = 0.35
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    minimum_consecutive_frames: int = 2
    frame_rate: float = 25.0


@dataclass
class SceneCutConfig:
    enabled: bool = True
    hist_correlation_floor: float = 0.55
    min_frames_between_cuts: int = 5


@dataclass
class PipelineConfig:
    camera_id: str = "cam-00"
    process_every_n_frames: int = 1
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    scene_cut: SceneCutConfig = field(default_factory=SceneCutConfig)