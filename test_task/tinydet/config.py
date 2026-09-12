from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionConfig:
    model_path: str = "yolo11n.pt"
    confidence_threshold: float = 0.25
    device: Optional[str] = None
    debug: bool = False


@dataclass
class TrackingConfig:
    iou_threshold: float = 0.3
    max_age: int = 10
    debug: bool = False