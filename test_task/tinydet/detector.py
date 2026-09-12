from typing import Any, Optional
import logging
from ultralytics import YOLO
from tinydet.config import DetectionConfig
from tinydet.device_utils import resolve_device

logger = logging.getLogger(__name__)

class Detector:
    def __init__(
        self,
        config: Optional[DetectionConfig] = None,) -> None:
        self.config = config or DetectionConfig()
        self.device = (
            self.config.device
            if self.config.device is not None
            else resolve_device()
        )

        self.model = YOLO(self.config.model_path)

        if self.config.debug:
            logger.debug(
                "Detector initialized: model=%s, device=%s, confidence=%.2f",
                self.config.model_path,
                self.device,
                self.config.confidence_threshold,
            )

    def detect(self, frame: Any) -> list[dict]:
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                bbox = box.xyxy[0].tolist()
                bbox = [float(value) for value in bbox]

                class_name = result.names[class_id]

                detections.append(
                    {
                        "bbox": bbox,
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name,
                    }
                )

        if self.config.debug:
            logger.debug(
                "Detected %d objects",
                len(detections),
            )

        return detections