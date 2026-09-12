from typing import Any
import logging

logger = logging.getLogger(__name__)


class Tracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10, debug: bool = False,) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.debug = debug
        self.next_track_id = 1
        self.tracks: list[dict] = []
        
        logger.debug(
            "Tracker initialized: iou_threshold=%.2f, max_age=%d",
            self.iou_threshold,
            self.max_age,
        )

    def update(
        self,
        detections: list[dict],
        frame: Any,
    ) -> list[dict]:

        for track in self.tracks:
            track["age"] += 1

        
        tracked_detections = []
        matched_track_ids = set()

        for detection in detections:
            best_track = None
            best_iou = 0.0

            for track in self.tracks:

                if track["track_id"] in matched_track_ids:
                    continue

                if detection["class_id"] != track["class_id"]:
                    continue

                iou = self._calculate_iou(
                    detection["bbox"],
                    track["bbox"],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track is not None and best_iou >= self.iou_threshold:

                track_id = best_track["track_id"]

                best_track["bbox"] = detection["bbox"]
                best_track["class_id"] = detection["class_id"]
                best_track["age"] = 0

                matched_track_ids.add(track_id)

            else:

                track_id = self.next_track_id
                self.next_track_id += 1

                self.tracks.append(
                    {
                        "track_id": track_id,
                        "bbox": detection["bbox"],
                        "class_id": detection["class_id"],
                        "age": 0,
                    }
                )
            tracked_detection = detection.copy()
            tracked_detection["track_id"] = track_id

            tracked_detections.append(tracked_detection)

        self.tracks = [
            track
            for track in self.tracks
            if track["age"] <= self.max_age
        ]

        return tracked_detections

    @staticmethod
    def _calculate_iou(
        bbox_a: list[float],
        bbox_b: list[float],
    ) -> float:

        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)
        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_width = max(
            0,
            intersection_x2 - intersection_x1,
        )

        intersection_height = max(
            0,
            intersection_y2 - intersection_y1,
        )

        intersection_area = (
            intersection_width * intersection_height
        )

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)

        union_area = area_a + area_b - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area