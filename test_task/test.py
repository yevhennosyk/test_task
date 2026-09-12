import json
import logging
import cv2

from tinydet.config import DetectionConfig, TrackingConfig
from tinydet.detector import Detector
from tinydet.tracker import Tracker


INPUT_VIDEO = "video.mp4"
OUTPUT_VIDEO = "output.mp4"
OUTPUT_JSON = "results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

def main() -> None:
    detector_config = DetectionConfig(
        confidence_threshold=0.25,
        debug=True,
    )

    tracking_config = TrackingConfig(
        iou_threshold=0.3,
        max_age=10,
    )

    detector = Detector(detector_config)

    tracker = Tracker(
        iou_threshold=tracking_config.iou_threshold,
        max_age=tracking_config.max_age,
    )

    video = cv2.VideoCapture(INPUT_VIDEO)

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: {INPUT_VIDEO}"
        )

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    results_log = []

    frame_number = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        frame_number += 1

        detections = detector.detect(frame)

        tracked_detections = tracker.update(
            detections,
            frame,
        )

        frame_result = {
            "frame": frame_number,
            "detections": tracked_detections,
        }

        results_log.append(frame_result)

        for detection in tracked_detections:
            x1, y1, x2, y2 = detection["bbox"]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            label = (
                f'{detection["class_name"]} '
                f'{detection["confidence"]:.2f} '
                f'ID:{detection["track_id"]}'
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        writer.write(frame)

    video.release()
    writer.release()

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(
            results_log,
            file,
            indent=2,
        )

    logger = logging.getLogger(__name__)
    logger.info("Processed frames: %d", frame_number)
    logger.info("Video saved to: %s", OUTPUT_VIDEO)
    logger.info("Results saved to: %s", OUTPUT_JSON)



if __name__ == "__main__":
    main()