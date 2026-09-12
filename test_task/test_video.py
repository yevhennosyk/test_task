import cv2

from tinydet.detector import Detector
from tinydet.tracker import Tracker


detector = Detector()
tracker = Tracker(iou_threshold=0.3, max_age=10)

video = cv2.VideoCapture("video.mp4")

if not video.isOpened():
    raise RuntimeError("Could not open video")


while True:
    success, frame = video.read()

    if not success:
        break

    detections = detector.detect(frame)

    tracked_detections = tracker.update(
        detections,
        frame,
    )

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
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imshow("TinyDet Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


   


video.release()
cv2.destroyAllWindows()