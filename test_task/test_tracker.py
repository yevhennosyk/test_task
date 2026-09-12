from tinydet.tracker import Tracker


tracker = Tracker(
    iou_threshold=0.3,
    max_age=3,
)


frame_1 = [
    {
        "bbox": [100, 100, 200, 200],
        "confidence": 0.9,
        "class_id": 4,
        "class_name": "airplane",
    },
    {
        "bbox": [500, 100, 600, 200],
        "confidence": 0.8,
        "class_id": 4,
        "class_name": "airplane",
    },
]


frame_2 = [
    {
        "bbox": [110, 105, 210, 205],
        "confidence": 0.9,
        "class_id": 4,
        "class_name": "airplane",
    },
    {
        "bbox": [510, 105, 610, 205],
        "confidence": 0.8,
        "class_id": 4,
        "class_name": "airplane",
    },
]


frame_3 = [
    {
        "bbox": [120, 110, 220, 210],
        "confidence": 0.9,
        "class_id": 4,
        "class_name": "airplane",
    },
    {
        "bbox": [520, 110, 620, 210],
        "confidence": 0.8,
        "class_id": 4,
        "class_name": "airplane",
    },
]


for frame_number, detections in enumerate(
    [frame_1, frame_2, frame_3],
    start=1,
):
    result = tracker.update(detections, None)

    print(f"Frame {frame_number}:")

    for detection in result:
        print(
            f'  ID {detection["track_id"]}: '
            f'{detection["bbox"]}'
        )