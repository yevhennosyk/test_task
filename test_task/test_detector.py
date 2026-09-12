from tinydet.detector import Detector


detector = Detector()

detections = detector.detect("image.jpg")

for detection in detections:
    print(detection)