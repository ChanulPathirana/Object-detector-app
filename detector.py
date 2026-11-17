"""YOLO inference and OpenCV drawing helpers."""

import cv2
from ultralytics import YOLO

from config import CONFIDENCE_THRESHOLD, DEVICE, MODEL_PATH


class ObjectDetector:
    """Run YOLO on images and draw the detected objects."""

    def __init__(
        self,
        model_path=MODEL_PATH,
        confidence=CONFIDENCE_THRESHOLD,
        device=DEVICE,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    def detect(self, frame):
        """Return the YOLO result for one video frame."""
        return self.model.predict(
            frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]

    def draw_detections(self, frame, result):
        """Draw boxes and class labels on a frame and return it."""
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            label = f"{result.names[class_id]} {score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return frame
