
"""YOLO object detection, tracking, and drawing helpers."""

import cv2
from collections import defaultdict, deque
from ultralytics import YOLO

from config import CONFIDENCE_THRESHOLD, DEVICE, MODEL_PATH


class ObjectDetector:
    """Detect and track objects using YOLO and ByteTrack."""

    def __init__(
        self,
        model_path=MODEL_PATH,
        confidence=CONFIDENCE_THRESHOLD,
        device=DEVICE,
    ):
        # Load YOLO model
        self.model = YOLO(model_path)

        # Configuration
        self.confidence = confidence
        self.device = device

        # Store the last 30 center positions for each track ID
        self.track_history = defaultdict(
            lambda: deque(maxlen=30)
        )

    def detect(self, frame):
        """Detect and track objects in a video frame."""

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        return results[0]

    def draw_detections(self, frame, result):
        """Draw boxes, labels, tracking IDs, and movement trails."""

        # Copy original frame
        annotated = frame.copy()

        # Store detected object names
        detected_objects = []

        for box in result.boxes:

            # 1. Extract bounding-box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # 2. Extract class and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = self.model.names[class_id]

            detected_objects.append(class_name)

            # 3. Extract tracking ID
            track_id = None

            if box.id is not None:
                track_id = int(box.id.item())

            # 4. Draw green bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # 5. Create detection label
            if track_id is not None:
                label = (
                    f"ID:{track_id} "
                    f"{class_name} {confidence:.2f}"
                )
            else:
                label = f"{class_name} {confidence:.2f}"

            # Draw label
            cv2.putText(
                annotated,
                label,
                (max(x1, 0), max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # 6. Track movement
            if track_id is not None:

                # Calculate bounding-box center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Store new position
                history = self.track_history[track_id]

                history.append((center_x, center_y))

                # Draw blue movement trail
                for i in range(1, len(history)):
                    cv2.line(
                        annotated,
                        history[i - 1],
                        history[i],
                        (255, 0, 0),
                        2
                    )

                # Draw red center point
                cv2.circle(
                    annotated,
                    (center_x, center_y),
                    4,
                    (0, 0, 255),
                    -1
                )

        # Return annotated frame and detected objects
        return annotated, detected_objects
