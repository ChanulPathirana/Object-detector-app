
import cv2
import time
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo26n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

previous_time = time.perf_counter()

while True:
    success, frame = cap.read()

    if not success:
        break

    # Run object detection
    results = model.predict(
        frame,
        conf=0.5,
        device="cpu",
        verbose=False
    )

    result = results[0]

    # Count objects detected in this frame
    object_count = len(result.boxes)

    # Draw predicted boxes and labels
    annotated = result.plot()

    # Calculate FPS
    current_time = time.perf_counter()
    elapsed = current_time - previous_time
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    previous_time = current_time

    # Display FPS
    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Display object count
    cv2.putText(
        annotated,
        f"Objects: {object_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO Object Detector", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
