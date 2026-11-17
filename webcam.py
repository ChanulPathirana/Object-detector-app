
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    success, frame = cap.read()

    if not success:
        break

    height, width = frame.shape[:2]

    # Define a fixed bounding box
    x1, y1 = 100, 100
    x2, y2 = 350, 300

    # Draw rectangle
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Draw label
    cv2.putText(
        frame,
        "Object Region",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Display image resolution
    cv2.putText(
        frame,
        f"{width} x {height}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Webcam Overlay", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
