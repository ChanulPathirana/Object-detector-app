import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Change this to your video filename
cap = cv2.VideoCapture("inputs/traffic.mp4")

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

while True:
    success, frame = cap.read()

    if not success:
        break

    # Detection only — no ByteTrack
    results = model.predict(
        frame,
        conf=0.25,
        device="cpu",
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow("YOLO Detection Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()