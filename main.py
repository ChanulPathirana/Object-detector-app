
"""Real-time YOLO detection, tracking, and video recording."""

import cv2
import time

from datetime import datetime
from pathlib import Path

from config import WEBCAM_INDEX, WINDOW_NAME
from detector import ObjectDetector


def main():
    # 1. Initialize object detector
    detector = ObjectDetector()

    # 2. Open webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    # 3. Create output folder
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Create a unique output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"detection_{timestamp}.mp4"

    # 5. Get webcam dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 6. Recording settings
    # Start with an estimated output frame rate
    recording_fps = 12.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        recording_fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot initialize video writer")

    previous_time = time.perf_counter()

    print(f"Recording to: {output_path}")

    try:
        while True:
            # 7. Capture frame
            success, frame = cap.read()

            if not success:
                print("Failed to capture frame")
                break

            # 8. Run YOLO + ByteTrack
            result = detector.detect(frame)

            # 9. Draw bounding boxes and trails
            annotated, objects = detector.draw_detections(
                frame, result
            )

            # 10. Calculate processing FPS
            current_time = time.perf_counter()
            elapsed = current_time - previous_time

            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous_time = current_time

            # 11. Draw information panel
            h, w = annotated.shape[:2]
            panel_height = 75
            panel_top = max(0, h - panel_height)

            cv2.rectangle(
                annotated,
                (0, panel_top),
                (w, h),
                (30, 30, 30),
                -1
            )

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (20, panel_top + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"Objects: {len(objects)}",
                (20, panel_top + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # 12. Save processed frame
            writer.write(annotated)

            # 13. Display processed frame
            cv2.imshow(WINDOW_NAME, annotated)

            # 14. Press Q to stop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        writer.release()
        cap.release()
        cv2.destroyAllWindows()

    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    main()
