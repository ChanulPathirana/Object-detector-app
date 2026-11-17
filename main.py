"""Run real-time object detection from the default webcam."""

import time

import cv2

from config import WEBCAM_INDEX, WINDOW_NAME
from detector import ObjectDetector


def main():
    detector = ObjectDetector()
    camera = cv2.VideoCapture(WEBCAM_INDEX)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Cannot open webcam at index {WEBCAM_INDEX}")

    previous_time = time.perf_counter()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the webcam.")
                break

            result = detector.detect(frame)
            detector.draw_detections(frame, result)
            object_count = len(result.boxes)

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous_time = current_time

            # Add a dark information panel below the video frame.
            display = cv2.copyMakeBorder(
                frame, 0, 55, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35)
            )
            cv2.putText(
                display,
                f"FPS: {fps:.1f}   Objects: {object_count}   Q: quit",
                (12, frame.shape[0] + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(WINDOW_NAME, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
