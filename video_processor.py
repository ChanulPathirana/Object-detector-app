"""Detect and track objects in a prerecorded video."""

import argparse
import math
from pathlib import Path

import cv2

from detector import ObjectDetector


def process_video(input_path, output_path):
    """Process one video and save the annotated frames as an MP4 file."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise ValueError(f"Input video must be an existing file: {input_path}")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Output video path must end in .mp4")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output video paths must be different")

    capture = cv2.VideoCapture(str(input_path))
    writer = None

    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")

        source_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        source_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        source_fps = capture.get(cv2.CAP_PROP_FPS)

        metadata = (source_width, source_height, source_fps)
        if any(not math.isfinite(value) or value <= 0 for value in metadata):
            raise ValueError(
                f"Invalid video resolution or FPS: {input_path} "
                f"(width={source_width}, height={source_height}, fps={source_fps})"
            )

        width, height = int(source_width), int(source_height)
        if width == 0 or height == 0:
            raise ValueError(f"Invalid resolution in input video: {input_path}")

        success, frame = capture.read()
        if not success or frame is None:
            raise ValueError(f"No readable frames in input video: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot initialize video writer: {output_path}")

        detector = ObjectDetector()
        processed_frames = 0

        while success and frame is not None:
            result = detector.detect(frame)
            annotated, objects = detector.draw_detections(frame, result)

            cv2.putText(
                annotated,
                f"Objects: {len(objects)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # VideoWriter needs every frame to match the source video metadata.
            if annotated.shape[:2] != (height, width):
                annotated = cv2.resize(annotated, (width, height))

            writer.write(annotated)
            processed_frames += 1

            cv2.imshow("Video Object Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            success, frame = capture.read()

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Video saved to: {output_path}")
    print(f"Processed frames: {processed_frames}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect and track objects in a local video file."
    )
    parser.add_argument("--input", required=True, help="Path to the source video")
    parser.add_argument("--output", required=True, help="Path for the annotated MP4")
    args = parser.parse_args()

    try:
        process_video(args.input, args.output)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
