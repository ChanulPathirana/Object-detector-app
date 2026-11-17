# Real-Time Object Detection System

This application uses an OpenCV webcam feed and Ultralytics YOLO to detect objects in real time. It draws boxes and labels for each detection and shows FPS and the number of detected objects below the video.

## Setup

Use the existing virtual environment and keep `yolo26n.pt` in the project directory. The default settings in `config.py` use the CPU, confidence threshold `0.5`, and webcam index `0`.

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

If the existing environment is missing dependencies, install the direct requirements with `pip install -r requirements.txt`.

## Run

```bash
python main.py
```

Press **Q** in the video window to quit. Webcam recordings are saved in `outputs/`. Change the values in `config.py` if you need another camera or model.

## Process a video file

Place a local video in `inputs/`, then run:

```bash
python video_processor.py \
  --input inputs/traffic.mp4 \
  --output outputs/traffic_detected.mp4
```

The video processor uses the same `ObjectDetector` class as the webcam application. YOLO and ByteTrack process frames in order, and the detector draws boxes, tracking IDs, confidence scores, and movement trails. The result is displayed while processing and saved as an MP4 at the source video's resolution and playback FPS. Press **Q** to stop early; processed frames are still saved.
