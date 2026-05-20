Project: Fast Drone Detection PoC

Goal:
Build a fast-and-dirty standalone drone detection PoC for field testing.

Development flow:
1. Develop and test on Mac
2. Deploy to Raspberry Pi 5
3. Later migrate to NVIDIA Jetson

System purpose:
- Detect drones/UAVs in sky video
- Show local visual alert
- No cloud dependency
- Minimal infrastructure
- Prioritize speed of implementation over production quality

High-level architecture:
RTSP camera/video file
    ↓
YOLO object detection
    ↓
simple tracking/persistence logic
    ↓
visual alert UI

Development platforms:
- MacBook for development/testing/training
- Raspberry Pi 5 for first edge deployment
- NVIDIA Jetson later for performance scaling

Preferred stack:
- Python
- OpenCV
- Ultralytics YOLO
- RTSP video streams
- Simple OpenCV GUI

Avoid initially:
- Docker complexity
- distributed systems
- PTZ automation
- cloud services
- custom CUDA optimization
- TensorRT tuning
- Frigate/NVR frameworks

Recommended repository structure:

project/
├── app/
│   ├── main.py
│   ├── detector.py
│   ├── tracker.py
│   ├── alert.py
│   ├── config.py
│   └── ui.py
│
├── configs/
│   └── config.yaml
│
├── videos/
│
├── models/
│
├── scripts/
│   ├── extract_frames.sh
│   └── test_rtsp.py
│
├── requirements.txt
└── README.md

Primary implementation goals:
1. Read RTSP stream or local video
2. Run YOLO inference
3. Draw bounding boxes
4. Show confidence
5. Trigger visual alert
6. Maintain clean modular structure

Initial implementation should support:
- local mp4 video
- RTSP streams
- webcam fallback

Recommended initial model:
- yolov8n.pt
or
- yolov8s.pt

Do NOT train initially.
Start with pretrained YOLO models.

Later possible improvements:
- VisDrone pretrained weights
- custom fine-tuning
- TensorRT
- Hailo accelerator
- Jetson optimization

Expected PoC limitations:
- small distant drones may be missed
- birds may create false positives
- camera optics/zoom matter enormously
- wide-angle cameras are poor for drone detection

Recommended camera characteristics:
- RTSP support
- 1080p or 4K
- optical zoom preferred
- stable tripod mounting

Simple alert logic:
Raise alert only if:
- confidence > threshold
- detection persists across several frames

Suggested persistence example:
- confidence > 0.5
- object seen in >= 5 frames during 2 seconds

Mac development environment:
Python venv:
    python3 -m venv venv
    source venv/bin/activate

Install:
    pip install ultralytics opencv-python pyyaml

Basic inference example:

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("Drone Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

Raspberry Pi 5 deployment notes:
- Use Raspberry Pi OS 64-bit
- Prefer Pi5 8GB
- Initially run CPU-only
- Downscale frames to 720p or 1080p
- Later optional acceleration:
    - Hailo-8L
    - NCNN export

Jetson migration later:
- same Python architecture
- same YOLO pipeline
- later:
    - TensorRT
    - CUDA optimization
    - higher FPS
    - larger models

Recommended next tasks for Codex:
1. Generate clean Python project skeleton
2. Implement RTSP/video ingestion
3. Implement YOLO wrapper
4. Add simple persistence logic
5. Add visual alert overlay
6. Add YAML config support
7. Add logging
8. Add README with Mac and RPi instructions

Success criteria for first PoC:
- live video inference working
- visible bounding boxes
- alert when UAV visible
- runs on Mac and RPi5
- modular enough for later Jetson migration

Important engineering guidance:
- Optimize for iteration speed
- Keep architecture simple
- Avoid premature optimization
- Avoid overengineering
- Focus on validating detection feasibility first

Useful references:
Ultralytics docs:
 [oai_citation:0‡docs.ultralytics.com](https://docs.ultralytics.com/?utm_source=chatgpt.com)

Ultralytics GitHub:
 [oai_citation:1‡github.com](https://github.com/ultralytics/ultralytics?utm_source=chatgpt.com)

Raspberry Pi YOLO deployment:
 [oai_citation:2‡docs.ultralytics.com](https://docs.ultralytics.com/guides/raspberry-pi/?utm_source=chatgpt.com)

Jetson deployment guide:
 [oai_citation:3‡docs.ultralytics.com](https://docs.ultralytics.com/guides/nvidia-jetson/?utm_source=chatgpt.com)