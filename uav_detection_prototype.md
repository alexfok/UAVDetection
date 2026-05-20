Project: Standalone Field Drone Recognition Prototype

Goal:
Build a standalone low-cost field system for visual drone detection. The system is NOT for home automation. It should detect drones in sky video and provide a local visual alert in v1.

Target v1 setup:
- Outdoor camera scans or observes the sky.
- Camera transmits video wirelessly to an analysis unit.
- Edge hardware runs detection software locally.
- If a drone is detected, the system displays a clear visual alert.
- No cloud dependency in v1.

Preferred hardware direction:
- Camera: 4K RTSP-capable outdoor camera, ideally PTZ or optical zoom capable.
- Wireless link: point-to-point Wi-Fi bridge, not generic unstable Wi-Fi.
- Edge compute: NVIDIA Jetson Orin Nano / Orin Nano Super class hardware.
- Display: HDMI monitor or local dashboard for visual alerts.
- Power: field battery / 12V power station.

Why NVIDIA Jetson:
- Drone recognition is a small-object detection and tracking problem.
- Raspberry Pi + Coral is likely insufficient for robust drone detection in sky.
- Jetson supports CUDA/TensorRT/GStreamer/OpenCV and is suitable for YOLO-style edge inference.
- Later optimization can use TensorRT.

Software architecture:
RTSP camera stream
  -> GStreamer / OpenCV capture
  -> object detector
  -> tracker
  -> temporal alert logic
  -> visual alert UI

Recommended model path:
- Start with YOLO family model: YOLOv8 / YOLOv9 / YOLO11 small or nano variant.
- Fine-tune on drone-in-sky data.
- Include negative samples: birds, clouds, planes, insects, trees, wires, sun glare.
- Detection alone is insufficient; add tracking and persistence filtering.

Alert logic:
Trigger alert only if:
- detection confidence exceeds threshold, e.g. > 0.55
- object persists across multiple frames, e.g. >= 5 frames within 2 seconds
- motion/track is consistent
- optional: reject objects with impossible size/motion patterns

Suggested modules:
1. video_input/
   - RTSP ingestion
   - reconnect handling
   - frame resizing / batching

2. detector/
   - YOLO inference
   - model loading
   - confidence filtering
   - TensorRT export path later

3. tracker/
   - ByteTrack / DeepSORT / Kalman tracker
   - track IDs
   - persistence counters

4. alert/
   - alert state machine
   - debounce logic
   - visual alert trigger

5. ui/
   - local display window
   - bounding boxes
   - confidence
   - “DRONE DETECTED” banner

6. config/
   - YAML config for camera URL, thresholds, model path, alert parameters

7. data_tools/
   - recording utility
   - frame extraction
   - annotation helpers

Prototype priority:
Phase 1:
- Fixed camera, no PTZ scan logic.
- Record real sky footage.
- Run detector on RTSP stream.
- Show bounding boxes and visual alert.

Phase 2:
- Add tracker and false-positive suppression.
- Evaluate against birds/clouds/planes.

Phase 3:
- Add PTZ scanning logic.
- Add zone-based detection.
- Add logging and event snapshots.

Core technical risk:
Small drones may occupy only a few pixels. Camera optics, zoom, resolution, exposure, and compression quality are as important as the AI model.

Implementation preference:
- Python first for fast prototyping.
- Use OpenCV + GStreamer where possible.
- Keep interfaces clean so detector can later move to TensorRT.
- Target NVIDIA Jetson deployment.
- Assume NVIDIA licensing/environment will be handled later.

Initial deliverable for Codex:
Create a clean Python prototype repository with:
- RTSP video ingestion
- YOLO inference wrapper
- simple tracker/persistence logic
- visual alert UI
- YAML configuration
- README with setup and run commands