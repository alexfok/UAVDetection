# Fast UAV Detection PoC

Standalone Python/OpenCV prototype for local UAV detection experiments from webcam, local video, RTSP streams, and offline media folders.

The current baseline uses Ultralytics YOLO with general COCO weights (`yolov8n.pt`). COCO does not include true `drone`, `uav`, or `quadcopter` classes, so this project currently treats `airplane`, `bird`, and `kite` as UAV-like proxy labels until a drone-specific model is added.

## What Is Included

- `app/`: live detection app with YOLO inference, simple tracking, persistence alerts, and OpenCV UI.
- `scripts/assess_media.py`: batch media assessment and category output generation.
- `scripts/export_assessment_pdf.py`: PDF exporter for customer-facing assessment reports.
- `configs/config.yaml`: runtime configuration for sources, thresholds, target classes, UI, and output behavior.
- `uav_detection_prototype*.md`: planning notes and prototype requirements.

Raw media, generated reports, annotated videos, local model weights, and local run artifacts are intentionally ignored by Git.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Ultralytics downloads `yolov8n.pt` on first use unless a model path is provided.

## Live Detection Examples

Webcam:

```bash
python -m app.main --source 0
```

Local video:

```bash
python -m app.main --source videos/test.mp4
```

RTSP stream:

```bash
python -m app.main --source "rtsp://user:password@camera-ip:554/stream1"
```

Use the YAML config:

```bash
python -m app.main --config configs/config.yaml
```

Quit the OpenCV UI with `q` or `Esc`.

## Batch Assessment Examples

Initial sampled assessment without saving annotated media:

```bash
python scripts/assess_media.py videos/Roni/drive-download-20260519T062344Z-3-001 --device cpu
```

Full annotated assessment with timestamped output folder:

```bash
python scripts/assess_media.py videos/Roni/raw_data \
  --save-annotated \
  --run-name roni_raw_data_detection_assessment \
  --device cpu \
  --annotate-batch-size 16
```

Analyze every third video frame while saving annotated output:

```bash
python scripts/assess_media.py videos/Roni/raw_data \
  --save-annotated \
  --run-name roni_raw_data_detection_assessment_stride3 \
  --device cpu \
  --frame-step 3
```

Export a customer-facing PDF for a completed run:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib XDG_CACHE_HOME=/private/tmp MPLBACKEND=Agg \
python scripts/export_assessment_pdf.py reports/roni_raw_data_detection_assessment_20260520_173524
```

## Media Categories

- Good media: at least one configured UAV-like target label was detected. With the current baseline model, those proxy labels are `airplane`, `bird`, and `kite`.
- Neutral media: one or more objects were detected, but no configured UAV-like target label was detected.
- Bad media: no object was detected above the configured confidence threshold.
- Unreadable media: the file could not be opened or decoded by the assessment pipeline.

## Output Layout

Annotated assessment runs are written under a timestamped folder:

```text
reports/<run-name>_<YYYYMMDD_HHMMSS>/
  assessment.md
  assessment.json
  assessment.pdf
  good/
  neutral/
  bad/
  unreadable/
  images/
    good/
    neutral/
    bad/
    unreadable/
```

Video outputs are saved as annotated `.mp4` files with labels and bounding boxes. Image outputs are saved in their corresponding category folders with labels and boxes drawn where detections exist.

## Run History

| Date/time | Dataset | Command summary | Result destination | Summary |
|---|---|---|---|---|
| 2026-05-19 09:42 IDT | `videos/Roni/drive-download-20260519T062344Z-3-001` | `scripts/assess_media.py ... --device cpu` | `reports/roni_media_detection_assessment.md`, `reports/roni_media_detection_assessment.json` | Videos: 6 good, 14 neutral, 0 bad. Images: 0 good, 1 neutral, 0 bad. |
| 2026-05-20 17:55 IDT | `videos/Roni/raw_data` | `scripts/assess_media.py ... --save-annotated --run-name roni_raw_data_detection_assessment --device cpu --annotate-batch-size 16` | `reports/roni_raw_data_detection_assessment_20260520_173524/` | Videos: 9 good, 11 neutral, 0 bad. Images: 8 good, 9 neutral, 72 bad. |

Generated report artifacts are local by default and are not intended to be pushed to GitHub unless explicitly needed.

## Configuration

Edit `configs/config.yaml` for:

- `video.source`: webcam index, local video path, or RTSP URL.
- `video.frame_skip`: process one frame and skip N frames. `frame_skip: 2` analyzes every third frame.
- `detector.model_path`: YOLO model, for example `yolov8n.pt` or `models/drone.pt`.
- `detector.confidence_threshold`: per-frame detection threshold.
- `detector.target_classes`: model class names to alert on.
- `alert.persistence_frames`: required hits before alerting.
- `alert.window_seconds`: time window for persistence.
- `ui.save_output`: save annotated live output video.

## RTSP Sanity Check

```bash
python scripts/test_rtsp.py "rtsp://user:password@camera-ip:554/stream1" --seconds 10
```

## Extract Frames

Requires `ffmpeg`:

```bash
scripts/extract_frames.sh videos/test.mp4 frames/test 1
```

## Next Steps

- Add or train a drone-specific model with real `drone`/`uav` classes.
- Re-run the assessment with the drone model via `--model models/drone.pt`.
- Compare proxy-model and drone-model results across the same run folders.
- Tune `confidence_threshold`, `image_size`, and `frame_step` for field performance.
