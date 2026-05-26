# Fast UAV Detection PoC

Standalone Python/OpenCV prototype for local UAV detection experiments from webcam, local video, RTSP streams, and offline media folders.

The current baseline uses Ultralytics YOLO with general COCO weights (`data_store/models/base/yolov8n.pt`). COCO does not include true `drone`, `uav`, or `quadcopter` classes, so this project can consolidate selected proxy labels. By default, `airplane` and `kite` are displayed and scored as `drone`.

## What Is Included

- `app/`: live detection app with YOLO inference, simple tracking, persistence alerts, and OpenCV UI.
- `scripts/assess_media.py`: batch media assessment and category output generation.
- `scripts/annotation_server.py`: local browser UI for manually annotating frames/images into YOLO labels.
- `scripts/export_assessment_pdf.py`: PDF exporter for customer-facing assessment reports.
- `configs/config.yaml`: runtime configuration for sources, thresholds, target classes, UI, and output behavior.
- `uav_detection_prototype*.md`: planning notes and prototype requirements.

Raw media, generated reports, annotated videos, local model weights, and local run artifacts are intentionally ignored by Git.

## Two-Step Deployment

On a new machine, project deployment is intentionally split into code and data:

1. Clone the Git repository:

```bash
git clone https://github.com/alexfok/UAVDetection.git
cd UAVDetection
```

2. Download the shared data store:

```bash
python3 scripts/datastore_sync.py sync-down --yes
```

After step 2, the local `data_store/` contains raw media, annotation datasets, detection results, model weights, system config, and generated stats. The command also recreates compatibility links such as `reports`, `certs`, `videos/Roni/raw_data`, and `annotations/web_drone_v1`.

Prerequisites for running the project are still needed on each machine, such as Python dependencies and an authenticated rclone remote named `uavdrive:`. If rclone is not configured yet, create the remote once:

```bash
rclone config create uavdrive drive scope drive root_folder_id 16qqTwiknaYpYArNKG_r-JaA7dUA816w9
```

If Google Drive Desktop is mounted instead of rclone, download with:

```bash
python3 scripts/datastore_sync.py sync-down --yes --backend local --local-remote-path <mounted-folder>
```

## Data Store

Project data lives under the canonical local data store:

```text
data_store/
  raw_data/              # source videos/images, e.g. data_store/raw_data/Roni/
  detection_results/     # timestamped assessments and comparison reports
  datasets/              # YOLO train/val annotation datasets
  models/                # base, external, and trained model weights
  system_config/         # local deployment config, certs, users/passwords
  stats/                 # generated dataset/raw-data summaries
  backups/               # optional snapshot copies
```

Initialize or repair the local layout:

```bash
python3 scripts/datastore_sync.py init --migrate-legacy
python3 scripts/datastore_sync.py stats
python3 scripts/datastore_sync.py doctor
```

The init command creates compatibility links for legacy paths such as `videos/Roni/raw_data`, `annotations/web_drone_v1`, `reports`, and `certs`, so older commands still resolve while new commands use `data_store/`.

The configured Google Drive folder for project data sync is:

```text
https://drive.google.com/drive/u/0/folders/16qqTwiknaYpYArNKG_r-JaA7dUA816w9
```

Recommended rclone setup is to create a Drive remote rooted at that folder, named `uavdrive`, then use the sync CLI:

```bash
rclone config create uavdrive drive scope drive root_folder_id 16qqTwiknaYpYArNKG_r-JaA7dUA816w9

python3 scripts/datastore_sync.py backup --dry-run
python3 scripts/datastore_sync.py backup
python3 scripts/datastore_sync.py sync-up --dry-run
python3 scripts/datastore_sync.py sync-up
python3 scripts/datastore_sync.py sync-down --dry-run
python3 scripts/datastore_sync.py sync-down --yes
```

For simultaneous multi-person editing, use `sync-up`/`sync-down` carefully at first; true two-way sync should be tested with `python3 scripts/datastore_sync.py bisync --dry-run --yes` before enabling periodic automation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Model weights are stored in `data_store/models/`. The base COCO model is expected at `data_store/models/base/yolov8n.pt`; if it is missing, Ultralytics can download `yolov8n.pt` when you explicitly pass that model name.

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
python scripts/assess_media.py data_store/raw_data/Roni \
  --save-annotated \
  --run-name roni_raw_data_detection_assessment \
  --device cpu \
  --annotate-batch-size 16
```

Analyze every third video frame while saving annotated output:

```bash
python scripts/assess_media.py data_store/raw_data/Roni \
  --save-annotated \
  --run-name roni_raw_data_detection_assessment_stride3 \
  --device cpu \
  --frame-step 3
```

## Manual Web Annotation

Start the annotation server. By default it listens on all interfaces (`0.0.0.0`) so another computer on the LAN can open it.

```bash
export ANNOTATION_SERVER_PASSWORD='choose-a-strong-password'
python3 scripts/annotation_server.py \
  --host 0.0.0.0 \
  --port 8765 \
  --default-folder data_store/raw_data/Roni \
  --project-dir data_store/datasets/web_drone_v1
```

Open:

```text
http://127.0.0.1:8765
```

From another computer on the same LAN, use this machine's IP address instead of `127.0.0.1`, for example:

```text
http://192.168.100.178:8765
```

The default username is `admin`. The password is read from `ANNOTATION_SERVER_PASSWORD`; if no password is provided, the server prints a generated one-time password at startup.

To run with HTTPS, create a local self-signed certificate:

```bash
mkdir -p data_store/system_config/certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout data_store/system_config/certs/annotation.key \
  -out data_store/system_config/certs/annotation.crt \
  -days 30 \
  -subj "/CN=drone-annotator"
```

Then start the server with TLS:

```bash
export ANNOTATION_SERVER_PASSWORD='choose-a-strong-password'
python3 scripts/annotation_server.py \
  --host 0.0.0.0 \
  --port 8765 \
  --certfile data_store/system_config/certs/annotation.crt \
  --keyfile data_store/system_config/certs/annotation.key \
  --default-folder data_store/raw_data/Roni \
  --project-dir data_store/datasets/web_drone_v1
```

Open:

```text
https://<this-machine-ip>:8765
```

Browsers will warn for a self-signed certificate; accept the warning only for your own trusted LAN.

If `8765` is already busy, either open the URL above to use the existing server, or start a second server on another port:

```bash
python3 scripts/annotation_server.py --port 8766
```

The web UI scans a local folder for movies/images. For videos, play to the desired moment, click `Capture`, draw boxes on the captured frame, then `Save Boxes`. For images, draw directly and save. `Save Negative` stores the current image/frame with an empty YOLO label file.

Saved annotations are written as a YOLO dataset:

```text
data_store/datasets/web_drone_v1/
  data.yaml
  manifest.csv
  images/
    train/
    val/
  labels/
    train/
    val/
```

## Training Smoke Test

Use this only to verify the training pipeline wiring. It creates a temporary tiny dataset from the current annotations, runs one CPU epoch, and writes the test model under `/private/tmp`; it is not expected to improve detection quality.

```bash
YOLO_CONFIG_DIR=/private/tmp/ultralytics MPLCONFIGDIR=/private/tmp/matplotlib \
.venv/bin/python scripts/train_yolov8n_drone.py \
  --smoke-from data_store/datasets/web_drone_v1 \
  --project /private/tmp/uav_train_runs \
  --name yolov8n_drone \
  --output-model /private/tmp/yolov8n_drone_smoke_best.pt \
  --device cpu \
  --workers 0
```

For real training later, first collect a reviewed train/val annotation set with enough positive and negative samples, then run the same script without `--smoke-from` and point `--data` at the reviewed dataset `data.yaml`.

Run the drone-specific YOLOv11x model from Hugging Face:

```bash
mkdir -p data_store/models/external
curl -L -o data_store/models/external/doguilmak_drone_yolo11x_best.pt \
  https://huggingface.co/doguilmak/Drone-Detection-YOLOv11x/resolve/main/weight/best.pt

python scripts/assess_media.py data_store/raw_data/Roni \
  --model data_store/models/external/doguilmak_drone_yolo11x_best.pt \
  --target-label drone \
  --conf 0.3 \
  --iou 0.5 \
  --save-annotated \
  --run-name roni_raw_data_drone_yolo11x \
  --device cpu
```

Compare two completed assessment JSON files:

```bash
python scripts/compare_assessments.py \
  data_store/detection_results/roni_raw_data_detection_assessment_20260520_173524/assessment.json \
  data_store/detection_results/<drone-yolo11x-run>/assessment.json \
  --baseline-name yolov8n-coco-proxy \
  --candidate-name drone-yolo11x \
  --output data_store/detection_results/model_comparison_yolov8n_vs_drone_yolo11x.md \
  --pdf-output data_store/detection_results/model_comparison_yolov8n_vs_drone_yolo11x.pdf
```

Export a customer-facing PDF for a completed run:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib XDG_CACHE_HOME=/private/tmp MPLBACKEND=Agg \
python scripts/export_assessment_pdf.py data_store/detection_results/roni_raw_data_detection_assessment_20260520_173524
```

## Media Categories

- Good media: at least one configured UAV-like target label was detected. With the current baseline model, `airplane` and `kite` are consolidated to the displayed label `drone`.
- Neutral media: one or more objects were detected, but no configured UAV-like target label was detected.
- Bad media: no object was detected above the configured confidence threshold.
- Unreadable media: the file could not be opened or decoded by the assessment pipeline.

## Output Layout

Annotated assessment runs are written under a timestamped folder:

```text
data_store/detection_results/<run-name>_<YYYYMMDD_HHMMSS>/
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

Each assessment run also writes `run_metadata.json` with total elapsed time, model load time, media processing time, analyzed frame counts, and category totals.

## Run History

| Date/time | Dataset | Command summary | Result destination | Summary |
|---|---|---|---|---|
| 2026-05-19 09:42 IDT | `videos/Roni/drive-download-20260519T062344Z-3-001` | `scripts/assess_media.py ... --device cpu` | `reports/roni_media_detection_assessment.md`, `reports/roni_media_detection_assessment.json` | Videos: 6 good, 14 neutral, 0 bad. Images: 0 good, 1 neutral, 0 bad. |
| 2026-05-20 17:55 IDT | `videos/Roni/raw_data` | `scripts/assess_media.py ... --save-annotated --run-name roni_raw_data_detection_assessment --device cpu --annotate-batch-size 16` | `reports/roni_raw_data_detection_assessment_20260520_173524/` | Videos: 9 good, 11 neutral, 0 bad. Images: 8 good, 9 neutral, 72 bad. |
| 2026-05-20 20:16 IDT | `videos/Roni/raw_data` | `scripts/assess_media.py ... --model yolov8n.pt --save-annotated --run-name roni_raw_data_yolov8n_coco_full_timed --device cpu` | `reports/roni_raw_data_yolov8n_coco_full_timed_20260520_201629/` | Full every-frame CPU run: 18m 44s. Videos: 9 good, 11 neutral, 0 bad. Images: 8 good, 9 neutral, 72 bad. |
| 2026-05-20 20:35 IDT | `videos/Roni/raw_data` | `scripts/assess_media.py ... --model models/doguilmak_drone_yolo11x_best.pt --target-label drone --conf 0.3 --save-annotated --run-name roni_raw_data_drone_yolo11x_full_timed --device cpu` | `reports/roni_raw_data_drone_yolo11x_full_timed_20260520_203524/` | Full every-frame CPU run: 2h 59m 54s. Videos: 19 good, 0 neutral, 1 bad. Images: 48 good, 0 neutral, 41 bad. |
| 2026-05-21 12:12 IDT | `videos/Roni/raw_data` | `scripts/compare_assessments.py ... --pdf-output ...` | `reports/model_comparison_yolov8n_vs_drone_yolo11x_full_20260521_121242/` | Comparative report with timing and per-video detected-frame KPI. Combined elapsed: 3h 18m 38s. |

Generated report artifacts are local by default and are not intended to be pushed to GitHub unless explicitly needed.

## Drone-Specific Model Evaluation

The Hugging Face model `doguilmak/Drone-Detection-YOLOv11x` is a YOLOv11x checkpoint trained for one class: `drone`.

Why it is useful here:

- It provides a true `drone` class instead of relying on COCO proxy labels such as `airplane`, `bird`, and `kite`.
- It loads directly with Ultralytics YOLO in the current code path.
- It can be used with `scripts/assess_media.py` by passing `--model data_store/models/external/doguilmak_drone_yolo11x_best.pt --target-label drone`.

Comparison caveat:

- `data_store/models/base/yolov8n.pt` is a broad COCO object detector. It can produce Good, Neutral, and Bad categories because it detects many object classes.
- `doguilmak/Drone-Detection-YOLOv11x` is a single-class drone detector. It will mostly produce Good or Bad categories; Neutral is not very meaningful because the model does not detect non-drone objects.
- Real performance comparison needs manually labeled ground truth. Without ground truth, the comparison report should be treated as triage: which files each model flags for review, where they agree, and where they disagree.

## Configuration

Edit `configs/config.yaml` for:

- `video.source`: webcam index, local video path, or RTSP URL.
- `video.frame_skip`: process one frame and skip N frames. `frame_skip: 2` analyzes every third frame.
- `detector.model_path`: YOLO model, for example `data_store/models/base/yolov8n.pt` or `data_store/models/trained/yolov8n_drone_best.pt`.
- `detector.confidence_threshold`: per-frame detection threshold.
- `detector.target_classes`: model class names to alert on.
- `detector.label_aliases`: displayed label consolidation, currently `airplane: drone` and `kite: drone`.
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
- Re-run the assessment with the drone model via `--model data_store/models/trained/yolov8n_drone_best.pt`.
- Compare proxy-model and drone-model results across the same run folders.
- Tune `confidence_threshold`, `image_size`, and `frame_step` for field performance.
