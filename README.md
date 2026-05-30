# Fast UAV Detection PoC

Standalone Python/OpenCV prototype for local UAV detection experiments from webcam, local video, RTSP streams, and offline media folders.

The default live detector uses the current trained single-class drone model (`data_store/models/trained/yolov8n_drone_best.pt`). The older COCO baseline (`data_store/models/base/yolov8n.pt`) is still available; COCO does not include true `drone`, `uav`, or `quadcopter` classes, so this project can consolidate selected proxy labels. By default, `airplane` and `kite` are displayed and scored as `drone`.

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

## Offline USB Deployment

Use this path when the target computer has no Internet access. The bundle contains the repo code, the current `data_store/`, install scripts, and a Python wheelhouse when package download succeeds on the packaging machine.

Prepare a USB bundle:

```bash
python3 scripts/prepare_offline_deployment.py /Volumes/ESD-USB
```

For a Windows x64 laptop target, build Windows wheels from the Mac packaging machine:

```bash
python3 scripts/prepare_offline_deployment.py /Volumes/ESD-USB \
  --bundle-name UAVDetection_offline_current \
  --wheel-platform windows-x64 \
  --wheel-python-version 311 \
  --force
```

The command creates a timestamped folder such as:

```text
/Volumes/ESD-USB/UAVDetection_offline_YYYYMMDD_HHMMSS/
```

On a macOS/Linux target computer, open a terminal and run:

```bash
cd /Volumes/ESD-USB/UAVDetection_offline_YYYYMMDD_HHMMSS
./install_offline.sh
```

On a Windows target computer, install Python 3.11 first, then open PowerShell and run:

```powershell
cd E:\UAVDetection_offline_current
.\install_offline.ps1
```

Or from Command Prompt:

```bat
E:\UAVDetection_offline_current\install_offline.cmd
```

The installer copies the bundled project to `~/UAVDetection` by default, then:

- creates `.venv`
- installs Python packages from `UAVDetection/wheelhouse/` without contacting the Internet
- initializes `data_store/` and compatibility links
- creates or reuses a self-signed HTTPS certificate
- installs automatic server startup on login/boot (`launchd` on macOS, user `systemd` on Linux, scheduled task on Windows)
- updates common browser home/start pages to the local server URL when browser profiles are found
- writes a local shortcut file, `UAVDetection_Server.url`

### Windows Patch Deployment

For a Windows laptop that already has the offline project installed, build a copy-only patch ZIP:

```bash
python3 scripts/prepare_windows_patch.py
```

The ZIP is written under `data_store/deployment_patches/`. On the Windows laptop:

```text
1. Download the patch ZIP from Google Drive.
2. Extract the ZIP.
3. Run install_patch.cmd.
```

By default, `install_patch.cmd` patches `%USERPROFILE%\UAVDetection`. If the project is installed elsewhere, pass the install directory:

```bat
install_patch.cmd D:\UAVDetection
```

The patch installer stops the `UAVDetection Annotation Server` scheduled task if it exists, copies the included files, then starts the task again.

Default local URL and login:

```text
https://127.0.0.1:8765
admin / admin123
```

Useful installer options:

```bash
./install_offline.sh --install-dir ~/UAVDetection
./install_offline.sh --install-dir ~/UAVDetection --force
./install_offline.sh --password admin123
./install_offline.sh --no-browser-homepage
./install_offline.sh --no-autostart
./install_offline.sh --allow-online
```

Important: Python wheels are platform-specific. A bundle prepared with the default wheelhouse on macOS is suitable for a compatible macOS target, but not for Windows or Jetson/Linux ARM. For the Windows laptop, use `--wheel-platform windows-x64 --wheel-python-version 311` and install Python 3.11 on the target first. For Jetson, prepare the bundle on Jetson or another compatible Linux ARM environment, or provide a matching wheelhouse manually.

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

Model weights are stored in `data_store/models/`. The default live model is expected at `data_store/models/trained/yolov8n_drone_best.pt`. The base COCO model is expected at `data_store/models/base/yolov8n.pt`; if it is missing, Ultralytics can download `yolov8n.pt` when you explicitly pass that model name.

## Live Detection Examples

Embedded or default USB camera:

```bash
python -m app.main --source 0
```

Specific USB camera:

```bash
python -m app.main --source usb:1
```

Local video:

```bash
python -m app.main --source videos/test.mp4
```

Local image:

```bash
python -m app.main --source data_store/raw_data/Roni/IMG_0980.PNG
```

RTSP stream:

```bash
python -m app.main --source "rtsp://user:password@camera-ip:554/stream1"
```

Named camera from the local registry, after you add a verified RTSP camera to `data_store/system_config/cameras.yaml`:

```bash
python -m app.main --list-cameras
export UAV_CAMERA_USER='camera-user'
export UAV_CAMERA_PASSWORD='camera-password'
python -m app.main --camera example_camera
```

The default camera registry path is `data_store/system_config/cameras.yaml`, which is ignored by Git because it is site-specific. It is empty by default; use `configs/cameras.example.yaml` as the tracked template when you are ready to add verified RTSP cameras.

Quick source/FPS test without loading YOLO:

```bash
python scripts/test_rtsp.py --list-cameras
python scripts/test_rtsp.py "rtsp://user:password@camera-ip:554/stream1" --seconds 10
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

The same server also has a `Live Detection` tab for source testing and live monitoring. It can stream annotated detections from named cameras, direct RTSP URLs, USB/embedded cameras, video files, or image files:

```bash
python3 scripts/annotation_server.py \
  --host 0.0.0.0 \
  --port 8765 \
  --default-folder data_store/raw_data/Roni \
  --project-dir data_store/datasets/web_drone_v1 \
  --camera-config data_store/system_config/cameras.yaml \
  --live-model data_store/models/trained/yolov8n_drone_best.pt
```

In the browser, open `Live Detection`, then choose a named camera, pick a cached local USB/embedded camera, pick a media file from the scanned annotation folder, or enter a custom source. Local camera discovery starts in the background at server startup when `data_store/system_config/local_cameras.json` is missing; later page loads use the saved camera list. Use `Scan Local` only when cameras are added/removed and you want to refresh that cache. Tune confidence, FPS, frame-skip, image size, and device (`auto`, `mps`, `cpu`, `cuda`). The `Fast` preset raises requested FPS, skips frames, and lowers inference image size; `Quality` does the opposite.

Enable `Record` before pressing `Start` to save the streamed frames into the selected media folder. Keep `Labels` checked to create shareable demo clips with the same detection boxes and object labels shown in Live Detection; uncheck `Labels` to save raw resized frames for later annotation. Recordings use browser-playable H.264 MP4 files named `record_DDMM_HH-MM.mp4`; labeled demo clips add `_labeled` before the extension. If a recording rolls over, later segments add `_02`, `_03`, and so on. The hyphen keeps filenames valid on Windows, and the media list refreshes after recording stops.

Recording files are capped with segment rollover: the server targets a conservative 28 MiB rollover point under a 30 MiB maximum. This keeps individual clips small enough to sync and annotate comfortably while preserving longer sessions as multiple ordered segments.

Live detection sessions write one JSON event per line to a daily log, with saved JPEG frames for drone detections:

```text
data_store/detection_results/live_events/YYYY-MM-DD/events.jsonl
data_store/detection_results/live_events/YYYY-MM-DD/frames/<session_id>/*.jpg
```

Event types include `start`, `stop`, `drone_detected`, `recording_started`, `recording_saved`, `recording_skipped`, and `error`. `drone_detected` rows include `tracks` with bounding boxes, object labels, confidence, and track IDs; recording events include whether the saved clip is a labeled demo or raw recording.

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

For real training, either point `--data` at a reviewed dataset snapshot, or let the script build a fresh snapshot from the annotation manifest. Snapshot modes are useful after recording/annotating new media:

```bash
# train from every annotated item
YOLO_CONFIG_DIR=/private/tmp/ultralytics MPLCONFIGDIR=/private/tmp/matplotlib \
.venv/bin/python scripts/train_yolov8n_drone.py \
  --dataset-scope all \
  --model data_store/models/trained/yolov8n_drone_best.pt \
  --project data_store/models/trained/runs \
  --name yolov8n_drone_incremental_all \
  --output-model data_store/models/trained/yolov8n_drone_best.pt

# train only annotations saved after the previous training metadata/model timestamp
YOLO_CONFIG_DIR=/private/tmp/ultralytics MPLCONFIGDIR=/private/tmp/matplotlib \
.venv/bin/python scripts/train_yolov8n_drone.py \
  --dataset-scope since-last \
  --model data_store/models/trained/yolov8n_drone_best.pt \
  --project data_store/models/trained/runs \
  --name yolov8n_drone_incremental_since_last \
  --output-model data_store/models/trained/yolov8n_drone_best.pt

# train annotations saved in a specific date range
YOLO_CONFIG_DIR=/private/tmp/ultralytics MPLCONFIGDIR=/private/tmp/matplotlib \
.venv/bin/python scripts/train_yolov8n_drone.py \
  --dataset-scope date-range \
  --from-date 2026-05-29 \
  --to-date 2026-05-29 \
  --model data_store/models/trained/yolov8n_drone_best.pt \
  --project data_store/models/trained/runs \
  --name yolov8n_drone_incremental_20260529 \
  --output-model data_store/models/trained/yolov8n_drone_best.pt
```

Use `--prepare-only` with any snapshot mode to validate the selected data without launching YOLO training. Each completed training run writes `uav_training_metadata.json` inside the Ultralytics run folder and updates `data_store/models/trained/yolov8n_drone_best.meta.json`; `--dataset-scope since-last` uses that metadata as its cutoff. If the selected rows have no validation split, the script carves a small validation item from the selection so YOLO still has train and val folders.

The web server also exposes these modes in the `Training` tab. Choose `Since last training`, `Date range`, or `All annotations`, keep `Prepare only` checked to validate the selected dataset, or clear it to launch training. The tab shows current state, elapsed time, approximate epoch progress, and a live training log tail. Only one training job runs at a time; `Stop` terminates the active process.

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

# Talk to Segal about fiber optic detection techniques

# check dataset:
  I found annoted drones dataset https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset
  I think it is from Ukraine (including Shaheds) - we can exclude them.  Did not download it yet but plan to do it and look into it.
# Jetson setup
# Local network setup for field
# Deploy project on Jetson
# UI for detection
  - choose source - folder, stream
  - detection - Support for video stream
  - detection notification - how to?
# Detection
  - Done
  - single video streams support
  - multiple video streams support
  - various video sources support - USB camera, RTSP camera, etc
  - align detection media player with annotation one
  - support WEB cam, remove home cameras
  - sort media by modification time - newset file on top
  - save detection log\events to file - by date: start, stop, drone detected + image frame
  - add record stream option, save it to media library for later annotation

  - verify default parameters
  - deploy and start automatically

  - video stream source from client side - e.g. mobile client support
  - lower resolution / frame-step filter


Demo feedback:
1. Done: local camera enumeration is cached in `data_store/system_config/local_cameras.json`; startup only scans in the background when the cache is missing, and browser refreshes use the cache unless `Scan Local` is pressed.
2. Done: Windows recording filenames now use `record_DDMM_HH-MM.mp4` because `:` is illegal in Windows paths.
3. CLI added: incremental training can build snapshots for all annotations, annotations since the last training metadata/model timestamp, or an inclusive saved-at date range.
4. Camera direction: prefer an outdoor IP camera with RTSP/H.264 for field use; for quick mobile tests, add a client-side `getUserMedia` phone/browser source that sends frames to the server.
