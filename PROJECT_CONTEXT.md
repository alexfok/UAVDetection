# UAV Detection Project Context

Last updated: 2026-05-26

This is a shareable handoff for human associates and AI coding agents. It summarizes what exists, what was measured, where the artifacts live, and what to do next.

## Executive Summary

The project is a local/offline UAV detection proof of concept. It currently supports:

- live YOLO detection from webcam, video file, or RTSP source
- batch assessment of media folders into `good`, `neutral`, `bad`, and `unreadable`
- annotated output videos/images with bounding boxes
- manual web annotation into YOLO train/val datasets
- local data-store sync to Google Drive for multi-machine handoff
- YOLOv8n fine-tuning on the current manually annotated drone dataset

Current best project model:

```text
data_store/models/trained/yolov8n_drone_best.pt
```

This model is a single-class `drone` detector trained from the current annotation dataset. It improves the number of files marked as drone-positive versus the COCO proxy baseline, but the result is not yet a ground-truth accuracy claim. The next priority is visual review and more annotation, especially false positives and false negatives.

## Repository And Data Store

GitHub repository:

```text
https://github.com/alexfok/UAVDetection.git
```

Current code checkpoint:

```text
b6bff8d Fix trained detector paths and labels
```

Deployment is intentionally two-step:

```bash
git clone https://github.com/alexfok/UAVDetection.git
cd UAVDetection
python3 scripts/datastore_sync.py sync-down --yes
```

The second step expects an authenticated rclone remote named `uavdrive:` rooted at this Google Drive folder:

```text
https://drive.google.com/drive/u/0/folders/16qqTwiknaYpYArNKG_r-JaA7dUA816w9
```

The data store was synced up after the latest trained-model detection run. A final dry-run on 2026-05-26 showed `0 B` remaining to transfer.

## Data Layout

Large/generated artifacts are intentionally not tracked in Git. They live in `data_store/` and sync separately.

```text
data_store/
  raw_data/              # source media, currently data_store/raw_data/Roni
  detection_results/     # assessment runs, annotated media, comparison reports
  datasets/              # YOLO train/val datasets
  models/                # base, external, and trained weights
  system_config/         # local certs/config/users; do not publish secrets
  stats/                 # generated dataset/raw-data summaries
  backups/               # optional snapshots
```

Current raw data stats:

```text
data_store/raw_data/Roni
109 files total
20 videos
89 images
```

Current annotation datasets:

```text
data_store/datasets/web_drone_v1
  train: 254 total, 223 positive, 31 negative, 237 boxes
  val:   0 total

data_store/datasets/web_drone_v1_trainval_20260526_130000
  train: 203 total, 178 positive, 25 negative, 190 boxes
  val:   51 total, 45 positive, 6 negative, 47 boxes
```

`web_drone_v1_trainval_20260526_130000` is the train/val snapshot used for the latest training run.

## Model Inventory

```text
data_store/models/base/yolov8n.pt
  General COCO YOLOv8n baseline. COCO has no real drone class.

data_store/models/external/doguilmak_drone_yolo11x_best.pt
  External one-class drone YOLOv11x checkpoint from Hugging Face.

data_store/models/trained/yolov8n_drone_best.pt
  Current project-trained one-class drone YOLOv8n checkpoint.
```

Label consolidation is implemented in the code path:

```text
airplane -> drone
kite -> drone
```

As of commit `b6bff8d`, saved annotated media also uses the consolidated display label. Older annotated media generated before that fix may still show raw `kite` or `airplane` overlays.

## Latest Training Run

Training command summary:

```bash
YOLO_CONFIG_DIR=/private/tmp/ultralytics MPLCONFIGDIR=/private/tmp/matplotlib \
.venv/bin/python scripts/train_yolov8n_drone.py \
  --data data_store/datasets/web_drone_v1_trainval_20260526_130000/data.yaml \
  --model data_store/models/base/yolov8n.pt \
  --epochs 25 \
  --imgsz 640 \
  --batch 8 \
  --device cpu \
  --workers 0 \
  --patience 8 \
  --project data_store/models/trained/runs \
  --name yolov8n_drone_web_drone_v1_20260526_130000 \
  --output-model data_store/models/trained/yolov8n_drone_best.pt
```

Training artifacts:

```text
data_store/models/trained/yolov8n_drone_best.pt
data_store/models/trained/runs/yolov8n_drone_web_drone_v1_20260526_130000/
```

Incremental training is now supported by `scripts/train_yolov8n_drone.py` snapshot modes:

```bash
# all current annotations
.venv/bin/python scripts/train_yolov8n_drone.py --dataset-scope all --prepare-only

# annotations saved after the previous training metadata/model timestamp
.venv/bin/python scripts/train_yolov8n_drone.py --dataset-scope since-last --prepare-only

# annotations saved in a specific inclusive saved_at range
.venv/bin/python scripts/train_yolov8n_drone.py --dataset-scope date-range --from-date 2026-05-29 --to-date 2026-05-29 --prepare-only
```

Remove `--prepare-only` to train. Completed runs write `uav_training_metadata.json` in the Ultralytics run folder and refresh `data_store/models/trained/yolov8n_drone_best.meta.json`, which becomes the cutoff source for the next `since-last` run.

The annotation web UI has a third `Training` tab for the same workflow. It can prepare or launch `all`, `since-last`, and `date-range` jobs, polls `/api/training/status`, shows elapsed time, approximate epoch progress, and a live log tail, and prevents starting a second job while one is running.

Final validation snapshot from epoch 25:

```text
precision: 0.640
recall:    0.660
mAP50:     0.546
mAP50-95:  0.194
```

Interpretation: useful first fine-tune, but still early. The validation set is small and derived from the same current annotation effort.

## Detection Results

### Baseline: YOLOv8n COCO Proxy

Run:

```text
data_store/detection_results/roni_raw_data_detection_assessment_20260526_122421/
```

Model:

```text
data_store/models/base/yolov8n.pt
```

Elapsed:

```text
18m 16s on CPU
```

Results:

| Kind | Total | Good | Neutral | Bad | Unreadable |
|---|---:|---:|---:|---:|---:|
| video | 20 | 8 | 12 | 0 | 0 |
| image | 89 | 7 | 10 | 72 | 0 |

Notes:

- `Good` is driven by proxy labels consolidated to `drone`, mainly COCO `airplane` and `kite`.
- The model also detects many non-drone objects, so `neutral` is meaningful here.

### Candidate: Trained YOLOv8n Drone

Run:

```text
data_store/detection_results/roni_raw_data_yolov8n_drone_trained_20260526_140419/
```

Model:

```text
data_store/models/trained/yolov8n_drone_best.pt
```

Elapsed:

```text
28m 6s on CPU
```

Results:

| Kind | Total | Good | Neutral | Bad | Unreadable |
|---|---:|---:|---:|---:|---:|
| video | 20 | 18 | 0 | 2 | 0 |
| image | 89 | 29 | 0 | 60 | 0 |

Notes:

- This is a single-class detector, so it reports `drone` only.
- New assessment labels were verified clean: only `drone`, no `kite` or `airplane`.
- `Neutral` is usually not meaningful for one-class models because they do not detect non-drone objects.

### Comparison Report

Primary comparison artifacts:

```text
data_store/detection_results/model_comparison_yolov8n_coco_vs_trained_drone_20260526.md
data_store/detection_results/model_comparison_yolov8n_coco_vs_trained_drone_20260526.pdf
```

Headline comparison:

```text
Common files: 109
Status changes: 40
Candidate newly marks Good: 34
Candidate misses baseline Good: 2
```

Candidate misses baseline good:

```text
IMG_0969.MOV
Screenshot 2026-05-19 191110.png
```

Important caveat:

The comparison is model-behavior triage, not final accuracy measurement. There is no fully reviewed ground-truth test set yet, so the `good/bad` changes must be visually reviewed.

## Older Comparative Result: YOLOv8n vs YOLO11x

Earlier full CPU comparison against the external `doguilmak/Drone-Detection-YOLOv11x` model found:

```text
YOLOv8n COCO proxy full run: 18m 44s
YOLO11x drone full run:      2h 59m 54s
```

YOLO11x fired many drone positives, but visual inspection suggested too many false alerts. Because it is much slower on CPU and likely over-alerting for this dataset, the current direction is improving YOLOv8n with local annotation.

Related report:

```text
data_store/detection_results/model_comparison_yolov8n_vs_drone_yolo11x_full_20260521_121242/comparison.md
```

## Annotation Server

Start local/LAN annotation UI:

```bash
export ANNOTATION_SERVER_PASSWORD='choose-a-strong-password'
python3 scripts/annotation_server.py \
  --host 0.0.0.0 \
  --port 8765 \
  --default-folder data_store/raw_data/Roni \
  --project-dir data_store/datasets/web_drone_v1
```

Open locally:

```text
http://127.0.0.1:8765
```

For HTTPS, use the local cert/key under `data_store/system_config/certs/` or create a new self-signed pair. Do not share passwords in this document; use local environment variables or system config for actual credentials.

## Offline USB Deployment

Prepare a no-Internet deployment bundle from the repo root:

```bash
python3 scripts/prepare_offline_deployment.py /Volumes/ESD-USB
```

For a Windows x64 laptop target:

```bash
python3 scripts/prepare_offline_deployment.py /Volumes/ESD-USB \
  --bundle-name UAVDetection_offline_current \
  --wheel-platform windows-x64 \
  --wheel-python-version 311 \
  --force
```

The bundle contains:

- repo source without Git metadata or generated compatibility symlink folders
- current `data_store/`
- `wheelhouse/` for offline Python dependency installation when wheels can be downloaded on the packaging machine
- `install_offline.sh` at the bundle root
- `install_offline.ps1` and `install_offline.cmd` for Windows
- `scripts/install_offline_deployment.py` inside the project copy

On a macOS/Linux target computer:

```bash
cd /Volumes/ESD-USB/UAVDetection_offline_YYYYMMDD_HHMMSS
./install_offline.sh
```

On a Windows target computer, install Python 3.11 first, then run:

```powershell
cd E:\UAVDetection_offline_current
.\install_offline.ps1
```

The installer copies the project to `~/UAVDetection` on macOS/Linux or `%USERPROFILE%\UAVDetection` on Windows by default, creates `.venv`, installs dependencies from the wheelhouse, initializes `data_store`, installs an autostart service, and points common browser home/start pages to `https://127.0.0.1:8765`. Default login is `admin / admin123` unless overridden with `--password <value>`. Use `--install-dir <path>` for a different target directory.

For an existing Windows install, run the same `install_offline.ps1` or `install_offline.cmd` from the new bundle. It updates code/support files and preserves the installed `data_store/`; only `data_store/system_config/cameras.yaml` is refreshed, with a timestamped backup of the previous camera file. Existing raw media, annotations, results, models, certs, and credentials stay in place. Use `--no-camera-config-update` to skip that camera refresh. Use `--force` only for a destructive full replace of the install directory, including `data_store/`.

Platform caveat: build the wheelhouse on the same platform family as the target. A macOS bundle is not a Jetson/Linux ARM dependency bundle.

For an already installed offline Windows laptop, build a smaller copy-only patch ZIP:

```bash
python3 scripts/prepare_windows_patch.py
```

The ZIP is written under `data_store/deployment_patches/`. Download it from Google Drive on the laptop, extract it, then run `install_patch.cmd`. The default target is `%USERPROFILE%\UAVDetection`; pass another install directory as the first argument if needed. The patch command stops the `UAVDetection Annotation Server` scheduled task if present, copies the included files, and starts the task again.

Annotation workflow:

1. Select media folder.
2. Choose a video/image.
3. For video, play or seek to a useful moment and click `Capture`.
4. Draw drone boxes.
5. Use `Save Boxes` for positives.
6. Use `Save Negative` for reviewed frames/images with no drone.

## Common Commands

Run live detection from a local camera, one-off RTSP source, video, or image:

```bash
python -m app.main --source 0
python -m app.main --source usb:1
python -m app.main --list-cameras
python -m app.main --source "rtsp://user:password@camera-ip:554/stream1"
python -m app.main --source data_store/raw_data/Roni/IMG_0980.PNG
python -m app.main --source data_store/raw_data/Roni/IMG_0796.MOV
```

The web annotation server now includes a separate `Live Detection` tab. It uses the same camera registry, caches local USB/embedded camera discovery in `data_store/system_config/local_cameras.json`, starts discovery in the background only when that cache is missing, can re-scan local cameras on demand, can pick files from the currently scanned annotation media folder, exposes FPS/frame-skip/image-size/device controls plus fast/balanced/quality presets, and streams annotated MJPEG frames through `/api/live/stream`. The tab can also record processed live frames back into the selected media folder as browser-playable H.264 MP4 `record_DDMM_HH-MM.mp4` segments. Keep `Labels` checked for shareable demo clips with boxes/object labels burned in (`record_DDMM_HH-MM_labeled.mp4`); uncheck it for raw resized clips for later annotation. Segments roll over conservatively at 28 MiB to stay under a 30 MiB target file size. Live sessions write one JSON event per line to `data_store/detection_results/live_events/YYYY-MM-DD/events.jsonl`, with drone detection snapshots saved under `frames/<session_id>/`. `drone_detected` events include track labels, bounding boxes, confidence, and track IDs.

Camera registry:

```text
configs/cameras.example.yaml                 # tracked template
data_store/system_config/cameras.yaml        # local site-specific camera list, empty by default
```

Camera credentials should be supplied via environment variables such as `UAV_CAMERA_USER` and `UAV_CAMERA_PASSWORD`, not committed to Git.

Quick source/FPS test without loading YOLO:

```bash
python scripts/test_rtsp.py --list-cameras
python scripts/test_rtsp.py "rtsp://user:password@camera-ip:554/stream1" --seconds 10
```

Refresh stats:

```bash
python3 scripts/datastore_sync.py stats
```

Sync local data store to Google Drive:

```bash
python3 scripts/datastore_sync.py sync-up --dry-run
python3 scripts/datastore_sync.py sync-up
```

Download data store on another machine:

```bash
python3 scripts/datastore_sync.py sync-down --yes
```

Run trained detector on current raw data:

```bash
YOLO_CONFIG_DIR=/private/tmp/ultralytics MPLCONFIGDIR=/private/tmp/matplotlib \
.venv/bin/python scripts/assess_media.py \
  data_store/raw_data/Roni \
  --model data_store/models/trained/yolov8n_drone_best.pt \
  --target-label drone \
  --save-annotated \
  --run-name roni_raw_data_yolov8n_drone_trained \
  --device cpu \
  --annotate-batch-size 16
```

Compare two assessment runs:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib XDG_CACHE_HOME=/private/tmp MPLBACKEND=Agg \
.venv/bin/python scripts/compare_assessments.py \
  data_store/detection_results/roni_raw_data_detection_assessment_20260526_122421/assessment.json \
  data_store/detection_results/roni_raw_data_yolov8n_drone_trained_20260526_140419/assessment.json \
  --baseline-name yolov8n-coco-proxy-20260526 \
  --candidate-name yolov8n-drone-trained \
  --output data_store/detection_results/model_comparison_yolov8n_coco_vs_trained_drone_20260526.md \
  --pdf-output data_store/detection_results/model_comparison_yolov8n_coco_vs_trained_drone_20260526.pdf
```

## Next Steps

1. Visually review the latest trained-model detections, especially the 34 newly-good files and the 2 baseline-good misses.
2. Add annotations for false positives and false negatives discovered during that review.
3. Build a more reliable validation/test split; keep a reviewed holdout set that is not used for training.
4. Retrain YOLOv8n after adding more diverse positives and negatives.
5. Tune confidence threshold and frame sampling for field use. Current detection used `conf=0.5`, `iou=0.45`, `imgsz=640`.
6. Compare the retrained model against both the COCO proxy baseline and the external YOLO11x drone model.
7. Deploy the best candidate to Jetson for RTSP/FPS testing and TensorRT export.
8. Later, evaluate external datasets such as `lgrzybowski/seraphim-drone-detection-dataset`, excluding irrelevant Shahed-style content if needed.

## Suggested Tasks For A Human Associate

- Open the latest trained detection result folders and judge whether each `good` case is a true drone.
- Start with:
  - `IMG_0969.MOV`
  - `IMG_0975.MOV`
  - `Screenshot 2026-05-19 191110.png`
  - the 34 files in the "Candidate Newly Marks Good" table of the comparison report
- Add manual labels for missed drones.
- Save negative samples where the trained detector fires on non-drone objects.
- Record notes about conditions that fail: sky/clouds, wires, birds, buildings, distance, blur, motion, camera angle.

## Suggested Prompt For Another AI Agent

Use this if continuing the project in a new AI session:

```text
You are working in the UAVDetection repo. Read PROJECT_CONTEXT.md and README.md first. The repo code is on GitHub, while large artifacts live in data_store/ and sync through scripts/datastore_sync.py with rclone remote uavdrive:. Current trained model is data_store/models/trained/yolov8n_drone_best.pt. Latest trained detection run is data_store/detection_results/roni_raw_data_yolov8n_drone_trained_20260526_140419. Latest comparison against the YOLOv8n COCO proxy baseline is data_store/detection_results/model_comparison_yolov8n_coco_vs_trained_drone_20260526.md/pdf. Do not treat the comparison as ground-truth accuracy; use it for triage. Next task is visual review, annotation of false positives/false negatives, retraining, and Jetson deployment testing.
```

## Jetson Notes

Recommended Jetson role for now:

- inference benchmark
- RTSP testing
- TensorRT export
- field deployment experiment

Identify Jetson model:

```bash
cat /proc/device-tree/model
cat /etc/nv_tegra_release
```

TensorRT export example on Jetson:

```python
from ultralytics import YOLO
model = YOLO("data_store/models/trained/yolov8n_drone_best.pt")
model.export(format="engine", half=True)
```
