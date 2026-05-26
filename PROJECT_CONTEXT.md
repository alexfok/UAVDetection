# UAV Detection Project Context

Last updated: 2026-05-26

This file is a handoff note for continuing the project on another machine, such as a MacBook Pro or NVIDIA Jetson.

## Repository

- GitHub: `https://github.com/alexfok/UAVDetection.git`
- Main branch: `main`
- Code baseline before datastore work: `3acb4d8 Improve annotation dashboard layout`

Project deployment on a new machine is two steps: clone the code, then download the shared data store.

```bash
git clone https://github.com/alexfok/UAVDetection.git
cd UAVDetection
python3 scripts/datastore_sync.py sync-down --yes
```

The data-store download expects rclone to have an authenticated `uavdrive:` remote rooted at the project Google Drive folder. If Google Drive Desktop is mounted instead, use `python3 scripts/datastore_sync.py sync-down --yes --backend local --local-remote-path <mounted-folder>`.

## Current Goal

Build a local/offline drone detection PoC:

- ingest local video, webcam, or RTSP stream
- run YOLO detection
- track/persist detections over time
- show a local visual alert
- collect manual annotations for a better drone-specific YOLOv8n fine-tune
- deploy/test inference on Jetson later

## Important Local Artifacts

The repository intentionally ignores large/generated/local artifacts:

- `data_store/` contents, except its README/gitkeep
- `videos/`
- `reports/`
- `annotations/`
- `datasets/`
- `models/`
- `runs/`
- `*.pt`, `*.onnx`, `*.engine`

This means a fresh clone has the code and docs, but not local videos, generated reports, trained weights, or annotation images/labels until the data-store download step is run.

The canonical local data layout is now:

```text
data_store/
  raw_data/
  detection_results/
  datasets/
  models/
  system_config/
  stats/
  backups/
```

Initialize/repair and check it with:

```bash
python3 scripts/datastore_sync.py init --migrate-legacy
python3 scripts/datastore_sync.py stats
python3 scripts/datastore_sync.py doctor
```

Legacy paths such as `videos/Roni/raw_data`, `reports`, `annotations/web_drone_v1`, and `certs` are compatibility links after migration.

Google Drive data-store sync target:

```text
https://drive.google.com/drive/u/0/folders/16qqTwiknaYpYArNKG_r-JaA7dUA816w9
```

For rclone, configure a remote rooted at that folder, then use `scripts/datastore_sync.py backup`, `sync-up`, `sync-down`, or later `bisync`.

## Setup On Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Quick syntax check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/uav_pycache python3 -m py_compile app/*.py scripts/*.py
```

## Setup On Jetson

First identify the Jetson model:

```bash
cat /proc/device-tree/model
cat /etc/nv_tegra_release
```

Recommended initial Jetson role:

- run inference and field tests
- benchmark RTSP/local video FPS
- export/test TensorRT engines

Training on Jetson is possible on stronger devices, but usually slower and less convenient than a desktop/cloud GPU. Treat Jetson as the deployment target unless the dataset is tiny or you only need a smoke test.

## Main App

Run live detection:

```bash
python -m app.main --source 0
python -m app.main --source videos/test.mp4
python -m app.main --source "rtsp://user:password@camera-ip:554/stream1"
```

Config:

```bash
configs/config.yaml
```

Current label consolidation:

- `airplane -> drone`
- `kite -> drone`

This lets the COCO `data_store/models/base/yolov8n.pt` proxy detections display and score as `drone` while a future single-class drone model can also use the same `drone` target.

## Web Annotation UI

Start server:

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

Open from another computer on the LAN:

```text
http://<server-ip>:8765
```

The default username is `admin`. The password is taken from `ANNOTATION_SERVER_PASSWORD`, or generated for one server run if the env var is absent.

For HTTPS, generate a self-signed cert and start with `--certfile` and `--keyfile`:

```bash
mkdir -p data_store/system_config/certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout data_store/system_config/certs/annotation.key \
  -out data_store/system_config/certs/annotation.crt \
  -days 30 \
  -subj "/CN=drone-annotator"

export ANNOTATION_SERVER_PASSWORD='choose-a-strong-password'
python3 scripts/annotation_server.py \
  --host 0.0.0.0 \
  --port 8765 \
  --certfile data_store/system_config/certs/annotation.crt \
  --keyfile data_store/system_config/certs/annotation.key \
  --default-folder data_store/raw_data/Roni \
  --project-dir data_store/datasets/web_drone_v1
```

Workflow:

- select a local media folder
- choose a video or image
- for video, play/pause/seek, then `Capture`
- draw drone boxes
- `Save Boxes` for positives
- `Save Negative` for reviewed no-drone frames

Output layout:

```text
data_store/datasets/web_drone_v1/
  data.yaml
  manifest.csv
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Current known local annotation state before this context file:

- only a few manually annotated frames existed locally
- not enough for meaningful training
- collect more positives and negatives before real training

Target before real training:

- at least `50-100` positive drone frames
- at least `50-100` negative frames
- include multiple videos, backgrounds, sizes, distances, lighting conditions
- include both `train` and `val` samples
- `200+` reviewed images is a better first serious target

## Training

Training script:

```bash
scripts/train_yolov8n_drone.py
```

Smoke test only:

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

This verifies the training pipeline only. It is not expected to improve the model.

Real training later:

```bash
.venv/bin/python scripts/train_yolov8n_drone.py \
  --data data_store/datasets/web_drone_v1/data.yaml \
  --model data_store/models/base/yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --output-model data_store/models/trained/yolov8n_drone_best.pt
```

Use `--device cpu` if no CUDA/MPS device is available. On Apple Silicon, try `--device mps` only after verifying PyTorch MPS works in the local environment.

## Assessment Scripts

Batch assessment:

```bash
python scripts/assess_media.py data_store/raw_data/Roni \
  --save-annotated \
  --run-name roni_raw_data_detection_assessment \
  --device cpu \
  --annotate-batch-size 16
```

Compare assessments:

```bash
python scripts/compare_assessments.py \
  data_store/detection_results/<baseline>/assessment.json \
  data_store/detection_results/<candidate>/assessment.json \
  --baseline-name yolov8n-coco-proxy \
  --candidate-name drone-model \
  --output data_store/detection_results/model_comparison.md
```

Export PDF:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib XDG_CACHE_HOME=/private/tmp MPLBACKEND=Agg \
python scripts/export_assessment_pdf.py data_store/detection_results/<run-folder>
```

## Jetson Transfer Pattern

After training elsewhere:

```bash
scp data_store/models/trained/yolov8n_drone_best.pt <user>@<jetson-ip>:~/UAVDetection/data_store/models/trained/
```

Run on Jetson:

```bash
python -m app.main --model data_store/models/trained/yolov8n_drone_best.pt --source "<rtsp-or-video>"
```

Later TensorRT export on Jetson:

```python
from ultralytics import YOLO
model = YOLO("data_store/models/trained/yolov8n_drone_best.pt")
model.export(format="engine", half=True)
```

## Current Next Steps

1. Continue manual annotation in `data_store/datasets/web_drone_v1`.
2. Add negative samples, not just positives.
3. Once there are enough reviewed images, run real YOLOv8n training.
4. Compare trained YOLOv8n against:
   - current COCO proxy behavior
   - existing drone YOLO11x checkpoint
5. Deploy best candidate to Jetson for RTSP/FPS testing.
