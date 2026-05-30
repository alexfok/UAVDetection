# UAVDetection Agent Notes

These notes are for AI coding agents and developers making changes in this repo.
Read `PROJECT_CONTEXT.md` and `README.md` before changing behavior; they describe the current system state, data-store layout, trained model, deployment flow, and latest results.

## Project Shape

- Source code lives in this Git repo.
- Large/runtime artifacts live under `data_store/` and sync through `scripts/datastore_sync.py`; do not move large media, model weights, detection results, or generated datasets into Git.
- The default trained model is `data_store/models/trained/yolov8n_drone_best.pt`.
- The main browser UI is served by `scripts/annotation_server.py` and uses `web/annotator/`.
- The live detector and source handling code lives under `app/`.

## Common Commands

Start the local HTTPS annotation/detection/training server:

```bash
.venv/bin/python scripts/annotation_server.py \
  --host 0.0.0.0 \
  --port 8765 \
  --password admin123 \
  --certfile data_store/system_config/certs/annotation.crt \
  --keyfile data_store/system_config/certs/annotation.key
```

Refresh data-store stats:

```bash
python3 scripts/datastore_sync.py stats
```

Sync data store to Google Drive:

```bash
python3 scripts/datastore_sync.py sync-up
```

Build a copy-only Windows patch for an already installed laptop:

```bash
python3 scripts/prepare_windows_patch.py
```

## Verification

For server/UI code changes, run at least:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/uav_pycache .venv/bin/python -m py_compile scripts/annotation_server.py app/sources.py
python3 -m html.parser web/annotator/index.html
```

If changing training code, also compile:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/uav_pycache .venv/bin/python -m py_compile scripts/train_yolov8n_drone.py
```

When the local server is running, a quick API check is:

```bash
curl -k -s -u admin:admin123 https://127.0.0.1:8765/api/training/status
```

## Workflow Rules

- Preserve the existing data-store layout: `raw_data`, `detection_results`, `datasets`, `models`, `system_config`, `stats`, `deployment_patches`.
- Keep browser UI changes mobile-aware; the app is used on laptops and phones.
- Keep labels consolidated as one class, `drone`, unless the user explicitly asks for multi-class training.
- Use `Prepare only` in the Training tab or `--prepare-only` in `scripts/train_yolov8n_drone.py` before launching longer training jobs.
- Treat model comparison outputs as triage/disagreement analysis unless a ground-truth validation set exists.
- Do not run long full-dataset CPU evaluations or training without making the runtime cost clear.
- Do not delete or rewrite user data in `data_store/` unless explicitly asked.

## Deployment Notes

- Full offline deployment is built with `scripts/prepare_offline_deployment.py`.
- Incremental Windows laptop updates should use `scripts/prepare_windows_patch.py`; the generated ZIP contains `install_patch.cmd`.
- Default Windows install target is `%USERPROFILE%\UAVDetection`.
- Default local URL is `https://127.0.0.1:8765`, default login is `admin / admin123`.
