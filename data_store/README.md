# UAVDetection Data Store

This directory is the canonical local data store. Large and sensitive contents are ignored by Git and can be synchronized separately.

Fresh deployment has two project-level steps:

1. Clone the repo.
2. Download this data store with `python3 scripts/datastore_sync.py sync-down --yes`.

The download step restores raw media, datasets, annotations, detection results, model weights, system config, stats, and compatibility links used by older paths.

For no-Internet deployment, create an offline USB bundle from the repo root:

```bash
python3 scripts/prepare_offline_deployment.py /Volumes/ESD-USB
```

For a Windows x64 laptop target, build a Windows wheelhouse:

```bash
python3 scripts/prepare_offline_deployment.py /Volumes/ESD-USB \
  --bundle-name UAVDetection_offline_current \
  --wheel-platform windows-x64 \
  --wheel-python-version 311 \
  --force
```

That bundle includes this `data_store/` directory, the project code, install scripts, and a Python wheelhouse when available. On macOS/Linux, run `./install_offline.sh` from the bundle root. On Windows, run `.\install_offline.ps1` or `install_offline.cmd` from the bundle root.

On a clean Windows machine, the installer copies the full bundled `data_store/`.
On an existing Windows install, the same installer preserves the installed
`data_store/` and updates only `system_config/cameras.yaml`, backing up the
previous file as `cameras.yaml.backup_YYYYMMDD_HHMMSS`. Use `--force` only when
you intentionally want to replace the whole install directory, including local
data.

Expected layout:

```text
data_store/
  raw_data/              # source videos/images, for example raw_data/Roni/
  detection_results/     # assessment outputs and comparison reports, grouped by run/date
  datasets/              # YOLO train/val datasets with images, labels, data.yaml, manifest.csv
  models/                # base, external, and trained model weights
  system_config/         # local deployment settings, certs, annotation users/passwords
  stats/                 # generated summaries such as dataset_stats.json
  backups/               # optional local or remote snapshot copies
```

Use `python3 scripts/datastore_sync.py init --migrate-legacy` to create the structure and migrate legacy ignored folders into this store.

Model layout:

```text
data_store/models/
  base/                  # base weights such as yolov8n.pt
  external/              # downloaded third-party checkpoints
  trained/               # locally trained project checkpoints
```
