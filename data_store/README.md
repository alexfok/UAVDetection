# UAVDetection Data Store

This directory is the canonical local data store. Large and sensitive contents are ignored by Git and can be synchronized separately.

Fresh deployment has two project-level steps:

1. Clone the repo.
2. Download this data store with `python3 scripts/datastore_sync.py sync-down --yes`.

The download step restores raw media, datasets, annotations, detection results, model weights, system config, stats, and compatibility links used by older paths.

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
