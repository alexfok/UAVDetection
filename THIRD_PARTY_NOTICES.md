# Third-Party Notices

This project is licensed under the GNU Affero General Public License v3.0
(`AGPL-3.0`). See `LICENSE`.

This file is a practical attribution and licensing note for the main third-party
software and model components used by the project. It is not legal advice and is
not a complete replacement for each dependency's own license metadata.

## Python Dependencies

Top-level runtime dependencies are listed in `requirements.txt`.

| Component | Role | License note |
|---|---|---|
| Ultralytics | YOLO model loading, inference, and training runtime | AGPL-3.0 by default. This is the main reason this project is licensed as AGPL-3.0 while Ultralytics remains in use. |
| OpenCV / `opencv-python` | Camera, video, image, and RTSP handling | Apache-2.0. |
| NumPy | Array processing | BSD-style license; see package metadata for bundled binary library notices. |
| PyYAML | YAML configuration parsing | MIT. |

Transitive dependencies installed by these packages keep their own licenses and
notices. When producing commercial distributions or offline bundles, keep the
Python environment's package metadata available alongside the source bundle.

## Model Weights

Model weights are stored outside Git under `data_store/models/`.

- `data_store/models/base/yolov8n.pt` is the base Ultralytics YOLOv8n model.
- `data_store/models/trained/yolov8n_drone_best.pt` is the current trained drone
  model used by the live detector. It is trained and loaded through the
  Ultralytics runtime and should be treated as part of the AGPL-3.0 stack unless
  a separate commercial Ultralytics license or permissive replacement stack is
  adopted.
- External checkpoints, such as `doguilmak/Drone-Detection-YOLOv11x`, should be
  checked against their source repository/model-card license before redistribution.

## Data Store

Raw media, annotations, generated datasets, detection results, and model files
are intentionally kept in `data_store/` and synchronized separately from Git.
Dataset publication should use a separate dataset license and should exclude
private or sensitive media unless all required permissions are in place.
