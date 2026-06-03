from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.torchvision_compat import install_torchvision_nms_fallback


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass
class ProjectSummary:
    media_files: int = 0
    extracted_images: int = 0
    train_images: int = 0
    val_images: int = 0
    prelabelled_images: int = 0
    prelabelled_boxes: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a YOLO annotation project from images/videos.")
    parser.add_argument("--media-dir", type=Path, default=Path("data_store/raw_data/Roni"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_store/datasets/roni_drone_v1"))
    parser.add_argument("--class-name", default="drone")
    parser.add_argument("--video-frame-step", type=int, default=15)
    parser.add_argument("--max-video-frames", type=int, default=0, help="0 means no per-video cap.")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--seed-proxy-labels", action="store_true")
    parser.add_argument(
        "--model",
        default="data_store/models/base/yolov8n.pt",
        help="Used only with --seed-proxy-labels.",
    )
    parser.add_argument("--source-label", action="append", default=["airplane", "kite"])
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    make_project_dirs(args.output_dir)

    media_paths = list_media(args.media_dir)
    if not media_paths:
        raise SystemExit(f"No supported media found under {args.media_dir}")

    model = None
    source_class_ids: list[int] = []
    model_names: dict[int, str] = {}
    if args.seed_proxy_labels:
        install_torchvision_nms_fallback()
        from ultralytics import YOLO

        model = YOLO(args.model)
        model_names = normalise_names(model.names)
        source_class_ids = resolve_source_class_ids(model_names, args.source_label)
        if not source_class_ids:
            raise SystemExit(f"None of {args.source_label} exist in model labels.")

    summary = ProjectSummary(media_files=len(media_paths))
    manifest_rows: list[dict[str, str]] = []

    for media_index, media_path in enumerate(media_paths, start=1):
        print(f"[{media_index}/{len(media_paths)}] {media_path.name}", flush=True)
        split = split_for_source(media_path, args.val_split)
        for frame_key, frame in iter_media_frames(media_path, args):
            labels = []
            if model is not None:
                labels = predict_proxy_labels(model, frame, args, source_class_ids)

            image_id = safe_image_id(media_path, frame_key)
            image_path = args.output_dir / "images" / split / f"{image_id}.jpg"
            label_path = args.output_dir / "labels" / split / f"{image_id}.txt"
            write_image_and_label(image_path, label_path, frame, labels)

            summary.extracted_images += 1
            summary.train_images += int(split == "train")
            summary.val_images += int(split == "val")
            if labels:
                summary.prelabelled_images += 1
                summary.prelabelled_boxes += len(labels)

            manifest_rows.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "source_media": str(media_path),
                    "frame_key": frame_key,
                    "reviewed": "false",
                    "box_count": str(len(labels)),
                    "notes": "",
                }
            )

    write_manifest(args.output_dir / "manifest.csv", manifest_rows)
    write_data_yaml(args.output_dir, args.class_name)
    write_project_readme(args, summary)
    write_metadata(args, summary, model_names, source_class_ids)

    print(f"Wrote annotation project: {args.output_dir}")
    print(
        f"Images={summary.extracted_images}, train={summary.train_images}, val={summary.val_images}, "
        f"prelabelled={summary.prelabelled_images}, boxes={summary.prelabelled_boxes}"
    )
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.video_frame_step < 1:
        raise SystemExit("--video-frame-step must be >= 1")
    if not 0 < args.val_split < 1:
        raise SystemExit("--val-split must be between 0 and 1")


def make_project_dirs(output_dir: Path) -> None:
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def list_media(media_dir: Path) -> list[Path]:
    suffixes = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
    return sorted(path for path in media_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def iter_media_frames(path: Path, args: argparse.Namespace):
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(path))
        if frame is not None:
            yield path.stem, resize_frame(frame, args.max_width, args.max_height)
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  skipped unreadable video: {path}", flush=True)
        return

    sampled = 0
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_index % args.video_frame_step == 0:
                yield f"frame_{frame_index:06d}", resize_frame(frame, args.max_width, args.max_height)
                sampled += 1
                if args.max_video_frames and sampled >= args.max_video_frames:
                    break
            frame_index += 1
    finally:
        cap.release()


def predict_proxy_labels(model: object, frame: np.ndarray, args: argparse.Namespace, source_class_ids: list[int]) -> list[str]:
    predict_args: dict[str, object] = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "classes": source_class_ids,
        "verbose": False,
    }
    if args.device:
        predict_args["device"] = args.device

    results = model.predict([frame], **predict_args)
    if not results:
        return []
    return yolo_result_to_single_class_labels(results[0], frame)


def yolo_result_to_single_class_labels(result: object, frame: np.ndarray) -> list[str]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    height, width = frame.shape[:2]
    labels: list[str] = []
    for x1, y1, x2, y2 in boxes.xyxy.cpu().numpy():
        x1 = float(np.clip(x1, 0, width - 1))
        y1 = float(np.clip(y1, 0, height - 1))
        x2 = float(np.clip(x2, 0, width - 1))
        y2 = float(np.clip(y2, 0, height - 1))
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        if box_width <= 1 or box_height <= 1:
            continue

        x_center = (x1 + x2) / 2 / width
        y_center = (y1 + y2) / 2 / height
        labels.append(f"0 {x_center:.6f} {y_center:.6f} {box_width / width:.6f} {box_height / height:.6f}")
    return labels


def write_image_and_label(image_path: Path, label_path: Path, frame: np.ndarray, labels: list[str]) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f"Unable to write {image_path}")
    label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")


def resize_frame(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    if max_width <= 0 or max_height <= 0:
        return frame
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def safe_image_id(source_path: Path, frame_key: str) -> str:
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    return f"{source_path.stem}_{digest}_{frame_key}".replace(" ", "_")


def split_for_source(path: Path, val_split: float) -> str:
    value = int(hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if value < val_split else "train"


def normalise_names(names: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def resolve_source_class_ids(names: dict[int, str], source_labels: list[str]) -> list[int]:
    wanted = {label.strip().lower() for label in source_labels if label.strip()}
    return [class_id for class_id, label in names.items() if label.lower() in wanted]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_data_yaml(output_dir: Path, class_name: str) -> None:
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_project_readme(args: argparse.Namespace, summary: ProjectSummary) -> None:
    seed_text = "yes" if args.seed_proxy_labels else "no"
    readme = f"""# Drone Annotation Project

- Created: {datetime.now().astimezone().isoformat()}
- Source media: `{args.media_dir}`
- Class: `{args.class_name}`
- Images: `{summary.extracted_images}`
- Proxy prelabels seeded: `{seed_text}`

Use `scripts/annotate_yolo.py --project-dir {args.output_dir}` to review and edit boxes.
Then export reviewed images with `scripts/export_reviewed_annotations.py --project-dir {args.output_dir}`.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")


def write_metadata(
    args: argparse.Namespace,
    summary: ProjectSummary,
    model_names: dict[int, str],
    source_class_ids: list[int],
) -> None:
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "summary": asdict(summary),
        "source_model_labels": {class_id: model_names[class_id] for class_id in source_class_ids},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
