from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or smoke-test a single-class yolov8n drone detector.")
    parser.add_argument("--data", type=Path, default=Path("data_store/datasets/web_drone_v1/data.yaml"))
    parser.add_argument("--model", default="data_store/models/base/yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", default="yolov8n_drone")
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("data_store/models/trained/yolov8n_drone_best.pt"),
    )
    parser.add_argument(
        "--smoke-from",
        type=Path,
        help="Build a tiny temporary dataset from an annotation project and train for a wiring smoke test.",
    )
    parser.add_argument("--smoke-dir", type=Path, default=Path("/private/tmp/uav_yolo_smoke_dataset"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    project_dir = args.project.resolve()
    output_model = args.output_model.resolve()

    if args.smoke_from:
        data_path = build_smoke_dataset(args.smoke_from, args.smoke_dir)
        args.epochs = min(args.epochs, 1)
        args.imgsz = min(args.imgsz, 320)
        args.batch = 1
        args.patience = 1
        args.name = f"{args.name}_smoke"

    if not data_path.exists():
        raise SystemExit(f"Dataset config not found: {data_path}")

    train_args: dict[str, object] = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(project_dir),
        "name": args.name,
        "patience": args.patience,
        "workers": args.workers,
        "exist_ok": True,
    }
    if args.device:
        train_args["device"] = args.device

    model = YOLO(args.model)
    results = model.train(**train_args)
    save_dir = Path(results.save_dir)
    best_model = save_dir / "weights" / "best.pt"
    if not best_model.exists():
        raise RuntimeError(f"Expected trained model not found: {best_model}")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, output_model)
    print(f"Copied best model to {output_model}")
    return 0


def build_smoke_dataset(project_dir: Path, smoke_dir: Path) -> Path:
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)

    for split in ("train", "val"):
        (smoke_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (smoke_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    images = sorted(
        path
        for path in (project_dir / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"No annotated images found under {project_dir / 'images'}")

    train_images = images[:-1] if len(images) > 1 else images
    val_images = images[-1:] if len(images) > 1 else images
    copy_pairs(project_dir, smoke_dir, train_images, "train")
    copy_pairs(project_dir, smoke_dir, val_images, "val")

    class_name = read_class_name(project_dir / "data.yaml")
    data = {
        "path": str(smoke_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
    }
    data_path = smoke_dir / "data.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(
        f"Built smoke dataset at {smoke_dir} "
        f"({len(train_images)} train images, {len(val_images)} val images)"
    )
    return data_path


def copy_pairs(project_dir: Path, smoke_dir: Path, images: list[Path], split: str) -> None:
    for image_path in images:
        label_path = label_path_for_image(project_dir, image_path)
        out_image = smoke_dir / "images" / split / image_path.name
        out_label = smoke_dir / "labels" / split / f"{image_path.stem}.txt"
        shutil.copy2(image_path, out_image)
        if label_path.exists():
            shutil.copy2(label_path, out_label)
        else:
            out_label.write_text("", encoding="utf-8")


def label_path_for_image(project_dir: Path, image_path: Path) -> Path:
    parts = image_path.relative_to(project_dir).parts
    if len(parts) >= 3 and parts[0] == "images":
        return project_dir / "labels" / parts[1] / f"{image_path.stem}.txt"
    return image_path.with_suffix(".txt")


def read_class_name(data_path: Path) -> str:
    if not data_path.exists():
        return "drone"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    names = data.get("names", {0: "drone"})
    if isinstance(names, list):
        return str(names[0]) if names else "drone"
    return str(names.get(0) or names.get("0") or "drone")


if __name__ == "__main__":
    raise SystemExit(main())
