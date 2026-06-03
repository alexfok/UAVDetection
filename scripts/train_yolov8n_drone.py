from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import date, datetime, time
from pathlib import Path

import yaml

from app.torchvision_compat import install_torchvision_nms_fallback

install_torchvision_nms_fallback()

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or smoke-test a single-class yolov8n drone detector.")
    parser.add_argument("--data", type=Path, default=Path("data_store/datasets/web_drone_v1/data.yaml"))
    parser.add_argument("--project-dir", type=Path, default=Path("data_store/datasets/web_drone_v1"))
    parser.add_argument(
        "--dataset-scope",
        choices=["data", "all", "since-last", "date-range"],
        default="data",
        help=(
            "data uses --data directly. all/since-last/date-range build a timestamped training "
            "snapshot from the annotation manifest."
        ),
    )
    parser.add_argument("--from-date", help="Inclusive manifest saved_at lower bound for --dataset-scope date-range.")
    parser.add_argument("--to-date", help="Inclusive manifest saved_at upper bound for --dataset-scope date-range.")
    parser.add_argument("--snapshot-root", type=Path, default=Path("data_store/datasets/training_snapshots"))
    parser.add_argument(
        "--last-training-metadata",
        type=Path,
        default=Path("data_store/models/trained/yolov8n_drone_best.meta.json"),
        help="Metadata file used to find the cutoff for --dataset-scope since-last.",
    )
    parser.add_argument(
        "--include-existing-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For incremental scopes, keep the existing val split as a stable validation set.",
    )
    parser.add_argument(
        "--min-val",
        type=int,
        default=1,
        help="Minimum validation rows to carve from selected rows when the snapshot has no val rows.",
    )
    parser.add_argument("--model", default="data_store/models/base/yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", default="yolov8n_drone")
    parser.add_argument("--prepare-only", action="store_true", help="Build/validate the selected dataset and exit.")
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
    started_at = datetime.now().astimezone()
    data_path = args.data.resolve()
    project_dir = args.project.resolve()
    output_model = args.output_model.resolve()
    dataset_metadata: dict[str, object] = {"scope": "data", "data": portable_path(data_path)}

    if args.smoke_from:
        data_path = build_smoke_dataset(args.smoke_from, args.smoke_dir)
        dataset_metadata = {"scope": "smoke", "data": portable_path(data_path)}
        args.epochs = min(args.epochs, 1)
        args.imgsz = min(args.imgsz, 320)
        args.batch = 1
        args.patience = 1
        args.name = f"{args.name}_smoke"
    elif args.dataset_scope != "data":
        data_path, dataset_metadata = build_filtered_dataset(args)

    if not data_path.exists():
        raise SystemExit(f"Dataset config not found: {data_path}")
    if args.prepare_only:
        print(f"Prepared dataset config: {data_path}")
        return 0

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
    ended_at = datetime.now().astimezone()
    save_dir = Path(results.save_dir)
    best_model = save_dir / "weights" / "best.pt"
    if not best_model.exists():
        raise RuntimeError(f"Expected trained model not found: {best_model}")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, output_model)
    write_training_metadata(args, started_at, ended_at, data_path, save_dir, best_model, output_model, dataset_metadata)
    print(f"Copied best model to {output_model}")
    return 0


def build_filtered_dataset(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    project_dir = args.project_dir.resolve()
    manifest_path = project_dir / "manifest.csv"
    rows = read_manifest(manifest_path)
    if not rows:
        raise SystemExit(f"No manifest rows found in {manifest_path}")

    selected_rows, filter_metadata = filter_manifest_rows(rows, args)
    if not selected_rows:
        raise SystemExit(f"No annotations matched dataset scope {args.dataset_scope!r}.")

    rows_by_split = assign_snapshot_splits(rows, selected_rows, args)
    if not rows_by_split["train"] or not rows_by_split["val"]:
        raise SystemExit(
            "Training snapshots need at least one train and one val item. "
            f"Got train={len(rows_by_split['train'])}, val={len(rows_by_split['val'])}."
        )

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = args.snapshot_root.resolve() / f"{project_dir.name}_{args.dataset_scope}_{stamp}"
    data_path = write_dataset_snapshot(project_dir, snapshot_dir, rows_by_split)

    metadata: dict[str, object] = {
        "scope": args.dataset_scope,
        "source_project": portable_path(project_dir),
        "snapshot_dir": portable_path(snapshot_dir),
        "data": portable_path(data_path),
        "selected_rows": len(selected_rows),
        "train_rows": len(rows_by_split["train"]),
        "val_rows": len(rows_by_split["val"]),
        **filter_metadata,
    }
    (snapshot_dir / "snapshot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Built training snapshot at {snapshot_dir} "
        f"({len(rows_by_split['train'])} train, {len(rows_by_split['val'])} val)"
    )
    return data_path, metadata


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def filter_manifest_rows(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    if args.dataset_scope == "all":
        return rows, {"filter": "all"}

    if args.dataset_scope == "since-last":
        cutoff = last_training_cutoff(args.last_training_metadata, args.output_model)
        selected = [row for row in rows if row_saved_at(row) and row_saved_at(row) > cutoff]
        return selected, {"filter": "since-last", "from_exclusive": cutoff.isoformat()}

    if args.dataset_scope == "date-range":
        start = parse_datetime_bound(args.from_date, end_of_day=False) if args.from_date else None
        end = parse_datetime_bound(args.to_date, end_of_day=True) if args.to_date else None
        if not start and not end:
            raise SystemExit("--dataset-scope date-range requires --from-date, --to-date, or both.")
        selected = []
        for row in rows:
            saved_at = row_saved_at(row)
            if not saved_at:
                continue
            if start and saved_at < start:
                continue
            if end and saved_at > end:
                continue
            selected.append(row)
        return selected, {
            "filter": "date-range",
            "from": start.isoformat() if start else None,
            "to": end.isoformat() if end else None,
        }

    raise SystemExit(f"Unsupported dataset scope: {args.dataset_scope}")


def assign_snapshot_splits(
    all_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, list[dict[str, str]]]:
    if args.dataset_scope == "all":
        train_rows = [row for row in selected_rows if row_split(row) == "train"]
        val_rows = [row for row in selected_rows if row_split(row) == "val"]
        if not val_rows:
            train_rows, val_rows = carve_validation_rows(train_rows, args.min_val)
        return {"train": train_rows, "val": val_rows}

    train_rows = [row for row in selected_rows if row_split(row) != "val"]
    if args.include_existing_val:
        val_rows = [row for row in all_rows if row_split(row) == "val"]
    else:
        val_rows = [row for row in selected_rows if row_split(row) == "val"]

    if not val_rows:
        train_rows, val_rows = carve_validation_rows(train_rows, args.min_val)
    return {"train": train_rows, "val": val_rows}


def carve_validation_rows(
    train_rows: list[dict[str, str]],
    min_val: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not train_rows:
        return train_rows, []
    val_count = max(1, min(min_val, len(train_rows)))
    if len(train_rows) <= val_count:
        return train_rows, train_rows[:]
    return train_rows[:-val_count], train_rows[-val_count:]


def write_dataset_snapshot(
    project_dir: Path,
    snapshot_dir: Path,
    rows_by_split: dict[str, list[dict[str, str]]],
) -> Path:
    for split in ("train", "val"):
        (snapshot_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    for split, rows in rows_by_split.items():
        for row in rows:
            image_path = resolve_manifest_path(row.get("image_path", ""), project_dir)
            label_path = resolve_manifest_path(row.get("label_path", ""), project_dir, must_exist=False)
            if not image_path.exists():
                raise SystemExit(f"Manifest image not found: {row.get('image_path')}")

            image_name = image_path.name
            label_name = f"{image_path.stem}.txt"
            out_image = snapshot_dir / "images" / split / image_name
            out_label = snapshot_dir / "labels" / split / label_name
            shutil.copy2(image_path, out_image)
            if label_path.exists():
                shutil.copy2(label_path, out_label)
            else:
                out_label.write_text("", encoding="utf-8")

            manifest_row = dict(row)
            manifest_row["split"] = split
            manifest_row["image_path"] = portable_path(out_image)
            manifest_row["label_path"] = portable_path(out_label)
            manifest_rows.append(manifest_row)

    write_snapshot_manifest(snapshot_dir / "manifest.csv", manifest_rows)
    data_path = snapshot_dir / "data.yaml"
    data = {
        "path": str(snapshot_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: read_class_name(project_dir / "data.yaml")},
    }
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_path


def write_snapshot_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_manifest_path(value: str, project_dir: Path, must_exist: bool = True) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path, project_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if must_exist else candidates[0]


def row_split(row: dict[str, str]) -> str:
    split = (row.get("split") or "train").strip().lower()
    return split if split in {"train", "val"} else "train"


def row_saved_at(row: dict[str, str]) -> datetime | None:
    value = (row.get("saved_at") or "").strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed


def parse_datetime_bound(value: str, end_of_day: bool) -> datetime:
    text = value.strip()
    try:
        if len(text) == 10:
            parsed_date = date.fromisoformat(text)
            parsed = datetime.combine(parsed_date, time.max if end_of_day else time.min)
        else:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"Invalid date/time value: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed


def last_training_cutoff(metadata_path: Path, output_model: Path) -> datetime:
    metadata_path = metadata_path.resolve()
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            ended_at = metadata.get("ended_at")
            if ended_at:
                return parse_datetime_bound(str(ended_at), end_of_day=False)
        except (OSError, json.JSONDecodeError):
            pass

    output_model = output_model.resolve()
    if output_model.exists():
        return datetime.fromtimestamp(output_model.stat().st_mtime).astimezone()

    raise SystemExit(
        "--dataset-scope since-last needs existing training metadata or an existing --output-model timestamp."
    )


def write_training_metadata(
    args: argparse.Namespace,
    started_at: datetime,
    ended_at: datetime,
    data_path: Path,
    save_dir: Path,
    best_model: Path,
    output_model: Path,
    dataset_metadata: dict[str, object],
) -> None:
    metadata = {
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": (ended_at - started_at).total_seconds(),
        "model": args.model,
        "output_model": portable_path(output_model),
        "best_model": portable_path(best_model),
        "save_dir": portable_path(save_dir),
        "data": portable_path(data_path),
        "dataset": dataset_metadata,
        "train_args": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "patience": args.patience,
            "project": portable_path(args.project.resolve()),
            "name": args.name,
        },
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "uav_training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    args.last_training_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.last_training_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


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
