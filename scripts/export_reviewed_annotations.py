from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reviewed annotation rows into a clean YOLO dataset.")
    parser.add_argument("--project-dir", type=Path, default=Path("annotations/roni_drone_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/roni_drone_reviewed_v1"))
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.project_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    for split in ("train", "val"):
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path)
    reviewed = [row for row in rows if row.get("reviewed", "").lower() == "true"]
    if not reviewed:
        raise SystemExit("No reviewed rows found.")

    copied = 0
    positives = 0
    for row in reviewed:
        split = row["split"]
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])
        if not image_path.exists():
            continue

        out_image = args.output_dir / "images" / split / image_path.name
        out_label = args.output_dir / "labels" / split / f"{image_path.stem}.txt"
        shutil.copy2(image_path, out_image)
        if label_path.exists():
            shutil.copy2(label_path, out_label)
        else:
            out_label.write_text("", encoding="utf-8")

        copied += 1
        positives += int(out_label.read_text(encoding="utf-8").strip() != "")

    data_yaml = read_data_yaml(args.project_dir / "data.yaml")
    data_yaml["path"] = str(args.output_dir.resolve())
    (args.output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps({"reviewed_images": copied, "positive_images": positives}, indent=2),
        encoding="utf-8",
    )

    print(f"Exported {copied} reviewed images to {args.output_dir}")
    print(f"Positive images: {positives}; negative images: {copied - positives}")
    return 0


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_data_yaml(path: Path) -> dict[object, object]:
    if not path.exists():
        return {"train": "images/train", "val": "images/val", "names": {0: "drone"}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


if __name__ == "__main__":
    raise SystemExit(main())

