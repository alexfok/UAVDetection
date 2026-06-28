from __future__ import annotations

import argparse
import csv
import contextlib
import io
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import yaml

from scripts.train_yolov8n_drone import (
    assign_snapshot_splits,
    build_filtered_dataset,
    carve_validation_rows,
    filter_manifest_rows,
    last_training_cutoff,
    parse_datetime_bound,
    read_class_name,
    row_saved_at,
)


class TrainingPrepareTests(unittest.TestCase):
    def test_date_range_filter_and_carved_validation_split(self) -> None:
        rows = [
            manifest_row("a", "train", "2026-06-20T10:00:00+00:00"),
            manifest_row("b", "train", "2026-06-21T10:00:00+00:00"),
            manifest_row("c", "train", "2026-06-22T10:00:00+00:00"),
        ]
        args = argparse.Namespace(dataset_scope="date-range", from_date="2026-06-21", to_date="2026-06-22")
        selected, metadata = filter_manifest_rows(rows, args)
        self.assertEqual([row["image_id"] for row in selected], ["b", "c"])
        self.assertEqual(metadata["filter"], "date-range")

        split_args = argparse.Namespace(dataset_scope="date-range", include_existing_val=False, min_val=1)
        splits = assign_snapshot_splits(rows, selected, split_args)
        self.assertEqual([row["image_id"] for row in splits["train"]], ["b"])
        self.assertEqual([row["image_id"] for row in splits["val"]], ["c"])

    def test_since_last_uses_metadata_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "meta.json"
            metadata.write_text('{"ended_at": "2026-06-21T12:00:00+00:00"}', encoding="utf-8")
            cutoff = last_training_cutoff(metadata, Path(tmp) / "missing.pt")
            self.assertEqual(cutoff.isoformat(), "2026-06-21T12:00:00+00:00")

            rows = [
                manifest_row("old", "train", "2026-06-21T11:59:00+00:00"),
                manifest_row("new", "train", "2026-06-21T12:01:00+00:00"),
            ]
            args = argparse.Namespace(
                dataset_scope="since-last",
                last_training_metadata=metadata,
                output_model=Path(tmp) / "missing.pt",
            )
            selected, _metadata = filter_manifest_rows(rows, args)
            self.assertEqual([row["image_id"] for row in selected], ["new"])

    def test_build_filtered_dataset_copies_images_and_empty_missing_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            image_dir = project / "images" / "train"
            label_dir = project / "labels" / "train"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            (project / "data.yaml").write_text("names:\n  0: drone\n", encoding="utf-8")
            image_a = image_dir / "a.png"
            image_b = image_dir / "b.png"
            image_a.write_bytes(b"image-a")
            image_b.write_bytes(b"image-b")
            (label_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            manifest_path = project / "manifest.csv"
            write_manifest(
                manifest_path,
                [
                    manifest_row("a", "train", "2026-06-20T10:00:00+00:00", image_a, label_dir / "a.txt"),
                    manifest_row("b", "train", "2026-06-21T10:00:00+00:00", image_b, label_dir / "b.txt"),
                ],
            )

            args = argparse.Namespace(
                dataset_scope="all",
                project_dir=project,
                snapshot_root=root / "snapshots",
                min_val=1,
                include_existing_val=True,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                data_path, metadata = build_filtered_dataset(args)
            self.assertTrue(data_path.exists())
            self.assertEqual(metadata["train_rows"], 1)
            self.assertEqual(metadata["val_rows"], 1)
            data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
            self.assertEqual(data["names"][0], "drone")
            empty_labels = list(data_path.parent.rglob("labels/val/b.txt"))
            self.assertEqual(len(empty_labels), 1)
            self.assertEqual(empty_labels[0].read_text(encoding="utf-8"), "")

    def test_datetime_and_class_helpers(self) -> None:
        self.assertEqual(parse_datetime_bound("2026-06-21", end_of_day=False).date().isoformat(), "2026-06-21")
        self.assertIsNone(row_saved_at({"saved_at": "not-a-date"}))
        self.assertEqual(read_class_name(Path("/no/such/data.yaml")), "drone")
        self.assertEqual(len(carve_validation_rows([manifest_row("a"), manifest_row("b")], 1)[1]), 1)


def manifest_row(
    image_id: str,
    split: str = "train",
    saved_at: str = "2026-06-20T10:00:00+00:00",
    image_path: Path | None = None,
    label_path: Path | None = None,
) -> dict[str, str]:
    return {
        "image_id": image_id,
        "split": split,
        "image_path": str(image_path or f"images/{split}/{image_id}.png"),
        "label_path": str(label_path or f"labels/{split}/{image_id}.txt"),
        "source_media": f"{image_id}.mp4",
        "media_kind": "image",
        "frame_time": "",
        "image_width": "32",
        "image_height": "24",
        "box_count": "1",
        "reviewed": "true",
        "saved_at": saved_at,
        "notes": "",
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
