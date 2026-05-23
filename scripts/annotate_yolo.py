from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass
class AnnotationState:
    image_path: Path
    label_path: Path
    class_name: str
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    scale: float = 1.0
    drawing: bool = False
    start_xy: tuple[int, int] | None = None
    cursor_xy: tuple[int, int] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny OpenCV YOLO annotator for one-class drone boxes.")
    parser.add_argument("--project-dir", type=Path, default=Path("annotations/roni_drone_v1"))
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-display-width", type=int, default=1600)
    parser.add_argument("--max-display-height", type=int, default=1000)
    parser.add_argument("--window-name", default="Drone YOLO Annotator")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = read_data_yaml(args.project_dir / "data.yaml")
    class_name = str(data.get("names", {0: "drone"}).get(0, "drone"))
    images = list_images(args.project_dir, args.split)
    if not images:
        raise SystemExit(f"No images found in {args.project_dir}")

    index = min(max(args.start, 0), len(images) - 1)
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    try:
        while 0 <= index < len(images):
            image_path = images[index]
            label_path = label_path_for_image(args.project_dir, image_path)
            image = cv2.imread(str(image_path))
            if image is None:
                index += 1
                continue

            state = AnnotationState(
                image_path=image_path,
                label_path=label_path,
                class_name=class_name,
                boxes=load_boxes(label_path, image.shape),
            )
            cv2.setMouseCallback(args.window_name, on_mouse, state)

            while True:
                display = render(image, state, index, len(images), args)
                cv2.imshow(args.window_name, display)
                key = cv2.waitKey(20) & 0xFF

                if key in {255, 0xFF}:
                    continue
                if key in {ord("q"), 27}:
                    save_boxes(state, image.shape)
                    mark_reviewed(args.project_dir, image_path, len(state.boxes))
                    return 0
                if key == ord("n"):
                    save_boxes(state, image.shape)
                    mark_reviewed(args.project_dir, image_path, len(state.boxes))
                    index += 1
                    break
                if key == ord("p"):
                    save_boxes(state, image.shape)
                    mark_reviewed(args.project_dir, image_path, len(state.boxes))
                    index = max(0, index - 1)
                    break
                if key == ord("s"):
                    save_boxes(state, image.shape)
                    mark_reviewed(args.project_dir, image_path, len(state.boxes))
                if key == ord("d") and state.boxes:
                    state.boxes.pop()
                if key == ord("c"):
                    state.boxes.clear()
                if key == ord("0"):
                    state.boxes.clear()
                    save_boxes(state, image.shape)
                    mark_reviewed(args.project_dir, image_path, 0)
                    index += 1
                    break
    finally:
        cv2.destroyWindow(args.window_name)

    return 0


def read_data_yaml(path: Path) -> dict[object, object]:
    if not path.exists():
        return {"names": {0: "drone"}}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = data.get("names", {0: "drone"})
    if isinstance(names, list):
        data["names"] = {index: value for index, value in enumerate(names)}
    return data


def list_images(project_dir: Path, split: str) -> list[Path]:
    splits = ["train", "val"] if split == "all" else [split]
    images: list[Path] = []
    for split_name in splits:
        image_dir = project_dir / "images" / split_name
        images.extend(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    return sorted(images)


def label_path_for_image(project_dir: Path, image_path: Path) -> Path:
    parts = image_path.relative_to(project_dir).parts
    split = parts[1]
    return project_dir / "labels" / split / f"{image_path.stem}.txt"


def load_boxes(label_path: Path, image_shape: tuple[int, ...]) -> list[tuple[int, int, int, int]]:
    if not label_path.exists():
        return []
    height, width = image_shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, x_center, y_center, box_width, box_height = parts
        cx = float(x_center) * width
        cy = float(y_center) * height
        bw = float(box_width) * width
        bh = float(box_height) * height
        x1 = int(round(cx - bw / 2))
        y1 = int(round(cy - bh / 2))
        x2 = int(round(cx + bw / 2))
        y2 = int(round(cy + bh / 2))
        boxes.append(clamp_box((x1, y1, x2, y2), width, height))
    return boxes


def save_boxes(state: AnnotationState, image_shape: tuple[int, ...]) -> None:
    height, width = image_shape[:2]
    lines = []
    for x1, y1, x2, y2 in state.boxes:
        x1, y1, x2, y2 = clamp_box((x1, y1, x2, y2), width, height)
        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)
        if box_width <= 1 or box_height <= 1:
            continue
        x_center = (x1 + x2) / 2 / width
        y_center = (y1 + y2) / 2 / height
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_width / width:.6f} {box_height / height:.6f}")
    state.label_path.parent.mkdir(parents=True, exist_ok=True)
    state.label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def on_mouse(event: int, x: int, y: int, _flags: int, state: AnnotationState) -> None:
    image_x = int(round(x / state.scale))
    image_y = int(round(y / state.scale))
    state.cursor_xy = (image_x, image_y)

    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.start_xy = (image_x, image_y)
    elif event == cv2.EVENT_LBUTTONUP and state.drawing and state.start_xy:
        x1, y1 = state.start_xy
        box = normalise_box((x1, y1, image_x, image_y))
        if abs(box[2] - box[0]) > 3 and abs(box[3] - box[1]) > 3:
            state.boxes.append(box)
        state.drawing = False
        state.start_xy = None


def render(image: np.ndarray, state: AnnotationState, index: int, total: int, args: argparse.Namespace) -> np.ndarray:
    display, scale = scaled_image(image, args.max_display_width, args.max_display_height)
    state.scale = scale

    for box in state.boxes:
        draw_box(display, scale_box(box, scale), state.class_name, (0, 220, 255))

    if state.drawing and state.start_xy and state.cursor_xy:
        draw_box(display, scale_box(normalise_box((*state.start_xy, *state.cursor_xy)), scale), state.class_name, (0, 0, 255))

    header = (
        f"{index + 1}/{total} | boxes={len(state.boxes)} | "
        "drag=add  d=delete-last  c=clear  0=negative+next  s=save  n/p=next/prev  q=quit"
    )
    cv2.rectangle(display, (0, 0), (display.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(display, header, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return display


def scaled_image(image: np.ndarray, max_width: int, max_height: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return image.copy(), 1.0
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA), scale


def draw_box(display: np.ndarray, box: tuple[int, int, int, int], label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
    cv2.putText(display, label, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def normalise_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = normalise_box(box)
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width - 1, x2)),
        max(0, min(height - 1, y2)),
    )


def scale_box(box: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    return tuple(int(round(value * scale)) for value in box)  # type: ignore[return-value]


def mark_reviewed(project_dir: Path, image_path: Path, box_count: int) -> None:
    manifest_path = project_dir / "manifest.csv"
    if not manifest_path.exists():
        return

    rows = []
    image_id = image_path.stem
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for row in reader:
            if row.get("image_id") == image_id:
                row["reviewed"] = "true"
                row["box_count"] = str(box_count)
                row["reviewed_at"] = datetime.now().astimezone().isoformat()
            rows.append(row)

    if "reviewed_at" not in fieldnames:
        fieldnames.append("reviewed_at")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

