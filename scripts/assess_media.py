from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from app.torchvision_compat import install_torchvision_nms_fallback

install_torchvision_nms_fallback()

from ultralytics import YOLO


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_TARGET_LABELS = ("drone",)
DEFAULT_LABEL_ALIASES = ("airplane=drone", "kite=drone")
DEFAULT_OUTPUT_MD = Path("data_store/detection_results/roni_media_detection_assessment.md")
DEFAULT_OUTPUT_JSON = Path("data_store/detection_results/roni_media_detection_assessment.json")


@dataclass
class DetectionSummary:
    label: str
    count: int
    max_confidence: float


@dataclass
class TopDetection:
    label: str
    confidence: float
    frame_index: int | None
    bbox_xyxy: tuple[int, int, int, int]


@dataclass
class MediaAssessment:
    path: str
    kind: str
    status: str
    object_detected: bool
    uav_proxy_detected: bool
    readable: bool
    error: str | None
    width: int | None
    height: int | None
    fps: float | None
    frame_count: int | None
    duration_seconds: float | None
    sampled_frames: int
    frames_with_objects: int
    frames_with_uav_proxy: int
    labels: list[DetectionSummary]
    top_detections: list[TopDetection]
    annotated_output: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Categorize media by YOLO detections.")
    parser.add_argument("media_dir", type=Path, help="Directory containing videos/images to assess.")
    parser.add_argument(
        "--model",
        default="data_store/models/base/yolov8n.pt",
        help="Ultralytics YOLO model path/name.",
    )
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("data_store/detection_results"),
        help="Parent folder for timestamped runs.",
    )
    parser.add_argument("--run-name", help="Base name for timestamped run folder.")
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated media into category folders.")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--max-video-frames", type=int, default=30, help="Max sampled frames per video.")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="When saving annotated videos, analyze and write every Nth frame.",
    )
    parser.add_argument("--annotate-batch-size", type=int, default=8, help="YOLO batch size for annotated video output.")
    parser.add_argument("--max-width", type=int, default=1280, help="Downscale frames wider than this before YOLO.")
    parser.add_argument("--max-height", type=int, default=720, help="Downscale frames taller than this before YOLO.")
    parser.add_argument("--device", default="", help="Optional Ultralytics device, such as cpu, mps, or 0.")
    parser.add_argument(
        "--target-label",
        action="append",
        dest="target_labels",
        help="Displayed label to treat as UAV/proxy. Repeat to provide multiple labels.",
    )
    parser.add_argument(
        "--label-alias",
        action="append",
        dest="label_aliases",
        default=list(DEFAULT_LABEL_ALIASES),
        help="Map model labels before reporting/targeting, e.g. airplane=drone. Repeatable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_started_at = datetime.now().astimezone()
    run_started_perf = time.perf_counter()
    if args.frame_step < 1:
        raise SystemExit("--frame-step must be >= 1")
    if args.annotate_batch_size < 1:
        raise SystemExit("--annotate-batch-size must be >= 1")

    media_paths = list_media(args.media_dir)
    if not media_paths:
        raise SystemExit(f"No supported media files found in {args.media_dir}")

    run_dir = prepare_run_dir(args)
    model_load_started = time.perf_counter()
    model = YOLO(args.model)
    model_load_elapsed = time.perf_counter() - model_load_started
    names = normalise_names(model.names)
    label_aliases = parse_label_aliases(args.label_aliases)
    display_names = {class_id: apply_label_alias(name, label_aliases) for class_id, name in names.items()}
    requested_targets = tuple(args.target_labels or DEFAULT_TARGET_LABELS)
    requested_target_names = {label.strip().lower() for label in requested_targets if label and label.strip()}
    available_targets = {
        label.lower()
        for label in display_names.values()
        if label.lower() in requested_target_names
    }

    assessments: list[MediaAssessment] = []
    media_processing_started = time.perf_counter()
    for index, path in enumerate(media_paths, start=1):
        print(f"[{index}/{len(media_paths)}] assessing {path.name}", flush=True)
        if args.save_annotated and path.suffix.lower() in VIDEO_EXTENSIONS:
            assessment = assess_video_with_annotation(path, model, args, available_targets, display_names, run_dir)
        elif args.save_annotated:
            assessment = assess_image_with_annotation(path, model, args, available_targets, display_names, run_dir)
        elif path.suffix.lower() in VIDEO_EXTENSIONS:
            assessment = assess_video(path, model, args, available_targets, display_names)
        else:
            assessment = assess_image(path, model, args, available_targets, display_names)
        assessments.append(assessment)
    media_processing_elapsed = time.perf_counter() - media_processing_started

    run_ended_at = datetime.now().astimezone()
    run_elapsed = time.perf_counter() - run_started_perf
    run_metadata = build_run_metadata(
        args=args,
        assessments=assessments,
        requested_targets=requested_targets,
        available_targets=available_targets,
        names=names,
        display_names=display_names,
        label_aliases=label_aliases,
        run_started_at=run_started_at,
        run_ended_at=run_ended_at,
        run_elapsed_seconds=run_elapsed,
        model_load_seconds=model_load_elapsed,
        media_processing_seconds=media_processing_elapsed,
    )
    args.run_metadata = run_metadata

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    report = build_markdown_report(args, assessments, requested_targets, available_targets, names, label_aliases)
    args.output_md.write_text(report, encoding="utf-8")
    args.output_json.write_text(
        json.dumps([assessment_to_dict(item) for item in assessments], indent=2),
        encoding="utf-8",
    )
    metadata_path = write_run_metadata(args, run_metadata)

    video_counts = Counter(item.status for item in assessments if item.kind == "video")
    image_counts = Counter(item.status for item in assessments if item.kind == "image")
    print()
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {metadata_path}")
    if args.save_annotated:
        print(f"Wrote annotated media under {run_dir}")
    print(f"Elapsed: {format_elapsed(run_elapsed)}")
    print(
        "Videos: "
        f"good={video_counts.get('good', 0)}, "
        f"neutral={video_counts.get('neutral', 0)}, "
        f"bad={video_counts.get('bad', 0)}, "
        f"unreadable={video_counts.get('unreadable', 0)}"
    )
    print(
        "Images: "
        f"good={image_counts.get('good', 0)}, "
        f"neutral={image_counts.get('neutral', 0)}, "
        f"bad={image_counts.get('bad', 0)}, "
        f"unreadable={image_counts.get('unreadable', 0)}"
    )
    return 0


def prepare_run_dir(args: argparse.Namespace) -> Path | None:
    if not args.save_annotated:
        return None

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    base_name = args.run_name or f"{args.media_dir.name}_media_detection_assessment"
    run_dir = args.run_root / f"{base_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    args.output_md = run_dir / "assessment.md"
    args.output_json = run_dir / "assessment.json"
    args.run_dir = run_dir

    for folder in ("good", "neutral", "bad", "unreadable"):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)
        (run_dir / "images" / folder).mkdir(parents=True, exist_ok=True)
    return run_dir


def build_run_metadata(
    args: argparse.Namespace,
    assessments: list[MediaAssessment],
    requested_targets: Iterable[str],
    available_targets: set[str],
    names: dict[int, str],
    display_names: dict[int, str],
    label_aliases: dict[str, str],
    run_started_at: datetime,
    run_ended_at: datetime,
    run_elapsed_seconds: float,
    model_load_seconds: float,
    media_processing_seconds: float,
) -> dict[str, object]:
    videos = [item for item in assessments if item.kind == "video"]
    images = [item for item in assessments if item.kind == "image"]
    video_counts = Counter(item.status for item in videos)
    image_counts = Counter(item.status for item in images)
    total_analyzed_frames = sum(item.sampled_frames for item in assessments)
    total_object_frames = sum(item.frames_with_objects for item in videos)
    total_target_frames = sum(item.frames_with_uav_proxy for item in videos)

    return {
        "started_at": run_started_at.isoformat(),
        "ended_at": run_ended_at.isoformat(),
        "elapsed_seconds": round(run_elapsed_seconds, 3),
        "elapsed_human": format_elapsed(run_elapsed_seconds),
        "model_load_seconds": round(model_load_seconds, 3),
        "media_processing_seconds": round(media_processing_seconds, 3),
        "media_processing_human": format_elapsed(media_processing_seconds),
        "dataset": str(args.media_dir),
        "model": args.model,
        "confidence": args.conf,
        "iou": args.iou,
        "image_size": args.imgsz,
        "device": args.device or "auto",
        "save_annotated": bool(args.save_annotated),
        "frame_step": args.frame_step,
        "max_video_frames": args.max_video_frames,
        "max_width": args.max_width,
        "max_height": args.max_height,
        "requested_target_labels": list(requested_targets),
        "available_target_labels": sorted(available_targets),
        "label_aliases": label_aliases,
        "model_labels": [names[index] for index in sorted(names)],
        "display_labels": [display_names[index] for index in sorted(display_names)],
        "total_media": len(assessments),
        "total_videos": len(videos),
        "total_images": len(images),
        "total_analyzed_frames": total_analyzed_frames,
        "total_video_any_object_detected_frames": total_object_frames,
        "total_video_target_detected_frames": total_target_frames,
        "videos": dict(video_counts),
        "images": dict(image_counts),
    }


def write_run_metadata(args: argparse.Namespace, run_metadata: dict[str, object]) -> Path:
    run_dir = getattr(args, "run_dir", None)
    if run_dir:
        metadata_path = run_dir / "run_metadata.json"
    else:
        metadata_path = args.output_json.with_name(f"{args.output_json.stem}_run_metadata.json")
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    return metadata_path


def format_elapsed(seconds: float) -> str:
    whole_seconds = int(round(seconds))
    minutes, sec = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def list_media(media_dir: Path) -> list[Path]:
    suffixes = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
    return sorted(path for path in media_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def assess_video(
    path: Path,
    model: YOLO,
    args: argparse.Namespace,
    target_labels: set[str],
    names: dict[int, str],
) -> MediaAssessment:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return unreadable_assessment(path, "video", "Unable to open video")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
        raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = raw_frame_count if raw_frame_count > 0 else None
        duration = frame_count / fps if frame_count and fps else None
        frame_indices = sampled_frame_indices(frame_count, args.max_video_frames)

        frames: list[np.ndarray] = []
        readable_indices: list[int] = []
        if frame_indices:
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(resize_frame(frame, args.max_width, args.max_height))
                    readable_indices.append(frame_index)
        else:
            read_sequential_frames(cap, frames, readable_indices, args)

        result = run_detection_batch(model, frames, readable_indices, args, target_labels, names)
        return build_assessment(
            path=path,
            kind="video",
            readable=True,
            error=None,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
            sampled_frames=len(frames),
            result=result,
        )
    finally:
        cap.release()


def assess_image(
    path: Path,
    model: YOLO,
    args: argparse.Namespace,
    target_labels: set[str],
    names: dict[int, str],
) -> MediaAssessment:
    frame = cv2.imread(str(path))
    if frame is None:
        return unreadable_assessment(path, "image", "Unable to read image")

    frame = resize_frame(frame, args.max_width, args.max_height)
    result = run_detection_batch(model, [frame], [0], args, target_labels, names)
    height, width = frame.shape[:2]
    return build_assessment(
        path=path,
        kind="image",
        readable=True,
        error=None,
        width=width,
        height=height,
        fps=None,
        frame_count=None,
        duration_seconds=None,
        sampled_frames=1,
        result=result,
    )


def assess_video_with_annotation(
    path: Path,
    model: YOLO,
    args: argparse.Namespace,
    target_labels: set[str],
    names: dict[int, str],
    run_dir: Path,
) -> MediaAssessment:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return unreadable_assessment(path, "video", "Unable to open video")

    tmp_dir = run_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_output = unique_output_path(tmp_dir, f"{path.stem}.mp4")
    writer: cv2.VideoWriter | None = None
    result = new_detection_result()
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []
    sampled_frames = 0
    read_frames = 0

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
        raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = raw_frame_count if raw_frame_count > 0 else None
        duration = frame_count / fps if frame_count and fps else None
        output_fps = max((fps or 20.0) / args.frame_step, 1.0)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if read_frames % args.frame_step == 0:
                batch_frames.append(resize_frame(frame, args.max_width, args.max_height))
                batch_indices.append(read_frames)
                sampled_frames += 1

            read_frames += 1
            if len(batch_frames) >= args.annotate_batch_size:
                writer = process_annotated_batch(
                    model,
                    batch_frames,
                    batch_indices,
                    args,
                    target_labels,
                    names,
                    tmp_output,
                    output_fps,
                    writer,
                    result,
                )
                batch_frames = []
                batch_indices = []

        if batch_frames:
            writer = process_annotated_batch(
                model,
                batch_frames,
                batch_indices,
                args,
                target_labels,
                names,
                tmp_output,
                output_fps,
                writer,
                result,
            )

        if frame_count is None and read_frames > 0:
            frame_count = read_frames
            duration = frame_count / fps if fps else None
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if sampled_frames == 0 or not tmp_output.exists():
        return unreadable_assessment(path, "video", "No readable frames")

    assessment = build_assessment(
        path=path,
        kind="video",
        readable=True,
        error=None,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
        sampled_frames=sampled_frames,
        result=result,
    )
    final_output = unique_output_path(annotated_category_dir(run_dir, assessment), tmp_output.name)
    shutil.move(str(tmp_output), str(final_output))
    assessment.annotated_output = str(final_output)
    return assessment


def process_annotated_batch(
    model: YOLO,
    frames: list[np.ndarray],
    frame_indices: list[int],
    args: argparse.Namespace,
    target_labels: set[str],
    names: dict[int, str],
    output_path: Path,
    output_fps: float,
    writer: cv2.VideoWriter | None,
    detection_result: dict[str, object],
) -> cv2.VideoWriter:
    results = model.predict(frames, **predict_args_from_args(args))
    for frame_index, yolo_result in zip(frame_indices, results):
        add_yolo_result_detections(yolo_result, frame_index, target_labels, names, detection_result)
        annotated = plot_yolo_result(yolo_result, names)
        if writer is None:
            writer = create_video_writer(output_path, annotated, output_fps)
        writer.write(annotated)

    if writer is None:
        raise RuntimeError(f"Unable to create annotated output for {output_path}")
    return writer


def assess_image_with_annotation(
    path: Path,
    model: YOLO,
    args: argparse.Namespace,
    target_labels: set[str],
    names: dict[int, str],
    run_dir: Path,
) -> MediaAssessment:
    frame = cv2.imread(str(path))
    if frame is None:
        return unreadable_assessment(path, "image", "Unable to read image")

    frame = resize_frame(frame, args.max_width, args.max_height)
    result = new_detection_result()
    yolo_results = model.predict([frame], **predict_args_from_args(args))
    if yolo_results:
        add_yolo_result_detections(yolo_results[0], 0, target_labels, names, result)
        annotated = plot_yolo_result(yolo_results[0], names)
    else:
        annotated = frame

    height, width = frame.shape[:2]
    assessment = build_assessment(
        path=path,
        kind="image",
        readable=True,
        error=None,
        width=width,
        height=height,
        fps=None,
        frame_count=None,
        duration_seconds=None,
        sampled_frames=1,
        result=result,
    )
    output_path = unique_output_path(annotated_category_dir(run_dir, assessment), path.name)
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Unable to write annotated image: {output_path}")
    assessment.annotated_output = str(output_path)
    return assessment


def unreadable_assessment(path: Path, kind: str, error: str) -> MediaAssessment:
    return MediaAssessment(
        path=str(path),
        kind=kind,
        status="unreadable",
        object_detected=False,
        uav_proxy_detected=False,
        readable=False,
        error=error,
        width=None,
        height=None,
        fps=None,
        frame_count=None,
        duration_seconds=None,
        sampled_frames=0,
        frames_with_objects=0,
        frames_with_uav_proxy=0,
        labels=[],
        top_detections=[],
        annotated_output=None,
    )


def sampled_frame_indices(frame_count: int | None, max_frames: int) -> list[int]:
    if not frame_count:
        return []
    sample_count = min(max_frames, frame_count)
    if sample_count <= 0:
        return []
    return sorted({int(value) for value in np.linspace(0, frame_count - 1, sample_count)})


def read_sequential_frames(
    cap: cv2.VideoCapture,
    frames: list[np.ndarray],
    readable_indices: list[int],
    args: argparse.Namespace,
) -> None:
    frame_index = 0
    while len(frames) < args.max_video_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            return
        frames.append(resize_frame(frame, args.max_width, args.max_height))
        readable_indices.append(frame_index)
        frame_index += 1


def resize_frame(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    if max_width <= 0 or max_height <= 0:
        return frame
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def predict_args_from_args(args: argparse.Namespace) -> dict[str, object]:
    predict_args: dict[str, object] = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        predict_args["device"] = args.device
    return predict_args


def plot_yolo_result(yolo_result: object, names: dict[int, str]) -> np.ndarray:
    original_names = getattr(yolo_result, "names", None)
    try:
        setattr(yolo_result, "names", names)
        return yolo_result.plot()
    finally:
        if original_names is not None:
            setattr(yolo_result, "names", original_names)


def parse_label_aliases(values: Iterable[str] | None) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in values or ():
        if "=" not in value:
            raise SystemExit(f"--label-alias must use source=target form: {value}")
        source, target = value.split("=", 1)
        source = source.strip().lower()
        target = target.strip()
        if not source or not target:
            raise SystemExit(f"--label-alias must use non-empty source=target values: {value}")
        aliases[source] = target
    return aliases


def apply_label_alias(label: str, aliases: dict[str, str]) -> str:
    return aliases.get(label.strip().lower(), label)


def new_detection_result() -> dict[str, object]:
    return {
        "label_counts": Counter(),
        "label_confidences": defaultdict(list),
        "top_detections": [],
        "frames_with_objects": 0,
        "frames_with_uav_proxy": 0,
    }


def add_yolo_result_detections(
    yolo_result: object,
    frame_index: int,
    target_labels: set[str],
    names: dict[int, str],
    detection_result: dict[str, object],
) -> None:
    boxes = getattr(yolo_result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return

    label_counts: Counter[str] = detection_result["label_counts"]  # type: ignore[assignment]
    label_confidences: defaultdict[str, list[float]] = detection_result["label_confidences"]  # type: ignore[assignment]
    top_detections: list[TopDetection] = detection_result["top_detections"]  # type: ignore[assignment]
    frame_has_uav_proxy = False

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    for bbox, confidence, class_id in zip(xyxy, confidences, classes):
        label = names.get(int(class_id), str(class_id))
        label_counts[label] += 1
        label_confidences[label].append(float(confidence))
        top_detections.append(
            TopDetection(
                label=label,
                confidence=round(float(confidence), 4),
                frame_index=int(frame_index),
                bbox_xyxy=tuple(int(value) for value in bbox),
            )
        )
        if label.lower() in target_labels:
            frame_has_uav_proxy = True

    detection_result["frames_with_objects"] = int(detection_result["frames_with_objects"]) + 1
    if frame_has_uav_proxy:
        detection_result["frames_with_uav_proxy"] = int(detection_result["frames_with_uav_proxy"]) + 1


def create_video_writer(output_path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create annotated video: {output_path}")
    return writer


def annotated_category_dir(run_dir: Path, assessment: MediaAssessment) -> Path:
    if assessment.kind == "image":
        return run_dir / "images" / assessment.status
    return run_dir / assessment.status


def unique_output_path(folder: Path, filename: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    desired = folder / filename
    if not desired.exists():
        return desired

    stem = desired.stem
    suffix = desired.suffix
    counter = 1
    while True:
        candidate = folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def run_detection_batch(
    model: YOLO,
    frames: list[np.ndarray],
    frame_indices: list[int],
    args: argparse.Namespace,
    target_labels: set[str],
    names: dict[int, str],
) -> dict[str, object]:
    label_counts: Counter[str] = Counter()
    label_confidences: defaultdict[str, list[float]] = defaultdict(list)
    top_detections: list[TopDetection] = []
    frames_with_objects = 0
    frames_with_uav_proxy = 0

    if not frames:
        return {
            "label_counts": label_counts,
            "label_confidences": label_confidences,
            "top_detections": top_detections,
            "frames_with_objects": frames_with_objects,
            "frames_with_uav_proxy": frames_with_uav_proxy,
        }

    predict_args = {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz, "verbose": False}
    if args.device:
        predict_args["device"] = args.device

    results = model.predict(frames, **predict_args)
    for frame_index, result in zip(frame_indices, results):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        frame_has_objects = False
        frame_has_uav_proxy = False
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for bbox, confidence, class_id in zip(xyxy, confidences, classes):
            label = names.get(int(class_id), str(class_id))
            label_counts[label] += 1
            label_confidences[label].append(float(confidence))
            top_detections.append(
                TopDetection(
                    label=label,
                    confidence=round(float(confidence), 4),
                    frame_index=int(frame_index),
                    bbox_xyxy=tuple(int(value) for value in bbox),
                )
            )
            frame_has_objects = True
            if label.lower() in target_labels:
                frame_has_uav_proxy = True

        frames_with_objects += int(frame_has_objects)
        frames_with_uav_proxy += int(frame_has_uav_proxy)

    return {
        "label_counts": label_counts,
        "label_confidences": label_confidences,
        "top_detections": sorted(top_detections, key=lambda item: item.confidence, reverse=True)[:20],
        "frames_with_objects": frames_with_objects,
        "frames_with_uav_proxy": frames_with_uav_proxy,
    }


def build_assessment(
    path: Path,
    kind: str,
    readable: bool,
    error: str | None,
    width: int | None,
    height: int | None,
    fps: float | None,
    frame_count: int | None,
    duration_seconds: float | None,
    sampled_frames: int,
    result: dict[str, object],
) -> MediaAssessment:
    label_counts: Counter[str] = result["label_counts"]  # type: ignore[assignment]
    label_confidences: defaultdict[str, list[float]] = result["label_confidences"]  # type: ignore[assignment]
    object_detected = bool(label_counts)
    uav_proxy_detected = bool(result["frames_with_uav_proxy"])
    if uav_proxy_detected:
        status = "good"
    elif object_detected:
        status = "neutral"
    else:
        status = "bad"

    labels = [
        DetectionSummary(label=label, count=count, max_confidence=round(max(label_confidences[label]), 4))
        for label, count in label_counts.most_common()
    ]
    top_detections: list[TopDetection] = result["top_detections"]  # type: ignore[assignment]

    return MediaAssessment(
        path=str(path),
        kind=kind,
        status=status,
        object_detected=object_detected,
        uav_proxy_detected=uav_proxy_detected,
        readable=readable,
        error=error,
        width=width,
        height=height,
        fps=round(fps, 3) if fps else None,
        frame_count=frame_count,
        duration_seconds=round(duration_seconds, 2) if duration_seconds else None,
        sampled_frames=sampled_frames,
        frames_with_objects=int(result["frames_with_objects"]),
        frames_with_uav_proxy=int(result["frames_with_uav_proxy"]),
        labels=labels,
        top_detections=sorted(top_detections, key=lambda item: item.confidence, reverse=True)[:20],
        annotated_output=None,
    )


def build_markdown_report(
    args: argparse.Namespace,
    assessments: list[MediaAssessment],
    requested_targets: Iterable[str],
    available_targets: set[str],
    names: dict[int, str],
    label_aliases: dict[str, str],
) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    videos = [item for item in assessments if item.kind == "video"]
    images = [item for item in assessments if item.kind == "image"]
    video_counts = Counter(item.status for item in videos)
    image_counts = Counter(item.status for item in images)
    model_labels = ", ".join(names[index] for index in sorted(names))
    alias_text = ", ".join(f"{source}->{target}" for source, target in sorted(label_aliases.items())) or "none"
    run_dir = getattr(args, "run_dir", None)
    run_metadata = getattr(args, "run_metadata", None)
    video_analysis = (
        f"every `{args.frame_step}` frame(s), with annotated video output"
        if args.save_annotated
        else f"up to `{args.max_video_frames}` evenly spaced frames per file"
    )

    lines = [
        "# Initial Media Detection Assessment",
        "",
        f"- Generated: {timestamp}",
        f"- Dataset: `{args.media_dir}`",
        *( [f"- Output run folder: `{run_dir}`"] if run_dir else [] ),
        f"- Model: `{args.model}`",
        f"- Confidence / IoU / image size: `{args.conf}` / `{args.iou}` / `{args.imgsz}`",
        f"- Video analysis: {video_analysis}",
        f"- Frame resize before inference: max `{args.max_width}x{args.max_height}`",
        f"- Label aliases: `{alias_text}`",
        "",
        "## Classification Rule",
        "",
        "- Good: any configured UAV/proxy label detected.",
        "- Neutral: at least one object detected, but no configured UAV/proxy label detected.",
        "- Bad: no objects detected in the analyzed frames/image.",
        "",
        "## Customer-Facing Media Definitions",
        "",
        (
            "- Good media: media where the detection model found at least one configured UAV-like target label. "
            f"In this assessment, the available UAV-like proxy labels are `{', '.join(sorted(available_targets)) or 'none'}`."
        ),
        (
            "- Neutral media: media where the detection model found one or more objects, but none of the configured "
            "UAV-like target labels were detected."
        ),
        "- Bad media: media where the detection model did not find any object above the configured confidence threshold.",
        "- Unreadable media: media that could not be opened or decoded by the assessment pipeline.",
        "",
        "## Important Model Caveat",
        "",
        (
            "For general COCO models, the report can consolidate proxy labels before scoring. "
            f"In this run, the available displayed UAV-like target labels are `{', '.join(sorted(available_targets)) or 'none'}`."
        ),
        f"Requested UAV/proxy labels: `{', '.join(requested_targets)}`.",
        f"Model label set: `{model_labels}`.",
        "",
        "## Summary",
        "",
        "| Media type | Total | Good | Neutral | Bad | Unreadable |",
        "|---|---:|---:|---:|---:|---:|",
        summary_row("Videos", videos, video_counts),
        summary_row("Images", images, image_counts),
        "",
    ]
    if run_metadata:
        lines.extend(
            [
                "## Timing",
                "",
                f"- Total elapsed: `{run_metadata['elapsed_human']}` (`{run_metadata['elapsed_seconds']}` seconds)",
                f"- Model load: `{run_metadata['model_load_seconds']}` seconds",
                (
                    "- Media processing: "
                    f"`{run_metadata['media_processing_human']}` (`{run_metadata['media_processing_seconds']}` seconds)"
                ),
                f"- Total analyzed frames/items: `{run_metadata['total_analyzed_frames']}`",
                "",
            ]
        )

    lines.extend(category_section("Good Movies", [item for item in videos if item.status == "good"]))
    lines.extend(category_section("Neutral Movies", [item for item in videos if item.status == "neutral"]))
    lines.extend(category_section("Bad Movies", [item for item in videos if item.status == "bad"]))
    lines.extend(category_section("Unreadable Movies", [item for item in videos if item.status == "unreadable"]))
    lines.extend(category_section("Images", images))

    notes = ["## Notes", ""]
    if args.save_annotated:
        notes.append(
            f"- Videos were analyzed at the configured frame step; this run used every `{args.frame_step}` frame(s)."
        )
    else:
        notes.append("- This is a sampled first-pass assessment, not exhaustive frame-by-frame review.")
    notes.extend(
        [
            "- A drone-specific model is needed before treating `good` as confirmed UAV detection.",
            "- Files marked `bad` may still contain objects below the confidence threshold or outside the model label set.",
            "",
        ]
    )
    lines.extend(notes)
    return "\n".join(lines)


def summary_row(label: str, items: list[MediaAssessment], counts: Counter[str]) -> str:
    return (
        f"| {label} | {len(items)} | {counts.get('good', 0)} | {counts.get('neutral', 0)} | "
        f"{counts.get('bad', 0)} | {counts.get('unreadable', 0)} |"
    )


def category_section(title: str, items: list[MediaAssessment]) -> list[str]:
    lines = [f"## {title}", ""]
    if not items:
        lines.extend(["None.", ""])
        return lines

    lines.extend(
        [
            "| File | Status | Duration | Analyzed frames | Any-object detected frames | Target-detected frames | Labels | Annotated output |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for item in items:
        labels = ", ".join(f"{label.label} x{label.count} ({label.max_confidence:.2f})" for label in item.labels)
        output = f"`{item.annotated_output}`" if item.annotated_output else "-"
        lines.append(
            "| "
            f"`{Path(item.path).name}` | {item.status} | {format_duration(item.duration_seconds)} | "
            f"{item.sampled_frames} | {item.frames_with_objects} | {item.frames_with_uav_proxy} | "
            f"{labels or '-'} | {output} |"
        )
    lines.append("")
    return lines


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds:.1f}s"


def normalise_names(names: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def assessment_to_dict(assessment: MediaAssessment) -> dict[str, object]:
    data = asdict(assessment)
    return data


if __name__ == "__main__":
    raise SystemExit(main())
