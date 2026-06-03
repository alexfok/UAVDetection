from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.sources import open_source_capture, resolve_source


DEFAULT_CAMERA_IDS = ["poe_194", "poe_196"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile live detection pipeline timing.")
    parser.add_argument("--camera", action="append", dest="camera_ids", help="Named camera id from cameras.yaml")
    parser.add_argument("--source", action="append", dest="sources", help="Raw source path/url/index")
    parser.add_argument("--cameras", dest="cameras_file", default="data_store/system_config/cameras.yaml", help="Camera registry YAML")
    parser.add_argument("--model", default="data_store/models/trained/yolov8n_drone_best.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="", help="Ultralytics device value, e.g. cpu, mps, 0/cuda:0. Empty means auto.")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--concurrent", action="store_true", help="Profile selected sources concurrently")
    return parser.parse_args()


def source_specs(args: argparse.Namespace):
    specs = []
    for camera_id in args.camera_ids or []:
        specs.append((camera_id, resolve_source(f"camera:{camera_id}", args.cameras_file)))
    for raw_source in args.sources or []:
        specs.append((raw_source, resolve_source(raw_source, args.cameras_file)))
    if not specs:
        for camera_id in DEFAULT_CAMERA_IDS:
            specs.append((camera_id, resolve_source(f"camera:{camera_id}", args.cameras_file)))
    return specs


def main() -> int:
    args = parse_args()
    specs = source_specs(args)
    contexts = []
    for source_id, source in specs:
        cap = open_source_capture(source)
        if not cap.isOpened():
            print(f"{source_id}: unable to open {source.label}")
            continue
        contexts.append((source_id, source, cap))

    if not contexts:
        print("No sources opened.")
        return 2

    import os
    import statistics
    import tempfile
    import threading

    temp_root = Path(tempfile.gettempdir())
    os.environ.setdefault("YOLO_CONFIG_DIR", str(temp_root / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(temp_root / "matplotlib"))

    import cv2

    from app.alert import AlertManager
    from app.config import AlertConfig, DetectorConfig, TrackerConfig, UIConfig
    from app.detector import DroneDetector
    from app.tracker import SimpleTracker
    from app.ui import OpenCVUI

    def percentile(values: list[float], percent: int) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        return values[min(len(values) - 1, round((percent / 100) * (len(values) - 1)))]

    def summary(values: list[float]) -> str:
        if not values:
            return "n/a"
        return (
            f"avg={statistics.mean(values):.1f}ms "
            f"p50={statistics.median(values):.1f}ms "
            f"p90={percentile(values, 90):.1f}ms"
        )

    class LockedDetector:
        def __init__(self) -> None:
            started = time.perf_counter()
            self.detector = DroneDetector(
                DetectorConfig(
                    model_path=args.model,
                    confidence_threshold=args.conf,
                    image_size=args.imgsz,
                    device=args.device,
                )
            )
            self.load_ms = (time.perf_counter() - started) * 1000
            self.lock = threading.Lock()

        def detect(self, frame):
            wait_started = time.perf_counter()
            with self.lock:
                wait_ms = (time.perf_counter() - wait_started) * 1000
                detect_started = time.perf_counter()
                detections = self.detector.detect(frame)
                detect_ms = (time.perf_counter() - detect_started) * 1000
                return detections, wait_ms, detect_ms

    detector = LockedDetector()
    print(f"model_load_ms={detector.load_ms:.1f}")

    def resize_frame(frame):
        height, width = frame.shape[:2]
        scale = 1.0
        if args.max_width > 0 and width > args.max_width:
            scale = min(scale, args.max_width / width)
        if args.max_height > 0 and height > args.max_height:
            scale = min(scale, args.max_height / height)
        if scale >= 1.0:
            return frame
        return cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)

    def processing_context():
        return (
            SimpleTracker(TrackerConfig(), AlertConfig().window_seconds),
            AlertManager(AlertConfig(confidence_threshold=args.conf)),
            OpenCVUI(UIConfig(show_window=False, draw_all_tracks=True, draw_status_bar=False)),
        )

    def process_frame(source, cap, tracker, alert_manager, ui):
        total_started = time.perf_counter()
        read_started = time.perf_counter()
        ok, frame = cap.read()
        read_done = time.perf_counter()
        if not ok or frame is None:
            return None
        source_shape = frame.shape[:2]
        frame = resize_frame(frame)
        resize_done = time.perf_counter()
        detections, wait_ms, detect_ms = detector.detect(frame)
        detect_done = time.perf_counter()
        tracks = tracker.update(detections, time.monotonic())
        alert = alert_manager.update(tracks, time.monotonic())
        annotated = ui.draw(frame, tracks, alert, 0.0, source.label)
        draw_done = time.perf_counter()
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        encode_done = time.perf_counter()
        if not ok:
            return None
        return {
            "source_shape": source_shape,
            "out_shape": annotated.shape[:2],
            "read": (read_done - read_started) * 1000,
            "resize": (resize_done - read_done) * 1000,
            "lock_wait": wait_ms,
            "detect": detect_ms,
            "track_draw": (draw_done - detect_done) * 1000,
            "jpeg": (encode_done - draw_done) * 1000,
            "total": (encode_done - total_started) * 1000,
            "jpeg_kb": len(encoded) / 1024,
        }

    def run_source(source_id, source, cap):
        tracker, alert_manager, ui = processing_context()
        timings: dict[str, list[float]] = {
            key: [] for key in ["read", "resize", "lock_wait", "detect", "track_draw", "jpeg", "total", "jpeg_kb"]
        }
        source_shape = None
        out_shape = None
        processed = 0
        started = time.perf_counter()
        while processed < args.samples + args.warmup:
            result = process_frame(source, cap, tracker, alert_manager, ui)
            if result is None:
                continue
            processed += 1
            if source_shape is None:
                source_shape = result["source_shape"]
                out_shape = result["out_shape"]
            if processed <= args.warmup:
                continue
            for key in timings:
                timings[key].append(float(result[key]))
        elapsed = time.perf_counter() - started
        return source_id, source.label, source_shape, out_shape, elapsed, timings

    results = []
    try:
        if args.concurrent and len(contexts) > 1:
            lock = threading.Lock()

            def worker(context):
                result = run_source(*context)
                with lock:
                    results.append(result)

            threads = [threading.Thread(target=worker, args=(context,)) for context in contexts]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            for context in contexts:
                results.append(run_source(*context))
    finally:
        for _source_id, _source, cap in contexts:
            cap.release()

    total_frames = 0
    max_elapsed = 0.0
    for source_id, label, source_shape, out_shape, elapsed, timings in results:
        total_frames += len(timings["total"])
        max_elapsed = max(max_elapsed, elapsed)
        print(f"\n{source_id}: {label}")
        print(f"  source_shape={source_shape} output_shape={out_shape} samples={len(timings['total'])} elapsed={elapsed:.2f}s")
        for key in ["read", "resize", "lock_wait", "detect", "track_draw", "jpeg", "total"]:
            print(f"  {key}: {summary(timings[key])}")
        print(f"  jpeg_size: {summary(timings['jpeg_kb'])} KB")
        print(f"  processed_fps={len(timings['total']) / elapsed:.2f}")
    if args.concurrent and max_elapsed:
        print(f"\naggregate_processed_fps={total_frames / max_elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
