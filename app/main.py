from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from app.alert import AlertManager
from app.config import AppConfig, load_config, parse_video_source
from app.detector import DroneDetector
from app.tracker import SimpleTracker
from app.ui import OpenCVUI

LOGGER = logging.getLogger(__name__)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    configure_logging(config.logging.level)

    source = parse_video_source(config.video.source)
    source_label = str(config.video.source)

    detector = DroneDetector(config.detector)
    tracker = SimpleTracker(config.tracker, config.alert.window_seconds)
    alert_manager = AlertManager(config.alert)
    ui = OpenCVUI(config.ui)

    try:
        cap = open_capture(source, config)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 2

    writer = None
    frame_index = 0
    fps_meter = FPSMeter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if not is_reconnectable_source(source):
                    LOGGER.info("Reached end of local video source: %s", source)
                    return 0

                cap = reconnect_capture(cap, source, config)
                if cap is None:
                    LOGGER.error("Unable to read video source after reconnect attempts.")
                    return 2
                continue

            frame_index += 1
            if config.video.frame_skip > 0 and frame_index % (config.video.frame_skip + 1) != 1:
                continue

            frame = resize_frame(frame, config.video.resize_width, config.video.resize_height)
            now = time.monotonic()

            detections = detector.detect(frame)
            tracks = tracker.update(detections, now)
            alert = alert_manager.update(tracks, now)
            fps = fps_meter.update()

            annotated = ui.draw(frame, tracks, alert, fps, source_label)

            if config.ui.save_output:
                writer = ensure_writer(writer, config.ui.output_path, annotated, fps)
                writer.write(annotated)

            result = ui.show(annotated)
            if result.should_quit:
                LOGGER.info("Quit requested by user.")
                return 0
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        return 0
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        ui.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast Drone Detection PoC")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file")
    parser.add_argument("--source", help="Override video source: mp4 path, RTSP URL, or webcam index")
    parser.add_argument("--model", help="Override YOLO model path, e.g. yolov8n.pt")
    parser.add_argument("--no-window", action="store_true", help="Run without opening the OpenCV window")
    parser.add_argument("--save-output", action="store_true", help="Save annotated video output")
    parser.add_argument("--log-level", help="Override log level")
    return parser.parse_args()


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    if args.source:
        config.video.source = args.source
    if args.model:
        config.detector.model_path = args.model
    if args.no_window:
        config.ui.show_window = False
    if args.save_output:
        config.ui.save_output = True
    if args.log_level:
        config.logging.level = args.log_level


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def open_capture(source: str | int, config: AppConfig) -> cv2.VideoCapture:
    LOGGER.info("Opening video source: %s", source)
    cap = cv2.VideoCapture(source)
    if config.video.buffer_size > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, config.video.buffer_size)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    return cap


def reconnect_capture(
    cap: cv2.VideoCapture,
    source: str | int,
    config: AppConfig,
) -> cv2.VideoCapture | None:
    cap.release()
    for attempt in range(1, config.video.reconnect_attempts + 1):
        LOGGER.warning("Lost video source; reconnect attempt %s/%s", attempt, config.video.reconnect_attempts)
        time.sleep(config.video.reconnect_delay_sec)
        candidate = cv2.VideoCapture(source)
        if config.video.buffer_size > 0:
            candidate.set(cv2.CAP_PROP_BUFFERSIZE, config.video.buffer_size)
        if candidate.isOpened():
            return candidate
        candidate.release()
    return None


def is_reconnectable_source(source: str | int) -> bool:
    if isinstance(source, int):
        return True

    source_lower = source.lower()
    return source_lower.startswith(("rtsp://", "rtmp://", "http://", "https://"))


def resize_frame(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    if max_width <= 0 or max_height <= 0:
        return frame

    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame

    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def ensure_writer(
    writer: cv2.VideoWriter | None,
    output_path: str,
    frame: np.ndarray,
    fps: float,
) -> cv2.VideoWriter:
    if writer is not None:
        return writer

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    safe_fps = fps if fps > 1 else 20.0
    video_writer = cv2.VideoWriter(str(path), fourcc, safe_fps, (width, height))
    if not video_writer.isOpened():
        raise RuntimeError(f"Unable to create output video: {path}")
    return video_writer


class FPSMeter:
    def __init__(self) -> None:
        self._last = time.monotonic()
        self._fps = 0.0

    def update(self) -> float:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        if elapsed <= 0:
            return self._fps

        instant = 1.0 / elapsed
        self._fps = instant if self._fps == 0 else (self._fps * 0.85) + (instant * 0.15)
        return self._fps


if __name__ == "__main__":
    raise SystemExit(main())
