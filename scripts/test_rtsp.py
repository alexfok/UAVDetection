from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.sources import camera_summary, open_source_capture, resolve_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick RTSP connectivity/FPS test")
    parser.add_argument("source", nargs="?", help="RTSP URL, local video/image path, webcam index, or camera:<id>")
    parser.add_argument("--camera", help="Use a named camera from the camera registry")
    parser.add_argument("--cameras", default="data_store/system_config/cameras.yaml", help="Path to camera registry YAML")
    parser.add_argument("--list-cameras", action="store_true", help="List configured cameras and exit")
    parser.add_argument("--seconds", type=float, default=10.0, help="How long to sample frames")
    args = parser.parse_args()

    if args.list_cameras:
        for line in camera_summary(args.cameras):
            print(line)
        return 0

    source_value = f"camera:{args.camera}" if args.camera else args.source
    if not source_value:
        parser.error("source or --camera is required unless --list-cameras is used")

    source = resolve_source(source_value, args.cameras)
    cap = open_source_capture(source)
    if not cap.isOpened():
        print(f"Unable to open source: {source.label}")
        return 2

    start = time.monotonic()
    frames = 0
    last_shape = None

    while time.monotonic() - start < args.seconds:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Frame read failed")
            break
        frames += 1
        last_shape = frame.shape

    cap.release()
    elapsed = max(time.monotonic() - start, 0.001)
    print(f"Source: {source.label}")
    print(f"Read {frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} FPS)")
    if last_shape:
        print(f"Last frame shape: {last_shape}")
    return 0 if frames > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
