from __future__ import annotations

import argparse
import time

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick RTSP connectivity/FPS test")
    parser.add_argument("source", help="RTSP URL, local video path, or webcam index")
    parser.add_argument("--seconds", type=float, default=10.0, help="How long to sample frames")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Unable to open source: {args.source}")
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
    print(f"Read {frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} FPS)")
    if last_shape:
        print(f"Last frame shape: {last_shape}")
    return 0 if frames > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

