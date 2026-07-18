from __future__ import annotations

import csv
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from scripts.annotation_server import (
    AsyncDetectionWorker,
    advance_video_capture_to_realtime,
    annotation_split_stats,
    parse_bool,
    parse_float,
    parse_int,
    query_value,
    raw_data_stats,
    read_recent_live_events,
    remove_live_event_rows,
    safe_name,
    upsert_manifest,
    yolo_label_text,
)


class AnnotationServerHelperTests(unittest.TestCase):
    def test_video_capture_drops_frames_to_match_wall_clock(self) -> None:
        class FakeCapture:
            def __init__(self) -> None:
                self.grabbed = 0

            def grab(self) -> bool:
                self.grabbed += 1
                return True

        capture = FakeCapture()
        consumed, available = advance_video_capture_to_realtime(
            capture,
            frames_consumed=0,
            source_fps=16.0,
            playback_started_at=100.0,
            now=100.5,
        )
        self.assertTrue(available)
        self.assertEqual(consumed, 8)
        self.assertEqual(capture.grabbed, 8)

    def test_async_detection_does_not_block_or_queue_stale_frames(self) -> None:
        class BlockingDetector:
            def __init__(self) -> None:
                self.started = threading.Event()
                self.release = threading.Event()

            def detect(self, frame):
                self.started.set()
                self.release.wait(timeout=1.0)
                return [frame]

        detector = BlockingDetector()
        worker = AsyncDetectionWorker(detector)
        try:
            self.assertTrue(worker.submit(["first"], 10))
            self.assertTrue(detector.started.wait(timeout=1.0))
            self.assertFalse(worker.submit(["stale"], 11))
            self.assertIsNone(worker.poll())
            detector.release.set()
            deadline = time.monotonic() + 1.0
            result = None
            while result is None and time.monotonic() < deadline:
                result = worker.poll()
                time.sleep(0.005)
            self.assertIsNotNone(result)
            self.assertEqual(result.frame_index, 10)
            self.assertEqual(result.detections, [["first"]])
        finally:
            detector.release.set()
            worker.close()

    def test_safe_parsers_and_label_text(self) -> None:
        self.assertEqual(safe_name(" Field Test: #1 "), "Field_Test_1")
        self.assertEqual(query_value({"a": [""]}, "a", "fallback"), "fallback")
        self.assertEqual(parse_int("12.9"), 12)
        self.assertEqual(parse_int("bad"), 0)
        self.assertEqual(parse_float("bad", 1.5), 1.5)
        self.assertTrue(parse_bool("yes"))
        self.assertFalse(parse_bool("no"))
        self.assertEqual(yolo_label_text([(10, 20, 30, 60)], 100, 100), "0 0.200000 0.400000 0.200000 0.400000\n")

    def test_raw_and_annotation_stats_are_folder_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            (raw / "frame.png").write_bytes(b"not a real image")
            (raw / "clip.mp4").write_bytes(b"not a real video")
            (raw / "notes.txt").write_text("ignore", encoding="utf-8")
            self.assertEqual(raw_data_stats(raw), {"exists": True, "files": 2, "videos": 1, "images": 1})

            project = root / "dataset"
            (project / "images" / "train").mkdir(parents=True)
            (project / "labels" / "train").mkdir(parents=True)
            (project / "images" / "train" / "a.png").write_bytes(b"image")
            (project / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (project / "labels" / "train" / "b.txt").write_text("", encoding="utf-8")
            stats = annotation_split_stats(project, "train")
            self.assertEqual(stats["total"], 2)
            self.assertEqual(stats["positive"], 1)
            self.assertEqual(stats["negative"], 1)
            self.assertEqual(stats["boxes"], 1)

    def test_upsert_manifest_reports_when_row_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.csv"
            created = upsert_manifest(
                manifest_path,
                {
                    "image_id": "frame_1",
                    "split": "train",
                    "image_path": "images/train/frame_1.jpg",
                    "label_path": "labels/train/frame_1.txt",
                    "box_count": "1",
                },
            )
            replaced = upsert_manifest(
                manifest_path,
                {
                    "image_id": "frame_1",
                    "split": "train",
                    "image_path": "images/train/frame_1.jpg",
                    "label_path": "labels/train/frame_1.txt",
                    "box_count": "0",
                },
            )

            self.assertFalse(created)
            self.assertTrue(replaced)
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["image_id"], "frame_1")
            self.assertEqual(rows[0]["box_count"], "0")

    def test_live_event_listing_and_removal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2026-06-20"
            frame_dir = day / "frames" / "s1"
            frame_dir.mkdir(parents=True)
            frame = frame_dir / "hit.jpg"
            frame.write_bytes(b"jpeg")
            event_path = day / "events.jsonl"
            rows = [
                {"event_type": "start", "session_id": "s1"},
                {"event_type": "drone_detected", "session_id": "s1", "image_path": str(frame)},
                {"event_type": "stop", "session_id": "s1"},
            ]
            event_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            recent = read_recent_live_events(root, limit=2)
            self.assertEqual([event["event_type"] for event in recent], ["stop", "drone_detected"])
            self.assertEqual(recent[0]["event_id"], "2026-06-20:3")

            result = remove_live_event_rows(root, ["2026-06-20:2", "bad"])
            self.assertEqual(result["removed"], ["2026-06-20:2"])
            self.assertFalse(frame.exists())
            self.assertEqual(len(result["failed"]), 1)
            remaining = event_path.read_text(encoding="utf-8")
            self.assertNotIn("drone_detected", remaining)


if __name__ == "__main__":
    unittest.main()
