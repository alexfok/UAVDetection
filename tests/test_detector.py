from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from app.config import DetectorConfig
from app.detector import DroneDetector
from app.sources import open_source_capture, resolve_source

from tests.test_sources import FIXTURE_MEDIA


class DetectorTests(unittest.TestCase):
    def test_detector_uses_runtime_parameters_aliases_and_known_frame(self) -> None:
        frame = read_fixture_frame("known_frame.png")
        fake_model = FakeModel(
            [
                FakeResult(
                    FakeBoxes(
                        xyxy=np.array([[4.2, 5.1, 20.9, 18.4]]),
                        conf=np.array([0.81]),
                        cls=np.array([2]),
                    )
                )
            ]
        )
        config = DetectorConfig(
            model_path="fake.pt",
            confidence_threshold=0.3,
            iou_threshold=0.4,
            image_size=960,
            device="cpu",
            target_classes=["drone"],
            label_aliases={"kite": "drone"},
        )
        with patch("app.detector.load_yolo_model", return_value=fake_model):
            detector = DroneDetector(config)
            detections = detector.detect(frame)

        self.assertEqual(fake_model.predict_calls[0]["conf"], 0.3)
        self.assertEqual(fake_model.predict_calls[0]["iou"], 0.4)
        self.assertEqual(fake_model.predict_calls[0]["imgsz"], 960)
        self.assertEqual(fake_model.predict_calls[0]["device"], "cpu")
        self.assertEqual(fake_model.predict_calls[0]["classes"], [0, 2])
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox, (4, 5, 20, 18))
        self.assertEqual(detections[0].label, "drone")

    def test_detector_pads_missing_batch_results_for_known_clip_frames(self) -> None:
        frames = read_fixture_clip_frames("known_clip.mp4", count=2)
        fake_model = FakeModel([])
        with patch("app.detector.load_yolo_model", return_value=fake_model):
            detector = DroneDetector(DetectorConfig(model_path="fake.pt", target_classes=[]))
            detections = detector.detect_batch(frames)

        self.assertEqual(len(detections), 2)
        self.assertEqual(detections, [[], []])
        self.assertEqual(len(fake_model.predict_calls[0]["frames"]), 2)


def read_fixture_frame(name: str):
    source = resolve_source(FIXTURE_MEDIA / name)
    cap = open_source_capture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise AssertionError(f"could not read fixture frame: {name}")
    return frame


def read_fixture_clip_frames(name: str, count: int):
    source = resolve_source(FIXTURE_MEDIA / name)
    cap = open_source_capture(source)
    frames = []
    try:
        for _ in range(count):
            ok, frame = cap.read()
            if not ok:
                raise AssertionError(f"could not read fixture clip frame: {name}")
            frames.append(frame)
    finally:
        cap.release()
    return frames


class FakeModel:
    names = {0: "drone", 1: "bird", 2: "kite"}

    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.predict_calls: list[dict[str, object]] = []

    def predict(self, frames, **kwargs):
        self.predict_calls.append({"frames": frames, **kwargs})
        return self.results


class FakeResult:
    def __init__(self, boxes: object) -> None:
        self.boxes = boxes


class FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)
        self.cls = FakeTensor(cls)

    def __len__(self) -> int:
        return len(self.conf.values)


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


if __name__ == "__main__":
    unittest.main()
