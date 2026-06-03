from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from app.torchvision_compat import install_torchvision_nms_fallback

install_torchvision_nms_fallback()

from ultralytics import YOLO

from app.config import DetectorConfig
from app.labels import display_label, normalise_label_aliases, resolve_target_class_ids
from app.types import Detection

LOGGER = logging.getLogger(__name__)


class DroneDetector:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        LOGGER.info("Loading YOLO model: %s", config.model_path)
        self.model = YOLO(config.model_path)
        self.names = self._normalise_names(self.model.names)
        self.label_aliases = normalise_label_aliases(config.label_aliases)
        self.target_class_ids = self._resolve_target_classes(config.target_classes)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        predict_args = {
            "conf": self.config.confidence_threshold,
            "iou": self.config.iou_threshold,
            "imgsz": self.config.image_size,
            "verbose": False,
        }
        if self.config.device:
            predict_args["device"] = self.config.device
        if self.target_class_ids:
            predict_args["classes"] = self.target_class_ids

        results = self.model.predict(frames, **predict_args)
        if not results:
            return [[] for _frame in frames]

        detections_by_frame = [self._detections_from_result(result) for result in results]
        if len(detections_by_frame) < len(frames):
            detections_by_frame.extend([[] for _frame in frames[len(detections_by_frame) :]])
        return detections_by_frame[: len(frames)]

    def _detections_from_result(self, result: object) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        detections: list[Detection] = []
        for bbox, confidence, class_id in zip(xyxy, confidences, classes):
            raw_label = self.names.get(int(class_id), str(class_id))
            label = display_label(raw_label, self.label_aliases)
            detections.append(
                Detection(
                    bbox=tuple(int(value) for value in bbox),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    label=label,
                )
            )
        return detections

    @staticmethod
    def _normalise_names(names: dict[int, str] | list[str]) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        return {index: str(value) for index, value in enumerate(names)}

    def _resolve_target_classes(self, targets: Iterable[str]) -> list[int] | None:
        target_names = {name.strip().lower() for name in targets if name and name.strip()}
        if not target_names:
            LOGGER.info("No target class filter configured; detector will return all classes.")
            return None

        ids = resolve_target_class_ids(self.names, targets, self.label_aliases)
        if not ids:
            LOGGER.warning(
                "None of the configured target classes exist in this model: %s. "
                "Detector will return all classes. Label aliases: %s",
                sorted(target_names),
                self.label_aliases,
            )
            return None

        LOGGER.info(
            "Filtering detector to classes: %s",
            ", ".join(
                f"{self.names[class_id]}->{display_label(self.names[class_id], self.label_aliases)}({class_id})"
                for class_id in ids
            ),
        )
        return ids
