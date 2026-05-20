from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from app.config import TrackerConfig
from app.types import Detection, TrackedDetection


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    label: str
    class_id: int
    confidence: float
    first_seen: float
    last_seen: float
    seen_frames: int = 1
    seen_timestamps: deque[float] = field(default_factory=deque)

    def snapshot(self, window_start: float) -> "TrackedDetection":
        recent_hits = sum(1 for timestamp in self.seen_timestamps if timestamp >= window_start)
        return TrackedDetection(
            track_id=self.track_id,
            bbox=self.bbox,
            label=self.label,
            class_id=self.class_id,
            confidence=self.confidence,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            seen_frames=self.seen_frames,
            recent_hits=recent_hits,
        )


class SimpleTracker:
    def __init__(self, config: TrackerConfig, alert_window_seconds: float) -> None:
        self.config = config
        self.alert_window_seconds = alert_window_seconds
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection], now: float) -> list[TrackedDetection]:
        self._expire_old_tracks(now)

        usable_detections = [
            detection for detection in detections if detection.area >= self.config.min_box_area
        ]
        usable_detections.sort(key=lambda detection: detection.confidence, reverse=True)

        matched_track_ids: set[int] = set()
        updated_track_ids: set[int] = set()

        for detection in usable_detections:
            track = self._best_match(detection, matched_track_ids)
            if track is None:
                track = self._new_track(detection, now)
            else:
                self._update_track(track, detection, now)

            matched_track_ids.add(track.track_id)
            updated_track_ids.add(track.track_id)

        window_start = now - self.alert_window_seconds
        return [
            self._tracks[track_id].snapshot(window_start)
            for track_id in updated_track_ids
            if track_id in self._tracks
        ]

    def _new_track(self, detection: Detection, now: float) -> Track:
        track = Track(
            track_id=self._next_id,
            bbox=detection.bbox,
            label=detection.label,
            class_id=detection.class_id,
            confidence=detection.confidence,
            first_seen=now,
            last_seen=now,
            seen_timestamps=deque([now]),
        )
        self._tracks[track.track_id] = track
        self._next_id += 1
        return track

    @staticmethod
    def _update_track(track: Track, detection: Detection, now: float) -> None:
        track.bbox = detection.bbox
        track.label = detection.label
        track.class_id = detection.class_id
        track.confidence = detection.confidence
        track.last_seen = now
        track.seen_frames += 1
        track.seen_timestamps.append(now)

    def _best_match(self, detection: Detection, matched_track_ids: set[int]) -> Track | None:
        best_track: Track | None = None
        best_iou = 0.0

        for track in self._tracks.values():
            if track.track_id in matched_track_ids:
                continue

            score = iou(detection.bbox, track.bbox)
            if score > best_iou:
                best_iou = score
                best_track = track

        if best_track is None or best_iou < self.config.iou_match_threshold:
            return None
        return best_track

    def _expire_old_tracks(self, now: float) -> None:
        max_age = self.config.max_track_age_sec
        expired_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if now - track.last_seen > max_age
        ]
        for track_id in expired_ids:
            del self._tracks[track_id]

        keep_after = now - max(max_age, self.alert_window_seconds)
        for track in self._tracks.values():
            while track.seen_timestamps and track.seen_timestamps[0] < keep_after:
                track.seen_timestamps.popleft()


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union
