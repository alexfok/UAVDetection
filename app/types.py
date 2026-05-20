from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    label: str

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    bbox: tuple[int, int, int, int]
    label: str
    class_id: int
    confidence: float
    first_seen: float
    last_seen: float
    seen_frames: int
    recent_hits: int

