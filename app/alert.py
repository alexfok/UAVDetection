from __future__ import annotations

from dataclasses import dataclass

from app.config import AlertConfig
from app.types import TrackedDetection


@dataclass(frozen=True)
class AlertStatus:
    active: bool
    message: str
    tracks: list[TrackedDetection]


class AlertManager:
    def __init__(self, config: AlertConfig) -> None:
        self.config = config
        self._active = False
        self._last_triggered_at = 0.0
        self._last_tracks: list[TrackedDetection] = []

    def update(self, tracks: list[TrackedDetection], now: float) -> AlertStatus:
        persistent_tracks = [
            track
            for track in tracks
            if track.confidence >= self.config.confidence_threshold
            and track.recent_hits >= self.config.persistence_frames
        ]

        if persistent_tracks:
            self._active = True
            self._last_triggered_at = now
            self._last_tracks = persistent_tracks
        elif self._active and now - self._last_triggered_at <= self.config.cooldown_seconds:
            persistent_tracks = self._last_tracks
        else:
            self._active = False
            self._last_tracks = []

        message = self._message(persistent_tracks) if self._active else "Monitoring"
        return AlertStatus(active=self._active, message=message, tracks=persistent_tracks)

    @staticmethod
    def _message(tracks: list[TrackedDetection]) -> str:
        if not tracks:
            return "DRONE ALERT"
        best = max(tracks, key=lambda track: track.confidence)
        return f"DRONE ALERT: {best.label} #{best.track_id} {best.confidence:.2f}"
