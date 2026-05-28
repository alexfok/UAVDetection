from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.alert import AlertStatus
from app.config import UIConfig
from app.types import TrackedDetection


@dataclass
class UIResult:
    should_quit: bool = False


class OpenCVUI:
    def __init__(self, config: UIConfig) -> None:
        self.config = config
        self._created_window = False

    def draw(
        self,
        frame: np.ndarray,
        tracks: list[TrackedDetection],
        alert: AlertStatus,
        fps: float,
        source: str,
    ) -> np.ndarray:
        output = frame.copy()
        visible_tracks = tracks if self.config.draw_all_tracks else alert.tracks
        alert_track_ids = {track.track_id for track in alert.tracks}

        for track in visible_tracks:
            color = (0, 0, 255) if track.track_id in alert_track_ids else (0, 180, 255)
            self._draw_track(output, track, color)

        if self.config.draw_status_bar:
            self._draw_status_bar(output, alert, fps, source)
        return output

    def show(self, frame: np.ndarray, wait_ms: int = 1) -> UIResult:
        if not self.config.show_window:
            return UIResult()

        if not self._created_window:
            cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
            if self.config.fullscreen:
                cv2.setWindowProperty(
                    self.config.window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )
            self._created_window = True

        cv2.imshow(self.config.window_name, frame)
        key = cv2.waitKey(wait_ms) & 0xFF
        return UIResult(should_quit=key in {27, ord("q")})

    def close(self) -> None:
        if self._created_window:
            cv2.destroyWindow(self.config.window_name)
            self._created_window = False

    @staticmethod
    def _draw_track(frame: np.ndarray, track: TrackedDetection, color: tuple[int, int, int]) -> None:
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"#{track.track_id} {track.label} {track.confidence:.2f} hits:{track.recent_hits}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_width, text_height = text_size
        cv2.rectangle(frame, (x1, max(0, y1 - text_height - 10)), (x1 + text_width + 8, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_status_bar(frame: np.ndarray, alert: AlertStatus, fps: float, source: str) -> None:
        _, width = frame.shape[:2]
        bar_height = 62 if alert.active else 44
        color = (0, 0, 180) if alert.active else (24, 24, 24)
        cv2.rectangle(frame, (0, 0), (width, bar_height), color, -1)

        status_text = alert.message if alert.active else "Monitoring sky"
        cv2.putText(
            frame,
            status_text,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        meta = f"FPS {fps:.1f} | {source} | {time.strftime('%H:%M:%S')}"
        cv2.putText(
            frame,
            meta,
            (16, bar_height - 10 if alert.active else 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
