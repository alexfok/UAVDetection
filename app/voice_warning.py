from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
from pathlib import Path

from app.config import VoiceWarningConfig


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VoiceWarningPlayer:
    """Play prerecorded alert transitions without blocking video processing."""

    def __init__(self, config: VoiceWarningConfig) -> None:
        self.config = config
        self._source_states: dict[str, bool] = {}
        self._aggregate_active = False
        self._last_warning_at = float("-inf")
        self._state_lock = threading.Lock()
        self._queue: queue.Queue[Path | None] = queue.Queue(maxsize=4)
        self._worker: threading.Thread | None = None
        self._closed = False
        self._reported_errors: set[str] = set()

    def update(self, active: bool, now: float, source_id: str = "default") -> str | None:
        """Update one detection source and return the clip transition, if any."""
        if not self.config.enabled or self._closed:
            return None

        clip_kind: str | None = None
        with self._state_lock:
            self._source_states[source_id] = active
            aggregate_active = any(self._source_states.values())
            if aggregate_active and (
                not self._aggregate_active or now - self._last_warning_at >= max(0.0, self.config.repeat_seconds)
            ):
                clip_kind = "warning"
                self._last_warning_at = now
            elif not aggregate_active and self._aggregate_active and self.config.play_all_clear:
                clip_kind = "all_clear"
            self._aggregate_active = aggregate_active

        if clip_kind:
            self._enqueue(self._clip_path(clip_kind))
        return clip_kind

    def release_source(self, source_id: str, now: float) -> str | None:
        """Remove a stopped stream and announce all-clear when it was the last active source."""
        if not self.config.enabled or self._closed:
            return None

        clip_kind: str | None = None
        with self._state_lock:
            self._source_states.pop(source_id, None)
            aggregate_active = any(self._source_states.values())
            if not aggregate_active and self._aggregate_active and self.config.play_all_clear:
                clip_kind = "all_clear"
            self._aggregate_active = aggregate_active

        if clip_kind:
            self._enqueue(self._clip_path(clip_kind))
        return clip_kind

    def close(self) -> None:
        self._closed = True
        worker = self._worker
        if worker is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=3.0)

    def _clip_path(self, clip_kind: str) -> Path:
        configured = self.config.warning_path if clip_kind == "warning" else self.config.all_clear_path
        path = Path(configured).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _enqueue(self, path: Path) -> None:
        if not path.is_file():
            self._log_once(f"missing:{path}", logging.ERROR, "Voice warning recording not found: %s", path)
            return
        if self._worker is None:
            self._worker = threading.Thread(target=self._playback_worker, name="voice-warning", daemon=True)
            self._worker.start()
        try:
            self._queue.put_nowait(path)
        except queue.Full:
            LOGGER.warning("Voice warning queue is full; skipping %s", path.name)

    def _playback_worker(self) -> None:
        while True:
            path = self._queue.get()
            if path is None:
                return
            try:
                self._play_file(path)
            finally:
                self._queue.task_done()

    def _play_file(self, path: Path) -> None:
        player = self._resolve_player()
        if not player:
            self._log_once(
                "no-player",
                logging.ERROR,
                "Voice warnings enabled, but neither aplay nor paplay is installed",
            )
            return

        if player == "aplay":
            command = [player, "-q"]
            if self.config.output_device:
                command.extend(["-D", self.config.output_device])
            command.append(str(path))
        else:
            command = [player]
            if self.config.output_device:
                command.append(f"--device={self.config.output_device}")
            command.append(str(path))

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._log_once(f"play:{exc}", logging.ERROR, "Voice warning playback failed: %s", exc)
            return
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            self._log_once(f"play:{detail}", logging.ERROR, "Voice warning playback failed: %s", detail)

    def _resolve_player(self) -> str:
        requested = self.config.player.strip().lower()
        candidates = ("aplay", "paplay") if requested == "auto" else (requested,)
        for candidate in candidates:
            if candidate in {"aplay", "paplay"} and shutil.which(candidate):
                return candidate
        return ""

    def _log_once(self, key: str, level: int, message: str, *args: object) -> None:
        if key in self._reported_errors:
            return
        self._reported_errors.add(key)
        LOGGER.log(level, message, *args)
