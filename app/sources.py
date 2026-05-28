from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CAMERA_NAME_CACHE_SECONDS = 300.0
_CAMERA_NAMES_CACHE: tuple[float, list[str]] = (0.0, [])


@dataclass(frozen=True)
class SourceSpec:
    capture_source: str | int
    label: str
    kind: str

    @property
    def is_image(self) -> bool:
        return self.kind == "image"


@dataclass(frozen=True)
class CameraEntry:
    camera_id: str
    name: str
    address: str = ""
    url: str = ""
    url_env: str = ""
    protocol: str = "rtsp"
    rtsp_port: int = 554
    rtsp_path: str = ""
    username: str = ""
    password: str = ""
    username_env: str = ""
    password_env: str = ""
    enabled: bool = True
    metadata: dict[str, Any] | None = None


class ImageCapture:
    def __init__(self, path: str | Path) -> None:
        import cv2

        self.path = str(path)
        self._frame = cv2.imread(self.path)
        self._read = False

    def isOpened(self) -> bool:
        return self._frame is not None

    def read(self):
        if self._frame is None or self._read:
            return False, None
        self._read = True
        return True, self._frame.copy()

    def release(self) -> None:
        return None

    def set(self, _prop_id: int, _value: float) -> bool:
        return False


def resolve_source(source: str | int, camera_config: str | Path | None = None) -> SourceSpec:
    if isinstance(source, int):
        return SourceSpec(capture_source=source, label=f"camera:{source}", kind="camera")

    text = str(source).strip()
    if not text:
        raise ValueError("Video source is empty.")

    if text.startswith("camera:"):
        camera_id = text.split(":", 1)[1].strip()
        return resolve_camera(camera_id, camera_config)

    if text in {"embedded", "usb"}:
        return SourceSpec(capture_source=0, label=text, kind="camera")

    if text.startswith("usb:"):
        index = text.split(":", 1)[1].strip()
        if not index.isdigit():
            raise ValueError(f"USB camera source must look like usb:0, got: {text}")
        return SourceSpec(capture_source=int(index), label=f"usb:{index}", kind="camera")

    if text.isdigit():
        return SourceSpec(capture_source=int(text), label=f"camera:{text}", kind="camera")

    path = Path(text).expanduser()
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return SourceSpec(capture_source=str(path), label=str(path), kind="image")

    return SourceSpec(capture_source=text, label=redact_source(text), kind=guess_stream_kind(text))


def resolve_camera(camera_id: str, camera_config: str | Path | None = None) -> SourceSpec:
    entries = load_camera_entries(camera_config)
    entry = entries.get(normalise_camera_id(camera_id))
    if entry is None:
        known = ", ".join(sorted(entries)) or "none"
        raise ValueError(f"Unknown camera '{camera_id}'. Known cameras: {known}")
    if not entry.enabled:
        raise ValueError(f"Camera '{camera_id}' is disabled in the camera config.")

    url = camera_url(entry)
    label = entry.name or entry.camera_id
    if entry.address:
        label = f"{label} ({entry.address})"
    return SourceSpec(capture_source=url, label=label, kind="rtsp")


def load_camera_entries(camera_config: str | Path | None = None) -> dict[str, CameraEntry]:
    config_path = Path(camera_config or "data_store/system_config/cameras.yaml")
    if not config_path.exists():
        return {}

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Camera config must contain a YAML mapping: {config_path}")

    defaults = data.get("defaults") or {}
    raw_cameras = data.get("cameras") or {}
    if not isinstance(defaults, dict):
        raise ValueError(f"Camera config defaults must be a mapping: {config_path}")

    entries: dict[str, CameraEntry] = {}
    if isinstance(raw_cameras, dict):
        iterable = [dict(value or {}, id=key) for key, value in raw_cameras.items()]
    elif isinstance(raw_cameras, list):
        iterable = raw_cameras
    else:
        raise ValueError(f"Camera config cameras must be a mapping or list: {config_path}")

    for raw_entry in iterable:
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Camera entries must be YAML mappings: {config_path}")
        merged = {**defaults, **raw_entry}
        camera_id = normalise_camera_id(str(merged.get("id") or merged.get("camera_id") or ""))
        if not camera_id:
            raise ValueError(f"Camera entry is missing id: {config_path}")

        entry = CameraEntry(
            camera_id=camera_id,
            name=str(merged.get("name") or camera_id),
            address=str(merged.get("address") or ""),
            url=str(merged.get("url") or ""),
            url_env=str(merged.get("url_env") or ""),
            protocol=str(merged.get("protocol") or "rtsp").lower(),
            rtsp_port=int(merged.get("rtsp_port") or merged.get("port") or 554),
            rtsp_path=str(merged.get("rtsp_path") or ""),
            username=str(merged.get("username") or ""),
            password=str(merged.get("password") or ""),
            username_env=str(merged.get("username_env") or ""),
            password_env=str(merged.get("password_env") or ""),
            enabled=bool(merged.get("enabled", True)),
            metadata={key: value for key, value in merged.items() if key not in CAMERA_FIELDS},
        )
        entries[camera_id] = entry
    return entries


def camera_url(entry: CameraEntry) -> str:
    if entry.url_env:
        env_value = os.environ.get(entry.url_env)
        if env_value:
            return env_value

    if entry.url:
        return os.path.expandvars(entry.url)

    if not entry.address:
        raise ValueError(f"Camera '{entry.camera_id}' needs either url, url_env, or address.")

    protocol = entry.protocol or "rtsp"
    if protocol != "rtsp":
        raise ValueError(
            f"Camera '{entry.camera_id}' uses protocol '{protocol}'. "
            "OpenCV live detection currently needs an RTSP URL."
        )

    username = entry.username or os.environ.get(entry.username_env, "")
    password = entry.password or os.environ.get(entry.password_env, "")
    credentials = ""
    if username:
        credentials = quote(username, safe="")
        if password:
            credentials += f":{quote(password, safe='')}"
        credentials += "@"

    path = entry.rtsp_path or ""
    if path and not path.startswith("/"):
        path = f"/{path}"
    return f"rtsp://{credentials}{entry.address}:{entry.rtsp_port}{path}"


def camera_summary(camera_config: str | Path | None = None) -> list[str]:
    entries = load_camera_entries(camera_config)
    lines: list[str] = []
    for camera_id in sorted(entries):
        entry = entries[camera_id]
        status = "enabled" if entry.enabled else "disabled"
        target = entry.address or entry.url or (f"${entry.url_env}" if entry.url_env else "missing target")
        model = ""
        if entry.metadata:
            model_value = entry.metadata.get("model")
            if model_value:
                model = f" | {model_value}"
        lines.append(f"{camera_id}: {entry.name} | {target} | {status}{model}")
    return lines


def open_source_capture(source: SourceSpec):
    import cv2

    if source.is_image:
        return ImageCapture(source.capture_source)
    quiet_opencv_logging(cv2)
    if has_url_credentials(source.capture_source):
        with suppress_native_stderr():
            return cv2.VideoCapture(source.capture_source)
    return cv2.VideoCapture(source.capture_source)


def scan_local_cameras(max_index: int = 5) -> list[dict[str, Any]]:
    import cv2

    quiet_opencv_logging(cv2)
    camera_names = local_camera_names()
    cameras: list[dict[str, Any]] = []
    max_index = max(0, min(max_index, 20))
    for index in range(max_index + 1):
        cap = None
        try:
            with suppress_native_stderr():
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    continue
                ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame_height, frame_width = frame.shape[:2]
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame_width)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame_height)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cameras.append(
                {
                    "id": f"local_{index}",
                    "name": camera_name_for_index(index, camera_names),
                    "source": str(index),
                    "kind": "local",
                    "width": width,
                    "height": height,
                    "fps": round(fps, 2) if fps > 0 else 0,
                }
            )
        finally:
            if cap is not None:
                cap.release()
    return cameras


def camera_name_for_index(index: int, camera_names: list[str]) -> str:
    if 0 <= index < len(camera_names):
        return camera_names[index]
    return f"Local camera {index}"


def local_camera_names() -> list[str]:
    global _CAMERA_NAMES_CACHE

    if sys.platform != "darwin":
        return []
    cached_at, cached_names = _CAMERA_NAMES_CACHE
    if cached_names and time.monotonic() - cached_at < CAMERA_NAME_CACHE_SECONDS:
        return list(cached_names)

    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    names = parse_macos_camera_names(result.stdout)
    if names:
        _CAMERA_NAMES_CACHE = (time.monotonic(), names)
    return names


def parse_macos_camera_names(output: str) -> list[str]:
    names: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "Camera:":
            continue
        if not stripped.endswith(":"):
            continue
        if ":" in stripped[:-1]:
            continue
        names.append(stripped[:-1])
    return names


def quiet_opencv_logging(cv2_module) -> None:
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "FATAL")
    set_log_level = getattr(cv2_module, "setLogLevel", None)
    if set_log_level is not None:
        set_log_level(1)


def has_url_credentials(source: str | int) -> bool:
    if not isinstance(source, str):
        return False
    try:
        parsed = urlsplit(source)
    except ValueError:
        return False
    return bool(parsed.username or parsed.password)


@contextmanager
def suppress_native_stderr():
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


def normalise_camera_id(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def guess_stream_kind(source: str) -> str:
    source_lower = source.lower()
    if source_lower.startswith("rtsp://"):
        return "rtsp"
    if source_lower.startswith(("rtmp://", "http://", "https://")):
        return "stream"
    return "video"


def redact_source(source: str) -> str:
    try:
        parsed = urlsplit(source)
    except ValueError:
        return source
    if not parsed.username and not parsed.password:
        return source

    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    if parsed.username or parsed.password:
        host = f"<credentials>@{host}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


CAMERA_FIELDS = {
    "id",
    "camera_id",
    "name",
    "address",
    "url",
    "url_env",
    "protocol",
    "rtsp_port",
    "port",
    "rtsp_path",
    "username",
    "password",
    "username_env",
    "password_env",
    "enabled",
}
