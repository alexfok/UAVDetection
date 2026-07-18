from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hmac
import hashlib
import json
import mimetypes
import os
import re
import secrets
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
STATIC_ROOT = PROJECT_ROOT / "web" / "annotator"
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
RECORDING_MAX_BYTES = 30 * 1024 * 1024
RECORDING_ROLLOVER_BYTES = 28 * 1024 * 1024
DEFAULT_PRESENCE_OUT_SECONDS = 2.0
MANIFEST_FIELDS = [
    "image_id",
    "split",
    "image_path",
    "label_path",
    "source_media",
    "media_kind",
    "frame_time",
    "image_width",
    "image_height",
    "box_count",
    "reviewed",
    "saved_at",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local web annotation server for UAV/drone YOLO labels.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--default-folder", type=Path, default=Path("data_store/raw_data/Roni"))
    parser.add_argument("--project-dir", type=Path, default=Path("data_store/datasets/web_drone_v1"))
    parser.add_argument("--class-name", default="drone")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", help="HTTP Basic Auth password. Prefer --password-env for shared machines.")
    parser.add_argument(
        "--password-env",
        default="ANNOTATION_SERVER_PASSWORD",
        help="Environment variable to read the Basic Auth password from.",
    )
    parser.add_argument("--no-auth", action="store_true", help="Disable Basic Auth. Not recommended off localhost.")
    parser.add_argument("--certfile", type=Path, help="TLS certificate file for HTTPS.")
    parser.add_argument("--keyfile", type=Path, help="TLS private key file for HTTPS.")
    parser.add_argument("--camera-config", type=Path, default=Path("data_store/system_config/cameras.yaml"))
    parser.add_argument("--live-model", type=Path, default=Path("data_store/models/trained/yolov8n_drone_best.pt"))
    parser.add_argument("--no-voice-warning", action="store_true", help="Disable prerecorded detection warnings.")
    parser.add_argument("--voice-warning-file", type=Path, default=Path("assets/audio/drone_warning.wav"))
    parser.add_argument("--voice-all-clear-file", type=Path, default=Path("assets/audio/drone_all_clear.wav"))
    parser.add_argument("--voice-repeat-seconds", type=float, default=15.0)
    parser.add_argument("--voice-player", choices=("auto", "aplay", "paplay"), default="auto")
    parser.add_argument("--voice-device", default="", help="Optional ALSA or PulseAudio output device.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = AnnotationServer((args.host, args.port), AnnotationHandler)
    server.default_folder = resolve_user_path(args.default_folder)
    server.default_project_dir = resolve_user_path(args.project_dir)
    server.camera_config = resolve_user_path(args.camera_config)
    server.local_camera_cache = PROJECT_ROOT / "data_store" / "system_config" / "local_cameras.json"
    server.local_camera_scan_running = False
    server.training_lock = threading.Lock()
    server.training_job = None
    server.diagnostics_lock = threading.Lock()
    server.diagnostics_job = None
    server.debug_session_lock = threading.Lock()
    server.debug_session_id = ""
    server.debug_session_path = None
    server.detector_cache_lock = threading.Lock()
    server.detector_cache = {}
    server.live_model = resolve_user_path(args.live_model)
    from app.config import VoiceWarningConfig
    from app.voice_warning import VoiceWarningPlayer

    server.voice_warning = VoiceWarningPlayer(
        VoiceWarningConfig(
            enabled=not args.no_voice_warning,
            warning_path=str(resolve_user_path(args.voice_warning_file)),
            all_clear_path=str(resolve_user_path(args.voice_all_clear_file)),
            repeat_seconds=max(0.0, args.voice_repeat_seconds),
            player=args.voice_player,
            output_device=args.voice_device,
        )
    )
    server.class_name = args.class_name
    server.auth_enabled = not args.no_auth
    server.auth_username = args.username
    server.auth_password = resolve_auth_password(args)

    scheme = "http"
    if args.certfile:
        certfile = resolve_user_path(args.certfile)
        keyfile = resolve_user_path(args.keyfile) if args.keyfile else certfile
        if not certfile.exists():
            raise SystemExit(f"TLS certfile not found: {certfile}")
        if not keyfile.exists():
            raise SystemExit(f"TLS keyfile not found: {keyfile}")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=certfile, keyfile=keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"

    generated_password = False
    if server.auth_enabled and not server.auth_password:
        server.auth_password = secrets.token_urlsafe(18)
        generated_password = True
        print("Generated one-time annotation password for this server run.")

    display_host = local_display_host(args.host)
    print(f"Annotation server: {scheme}://{display_host}:{args.port}")
    print(f"Default media folder: {server.default_folder}")
    print(f"Default annotation project: {server.default_project_dir}")
    print(f"Camera registry: {server.camera_config}")
    print(f"Default live model: {server.live_model}")
    if server.auth_enabled:
        print(f"Username: {server.auth_username}")
        if generated_password:
            print(f"Password: {server.auth_password}")
        else:
            print("Password: configured")
    else:
        print("WARNING: Basic Auth is disabled.")
    if scheme == "http" and args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("WARNING: server is reachable over plain HTTP. Use --certfile/--keyfile for HTTPS.")
    initialise_local_camera_cache(server)
    threading.Thread(target=prewarm_live_detector, args=(server,), daemon=True).start()
    try:
        server.serve_forever()
    finally:
        server.voice_warning.close()
        server.server_close()
    return 0


class AnnotationServer(ThreadingHTTPServer):
    default_folder: Path
    default_project_dir: Path
    camera_config: Path
    local_camera_cache: Path
    local_camera_scan_running: bool
    training_lock: threading.Lock
    training_job: dict[str, object] | None
    diagnostics_lock: threading.Lock
    diagnostics_job: dict[str, object] | None
    debug_session_lock: threading.Lock
    debug_session_id: str
    debug_session_path: Path | None
    detector_cache_lock: threading.Lock
    detector_cache: dict[tuple[object, ...], object]
    live_model: Path
    voice_warning: object
    class_name: str
    auth_enabled: bool
    auth_username: str
    auth_password: str


class SharedDetector:
    def __init__(self, detector: object) -> None:
        self.detector = detector
        self.condition = threading.Condition()
        self.pending = []
        self.batch_window_seconds = 0.015
        self.worker = threading.Thread(target=self._run_batches, daemon=True)
        self.worker.start()

    def detect(self, frame):
        request = DetectionBatchRequest(frame)
        with self.condition:
            self.pending.append(request)
            self.condition.notify()
        request.done.wait()
        if request.error is not None:
            raise request.error
        return request.result or []

    def _run_batches(self) -> None:
        while True:
            with self.condition:
                while not self.pending:
                    self.condition.wait()
                deadline = time.monotonic() + self.batch_window_seconds
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self.condition.wait(timeout=remaining)
                requests = self.pending
                self.pending = []

            frames = [request.frame for request in requests]
            try:
                if hasattr(self.detector, "detect_batch"):
                    results = self.detector.detect_batch(frames)
                else:
                    results = [self.detector.detect(frame) for frame in frames]
                if len(results) < len(requests):
                    results = list(results) + [[] for _request in requests[len(results) :]]
                for request, result in zip(requests, results):
                    request.result = result
                    request.done.set()
            except Exception as exc:
                for request in requests:
                    request.error = exc
                    request.done.set()


class DetectionBatchRequest:
    def __init__(self, frame: object) -> None:
        self.frame = frame
        self.done = threading.Event()
        self.result = None
        self.error = None


def detector_cache_key(config: object) -> tuple[object, ...]:
    label_aliases = getattr(config, "label_aliases", {}) or {}
    target_classes = getattr(config, "target_classes", []) or []
    return (
        str(getattr(config, "model_path", "")),
        float(getattr(config, "confidence_threshold", 0.5)),
        float(getattr(config, "iou_threshold", 0.45)),
        int(getattr(config, "image_size", 640)),
        str(getattr(config, "device", "")),
        tuple(str(value) for value in target_classes),
        tuple(sorted((str(key), str(value)) for key, value in label_aliases.items())),
    )


def shared_detector_for(server: AnnotationServer, config: object, detector_class: object) -> SharedDetector:
    key = detector_cache_key(config)
    with server.detector_cache_lock:
        detector = server.detector_cache.get(key)
        if detector is None:
            detector = SharedDetector(detector_class(config))
            server.detector_cache[key] = detector
        return detector


def prewarm_live_detector(server: AnnotationServer) -> None:
    try:
        from app.config import DetectorConfig
        from app.detector import DroneDetector

        shared_detector_for(server, DetectorConfig(model_path=str(server.live_model)), DroneDetector)
        print("Default live detector prewarmed.")
    except Exception as exc:
        print(f"Default live detector prewarm failed: {exc}")


class AnnotationHandler(BaseHTTPRequestHandler):
    server: AnnotationServer

    def do_GET(self) -> None:
        if not self.require_auth():
            return

        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.send_static(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path.startswith("/static/"):
            static_path = STATIC_ROOT / parsed.path.removeprefix("/static/")
            self.send_static(static_path)
            return
        if parsed.path == "/api/defaults":
            self.send_json(
                {
                    "folder": str(self.server.default_folder),
                    "project_dir": str(self.server.default_project_dir),
                    "class_name": self.server.class_name,
                    "camera_config": str(self.server.camera_config),
                    "live_model": str(self.server.live_model),
                }
            )
            return
        if parsed.path == "/api/live/cameras":
            self.send_live_cameras()
            return
        if parsed.path == "/api/live/local-cameras":
            query = parse_qs(parsed.query)
            max_index = min(20, max(0, parse_int(query_value(query, "max_index", "5"))))
            refresh = parse_bool(query_value(query, "refresh", "0"))
            self.send_live_local_cameras(max_index, refresh=refresh)
            return
        if parsed.path == "/api/live/events":
            query = parse_qs(parsed.query)
            limit = min(200, max(1, parse_int(query_value(query, "limit", "50"))))
            self.send_live_events(limit)
            return
        if parsed.path == "/api/training/status":
            self.send_training_status()
            return
        if parsed.path == "/api/diagnostics/status":
            self.send_json(diagnostics_status_payload(self.server))
            return
        if parsed.path == "/api/diagnostics/download":
            query = parse_qs(parsed.query)
            self.send_diagnostics_artifact(
                query_value(query, "run_id", ""),
                query_value(query, "artifact", "report"),
            )
            return
        if parsed.path == "/api/live/stream":
            query = parse_qs(parsed.query)
            self.stream_live_detection(query)
            return
        if parsed.path == "/api/scan":
            query = parse_qs(parsed.query)
            folder = query.get("folder", [str(self.server.default_folder)])[0]
            self.scan_folder(folder)
            return
        if parsed.path == "/api/media":
            query = parse_qs(parsed.query)
            media_path = query.get("path", [""])[0]
            self.send_media(media_path)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if not self.require_auth():
            return

        parsed = urlparse(self.path)
        if parsed.path == "/api/scan":
            payload = self.read_json()
            self.scan_folder(str(payload.get("folder") or self.server.default_folder))
            return
        if parsed.path == "/api/stats":
            payload = self.read_json()
            self.send_json(
                build_dashboard_stats(
                    str(payload.get("folder") or self.server.default_folder),
                    str(payload.get("project_dir") or self.server.default_project_dir),
                )
            )
            return
        if parsed.path == "/api/save":
            payload = self.read_json()
            self.save_annotation(payload)
            return
        if parsed.path == "/api/training/start":
            payload = self.read_json()
            self.start_training_job(payload)
            return
        if parsed.path == "/api/training/stop":
            self.stop_training_job()
            return
        if parsed.path == "/api/diagnostics/run":
            payload = self.read_json()
            self.start_diagnostics_job(payload)
            return
        if parsed.path == "/api/debug-session/start":
            payload = self.read_json()
            self.start_debug_session(payload)
            return
        if parsed.path == "/api/debug-session/event":
            payload = self.read_json()
            self.record_debug_session_event(payload)
            return
        if parsed.path == "/api/debug-session/stop":
            payload = self.read_json()
            self.stop_debug_session(payload)
            return
        if parsed.path == "/api/media/remove":
            payload = self.read_json()
            self.remove_media(payload)
            return
        if parsed.path == "/api/live/events/remove":
            payload = self.read_json()
            self.remove_live_events(payload)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def scan_folder(self, folder_value: str) -> None:
        folder = resolve_user_path(folder_value)
        if not folder.exists() or not folder.is_dir():
            self.send_json({"error": f"Folder not found: {folder}"}, status=HTTPStatus.BAD_REQUEST)
            return

        media = []
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
                continue
            stat = path.stat()
            media.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "relative": str(path.relative_to(folder)),
                    "kind": "video" if path.suffix.lower() in VIDEO_EXTENSIONS else "image",
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
        media.sort(key=lambda item: (-float(item["mtime"]), str(item["relative"]).lower()))

        self.send_json({"folder": str(folder), "media": media})

    def save_annotation(self, payload: dict[str, object]) -> None:
        source_path = resolve_user_path(str(payload.get("source_path") or ""))
        if not source_path.exists():
            self.send_json({"error": f"Source media not found: {source_path}"}, status=HTTPStatus.BAD_REQUEST)
            return

        image_data = str(payload.get("image_data") or "")
        image_bytes = decode_data_url(image_data)
        if not image_bytes:
            self.send_json({"error": "Missing image_data"}, status=HTTPStatus.BAD_REQUEST)
            return

        split = str(payload.get("split") or "train")
        if split not in {"train", "val"}:
            self.send_json({"error": "split must be train or val"}, status=HTTPStatus.BAD_REQUEST)
            return

        project_dir = resolve_user_path(str(payload.get("project_dir") or self.server.default_project_dir))
        class_name = str(payload.get("class_name") or self.server.class_name or "drone")
        media_kind = str(payload.get("media_kind") or media_kind_for_path(source_path))
        frame_time = payload.get("frame_time")
        width = int(float(payload.get("image_width") or 0))
        height = int(float(payload.get("image_height") or 0))
        if width <= 0 or height <= 0:
            self.send_json({"error": "image_width and image_height must be positive"}, status=HTTPStatus.BAD_REQUEST)
            return

        boxes = normalise_boxes(payload.get("boxes"), width, height)
        image_id = str(payload.get("image_id") or image_id_for(source_path, media_kind, frame_time))
        safe_id = safe_name(image_id)

        image_path = project_dir / "images" / split / f"{safe_id}.jpg"
        label_path = project_dir / "labels" / split / f"{safe_id}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)
        label_path.write_text(yolo_label_text(boxes, width, height), encoding="utf-8")

        write_data_yaml(project_dir, class_name)
        overwrote = upsert_manifest(
            project_dir / "manifest.csv",
            {
                "image_id": safe_id,
                "split": split,
                "image_path": portable_path(image_path),
                "label_path": portable_path(label_path),
                "source_media": portable_path(source_path),
                "media_kind": media_kind,
                "frame_time": "" if frame_time is None else str(frame_time),
                "image_width": str(width),
                "image_height": str(height),
                "box_count": str(len(boxes)),
                "reviewed": "true",
                "saved_at": datetime.now().astimezone().isoformat(),
                "notes": str(payload.get("notes") or ""),
            },
        )

        self.send_json(
            {
                "ok": True,
                "overwrote": overwrote,
                "image_id": safe_id,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "box_count": len(boxes),
                "project_dir": str(project_dir),
            }
        )

    def send_live_cameras(self) -> None:
        try:
            from app.sources import load_camera_entries

            entries = load_camera_entries(self.server.camera_config)
            cameras = []
            for camera_id in sorted(entries):
                entry = entries[camera_id]
                cameras.append(
                    {
                        "id": camera_id,
                        "name": entry.name,
                        "address": entry.address,
                        "enabled": entry.enabled,
                        "model": (entry.metadata or {}).get("model", ""),
                        "vendor": (entry.metadata or {}).get("vendor", ""),
                    }
                )
            self.send_json({"camera_config": str(self.server.camera_config), "cameras": cameras})
        except Exception as exc:
            self.send_json({"error": str(exc), "cameras": []}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_live_local_cameras(self, max_index: int, refresh: bool = False) -> None:
        try:
            self.send_json(
                load_or_scan_local_cameras(
                    self.server.local_camera_cache,
                    max_index,
                    refresh=refresh,
                    scan_running=self.server.local_camera_scan_running,
                )
            )
        except Exception as exc:
            self.send_json({"error": str(exc), "cameras": []}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_live_events(self, limit: int) -> None:
        events_root = PROJECT_ROOT / "data_store" / "detection_results" / "live_events"
        self.send_json(
            {
                "events_root": str(events_root),
                "events": read_recent_live_events(events_root, limit),
            }
        )

    def send_training_status(self) -> None:
        self.send_json(training_status_payload(self.server))

    def start_training_job(self, payload: dict[str, object]) -> None:
        try:
            status = start_training_process(self.server, payload)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except RuntimeError as exc:
            self.send_json({"error": str(exc), **training_status_payload(self.server)}, status=HTTPStatus.CONFLICT)
            return
        self.send_json(status)

    def stop_training_job(self) -> None:
        self.send_json(stop_training_process(self.server))

    def start_diagnostics_job(self, payload: dict[str, object]) -> None:
        try:
            self.send_json(start_diagnostics_process(self.server, payload))
        except (RuntimeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def start_debug_session(self, payload: dict[str, object]) -> None:
        self.send_json(start_debug_session(self.server, payload))

    def record_debug_session_event(self, payload: dict[str, object]) -> None:
        self.send_json(record_debug_event(self.server, payload))

    def stop_debug_session(self, payload: dict[str, object]) -> None:
        self.send_json(stop_debug_session(self.server, payload))

    def send_diagnostics_artifact(self, run_id: str, artifact: str) -> None:
        run_id = safe_name(run_id)
        artifact = artifact if artifact in {"report", "sysdump", "json"} else "report"
        root = PROJECT_ROOT / "data_store" / "detection_results" / "sysdumps"
        if artifact == "sysdump":
            path = root / f"{run_id}.tar.gz"
            content_type = "application/gzip"
        elif artifact == "json":
            path = root / run_id / "checks.json"
            content_type = "application/json; charset=utf-8"
        else:
            path = root / run_id / f"{run_id}_report.md"
            content_type = "text/markdown; charset=utf-8"
        try:
            path.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Diagnostics artifact not found")
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.send_no_cache_headers()
        self.end_headers()
        self.wfile.write(data)

    def remove_media(self, payload: dict[str, object]) -> None:
        folder = resolve_user_path(str(payload.get("folder") or self.server.default_folder))
        paths = payload.get("paths")
        if not isinstance(paths, list):
            self.send_json({"error": "paths must be a list"}, status=HTTPStatus.BAD_REQUEST)
            return
        if not folder.exists() or not folder.is_dir():
            self.send_json({"error": f"Folder not found: {folder}"}, status=HTTPStatus.BAD_REQUEST)
            return

        trash_root = PROJECT_ROOT / "data_store" / "trash" / "media" / datetime.now().strftime("%Y%m%d_%H%M%S")
        removed: list[dict[str, str]] = []
        failed: list[dict[str, str]] = []
        folder_root = folder.resolve()
        for value in paths:
            source = resolve_user_path(str(value or ""))
            try:
                resolved = source.resolve()
                relative = resolved.relative_to(folder_root)
            except (OSError, ValueError):
                failed.append({"path": str(source), "error": "media must be inside the selected media folder"})
                continue
            if not source.exists() or not source.is_file() or source.suffix.lower() not in MEDIA_EXTENSIONS:
                failed.append({"path": str(source), "error": "media file not found"})
                continue

            target = unique_path(trash_root / relative)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
            except OSError as exc:
                failed.append({"path": str(source), "error": str(exc)})
                continue
            removed.append({"path": str(source), "trash_path": str(target)})

        self.send_json({"removed": removed, "failed": failed, "trash_root": str(trash_root)})

    def remove_live_events(self, payload: dict[str, object]) -> None:
        event_ids = payload.get("event_ids")
        if not isinstance(event_ids, list):
            self.send_json({"error": "event_ids must be a list"}, status=HTTPStatus.BAD_REQUEST)
            return
        events_root = PROJECT_ROOT / "data_store" / "detection_results" / "live_events"
        try:
            result = remove_live_event_rows(events_root, [str(value) for value in event_ids])
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self.send_json(result)

    def stream_live_detection(self, query: dict[str, list[str]]) -> None:
        source_value = query_value(query, "source", "0")
        camera_id = query_value(query, "camera", "")
        camera_profile = query_value(query, "camera_profile", "main").strip().lower() or "main"

        model_path = resolve_user_path(query_value(query, "model", str(self.server.live_model)))
        confidence = parse_float(query_value(query, "conf", "0.3"), 0.3)
        iou = parse_float(query_value(query, "iou", "0.45"), 0.45)
        image_size = parse_int(query_value(query, "imgsz", "960")) or 960
        device = normalise_device_value(query_value(query, "device", ""))
        frame_skip = max(0, parse_int(query_value(query, "frame_skip", "0")))
        preview_fps = max(0.1, parse_float(query_value(query, "preview_fps", query_value(query, "max_fps", "4")), 4.0))
        legacy_max_fps = query_value(query, "max_fps", "")
        detect_fps_default = legacy_max_fps if legacy_max_fps and not query_value(query, "preview_fps", "") else "2"
        detect_fps = max(0.1, parse_float(query_value(query, "detect_fps", detect_fps_default), 2.0))
        max_width = max(0, parse_int(query_value(query, "max_width", "1920")))
        max_height = max(0, parse_int(query_value(query, "max_height", "1080")))
        jpeg_quality = min(95, max(35, parse_int(query_value(query, "quality", "85"))))
        read_failure_limit = max(5, parse_int(query_value(query, "read_failure_limit", "30")))
        reconnect_attempts = max(0, parse_int(query_value(query, "reconnect_attempts", "5")))
        reconnect_delay = max(0.1, parse_float(query_value(query, "reconnect_delay", "1.0"), 1.0))
        presence_out_seconds = max(
            0.1,
            parse_float(
                query_value(query, "presence_out_seconds", str(DEFAULT_PRESENCE_OUT_SECONDS)),
                DEFAULT_PRESENCE_OUT_SECONDS,
            ),
        )
        record_enabled = parse_bool(query_value(query, "record", "0"))
        record_labels = parse_bool(query_value(query, "record_labels", "0"))
        record_name_suffix = query_value(query, "record_name_suffix", "").strip()
        client_run_id = safe_name(query_value(query, "client_run_id", "").strip())[:80]
        record_dir = resolve_user_path(query_value(query, "record_dir", str(self.server.default_folder)))
        record_max_bytes = min(
            RECORDING_MAX_BYTES,
            max(1, parse_int(query_value(query, "record_max_mb", "30"))) * 1024 * 1024,
        )
        record_rollover_bytes = min(RECORDING_ROLLOVER_BYTES, max(1, record_max_bytes - (2 * 1024 * 1024)))

        try:
            import cv2

            from app.alert import AlertManager
            from app.config import AlertConfig, DetectorConfig, TrackerConfig, UIConfig
            from app.detector import DroneDetector
            from app.sources import open_source_capture, resolve_camera, resolve_source
            from app.tracker import SimpleTracker
            from app.ui import OpenCVUI
        except Exception as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Live detection dependencies failed to load: {exc}")
            return

        cap = None
        try:
            if camera_id:
                source = resolve_camera(camera_id, self.server.camera_config, profile=camera_profile)
            else:
                source = resolve_source(source_value, self.server.camera_config)
            cap = open_source_capture(source)
            if not cap.isOpened():
                self.send_error(HTTPStatus.BAD_REQUEST, f"Unable to open source: {source.label}")
                return

            detector_config = DetectorConfig(
                model_path=str(model_path),
                confidence_threshold=confidence,
                iou_threshold=iou,
                image_size=image_size,
                device=device,
            )
            alert_config = AlertConfig(confidence_threshold=confidence)
            tracker = SimpleTracker(TrackerConfig(), alert_config.window_seconds)
            alert_manager = AlertManager(alert_config)
            ui = OpenCVUI(UIConfig(show_window=False, draw_all_tracks=True, draw_status_bar=False))
        except Exception as exc:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        fps_meter = StreamFPSMeter()
        frame_index = 0
        failures = 0
        preview_interval = 1.0 / preview_fps
        detection_interval = 1.0 / detect_fps
        last_frame_at = 0.0
        last_detection_at = 0.0
        last_tracks = []
        last_alert = alert_manager.update([], time.monotonic())
        voice_source_id = uuid.uuid4().hex
        stop_reason = "completed"
        event_logger = LiveEventLogger(
            PROJECT_ROOT / "data_store" / "detection_results" / "live_events",
            source_label=source.label,
            source_kind=source.kind,
            source_id=camera_id or source.label,
            model_path=model_path,
            settings={
                "confidence": confidence,
                "iou": iou,
                "image_size": image_size,
                "device": device or "auto",
                "frame_skip": frame_skip,
                "preview_fps": preview_fps,
                "detect_fps": detect_fps,
                "max_width": max_width,
                "max_height": max_height,
                "jpeg_quality": jpeg_quality,
                "read_failure_limit": read_failure_limit,
                "reconnect_attempts": reconnect_attempts,
                "reconnect_delay": reconnect_delay,
                "presence_out_seconds": presence_out_seconds,
                "recording_enabled": record_enabled,
                "recording_labels": record_labels,
                "recording_max_mb": round(record_max_bytes / 1024 / 1024, 1),
                "camera_profile": camera_profile if camera_id else "",
            },
            client_run_id=client_run_id,
        )
        event_logger.log_start()
        presence_state = DronePresenceState(out_seconds=presence_out_seconds)
        detector_state: dict[str, object] = {"detector": None, "error": ""}

        def load_detector_for_stream() -> None:
            started_at = time.monotonic()
            try:
                detector_state["detector"] = shared_detector_for(self.server, detector_config, DroneDetector)
                event_logger.log_detector_ready(time.monotonic() - started_at)
            except Exception as exc:
                detector_state["error"] = str(exc)
                event_logger.log_error(f"detector unavailable: {exc}")

        threading.Thread(target=load_detector_for_stream, daemon=True).start()
        last_detection_event_at = 0.0
        recorder: StreamRecorder | None = None
        recording_confirmed = False
        if record_enabled:
            if source.is_image:
                event_logger.log_recording_skipped("image source does not need stream recording")
            else:
                try:
                    suffix_parts = []
                    if record_name_suffix:
                        suffix_parts.append(safe_name(record_name_suffix))
                    if record_labels:
                        suffix_parts.append("labeled")
                    recorder = StreamRecorder(
                        record_dir,
                        preview_fps,
                        record_max_bytes,
                        record_rollover_bytes,
                        name_suffix=f"_{'_'.join(suffix_parts)}" if suffix_parts else "",
                    )
                except Exception as exc:
                    event_logger.log_recording_failed(f"recording disabled: {exc}")
        capture_worker = None
        latest_frame_token = 0
        use_capture_worker = source.kind in {"camera", "rtsp", "stream"}
        if use_capture_worker:
            capture_worker = LatestFrameCapture(
                source,
                cap,
                open_source_capture,
                event_logger,
                read_failure_limit,
                reconnect_attempts,
                reconnect_delay,
            )
            capture_worker.start()
        try:
            while True:
                if capture_worker is not None:
                    latest_frame_token, frame, frame_index, capture_stop_reason = capture_worker.latest_after(latest_frame_token)
                    if frame is None:
                        if capture_stop_reason:
                            stop_reason = capture_stop_reason
                            break
                        time.sleep(0.01)
                        continue
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        stop_reason = "source_exhausted"
                        break
                    failures = 0
                    frame_index += 1

                now = time.monotonic()
                wait = preview_interval - (now - last_frame_at)
                if wait > 0:
                    time.sleep(wait)
                last_frame_at = time.monotonic()

                frame = resize_frame(frame, max_width, max_height)
                detector = detector_state.get("detector")
                detector_status = ""
                detection_ran = False
                if detector is None:
                    tracks = last_tracks
                    alert = last_alert
                    detector_status = " | detector error" if detector_state.get("error") else " | loading detector"
                else:
                    detection_due = last_detection_at <= 0 or time.monotonic() - last_detection_at >= detection_interval
                    if frame_skip and frame_index % (frame_skip + 1) != 1:
                        detection_due = False
                    if detection_due:
                        detections = detector.detect(frame)
                        detection_ran = True
                        last_detection_at = time.monotonic()
                        last_tracks = tracker.update(detections, last_detection_at)
                        last_alert = alert_manager.update(last_tracks, last_detection_at)
                        self.server.voice_warning.update(last_alert.active, last_detection_at, voice_source_id)
                    tracks = last_tracks
                    alert = last_alert
                fps = fps_meter.update()
                annotated = ui.draw(frame, tracks, alert, fps, f"{source.label}{detector_status}")
                if recorder is not None:
                    try:
                        recorder.write(annotated if record_labels else frame)
                        if not recording_confirmed:
                            recording_confirmed = True
                            event_logger.log_recording_started(
                                record_dir,
                                record_max_bytes,
                                record_labels,
                                recorder.current_path,
                            )
                    except Exception as exc:
                        event_logger.log_recording_failed(f"recording disabled: {exc}")
                        recorder.close()
                        recorder = None

                ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if not ok:
                    continue
                payload = encoded.tobytes()
                event_tracks = alert.tracks if alert.active and alert.tracks else tracks
                event_now = time.monotonic()
                if detection_ran:
                    # Presence episodes should follow persisted alerts, not one-frame raw detections.
                    presence_tracks = alert.tracks if alert.active else []
                    transition = presence_state.update(presence_tracks, frame_index, event_now)
                    if transition is not None:
                        event_logger.log_presence_transition(transition)
                if detection_ran and event_tracks and event_logger.should_log_detection(last_detection_event_at, event_now):
                    last_detection_event_at = event_now
                    event_logger.log_detection(event_tracks, frame_index, fps, payload)
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

                if source.is_image:
                    stop_reason = "image_completed"
                    break
        except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
            stop_reason = "client_disconnected"
        except Exception as exc:
            stop_reason = f"error:{type(exc).__name__}"
            event_logger.log_error(str(exc))
        finally:
            if recorder is not None:
                for segment in recorder.close():
                    event_logger.log_recording_saved(segment["path"], int(segment["size_bytes"]), record_labels)
            self.server.voice_warning.release_source(voice_source_id, time.monotonic())
            if capture_worker is not None:
                capture_worker.close()
            else:
                cap.release()
            transition = presence_state.close(frame_index, time.monotonic(), stop_reason)
            if transition is not None:
                event_logger.log_presence_transition(transition)
            event_logger.log_stop(stop_reason, frame_index)

    def send_static(self, path: Path, content_type: str | None = None) -> None:
        try:
            resolved = path.resolve()
            resolved.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return

        if not resolved.exists() or not resolved.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        data = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or guess_content_type(resolved))
        self.send_header("Content-Length", str(len(data)))
        self.send_no_cache_headers()
        self.end_headers()
        self.wfile.write(data)

    def send_media(self, media_path_value: str) -> None:
        media_path = resolve_user_path(unquote(media_path_value))
        if not media_path.exists() or not media_path.is_file() or media_path.suffix.lower() not in MEDIA_EXTENSIONS:
            self.send_error(HTTPStatus.NOT_FOUND, "Media not found")
            return

        file_size = media_path.stat().st_size
        content_type = guess_content_type(media_path)
        range_header = self.headers.get("Range")
        if not range_header:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            stream_file(media_path, self.wfile, 0, file_size - 1)
            return

        start, end = parse_range_header(range_header, file_size)
        if start is None or end is None:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return

        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        stream_file(media_path, self.wfile, start, end)

    def read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_no_cache_headers()
        self.end_headers()
        self.wfile.write(data)

    def send_no_cache_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def require_auth(self) -> bool:
        if not self.server.auth_enabled:
            return True

        header = self.headers.get("Authorization", "")
        prefix = "Basic "
        if not header.startswith(prefix):
            self.send_auth_required()
            return False

        try:
            decoded = base64.b64decode(header[len(prefix) :], validate=True).decode("utf-8")
            username, password = decoded.split(":", 1)
        except (binascii.Error, ValueError, UnicodeDecodeError):
            self.send_auth_required()
            return False

        user_ok = hmac.compare_digest(username, self.server.auth_username)
        pass_ok = hmac.compare_digest(password, self.server.auth_password)
        if not (user_ok and pass_ok):
            self.send_auth_required()
            return False
        return True

    def send_auth_required(self) -> None:
        body = b"Authentication required\n"
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("WWW-Authenticate", 'Basic realm="Drone Annotation"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def resolve_user_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_auth_password(args: argparse.Namespace) -> str:
    if args.no_auth:
        return ""
    if args.password:
        return args.password
    if args.password_env:
        return os.environ.get(args.password_env, "")
    return ""


def local_display_host(host: str) -> str:
    if host == "0.0.0.0":
        return "localhost"
    if host == "::":
        return "[::1]"
    return host


def initialise_local_camera_cache(server: AnnotationServer) -> None:
    if server.local_camera_cache.exists():
        print(f"Local camera cache: {server.local_camera_cache}")
        return

    print("Local camera cache missing. Background local camera scan started.")
    server.local_camera_scan_running = True
    thread = threading.Thread(target=startup_local_camera_scan, args=(server,), daemon=True)
    thread.start()


def startup_local_camera_scan(server: AnnotationServer) -> None:
    try:
        payload = scan_and_save_local_camera_cache(server.local_camera_cache, max_index=5)
    except Exception as exc:
        print(f"WARNING: local camera startup scan failed: {exc}")
    else:
        print(f"Local camera cache saved: {len(payload.get('cameras', []))} cameras")
    finally:
        server.local_camera_scan_running = False


def load_or_scan_local_cameras(
    cache_path: Path,
    max_index: int,
    refresh: bool = False,
    scan_running: bool = False,
) -> dict[str, object]:
    if scan_running:
        cached = read_local_camera_cache(cache_path)
        if cached is not None:
            cached["source"] = "cache"
            cached["cache_path"] = str(cache_path)
            cached["scan_running"] = True
            return cached
        return {
            "source": "scanning",
            "cache_path": str(cache_path),
            "max_index": max_index,
            "cameras": [],
            "scan_running": True,
            "message": "Local camera discovery is running in the background.",
        }

    if refresh:
        return scan_and_save_local_camera_cache(cache_path, max_index=max_index)

    cached = read_local_camera_cache(cache_path)
    if cached is not None:
        cached["source"] = "cache"
        cached["cache_path"] = str(cache_path)
        return cached

    return {
        "source": "missing",
        "cache_path": str(cache_path),
        "max_index": max_index,
        "cameras": [],
        "message": "Local camera cache does not exist. Use Scan Local to refresh it.",
    }


def scan_and_save_local_camera_cache(cache_path: Path, max_index: int) -> dict[str, object]:
    from app.sources import scan_local_cameras

    cameras = scan_local_cameras(max_index)
    payload: dict[str, object] = {
        "source": "scan",
        "cache_path": str(cache_path),
        "generated_at": datetime.now().astimezone().isoformat(),
        "max_index": max_index,
        "cameras": cameras,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(cache_path)
    return payload


def read_local_camera_cache(cache_path: Path) -> dict[str, object] | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    cameras = payload.get("cameras")
    if not isinstance(cameras, list):
        payload["cameras"] = []
    return payload


def guess_content_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def media_kind_for_path(path: Path) -> str:
    return "video" if path.suffix.lower() in VIDEO_EXTENSIONS else "image"


def stream_file(path: Path, output, start: int, end: int) -> None:
    remaining = end - start + 1
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            output.write(chunk)
            remaining -= len(chunk)


def parse_range_header(header: str, file_size: int) -> tuple[int | None, int | None]:
    match = re.match(r"bytes=(\d*)-(\d*)$", header.strip())
    if not match:
        return None, None
    start_text, end_text = match.groups()
    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
    else:
        suffix_length = int(end_text)
        start = max(file_size - suffix_length, 0)
        end = file_size - 1
    if start > end or start >= file_size:
        return None, None
    return start, min(end, file_size - 1)


def decode_data_url(data_url: str) -> bytes:
    if "," not in data_url:
        return b""
    header, encoded = data_url.split(",", 1)
    if ";base64" not in header:
        return b""
    return base64.b64decode(encoded)


def normalise_boxes(raw_boxes: object, width: int, height: int) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    if not isinstance(raw_boxes, list):
        return boxes

    for item in raw_boxes:
        if not isinstance(item, dict):
            continue
        x1 = float(item.get("x1", 0))
        y1 = float(item.get("y1", 0))
        x2 = float(item.get("x2", 0))
        y2 = float(item.get("y2", 0))
        left = max(0.0, min(width - 1.0, min(x1, x2)))
        top = max(0.0, min(height - 1.0, min(y1, y2)))
        right = max(0.0, min(width - 1.0, max(x1, x2)))
        bottom = max(0.0, min(height - 1.0, max(y1, y2)))
        if right - left > 1 and bottom - top > 1:
            boxes.append((left, top, right, bottom))
    return boxes


def yolo_label_text(boxes: list[tuple[float, float, float, float]], width: int, height: int) -> str:
    lines = []
    for left, top, right, bottom in boxes:
        box_width = right - left
        box_height = bottom - top
        x_center = (left + right) / 2 / width
        y_center = (top + bottom) / 2 / height
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_width / width:.6f} {box_height / height:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def image_id_for(source_path: Path, media_kind: str, frame_time: object) -> str:
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    if media_kind == "video":
        try:
            millis = int(round(float(frame_time or 0) * 1000))
        except (TypeError, ValueError):
            millis = 0
        suffix = f"t{millis:09d}"
    else:
        suffix = "image"
    return f"{source_path.stem}_{digest}_{suffix}"


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "annotation"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def write_data_yaml(project_dir: Path, class_name: str) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    content = (
        f"path: {json.dumps(portable_path(project_dir))}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {class_name}\n"
    )
    (project_dir / "data.yaml").write_text(content, encoding="utf-8")


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_dashboard_stats(folder_value: str, project_dir_value: str) -> dict[str, object]:
    folder = resolve_user_path(folder_value)
    project_dir = resolve_user_path(project_dir_value)
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "folder": str(folder),
        "project_dir": str(project_dir),
        "raw": raw_data_stats(folder),
        "annotations": annotation_stats(project_dir),
    }


def raw_data_stats(folder: Path) -> dict[str, object]:
    stats: dict[str, object] = {
        "exists": folder.exists() and folder.is_dir(),
        "files": 0,
        "videos": 0,
        "images": 0,
    }
    if not stats["exists"]:
        return stats

    files = videos = images = 0
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            files += 1
            videos += 1
        elif suffix in IMAGE_EXTENSIONS:
            files += 1
            images += 1

    stats.update({"files": files, "videos": videos, "images": images})
    return stats


def annotation_stats(project_dir: Path) -> dict[str, object]:
    split_stats = {split: annotation_split_stats(project_dir, split) for split in ("train", "val")}
    total = {
        key: sum(int(split_stats[split][key]) for split in split_stats)
        for key in ("total", "positive", "negative", "boxes", "unlabeled")
    }
    return {
        "exists": project_dir.exists() and project_dir.is_dir(),
        "total": total,
        "splits": split_stats,
        "sources": source_frame_stats(project_dir / "manifest.csv"),
    }


def annotation_split_stats(project_dir: Path, split: str) -> dict[str, int]:
    image_dir = project_dir / "images" / split
    label_dir = project_dir / "labels" / split
    image_stems = {
        path.stem
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    label_paths = [path for path in label_dir.rglob("*.txt") if path.is_file()]
    label_stems = {path.stem for path in label_paths}

    positive = 0
    negative = 0
    boxes = 0
    for label_path in label_paths:
        label_boxes = count_yolo_boxes(label_path)
        boxes += label_boxes
        if label_boxes > 0:
            positive += 1
        else:
            negative += 1

    sample_stems = image_stems | label_stems
    return {
        "total": len(sample_stems),
        "positive": positive,
        "negative": negative,
        "boxes": boxes,
        "unlabeled": len(image_stems - label_stems),
    }


def count_yolo_boxes(label_path: Path) -> int:
    try:
        text = label_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    return sum(1 for line in text.splitlines() if line.strip())


def source_frame_stats(manifest_path: Path) -> list[dict[str, object]]:
    if not manifest_path.exists():
        return []

    sources: dict[str, dict[str, object]] = {}
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return []

    for row in rows:
        source_path = row.get("source_media", "")
        source_name = Path(source_path).name or "unknown"
        split = row.get("split") if row.get("split") in {"train", "val"} else "unknown"
        box_count = parse_int(row.get("box_count"))
        source = sources.setdefault(
            source_name,
            {
                "source": source_name,
                "media_kind": row.get("media_kind") or media_kind_for_path(Path(source_name)),
                "frames": 0,
                "train": 0,
                "val": 0,
                "positive": 0,
                "negative": 0,
                "boxes": 0,
            },
        )
        source["frames"] = int(source["frames"]) + 1
        if split in {"train", "val"}:
            source[split] = int(source[split]) + 1
        if box_count > 0:
            source["positive"] = int(source["positive"]) + 1
        else:
            source["negative"] = int(source["negative"]) + 1
        source["boxes"] = int(source["boxes"]) + box_count

    return sorted(sources.values(), key=lambda item: (-int(item["frames"]), str(item["source"]).lower()))


def read_recent_live_events(root_dir: Path, limit: int) -> list[dict[str, object]]:
    if not root_dir.exists() or not root_dir.is_dir():
        return []

    events: list[dict[str, object]] = []
    day_dirs = sorted((path for path in root_dir.iterdir() if path.is_dir()), reverse=True)
    for day_dir in day_dirs:
        event_path = day_dir / "events.jsonl"
        if not event_path.exists():
            continue
        try:
            lines = event_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_number, line in reversed(list(enumerate(lines, start=1))):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                event["event_id"] = f"{day_dir.name}:{line_number}"
                events.append(event)
            if len(events) >= limit:
                return events
    return events


def remove_live_event_rows(root_dir: Path, event_ids: list[str]) -> dict[str, object]:
    requested = {event_id for event_id in event_ids if event_id}
    if not requested:
        return {"removed": [], "failed": []}

    by_day: dict[str, set[int]] = {}
    failed: list[dict[str, str]] = []
    for event_id in requested:
        match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}):(\d+)", event_id)
        if not match:
            failed.append({"event_id": event_id, "error": "invalid event id"})
            continue
        by_day.setdefault(match.group(1), set()).add(int(match.group(2)))

    removed: list[str] = []
    removed_images: list[str] = []
    for day, line_numbers in by_day.items():
        event_path = root_dir / day / "events.jsonl"
        if not event_path.exists():
            failed.extend({"event_id": f"{day}:{line_number}", "error": "event log not found"} for line_number in line_numbers)
            continue
        try:
            lines = event_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            failed.extend({"event_id": f"{day}:{line_number}", "error": str(exc)} for line_number in line_numbers)
            continue

        kept: list[str] = []
        changed = False
        for line_number, line in enumerate(lines, start=1):
            if line_number not in line_numbers:
                kept.append(line)
                continue
            changed = True
            removed.append(f"{day}:{line_number}")
            maybe_remove_event_image(root_dir, line, removed_images)

        if not changed:
            continue
        try:
            event_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        except OSError as exc:
            failed.extend({"event_id": f"{day}:{line_number}", "error": str(exc)} for line_number in line_numbers)

    return {"removed": removed, "failed": failed, "removed_images": removed_images}


def maybe_remove_event_image(events_root: Path, line: str, removed_images: list[str]) -> None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return
    if not isinstance(event, dict):
        return
    image_value = str(event.get("image_path") or "")
    if not image_value:
        return
    image_path = resolve_user_path(image_value)
    try:
        image_path.resolve().relative_to(events_root.resolve())
    except (OSError, ValueError):
        return
    try:
        if image_path.exists() and image_path.is_file():
            image_path.unlink()
            removed_images.append(str(image_path))
    except OSError:
        return


def query_value(query: dict[str, list[str]], name: str, default: str) -> str:
    values = query.get(name)
    if not values:
        return default
    return str(values[0] or default)


def normalise_device_value(value: str) -> str:
    device = str(value or "").strip()
    return "" if device.lower() == "auto" else device


def parse_float(value: object, default: float) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def parse_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def parse_int(value: object) -> int:
    try:
        return int(float(str(value or 0)))
    except ValueError:
        return 0


def resize_frame(frame, max_width: int, max_height: int):
    if max_width <= 0 or max_height <= 0:
        return frame

    import cv2

    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


class LatestFrameCapture:
    def __init__(
        self,
        source: object,
        cap: object,
        open_capture: object,
        event_logger: object,
        read_failure_limit: int,
        reconnect_attempts: int,
        reconnect_delay: float,
    ) -> None:
        self.source = source
        self.cap = cap
        self.open_capture = open_capture
        self.event_logger = event_logger
        self.read_failure_limit = read_failure_limit
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.frame = None
        self.token = 0
        self.frames_seen = 0
        self.stop_reason = ""

    def start(self) -> None:
        self.thread.start()

    def latest_after(self, previous_token: int):
        with self.lock:
            if self.frame is None or self.token == previous_token:
                return self.token, None, self.frames_seen, self.stop_reason
            return self.token, self.frame.copy(), self.frames_seen, self.stop_reason

    def close(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        try:
            self.cap.release()
        except Exception:
            pass

    def _set_stop_reason(self, reason: str) -> None:
        with self.lock:
            self.stop_reason = reason

    def _set_frame(self, frame) -> None:
        with self.lock:
            self.frame = frame
            self.frames_seen += 1
            self.token += 1

    def _run(self) -> None:
        failures = 0
        reconnect_count = 0
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                failures = 0
                self._set_frame(frame)
                continue

            if getattr(self.source, "is_image", False) or getattr(self.source, "kind", "") == "video":
                self._set_stop_reason("source_exhausted")
                return

            failures += 1
            if failures <= self.read_failure_limit:
                time.sleep(0.05)
                continue

            reconnected = False
            while not self.stop_event.is_set() and reconnect_count < self.reconnect_attempts:
                reconnect_count += 1
                self.event_logger.log_source_reconnect("read_failed", reconnect_count, self.reconnect_attempts)
                try:
                    self.cap.release()
                except Exception:
                    pass
                time.sleep(self.reconnect_delay)
                new_cap = None
                try:
                    new_cap = self.open_capture(self.source)
                except Exception as exc:
                    self.event_logger.log_error(f"source reconnect failed: {exc}")
                if new_cap is not None and new_cap.isOpened():
                    self.cap = new_cap
                    failures = 0
                    self.event_logger.log_source_reconnected(reconnect_count)
                    reconnected = True
                    break
                if new_cap is not None:
                    try:
                        new_cap.release()
                    except Exception:
                        pass

            if not reconnected:
                self._set_stop_reason("read_failed")
                return


def start_training_process(server: AnnotationServer, payload: dict[str, object]) -> dict[str, object]:
    scope = str(payload.get("dataset_scope") or "since-last").strip()
    if scope not in {"all", "since-last", "date-range"}:
        raise ValueError("dataset_scope must be all, since-last, or date-range")

    with server.training_lock:
        existing = server.training_job
        if existing and existing.get("status") in {"starting", "running", "stopping"}:
            process = existing.get("process")
            if process is not None and getattr(process, "poll")() is None:
                raise RuntimeError("Training job is already running")

    project_dir = resolve_user_path(str(payload.get("project_dir") or server.default_project_dir))
    model_path = resolve_user_path(str(payload.get("model") or server.live_model))
    output_model = resolve_user_path(str(payload.get("output_model") or server.live_model))
    run_project = resolve_user_path(str(payload.get("run_project") or "data_store/models/trained/runs"))
    snapshot_root = resolve_user_path(str(payload.get("snapshot_root") or "data_store/datasets/training_snapshots"))
    metadata_path = output_model.with_suffix(".meta.json")
    epochs = clamp_int(payload.get("epochs"), 1, 300, 25)
    imgsz = clamp_int(payload.get("imgsz"), 256, 1536, 640)
    batch = clamp_int(payload.get("batch"), 1, 128, 8)
    workers = clamp_int(payload.get("workers"), 0, 16, 0)
    patience = clamp_int(payload.get("patience"), 1, 100, 8)
    device = normalise_device_value(str(payload.get("device") or ""))
    prepare_only = parse_bool(payload.get("prepare_only"))
    from_date = str(payload.get("from_date") or "").strip()
    to_date = str(payload.get("to_date") or "").strip()
    if scope == "date-range" and not (from_date or to_date):
        raise ValueError("date-range training requires from_date, to_date, or both")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    job_id = f"train_{stamp}_{uuid.uuid4().hex[:8]}"
    name = safe_name(str(payload.get("name") or f"yolov8n_drone_{scope}_{stamp}"))
    log_dir = PROJECT_ROOT / "data_store" / "models" / "trained" / "runs" / "_web_jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id}.log"

    command = [
        sys.executable,
        "scripts/train_yolov8n_drone.py",
        "--dataset-scope",
        scope,
        "--project-dir",
        portable_path(project_dir),
        "--model",
        portable_path(model_path),
        "--epochs",
        str(epochs),
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
        "--workers",
        str(workers),
        "--patience",
        str(patience),
        "--project",
        portable_path(run_project),
        "--name",
        name,
        "--output-model",
        portable_path(output_model),
        "--snapshot-root",
        portable_path(snapshot_root),
        "--last-training-metadata",
        portable_path(metadata_path),
    ]
    if device:
        command.extend(["--device", device])
    if from_date:
        command.extend(["--from-date", from_date])
    if to_date:
        command.extend(["--to-date", to_date])
    if prepare_only:
        command.append("--prepare-only")

    env = os.environ.copy()
    temp_root = Path(tempfile.gettempdir())
    env.setdefault("YOLO_CONFIG_DIR", str(temp_root / "ultralytics"))
    env.setdefault("MPLCONFIGDIR", str(temp_root / "matplotlib"))
    Path(env["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    with log_path.open("ab") as handle:
        handle.write(f"$ {subprocess.list2cmdline(command)}\n\n".encode("utf-8"))
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )

    job: dict[str, object] = {
        "id": job_id,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "ended_at": "",
        "returncode": None,
        "stop_requested": False,
        "dataset_scope": scope,
        "prepare_only": prepare_only,
        "from_date": from_date,
        "to_date": to_date,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device or "auto",
        "project_dir": portable_path(project_dir),
        "model": portable_path(model_path),
        "output_model": portable_path(output_model),
        "run_project": portable_path(run_project),
        "snapshot_root": portable_path(snapshot_root),
        "metadata_path": portable_path(metadata_path),
        "name": name,
        "log_path": portable_path(log_path),
        "command": command,
        "process": process,
    }
    with server.training_lock:
        server.training_job = job

    thread = threading.Thread(target=watch_training_process, args=(server, job_id), daemon=True)
    thread.start()
    return training_status_payload(server)


def watch_training_process(server: AnnotationServer, job_id: str) -> None:
    with server.training_lock:
        job = server.training_job if server.training_job and server.training_job.get("id") == job_id else None
        process = job.get("process") if job else None
    if process is None:
        return

    returncode = process.wait()
    with server.training_lock:
        job = server.training_job if server.training_job and server.training_job.get("id") == job_id else None
        if job is None:
            return
        finish_training_job_locked(job, returncode)


def stop_training_process(server: AnnotationServer) -> dict[str, object]:
    with server.training_lock:
        job = server.training_job
        if not job:
            return training_status_payload_from_job(None)
        process = job.get("process")
        if job.get("status") not in {"running", "starting"} or process is None or getattr(process, "poll")() is not None:
            return training_status_payload_from_job(job)
        job["status"] = "stopping"
        job["stop_requested"] = True
        try:
            getattr(process, "terminate")()
        except OSError as exc:
            job["status"] = "error"
            job["error"] = str(exc)
        return training_status_payload_from_job(job)


def start_diagnostics_process(server: AnnotationServer, payload: dict[str, object]) -> dict[str, object]:
    mode = str(payload.get("mode") or "quick").strip().lower()
    if mode not in {"quick", "sysdump"}:
        raise ValueError("mode must be quick or sysdump")

    with server.diagnostics_lock:
        existing = server.diagnostics_job
        if existing and existing.get("status") == "running":
            raise RuntimeError("Diagnostics job is already running")

        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        job_id = f"diag_{stamp}_{uuid.uuid4().hex[:8]}"
        job: dict[str, object] = {
            "id": job_id,
            "status": "running",
            "result_status": "",
            "started_at": datetime.now().astimezone().isoformat(),
            "ended_at": "",
            "mode": mode,
            "camera_id": str(payload.get("camera_id") or "").strip(),
            "camera_profile": str(payload.get("camera_profile") or "main").strip().lower() or "main",
            "camera_seconds": max(0.5, parse_float(payload.get("camera_seconds"), 3.0)),
            "include_camera": parse_bool(payload.get("include_camera", True)),
            "include_performance": parse_bool(payload.get("include_performance", False)),
            "refresh_stats": parse_bool(payload.get("refresh_stats", False)),
            "privacy": str(payload.get("privacy") or "normal").strip().lower() or "normal",
            "checks": [],
            "error": "",
            "report": "",
            "archive_path": "",
            "report_path": "",
            "run_id": "",
            "sysdump_id": "",
        }
        server.diagnostics_job = job

    thread = threading.Thread(target=run_diagnostics_job, args=(server, job_id), daemon=True)
    thread.start()
    return diagnostics_status_payload(server)


def run_diagnostics_job(server: AnnotationServer, job_id: str) -> None:
    with server.diagnostics_lock:
        job = server.diagnostics_job if server.diagnostics_job and server.diagnostics_job.get("id") == job_id else None
        if job is None:
            return
        options_payload = dict(job)

    try:
        from app.diagnostics import DiagnosticsOptions, run_diagnostics

        result = run_diagnostics(
            DiagnosticsOptions(
                mode=str(options_payload.get("mode") or "quick"),
                camera_id=str(options_payload.get("camera_id") or ""),
                camera_profile=str(options_payload.get("camera_profile") or "main"),
                camera_seconds=float(options_payload.get("camera_seconds") or 3.0),
                include_camera=bool(options_payload.get("include_camera")),
                include_performance=bool(options_payload.get("include_performance")),
                refresh_stats=bool(options_payload.get("refresh_stats")),
                privacy=str(options_payload.get("privacy") or "normal"),
                server_context=diagnostics_server_context(server),
            )
        )
        report_text = result.report_path.read_text(encoding="utf-8") if result.report_path.exists() else ""
        with server.diagnostics_lock:
            job = server.diagnostics_job if server.diagnostics_job and server.diagnostics_job.get("id") == job_id else None
            if job is None:
                return
            job.update(
                {
                    "status": "completed",
                    "result_status": result.status,
                    "ended_at": datetime.now().astimezone().isoformat(),
                    "checks": result.checks,
                    "archive_path": portable_path(result.archive_path),
                    "report_path": portable_path(result.report_path),
                    "run_id": result.run_id,
                    "sysdump_id": result.sysdump_id,
                    "report": report_text,
                }
            )
    except Exception as exc:
        with server.diagnostics_lock:
            job = server.diagnostics_job if server.diagnostics_job and server.diagnostics_job.get("id") == job_id else None
            if job is not None:
                job.update(
                    {
                        "status": "failed",
                        "result_status": "fail",
                        "ended_at": datetime.now().astimezone().isoformat(),
                        "error": str(exc),
                    }
                )


def diagnostics_status_payload(server: AnnotationServer) -> dict[str, object]:
    with server.diagnostics_lock:
        job = dict(server.diagnostics_job or {})
    if not job:
        return {"status": "idle", "checks": [], "report": ""}
    job.pop("thread", None)
    run_id = str(job.get("run_id") or "")
    if run_id:
        job["downloads"] = {
            "report": f"/api/diagnostics/download?run_id={run_id}&artifact=report",
            "sysdump": f"/api/diagnostics/download?run_id={run_id}&artifact=sysdump",
            "json": f"/api/diagnostics/download?run_id={run_id}&artifact=json",
        }
    return job


def diagnostics_server_context(server: AnnotationServer) -> dict[str, object]:
    return {
        "default_folder": str(server.default_folder),
        "default_project_dir": str(server.default_project_dir),
        "camera_config": str(server.camera_config),
        "live_model": str(server.live_model),
        "class_name": server.class_name,
        "auth_enabled": server.auth_enabled,
        "training": training_status_payload(server),
        "debug_session_active": bool(server.debug_session_id),
    }


def start_debug_session(server: AnnotationServer, payload: dict[str, object]) -> dict[str, object]:
    with server.debug_session_lock:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        session_id = f"debug_{stamp}_{uuid.uuid4().hex[:8]}"
        path = PROJECT_ROOT / "data_store" / "detection_results" / "debug_sessions" / f"{session_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        server.debug_session_id = session_id
        server.debug_session_path = path
    append_debug_event(server, "debug_session_started", {"source": payload.get("source") or "ui"})
    return {"active": True, "session_id": session_id}


def record_debug_event(server: AnnotationServer, payload: dict[str, object]) -> dict[str, object]:
    session_id = str(payload.get("session_id") or "")
    with server.debug_session_lock:
        active_id = server.debug_session_id
    if not active_id:
        return {"active": False, "recorded": False}
    if session_id and session_id != active_id:
        return {"active": True, "recorded": False, "error": "debug session id mismatch", "session_id": active_id}
    event_type = safe_name(str(payload.get("event_type") or "ui_event"))
    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    append_debug_event(server, event_type, details)
    return {"active": True, "recorded": True, "session_id": active_id}


def stop_debug_session(server: AnnotationServer, payload: dict[str, object]) -> dict[str, object]:
    session_id = str(payload.get("session_id") or "")
    with server.debug_session_lock:
        active_id = server.debug_session_id
    if active_id and (not session_id or session_id == active_id):
        append_debug_event(server, "debug_session_stopped", {"source": payload.get("source") or "ui"})
        with server.debug_session_lock:
            server.debug_session_id = ""
            server.debug_session_path = None
        return {"active": False, "session_id": active_id}
    return {"active": bool(active_id), "session_id": active_id}


def append_debug_event(server: AnnotationServer, event_type: str, details: dict[str, object]) -> None:
    from app.diagnostics.redaction import redact_mapping

    with server.debug_session_lock:
        session_id = server.debug_session_id
        path = server.debug_session_path
    if not session_id or path is None:
        return
    row = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "session_id": session_id,
        "event_type": safe_name(event_type),
        "details": redact_mapping(details),
    }
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        return


def training_status_payload(server: AnnotationServer) -> dict[str, object]:
    with server.training_lock:
        job = server.training_job
        if job and job.get("status") in {"running", "starting", "stopping"}:
            process = job.get("process")
            if process is not None:
                returncode = getattr(process, "poll")()
                if returncode is not None:
                    finish_training_job_locked(job, returncode)
        return training_status_payload_from_job(job)


def training_status_payload_from_job(job: dict[str, object] | None) -> dict[str, object]:
    if not job:
        return {
            "job": None,
            "status": "idle",
            "running": False,
            "log": "",
            "progress": {"current": 0, "total": 0, "percent": 0},
        }

    public = {key: value for key, value in job.items() if key != "process"}
    log_path = resolve_user_path(str(job.get("log_path") or ""))
    log_tail = read_text_tail(log_path)
    progress = parse_training_progress(log_tail, int(job.get("epochs") or 0), str(job.get("status") or ""))
    started_at = parse_datetime_for_elapsed(str(job.get("started_at") or ""))
    ended_at = parse_datetime_for_elapsed(str(job.get("ended_at") or ""))
    if started_at:
        end = ended_at or datetime.now().astimezone()
        public["elapsed_seconds"] = round((end - started_at).total_seconds(), 1)
    return {
        "job": public,
        "status": public.get("status", "unknown"),
        "running": public.get("status") in {"starting", "running", "stopping"},
        "log": log_tail,
        "progress": progress,
    }


def finish_training_job_locked(job: dict[str, object], returncode: int) -> None:
    job["returncode"] = returncode
    job["ended_at"] = datetime.now().astimezone().isoformat()
    if job.get("stop_requested"):
        job["status"] = "stopped"
    elif returncode == 0:
        job["status"] = "completed"
    else:
        job["status"] = "failed"


def read_text_tail(path: Path, max_bytes: int = 32 * 1024) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def parse_training_progress(log_text: str, epochs: int, status: str) -> dict[str, object]:
    if status == "completed":
        return {"current": epochs, "total": epochs, "percent": 100 if epochs else 100}
    matches = list(re.finditer(r"(?:^|[\r\n])\s*(\d{1,4})/(\d{1,4})(?=\s)", log_text))
    if not matches:
        matches = list(re.finditer(r"Epoch\s+(\d{1,4})/(\d{1,4})", log_text, flags=re.IGNORECASE))
    if matches:
        current = int(matches[-1].group(1))
        total = int(matches[-1].group(2))
    else:
        current = 0
        total = epochs
    percent = round((current / total) * 100, 1) if total else 0
    return {"current": current, "total": total, "percent": min(100, max(0, percent))}


def parse_datetime_for_elapsed(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed


def clamp_int(value: object, minimum: int, maximum: int, default: int) -> int:
    parsed = parse_int(value)
    if parsed <= 0 and default > 0:
        parsed = default
    return min(maximum, max(minimum, parsed))


class StreamRecorder:
    def __init__(
        self,
        output_dir: Path,
        fps: float,
        max_bytes: int,
        rollover_bytes: int,
        name_suffix: str = "",
    ) -> None:
        self.output_dir = output_dir
        self.fps = max(1.0, min(float(fps or 5.0), 30.0))
        self.max_bytes = max(1, max_bytes)
        self.rollover_bytes = min(max(1, rollover_bytes), self.max_bytes)
        suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", name_suffix)
        self.stamp = f"{datetime.now().strftime('record_%d%m_%H-%M')}{suffix}"
        self.segment_index = 0
        self.writer = None
        self.current_path: Path | None = None
        self.current_size: tuple[int, int] | None = None
        self.frames_in_segment = 0
        self.segment_started_at = 0.0
        self.completed_paths: list[Path] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame) -> None:
        height, width = frame.shape[:2]
        frame_size = (width, height)
        if self.writer is None or self.current_size != frame_size or self.should_rollover_before_write():
            self.open_segment(frame_size)

        self.writer.write(frame)
        self.frames_in_segment += 1
        if self.current_path and self.current_path.exists() and self.current_path.stat().st_size >= self.rollover_bytes:
            self.release_current()

    def close(self) -> list[dict[str, object]]:
        self.release_current()
        segments: list[dict[str, object]] = []
        for path in self.completed_paths:
            if path.exists() and path.stat().st_size > 0:
                segments.append({"path": path, "size_bytes": path.stat().st_size})
        return segments

    def should_rollover_before_write(self) -> bool:
        if self.current_path is None or self.frames_in_segment <= 0:
            return False
        if time.monotonic() - self.segment_started_at >= 60.0:
            return True
        if not self.current_path.exists():
            return False
        current_bytes = self.current_path.stat().st_size
        if current_bytes >= self.rollover_bytes:
            return True
        average_frame_bytes = current_bytes / max(self.frames_in_segment, 1)
        if average_frame_bytes <= 0:
            return False
        return current_bytes + max(average_frame_bytes * 2, 512 * 1024) >= self.max_bytes

    def open_segment(self, frame_size: tuple[int, int]) -> None:
        self.release_current()

        self.segment_index += 1
        candidates = [
            (".mp4", "avc1"),
            (".mp4", "H264"),
            (".webm", "VP90"),
            (".webm", "VP80"),
            (".mp4", "mp4v"),
            (".avi", "MJPG"),
        ]
        path = None
        writer = None
        for suffix, codec in candidates:
            path, writer = self.try_open_writer(suffix, codec, frame_size)
            if writer is not None:
                break
        if path is None or writer is None:
            raise RuntimeError(f"unable to create recording file in {self.output_dir}")

        self.writer = writer
        self.current_path = path
        self.current_size = frame_size
        self.frames_in_segment = 0
        self.segment_started_at = time.monotonic()

    def try_open_writer(self, suffix: str, codec: str, frame_size: tuple[int, int]):
        import cv2

        path = self.next_segment_path(suffix)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), self.fps, frame_size)
        if writer.isOpened():
            return path, writer
        writer.release()
        if path.exists() and path.stat().st_size == 0:
            try:
                path.unlink()
            except OSError:
                pass
        return None, None

    def release_current(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.current_path is not None:
            if self.current_path.exists() and self.current_path.stat().st_size > 0:
                self.completed_paths.append(self.current_path)
            elif self.current_path.exists():
                try:
                    self.current_path.unlink()
                except OSError:
                    pass
        self.current_path = None
        self.current_size = None
        self.frames_in_segment = 0
        self.segment_started_at = 0.0

    def next_segment_path(self, suffix: str) -> Path:
        base = self.stamp if self.segment_index == 1 else f"{self.stamp}_{self.segment_index:02d}"
        candidate = self.output_dir / f"{base}{suffix}"
        collision = 2
        while candidate.exists():
            candidate = self.output_dir / f"{base}_{collision}{suffix}"
            collision += 1
        return candidate


@dataclass
class DronePresenceTransition:
    event_type: str
    episode_id: int
    frame_index: int
    entry_frame_index: int
    last_seen_frame_index: int
    best_track: object | None = None
    tracks: list[object] | None = None
    exit_frame_index: int | None = None
    duration_seconds: float | None = None
    absence_seconds: float | None = None
    reason: str = ""


class DronePresenceState:
    def __init__(self, out_seconds: float = DEFAULT_PRESENCE_OUT_SECONDS) -> None:
        self.out_seconds = out_seconds
        self.in_frame = False
        self.episode_id = 0
        self.entry_frame_index = 0
        self.entry_at = 0.0
        self.last_seen_frame_index = 0
        self.last_seen_at = 0.0
        self.last_tracks: list[object] = []

    def update(self, tracks: list[object], frame_index: int, now: float) -> DronePresenceTransition | None:
        if tracks:
            self.last_tracks = list(tracks)
            best = max(tracks, key=lambda track: getattr(track, "confidence", 0.0))
            if not self.in_frame:
                self.episode_id += 1
                self.in_frame = True
                self.entry_frame_index = frame_index
                self.entry_at = now
                self.last_seen_frame_index = frame_index
                self.last_seen_at = now
                return DronePresenceTransition(
                    event_type="drone_in_frame",
                    episode_id=self.episode_id,
                    frame_index=frame_index,
                    entry_frame_index=frame_index,
                    last_seen_frame_index=frame_index,
                    best_track=best,
                    tracks=list(tracks),
                )

            self.last_seen_frame_index = frame_index
            self.last_seen_at = now
            return None

        if self.in_frame and self.last_seen_at > 0 and now - self.last_seen_at >= self.out_seconds:
            return self._exit(frame_index, now, reason="absence_timeout")
        return None

    def close(self, frame_index: int, now: float, reason: str) -> DronePresenceTransition | None:
        if not self.in_frame:
            return None
        return self._exit(frame_index, now, reason=reason or "stream_closed")

    def _exit(self, frame_index: int, now: float, reason: str) -> DronePresenceTransition:
        transition = DronePresenceTransition(
            event_type="drone_out_frame",
            episode_id=self.episode_id,
            frame_index=frame_index,
            entry_frame_index=self.entry_frame_index,
            last_seen_frame_index=self.last_seen_frame_index,
            exit_frame_index=frame_index,
            duration_seconds=round(max(0.0, self.last_seen_at - self.entry_at), 3),
            absence_seconds=round(max(0.0, now - self.last_seen_at), 3),
            reason=reason,
            tracks=list(self.last_tracks),
        )
        self.in_frame = False
        self.entry_frame_index = 0
        self.entry_at = 0.0
        self.last_seen_frame_index = 0
        self.last_seen_at = 0.0
        self.last_tracks = []
        return transition


class LiveEventLogger:
    def __init__(
        self,
        root_dir: Path,
        source_label: str,
        source_kind: str,
        source_id: str,
        model_path: Path,
        settings: dict[str, object],
        client_run_id: str = "",
    ) -> None:
        now = datetime.now().astimezone()
        self.root_dir = root_dir
        self.day_dir = root_dir / now.strftime("%Y-%m-%d")
        self.session_id = f"{now.strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.frame_dir = self.day_dir / "frames" / self.session_id
        self.event_path = self.day_dir / "events.jsonl"
        self.source_label = source_label
        self.source_kind = source_kind
        self.source_id = source_id
        self.model_path = model_path
        self.settings = settings
        self.client_run_id = client_run_id
        self.started_at = time.monotonic()
        self.detection_cooldown_seconds = 5.0
        self.detection_count = 0
        self.lock = threading.Lock()

    def log_start(self) -> None:
        self._append(
            {
                "event_type": "start",
                "model_path": portable_path(self.model_path),
                "settings": self.settings,
            }
        )

    def log_stop(self, reason: str, frame_index: int) -> None:
        self._append(
            {
                "event_type": "stop",
                "reason": reason,
                "frames_seen": frame_index,
                "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
                "detection_events": self.detection_count,
            }
        )

    def log_error(self, message: str) -> None:
        self._append({"event_type": "error", "message": message})

    def log_detector_ready(self, elapsed_seconds: float) -> None:
        self._append(
            {
                "event_type": "detector_ready",
                "message": f"detector ready in {elapsed_seconds:.2f}s",
                "elapsed_seconds": round(elapsed_seconds, 3),
            }
        )

    def log_recording_started(
        self,
        output_dir: Path,
        max_bytes: int,
        labels: bool = False,
        path: Path | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "event_type": "recording_started",
            "message": f"recording to {portable_path(output_dir)}",
            "recording_dir": portable_path(output_dir),
            "max_size_mb": round(max_bytes / 1024 / 1024, 1),
            "labels": labels,
        }
        if path is not None:
            payload["recording_path"] = portable_path(path)
        self._append(payload)

    def log_recording_failed(self, reason: str) -> None:
        self._append({"event_type": "recording_failed", "message": reason, "reason": reason})

    def log_recording_saved(self, path: Path, size_bytes: int, labels: bool = False) -> None:
        self._append(
            {
                "event_type": "recording_saved",
                "message": f"saved {portable_path(path)}",
                "recording_path": portable_path(path),
                "size_bytes": size_bytes,
                "labels": labels,
            }
        )

    def log_recording_skipped(self, reason: str) -> None:
        self._append({"event_type": "recording_skipped", "message": reason})

    def log_source_reconnect(self, reason: str, attempt: int, max_attempts: int) -> None:
        self._append(
            {
                "event_type": "source_reconnect",
                "message": f"{reason}; reconnect attempt {attempt}/{max_attempts}",
                "reason": reason,
                "attempt": attempt,
                "max_attempts": max_attempts,
            }
        )

    def log_source_reconnected(self, attempt: int) -> None:
        self._append(
            {
                "event_type": "source_reconnected",
                "message": f"source reconnected on attempt {attempt}",
                "attempt": attempt,
            }
        )

    def should_log_detection(self, last_logged_at: float, now: float) -> bool:
        return last_logged_at <= 0 or now - last_logged_at >= self.detection_cooldown_seconds

    def log_presence_transition(self, transition: DronePresenceTransition) -> None:
        payload: dict[str, object] = {
            "event_type": transition.event_type,
            "episode_id": transition.episode_id,
            "frame_index": transition.frame_index,
            "entry_frame_index": transition.entry_frame_index,
            "last_seen_frame_index": transition.last_seen_frame_index,
            "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
        }
        if transition.exit_frame_index is not None:
            payload["exit_frame_index"] = transition.exit_frame_index
        if transition.duration_seconds is not None:
            payload["duration_seconds"] = transition.duration_seconds
        if transition.absence_seconds is not None:
            payload["absence_seconds"] = transition.absence_seconds
        if transition.reason:
            payload["reason"] = transition.reason
        if transition.best_track is not None:
            payload["best_track"] = serialise_track(transition.best_track)
        if transition.tracks:
            payload["tracks"] = [serialise_track(track) for track in transition.tracks]
        self._append(payload)

    def log_detection(self, tracks: list[object], frame_index: int, fps: float, jpeg_bytes: bytes) -> None:
        self.detection_count += 1
        event_time = datetime.now().astimezone()
        image_path = self.frame_dir / f"{event_time.strftime('%H%M%S_%f')}_{self.detection_count:04d}.jpg"
        try:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(jpeg_bytes)
        except OSError as exc:
            self._append({"event_type": "error", "message": f"failed to save detection frame: {exc}"})
            return

        best = max(tracks, key=lambda track: getattr(track, "confidence", 0.0))
        self._append(
            {
                "event_type": "drone_detected",
                "frame_index": frame_index,
                "fps": round(fps, 2),
                "image_path": portable_path(image_path),
                "best_track": serialise_track(best),
                "tracks": [serialise_track(track) for track in tracks],
            },
            event_time,
        )

    def _append(self, payload: dict[str, object], event_time: datetime | None = None) -> None:
        event_time = event_time or datetime.now().astimezone()
        row = {
            "timestamp": event_time.isoformat(),
            "session_id": self.session_id,
            "source": self.source_label,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            **payload,
        }
        if self.client_run_id:
            row["client_run_id"] = self.client_run_id
        try:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock:
                with self.event_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError:
            return


def serialise_track(track: object) -> dict[str, object]:
    return {
        "track_id": int(getattr(track, "track_id", 0)),
        "label": str(getattr(track, "label", "")),
        "class_id": int(getattr(track, "class_id", 0)),
        "confidence": round(float(getattr(track, "confidence", 0.0)), 4),
        "bbox": [int(value) for value in getattr(track, "bbox", ())],
        "seen_frames": int(getattr(track, "seen_frames", 0)),
        "recent_hits": int(getattr(track, "recent_hits", 0)),
    }


class StreamFPSMeter:
    def __init__(self) -> None:
        self._last = time.monotonic()
        self._fps = 0.0

    def update(self) -> float:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        if elapsed <= 0:
            return self._fps

        instant = 1.0 / elapsed
        self._fps = instant if self._fps == 0 else (self._fps * 0.85) + (instant * 0.15)
        return self._fps


def upsert_manifest(path: Path, row: dict[str, str]) -> bool:
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    replaced = False
    for index, existing in enumerate(rows):
        if existing.get("image_id") == row["image_id"]:
            rows[index] = {field: row.get(field, existing.get(field, "")) for field in MANIFEST_FIELDS}
            replaced = True
            break
    if not replaced:
        rows.append({field: row.get(field, "") for field in MANIFEST_FIELDS})

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return replaced


if __name__ == "__main__":
    raise SystemExit(main())
