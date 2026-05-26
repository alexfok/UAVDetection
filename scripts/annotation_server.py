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
import ssl
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = PROJECT_ROOT / "web" / "annotator"
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
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
    parser.add_argument("--default-folder", type=Path, default=Path("videos/Roni/raw_data"))
    parser.add_argument("--project-dir", type=Path, default=Path("annotations/web_drone_v1"))
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = AnnotationServer((args.host, args.port), AnnotationHandler)
    server.default_folder = resolve_user_path(args.default_folder)
    server.default_project_dir = resolve_user_path(args.project_dir)
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
    server.serve_forever()
    return 0


class AnnotationServer(ThreadingHTTPServer):
    default_folder: Path
    default_project_dir: Path
    class_name: str
    auth_enabled: bool
    auth_username: str
    auth_password: str


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
                }
            )
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

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def scan_folder(self, folder_value: str) -> None:
        folder = resolve_user_path(folder_value)
        if not folder.exists() or not folder.is_dir():
            self.send_json({"error": f"Folder not found: {folder}"}, status=HTTPStatus.BAD_REQUEST)
            return

        media = []
        for path in sorted(folder.rglob("*")):
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
        upsert_manifest(
            project_dir / "manifest.csv",
            {
                "image_id": safe_id,
                "split": split,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "source_media": str(source_path),
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
                "image_id": safe_id,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "box_count": len(boxes),
                "project_dir": str(project_dir),
            }
        )

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


def write_data_yaml(project_dir: Path, class_name: str) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    content = (
        f"path: {project_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {class_name}\n"
    )
    (project_dir / "data.yaml").write_text(content, encoding="utf-8")


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


def parse_int(value: object) -> int:
    try:
        return int(float(str(value or 0)))
    except ValueError:
        return 0


def upsert_manifest(path: Path, row: dict[str, str]) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())
