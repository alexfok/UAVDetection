from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from app.diagnostics.redaction import redact_mapping, redact_text
from app.diagnostics.report import render_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data_store" / "detection_results" / "sysdumps"
SECRET_ENV_KEYS = {"ANNOTATION_SERVER_PASSWORD", "UAV_CAMERA_USER", "UAV_CAMERA_PASSWORD"}


@dataclass
class DiagnosticsOptions:
    mode: str = "quick"
    output_root: Path = DEFAULT_OUTPUT_ROOT
    camera_id: str = ""
    camera_profile: str = "main"
    camera_seconds: float = 3.0
    include_camera: bool = True
    include_performance: bool = False
    include_logs: bool = True
    refresh_stats: bool = False
    privacy: str = "normal"
    server_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticsResult:
    run_id: str
    sysdump_id: str
    status: str
    output_dir: Path
    archive_path: Path
    report_path: Path
    checks: list[dict[str, Any]]
    manifest: dict[str, Any]
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("output_dir", "archive_path", "report_path"):
            data[key] = str(data[key])
        return data


def run_diagnostics(options: DiagnosticsOptions | None = None) -> DiagnosticsResult:
    options = options or DiagnosticsOptions()
    now = datetime.now().astimezone()
    sysdump_id = now.strftime("sysdump_%d%m%y:%H%M")
    run_id = now.strftime("sysdump_%d%m%y_%H%M%S")
    output_dir = options.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "schema": 1,
        "run_id": run_id,
        "sysdump_id": sysdump_id,
        "generated_at": now.isoformat(),
        "mode": options.mode,
        "host": socket.gethostname(),
        "project_root": str(PROJECT_ROOT),
        "privacy": options.privacy,
        "included_raw_media": False,
        "included_model_weights": False,
    }

    write_json(output_dir / "manifest.json", manifest)
    collect_system(output_dir, checks, options)
    collect_repo(output_dir, checks, options)
    collect_configs(output_dir, checks, options)
    collect_datastore(output_dir, checks, options)
    collect_server_context(output_dir, checks, options)
    collect_recent_events(output_dir, checks, options)
    collect_debug_sessions(output_dir, checks, options)
    if options.include_logs:
        collect_logs(output_dir, checks, options)
    if options.include_camera:
        collect_camera(output_dir, checks, options)
    if options.include_performance:
        collect_performance_placeholder(output_dir, checks)

    write_json(output_dir / "checks.json", checks)
    report_path = output_dir / f"{run_id}_report.md"
    archive_path = output_dir.with_suffix(".tar.gz")
    manifest["report_path"] = str(report_path)
    manifest["archive_path"] = str(archive_path)
    write_json(output_dir / "manifest.json", manifest)
    report_path.write_text(render_report(manifest, checks), encoding="utf-8")

    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_dir, arcname=run_id)

    status = "fail" if any(check["status"] == "fail" for check in checks) else "warn" if any(
        check["status"] == "warn" for check in checks
    ) else "pass"
    return DiagnosticsResult(run_id, sysdump_id, status, output_dir, archive_path, report_path, checks, manifest)


def collect_system(output_dir: Path, checks: list[dict[str, Any]], options: DiagnosticsOptions) -> None:
    packages = {}
    for package in ("cv2", "numpy", "yaml", "torch", "ultralytics"):
        packages[package] = package_version(package)
    system = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "executable": os.sys.executable,
        "packages": packages,
        "disk": disk_summary(PROJECT_ROOT),
        "env_keys": {key: {"present": bool(os.environ.get(key))} for key in sorted(SECRET_ENV_KEYS)},
    }
    write_json(output_dir / "system.json", system)
    add_check(checks, "system", "python", "pass", f"Python {system['python']} at {system['executable']}")
    disk = system["disk"]
    free_gb = float(disk.get("free_gb", 0.0))
    add_check(
        checks,
        "system",
        "disk_space",
        "warn" if free_gb < 5 else "pass",
        f"{free_gb:.1f} GB free on {disk.get('path')}",
    )


def collect_repo(output_dir: Path, checks: list[dict[str, Any]], options: DiagnosticsOptions) -> None:
    info = {
        "commit": run_text(["git", "rev-parse", "--short", "HEAD"]),
        "branch": run_text(["git", "branch", "--show-current"]),
        "status_short": run_text(["git", "status", "--short"]),
    }
    write_json(output_dir / "repo.json", info)
    dirty = bool(info["status_short"].strip())
    add_check(checks, "repo", "git_status", "warn" if dirty else "pass", "working tree has local changes" if dirty else "working tree clean")


def collect_configs(output_dir: Path, checks: list[dict[str, Any]], options: DiagnosticsOptions) -> None:
    config_dir = output_dir / "config"
    config_dir.mkdir(exist_ok=True)
    paths = [
        PROJECT_ROOT / "configs" / "config.yaml",
        PROJECT_ROOT / "data_store" / "system_config" / "cameras.yaml",
        PROJECT_ROOT / "data_store" / "system_config" / "local_cameras.json",
        PROJECT_ROOT / "data_store" / "system_config" / "annotation_server.env",
    ]
    for path in paths:
        if not path.exists():
            continue
        target = config_dir / path.name
        if path.suffix in {".yaml", ".yml"}:
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                write_json(target.with_suffix(target.suffix + ".redacted.json"), redact_mapping(data, options.privacy))
            except Exception as exc:
                add_check(checks, "config", path.name, "warn", f"could not parse: {exc}")
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            target.with_suffix(target.suffix + ".redacted.txt").write_text(redact_text(text, options.privacy), encoding="utf-8")

    camera_config = PROJECT_ROOT / "data_store" / "system_config" / "cameras.yaml"
    add_check(
        checks,
        "config",
        "camera_config",
        "pass" if camera_config.exists() else "fail",
        str(camera_config) if camera_config.exists() else "camera config missing",
    )


def collect_datastore(output_dir: Path, checks: list[dict[str, Any]], options: DiagnosticsOptions) -> None:
    data_store = PROJECT_ROOT / "data_store"
    if options.refresh_stats:
        try:
            from scripts.datastore_sync import write_stats

            write_stats(data_store)
            add_check(checks, "datastore", "refresh_stats", "pass", "dataset stats refreshed")
        except Exception as exc:
            add_check(checks, "datastore", "refresh_stats", "warn", f"could not refresh stats: {exc}")
    required = ["raw_data", "detection_results", "datasets", "models", "system_config", "stats"]
    summary = {}
    for name in required:
        path = data_store / name
        summary[name] = {"exists": path.exists(), "path": str(path)}
    model = data_store / "models" / "trained" / "yolov8n_drone_best.pt"
    summary["default_model"] = {
        "exists": model.exists(),
        "path": str(model),
        "size_bytes": model.stat().st_size if model.exists() else 0,
    }
    stats = data_store / "stats" / "dataset_stats.json"
    if stats.exists():
        try:
            summary["dataset_stats"] = json.loads(stats.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary["dataset_stats"] = {"error": "invalid json"}
    write_json(output_dir / "datastore.json", summary)
    missing = [name for name in required if not summary[name]["exists"]]
    add_check(
        checks,
        "datastore",
        "required_folders",
        "fail" if missing else "pass",
        f"missing: {', '.join(missing)}" if missing else "all required data_store folders exist",
    )
    add_check(
        checks,
        "datastore",
        "default_model",
        "fail" if not model.exists() else "pass",
        str(model) if model.exists() else "default trained model missing",
    )


def collect_server_context(output_dir: Path, checks: list[dict[str, Any]], options: DiagnosticsOptions) -> None:
    if not options.server_context:
        add_check(checks, "server", "server_context", "skip", "not run from annotation server")
        return
    context = redact_mapping(options.server_context, options.privacy)
    write_json(output_dir / "server_context.json", context)
    add_check(checks, "server", "server_context", "pass", "annotation server context captured")


def collect_recent_events(output_dir: Path, checks: list[dict[str, Any]], options: DiagnosticsOptions) -> None:
    events_root = PROJECT_ROOT / "data_store" / "detection_results" / "live_events"
    recent: list[dict[str, Any]] = []
    if events_root.exists():
        for event_file in sorted(events_root.rglob("events.jsonl"), reverse=True)[:5]:
            for line in tail_lines(event_file, 80):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    event.pop("image_path", None)
                    recent.append(redact_mapping(event, options.privacy))
    write_json(output_dir / "recent_live_events.json", recent[-200:])
    errors = [event for event in recent if str(event.get("event_type")) in {"error", "recording_failed"}]
    add_check(checks, "logs", "recent_live_errors", "warn" if errors else "pass", f"{len(errors)} recent error events")


def collect_debug_sessions(output_dir: Path, checks: list[dict[str, Any]], options: DiagnosticsOptions) -> None:
    sessions_root = PROJECT_ROOT / "data_store" / "detection_results" / "debug_sessions"
    recent: list[dict[str, Any]] = []
    if sessions_root.exists():
        for event_file in sorted(sessions_root.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)[:3]:
            for line in tail_lines(event_file, 120):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    recent.append(redact_mapping(event, options.privacy))
    write_json(output_dir / "debug_activity.json", recent[-300:])
    add_check(checks, "activity", "debug_session_events", "pass" if recent else "skip", f"{len(recent)} recent debug activity events")


def collect_logs(output_dir: Path, checks: list[dict[str, Any]], options: DiagnosticsOptions) -> None:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    candidates: list[Path] = []
    candidates.extend((PROJECT_ROOT / "data_store" / "system_config" / "logs").glob("*.log"))
    candidates.extend((PROJECT_ROOT / "data_store" / "models" / "trained" / "runs" / "_web_jobs").glob("*.log"))
    for root in (
        PROJECT_ROOT / "data_store" / "detection_results" / "performance",
        PROJECT_ROOT / "data_store" / "detection_results" / "profile",
        PROJECT_ROOT / "data_store" / "detection_results" / "profiles",
    ):
        candidates.extend(find_named_logs(root, ("perf", "profile", "log")))
    candidates = sorted({path for path in candidates if path.exists() and path.is_file()}, key=lambda path: path.stat().st_mtime, reverse=True)[:12]

    index = []
    for path in candidates:
        try:
            stat = path.stat()
            target = logs_dir / f"{safe_filename(path.stem)}{path.suffix}.tail.txt"
            target.write_text(redact_text("\n".join(tail_lines(path, 400)), options.privacy), encoding="utf-8")
            index.append(
                {
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
                    "tail_file": target.name,
                }
            )
        except OSError as exc:
            index.append({"path": str(path), "error": str(exc)})
    write_json(output_dir / "logs.json", index)
    add_check(checks, "logs", "recent_log_tails", "pass" if index else "skip", f"{len(index)} log tails captured")


def collect_camera(output_dir: Path, checks: list[dict[str, Any]], options: DiagnosticsOptions) -> None:
    if not options.camera_id:
        add_check(checks, "camera", "camera_selected", "skip", "no camera selected")
        return
    try:
        from app.sources import load_camera_entries, resolve_camera

        camera_config = PROJECT_ROOT / "data_store" / "system_config" / "cameras.yaml"
        entries = load_camera_entries(camera_config)
        entry = entries.get(options.camera_id)
        if entry is None:
            add_check(checks, "camera", "camera_selected", "fail", f"camera not found: {options.camera_id}")
            return
        source = resolve_camera(options.camera_id, camera_config, profile=options.camera_profile)
        camera_summary = {
            "camera_id": options.camera_id,
            "profile": options.camera_profile,
            "label": source.label,
            "address": entry.address,
            "rtsp_port": entry.rtsp_port,
            "rtsp_path": entry.rtsp_path,
            "preview_rtsp_path": entry.preview_rtsp_path,
            "enabled": entry.enabled,
            "capture_source": redact_text(str(source.capture_source), options.privacy),
        }
        port_status = check_tcp(entry.address, entry.rtsp_port)
        camera_summary["tcp"] = port_status
        add_check(checks, "camera", "camera_reachability", "pass" if port_status["ok"] else "fail", port_status["detail"])
        frame_status = read_camera_frames(source.capture_source, options.camera_seconds)
        camera_summary["frame_read"] = frame_status
        add_check(checks, "camera", "camera_frame_read", "pass" if frame_status["ok"] else "fail", frame_status["detail"])
        write_json(output_dir / "camera.json", camera_summary)
    except Exception as exc:
        add_check(checks, "camera", "camera_probe", "fail", str(exc))


def collect_performance_placeholder(output_dir: Path, checks: list[dict[str, Any]]) -> None:
    write_json(output_dir / "performance.json", {"status": "not_implemented", "detail": "reserved for profile_live_performance integration"})
    add_check(checks, "performance", "performance_probe", "skip", "performance probe is reserved for a later diagnostics phase")


def package_version(package: str) -> str:
    try:
        module = __import__(package)
    except Exception:
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


def disk_summary(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_gb": round(usage.total / 1024 / 1024 / 1024, 2),
        "used_gb": round(usage.used / 1024 / 1024 / 1024, 2),
        "free_gb": round(usage.free / 1024 / 1024 / 1024, 2),
    }


def check_tcp(host: str, port: int) -> dict[str, Any]:
    if not host:
        return {"ok": False, "detail": "camera has no address"}
    started = time.monotonic()
    try:
        with socket.create_connection((host, int(port)), timeout=2.0):
            elapsed = (time.monotonic() - started) * 1000
            return {"ok": True, "detail": f"{host}:{port} reachable in {elapsed:.0f} ms"}
    except OSError as exc:
        return {"ok": False, "detail": f"{host}:{port} unreachable: {exc}"}


def read_camera_frames(capture_source: str, seconds: float) -> dict[str, Any]:
    try:
        import cv2
    except Exception as exc:
        return {"ok": False, "detail": f"OpenCV unavailable: {exc}"}
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        return {"ok": False, "detail": "OpenCV could not open stream"}
    started = time.monotonic()
    frames = 0
    shape = None
    try:
        while time.monotonic() - started < max(0.5, seconds):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames += 1
            shape = list(frame.shape[:2])
    finally:
        cap.release()
    elapsed = max(time.monotonic() - started, 0.001)
    fps = frames / elapsed
    detail = f"read {frames} frames in {elapsed:.1f}s ({fps:.1f} FPS)"
    if shape:
        detail += f", shape {shape[1]}x{shape[0]}"
    return {"ok": frames > 0, "detail": detail, "frames": frames, "elapsed_seconds": round(elapsed, 3), "fps": round(fps, 2), "shape": shape}


def run_text(command: list[str]) -> str:
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False, timeout=8)
    except Exception as exc:
        return str(exc)
    return (result.stdout or result.stderr).strip()


def tail_lines(path: Path, limit: int) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_check(checks: list[dict[str, Any]], category: str, name: str, status: str, detail: str) -> None:
    checks.append({"category": category, "name": name, "status": status, "detail": detail})


def find_named_logs(root: Path, tokens: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    matches: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".log", ".txt", ".json", ".jsonl", ".csv"}:
            continue
        lowered = path.name.lower()
        if any(token in lowered for token in tokens):
            matches.append(path)
    return matches


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)[:80] or "log"
