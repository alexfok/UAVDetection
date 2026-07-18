from __future__ import annotations

import argparse
import json
import os
import platform
import plistlib
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTALL_DIR = Path.home() / "UAVDetection"
DEFAULT_SERVICE_NAME = "uav-detection.service"
DEFAULT_VENV = ""
JETSON_VENV = ".venv_cuda"
DEFAULT_FALLBACK_VENV = ".venv"
DEFAULT_PORT = 8765
LAUNCHD_LABEL = "com.uavdetection.annotation-server"
WINDOWS_TASK_NAME = "UAVDetection Annotation Server"
SERVICE_FILE_ROOT = Path("/etc/systemd/system")
SHELL_USERNAME_ARG = '"${ANNOTATION_SERVER_USERNAME:-admin}"'
REQUIRED_SOURCE_PATHS = (
    "app",
    "assets/audio/drone_warning.wav",
    "assets/audio/drone_all_clear.wav",
    "configs",
    "scripts/annotation_server.py",
    "scripts/datastore_sync.py",
    "web/annotator/index.html",
    "web/annotator/app.js",
    "requirements.txt",
)
DATASTORE_DIRS = (
    "raw_data",
    "detection_results",
    "datasets",
    "models",
    "models/base",
    "models/external",
    "models/trained",
    "system_config",
    "system_config/certs",
    "stats",
    "backups",
    "deployment_patches",
)
COPY_EXCLUDED_NAMES = {
    ".git",
    ".venv",
    ".venv_cuda",
    "__pycache__",
    ".pytest_cache",
    ".DS_Store",
    "data_store",
    "annotations",
    "certs",
    "models",
    "reports",
    "runs",
    "videos",
    "yolov8n.pt",
    "yolo26n.pt",
}
COPY_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


@dataclass
class Check:
    name: str
    status: str
    detail: str


@dataclass
class DeployContext:
    action: str
    source_dir: Path
    install_dir: Path
    service_name: str
    service_mode: str
    venv: str
    port: int
    skip_deps: bool
    no_service: bool
    allow_existing: bool = False
    target: str = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage UAVDetection install, upgrade, and uninstall.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("preflight", "install", "upgrade", "uninstall", "status"):
        add_common_args(subparsers.add_parser(name, help=f"{name} deployment"))

    install = subparsers.choices["install"]
    install.add_argument("--replace-existing", action="store_true", help="Move an existing install aside before clean install.")
    install.add_argument("--include-source-data-store", action="store_true", help="Copy source data_store into a clean install.")

    uninstall = subparsers.choices["uninstall"]
    uninstall.add_argument("--yes", action="store_true", help="Confirm uninstall file changes.")
    uninstall.add_argument("--delete-data", action="store_true", help="Delete data_store instead of preserving it beside the install.")

    for name in ("install", "upgrade"):
        command = subparsers.choices[name]
        command.add_argument("--allow-online", action="store_true", help="Allow pip to use package indexes if dependencies are installed.")
        command.add_argument("--skip-smoke", action="store_true", help="Skip post-copy Python/HTML smoke checks.")
        command.add_argument("--no-https", action="store_true", help="Configure the service without HTTPS cert/key arguments.")
        command.add_argument("--username", default=os.environ.get("ANNOTATION_SERVER_USERNAME", "admin"))
        command.add_argument("--password", default=os.environ.get("ANNOTATION_SERVER_PASSWORD", "admin123"))
        command.add_argument("--host", default="0.0.0.0")
        command.add_argument("--default-folder", default="data_store/raw_data/Roni")
        command.add_argument("--project-dir", default="data_store/datasets/web_drone_v1")
        command.add_argument("--camera-config", default="data_store/system_config/cameras.yaml")
        command.add_argument("--live-model", default="data_store/models/trained/yolov8n_drone_best.pt")

    return parser.parse_args()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT, help="Project source tree to deploy from.")
    parser.add_argument("--install-dir", type=Path, default=DEFAULT_INSTALL_DIR, help="Target install directory.")
    parser.add_argument("--target", choices=("auto", "jetson", "linux", "macos", "windows"), default="auto")
    parser.add_argument("--service-name", default="", help="Service/task name. Defaults depend on the target platform.")
    parser.add_argument("--service-mode", choices=("auto", "system", "user", "launchd", "windows-task", "none"), default="auto")
    parser.add_argument("--venv", default=DEFAULT_VENV, help="Preferred virtualenv under install dir. Defaults to .venv, or .venv_cuda on Jetson.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--skip-deps", action="store_true", help="Do not create/update virtualenv dependencies.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip the prerequisite check.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing files/services.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable preflight/status output.")


def main() -> int:
    args = parse_args()
    context = context_from_args(args)

    if args.command == "preflight":
        return run_preflight_command(context, json_output=args.json)
    if args.command == "status":
        return run_status_command(context, json_output=args.json)

    if not args.skip_preflight:
        checks = collect_preflight_checks(context)
        print_checks(checks)
        if has_failures(checks):
            print("Preflight failed; deployment stopped before making changes.")
            return 2

    if args.command == "install":
        return run_install(args, context)
    if args.command == "upgrade":
        return run_upgrade(args, context)
    if args.command == "uninstall":
        return run_uninstall(args, context)
    raise SystemExit(f"Unknown command: {args.command}")


def context_from_args(args: argparse.Namespace) -> DeployContext:
    target = resolve_target(args.target)
    service_mode = resolve_service_mode(args.service_mode, target)
    no_service = service_mode == "none"
    service_name = args.service_name or default_service_name(service_mode)
    venv = args.venv or default_venv(target)
    return DeployContext(
        action=args.command,
        source_dir=args.source_dir.expanduser().resolve(),
        install_dir=args.install_dir.expanduser().resolve(),
        service_name=service_name,
        service_mode=service_mode,
        venv=venv,
        port=args.port,
        skip_deps=args.skip_deps,
        no_service=no_service,
        allow_existing=bool(getattr(args, "replace_existing", False)),
        target=target,
    )


def resolve_target(target: str) -> str:
    if target != "auto":
        return target
    system = platform.system()
    if system == "Darwin":
        return "macos"
    if system == "Windows":
        return "windows"
    if system == "Linux":
        machine = platform.machine().lower()
        return "jetson" if machine in {"aarch64", "arm64"} else "linux"
    return "linux"


def resolve_service_mode(service_mode: str, target: str) -> str:
    if service_mode != "auto":
        return service_mode
    if target == "macos":
        return "launchd"
    if target == "windows":
        return "windows-task"
    if target == "jetson":
        return "system"
    return "user"


def default_venv(target: str) -> str:
    return JETSON_VENV if target == "jetson" else DEFAULT_FALLBACK_VENV


def default_service_name(service_mode: str) -> str:
    if service_mode == "launchd":
        return LAUNCHD_LABEL
    if service_mode == "windows-task":
        return WINDOWS_TASK_NAME
    return DEFAULT_SERVICE_NAME


def run_preflight_command(context: DeployContext, json_output: bool) -> int:
    checks = collect_preflight_checks(context)
    if json_output:
        print(json.dumps([asdict(check) for check in checks], indent=2) + "\n")
    else:
        print_checks(checks)
    return 1 if has_failures(checks) else 0


def run_status_command(context: DeployContext, json_output: bool) -> int:
    payload = {
        "install_dir": str(context.install_dir),
        "install_exists": context.install_dir.exists(),
        "data_store_exists": (context.install_dir / "data_store").exists(),
        "service_name": context.service_name,
        "service_active": service_active(context),
        "service_file": str(service_file_path(context)),
        "service_file_exists": service_file_path(context).exists(),
        "preferred_python": str(preferred_python(context.install_dir, context.venv)),
    }
    if json_output:
        print(json.dumps(payload, indent=2) + "\n")
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def collect_preflight_checks(context: DeployContext) -> list[Check]:
    checks: list[Check] = []
    add_platform_checks(checks)
    add_target_check(checks, context)
    add_source_checks(checks, context)
    add_install_checks(checks, context)
    add_runtime_checks(checks, context)
    add_service_checks(checks, context)
    add_network_checks(checks, context)
    return checks


def add_platform_checks(checks: list[Check]) -> None:
    system = platform.system()
    machine = platform.machine()
    supported = system in {"Linux", "Darwin", "Windows"}
    checks.append(Check("platform", "pass" if supported else "fail", f"{system} {machine}"))
    version = sys.version_info
    checks.append(
        Check(
            "python_version",
            "pass" if version >= (3, 9) else "fail",
            f"{version.major}.{version.minor}.{version.micro} at {sys.executable}",
        )
    )
    checks.append(Check("python_venv_module", "pass" if python_has_venv() else "fail", "python3 -m venv available"))


def add_target_check(checks: list[Check], context: DeployContext) -> None:
    system = platform.system()
    machine = platform.machine().lower()
    expected = {
        "macos": system == "Darwin",
        "windows": system == "Windows",
        "linux": system == "Linux",
        "jetson": system == "Linux" and machine in {"aarch64", "arm64"},
    }.get(context.target, True)
    checks.append(Check("target", "pass" if expected else "fail", f"{context.target} on {system} {machine}"))


def add_source_checks(checks: list[Check], context: DeployContext) -> None:
    if context.action not in {"preflight", "install", "upgrade"}:
        return
    checks.append(Check("source_dir", "pass" if context.source_dir.exists() else "fail", str(context.source_dir)))
    for relative in REQUIRED_SOURCE_PATHS:
        path = context.source_dir / relative
        checks.append(Check(f"source:{relative}", "pass" if path.exists() else "fail", str(path)))
    dirty = git_status_short(context.source_dir)
    if dirty is None:
        checks.append(Check("source_git", "warn", "source is not a git checkout"))
    else:
        checks.append(Check("source_git", "warn" if dirty else "pass", "working tree has local changes" if dirty else "working tree clean"))


def add_install_checks(checks: list[Check], context: DeployContext) -> None:
    parent = context.install_dir.parent
    if context.action == "install":
        if context.install_dir.exists() and not context.allow_existing:
            checks.append(Check("install_dir_available", "fail", f"{context.install_dir} already exists; use --replace-existing"))
        else:
            checks.append(Check("install_dir_available", "pass", str(context.install_dir)))
        checks.append(Check("install_parent_writable", "pass" if os.access(parent, os.W_OK) else "fail", str(parent)))
    elif context.action in {"upgrade", "uninstall", "status", "preflight"}:
        checks.append(Check("install_dir_exists", "pass" if context.install_dir.exists() else "fail", str(context.install_dir)))
    if context.install_dir.exists():
        data_store = context.install_dir / "data_store"
        checks.append(Check("data_store", "pass" if data_store.exists() else "warn", str(data_store) if data_store.exists() else "data_store missing"))
        model = data_store / "models/trained/yolov8n_drone_best.pt"
        checks.append(Check("default_model", "pass" if model.exists() else "warn", str(model) if model.exists() else "default model missing"))


def add_runtime_checks(checks: list[Check], context: DeployContext) -> None:
    if context.action not in {"upgrade", "uninstall", "status", "preflight"} or not context.install_dir.exists():
        return
    preferred = preferred_python(context.install_dir, context.venv)
    if preferred.exists():
        checks.append(Check("venv_python", "pass", str(preferred)))
    elif (fallback := context.install_dir / DEFAULT_FALLBACK_VENV / bin_dir() / python_executable_name()).exists():
        checks.append(Check("venv_python", "warn", f"preferred missing; fallback exists: {fallback}"))
    elif context.no_service:
        checks.append(Check("venv_python", "warn", f"{preferred} missing; service disabled, smoke checks can use {sys.executable}"))
    elif context.skip_deps:
        checks.append(Check("venv_python", "fail", f"{preferred} missing and --skip-deps was set"))
    else:
        checks.append(Check("venv_python", "warn", f"{preferred} missing; install/upgrade will create it"))


def add_service_checks(checks: list[Check], context: DeployContext) -> None:
    if context.no_service:
        checks.append(Check("service_mode", "skip", "service operations disabled"))
        return
    checks.append(Check("service_mode", "pass", context.service_mode))
    if context.service_mode in {"system", "user"}:
        systemctl = shutil.which("systemctl")
        checks.append(Check("systemctl", "pass" if systemctl else "fail", systemctl or "systemctl not found"))
    elif context.service_mode == "launchd":
        launchctl = shutil.which("launchctl")
        checks.append(Check("launchctl", "pass" if launchctl else "fail", launchctl or "launchctl not found"))
    elif context.service_mode == "windows-task":
        schtasks = shutil.which("schtasks")
        checks.append(Check("schtasks", "pass" if schtasks else "fail", schtasks or "schtasks not found"))
    if context.service_mode == "system":
        sudo = shutil.which("sudo")
        checks.append(Check("sudo", "pass" if sudo else "fail", sudo or "sudo not found"))
        sudo_ready = command_ok(["sudo", "-n", "true"]) if sudo else False
        checks.append(
            Check(
                "sudo_noninteractive",
                "pass" if sudo_ready else "warn",
                "sudo can run without a password" if sudo_ready else "service install/restart may require an interactive sudo password",
            )
        )
    path = service_file_path(context)
    exists = path.exists()
    if context.action in {"upgrade", "uninstall", "status", "preflight"}:
        checks.append(Check("service_file", "pass" if exists else "warn", str(path) if exists else f"{path} missing"))
        active = service_active(context)
        checks.append(Check("service_active", "pass" if active else "warn", "active" if active else "inactive or unavailable"))


def add_network_checks(checks: list[Check], context: DeployContext) -> None:
    if context.action == "install" and port_open("127.0.0.1", context.port):
        checks.append(Check("port_available", "warn", f"127.0.0.1:{context.port} is already accepting connections"))
    else:
        checks.append(Check("port_check", "pass", f"checked 127.0.0.1:{context.port}"))


def run_install(args: argparse.Namespace, context: DeployContext) -> int:
    if context.install_dir.exists():
        if not args.replace_existing:
            print(f"Install directory exists: {context.install_dir}")
            return 2
        backup = context.install_dir.with_name(f"{context.install_dir.name}.backup_{timestamp()}")
        print_action(args.dry_run, f"move existing install {context.install_dir} -> {backup}")
        if not args.dry_run:
            shutil.move(str(context.install_dir), str(backup))

    copy_source_tree(context.source_dir, context.install_dir, dry_run=args.dry_run, include_data_store=args.include_source_data_store)
    prepare_install_runtime(args, context)
    maybe_install_service(args, context)
    maybe_run_smoke(args, context)
    print(f"Clean install complete: {context.install_dir}")
    return 0


def run_upgrade(args: argparse.Namespace, context: DeployContext) -> int:
    if not context.install_dir.exists():
        print(f"Install directory missing: {context.install_dir}")
        return 2
    copy_source_tree(context.source_dir, context.install_dir, dry_run=args.dry_run, include_data_store=False)
    prepare_install_runtime(args, context)
    maybe_install_service(args, context)
    maybe_run_smoke(args, context)
    if not context.no_service:
        restart_service(context, args.dry_run)
    print(f"Upgrade complete: {context.install_dir}")
    return 0


def run_uninstall(args: argparse.Namespace, context: DeployContext) -> int:
    if not args.yes:
        print("Uninstall requires --yes. By default data_store is preserved beside the install directory.")
        return 2
    if not context.no_service:
        stop_disable_service(context, args.dry_run)
    uninstall_files(context.install_dir, delete_data=args.delete_data, dry_run=args.dry_run)
    print(f"Uninstall complete: {context.install_dir}")
    return 0


def prepare_install_runtime(args: argparse.Namespace, context: DeployContext) -> None:
    if args.dry_run:
        print_action(True, "prepare data_store layout, env, certs, and dependencies")
        return
    ensure_datastore_layout(context.install_dir / "data_store")
    ensure_server_env(context.install_dir, args.username, args.password)
    if not args.no_https:
        ensure_certificate(context.install_dir)
    if not args.skip_deps:
        ensure_dependencies(context.install_dir, context.venv, allow_online=args.allow_online)


def copy_source_tree(source: Path, destination: Path, dry_run: bool = False, include_data_store: bool = False) -> None:
    destination.mkdir(parents=True, exist_ok=True) if not dry_run else None
    for item in source.iterdir():
        if should_skip_copy(item, include_data_store=include_data_store):
            continue
        target = destination / item.name
        print_action(dry_run, f"copy {item.relative_to(source)} -> {target}")
        if dry_run:
            continue
        if item.is_symlink():
            copy_symlink_target(item, target)
        elif item.is_dir():
            if target.exists() and not target.is_dir():
                target.unlink()
            shutil.copytree(item, target, ignore=copy_ignore(include_data_store), dirs_exist_ok=True, copy_function=safe_copy_file)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            safe_copy_file(item, target)


def should_skip_copy(path: Path, include_data_store: bool) -> bool:
    if path.name == "data_store" and include_data_store:
        return False
    if path.name in COPY_EXCLUDED_NAMES:
        return True
    if path.suffix in COPY_EXCLUDED_SUFFIXES:
        return True
    return False


def copy_ignore(include_data_store: bool):
    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            path = Path(directory) / name
            if should_skip_copy(path, include_data_store=include_data_store):
                ignored.add(name)
        return ignored

    return ignore


def copy_symlink_target(source: Path, destination: Path) -> None:
    try:
        resolved = source.resolve(strict=True)
    except OSError:
        return
    if resolved.is_dir():
        shutil.copytree(resolved, destination, ignore=copy_ignore(False), dirs_exist_ok=True, copy_function=safe_copy_file)
    elif resolved.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        safe_copy_file(resolved, destination)


def safe_copy_file(source: Path | str, destination: Path | str) -> str:
    shutil.copyfile(source, destination)
    try:
        shutil.copymode(source, destination)
    except OSError:
        pass
    return str(destination)


def ensure_datastore_layout(data_store: Path) -> None:
    for relative in DATASTORE_DIRS:
        (data_store / relative).mkdir(parents=True, exist_ok=True)
    cameras = data_store / "system_config/cameras.yaml"
    if not cameras.exists():
        cameras.write_text("cameras: {}\n", encoding="utf-8")


def ensure_server_env(install_dir: Path, username: str, password: str) -> None:
    env_path = install_dir / "data_store/system_config/annotation_server.env"
    if env_path.exists():
        return
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        f"ANNOTATION_SERVER_USERNAME={username}\nANNOTATION_SERVER_PASSWORD={password}\n",
        encoding="utf-8",
    )
    try:
        env_path.chmod(0o600)
    except OSError:
        pass


def ensure_certificate(install_dir: Path) -> None:
    cert = install_dir / "data_store/system_config/certs/annotation.crt"
    key = install_dir / "data_store/system_config/certs/annotation.key"
    if cert.exists() and key.exists():
        return
    openssl = shutil.which("openssl")
    if not openssl:
        print("openssl not found; HTTPS certificate was not created")
        return
    cert.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "365",
            "-subj",
            "/CN=uav-detection-jetson",
        ],
        check=True,
        cwd=install_dir,
    )


def ensure_dependencies(install_dir: Path, venv_name: str, allow_online: bool) -> None:
    venv_dir = install_dir / venv_name
    venv_python = venv_dir / bin_dir() / python_executable_name()
    if not venv_python.exists():
        run([sys.executable, "-m", "venv", str(venv_dir)], check=True, cwd=install_dir)
    command = [str(venv_python), "-m", "pip", "install"]
    wheelhouse = install_dir / "wheelhouse"
    if wheelhouse.exists() and any(wheelhouse.iterdir()):
        command.extend(["--no-index", "--find-links", str(wheelhouse)])
    elif not allow_online:
        print("No wheelhouse found; dependency install skipped. Use --allow-online to install from package indexes.")
        return
    command.extend(["-r", str(install_dir / "requirements.txt")])
    run(command, check=True, cwd=install_dir)


def maybe_install_service(args: argparse.Namespace, context: DeployContext) -> None:
    if context.no_service:
        return
    if context.service_mode == "launchd":
        install_launchd_service(args, context)
        return
    if context.service_mode == "windows-task":
        install_windows_task(args, context)
        return
    if args.dry_run:
        print_action(True, f"write service file {service_file_path(context)}")
        return
    content = render_service_file(args, context)
    path = service_file_path(context)
    tmp = Path("/tmp") / f"{context.service_name}.{os.getpid()}.tmp"
    tmp.write_text(content, encoding="utf-8")
    if context.service_mode == "system":
        run(["sudo", "mv", str(tmp), str(path)], check=True)
        run(["sudo", "systemctl", "daemon-reload"], check=True)
        run(["sudo", "systemctl", "enable", context.service_name], check=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp), str(path))
        run(["systemctl", "--user", "daemon-reload"], check=False)
        run(["systemctl", "--user", "enable", context.service_name], check=False)


def install_launchd_service(args: argparse.Namespace, context: DeployContext) -> None:
    path = service_file_path(context)
    if args.dry_run:
        print_action(True, f"write launch agent {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        plistlib.dump(render_launchd_plist(args, context), handle)
    run(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)], check=False)
    run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=False)
    run(["launchctl", "enable", f"gui/{os.getuid()}/{launchd_label(context)}"], check=False)


def install_windows_task(args: argparse.Namespace, context: DeployContext) -> None:
    script = write_windows_start_script(args, context, dry_run=args.dry_run)
    if args.dry_run:
        print_action(True, f"create scheduled task {context.service_name}")
        return
    run(["schtasks", "/Create", "/TN", context.service_name, "/SC", "ONLOGON", "/TR", str(script), "/F"], check=False)


def render_service_file(args: argparse.Namespace, context: DeployContext) -> str:
    install_dir = context.install_dir
    shell_command = server_shell_command(args, context)
    user_lines = []
    if context.service_mode == "system":
        user_lines = [f"User={os.environ.get('USER', 'ubuntu')}", f"Group={os.environ.get('USER', 'ubuntu')}"]
    return "\n".join(
        [
            "[Unit]",
            "Description=UAVDetection web server",
            "After=network-online.target",
            "Wants=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            *user_lines,
            f"WorkingDirectory={install_dir}",
            f"EnvironmentFile={install_dir / 'data_store/system_config/annotation_server.env'}",
            "Environment=PYTHONUNBUFFERED=1",
            f"Environment=YOLO_CONFIG_DIR={install_dir / 'data_store/system_config/ultralytics'}",
            f"Environment=MPLCONFIGDIR={install_dir / 'data_store/system_config/matplotlib'}",
            f"ExecStart=/bin/bash -lc 'exec {shell_command}'",
            "Restart=always",
            "RestartSec=5",
            "",
            "[Install]",
            "WantedBy=multi-user.target" if context.service_mode == "system" else "WantedBy=default.target",
            "",
        ]
    )


def server_command_parts(args: argparse.Namespace, context: DeployContext) -> list[str]:
    install_dir = context.install_dir
    python = preferred_python(install_dir, context.venv)
    cert = install_dir / "data_store/system_config/certs/annotation.crt"
    key = install_dir / "data_store/system_config/certs/annotation.key"
    username_value = "%ANNOTATION_SERVER_USERNAME%" if context.service_mode == "windows-task" else SHELL_USERNAME_ARG
    command = [
        str(python),
        "scripts/annotation_server.py",
        "--host",
        args.host,
        "--port",
        str(context.port),
        "--username",
        username_value,
        "--password-env",
        "ANNOTATION_SERVER_PASSWORD",
        "--default-folder",
        args.default_folder,
        "--project-dir",
        args.project_dir,
        "--camera-config",
        args.camera_config,
        "--live-model",
        args.live_model,
    ]
    if not args.no_https:
        command.extend(["--certfile", str(cert.relative_to(install_dir)), "--keyfile", str(key.relative_to(install_dir))])
    return command


def server_shell_command(args: argparse.Namespace, context: DeployContext) -> str:
    return " ".join(shell_quote(part) for part in server_command_parts(args, context))


def shell_quote(part: str) -> str:
    if part == SHELL_USERNAME_ARG:
        return part
    return shlex.quote(part)


def render_launchd_plist(args: argparse.Namespace, context: DeployContext) -> dict[str, object]:
    logs_dir = context.install_dir / "data_store/system_config/logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env_path = context.install_dir / "data_store/system_config/annotation_server.env"
    command = (
        f"set -a; [ -f {shlex.quote(str(env_path))} ] && . {shlex.quote(str(env_path))}; set +a; "
        f"cd {shlex.quote(str(context.install_dir))}; "
        f"exec {server_shell_command(args, context)}"
    )
    return {
        "Label": launchd_label(context),
        "ProgramArguments": ["/bin/bash", "-lc", command],
        "WorkingDirectory": str(context.install_dir),
        "RunAtLoad": True,
        "KeepAlive": True,
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
            "YOLO_CONFIG_DIR": str(context.install_dir / "data_store/system_config/ultralytics"),
            "MPLCONFIGDIR": str(context.install_dir / "data_store/system_config/matplotlib"),
        },
        "StandardOutPath": str(logs_dir / "annotation_server.out.log"),
        "StandardErrorPath": str(logs_dir / "annotation_server.err.log"),
    }


def write_windows_start_script(args: argparse.Namespace, context: DeployContext, dry_run: bool) -> Path:
    script = service_file_path(context)
    command = " ".join(windows_quote(part) for part in server_command_parts(args, context))
    lines = [
        "@echo off",
        "setlocal EnableExtensions",
        f'cd /d "{context.install_dir}"',
        'set "ENV_FILE=data_store\\system_config\\annotation_server.env"',
        'if exist "%ENV_FILE%" (',
        '  for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (',
        '    if not "%%A"=="" if not "%%A:~0,1%"=="#" set "%%A=%%B"',
        "  )",
        ")",
        f'set "YOLO_CONFIG_DIR={context.install_dir / "data_store/system_config/ultralytics"}"',
        f'set "MPLCONFIGDIR={context.install_dir / "data_store/system_config/matplotlib"}"',
        "set PYTHONUNBUFFERED=1",
        command,
        "",
    ]
    if dry_run:
        print_action(True, f"write Windows start script {script}")
        return script
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("\r\n".join(lines), encoding="utf-8")
    return script


def windows_quote(value: str) -> str:
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'


def maybe_run_smoke(args: argparse.Namespace, context: DeployContext) -> None:
    if args.skip_smoke:
        return
    python = preferred_python(context.install_dir, context.venv)
    if not python.exists():
        python = Path(sys.executable)
    commands = [
        [str(python), "-m", "py_compile", "scripts/annotation_server.py", "app/sources.py"],
        [str(python), "-m", "html.parser", "web/annotator/index.html"],
    ]
    for command in commands:
        if args.dry_run:
            print_action(True, " ".join(command))
        else:
            run(command, check=True, cwd=context.install_dir)


def restart_service(context: DeployContext, dry_run: bool) -> None:
    if context.service_mode == "launchd":
        command = ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{launchd_label(context)}"]
        print_action(dry_run, " ".join(command))
        if not dry_run:
            run(command, check=False)
        return
    if context.service_mode == "windows-task":
        for command in (["schtasks", "/End", "/TN", context.service_name], ["schtasks", "/Run", "/TN", context.service_name]):
            print_action(dry_run, " ".join(command))
            if not dry_run:
                run(command, check=False)
        return
    command = service_command(context, "restart")
    print_action(dry_run, " ".join(command))
    if not dry_run:
        run(command, check=True)


def stop_disable_service(context: DeployContext, dry_run: bool) -> None:
    if context.service_mode == "launchd":
        path = service_file_path(context)
        for command in (
            ["launchctl", "bootout", f"gui/{os.getuid()}", str(path)],
            ["launchctl", "disable", f"gui/{os.getuid()}/{launchd_label(context)}"],
        ):
            print_action(dry_run, " ".join(command))
            if not dry_run:
                run(command, check=False)
        print_action(dry_run, f"remove launch agent {path}")
        if not dry_run and path.exists():
            path.unlink()
        return
    if context.service_mode == "windows-task":
        for command in (["schtasks", "/End", "/TN", context.service_name], ["schtasks", "/Delete", "/TN", context.service_name, "/F"]):
            print_action(dry_run, " ".join(command))
            if not dry_run:
                run(command, check=False)
        path = service_file_path(context)
        print_action(dry_run, f"remove Windows start script {path}")
        if not dry_run and path.exists():
            path.unlink()
        return
    for action in ("stop", "disable"):
        command = service_command(context, action)
        print_action(dry_run, " ".join(command))
        if not dry_run:
            run(command, check=False)
    path = service_file_path(context)
    print_action(dry_run, f"remove service file {path}")
    if not dry_run and path.exists():
        if context.service_mode == "system":
            run(["sudo", "rm", "-f", str(path)], check=False)
            run(["sudo", "systemctl", "daemon-reload"], check=False)
        else:
            path.unlink()
            run(["systemctl", "--user", "daemon-reload"], check=False)


def uninstall_files(install_dir: Path, delete_data: bool, dry_run: bool) -> None:
    if not install_dir.exists():
        print(f"Install directory already absent: {install_dir}")
        return
    data_store = install_dir / "data_store"
    preserved = None
    if data_store.exists() and not delete_data:
        preserved = install_dir.with_name(f"{install_dir.name}_data_store_backup_{timestamp()}")
        print_action(dry_run, f"preserve data_store {data_store} -> {preserved}")
        if not dry_run:
            shutil.move(str(data_store), str(preserved))
    print_action(dry_run, f"remove install directory {install_dir}")
    if not dry_run:
        shutil.rmtree(install_dir)
    if preserved:
        print(f"Preserved data_store: {preserved}")


def preferred_python(install_dir: Path, venv_name: str) -> Path:
    preferred = install_dir / venv_name / bin_dir() / python_executable_name()
    if preferred.exists():
        return preferred
    fallback = install_dir / DEFAULT_FALLBACK_VENV / bin_dir() / python_executable_name()
    if fallback.exists():
        return fallback
    return preferred


def service_file_path(context: DeployContext) -> Path:
    if context.service_mode == "launchd":
        name = context.service_name if context.service_name.endswith(".plist") else f"{context.service_name}.plist"
        return Path.home() / "Library/LaunchAgents" / name
    if context.service_mode == "windows-task":
        return context.install_dir / "data_store/system_config/start_uav_detection.cmd"
    if context.service_mode == "user":
        return Path.home() / ".config/systemd/user" / context.service_name
    return SERVICE_FILE_ROOT / context.service_name


def service_command(context: DeployContext, action: str) -> list[str]:
    if context.service_mode == "user":
        return ["systemctl", "--user", action, context.service_name]
    return ["sudo", "systemctl", action, context.service_name]


def service_active(context: DeployContext) -> bool:
    if context.no_service:
        return False
    if context.service_mode == "launchd":
        return command_ok(["launchctl", "print", f"gui/{os.getuid()}/{launchd_label(context)}"])
    if context.service_mode == "windows-task":
        return command_ok(["schtasks", "/Query", "/TN", context.service_name])
    command = ["systemctl", "--user", "is-active", "--quiet", context.service_name] if context.service_mode == "user" else [
        "systemctl",
        "is-active",
        "--quiet",
        context.service_name,
    ]
    return command_ok(command)


def python_has_venv() -> bool:
    return command_ok([sys.executable, "-m", "venv", "--help"])


def git_status_short(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        result = subprocess.run(["git", "status", "--short"], cwd=path, text=True, capture_output=True, check=False, timeout=8)
    except Exception:
        return None
    return result.stdout.strip()


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def command_ok(command: list[str]) -> bool:
    try:
        return subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=8).returncode == 0
    except Exception:
        return False


def run(command: list[str], check: bool, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(str(part) for part in command), flush=True)
    return subprocess.run(command, cwd=cwd, check=check, text=True)


def print_checks(checks: list[Check]) -> None:
    width = max(len(check.name) for check in checks) if checks else 10
    for check in checks:
        print(f"{check.status.upper():5} {check.name:<{width}} {check.detail}")


def has_failures(checks: list[Check]) -> bool:
    return any(check.status == "fail" for check in checks)


def print_action(dry_run: bool, message: str) -> None:
    prefix = "DRY RUN: " if dry_run else ""
    print(prefix + message)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def bin_dir() -> str:
    return "Scripts" if platform.system() == "Windows" else "bin"


def python_executable_name() -> str:
    return "python.exe" if platform.system() == "Windows" else "python"


def launchd_label(context: DeployContext) -> str:
    return context.service_name[:-6] if context.service_name.endswith(".plist") else context.service_name


if __name__ == "__main__":
    raise SystemExit(main())
