from __future__ import annotations

import argparse
import json
import os
import plistlib
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL_HOST = "127.0.0.1"
SERVICE_LABEL = "com.uavdetection.annotation-server"
CAMERA_CONFIG_RELATIVE = Path("data_store/system_config/cameras.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install UAVDetection from an offline USB bundle.")
    parser.add_argument(
        "--install-dir",
        default=str(Path.home() / "UAVDetection"),
        help="Local target directory for the deployed project. Defaults to ~/UAVDetection.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Configure the project in the current directory instead of first copying to --install-dir.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and fully replace an existing --install-dir, including data_store. Use only when local data can be discarded.",
    )
    parser.add_argument(
        "--no-camera-config-update",
        action="store_true",
        help="Do not refresh data_store/system_config/cameras.yaml from the USB bundle when updating an existing install.",
    )
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Server port.")
    parser.add_argument("--username", default="admin", help="Annotation server username.")
    parser.add_argument("--password", default=os.environ.get("ANNOTATION_SERVER_PASSWORD", "admin123"))
    parser.add_argument("--default-folder", default="data_store/raw_data/Roni")
    parser.add_argument("--project-dir", default="data_store/datasets/web_drone_v1")
    parser.add_argument("--camera-config", default="data_store/system_config/cameras.yaml")
    parser.add_argument("--live-model", default="data_store/models/trained/yolov8n_drone_best.pt")
    parser.add_argument("--venv", default=".venv", help="Virtualenv directory under the project root.")
    parser.add_argument("--wheelhouse", default="wheelhouse", help="Offline wheel directory under the project root.")
    parser.add_argument("--allow-online", action="store_true", help="Allow pip to use indexes if wheelhouse is incomplete.")
    parser.add_argument("--skip-deps", action="store_true", help="Do not create venv or install dependencies.")
    parser.add_argument("--no-https", action="store_true", help="Run HTTP instead of HTTPS.")
    parser.add_argument("--no-autostart", action="store_true", help="Do not install launchd/systemd auto-start service.")
    parser.add_argument("--no-browser-homepage", action="store_true", help="Do not update common browser home/start pages.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing service/browser changes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    install_dir = Path(args.install_dir).expanduser().resolve()
    if not args.in_place and PROJECT_ROOT.resolve() != install_dir:
        return copy_and_reinvoke(args, install_dir)

    os.chdir(PROJECT_ROOT)
    preserve_data_store = args.update_existing and (PROJECT_ROOT / "data_store").exists()
    if preserve_data_store:
        preserve_existing_credentials(args)

    venv_python = PROJECT_ROOT / args.venv / bin_dir() / "python"
    if not args.skip_deps:
        create_venv(PROJECT_ROOT / args.venv, args.dry_run)
        install_dependencies(venv_python, PROJECT_ROOT / args.wheelhouse, allow_online=args.allow_online, dry_run=args.dry_run)
    elif not venv_python.exists():
        venv_python = Path(sys.executable)

    if not args.dry_run and not preserve_data_store:
        run([str(venv_python), "scripts/datastore_sync.py", "init"], check=True)
        write_server_env(args)
    elif not args.dry_run:
        maybe_write_server_env_for_update(args)

    scheme = "http" if args.no_https else "https"
    certfile = PROJECT_ROOT / "data_store/system_config/certs/annotation.crt"
    keyfile = PROJECT_ROOT / "data_store/system_config/certs/annotation.key"
    if scheme == "https":
        ensure_certificate(certfile, keyfile, args.dry_run)
        if not certfile.exists() or not keyfile.exists():
            print("HTTPS certificate unavailable; falling back to HTTP.")
            scheme = "http"

    service_command = server_command(venv_python, args, scheme, certfile, keyfile)
    env = server_env(args)
    local_url = f"{scheme}://{DEFAULT_URL_HOST}:{args.port}"

    if not args.no_autostart:
        install_autostart(service_command, env, args.dry_run)

    if not args.no_browser_homepage:
        configure_browser_start_pages(local_url, args.dry_run)

    write_local_shortcut(local_url, args.dry_run)

    print("Offline deployment configured.")
    print(f"Server URL: {local_url}")
    print(f"Login: {args.username} / {args.password}")
    if args.no_autostart:
        print("Autostart skipped. Manual command:")
        print(" ".join(shlex.quote(part) for part in service_command))
    return 0


def copy_and_reinvoke(args: argparse.Namespace, install_dir: Path) -> int:
    if args.dry_run:
        print(f"DRY RUN: copy {PROJECT_ROOT} -> {install_dir}")
        print("DRY RUN: re-run installer from local install directory")
        return 0

    update_existing = copy_project_to_install_dir(
        PROJECT_ROOT,
        install_dir,
        force=args.force,
        update_camera_config=not args.no_camera_config_update,
    )
    command = [sys.executable, str(install_dir / "scripts/install_offline_deployment.py"), *reinvoke_args(update_existing)]
    print(f"Copied project to local install directory: {install_dir}")
    return subprocess.run(command, cwd=install_dir).returncode


def reinvoke_args(update_existing: bool) -> list[str]:
    args = list(sys.argv[1:])
    if "--in-place" not in args:
        args.append("--in-place")
    if update_existing and "--update-existing" not in args:
        args.append("--update-existing")
    return args


def copy_project_to_install_dir(source: Path, destination: Path, force: bool, update_camera_config: bool) -> bool:
    if destination.exists():
        if force:
            shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination, ignore=install_copy_ignore, copy_function=safe_copy_file)
            return False

        update_existing_install(source, destination, update_camera_config)
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, ignore=install_copy_ignore, copy_function=safe_copy_file)
    return False


def update_existing_install(source: Path, destination: Path, update_camera_config: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in {".venv", "data_store", "__pycache__", ".pytest_cache", ".DS_Store"}:
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, ignore=install_update_ignore, copy_function=safe_copy_file, dirs_exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            safe_copy_file(item, target)

    if update_camera_config:
        refresh_camera_config(source / CAMERA_CONFIG_RELATIVE, destination / CAMERA_CONFIG_RELATIVE)


def install_update_ignore(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path = Path(directory) / name
        if name in {".venv", "data_store", "__pycache__", ".pytest_cache", ".DS_Store"}:
            ignored.add(name)
        elif path.suffix == ".pyc":
            ignored.add(name)
    return ignored


def refresh_camera_config(source: Path, destination: Path) -> None:
    if not source.exists():
        print(f"Camera config update skipped; bundle file not found: {source}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() == source.read_bytes():
        print(f"Camera config already current: {destination}")
        return
    if destination.exists():
        backup = destination.with_name(f"{destination.name}.backup_{timestamp()}")
        safe_copy_file(destination, backup)
        print(f"Backed up existing camera config: {backup}")
    safe_copy_file(source, destination)
    print(f"Updated camera config: {destination}")


def install_copy_ignore(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in {".venv", "__pycache__", ".pytest_cache", ".DS_Store"}:
            ignored.add(name)
    return ignored


def safe_copy_file(source: Path | str, destination: Path | str) -> str:
    shutil.copyfile(source, destination)
    try:
        shutil.copymode(source, destination)
    except OSError:
        pass
    return str(destination)


def bin_dir() -> str:
    return "Scripts" if platform.system() == "Windows" else "bin"


def create_venv(venv: Path, dry_run: bool) -> None:
    if (venv / bin_dir() / "python").exists():
        return
    command = [sys.executable, "-m", "venv", str(venv)]
    if dry_run:
        print("DRY RUN:", " ".join(command))
        return
    run(command, check=True)


def install_dependencies(venv_python: Path, wheelhouse: Path, allow_online: bool, dry_run: bool) -> None:
    requirements = PROJECT_ROOT / "requirements.txt"
    command = [str(venv_python), "-m", "pip", "install"]
    if wheelhouse.exists() and any(wheelhouse.iterdir()):
        command.extend(["--no-index", "--find-links", str(wheelhouse)])
    elif not allow_online:
        raise SystemExit(
            f"Wheelhouse is missing or empty: {wheelhouse}. "
            "Rebuild the USB bundle with wheels, or rerun installer with --allow-online."
        )
    command.extend(["-r", str(requirements)])
    if dry_run:
        print("DRY RUN:", " ".join(shlex.quote(part) for part in command))
        return
    run(command, check=True)


def write_server_env(args: argparse.Namespace) -> None:
    config_dir = PROJECT_ROOT / "data_store/system_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    env_path = config_dir / "annotation_server.env"
    env_path.write_text(
        "\n".join(
            [
                f"ANNOTATION_SERVER_USERNAME={args.username}",
                f"ANNOTATION_SERVER_PASSWORD={args.password}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    try:
        env_path.chmod(0o600)
    except OSError:
        pass


def maybe_write_server_env_for_update(args: argparse.Namespace) -> None:
    env_path = PROJECT_ROOT / "data_store/system_config/annotation_server.env"
    if env_path.exists() and not (option_was_passed("--username") or option_was_passed("--password")):
        return
    write_server_env(args)


def preserve_existing_credentials(args: argparse.Namespace) -> None:
    env_path = PROJECT_ROOT / "data_store/system_config/annotation_server.env"
    values = read_env_file(env_path)
    if not option_was_passed("--username") and values.get("ANNOTATION_SERVER_USERNAME"):
        args.username = values["ANNOTATION_SERVER_USERNAME"]
    if not option_was_passed("--password") and values.get("ANNOTATION_SERVER_PASSWORD"):
        args.password = values["ANNOTATION_SERVER_PASSWORD"]


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def option_was_passed(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def server_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env["ANNOTATION_SERVER_USERNAME"] = args.username
    env["ANNOTATION_SERVER_PASSWORD"] = args.password
    return env


def ensure_certificate(certfile: Path, keyfile: Path, dry_run: bool) -> None:
    if certfile.exists() and keyfile.exists():
        return
    openssl = shutil.which("openssl")
    if not openssl:
        print("openssl not found; cannot create self-signed HTTPS certificate.")
        return
    certfile.parent.mkdir(parents=True, exist_ok=True)
    command = [
        openssl,
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(keyfile),
        "-out",
        str(certfile),
        "-days",
        "365",
        "-subj",
        "/CN=uav-detection-local",
    ]
    if dry_run:
        print("DRY RUN:", " ".join(shlex.quote(part) for part in command))
        return
    run(command, check=True)


def server_command(
    venv_python: Path,
    args: argparse.Namespace,
    scheme: str,
    certfile: Path,
    keyfile: Path,
) -> list[str]:
    command = [
        str(venv_python),
        str(PROJECT_ROOT / "scripts/annotation_server.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--username",
        args.username,
        "--password-env",
        "ANNOTATION_SERVER_PASSWORD",
        "--default-folder",
        str(PROJECT_ROOT / args.default_folder),
        "--project-dir",
        str(PROJECT_ROOT / args.project_dir),
        "--camera-config",
        str(PROJECT_ROOT / args.camera_config),
        "--live-model",
        str(PROJECT_ROOT / args.live_model),
    ]
    if scheme == "https":
        command.extend(["--certfile", str(certfile), "--keyfile", str(keyfile)])
    return command


def install_autostart(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    system = platform.system()
    if system == "Darwin":
        install_launch_agent(command, env, dry_run)
    elif system == "Linux":
        install_systemd_user_service(command, env, dry_run)
    elif system == "Windows":
        install_windows_scheduled_task(command, env, dry_run)
    else:
        print(f"Autostart is not implemented for {system}. Manual command:")
        print(" ".join(shlex.quote(part) for part in command))


def install_launch_agent(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    launch_agents = Path.home() / "Library/LaunchAgents"
    plist_path = launch_agents / f"{SERVICE_LABEL}.plist"
    logs_dir = PROJECT_ROOT / "data_store/system_config/logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plist = {
        "Label": SERVICE_LABEL,
        "ProgramArguments": command,
        "WorkingDirectory": str(PROJECT_ROOT),
        "RunAtLoad": True,
        "KeepAlive": True,
        "EnvironmentVariables": {
            "ANNOTATION_SERVER_USERNAME": env["ANNOTATION_SERVER_USERNAME"],
            "ANNOTATION_SERVER_PASSWORD": env["ANNOTATION_SERVER_PASSWORD"],
        },
        "StandardOutPath": str(logs_dir / "annotation_server.out.log"),
        "StandardErrorPath": str(logs_dir / "annotation_server.err.log"),
    }
    if dry_run:
        print(f"DRY RUN: write launch agent {plist_path}")
        return
    launch_agents.mkdir(parents=True, exist_ok=True)
    with plist_path.open("wb") as handle:
        plistlib.dump(plist, handle)
    uid = os.getuid()
    run(["launchctl", "bootout", f"gui/{uid}", str(plist_path)], check=False)
    run(["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)], check=False)
    run(["launchctl", "enable", f"gui/{uid}/{SERVICE_LABEL}"], check=False)
    run(["launchctl", "kickstart", "-k", f"gui/{uid}/{SERVICE_LABEL}"], check=False)


def install_systemd_user_service(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    systemd_dir = Path.home() / ".config/systemd/user"
    service_path = systemd_dir / "uav-annotation-server.service"
    env_path = PROJECT_ROOT / "data_store/system_config/annotation_server.env"
    content = "\n".join(
        [
            "[Unit]",
            "Description=UAVDetection annotation and live detection server",
            "After=network-online.target",
            "",
            "[Service]",
            f"WorkingDirectory={PROJECT_ROOT}",
            f"EnvironmentFile={env_path}",
            "ExecStart=" + " ".join(shlex.quote(part) for part in command),
            "Restart=always",
            "RestartSec=5",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )
    if dry_run:
        print(f"DRY RUN: write systemd user service {service_path}")
        return
    systemd_dir.mkdir(parents=True, exist_ok=True)
    service_path.write_text(content, encoding="utf-8")
    run(["systemctl", "--user", "daemon-reload"], check=False)
    run(["systemctl", "--user", "enable", "--now", "uav-annotation-server.service"], check=False)
    print("For start at boot before login on Linux, run once if permitted:")
    print(f"  sudo loginctl enable-linger {os.environ.get('USER', '<user>')}")


def install_windows_scheduled_task(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    start_script = write_windows_start_script(command, env, dry_run)
    task_name = "UAVDetection Annotation Server"
    task_command = f'"{start_script}"'
    schtasks = shutil.which("schtasks")
    if not schtasks:
        print("schtasks not found. Manual startup script:")
        print(start_script)
        return
    command_line = [
        schtasks,
        "/Create",
        "/TN",
        task_name,
        "/SC",
        "ONLOGON",
        "/TR",
        task_command,
        "/F",
    ]
    if dry_run:
        print("DRY RUN:", " ".join(shlex.quote(part) for part in command_line))
        return
    run(command_line, check=False)


def write_windows_start_script(command: list[str], env: dict[str, str], dry_run: bool) -> Path:
    config_dir = PROJECT_ROOT / "data_store/system_config"
    script_path = config_dir / "start_annotation_server.cmd"
    lines = [
        "@echo off",
        f'cd /d "{PROJECT_ROOT}"',
        f'set "ANNOTATION_SERVER_USERNAME={env["ANNOTATION_SERVER_USERNAME"]}"',
        f'set "ANNOTATION_SERVER_PASSWORD={env["ANNOTATION_SERVER_PASSWORD"]}"',
        " ".join(windows_quote(part) for part in command),
        "",
    ]
    if dry_run:
        print(f"DRY RUN: write Windows startup script {script_path}")
        return script_path
    config_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\r\n".join(lines), encoding="utf-8")
    return script_path


def windows_quote(value: str) -> str:
    escaped = value.replace('"', r'\"')
    return f'"{escaped}"'


def timestamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def configure_browser_start_pages(url: str, dry_run: bool) -> None:
    configured = []
    configured.extend(configure_chromium_family(url, dry_run))
    configured.extend(configure_firefox(url, dry_run))
    if platform.system() == "Darwin":
        configured.extend(configure_safari(url, dry_run))
    if configured:
        print("Browser start pages updated:")
        for item in configured:
            print(f"  {item}")
    else:
        print("No supported browser profiles found for homepage update.")


def configure_chromium_family(url: str, dry_run: bool) -> list[str]:
    candidates: list[Path]
    if platform.system() == "Darwin":
        base = Path.home() / "Library/Application Support"
        candidates = [
            base / "Google/Chrome/Default/Preferences",
            base / "Microsoft Edge/Default/Preferences",
            base / "BraveSoftware/Brave-Browser/Default/Preferences",
            base / "Chromium/Default/Preferences",
        ]
    elif platform.system() == "Windows":
        local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
        candidates = [
            local_app_data / "Google/Chrome/User Data/Default/Preferences",
            local_app_data / "Microsoft/Edge/User Data/Default/Preferences",
            local_app_data / "BraveSoftware/Brave-Browser/User Data/Default/Preferences",
            local_app_data / "Chromium/User Data/Default/Preferences",
        ]
    else:
        base = Path.home() / ".config"
        candidates = [
            base / "google-chrome/Default/Preferences",
            base / "chromium/Default/Preferences",
            base / "microsoft-edge/Default/Preferences",
            base / "BraveSoftware/Brave-Browser/Default/Preferences",
        ]
    updated = []
    for prefs_path in candidates:
        if not prefs_path.exists():
            continue
        if dry_run:
            updated.append(str(prefs_path))
            continue
        try:
            prefs = json.loads(prefs_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        prefs["homepage"] = url
        prefs["homepage_is_newtabpage"] = False
        session = prefs.setdefault("session", {})
        session["restore_on_startup"] = 4
        session["startup_urls"] = [url]
        prefs_path.write_text(json.dumps(prefs, indent=2, sort_keys=True), encoding="utf-8")
        updated.append(str(prefs_path))
    return updated


def configure_firefox(url: str, dry_run: bool) -> list[str]:
    profiles_root = Path.home() / ".mozilla/firefox"
    if platform.system() == "Darwin":
        profiles_root = Path.home() / "Library/Application Support/Firefox/Profiles"
    elif platform.system() == "Windows":
        profiles_root = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / "Mozilla/Firefox/Profiles"
    if not profiles_root.exists():
        return []
    updated = []
    for prefs_path in profiles_root.glob("*/prefs.js"):
        if dry_run:
            updated.append(str(prefs_path))
            continue
        try:
            text = prefs_path.read_text(encoding="utf-8")
        except OSError:
            continue
        text = upsert_firefox_pref(text, "browser.startup.homepage", json.dumps(url))
        text = upsert_firefox_pref(text, "browser.startup.page", "1")
        prefs_path.write_text(text, encoding="utf-8")
        updated.append(str(prefs_path))
    return updated


def upsert_firefox_pref(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf'^user_pref\("{re.escape(key)}", .*?\);\s*$', re.MULTILINE)
    replacement = f'user_pref("{key}", {value});'
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text.rstrip() + "\n" + replacement + "\n"


def configure_safari(url: str, dry_run: bool) -> list[str]:
    commands = [
        ["defaults", "write", "com.apple.Safari", "HomePage", url],
        ["defaults", "write", "com.apple.Safari", "NewWindowBehavior", "-int", "0"],
        ["defaults", "write", "com.apple.Safari", "NewTabBehavior", "-int", "0"],
    ]
    if dry_run:
        return ["Safari defaults"]
    for command in commands:
        run(command, check=False)
    return ["Safari defaults"]


def write_local_shortcut(url: str, dry_run: bool) -> None:
    shortcut = PROJECT_ROOT / "UAVDetection_Server.url"
    if dry_run:
        print(f"DRY RUN: write {shortcut}")
        return
    shortcut.write_text(f"[InternetShortcut]\nURL={url}\n", encoding="utf-8")


def run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=PROJECT_ROOT, check=check, text=True)


if __name__ == "__main__":
    raise SystemExit(main())
