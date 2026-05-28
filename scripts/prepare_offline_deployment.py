from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PREFIX = "UAVDetection_offline"
EXCLUDED_SOURCE_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "annotations",
    "certs",
    "data_store",
    "models",
    "reports",
    "runs",
    "videos",
}
EXCLUDED_SOURCE_SUFFIXES = {".pyc", ".pyo"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a USB-ready offline deployment bundle with project code, "
            "data_store, install scripts, and optional Python wheels."
        )
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="USB drive path or local output directory that will receive the bundle.",
    )
    parser.add_argument(
        "--bundle-name",
        default="",
        help="Bundle folder name. Defaults to UAVDetection_offline_<YYYYMMDD_HHMMSS>.",
    )
    parser.add_argument(
        "--no-data-store",
        action="store_true",
        help="Do not copy data_store into the bundle.",
    )
    parser.add_argument(
        "--no-wheelhouse",
        action="store_true",
        help="Skip building wheelhouse. Target install will then need dependencies already available.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to download wheels. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--wheel-platform",
        choices=("current", "windows-x64"),
        default="current",
        help="Wheelhouse target platform. Use windows-x64 for an offline Windows laptop bundle.",
    )
    parser.add_argument(
        "--wheel-python-version",
        default="311",
        help="Target CPython version for cross-platform wheels, for example 311 for Python 3.11.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing bundle directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.destination.expanduser().resolve()
    bundle_name = args.bundle_name or f"{DEFAULT_BUNDLE_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    bundle_root = destination / bundle_name
    project_bundle = bundle_root / "UAVDetection"

    if bundle_root.exists():
        if not args.force:
            raise SystemExit(f"Bundle already exists: {bundle_root}. Use --force to replace it.")
        shutil.rmtree(bundle_root, ignore_errors=True)
        if bundle_root.exists():
            raise SystemExit(f"Could not fully replace existing bundle: {bundle_root}")

    destination.mkdir(parents=True, exist_ok=True)
    project_bundle.mkdir(parents=True)

    copy_project_source(project_bundle)
    if not args.no_data_store:
        copy_data_store(project_bundle / "data_store")
    else:
        ensure_minimal_data_store(project_bundle / "data_store")

    wheelhouse_status = "skipped"
    if not args.no_wheelhouse:
        wheelhouse_status = build_wheelhouse(args.python, project_bundle, args.wheel_platform, args.wheel_python_version)

    write_bundle_helpers(bundle_root, project_bundle, args.wheel_python_version)
    write_manifest(
        bundle_root,
        project_bundle,
        include_data_store=not args.no_data_store,
        wheelhouse_status=wheelhouse_status,
        wheel_platform=args.wheel_platform,
        wheel_python_version=args.wheel_python_version,
    )

    print(f"Offline deployment bundle ready: {bundle_root}")
    print(f"Project copy: {project_bundle}")
    print(f"Wheelhouse: {wheelhouse_status}")
    print("On the target machine, run:")
    print(f"  cd {shell_display_path(bundle_root)}")
    print("  ./install_offline.sh")
    print("On Windows, run:")
    print(f"  {bundle_root}\\install_offline.ps1")
    return 0


def copy_project_source(target: Path) -> None:
    for item in PROJECT_ROOT.iterdir():
        if should_skip_source_item(item):
            continue
        destination = target / item.name
        if item.is_symlink():
            copy_symlink_or_target(item, destination)
        elif item.is_dir():
            shutil.copytree(item, destination, ignore=source_ignore, copy_function=safe_copy_file)
        else:
            safe_copy_file(item, destination)


def should_skip_source_item(path: Path) -> bool:
    if path.name in EXCLUDED_SOURCE_NAMES:
        return True
    if path.suffix in EXCLUDED_SOURCE_SUFFIXES:
        return True
    return False


def source_ignore(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path = Path(directory) / name
        if name in EXCLUDED_SOURCE_NAMES:
            ignored.add(name)
        elif path.suffix in EXCLUDED_SOURCE_SUFFIXES:
            ignored.add(name)
    return ignored


def copy_symlink_or_target(source: Path, destination: Path) -> None:
    try:
        target_path = source.resolve(strict=True)
    except FileNotFoundError:
        return
    if target_path.is_dir():
        shutil.copytree(target_path, destination, ignore=source_ignore, copy_function=safe_copy_file)
    elif target_path.is_file():
        safe_copy_file(target_path, destination)


def copy_data_store(target: Path) -> None:
    source = PROJECT_ROOT / "data_store"
    if not source.exists():
        ensure_minimal_data_store(target)
        return
    shutil.copytree(source, target, ignore=data_store_ignore, copy_function=safe_copy_file)


def data_store_ignore(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name == "__pycache__" or name == ".DS_Store":
            ignored.add(name)
    return ignored


def ensure_minimal_data_store(target: Path) -> None:
    for relative in (
        "raw_data",
        "detection_results",
        "datasets",
        "models/base",
        "models/external",
        "models/trained",
        "system_config/certs",
        "stats",
        "backups",
    ):
        (target / relative).mkdir(parents=True, exist_ok=True)
    readme = PROJECT_ROOT / "data_store" / "README.md"
    if readme.exists():
        safe_copy_file(readme, target / "README.md")


def safe_copy_file(source: Path | str, destination: Path | str) -> str:
    """Copy file content without macOS extended metadata that some USB filesystems reject."""
    shutil.copyfile(source, destination)
    try:
        shutil.copymode(source, destination)
    except OSError:
        pass
    return str(destination)


def build_wheelhouse(python: str, project_bundle: Path, wheel_platform: str, wheel_python_version: str) -> str:
    requirements = project_bundle / "requirements.txt"
    wheelhouse = project_bundle / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        "-m",
        "pip",
        "download",
        "--dest",
        str(wheelhouse),
    ]
    if wheel_platform == "windows-x64":
        command.extend(
            [
                "--platform",
                "win_amd64",
                "--implementation",
                "cp",
                "--python-version",
                wheel_python_version,
                "--abi",
                f"cp{wheel_python_version}",
                "--abi",
                "abi3",
                "--abi",
                "none",
                "--only-binary",
                ":all:",
            ]
        )
    command.extend(["-r", str(requirements)])
    print("Building wheelhouse for offline dependency install...")
    result = subprocess.run(command, cwd=project_bundle)
    if result.returncode != 0:
        marker = wheelhouse / "WHEELHOUSE_INCOMPLETE.txt"
        marker.write_text(
            "\n".join(
                [
                    "pip download failed while preparing this bundle.",
                    "The target can still install if dependencies are already cached or already installed.",
                    "For true offline deployment, rebuild this bundle on a machine with package access.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return "incomplete"
    return "ready"


def write_bundle_helpers(bundle_root: Path, project_bundle: Path, wheel_python_version: str) -> None:
    install_sh = bundle_root / "install_offline.sh"
    install_sh.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                'cd "$SCRIPT_DIR/UAVDetection"',
                'python3 scripts/install_offline_deployment.py "$@"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    install_sh.chmod(0o755)

    install_ps1 = bundle_root / "install_offline.ps1"
    install_ps1.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                "$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path",
                "Set-Location (Join-Path $ScriptDir 'UAVDetection')",
                "$Py = Get-Command py -ErrorAction SilentlyContinue",
                "if ($Py) {",
                f"  & py -{wheel_python_version[0]}.{wheel_python_version[1:]} --version *> $null",
                "  if ($LASTEXITCODE -eq 0) {",
                f"    & py -{wheel_python_version[0]}.{wheel_python_version[1:]} scripts/install_offline_deployment.py @args",
                "    exit $LASTEXITCODE",
                "  }",
                "}",
                "& python scripts/install_offline_deployment.py @args",
                "exit $LASTEXITCODE",
                "",
            ]
        ),
        encoding="utf-8",
    )

    install_cmd = bundle_root / "install_offline.cmd"
    install_cmd.write_text(
        "\r\n".join(
            [
                "@echo off",
                "set SCRIPT_DIR=%~dp0",
                'powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install_offline.ps1" %*',
                "exit /b %ERRORLEVEL%",
                "",
            ]
        ),
        encoding="utf-8",
    )

    readme = bundle_root / "README_OFFLINE_DEPLOY.md"
    readme.write_text(
        "\n".join(
            [
                "# UAVDetection Offline USB Deployment",
                "",
                "This bundle is intended for deployment without Internet access.",
                "",
                "## Install On Target",
                "",
                "macOS/Linux:",
                "",
                "```bash",
                "cd <USB>/UAVDetection_offline_*",
                "./install_offline.sh",
                "```",
                "",
                "Windows PowerShell:",
                "",
                "```powershell",
                "cd E:\\UAVDetection_offline_current",
                ".\\install_offline.ps1",
                "```",
                "",
                "Windows Command Prompt:",
                "",
                "```bat",
                "E:\\UAVDetection_offline_current\\install_offline.cmd",
                "```",
                "",
                "The installer copies the bundled project to `~/UAVDetection` on macOS/Linux or `%USERPROFILE%\\UAVDetection` on Windows by default, creates `.venv`, installs Python dependencies from the copied `wheelhouse`, initializes `data_store`, creates or reuses HTTPS certificates, installs an automatic-start service, and updates common browser home/start pages to the local server URL.",
                "",
                "Default local URL:",
                "",
                "```text",
                "https://127.0.0.1:8765",
                "```",
                "",
                "Default login:",
                "",
                "```text",
                "admin / admin123",
                "```",
                "",
                "## Important Platform Note",
                "",
                "Python wheels are platform-specific. For a Windows laptop, prepare the bundle with `--wheel-platform windows-x64 --wheel-python-version 311` and install Python 3.11 on the target first. For Jetson, prepare the bundle on Jetson or another compatible Linux ARM environment, or provide a matching wheelhouse manually.",
                "",
                "## Useful Options",
                "",
                "```bash",
                "./install_offline.sh --install-dir ~/UAVDetection",
                "./install_offline.sh --install-dir ~/UAVDetection --force",
                "./install_offline.sh --port 8765 --password admin123",
                "./install_offline.sh --no-browser-homepage",
                "./install_offline.sh --no-autostart",
                "./install_offline.sh --allow-online",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_manifest(
    bundle_root: Path,
    project_bundle: Path,
    include_data_store: bool,
    wheelhouse_status: str,
    wheel_platform: str,
    wheel_python_version: str,
) -> None:
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_root": str(PROJECT_ROOT),
        "bundle_root": str(bundle_root),
        "project_bundle": str(project_bundle),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "git": git_info(),
        "include_data_store": include_data_store,
        "wheelhouse_status": wheelhouse_status,
        "wheel_platform": wheel_platform,
        "wheel_python_version": wheel_python_version,
        "data_store_bytes": directory_size(project_bundle / "data_store") if include_data_store else 0,
        "project_bytes": directory_size(project_bundle),
    }
    (bundle_root / "OFFLINE_BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def git_info() -> dict[str, object]:
    def run_git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            return ""
        return result.stdout.strip()

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(run_git("status", "--short")),
    }


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def shell_display_path(path: Path) -> str:
    return str(path).replace(" ", "\\ ")


if __name__ == "__main__":
    raise SystemExit(main())
