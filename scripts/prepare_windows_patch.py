from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATCH_FILES = [
    "README.md",
    "PROJECT_CONTEXT.md",
    "scripts/annotation_server.py",
    "scripts/train_yolov8n_drone.py",
    "web/annotator/app.js",
    "web/annotator/index.html",
    "web/annotator/styles.css",
]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data_store" / "deployment_patches"
TASK_NAME = "UAVDetection Annotation Server"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Windows copy-only patch ZIP for an offline UAVDetection install.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that will receive the patch folder and ZIP.",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Patch folder/ZIP base name. Defaults to UAVDetection_windows_patch_<timestamp>.",
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        default=[],
        help="Additional project-relative file to include. Can be passed more than once.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing patch folder/ZIP with the same name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patch_name = args.name or f"UAVDetection_windows_patch_{stamp}"
    output_dir = args.output_dir.expanduser().resolve()
    patch_dir = output_dir / patch_name
    zip_path = output_dir / f"{patch_name}.zip"

    if patch_dir.exists():
        if not args.force:
            raise SystemExit(f"Patch directory already exists: {patch_dir}. Use --force to replace it.")
        shutil.rmtree(patch_dir)
    if zip_path.exists():
        if not args.force:
            raise SystemExit(f"Patch ZIP already exists: {zip_path}. Use --force to replace it.")
        zip_path.unlink()

    files = unique_files([*DEFAULT_PATCH_FILES, *args.files])
    output_dir.mkdir(parents=True, exist_ok=True)
    (patch_dir / "files").mkdir(parents=True)
    copy_patch_files(files, patch_dir / "files")
    write_install_cmd(patch_dir / "install_patch.cmd")
    write_readme(patch_dir / "README_PATCH.md", files)
    write_manifest(patch_dir / "PATCH_MANIFEST.json", patch_name, files)
    write_zip(patch_dir, zip_path)

    print(f"Windows patch folder: {patch_dir}")
    print(f"Windows patch ZIP: {zip_path}")
    print("Target usage:")
    print("  1. Download the ZIP to the Windows laptop.")
    print("  2. Extract it.")
    print("  3. Run install_patch.cmd, optionally with the install dir as the first argument.")
    return 0


def unique_files(files: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in files:
        normalized = item.replace("\\", "/").strip("/")
        if not normalized or normalized in seen:
            continue
        path = PROJECT_ROOT / normalized
        if not path.exists() or not path.is_file():
            raise SystemExit(f"Patch file not found: {normalized}")
        seen.add(normalized)
        result.append(normalized)
    return result


def copy_patch_files(files: list[str], target_root: Path) -> None:
    for relative in files:
        source = PROJECT_ROOT / relative
        destination = target_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def write_install_cmd(path: Path) -> None:
    path.write_text(
        "\r\n".join(
            [
                "@echo off",
                "setlocal EnableExtensions",
                "set PATCH_DIR=%~dp0",
                "set INSTALL_DIR=%~1",
                "if \"%INSTALL_DIR%\"==\"\" set INSTALL_DIR=%USERPROFILE%\\UAVDetection",
                "",
                "echo UAVDetection patch installer",
                "echo Patch: %PATCH_DIR%",
                "echo Install dir: %INSTALL_DIR%",
                "",
                "if not exist \"%PATCH_DIR%files\" (",
                "  echo ERROR: Patch files directory not found.",
                "  exit /b 1",
                ")",
                "",
                "if not exist \"%INSTALL_DIR%\" (",
                "  echo ERROR: Install directory not found: %INSTALL_DIR%",
                "  echo Pass the install directory as the first argument, for example:",
                "  echo install_patch.cmd C:\\Users\\YOUR_USER\\UAVDetection",
                "  exit /b 1",
                ")",
                "",
                f"echo Stopping scheduled task if it exists: {TASK_NAME}",
                f"schtasks /End /TN \"{TASK_NAME}\" >nul 2>nul",
                "timeout /t 2 /nobreak >nul 2>nul",
                "",
                "echo Copying patch files...",
                "robocopy \"%PATCH_DIR%files\" \"%INSTALL_DIR%\" /E /R:2 /W:1 /NFL /NDL /NJH /NJS /NP",
                "set ROBOCOPY_RC=%ERRORLEVEL%",
                "if %ROBOCOPY_RC% GEQ 8 (",
                "  echo ERROR: robocopy failed with code %ROBOCOPY_RC%.",
                "  exit /b %ROBOCOPY_RC%",
                ")",
                "",
                f"echo Starting scheduled task if it exists: {TASK_NAME}",
                f"schtasks /Run /TN \"{TASK_NAME}\" >nul 2>nul",
                "",
                "echo Patch installed successfully.",
                "echo Open https://127.0.0.1:8765 and refresh the browser.",
                "exit /b 0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_readme(path: Path, files: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "# UAVDetection Windows Patch",
                "",
                "This is a copy-only patch for an already installed Windows laptop.",
                "",
                "## Install",
                "",
                "1. Download the ZIP from Google Drive.",
                "2. Extract the ZIP.",
                "3. Double-click `install_patch.cmd`, or run it from Command Prompt.",
                "",
                "Default target install directory:",
                "",
                "```text",
                "%USERPROFILE%\\UAVDetection",
                "```",
                "",
                "If the project was installed somewhere else, pass that folder:",
                "",
                "```bat",
                "install_patch.cmd D:\\UAVDetection",
                "```",
                "",
                "The script tries to stop the `UAVDetection Annotation Server` scheduled task, copies the patch files, then starts the task again.",
                "",
                "## Included Files",
                "",
                *[f"- `{relative}`" for relative in files],
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_manifest(path: Path, patch_name: str, files: list[str]) -> None:
    manifest = {
        "name": patch_name,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_root": str(PROJECT_ROOT),
        "git": git_info(),
        "task_name": TASK_NAME,
        "default_install_dir": "%USERPROFILE%\\UAVDetection",
        "files": files,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_zip(patch_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(patch_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(patch_dir.parent))


def git_info() -> dict[str, str]:
    def run_git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except OSError:
            return ""
        return result.stdout.strip()

    return {
        "branch": run_git("branch", "--show-current"),
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": run_git("status", "--short"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
