from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deploy_jetson import COPY_EXCLUDED_NAMES, COPY_EXCLUDED_SUFFIXES, safe_copy_file


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data_store" / "deployment_artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a UAVDetection deployment artifact.")
    parser.add_argument("output_dir", nargs="?", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default="", help="Artifact base name. Defaults to UAVDetection_deploy_<timestamp>.")
    parser.add_argument("--format", choices=("zip", "tar.gz"), default="zip")
    parser.add_argument("--include-data-store", action="store_true", help="Include source data_store. Default is code-only.")
    parser.add_argument("--force", action="store_true", help="Replace an existing artifact with the same name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    artifact_name = args.name or f"UAVDetection_deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifact_path = output_dir / f"{artifact_name}.{args.format}"
    if artifact_path.exists():
        if not args.force:
            raise SystemExit(f"Artifact already exists: {artifact_path}. Use --force to replace it.")
        artifact_path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="uav_deploy_build_") as tmp:
        staging = Path(tmp) / artifact_name
        copy_project(staging, include_data_store=args.include_data_store)
        write_manifest(staging, artifact_name, include_data_store=args.include_data_store)
        if args.format == "zip":
            write_zip(staging, artifact_path)
        else:
            write_tar_gz(staging, artifact_path)

    print(f"Deployment artifact: {artifact_path}")
    print("Target usage after extracting the artifact:")
    print("  python3 scripts/deploy.py preflight --install-dir ~/UAVDetection")
    print("  python3 scripts/deploy.py upgrade --install-dir ~/UAVDetection --skip-deps")
    return 0


def copy_project(target: Path, include_data_store: bool) -> None:
    target.mkdir(parents=True)
    for item in PROJECT_ROOT.iterdir():
        if should_skip(item, include_data_store=include_data_store):
            continue
        destination = target / item.name
        if item.is_symlink():
            copy_symlink_target(item, destination, include_data_store=include_data_store)
        elif item.is_dir():
            shutil.copytree(item, destination, ignore=ignore_factory(include_data_store), copy_function=safe_copy_file)
        elif item.is_file():
            safe_copy_file(item, destination)


def should_skip(path: Path, include_data_store: bool) -> bool:
    if path.name == "data_store" and include_data_store:
        return False
    if path.name in COPY_EXCLUDED_NAMES:
        return True
    if path.suffix in COPY_EXCLUDED_SUFFIXES:
        return True
    return False


def ignore_factory(include_data_store: bool):
    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            if should_skip(Path(directory) / name, include_data_store=include_data_store):
                ignored.add(name)
        return ignored

    return ignore


def copy_symlink_target(source: Path, destination: Path, include_data_store: bool) -> None:
    try:
        resolved = source.resolve(strict=True)
    except OSError:
        return
    if should_skip(resolved, include_data_store=include_data_store):
        return
    if resolved.is_dir():
        shutil.copytree(resolved, destination, ignore=ignore_factory(include_data_store), copy_function=safe_copy_file)
    elif resolved.is_file():
        safe_copy_file(resolved, destination)


def write_manifest(staging: Path, artifact_name: str, include_data_store: bool) -> None:
    manifest = {
        "name": artifact_name,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_root": str(PROJECT_ROOT),
        "include_data_store": include_data_store,
        "artifact_kind": "code+data" if include_data_store else "code-only",
        "git": git_info(),
    }
    (staging / "DEPLOYMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_zip(staging: Path, artifact_path: Path) -> None:
    with zipfile.ZipFile(artifact_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staging.parent))


def write_tar_gz(staging: Path, artifact_path: Path) -> None:
    with tarfile.open(artifact_path, "w:gz") as archive:
        archive.add(staging, arcname=staging.name)


def git_info() -> dict[str, str]:
    def run_git(*args: str) -> str:
        try:
            import subprocess

            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
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
