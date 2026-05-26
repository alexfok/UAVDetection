from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_STORE = Path("data_store")
DRIVE_FOLDER_ID = "16qqTwiknaYpYArNKG_r-JaA7dUA816w9"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/u/0/folders/{DRIVE_FOLDER_ID}"
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DATASET_IMAGE_EXTENSIONS = IMAGE_EXTENSIONS | {".jpg", ".jpeg"}

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
)
SYNC_EXCLUDED_ROOT_FILES = {"README.md", ".gitkeep"}
RCLONE_SYNC_FILTERS = [
    "--exclude",
    "/README.md",
    "--exclude",
    "/.gitkeep",
]

LEGACY_LINKS = (
    ("videos/Roni/raw_data", "raw_data/Roni"),
    ("annotations/web_drone_v1", "datasets/web_drone_v1"),
    ("reports", "detection_results"),
    ("certs", "system_config/certs"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage and synchronize the UAVDetection local data store.")
    parser.add_argument("--data-store", type=Path, default=DEFAULT_DATA_STORE)

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create the canonical data store structure.")
    init_parser.add_argument("--migrate-legacy", action="store_true", help="Move current local data into data_store.")
    init_parser.add_argument("--no-legacy-links", action="store_true", help="Do not create compatibility symlinks.")
    init_parser.add_argument("--username", default=os.environ.get("ANNOTATION_SERVER_USERNAME", "admin"))
    init_parser.add_argument("--password", default=os.environ.get("ANNOTATION_SERVER_PASSWORD", "admin123"))

    subparsers.add_parser("stats", help="Refresh data_store/stats/dataset_stats.json.")
    subparsers.add_parser("doctor", help="Check local structure and sync tool availability.")

    for name in ("backup", "restore", "sync-up", "sync-down", "bisync"):
        command_parser = subparsers.add_parser(name, help=f"Run {name} against the remote data store.")
        add_sync_args(command_parser)
        if name == "restore":
            command_parser.add_argument("--backup-name", help="Backup folder name. Defaults to newest backup.")
            command_parser.add_argument(
                "--replace",
                action="store_true",
                help="Restore directly into --data-store. Without this, restore into a timestamped side folder.",
            )
        if name == "bisync":
            command_parser.add_argument("--resync", action="store_true", help="Pass --resync to rclone bisync.")

    return parser.parse_args()


def add_sync_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("rclone", "local"), default="rclone")
    parser.add_argument(
        "--remote",
        default=os.environ.get("UAV_DATASTORE_RCLONE_REMOTE", "uavdrive:"),
        help="Rclone remote. Configure it to point at the Google Drive folder root.",
    )
    parser.add_argument(
        "--remote-path",
        default=os.environ.get("UAV_DATASTORE_REMOTE_PATH", "current"),
        help="Path under the remote root for current synchronized data.",
    )
    parser.add_argument(
        "--local-remote-path",
        type=Path,
        default=os.environ.get("UAV_DATASTORE_LOCAL_REMOTE", ""),
        help="Mounted Google Drive folder path for --backend local.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Acknowledge overwrite/delete-capable operations.")


def main() -> int:
    args = parse_args()
    data_store = resolve_project_path(args.data_store)

    if args.command == "init":
        init_datastore(data_store, migrate_legacy=args.migrate_legacy, create_links=not args.no_legacy_links)
        write_system_config(data_store, args.username, args.password)
        normalize_datasets(data_store)
        stats_path = write_stats(data_store)
        print(f"Data store ready: {data_store}")
        print(f"Wrote stats: {stats_path}")
        return 0

    if args.command == "stats":
        normalize_datasets(data_store)
        print(f"Wrote stats: {write_stats(data_store)}")
        return 0

    if args.command == "doctor":
        doctor(data_store)
        return 0

    if args.command == "backup":
        ensure_datastore(data_store)
        if not args.dry_run:
            write_stats(data_store)
        backup(data_store, args)
        return 0

    if args.command == "restore":
        restore(data_store, args)
        return 0

    if args.command == "sync-up":
        ensure_datastore(data_store)
        if not args.dry_run:
            write_stats(data_store)
        sync_up(data_store, args)
        return 0

    if args.command == "sync-down":
        require_yes(args, "sync-down can delete local files that are absent from the remote.")
        sync_down(data_store, args)
        if not args.dry_run:
            init_datastore(data_store, migrate_legacy=False, create_links=True)
            normalize_datasets(data_store)
            stats_path = write_stats(data_store)
            print(f"Local data store ready: {data_store}")
            print(f"Wrote stats: {stats_path}")
        return 0

    if args.command == "bisync":
        require_yes(args, "bisync is a two-way operation. Run only after testing with --dry-run.")
        bisync(data_store, args)
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


def init_datastore(data_store: Path, migrate_legacy: bool, create_links: bool) -> None:
    for relative in DATASTORE_DIRS:
        (data_store / relative).mkdir(parents=True, exist_ok=True)

    if migrate_legacy:
        for legacy_relative, target_relative in LEGACY_LINKS:
            legacy = PROJECT_ROOT / legacy_relative
            target = data_store / target_relative
            move_legacy_path(legacy, target)

    if create_links:
        for legacy_relative, target_relative in LEGACY_LINKS:
            legacy = PROJECT_ROOT / legacy_relative
            target = data_store / target_relative
            create_compat_link(legacy, target)


def write_system_config(data_store: Path, username: str, password: str) -> None:
    system_config = data_store / "system_config"
    env_path = system_config / "annotation_server.env"
    if not env_path.exists():
        env_path.write_text(
            "\n".join(
                [
                    f"ANNOTATION_SERVER_USERNAME={username}",
                    f"ANNOTATION_SERVER_PASSWORD={password}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    metadata_path = system_config / "datastore.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    metadata.update(
        {
            "schema_version": 1,
            "google_drive_folder_id": DRIVE_FOLDER_ID,
            "google_drive_folder_url": DRIVE_FOLDER_URL,
            "layout": {
                "raw_data": "raw source media",
                "detection_results": "timestamped assessment outputs and comparison reports",
                "datasets": "YOLO train/val image and label datasets",
                "models": "base, external, and trained model weights",
                "system_config": "local deployment settings, users/passwords, certs",
                "stats": "generated dataset, model, and raw-data summaries",
            },
        }
    )
    metadata.setdefault("created_at", datetime.now().astimezone().isoformat())
    metadata["updated_at"] = datetime.now().astimezone().isoformat()
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def write_stats(data_store: Path) -> Path:
    stats = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "raw_data": raw_data_stats(data_store / "raw_data"),
        "datasets": dataset_stats(data_store / "datasets"),
        "models": model_stats(data_store / "models"),
        "detection_results": detection_result_stats(data_store / "detection_results"),
    }
    stats_path = data_store / "stats" / "dataset_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats_path


def normalize_datasets(data_store: Path) -> None:
    datasets_root = data_store / "datasets"
    if not datasets_root.exists():
        return
    for dataset_dir in sorted(path for path in datasets_root.iterdir() if path.is_dir()):
        normalize_data_yaml(dataset_dir)
        normalize_manifest(dataset_dir, data_store)


def normalize_data_yaml(dataset_dir: Path) -> None:
    class_name = read_class_name(dataset_dir / "data.yaml")
    content = (
        f"path: {json.dumps(project_relative_path(dataset_dir))}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {class_name}\n"
    )
    (dataset_dir / "data.yaml").write_text(content, encoding="utf-8")


def project_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def read_class_name(data_yaml: Path) -> str:
    if not data_yaml.exists():
        return "drone"
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("0:"):
            return stripped.split(":", 1)[1].strip() or "drone"
    return "drone"


def normalize_manifest(dataset_dir: Path, data_store: Path) -> None:
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        return
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []
    if not rows:
        return
    changed = False
    source_index = build_raw_source_index(data_store / "raw_data")
    for row in rows:
        image_id = row.get("image_id", "")
        split = row.get("split", "train") or "train"
        if image_id:
            image_path = f"data_store/datasets/{dataset_dir.name}/images/{split}/{image_id}.jpg"
            label_path = f"data_store/datasets/{dataset_dir.name}/labels/{split}/{image_id}.txt"
            changed |= row.get("image_path") != image_path
            changed |= row.get("label_path") != label_path
            row["image_path"] = image_path
            row["label_path"] = label_path
        source_name = Path(row.get("source_media", "")).name
        if source_name in source_index:
            changed |= row.get("source_media") != source_index[source_name]
            row["source_media"] = source_index[source_name]
    if changed:
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def build_raw_source_index(raw_root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    if not raw_root.exists():
        return index
    for path in raw_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS:
            index[path.name] = str(path.relative_to(PROJECT_ROOT))
    return index


def raw_data_stats(raw_root: Path) -> dict[str, object]:
    files = videos = images = bytes_total = 0
    by_source: Counter[str] = Counter()
    if raw_root.exists():
        for path in raw_root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in VIDEO_EXTENSIONS and suffix not in IMAGE_EXTENSIONS:
                continue
            files += 1
            videos += int(suffix in VIDEO_EXTENSIONS)
            images += int(suffix in IMAGE_EXTENSIONS)
            bytes_total += path.stat().st_size
            source = path.relative_to(raw_root).parts[0] if path.relative_to(raw_root).parts else "root"
            by_source[source] += 1
    return {
        "files": files,
        "videos": videos,
        "images": images,
        "bytes": bytes_total,
        "by_source": dict(sorted(by_source.items())),
    }


def dataset_stats(datasets_root: Path) -> dict[str, object]:
    datasets: dict[str, object] = {}
    if not datasets_root.exists():
        return datasets
    for dataset_dir in sorted(path for path in datasets_root.iterdir() if path.is_dir()):
        datasets[dataset_dir.name] = single_dataset_stats(dataset_dir)
    return datasets


def single_dataset_stats(dataset_dir: Path) -> dict[str, object]:
    splits: dict[str, object] = {}
    totals = Counter()
    for split in ("train", "val"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        image_stems = {
            path.stem
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in DATASET_IMAGE_EXTENSIONS
        }
        label_paths = [path for path in label_dir.rglob("*.txt") if path.is_file()]
        label_stems = {path.stem for path in label_paths}
        positive = negative = boxes = 0
        for label_path in label_paths:
            count = count_label_lines(label_path)
            boxes += count
            if count:
                positive += 1
            else:
                negative += 1
        split_stats = {
            "total": len(image_stems | label_stems),
            "positive": positive,
            "negative": negative,
            "boxes": boxes,
            "unlabeled": len(image_stems - label_stems),
        }
        splits[split] = split_stats
        totals.update(split_stats)
    return {"splits": splits, "total": dict(totals)}


def detection_result_stats(results_root: Path) -> dict[str, object]:
    runs = []
    if results_root.exists():
        for path in sorted(results_root.iterdir()):
            if path.is_dir() and (path / "assessment.json").exists():
                runs.append(path.name)
    return {"runs": len(runs), "run_names": runs[-20:]}


def model_stats(models_root: Path) -> dict[str, object]:
    model_files = []
    bytes_total = 0
    if models_root.exists():
        for path in sorted(models_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".pt", ".onnx", ".engine"}:
                continue
            size = path.stat().st_size
            bytes_total += size
            model_files.append({"path": str(path.relative_to(models_root)), "bytes": size})
    return {
        "files": len(model_files),
        "bytes": bytes_total,
        "items": model_files,
    }


def count_label_lines(label_path: Path) -> int:
    try:
        return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def backup(data_store: Path, args: argparse.Namespace) -> None:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    destination = remote_join(args, "backups", f"uav_datastore_{stamp}")
    copy_or_sync(data_store, destination, args, sync=False)
    print(f"Backup written to {destination}")


def restore(data_store: Path, args: argparse.Namespace) -> None:
    backup_name = args.backup_name or newest_backup(args)
    source = remote_join(args, "backups", backup_name)
    if args.replace:
        require_yes(args, "restore --replace overwrites the local data store.")
        destination = data_store
    else:
        destination = data_store.with_name(f"{data_store.name}_restored_{timestamp()}")
    copy_or_sync(source, destination, args, sync=False)
    print(f"Restored {source} to {destination}")


def sync_up(data_store: Path, args: argparse.Namespace) -> None:
    destination = remote_join(args, args.remote_path)
    copy_or_sync(data_store, destination, args, sync=True)
    print(f"Synchronized local data store to {destination}")


def sync_down(data_store: Path, args: argparse.Namespace) -> None:
    source = remote_join(args, args.remote_path)
    copy_or_sync(source, data_store, args, sync=True)
    print(f"Synchronized {source} to local data store")


def bisync(data_store: Path, args: argparse.Namespace) -> None:
    if args.backend != "rclone":
        raise SystemExit("bisync requires --backend rclone")
    require_rclone()
    command = ["rclone", "bisync", str(data_store), remote_join(args, args.remote_path), *RCLONE_SYNC_FILTERS]
    if args.resync:
        command.append("--resync")
    if args.dry_run:
        command.append("--dry-run")
    run_command(command)


def copy_or_sync(source: Path | str, destination: Path | str, args: argparse.Namespace, sync: bool) -> None:
    if args.backend == "rclone":
        require_rclone()
        command = ["rclone", "sync" if sync else "copy", str(source), str(destination), *RCLONE_SYNC_FILTERS]
        if args.dry_run:
            command.append("--dry-run")
        run_command(command)
        return

    if args.backend == "local":
        local_source = Path(source) if isinstance(source, str) else source
        local_destination = Path(destination) if isinstance(destination, str) else destination
        local_copy(local_source, local_destination, delete_extra=sync, dry_run=args.dry_run)
        return

    raise SystemExit(f"Unsupported backend: {args.backend}")


def local_copy(source: Path, destination: Path, delete_extra: bool, dry_run: bool) -> None:
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if is_sync_excluded(relative):
            continue
        target = destination / relative
        if path.is_dir():
            if not dry_run:
                target.mkdir(parents=True, exist_ok=True)
            continue
        print(f"copy {path} -> {target}")
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    if delete_extra and destination.exists():
        source_paths = {path.relative_to(source) for path in source.rglob("*")}
        for path in sorted(destination.rglob("*"), reverse=True):
            relative = path.relative_to(destination)
            if relative in source_paths or is_sync_excluded(relative):
                continue
            print(f"delete {path}")
            if not dry_run:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()


def is_sync_excluded(relative: Path) -> bool:
    return len(relative.parts) == 1 and relative.name in SYNC_EXCLUDED_ROOT_FILES


def newest_backup(args: argparse.Namespace) -> str:
    if args.backend == "rclone":
        require_rclone()
        remote = remote_join(args, "backups")
        result = subprocess.run(["rclone", "lsf", "--dirs-only", remote], check=True, capture_output=True, text=True)
        names = sorted(name.rstrip("/") for name in result.stdout.splitlines() if name.strip())
    else:
        backup_root = remote_join(args, "backups")
        names = sorted(path.name for path in Path(backup_root).iterdir() if path.is_dir())
    if not names:
        raise SystemExit("No backups found.")
    return names[-1]


def remote_join(args: argparse.Namespace, *parts: str) -> str:
    clean_parts = [str(part).strip("/") for part in parts if str(part).strip("/")]
    if args.backend == "local":
        if not args.local_remote_path:
            raise SystemExit("--backend local requires --local-remote-path")
        return str(Path(args.local_remote_path).expanduser().joinpath(*clean_parts))

    remote = args.remote
    suffix = "/".join(clean_parts)
    if not suffix:
        return remote
    if remote.endswith(":") or remote.endswith("/"):
        return f"{remote}{suffix}"
    return f"{remote}/{suffix}"


def move_legacy_path(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print(f"move {source} -> {target}")
        shutil.move(str(source), str(target))
        return
    if source.is_dir() and target.is_dir():
        print(f"merge {source} -> {target}")
        merge_tree(source, target)
        source.rmdir()
        return
    raise SystemExit(f"Cannot migrate {source}: target already exists at {target}")


def merge_tree(source: Path, target: Path) -> None:
    for child in source.iterdir():
        destination = target / child.name
        if child.is_dir() and destination.is_dir():
            merge_tree(child, destination)
            child.rmdir()
        elif not destination.exists():
            shutil.move(str(child), str(destination))
        else:
            conflict = destination.with_name(f"{destination.name}.conflict.{timestamp()}")
            shutil.move(str(child), str(conflict))


def create_compat_link(link_path: Path, target: Path) -> None:
    if link_path.is_symlink():
        return
    if link_path.exists():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target, start=link_path.parent)
    print(f"link {link_path} -> {relative_target}")
    link_path.symlink_to(relative_target, target_is_directory=True)


def doctor(data_store: Path) -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data store: {data_store} ({'exists' if data_store.exists() else 'missing'})")
    print(f"Google Drive folder: {DRIVE_FOLDER_URL}")
    print(f"rclone: {shutil.which('rclone') or 'not found'}")
    for relative in DATASTORE_DIRS:
        path = data_store / relative
        print(f"{relative}: {'ok' if path.exists() else 'missing'}")
    for legacy_relative, target_relative in LEGACY_LINKS:
        legacy = PROJECT_ROOT / legacy_relative
        target = data_store / target_relative
        if legacy.is_symlink():
            status = f"link -> {os.readlink(legacy)}"
        elif legacy.exists():
            status = "legacy path exists"
        else:
            status = "missing"
        print(f"{legacy_relative}: {status} (target {target})")


def ensure_datastore(data_store: Path) -> None:
    missing = [relative for relative in DATASTORE_DIRS if not (data_store / relative).exists()]
    if missing:
        raise SystemExit(f"Data store is not initialized. Missing: {', '.join(missing)}")


def require_rclone() -> None:
    if not shutil.which("rclone"):
        raise SystemExit("rclone is not installed. Install/configure rclone or use --backend local.")


def require_yes(args: argparse.Namespace, message: str) -> None:
    if args.dry_run:
        return
    if not args.yes:
        raise SystemExit(f"{message} Re-run with --yes after reviewing the command.")


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def resolve_project_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
