from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("data_store/raw_data/Roni/demo_corrected_record_1107_15-57_segments_10-16.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild timing-corrected demo clips as browser-compatible H.264 MP4 files."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--apply", action="store_true", help="Create backups and replace the corrected outputs.")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fourcc_text(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))


def probe_video(path: Path, cv2_module) -> dict[str, object]:
    capture = cv2_module.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"unable to open video: {path}")
        fps = float(capture.get(cv2_module.CAP_PROP_FPS))
        frames = int(capture.get(cv2_module.CAP_PROP_FRAME_COUNT))
        return {
            "codec": fourcc_text(capture.get(cv2_module.CAP_PROP_FOURCC)),
            "fps": fps,
            "frames": frames,
            "width": int(capture.get(cv2_module.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT)),
            "duration": frames / fps if fps > 0 else 0.0,
        }
    finally:
        capture.release()


def encode_h264(source: Path, destination: Path, fps: float, cv2_module) -> int:
    capture = cv2_module.VideoCapture(str(source))
    writer = None
    frames_written = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(f"unable to open source: {source}")
        width = int(capture.get(cv2_module.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT))
        writer = cv2_module.VideoWriter(
            str(destination),
            cv2_module.VideoWriter_fourcc(*"avc1"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("the installed OpenCV build cannot create H.264/avc1 MP4 files")
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            writer.write(frame)
            frames_written += 1
    finally:
        if writer is not None:
            writer.release()
        capture.release()
    return frames_written


def main() -> int:
    args = parse_args()
    manifest_path = resolve_project_path(args.manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        raise SystemExit(f"manifest has no clips: {manifest_path}")

    print(f"Manifest: {manifest_path}")
    for clip in clips:
        print(
            f"segment {clip['segment']}: {clip['source']} -> {clip['output']} "
            f"at {float(clip['corrected_fps_encoded']):.3f} FPS"
        )
    if not args.apply:
        print("Dry run only. Pass --apply to encode, validate, back up, and replace the outputs.")
        return 0

    import cv2

    temporary_outputs: list[tuple[dict[str, object], Path, Path, dict[str, object]]] = []
    created_temporaries: list[Path] = []
    try:
        for clip in clips:
            source = resolve_project_path(str(clip["source"]))
            output = resolve_project_path(str(clip["output"]))
            fps = float(clip["corrected_fps_encoded"])
            temporary = output.with_name(f".{output.stem}.browser_h264.tmp.mp4")
            if temporary.exists():
                temporary.unlink()
            created_temporaries.append(temporary)
            expected = probe_video(source, cv2)
            written = encode_h264(source, temporary, fps, cv2)
            actual = probe_video(temporary, cv2)
            if actual["codec"].lower() not in {"avc1", "h264"}:
                raise RuntimeError(f"unexpected output codec for {temporary}: {actual['codec']}")
            if written != expected["frames"] or actual["frames"] != expected["frames"]:
                raise RuntimeError(
                    f"frame-count mismatch for segment {clip['segment']}: "
                    f"source={expected['frames']} written={written} output={actual['frames']}"
                )
            if abs(float(actual["fps"]) - fps) > 0.002:
                raise RuntimeError(
                    f"FPS mismatch for segment {clip['segment']}: requested={fps} output={actual['fps']}"
                )
            temporary_outputs.append((clip, output, temporary, actual))
            print(
                f"validated segment {clip['segment']}: codec={actual['codec']} "
                f"frames={actual['frames']} fps={actual['fps']:.3f} duration={actual['duration']:.3f}s"
            )

        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        backup_dir = PROJECT_ROOT / "data_store" / "backups" / f"demo_corrected_mp4v_{stamp}"
        backup_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(manifest_path, backup_dir / manifest_path.name)
        converted_at = datetime.now().astimezone().isoformat()
        for clip, output, temporary, actual in temporary_outputs:
            if output.exists():
                shutil.copy2(output, backup_dir / output.name)
                clip["previous_output_codec"] = probe_video(output, cv2)["codec"]
                clip["previous_output_sha256"] = sha256(output)
            temporary.replace(output)
            clip["output_codec"] = actual["codec"]
            clip["output_bytes"] = output.stat().st_size
            clip["output_sha256"] = sha256(output)
            clip["browser_compatible_h264"] = True
            clip["browser_conversion_at"] = converted_at

        payload["browser_compatibility"] = {
            "codec": "H.264/avc1",
            "converted_at": converted_at,
            "backup_dir": str(backup_dir.relative_to(PROJECT_ROOT)),
        }
        manifest_temporary = manifest_path.with_suffix(".json.tmp")
        manifest_temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifest_temporary.replace(manifest_path)
        print(f"Backup: {backup_dir}")
        print(f"Updated manifest: {manifest_path}")
    finally:
        for temporary in created_temporaries:
            if temporary.exists():
                temporary.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
