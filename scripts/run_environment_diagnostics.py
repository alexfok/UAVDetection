from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.diagnostics import DiagnosticsOptions, run_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run setup/environment diagnostics and build a sysdump tarball.")
    parser.add_argument("mode", nargs="?", choices=["quick", "sysdump"], default="quick")
    parser.add_argument("--output-root", type=Path, default=Path("data_store/detection_results/sysdumps"))
    parser.add_argument("--camera", default="", help="Configured camera id to probe, e.g. ip_camera_196.")
    parser.add_argument("--profile", default="main", choices=["main", "preview"], help="Camera stream profile.")
    parser.add_argument("--seconds", type=float, default=3.0, help="Camera frame-read probe duration.")
    parser.add_argument("--no-camera", action="store_true", help="Skip camera probing.")
    parser.add_argument("--include-performance", action="store_true", help="Reserve performance probe output in the sysdump.")
    parser.add_argument("--refresh-stats", action="store_true", help="Refresh data_store stats before collecting the sysdump.")
    parser.add_argument("--privacy", choices=["normal", "high"], default="normal")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_diagnostics(
        DiagnosticsOptions(
            mode=args.mode,
            output_root=args.output_root,
            camera_id=args.camera,
            camera_profile=args.profile,
            camera_seconds=args.seconds,
            include_camera=not args.no_camera,
            include_performance=args.include_performance,
            refresh_stats=args.refresh_stats,
            privacy=args.privacy,
        )
    )
    print(f"Status: {result.status}")
    print(f"Sysdump: {result.archive_path}")
    print(f"Report: {result.report_path}")
    return 1 if result.status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
