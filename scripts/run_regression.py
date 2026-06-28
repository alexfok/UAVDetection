from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UAVDetection regression checks.")
    parser.add_argument("--code", action="store_true", help="Run code regression checks.")
    parser.add_argument("--setup", action="store_true", help="Run setup/environment diagnostics.")
    parser.add_argument("--require-node", action="store_true", help="Fail if Node.js is unavailable for JS checks.")
    parser.add_argument("--skip-js", action="store_true", help="Skip JavaScript syntax checks.")
    parser.add_argument("--setup-camera", default="", help="Configured camera id for setup diagnostics.")
    parser.add_argument("--setup-profile", default="main", choices=["main", "preview"], help="Camera profile for setup diagnostics.")
    parser.add_argument("--setup-seconds", type=float, default=3.0, help="Camera probe duration for setup diagnostics.")
    parser.add_argument("--setup-privacy", default="normal", choices=["normal", "high"], help="Redaction level for setup diagnostics.")
    parser.add_argument("--setup-refresh-stats", action="store_true", help="Refresh data_store stats during setup diagnostics.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.code and not args.setup:
        args.code = True
    if args.setup:
        return run_setup_checks(args)
    return run_code_checks(require_node=args.require_node, skip_js=args.skip_js)


def run_setup_checks(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "uav_pycache"))
    command = [
        sys.executable,
        "scripts/run_environment_diagnostics.py",
        "quick",
        "--profile",
        args.setup_profile,
        "--seconds",
        str(args.setup_seconds),
        "--privacy",
        args.setup_privacy,
    ]
    if args.setup_camera:
        command.extend(["--camera", args.setup_camera])
    else:
        command.append("--no-camera")
    if args.setup_refresh_stats:
        command.append("--refresh-stats")
    return run(command, env)


def run_code_checks(require_node: bool, skip_js: bool) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "uav_pycache"))
    commands = [
        [sys.executable, "-m", "py_compile", *python_files()],
        [sys.executable, "-m", "html.parser", "web/annotator/index.html"],
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
    ]
    if not skip_js:
        node = shutil.which("node")
        if node:
            commands.insert(2, [node, "--check", "web/annotator/app.js"])
        elif require_node:
            print("Node.js is required for JavaScript syntax checks but was not found.", file=sys.stderr)
            return 1
        else:
            print("Skipping JavaScript syntax check: node not found.")

    for command in commands:
        result = run(command, env)
        if result != 0:
            return result
    return 0


def python_files() -> list[str]:
    roots = ["app", "scripts", "tests"]
    files: list[str] = []
    for root in roots:
        files.extend(str(path.relative_to(PROJECT_ROOT)) for path in sorted((PROJECT_ROOT / root).rglob("*.py")))
    return files


def run(command: list[str], env: dict[str, str]) -> int:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
