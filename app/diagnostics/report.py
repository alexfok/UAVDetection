from __future__ import annotations

from pathlib import Path
from typing import Any


def render_report(manifest: dict[str, Any], checks: list[dict[str, Any]]) -> str:
    counts = status_counts(checks)
    lines = [
        f"# {manifest['sysdump_id']} Report",
        "",
        "## Executive Summary",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Host: `{manifest.get('host', 'unknown')}`",
        f"- Mode: `{manifest.get('mode', 'quick')}`",
        f"- Checks: `{counts.get('pass', 0)}` pass, `{counts.get('warn', 0)}` warn, `{counts.get('fail', 0)}` fail, `{counts.get('skip', 0)}` skip",
        f"- Sysdump archive: `{Path(str(manifest.get('archive_path', ''))).name}`",
        "",
    ]

    blocking = [check for check in checks if check["status"] == "fail"]
    warnings = [check for check in checks if check["status"] == "warn"]
    lines.extend(section("Blocking Failures", blocking))
    lines.extend(section("Warnings", warnings))
    lines.extend(["## Check Details", ""])
    lines.extend(["| Status | Category | Check | Detail |", "|---|---|---|---|"])
    for check in checks:
        lines.append(
            f"| {check['status']} | {check['category']} | {check['name']} | "
            f"{markdown_inline(check.get('detail', ''))} |"
        )
    lines.extend(
        [
            "",
            "## Suggested Next Actions",
            "",
            *suggestions(checks),
            "",
        ]
    )
    return "\n".join(lines)


def status_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
    for check in checks:
        status = str(check.get("status") or "skip")
        counts[status] = counts.get(status, 0) + 1
    return counts


def section(title: str, checks: list[dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", ""]
    if not checks:
        lines.extend(["None.", ""])
        return lines
    for check in checks:
        lines.append(f"- `{check['category']}` / `{check['name']}`: {check.get('detail', '')}")
    lines.append("")
    return lines


def suggestions(checks: list[dict[str, Any]]) -> list[str]:
    output: list[str] = []
    failed_names = {str(check.get("name", "")) for check in checks if check.get("status") == "fail"}
    warning_names = {str(check.get("name", "")) for check in checks if check.get("status") == "warn"}
    if "default_model" in failed_names:
        output.append("- Sync or restore `data_store/models/trained/yolov8n_drone_best.pt` before live detection.")
    if "camera_config" in failed_names:
        output.append("- Fix `data_store/system_config/cameras.yaml` before testing named cameras.")
    if "camera_reachability" in failed_names:
        output.append("- Verify camera power, LAN, address, and RTSP port before debugging credentials.")
    if "camera_frame_read" in failed_names:
        output.append("- Camera port is reachable but frame read failed; check credentials, RTSP path, codec, or profile.")
    if "disk_space" in warning_names:
        output.append("- Free disk space or sync/delete old recordings before long field sessions.")
    if not output:
        output.append("- No blocking setup failures were detected. If detection quality is poor, review model thresholds, stream profile, and labeling coverage.")
    return output


def markdown_inline(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
