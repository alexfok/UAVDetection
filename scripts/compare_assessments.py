from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PAGE_SIZE = (11.69, 8.27)
LEFT = 0.055
TOP = 0.92
BOTTOM = 0.08
LINE = 0.028


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two media assessment JSON reports.")
    parser.add_argument("baseline_json", type=Path, help="Baseline assessment.json path.")
    parser.add_argument("candidate_json", type=Path, help="Candidate assessment.json path.")
    parser.add_argument("--baseline-name", default="baseline", help="Human-friendly baseline model/run name.")
    parser.add_argument("--candidate-name", default="candidate", help="Human-friendly candidate model/run name.")
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    parser.add_argument("--pdf-output", type=Path, help="Optional PDF output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = load_by_path(args.baseline_json)
    candidate = load_by_path(args.candidate_json)
    baseline_metadata = load_run_metadata(args.baseline_json)
    candidate_metadata = load_run_metadata(args.candidate_json)
    report = build_report(args, baseline, candidate, baseline_metadata, candidate_metadata)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(args.output)
    else:
        print(report)

    if args.pdf_output:
        args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
        write_pdf(args, baseline, candidate, baseline_metadata, candidate_metadata, args.pdf_output)
        print(args.pdf_output)
    return 0


def load_by_path(path: Path) -> dict[str, dict[str, object]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["path"]): item for item in items}


def load_run_metadata(assessment_json: Path) -> dict[str, object] | None:
    candidates = [
        assessment_json.with_name("run_metadata.json"),
        assessment_json.with_name(f"{assessment_json.stem}_run_metadata.json"),
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def build_report(
    args: argparse.Namespace,
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
    baseline_metadata: dict[str, object] | None,
    candidate_metadata: dict[str, object] | None,
) -> str:
    all_paths = sorted(set(baseline) | set(candidate))
    common_paths = [path for path in all_paths if path in baseline and path in candidate]
    missing_from_candidate = [path for path in all_paths if path in baseline and path not in candidate]
    new_in_candidate = [path for path in all_paths if path in candidate and path not in baseline]

    lines = [
        "# Model Assessment Comparison",
        "",
        f"- Baseline: `{args.baseline_name}` from `{args.baseline_json}`",
        f"- Candidate: `{args.candidate_name}` from `{args.candidate_json}`",
        f"- Common files: `{len(common_paths)}`",
        f"- Missing from candidate: `{len(missing_from_candidate)}`",
        f"- New in candidate: `{len(new_in_candidate)}`",
        "",
        "## Summary",
        "",
        "| Run | Kind | Total | Good | Neutral | Bad | Unreadable |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(summary_rows(args.baseline_name, baseline.values()))
    lines.extend(summary_rows(args.candidate_name, candidate.values()))
    lines.extend(timing_lines(args, baseline_metadata, candidate_metadata))
    lines.extend(
        [
            "",
            "## Detection Frame KPI",
            "",
            (
                "`Any-object detected frames` counts analyzed video frames with any model detection. "
                "`Target-detected frames` counts analyzed video frames with the configured target labels "
                "that drive the Good category."
            ),
            "",
            "| File | Baseline status | Baseline any-object frames | Baseline target frames | Candidate status | Candidate any-object frames | Candidate target frames |",
            "|---|---|---:|---:|---|---:|---:|",
        ]
    )
    lines.extend(video_kpi_rows(common_paths, baseline, candidate))

    transitions = [
        (path, baseline[path], candidate[path])
        for path in common_paths
        if baseline[path]["status"] != candidate[path]["status"]
    ]
    candidate_new_good = [
        (path, baseline[path], candidate[path])
        for path in common_paths
        if baseline[path]["status"] != "good" and candidate[path]["status"] == "good"
    ]
    candidate_missed_good = [
        (path, baseline[path], candidate[path])
        for path in common_paths
        if baseline[path]["status"] == "good" and candidate[path]["status"] != "good"
    ]

    lines.extend(
        [
            "",
            "## Key Differences",
            "",
            f"- Status changes: `{len(transitions)}`",
            f"- Candidate newly marks Good: `{len(candidate_new_good)}`",
            f"- Candidate misses baseline Good: `{len(candidate_missed_good)}`",
            "",
        ]
    )
    lines.extend(change_table("Candidate Newly Marks Good", candidate_new_good))
    lines.extend(change_table("Candidate Misses Baseline Good", candidate_missed_good))
    lines.extend(change_table("All Status Changes", transitions))

    if missing_from_candidate:
        lines.extend(path_list("Missing From Candidate", missing_from_candidate))
    if new_in_candidate:
        lines.extend(path_list("New In Candidate", new_in_candidate))
    return "\n".join(lines)


def timing_lines(
    args: argparse.Namespace,
    baseline_metadata: dict[str, object] | None,
    candidate_metadata: dict[str, object] | None,
) -> list[str]:
    lines = ["", "## Timing", ""]
    if not baseline_metadata and not candidate_metadata:
        lines.append("Timing metadata was not found next to the assessment JSON files.")
        return lines

    lines.extend(
        [
            "| Run | Total elapsed | Total seconds | Model load seconds | Media processing seconds |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    total_seconds = 0.0
    for name, metadata in ((args.baseline_name, baseline_metadata), (args.candidate_name, candidate_metadata)):
        if not metadata:
            lines.append(f"| {name} | missing | - | - | - |")
            continue
        seconds = float(metadata.get("elapsed_seconds", 0.0))
        total_seconds += seconds
        lines.append(
            f"| {name} | {metadata.get('elapsed_human', format_elapsed(seconds))} | "
            f"{seconds:.3f} | {float(metadata.get('model_load_seconds', 0.0)):.3f} | "
            f"{float(metadata.get('media_processing_seconds', 0.0)):.3f} |"
        )
    lines.extend(["", f"- Combined elapsed: `{format_elapsed(total_seconds)}` (`{total_seconds:.3f}` seconds)", ""])
    return lines


def format_elapsed(seconds: float) -> str:
    whole_seconds = int(round(seconds))
    minutes, sec = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def video_kpi_rows(
    paths: list[str],
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
) -> list[str]:
    rows: list[str] = []
    for path in paths:
        base = baseline[path]
        cand = candidate[path]
        if base["kind"] != "video":
            continue
        rows.append(
            f"| `{Path(path).name}` | {base['status']} | {base['frames_with_objects']} | "
            f"{base['frames_with_uav_proxy']} | {cand['status']} | {cand['frames_with_objects']} | "
            f"{cand['frames_with_uav_proxy']} |"
        )
    return rows


def summary_rows(name: str, items) -> list[str]:
    rows: list[str] = []
    for kind in ("video", "image"):
        subset = [item for item in items if item["kind"] == kind]
        counts = Counter(item["status"] for item in subset)
        rows.append(
            f"| {name} | {kind} | {len(subset)} | {counts['good']} | {counts['neutral']} | "
            f"{counts['bad']} | {counts['unreadable']} |"
        )
    return rows


def change_table(title: str, rows: list[tuple[str, dict[str, object], dict[str, object]]]) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        lines.extend(["None.", ""])
        return lines

    lines.extend(
        [
            "| File | Kind | Baseline status | Candidate status | Baseline labels | Candidate labels |",
            "|---|---|---|---|---|---|",
        ]
    )
    for path, base, cand in rows:
        lines.append(
            f"| `{Path(path).name}` | {base['kind']} | {base['status']} | {cand['status']} | "
            f"{labels_text(base)} | {labels_text(cand)} |"
        )
    lines.append("")
    return lines


def path_list(title: str, paths: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    lines.extend(f"- `{path}`" for path in paths)
    lines.append("")
    return lines


def labels_text(item: dict[str, object]) -> str:
    labels = item.get("labels") or []
    if not labels:
        return "-"
    return ", ".join(
        f"{label['label']} x{label['count']} ({float(label['max_confidence']):.2f})"
        for label in labels
    )


def write_pdf(
    args: argparse.Namespace,
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
    baseline_metadata: dict[str, object] | None,
    candidate_metadata: dict[str, object] | None,
    output_path: Path,
) -> None:
    common_paths = sorted(set(baseline) & set(candidate))
    transitions = [
        (path, baseline[path], candidate[path])
        for path in common_paths
        if baseline[path]["status"] != candidate[path]["status"]
    ]
    candidate_new_good = [
        (path, baseline[path], candidate[path])
        for path in common_paths
        if baseline[path]["status"] != "good" and candidate[path]["status"] == "good"
    ]
    candidate_missed_good = [
        (path, baseline[path], candidate[path])
        for path in common_paths
        if baseline[path]["status"] == "good" and candidate[path]["status"] != "good"
    ]

    with PdfPages(output_path) as pdf:
        add_cover_page(
            pdf,
            args,
            baseline,
            candidate,
            baseline_metadata,
            candidate_metadata,
            transitions,
            candidate_new_good,
            candidate_missed_good,
        )
        add_video_kpi_pages(pdf, args, common_paths, baseline, candidate)
        add_change_pages(pdf, "Candidate Newly Marks Good", candidate_new_good)
        add_change_pages(pdf, "Candidate Misses Baseline Good", candidate_missed_good)
        add_change_pages(pdf, "All Status Changes", transitions)

        info = pdf.infodict()
        info["Title"] = "Model Assessment Comparison"
        info["Author"] = "UAV Detection Assessment Pipeline"
        info["Subject"] = "Comparative detection report"
        info["CreationDate"] = datetime.now()


def add_cover_page(
    pdf: PdfPages,
    args: argparse.Namespace,
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
    baseline_metadata: dict[str, object] | None,
    candidate_metadata: dict[str, object] | None,
    transitions: list[tuple[str, dict[str, object], dict[str, object]]],
    candidate_new_good: list[tuple[str, dict[str, object], dict[str, object]]],
    candidate_missed_good: list[tuple[str, dict[str, object], dict[str, object]]],
) -> None:
    fig, ax = new_page()
    y = draw_title(ax, TOP, "Model Assessment Comparison")
    y = draw_wrapped(ax, y, f"Baseline: {args.baseline_name} ({args.baseline_json})", size=9.5)
    y = draw_wrapped(ax, y, f"Candidate: {args.candidate_name} ({args.candidate_json})", size=9.5)
    y -= LINE

    y = draw_heading(ax, y, "Summary")
    headers = ["Run", "Kind", "Total", "Good", "Neutral", "Bad", "Unreadable"]
    table_rows = summary_table_rows(args.baseline_name, baseline.values())
    table_rows.extend(summary_table_rows(args.candidate_name, candidate.values()))
    y = draw_simple_table(ax, y, headers, table_rows, widths=[0.30, 0.10, 0.09, 0.09, 0.10, 0.08, 0.12])
    y -= LINE

    y = draw_heading(ax, y, "Timing")
    total_seconds = 0.0
    for name, metadata in ((args.baseline_name, baseline_metadata), (args.candidate_name, candidate_metadata)):
        if not metadata:
            y = draw_wrapped(ax, y, f"{name}: timing metadata missing", size=9.2)
            continue
        seconds = float(metadata.get("elapsed_seconds", 0.0))
        total_seconds += seconds
        y = draw_wrapped(
            ax,
            y,
            (
                f"{name}: {metadata.get('elapsed_human', format_elapsed(seconds))} total; "
                f"{float(metadata.get('media_processing_seconds', 0.0)):.1f}s media processing; "
                f"{float(metadata.get('model_load_seconds', 0.0)):.1f}s model load"
            ),
            size=9.2,
        )
    y = draw_wrapped(ax, y, f"Combined elapsed: {format_elapsed(total_seconds)} ({total_seconds:.1f}s)", size=9.2)
    y -= LINE * 0.5

    y = draw_heading(ax, y, "Key Differences")
    for text in [
        f"Status changes: {len(transitions)}",
        f"Candidate newly marks Good: {len(candidate_new_good)}",
        f"Candidate misses baseline Good: {len(candidate_missed_good)}",
    ]:
        y = draw_wrapped(ax, y, text, size=9.5)

    y -= LINE * 0.5
    y = draw_heading(ax, y, "Detected-Frame KPI Definition")
    for text in [
        "Any-object detected frames: analyzed video frames with any bounding-box detection.",
        "Target-detected frames: analyzed video frames with the configured UAV/proxy or drone target labels.",
        "Target-detected frames are the primary per-video KPI for comparing UAV/drone detection behavior.",
    ]:
        y = draw_wrapped(ax, y, text, size=9.2)

    draw_footer(fig)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_video_kpi_pages(
    pdf: PdfPages,
    args: argparse.Namespace,
    paths: list[str],
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
) -> None:
    rows: list[list[str]] = []
    for path in paths:
        base = baseline[path]
        cand = candidate[path]
        if base["kind"] != "video":
            continue
        rows.append(
            [
                Path(path).name,
                str(base["status"]),
                str(base["frames_with_objects"]),
                str(base["frames_with_uav_proxy"]),
                str(cand["status"]),
                str(cand["frames_with_objects"]),
                str(cand["frames_with_uav_proxy"]),
            ]
        )

    add_table_pages(
        pdf,
        "Per-Video Detected-Frame KPI",
        ["File", "Base status", "Base any", "Base target", "Cand status", "Cand any", "Cand target"],
        rows,
        widths=[0.26, 0.12, 0.09, 0.10, 0.12, 0.09, 0.10],
    )


def add_change_pages(
    pdf: PdfPages,
    title: str,
    rows: list[tuple[str, dict[str, object], dict[str, object]]],
) -> None:
    table_rows = [
        [
            Path(path).name,
            str(base["kind"]),
            f"{base['status']} -> {cand['status']}",
            labels_text(base),
            labels_text(cand),
        ]
        for path, base, cand in rows
    ]
    add_table_pages(
        pdf,
        title,
        ["File", "Kind", "Status", "Baseline labels", "Candidate labels"],
        table_rows,
        widths=[0.22, 0.08, 0.13, 0.24, 0.24],
    )


def add_table_pages(
    pdf: PdfPages,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
) -> None:
    if not rows:
        fig, ax = new_page()
        y = draw_title(ax, TOP, title)
        draw_wrapped(ax, y, "None.", size=10)
        draw_footer(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    page = 1
    index = 0
    while index < len(rows):
        fig, ax = new_page()
        y = draw_title(ax, TOP, title if page == 1 else f"{title} continued")
        y = draw_wrapped(ax, y, f"Rows: {len(rows)}", size=9.5)
        y = draw_table_header(ax, y, headers, widths)
        while index < len(rows) and y > BOTTOM + LINE * 1.5:
            y = draw_table_row(ax, y, rows[index], widths)
            index += 1
        draw_footer(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page += 1


def summary_table_rows(name: str, items) -> list[list[str]]:
    rows: list[list[str]] = []
    for kind in ("video", "image"):
        subset = [item for item in items if item["kind"] == kind]
        counts = Counter(item["status"] for item in subset)
        rows.append(
            [
                name,
                kind,
                str(len(subset)),
                str(counts["good"]),
                str(counts["neutral"]),
                str(counts["bad"]),
                str(counts["unreadable"]),
            ]
        )
    return rows


def draw_simple_table(ax, y: float, headers: list[str], rows: list[list[str]], widths: list[float]) -> float:
    y = draw_table_header(ax, y, headers, widths)
    for row in rows:
        y = draw_table_row(ax, y, row, widths)
    return y


def draw_table_header(ax, y: float, headers: list[str], widths: list[float]) -> float:
    x = LEFT
    for header, width in zip(headers, widths):
        ax.text(x, y, header, fontsize=7.5, weight="bold", va="top", color="#111827")
        x += width
    return y - LINE


def draw_table_row(ax, y: float, row: list[str], widths: list[float]) -> float:
    wrapped_cells = [wrap(str(cell), width=max(8, int(width * 115))) or [""] for cell, width in zip(row, widths)]
    line_count = max(len(cell) for cell in wrapped_cells)
    x = LEFT
    for cell_lines, width in zip(wrapped_cells, widths):
        text = "\n".join(cell_lines[:3])
        ax.text(x, y, text, fontsize=6.8, va="top", color="#1f2937")
        x += width
    return y - (LINE * 0.78 * line_count) - (LINE * 0.25)


def new_page():
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def draw_title(ax, y: float, text: str) -> float:
    ax.text(LEFT, y, text, fontsize=19, weight="bold", va="top", color="#111827")
    return y - LINE * 1.7


def draw_heading(ax, y: float, text: str) -> float:
    ax.text(LEFT, y, text, fontsize=12, weight="bold", va="top", color="#111827")
    return y - LINE * 1.2


def draw_wrapped(ax, y: float, text: str, size: float = 9.5) -> float:
    for line in wrap(text, width=145) or [""]:
        ax.text(LEFT, y, line, fontsize=size, va="top", color="#1f2937")
        y -= LINE
    return y


def draw_footer(fig) -> None:
    fig.text(0.5, 0.035, "Generated from assessment JSON reports", ha="center", fontsize=7, color="#6b7280")


if __name__ == "__main__":
    raise SystemExit(main())
