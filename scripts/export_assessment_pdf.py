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
RIGHT = 0.965
TOP = 0.92
BOTTOM = 0.08
LINE = 0.028


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a media assessment run folder to PDF.")
    parser.add_argument("run_dir", type=Path, help="Run folder containing assessment.json and assessment.md.")
    parser.add_argument("--output", type=Path, help="Output PDF path. Defaults to <run_dir>/assessment.pdf.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_path = args.run_dir / "assessment.json"
    md_path = args.run_dir / "assessment.md"
    output_path = args.output or args.run_dir / "assessment.pdf"

    assessments = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = parse_markdown_metadata(md_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        add_cover_page(pdf, args.run_dir, metadata, assessments)
        add_section_pages(pdf, "Good Movies", filter_items(assessments, "video", "good"), video_rows)
        add_section_pages(pdf, "Neutral Movies", filter_items(assessments, "video", "neutral"), video_rows)
        add_section_pages(pdf, "Bad Movies", filter_items(assessments, "video", "bad"), video_rows)
        add_section_pages(pdf, "Good Images", filter_items(assessments, "image", "good"), image_rows)
        add_section_pages(pdf, "Neutral Images", filter_items(assessments, "image", "neutral"), image_rows)
        add_section_pages(pdf, "Bad Images", filter_items(assessments, "image", "bad"), image_rows)

        info = pdf.infodict()
        info["Title"] = "Initial Media Detection Assessment"
        info["Author"] = "UAV Detection Assessment Pipeline"
        info["Subject"] = "Media categorization report"
        info["CreationDate"] = datetime.now()

    print(output_path)
    return 0


def parse_markdown_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not path.exists():
        return metadata
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("- "):
            continue
        text = line[2:]
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        metadata[key.strip()] = value.strip().strip("`")
    return metadata


def filter_items(items: list[dict[str, object]], kind: str, status: str) -> list[dict[str, object]]:
    return [item for item in items if item["kind"] == kind and item["status"] == status]


def add_cover_page(
    pdf: PdfPages,
    run_dir: Path,
    metadata: dict[str, str],
    assessments: list[dict[str, object]],
) -> None:
    videos = [item for item in assessments if item["kind"] == "video"]
    images = [item for item in assessments if item["kind"] == "image"]
    video_counts = Counter(item["status"] for item in videos)
    image_counts = Counter(item["status"] for item in images)

    fig, ax = new_page()
    y = TOP
    y = draw_title(ax, y, "Initial Media Detection Assessment")
    y = draw_wrapped(ax, y, f"Generated: {metadata.get('Generated', '-')}", size=10)
    y = draw_wrapped(ax, y, f"Dataset: {metadata.get('Dataset', '-')}", size=10)
    y = draw_wrapped(ax, y, f"Output run folder: {run_dir}", size=10)
    y = draw_wrapped(ax, y, f"Model: {metadata.get('Model', '-')}", size=10)
    y = draw_wrapped(ax, y, f"Confidence / IoU / image size: {metadata.get('Confidence / IoU / image size', '-')}", size=10)
    y = draw_wrapped(ax, y, f"Video analysis: {metadata.get('Video analysis', '-')}", size=10)
    if "Total elapsed" in metadata:
        y = draw_wrapped(ax, y, f"Total elapsed: {metadata.get('Total elapsed')}", size=10)
    if "Media processing" in metadata:
        y = draw_wrapped(ax, y, f"Media processing: {metadata.get('Media processing')}", size=10)
    y -= LINE

    y = draw_heading(ax, y, "Media Definitions")
    definitions = [
        (
            "Good media",
            "The model found at least one configured UAV-like target label. With the current general model, "
            "the available UAV-like proxy labels are airplane, bird, and kite.",
        ),
        (
            "Neutral media",
            "The model found one or more objects, but none of the configured UAV-like target labels were detected.",
        ),
        (
            "Bad media",
            "The model did not find any object above the configured confidence threshold.",
        ),
        (
            "Unreadable media",
            "The file could not be opened or decoded by the assessment pipeline.",
        ),
    ]
    for label, description in definitions:
        y = draw_wrapped(ax, y, f"{label}: {description}", size=9.2, bold_prefix=label)
    y -= LINE * 0.5

    y = draw_heading(ax, y, "Model Caveat")
    y = draw_wrapped(
        ax,
        y,
        "This assessment used a general COCO detector. It does not contain true drone, UAV, or quadcopter "
        "classes, so Good items should be treated as candidate UAV/proxy detections rather than confirmed drones.",
        size=9.2,
    )
    y -= LINE

    y = draw_heading(ax, y, "Summary")
    draw_summary_table(ax, y, videos, images, video_counts, image_counts)
    y -= 0.18

    y = draw_heading(ax, y, "Annotated Output Locations")
    for text in [
        f"Videos: {run_dir}/good, {run_dir}/neutral, {run_dir}/bad",
        f"Images: {run_dir}/images/good, {run_dir}/images/neutral, {run_dir}/images/bad",
    ]:
        y = draw_wrapped(ax, y, text, size=9.2)

    draw_footer(fig)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_section_pages(
    pdf: PdfPages,
    title: str,
    items: list[dict[str, object]],
    row_builder,
) -> None:
    if not items:
        fig, ax = new_page()
        y = draw_title(ax, TOP, title)
        draw_wrapped(ax, y, "None.", size=10)
        draw_footer(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    rows = row_builder(items)
    page = 1
    while rows:
        fig, ax = new_page()
        y = draw_title(ax, TOP, title if page == 1 else f"{title} continued")
        y = draw_wrapped(ax, y, f"Total items: {len(items)}", size=9.5)
        y -= LINE * 0.4
        while rows and y > BOTTOM + LINE:
            row_lines = rows[0]
            needed = LINE * (len(row_lines) + 0.8)
            if y - needed < BOTTOM:
                break
            for idx, line in enumerate(row_lines):
                ax.text(
                    LEFT,
                    y,
                    line,
                    fontsize=7.3 if idx else 7.8,
                    family="DejaVu Sans Mono",
                    weight="bold" if idx == 0 else "normal",
                    va="top",
                    color="#17202a" if idx == 0 else "#34495e",
                )
                y -= LINE * 0.82
            y -= LINE * 0.35
            rows.pop(0)
        draw_footer(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page += 1


def video_rows(items: list[dict[str, object]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in items:
        labels = labels_text(item)
        first = (
            f"{Path(str(item['path'])).name} | duration {item.get('duration_seconds') or '-'}s | "
            f"analyzed frames {item['sampled_frames']} | any-object detected frames {item['frames_with_objects']} | "
            f"target-detected frames {item['frames_with_uav_proxy']}"
        )
        rows.append([first, *wrap(f"Labels: {labels}", width=150)])
    return rows


def image_rows(items: list[dict[str, object]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in items:
        first = f"{Path(str(item['path'])).name} | status {item['status']} | object frames {item['frames_with_objects']}"
        rows.append([first, *wrap(f"Labels: {labels_text(item)}", width=150)])
    return rows


def labels_text(item: dict[str, object]) -> str:
    labels = item.get("labels") or []
    if not labels:
        return "-"
    return ", ".join(
        f"{label['label']} x{label['count']} ({float(label['max_confidence']):.2f})"
        for label in labels
    )


def draw_summary_table(
    ax,
    y: float,
    videos: list[dict[str, object]],
    images: list[dict[str, object]],
    video_counts: Counter,
    image_counts: Counter,
) -> None:
    x_positions = [LEFT, 0.22, 0.34, 0.46, 0.58, 0.70]
    headers = ["Media type", "Total", "Good", "Neutral", "Bad", "Unreadable"]
    rows = [
        ["Videos", len(videos), video_counts["good"], video_counts["neutral"], video_counts["bad"], video_counts["unreadable"]],
        ["Images", len(images), image_counts["good"], image_counts["neutral"], image_counts["bad"], image_counts["unreadable"]],
    ]
    for x, header in zip(x_positions, headers):
        ax.text(x, y, header, fontsize=9, weight="bold", va="top")
    y -= LINE
    for row in rows:
        for x, value in zip(x_positions, row):
            ax.text(x, y, str(value), fontsize=9, va="top")
        y -= LINE


def new_page():
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def draw_title(ax, y: float, text: str) -> float:
    ax.text(LEFT, y, text, fontsize=20, weight="bold", va="top", color="#111827")
    return y - LINE * 1.8


def draw_heading(ax, y: float, text: str) -> float:
    ax.text(LEFT, y, text, fontsize=12, weight="bold", va="top", color="#111827")
    return y - LINE * 1.25


def draw_wrapped(ax, y: float, text: str, size: float = 9.5, bold_prefix: str | None = None) -> float:
    for line in wrap(text, width=150) or [""]:
        ax.text(LEFT, y, line, fontsize=size, va="top", color="#1f2937")
        y -= LINE
    return y


def draw_footer(fig) -> None:
    fig.text(0.5, 0.035, "Generated from assessment.md / assessment.json", ha="center", fontsize=7, color="#6b7280")


if __name__ == "__main__":
    raise SystemExit(main())
