from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two media assessment JSON reports.")
    parser.add_argument("baseline_json", type=Path, help="Baseline assessment.json path.")
    parser.add_argument("candidate_json", type=Path, help="Candidate assessment.json path.")
    parser.add_argument("--baseline-name", default="baseline", help="Human-friendly baseline model/run name.")
    parser.add_argument("--candidate-name", default="candidate", help="Human-friendly candidate model/run name.")
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = load_by_path(args.baseline_json)
    candidate = load_by_path(args.candidate_json)
    report = build_report(args, baseline, candidate)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(args.output)
    else:
        print(report)
    return 0


def load_by_path(path: Path) -> dict[str, dict[str, object]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["path"]): item for item in items}


def build_report(
    args: argparse.Namespace,
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
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


if __name__ == "__main__":
    raise SystemExit(main())
