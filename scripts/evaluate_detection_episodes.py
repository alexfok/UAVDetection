from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


TEMPLATE_FIELDS = ["episode_id", "session_id", "source_id", "source", "entry_frame", "exit_frame", "notes"]


@dataclass(frozen=True)
class PredictionEpisode:
    uid: str
    session_id: str
    source_id: str
    source: str
    episode_id: str
    entry_frame: int
    exit_frame: int | None
    last_seen_frame: int
    entry_timestamp: str
    source_type: str


@dataclass(frozen=True)
class ReviewedEpisode:
    episode_id: str
    session_id: str
    source_id: str
    source: str
    entry_frame: int
    exit_frame: int | None
    notes: str


@dataclass(frozen=True)
class EntryMatch:
    reviewed: ReviewedEpisode
    prediction: PredictionEpisode | None
    latency_frames: int | None
    on_time: bool


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    source_id: str
    source: str
    first_timestamp: str
    first_frame: int | None
    last_frame: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate first-entry drone detection KPIs from Live Detection event logs."
    )
    parser.add_argument(
        "event_paths",
        nargs="+",
        type=Path,
        help="events.jsonl file, daily live_events directory, or live_events root directory.",
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        help="Reviewed CSV with episode_id, session_id/source_id/source, entry_frame, and optional exit_frame.",
    )
    parser.add_argument("--write-template", type=Path, help="Write a review CSV template and exit.")
    parser.add_argument(
        "--deadline-frames",
        type=int,
        default=10,
        help="Maximum acceptable entry latency in frames for on-time detection.",
    )
    parser.add_argument(
        "--out-gap-frames",
        type=int,
        default=30,
        help="Frame gap used to collapse legacy drone_detected events into episodes when timestamps are missing.",
    )
    parser.add_argument(
        "--out-gap-seconds",
        type=float,
        default=8.0,
        help="Timestamp gap used to collapse legacy drone_detected events into episodes.",
    )
    parser.add_argument(
        "--open-ended-match-window-frames",
        type=int,
        default=300,
        help="Match window when a reviewed episode has no exit_frame.",
    )
    parser.add_argument("--output-md", type=Path, help="Write the Markdown report to this path.")
    parser.add_argument("--output-json", type=Path, help="Write machine-readable metrics to this path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    event_files = find_event_files(args.event_paths)
    events, bad_lines = load_events(event_files)
    predictions = extract_predictions(events, args.out_gap_frames, args.out_gap_seconds)

    if args.write_template:
        write_template(args.write_template, events, predictions)
        print(args.write_template)
        return 0

    reviewed = load_reviewed_episodes(args.episodes) if args.episodes else []
    matches, false_entries = evaluate_entries(
        reviewed,
        predictions,
        deadline_frames=max(0, args.deadline_frames),
        open_window_frames=max(1, args.open_ended_match_window_frames),
    )
    report_data = build_report_data(event_files, bad_lines, reviewed, predictions, matches, false_entries, args)
    report = render_markdown(report_data)

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report, encoding="utf-8")
        print(args.output_md)
    else:
        print(report)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.output_json)
    return 0


def find_event_files(paths: list[Path]) -> list[Path]:
    event_files: list[Path] = []
    for path in paths:
        if path.is_file():
            event_files.append(path)
            continue
        if not path.exists():
            continue
        direct = path / "events.jsonl"
        if direct.exists():
            event_files.append(direct)
            continue
        event_files.extend(sorted(path.rglob("events.jsonl")))
    return sorted(set(event_files))


def load_events(event_files: list[Path]) -> tuple[list[dict[str, object]], int]:
    events: list[dict[str, object]] = []
    bad_lines = 0
    for event_file in event_files:
        try:
            lines = event_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if isinstance(event, dict):
                event["event_log_path"] = str(event_file)
                event["event_line"] = line_number
                events.append(event)
    return events, bad_lines


def extract_predictions(
    events: list[dict[str, object]],
    out_gap_frames: int,
    out_gap_seconds: float,
) -> list[PredictionEpisode]:
    predictions: list[PredictionEpisode] = []
    for group in grouped_events(events).values():
        presence_predictions = extract_presence_predictions(group)
        if presence_predictions:
            predictions.extend(presence_predictions)
        else:
            predictions.extend(extract_legacy_detection_predictions(group, out_gap_frames, out_gap_seconds))
    return sorted(predictions, key=lambda item: (item.session_id, item.source_id, item.source, item.entry_frame))


def grouped_events(events: list[dict[str, object]]) -> dict[tuple[str, str, str, str], list[dict[str, object]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for event in events:
        key = (
            text_value(event.get("session_id")),
            text_value(event.get("source_id")),
            text_value(event.get("source")),
            text_value(event.get("event_log_path")),
        )
        groups.setdefault(key, []).append(event)
    return groups


def extract_presence_predictions(group: list[dict[str, object]]) -> list[PredictionEpisode]:
    entries = [event for event in group if event.get("event_type") == "drone_in_frame"]
    exits = [event for event in group if event.get("event_type") == "drone_out_frame"]
    exit_by_episode: dict[str, list[dict[str, object]]] = {}
    for event in exits:
        exit_by_episode.setdefault(text_value(event.get("episode_id")), []).append(event)

    predictions: list[PredictionEpisode] = []
    for event in sorted(entries, key=lambda row: frame_value(row, "entry_frame_index", "frame_index") or 0):
        episode_id = text_value(event.get("episode_id")) or str(len(predictions) + 1)
        entry_frame = frame_value(event, "entry_frame_index", "frame_index")
        if entry_frame is None:
            continue
        exit_event = first_exit_after(exit_by_episode.get(episode_id, []), entry_frame)
        exit_frame = frame_value(exit_event, "exit_frame_index", "frame_index") if exit_event else None
        last_seen_frame = (
            frame_value(exit_event, "last_seen_frame_index")
            if exit_event
            else frame_value(event, "last_seen_frame_index", "frame_index")
        )
        predictions.append(
            PredictionEpisode(
                uid=prediction_uid(event, "presence", episode_id, entry_frame),
                session_id=text_value(event.get("session_id")),
                source_id=text_value(event.get("source_id")),
                source=text_value(event.get("source")),
                episode_id=episode_id,
                entry_frame=entry_frame,
                exit_frame=exit_frame,
                last_seen_frame=last_seen_frame if last_seen_frame is not None else entry_frame,
                entry_timestamp=text_value(event.get("timestamp")),
                source_type="presence",
            )
        )
    return predictions


def first_exit_after(events: list[dict[str, object]], entry_frame: int) -> dict[str, object] | None:
    exits = [
        event
        for event in events
        if (frame_value(event, "exit_frame_index", "frame_index") or 0) >= entry_frame
    ]
    if not exits:
        return None
    return min(exits, key=lambda row: frame_value(row, "exit_frame_index", "frame_index") or 0)


def extract_legacy_detection_predictions(
    group: list[dict[str, object]],
    out_gap_frames: int,
    out_gap_seconds: float,
) -> list[PredictionEpisode]:
    detections = [
        event
        for event in group
        if event.get("event_type") == "drone_detected" and frame_value(event, "frame_index") is not None
    ]
    detections.sort(key=lambda row: frame_value(row, "frame_index") or 0)
    predictions: list[PredictionEpisode] = []
    if not detections:
        return predictions

    out_gap_frames = max(1, out_gap_frames)
    current_start = detections[0]
    last = detections[0]
    for event in detections[1:]:
        if legacy_episode_gap_exceeded(last, event, out_gap_frames, out_gap_seconds):
            predictions.append(legacy_prediction(current_start, last, len(predictions) + 1))
            current_start = event
        last = event
    predictions.append(legacy_prediction(current_start, last, len(predictions) + 1))
    return predictions


def legacy_episode_gap_exceeded(
    previous: dict[str, object],
    current: dict[str, object],
    out_gap_frames: int,
    out_gap_seconds: float,
) -> bool:
    previous_seconds = timestamp_seconds(previous)
    current_seconds = timestamp_seconds(current)
    if previous_seconds is not None and current_seconds is not None:
        return current_seconds - previous_seconds > max(0.1, out_gap_seconds)
    frame = frame_value(current, "frame_index") or 0
    previous_frame = frame_value(previous, "frame_index") or 0
    return frame - previous_frame > max(1, out_gap_frames)


def legacy_prediction(start: dict[str, object], last: dict[str, object], index: int) -> PredictionEpisode:
    entry_frame = frame_value(start, "frame_index") or 0
    last_seen_frame = frame_value(last, "frame_index") or entry_frame
    episode_id = f"legacy_{index}"
    return PredictionEpisode(
        uid=prediction_uid(start, "legacy", episode_id, entry_frame),
        session_id=text_value(start.get("session_id")),
        source_id=text_value(start.get("source_id")),
        source=text_value(start.get("source")),
        episode_id=episode_id,
        entry_frame=entry_frame,
        exit_frame=last_seen_frame,
        last_seen_frame=last_seen_frame,
        entry_timestamp=text_value(start.get("timestamp")),
        source_type="legacy_detection",
    )


def prediction_uid(event: dict[str, object], source_type: str, episode_id: str, entry_frame: int) -> str:
    return ":".join(
        [
            text_value(event.get("event_log_path")),
            text_value(event.get("session_id")),
            source_type,
            episode_id,
            str(entry_frame),
        ]
    )


def load_reviewed_episodes(path: Path | None) -> list[ReviewedEpisode]:
    if path is None:
        return []
    episodes: list[ReviewedEpisode] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            entry_frame = parse_int(row.get("entry_frame"))
            if entry_frame is None:
                continue
            episodes.append(
                ReviewedEpisode(
                    episode_id=text_value(row.get("episode_id")) or f"review_{row_number}",
                    session_id=text_value(row.get("session_id")),
                    source_id=text_value(row.get("source_id")),
                    source=text_value(row.get("source")),
                    entry_frame=entry_frame,
                    exit_frame=parse_int(row.get("exit_frame")),
                    notes=text_value(row.get("notes")),
                )
            )
    return episodes


def evaluate_entries(
    reviewed: list[ReviewedEpisode],
    predictions: list[PredictionEpisode],
    deadline_frames: int,
    open_window_frames: int,
) -> tuple[list[EntryMatch], list[PredictionEpisode]]:
    matched_prediction_ids: set[str] = set()
    matches: list[EntryMatch] = []

    for review in sorted(reviewed, key=lambda item: (item.session_id, item.source_id, item.source, item.entry_frame)):
        window_end = reviewed_window_end(review, open_window_frames)
        candidates = [
            prediction
            for prediction in predictions
            if prediction.uid not in matched_prediction_ids
            and prediction_matches_scope(review, prediction)
            and review.entry_frame <= prediction.entry_frame <= window_end
        ]
        if not candidates:
            matches.append(EntryMatch(review, None, None, False))
            continue
        prediction = min(candidates, key=lambda item: (item.entry_frame - review.entry_frame, item.entry_frame))
        matched_prediction_ids.add(prediction.uid)
        latency = prediction.entry_frame - review.entry_frame
        matches.append(EntryMatch(review, prediction, latency, latency <= deadline_frames))

    false_entries = [
        prediction
        for prediction in predictions
        if prediction.uid not in matched_prediction_ids and prediction_in_reviewed_scope(prediction, reviewed)
    ]
    return matches, false_entries


def reviewed_window_end(review: ReviewedEpisode, open_window_frames: int) -> int:
    if review.exit_frame is not None:
        return max(review.entry_frame, review.exit_frame)
    return review.entry_frame + open_window_frames


def prediction_matches_scope(review: ReviewedEpisode, prediction: PredictionEpisode) -> bool:
    if review.session_id:
        return review.session_id == prediction.session_id
    if review.source_id:
        return review.source_id == prediction.source_id
    if review.source:
        return review.source == prediction.source
    return True


def prediction_in_reviewed_scope(prediction: PredictionEpisode, reviewed: list[ReviewedEpisode]) -> bool:
    return any(prediction_matches_scope(review, prediction) for review in reviewed)


def build_report_data(
    event_files: list[Path],
    bad_lines: int,
    reviewed: list[ReviewedEpisode],
    predictions: list[PredictionEpisode],
    matches: list[EntryMatch],
    false_entries: list[PredictionEpisode],
    args: argparse.Namespace,
) -> dict[str, object]:
    detected_matches = [match for match in matches if match.prediction is not None]
    on_time_matches = [match for match in detected_matches if match.on_time]
    latencies = [match.latency_frames for match in detected_matches if match.latency_frames is not None]
    metrics = {
        "event_files": len(event_files),
        "bad_json_lines": bad_lines,
        "reviewed_episodes": len(reviewed),
        "predicted_episodes": len(predictions),
        "presence_predictions": sum(1 for prediction in predictions if prediction.source_type == "presence"),
        "legacy_detection_predictions": sum(
            1 for prediction in predictions if prediction.source_type == "legacy_detection"
        ),
        "detected_entries": len(detected_matches),
        "missed_entries": len(matches) - len(detected_matches),
        "false_entries_in_reviewed_scope": len(false_entries),
        "entry_recall": ratio(len(detected_matches), len(reviewed)),
        "on_time_entries": len(on_time_matches),
        "on_time_recall": ratio(len(on_time_matches), len(reviewed)),
        "deadline_frames": args.deadline_frames,
        "mean_latency_frames": mean(latencies),
        "median_latency_frames": median(latencies),
        "max_latency_frames": max(latencies) if latencies else None,
    }
    return {
        "metrics": metrics,
        "matches": [match_to_dict(match) for match in matches],
        "false_entries": [asdict(prediction) for prediction in false_entries],
        "predictions": [asdict(prediction) for prediction in predictions],
        "event_files": [str(path) for path in event_files],
    }


def match_to_dict(match: EntryMatch) -> dict[str, object]:
    return {
        "reviewed": asdict(match.reviewed),
        "prediction": asdict(match.prediction) if match.prediction else None,
        "latency_frames": match.latency_frames,
        "on_time": match.on_time,
    }


def render_markdown(report_data: dict[str, object]) -> str:
    metrics = report_data["metrics"]
    lines = [
        "# Episode Detection KPI",
        "",
        f"- Event files: `{metrics['event_files']}`",
        f"- Reviewed drone-entry episodes: `{metrics['reviewed_episodes']}`",
        f"- Predicted episodes: `{metrics['predicted_episodes']}` "
        f"(`{metrics['presence_predictions']}` presence, `{metrics['legacy_detection_predictions']}` legacy)",
        f"- Entry recall: `{format_ratio(metrics['entry_recall'])}` "
        f"({metrics['detected_entries']}/{metrics['reviewed_episodes']})",
        f"- On-time entry recall <= {metrics['deadline_frames']} frames: "
        f"`{format_ratio(metrics['on_time_recall'])}` ({metrics['on_time_entries']}/{metrics['reviewed_episodes']})",
        f"- Entry latency frames: mean `{format_number(metrics['mean_latency_frames'])}`, "
        f"median `{format_number(metrics['median_latency_frames'])}`, max `{format_number(metrics['max_latency_frames'])}`",
        f"- Missed entries: `{metrics['missed_entries']}`",
        f"- False entries in reviewed scope: `{metrics['false_entries_in_reviewed_scope']}`",
    ]

    if not report_data["matches"]:
        lines.extend(
            [
                "",
                "No reviewed episode CSV was supplied or no rows had an entry_frame.",
                "Create one with `--write-template`, then fill entry_frame and exit_frame from visual review.",
            ]
        )
        return "\n".join(lines)

    lines.extend(["", "## Reviewed Episodes", ""])
    lines.extend(
        [
            "| Episode | Scope | Entry | Exit | Predicted entry | Latency | Status |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for match in report_data["matches"]:
        reviewed = match["reviewed"]
        prediction = match["prediction"]
        scope = episode_scope(reviewed)
        exit_frame = reviewed["exit_frame"] if reviewed["exit_frame"] is not None else ""
        if prediction:
            status = "on-time" if match["on_time"] else "late"
            lines.append(
                f"| {reviewed['episode_id']} | {scope} | {reviewed['entry_frame']} | {exit_frame} | "
                f"{prediction['entry_frame']} | {match['latency_frames']} | {status} |"
            )
        else:
            lines.append(
                f"| {reviewed['episode_id']} | {scope} | {reviewed['entry_frame']} | {exit_frame} |  |  | missed |"
            )

    if report_data["false_entries"]:
        lines.extend(["", "## False Entry Events", ""])
        lines.extend(["| Source | Session | Entry frame | Kind |", "|---|---|---:|---|"])
        for prediction in report_data["false_entries"]:
            lines.append(
                f"| {prediction['source_id'] or prediction['source']} | {prediction['session_id']} | "
                f"{prediction['entry_frame']} | {prediction['source_type']} |"
            )
    return "\n".join(lines)


def write_template(path: Path, events: list[dict[str, object]], predictions: list[PredictionEpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sessions = session_infos(events)
    predictions_by_session: dict[str, list[PredictionEpisode]] = {}
    for prediction in predictions:
        predictions_by_session.setdefault(prediction.session_id, []).append(prediction)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEMPLATE_FIELDS)
        writer.writeheader()
        row_index = 1
        for session in sessions:
            session_predictions = predictions_by_session.get(session.session_id, [])
            if not session_predictions:
                writer.writerow(template_row(row_index, session, None))
                row_index += 1
                continue
            for prediction in session_predictions:
                writer.writerow(template_row(row_index, session, prediction))
                row_index += 1


def template_row(
    row_index: int,
    session: SessionInfo,
    prediction: PredictionEpisode | None,
) -> dict[str, object]:
    notes = f"review visible drone entry/exit; session frames {session.first_frame or ''}-{session.last_frame or ''}"
    if prediction:
        exit_frame = prediction.exit_frame if prediction.exit_frame is not None else ""
        notes = f"prediction entry {prediction.entry_frame}, exit {exit_frame}; {notes}"
    return {
        "episode_id": f"review_{row_index}",
        "session_id": session.session_id,
        "source_id": session.source_id,
        "source": session.source,
        "entry_frame": "",
        "exit_frame": "",
        "notes": notes,
    }


def session_infos(events: list[dict[str, object]]) -> list[SessionInfo]:
    sessions: dict[str, dict[str, object]] = {}
    for event in events:
        session_id = text_value(event.get("session_id"))
        if not session_id:
            continue
        session = sessions.setdefault(
            session_id,
            {
                "session_id": session_id,
                "source_id": text_value(event.get("source_id")),
                "source": text_value(event.get("source")),
                "first_timestamp": text_value(event.get("timestamp")),
                "first_frame": None,
                "last_frame": None,
            },
        )
        frame = frame_value(event, "frame_index", "frames_seen", "entry_frame_index", "exit_frame_index")
        if frame is not None:
            first_frame = session["first_frame"]
            last_frame = session["last_frame"]
            session["first_frame"] = frame if first_frame is None else min(int(first_frame), frame)
            session["last_frame"] = frame if last_frame is None else max(int(last_frame), frame)

    return [
        SessionInfo(
            session_id=text_value(item["session_id"]),
            source_id=text_value(item["source_id"]),
            source=text_value(item["source"]),
            first_timestamp=text_value(item["first_timestamp"]),
            first_frame=item["first_frame"] if isinstance(item["first_frame"], int) else None,
            last_frame=item["last_frame"] if isinstance(item["last_frame"], int) else None,
        )
        for item in sorted(sessions.values(), key=lambda row: (text_value(row["first_timestamp"]), text_value(row["session_id"])))
    ]


def episode_scope(reviewed: dict[str, object]) -> str:
    if reviewed["session_id"]:
        return f"session:{reviewed['session_id']}"
    if reviewed["source_id"]:
        return f"source_id:{reviewed['source_id']}"
    if reviewed["source"]:
        return f"source:{reviewed['source']}"
    return "all"


def frame_value(event: dict[str, object] | None, *keys: str) -> int | None:
    if event is None:
        return None
    for key in keys:
        value = parse_int(event.get(key))
        if value is not None:
            return value
    return None


def parse_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return None


def text_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def timestamp_seconds(event: dict[str, object]) -> float | None:
    timestamp = text_value(event.get("timestamp"))
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def mean(values: list[int]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def median(values: list[int]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def format_ratio(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def format_number(value: object) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
