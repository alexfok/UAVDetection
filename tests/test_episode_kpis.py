from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.evaluate_detection_episodes import (
    ReviewedEpisode,
    build_report_data,
    evaluate_entries,
    extract_predictions,
    find_event_files,
    load_events,
    render_markdown,
    write_template,
)


class EpisodeKpiTests(unittest.TestCase):
    def test_presence_events_match_reviewed_entries_and_report_latency(self) -> None:
        events = [
            event("start", session_id="s1", frame_index=0),
            event("drone_in_frame", session_id="s1", episode_id=1, frame_index=15, entry_frame_index=15),
            event("drone_out_frame", session_id="s1", episode_id=1, frame_index=40, last_seen_frame_index=38),
            event("drone_in_frame", session_id="s1", episode_id=2, frame_index=100, entry_frame_index=100),
        ]
        predictions = extract_predictions(events, out_gap_frames=30, out_gap_seconds=8.0)
        reviewed = [
            ReviewedEpisode("r1", "s1", "", "", 10, 50, ""),
            ReviewedEpisode("r2", "s1", "", "", 70, 80, ""),
        ]
        matches, false_entries = evaluate_entries(reviewed, predictions, deadline_frames=10, open_window_frames=300)

        self.assertEqual(matches[0].latency_frames, 5)
        self.assertTrue(matches[0].on_time)
        self.assertIsNone(matches[1].prediction)
        self.assertEqual([prediction.entry_frame for prediction in false_entries], [100])

        report_data = build_report_data([], 0, reviewed, predictions, matches, false_entries, Namespace(deadline_frames=10))
        self.assertEqual(report_data["metrics"]["entry_recall"], 0.5)
        self.assertIn("False Entry Events", render_markdown(report_data))

    def test_legacy_detection_events_group_by_timestamp_gap(self) -> None:
        events = [
            event("drone_detected", session_id="s1", frame_index=10, timestamp="2026-06-20T10:00:00+00:00"),
            event("drone_detected", session_id="s1", frame_index=200, timestamp="2026-06-20T10:00:05+00:00"),
            event("drone_detected", session_id="s1", frame_index=220, timestamp="2026-06-20T10:00:20+00:00"),
        ]
        predictions = extract_predictions(events, out_gap_frames=30, out_gap_seconds=8.0)
        self.assertEqual([prediction.entry_frame for prediction in predictions], [10, 220])
        self.assertEqual(predictions[0].exit_frame, 200)

    def test_event_file_discovery_loading_and_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2026-06-20"
            day.mkdir()
            event_path = day / "events.jsonl"
            event_path.write_text(
                "\n".join(
                    [
                        json.dumps(event("start", session_id="s1")),
                        "{bad json",
                        json.dumps(event("drone_detected", session_id="s1", frame_index=50)),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            files = find_event_files([root])
            events, bad_lines = load_events(files)
            self.assertEqual(files, [event_path])
            self.assertEqual(bad_lines, 1)
            self.assertEqual(len(events), 2)

            template_path = root / "review.csv"
            write_template(template_path, events, extract_predictions(events, 30, 8.0))
            text = template_path.read_text(encoding="utf-8")
            self.assertIn("entry_frame", text)
            self.assertIn("prediction entry 50", text)


def event(event_type: str, session_id: str = "s1", **kwargs: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_type": event_type,
        "timestamp": kwargs.pop("timestamp", "2026-06-20T10:00:00+00:00"),
        "session_id": session_id,
        "source_id": "ip_camera_196",
        "source": "IP Camera 196",
    }
    payload.update(kwargs)
    return payload


if __name__ == "__main__":
    unittest.main()
