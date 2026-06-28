from __future__ import annotations

import unittest

from app.alert import AlertManager
from app.config import AlertConfig, TrackerConfig
from app.tracker import SimpleTracker, iou
from app.types import Detection, TrackedDetection
from scripts.annotation_server import DronePresenceState


class TrackerAlertPresenceTests(unittest.TestCase):
    def test_tracker_matches_same_object_and_filters_tiny_boxes(self) -> None:
        tracker = SimpleTracker(TrackerConfig(iou_match_threshold=0.2, max_track_age_sec=1.0, min_box_area=10), 2.0)
        tracks = tracker.update([Detection((10, 10, 30, 30), 0.8, 0, "drone")], now=1.0)
        self.assertEqual(len(tracks), 1)
        first_id = tracks[0].track_id

        tracks = tracker.update([Detection((12, 12, 32, 32), 0.9, 0, "drone")], now=1.2)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, first_id)
        self.assertEqual(tracks[0].seen_frames, 2)

        self.assertEqual(tracker.update([Detection((1, 1, 2, 2), 0.99, 0, "drone")], now=1.3), [])
        self.assertEqual(tracker.update([], now=3.0), [])

    def test_iou_handles_overlap_and_empty_union(self) -> None:
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertEqual(iou((0, 0, 0, 0), (1, 1, 1, 1)), 0.0)

    def test_alert_requires_persistence_and_holds_cooldown(self) -> None:
        manager = AlertManager(AlertConfig(confidence_threshold=0.3, persistence_frames=2, cooldown_seconds=1.0))
        track = tracked(confidence=0.8, recent_hits=1)
        self.assertFalse(manager.update([track], now=10.0).active)

        active = manager.update([tracked(confidence=0.8, recent_hits=2)], now=10.1)
        self.assertTrue(active.active)
        self.assertIn("DRONE ALERT", active.message)

        cooldown = manager.update([], now=10.8)
        self.assertTrue(cooldown.active)
        self.assertEqual(len(cooldown.tracks), 1)

        cleared = manager.update([], now=11.3)
        self.assertFalse(cleared.active)
        self.assertEqual(cleared.message, "Monitoring")

    def test_presence_state_emits_entry_and_delayed_exit(self) -> None:
        state = DronePresenceState(out_seconds=2.0)
        entry = state.update([tracked(confidence=0.9, recent_hits=5)], frame_index=100, now=1.0)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.event_type, "drone_in_frame")
        self.assertEqual(entry.entry_frame_index, 100)

        self.assertIsNone(state.update([tracked(confidence=0.8, recent_hits=5)], frame_index=102, now=1.5))
        self.assertIsNone(state.update([], frame_index=103, now=2.0))

        exit_event = state.update([], frame_index=110, now=3.6)
        self.assertIsNotNone(exit_event)
        self.assertEqual(exit_event.event_type, "drone_out_frame")
        self.assertEqual(exit_event.last_seen_frame_index, 102)
        self.assertEqual(exit_event.reason, "absence_timeout")

    def test_presence_close_flushes_active_episode(self) -> None:
        state = DronePresenceState(out_seconds=2.0)
        state.update([tracked(confidence=0.9, recent_hits=5)], frame_index=1, now=10.0)
        closed = state.close(frame_index=5, now=10.5, reason="client_disconnected")
        self.assertIsNotNone(closed)
        self.assertEqual(closed.event_type, "drone_out_frame")
        self.assertEqual(closed.reason, "client_disconnected")
        self.assertIsNone(state.close(frame_index=6, now=11.0, reason="done"))


def tracked(confidence: float, recent_hits: int) -> TrackedDetection:
    return TrackedDetection(
        track_id=7,
        bbox=(10, 10, 30, 30),
        label="drone",
        class_id=0,
        confidence=confidence,
        first_seen=0.0,
        last_seen=0.0,
        seen_frames=recent_hits,
        recent_hits=recent_hits,
    )


if __name__ == "__main__":
    unittest.main()
