from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from app.sources import (
    camera_url,
    has_url_credentials,
    infer_preview_rtsp_path,
    load_camera_entries,
    open_source_capture,
    redact_source,
    resolve_camera,
    resolve_source,
)


FIXTURE_MEDIA = Path(__file__).resolve().parent / "fixtures" / "media"


class SourceTests(unittest.TestCase):
    def test_camera_registry_builds_main_and_preview_urls_with_env_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "cameras.yaml"
            config_path.write_text(
                """
defaults:
  protocol: rtsp
  rtsp_port: 554
  username_env: TEST_CAMERA_USER
  password_env: TEST_CAMERA_PASSWORD
cameras:
  ip-camera:
    name: Field Camera
    address: 192.168.100.196
    rtsp_path: /cam/realmonitor?channel=1&subtype=0
    preview_rtsp_path: /cam/realmonitor?channel=1&subtype=1
""",
                encoding="utf-8",
            )
            previous_user = os.environ.get("TEST_CAMERA_USER")
            previous_password = os.environ.get("TEST_CAMERA_PASSWORD")
            os.environ["TEST_CAMERA_USER"] = "field user"
            os.environ["TEST_CAMERA_PASSWORD"] = "p@ss word"
            try:
                entry = load_camera_entries(config_path)["ip_camera"]
                self.assertEqual(
                    camera_url(entry),
                    "rtsp://field%20user:p%40ss%20word@192.168.100.196:554/cam/realmonitor?channel=1&subtype=0",
                )
                self.assertEqual(
                    camera_url(entry, profile="preview"),
                    "rtsp://field%20user:p%40ss%20word@192.168.100.196:554/cam/realmonitor?channel=1&subtype=1",
                )
                source = resolve_camera("ip-camera", config_path, profile="preview")
                self.assertEqual(source.kind, "rtsp")
                self.assertIn("[preview]", source.label)
            finally:
                restore_env("TEST_CAMERA_USER", previous_user)
                restore_env("TEST_CAMERA_PASSWORD", previous_password)

    def test_resolve_source_redacts_credentials_and_detects_images(self) -> None:
        source = resolve_source("rtsp://admin:secret@example.test:554/live")
        self.assertEqual(source.kind, "rtsp")
        self.assertEqual(source.label, "rtsp://<credentials>@example.test:554/live")
        self.assertTrue(has_url_credentials(source.capture_source))

        image = resolve_source(FIXTURE_MEDIA / "known_frame.png")
        self.assertEqual(image.kind, "image")
        self.assertTrue(image.is_image)

    def test_image_capture_reads_known_frame_once(self) -> None:
        source = resolve_source(FIXTURE_MEDIA / "known_frame.png")
        cap = open_source_capture(source)
        self.assertTrue(cap.isOpened())
        ok, frame = cap.read()
        self.assertTrue(ok)
        self.assertEqual(tuple(frame.shape[:2]), (24, 32))
        ok, frame = cap.read()
        self.assertFalse(ok)
        self.assertIsNone(frame)

    def test_known_clip_fixture_opens_and_has_frames(self) -> None:
        source = resolve_source(FIXTURE_MEDIA / "known_clip.mp4")
        cap = open_source_capture(source)
        self.assertTrue(cap.isOpened())
        frames = 0
        while frames < 4:
            ok, frame = cap.read()
            self.assertTrue(ok)
            self.assertEqual(tuple(frame.shape[:2]), (24, 32))
            frames += 1
        cap.release()

    def test_preview_path_inference_and_redaction(self) -> None:
        self.assertEqual(
            infer_preview_rtsp_path("/cam/realmonitor?channel=1&subtype=0"),
            "/cam/realmonitor?channel=1&subtype=1",
        )
        self.assertEqual(infer_preview_rtsp_path("/Streaming/Channels/101"), "/Streaming/Channels/102")
        self.assertEqual(redact_source("https://user:pass@example.test/path?q=1"), "https://<credentials>@example.test/path?q=1")


def restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
