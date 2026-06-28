from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.diagnostics import core as diagnostics_core
from app.diagnostics.core import DiagnosticsOptions
from app.diagnostics.redaction import redact_mapping, redact_text
from scripts.annotation_server import record_debug_event


class DiagnosticsTests(unittest.TestCase):
    def test_redaction_masks_secrets_auth_and_credentialed_urls(self) -> None:
        text = "\n".join(
            [
                "export UAV_CAMERA_PASSWORD=field-secret",
                "Authorization: Bearer abc.def",
                "stream=rtsp://admin:camera-pass@192.168.100.196:554/live?token=abc&quality=main",
            ]
        )

        redacted = redact_text(text, privacy="high")

        self.assertNotIn("field-secret", redacted)
        self.assertNotIn("camera-pass", redacted)
        self.assertNotIn("abc.def", redacted)
        self.assertNotIn("192.168.100.196", redacted)
        self.assertIn("UAV_CAMERA_PASSWORD=[redacted-present:True]", redacted)
        self.assertIn("Bearer [redacted]", redacted)
        self.assertIn("<credentials>@192.168.100.x:554", redacted)

    def test_redaction_masks_secret_mapping_values(self) -> None:
        redacted = redact_mapping(
            {
                "password": "admin123",
                "camera": {
                    "url": "rtsp://admin:pw@10.0.0.5/live?token=secret&profile=main",
                },
            }
        )

        rendered = json.dumps(redacted)
        self.assertEqual(redacted["password"], {"present": True})
        self.assertNotIn("admin123", rendered)
        self.assertNotIn("admin:pw", rendered)
        self.assertNotIn("token=secret", rendered)

    def test_diagnostics_run_writes_report_tarball_and_redacted_context(self) -> None:
        original_package_version = diagnostics_core.package_version
        diagnostics_core.package_version = lambda _package: "test-version"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = diagnostics_core.run_diagnostics(
                    DiagnosticsOptions(
                        mode="quick",
                        output_root=Path(tmp),
                        include_camera=False,
                        include_performance=True,
                        server_context={
                            "camera_url": "rtsp://admin:pw@192.168.100.196/live?token=secret",
                            "password": "admin123",
                        },
                    )
                )

                self.assertTrue(result.output_dir.exists())
                self.assertTrue(result.archive_path.exists())
                self.assertTrue(result.report_path.exists())
                self.assertIn(result.status, {"pass", "warn", "fail"})
                self.assertTrue((result.output_dir / "manifest.json").exists())
                self.assertTrue((result.output_dir / "checks.json").exists())

                context = json.loads((result.output_dir / "server_context.json").read_text(encoding="utf-8"))
                rendered = json.dumps(context)
                self.assertEqual(context["password"], {"present": True})
                self.assertNotIn("admin123", rendered)
                self.assertNotIn("admin:pw", rendered)
                self.assertNotIn("token=secret", rendered)
        finally:
            diagnostics_core.package_version = original_package_version

    def test_debug_activity_event_writes_redacted_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            event_path = Path(tmp) / "debug.jsonl"
            server = SimpleNamespace(
                debug_session_lock=threading.Lock(),
                debug_session_id="debug_test",
                debug_session_path=event_path,
            )

            result = record_debug_event(
                server,
                {
                    "session_id": "debug_test",
                    "event_type": "ui/control changed!",
                    "details": {
                        "control": "liveSourceInput",
                        "value": "rtsp://admin:pw@10.0.0.5/live?token=secret",
                        "password": "admin123",
                    },
                },
            )

            self.assertTrue(result["recorded"])
            row = json.loads(event_path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["event_type"], "ui_control_changed")
            rendered = json.dumps(row)
            self.assertNotIn("admin123", rendered)
            self.assertNotIn("admin:pw", rendered)
            self.assertNotIn("token=secret", rendered)


if __name__ == "__main__":
    unittest.main()
