from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.deploy_jetson import (
    DeployContext,
    collect_preflight_checks,
    copy_source_tree,
    render_service_file,
    uninstall_files,
)


class JetsonDeploymentTests(unittest.TestCase):
    def test_preflight_fails_when_required_source_files_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = DeployContext(
                action="install",
                source_dir=root / "missing_source",
                install_dir=root / "install",
                service_name="uav-detection.service",
                service_mode="none",
                venv=".venv_cuda",
                port=8765,
                skip_deps=True,
                no_service=True,
            )

            checks = collect_preflight_checks(context)
            failed = {check.name for check in checks if check.status == "fail"}

            self.assertIn("source_dir", failed)
            self.assertIn("source:scripts/annotation_server.py", failed)

    def test_copy_source_tree_preserves_target_data_store_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            (source / "app").mkdir(parents=True)
            (source / "data_store" / "raw_data").mkdir(parents=True)
            (source / "app" / "main.py").write_text("print('new')\n", encoding="utf-8")
            (source / "data_store" / "raw_data" / "source_only.txt").write_text("source", encoding="utf-8")
            (target / "data_store" / "raw_data").mkdir(parents=True)
            (target / "data_store" / "raw_data" / "field_clip.txt").write_text("keep", encoding="utf-8")

            with redirect_stdout(StringIO()):
                copy_source_tree(source, target, dry_run=False, include_data_store=False)

            self.assertEqual((target / "app" / "main.py").read_text(encoding="utf-8"), "print('new')\n")
            self.assertTrue((target / "data_store" / "raw_data" / "field_clip.txt").exists())
            self.assertFalse((target / "data_store" / "raw_data" / "source_only.txt").exists())

    def test_uninstall_preserves_data_store_backup_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            install = Path(tmp) / "UAVDetection"
            (install / "data_store" / "raw_data").mkdir(parents=True)
            (install / "data_store" / "raw_data" / "field_clip.txt").write_text("keep", encoding="utf-8")
            (install / "app").mkdir()
            (install / "app" / "__init__.py").write_text("", encoding="utf-8")

            with redirect_stdout(StringIO()):
                uninstall_files(install, delete_data=False, dry_run=False)

            self.assertFalse(install.exists())
            backups = list(Path(tmp).glob("UAVDetection_data_store_backup_*"))
            self.assertEqual(len(backups), 1)
            self.assertTrue((backups[0] / "raw_data" / "field_clip.txt").exists())

    def test_service_file_uses_preferred_venv_and_preserved_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            install = Path(tmp) / "UAVDetection"
            (install / ".venv_cuda" / "bin").mkdir(parents=True)
            (install / ".venv_cuda" / "bin" / "python").write_text("", encoding="utf-8")
            context = DeployContext(
                action="upgrade",
                source_dir=Path(tmp),
                install_dir=install,
                service_name="uav-detection.service",
                service_mode="system",
                venv=".venv_cuda",
                port=8765,
                skip_deps=True,
                no_service=False,
            )
            args = type(
                "Args",
                (),
                {
                    "host": "0.0.0.0",
                    "default_folder": "data_store/raw_data/Roni",
                    "project_dir": "data_store/datasets/web_drone_v1",
                    "camera_config": "data_store/system_config/cameras.yaml",
                    "live_model": "data_store/models/trained/yolov8n_drone_best.pt",
                    "no_https": False,
                },
            )()

            content = render_service_file(args, context)

            self.assertIn(str(install / ".venv_cuda" / "bin" / "python"), content)
            self.assertIn("EnvironmentFile=", content)
            self.assertIn("annotation_server.env", content)
            self.assertIn("--password-env ANNOTATION_SERVER_PASSWORD", content)


if __name__ == "__main__":
    unittest.main()
