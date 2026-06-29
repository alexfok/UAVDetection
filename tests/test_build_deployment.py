from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_deployment import copy_project, write_manifest, write_zip


class BuildDeploymentTests(unittest.TestCase):
    def test_code_artifact_excludes_runtime_data_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / "UAVDetection_deploy_test"
            artifact = Path(tmp) / "UAVDetection_deploy_test.zip"

            copy_project(staging, include_data_store=False)
            write_manifest(staging, "UAVDetection_deploy_test", include_data_store=False)
            write_zip(staging, artifact)

            manifest = json.loads((staging / "DEPLOYMENT_MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact_kind"], "code-only")
            self.assertTrue((staging / "scripts" / "deploy.py").exists())
            self.assertFalse((staging / "data_store").exists())

            with zipfile.ZipFile(artifact) as archive:
                names = set(archive.namelist())
            self.assertIn("UAVDetection_deploy_test/scripts/deploy.py", names)
            self.assertIn("UAVDetection_deploy_test/DEPLOYMENT_MANIFEST.json", names)
            self.assertFalse(any(name.startswith("UAVDetection_deploy_test/data_store/") for name in names))


if __name__ == "__main__":
    unittest.main()
