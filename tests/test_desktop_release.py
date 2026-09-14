"""Checks for desktop runtime isolation and Windows release wiring."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import desktop_runtime


class DesktopReleaseTests(unittest.TestCase):
    def test_explicit_data_directory_has_priority(self):
        with tempfile.TemporaryDirectory() as folder:
            result = desktop_runtime.default_user_data_dir(
                system="Windows",
                environ={desktop_runtime.DATA_DIRECTORY_ENV: folder},
                home="C:/Users/Reviewer",
            )
            self.assertEqual(result, Path(folder))

    def test_configure_runtime_redirects_model_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            old_hf_home = os.environ.get("HF_HOME")
            old_destination = desktop_runtime.prepare_maest_hf.DEST
            try:
                root, model = desktop_runtime.configure_runtime(folder)
                self.assertEqual(root, Path(folder))
                self.assertEqual(model, Path(folder) / "models/mtg-upf-maest-519l")
                self.assertEqual(desktop_runtime.prepare_maest_hf.DEST, model)
                self.assertEqual(os.environ["HF_HOME"], str(Path(folder) / ".cache/huggingface"))
            finally:
                desktop_runtime.prepare_maest_hf.DEST = old_destination
                if old_hf_home is None:
                    os.environ.pop("HF_HOME", None)
                else:
                    os.environ["HF_HOME"] = old_hf_home

    def test_model_presence_requires_pinned_receipt_and_sizes(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            (model / "sample.bin").write_bytes(b"abc")
            (model / "SOURCES.json").write_text(
                json.dumps(
                    {
                        "model_id": desktop_runtime.MODEL_ID,
                        "revision": desktop_runtime.REVISION,
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(desktop_runtime, "FILES", {"sample.bin": (3, "unused")}):
                self.assertTrue(desktop_runtime.model_files_present(model))
                (model / "sample.bin").write_bytes(b"ab")
                self.assertFalse(desktop_runtime.model_files_present(model))

    def test_installer_removes_only_its_per_user_data(self):
        installer = Path("packaging/windows/installer.iss").read_text(encoding="utf-8")
        self.assertIn('Name: "{localappdata}\\MusicClassification"', installer)
        self.assertNotIn("{userdocs}", installer.lower())
        workflow = Path(".github/workflows/windows-installer.yml").read_text(encoding="utf-8")
        self.assertIn("Run packaged end-to-end inference", workflow)
        self.assertIn("gh release create", workflow)


if __name__ == "__main__":
    unittest.main()
