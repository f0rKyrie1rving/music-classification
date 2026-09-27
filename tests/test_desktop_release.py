"""Checks for desktop runtime isolation and Windows release wiring."""

import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import desktop_app
import desktop_runtime


class DesktopReleaseTests(unittest.TestCase):
    def prediction_result(self):
        return {
            "score_mode": "weight_corrected",
            "predicted_tags": ["ambient"],
            "scores": [
                {"label": "ambient", "score": 0.25001, "threshold": 0.25,
                 "raw_score": 0.70001, "raw_threshold": 0.7, "selected": True},
                {"label": "pop", "score": 0.3, "threshold": 0.5,
                 "raw_score": 0.3, "raw_threshold": 0.5, "selected": False},
            ],
        }

    def test_score_views_preserve_payload_and_decisions(self):
        result = self.prediction_result()
        original_payload = deepcopy(result)
        self.assertEqual(
            desktop_app.score_display_rows(result),
            [("ambient", ("0.2500", "0.2500", "Yes")), ("pop", ("0.3000", "0.5000", "No"))],
        )
        self.assertEqual(
            desktop_app.score_display_rows(result, show_original=True),
            [("ambient", ("0.7000", "0.7000", "Yes")), ("pop", ("0.3000", "0.5000", "No"))],
        )
        self.assertEqual(result, original_payload)
        # Rounding or a display value must never be used to infer a fresh decision.
        result["scores"][0]["selected"] = False
        self.assertEqual(desktop_app.score_display_rows(result)[0][1][-1], "No")
        self.assertEqual(desktop_app.score_display_rows(result, True)[0][1][-1], "No")

    def test_view_toggle_uses_cached_result_without_prediction(self):
        app = desktop_app.MusicTaggingApp.__new__(desktop_app.MusicTaggingApp)
        toggle = SimpleNamespace(value=False)
        app.show_original = SimpleNamespace(get=lambda: toggle.value)
        app.results, app.score_note = Mock(), Mock()
        app.last_result = self.prediction_result()
        with patch.object(desktop_app, "run_prediction") as prediction:
            app.refresh_result_view()
            app.results.heading.assert_called_with("score", text="Estimate")
            toggle.value = True
            app.refresh_result_view()
            app.results.heading.assert_called_with("score", text="Original score")
            app.results.item.assert_any_call("ambient", values=("0.7000", "0.7000", "Yes"))
            prediction.assert_not_called()

    def test_view_toggle_before_analysis_has_no_result_rows(self):
        app = desktop_app.MusicTaggingApp.__new__(desktop_app.MusicTaggingApp)
        app.show_original = SimpleNamespace(get=lambda: True)
        app.results, app.score_note = Mock(), Mock()
        app.last_result = None
        app.refresh_result_view()
        app.results.item.assert_not_called()

    def test_desktop_prediction_uses_new_application_entrypoint(self):
        encoder = Mock()
        encoder_type = Mock(return_value=encoder)
        classify = Mock(return_value=self.prediction_result())
        callback = Mock()
        with patch.object(desktop_app, "prepare_desktop_model") as prepare:
            with patch.dict("sys.modules", {
                "maest_hf_features": SimpleNamespace(HfMaestEncoder=encoder_type),
                "predict_app": SimpleNamespace(classify=classify),
            }):
                result = desktop_app.run_prediction(Path("example.wav"), callback)
        prepare.assert_called_once_with(data_root=desktop_app.APP_DATA_ROOT, callback=callback)
        encoder_type.assert_called_once_with(device="cpu")
        classify.assert_called_once_with(Path("example.wav"), device="cpu", encoder=encoder)
        self.assertEqual(result["score_mode"], "weight_corrected")

    def test_release_includes_score_correction_and_version(self):
        spec = Path("packaging/windows/MusicClassification.spec").read_text(encoding="utf-8")
        self.assertIn('(str(project_root / "artifacts/score_correction.json"), "artifacts")', spec)
        self.assertIn('"predict_app",', spec)
        self.assertIn('"score_correction",', spec)
        self.assertEqual(Path("app_version.txt").read_text(encoding="utf-8").strip(), "1.1.0")
        installer = Path("packaging/windows/installer.iss").read_text(encoding="utf-8")
        self.assertIn('#define AppVersion "1.1.0"', installer)

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
