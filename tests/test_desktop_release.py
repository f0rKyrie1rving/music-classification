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
            "application_version": "1.2.0",
            "score_mode": "validated_candidate",
            "predicted_tags": ["ambient"],
            "scores": [
                {"label": "ambient", "score": 0.20001, "threshold": 0.2,
                 "selected": True},
                {"label": "pop", "score": 0.27499, "threshold": 0.275,
                 "selected": False},
            ],
            "baseline": {
                "application_version": "1.1.0",
                "score_mode": "weight_corrected",
                "shared_feature": True,
                "predicted_tags": ["pop"],
                "scores": [
                    {"label": "ambient", "score": 0.12, "threshold": 0.13,
                     "raw_score": 0.3, "raw_threshold": 0.375, "selected": False},
                    {"label": "pop", "score": 0.27501, "threshold": 0.275,
                     "raw_score": 0.27501, "raw_threshold": 0.275, "selected": True},
                ],
            },
        }

    def test_score_views_select_each_models_scores_thresholds_and_decisions(self):
        result = self.prediction_result()
        original_payload = deepcopy(result)
        self.assertEqual(
            desktop_app.score_display_rows(result),
            [("ambient", ("0.2000", "0.2000", "Yes")), ("pop", ("0.2750", "0.2750", "No"))],
        )
        self.assertEqual(
            desktop_app.score_display_rows(result, compare_v1_1=True),
            [("ambient", ("0.1200", "0.1300", "No")), ("pop", ("0.2750", "0.2750", "Yes"))],
        )
        self.assertEqual(result, original_payload)
        # Equal rounded values must still show each model's original decision.
        self.assertEqual(desktop_app.score_display_rows(result)[1][1][-1], "No")
        self.assertEqual(desktop_app.score_display_rows(result, True)[1][1][-1], "Yes")
        result["scores"][0]["selected"] = False
        self.assertEqual(desktop_app.score_display_rows(result)[0][1][-1], "No")
        self.assertEqual(desktop_app.score_display_rows(result, True)[0][1][-1], "No")

    def test_view_toggle_uses_cached_result_without_prediction(self):
        app = desktop_app.MusicTaggingApp.__new__(desktop_app.MusicTaggingApp)
        toggle = SimpleNamespace(value=False)
        app.compare_v1_1 = SimpleNamespace(get=lambda: toggle.value)
        app.results, app.score_note, app.selected_tags = Mock(), Mock(), Mock()
        app.last_result = self.prediction_result()
        with patch.object(desktop_app, "run_prediction") as prediction:
            app.refresh_result_view()
            app.results.heading.assert_called_with("score", text="Estimate (v1.2.0)")
            app.selected_tags.set.assert_called_with("v1.2.0 · Selected tags: ambient")
            toggle.value = True
            app.refresh_result_view()
            app.results.heading.assert_called_with("score", text="Estimate (v1.1.0)")
            app.results.item.assert_any_call("ambient", values=("0.1200", "0.1300", "No"))
            app.results.item.assert_any_call("pop", values=("0.2750", "0.2750", "Yes"))
            app.selected_tags.set.assert_called_with("v1.1.0 · Selected tags: pop")
            toggle.value = False
            app.refresh_result_view()
            app.results.item.assert_any_call("ambient", values=("0.2000", "0.2000", "Yes"))
            app.selected_tags.set.assert_called_with("v1.2.0 · Selected tags: ambient")
            prediction.assert_not_called()

    def test_view_toggle_before_analysis_has_no_result_rows(self):
        app = desktop_app.MusicTaggingApp.__new__(desktop_app.MusicTaggingApp)
        app.compare_v1_1 = SimpleNamespace(get=lambda: True)
        app.results, app.score_note, app.selected_tags = Mock(), Mock(), Mock()
        app.last_result = None
        app.refresh_result_view()
        app.results.item.assert_not_called()
        app.selected_tags.set.assert_not_called()

    def test_comparison_view_empty_tags_are_not_inherited_from_current_model(self):
        app = desktop_app.MusicTaggingApp.__new__(desktop_app.MusicTaggingApp)
        app.compare_v1_1 = SimpleNamespace(get=lambda: True)
        app.results, app.score_note, app.selected_tags = Mock(), Mock(), Mock()
        app.last_result = self.prediction_result()
        app.last_result["baseline"]["predicted_tags"] = []
        app.refresh_result_view()
        app.selected_tags.set.assert_called_with("v1.1.0 · Selected tags: none")

    def test_desktop_prediction_uses_new_application_entrypoint(self):
        encoder = Mock()
        encoder_type = Mock(return_value=encoder)
        classify = Mock(return_value=self.prediction_result())
        validate = Mock()
        callback = Mock()
        order = Mock()
        with patch.object(desktop_app, "prepare_desktop_model") as prepare:
            order.attach_mock(validate, "validate")
            order.attach_mock(prepare, "prepare")
            order.attach_mock(encoder_type, "encoder")
            order.attach_mock(classify, "classify")
            with patch.dict("sys.modules", {
                "maest_hf_features": SimpleNamespace(HfMaestEncoder=encoder_type),
                "application_predict": SimpleNamespace(classify=classify, load_application=validate),
            }):
                result = desktop_app.run_prediction(Path("example.wav"), callback)
        self.assertEqual([call[0] for call in order.mock_calls], ["validate", "prepare", "encoder", "classify"])
        validate.assert_called_once_with(root=desktop_app.BUNDLE_ROOT)
        prepare.assert_called_once_with(data_root=desktop_app.APP_DATA_ROOT, callback=callback)
        encoder_type.assert_called_once_with(device="cpu")
        classify.assert_called_once_with(
            Path("example.wav"), device="cpu", encoder=encoder,
            compare_baseline=True, root=desktop_app.BUNDLE_ROOT,
        )
        self.assertEqual(result["score_mode"], "validated_candidate")

    def test_invalid_application_release_fails_before_model_download(self):
        validate = Mock(side_effect=ValueError("Candidate weights checksum mismatch"))
        classify, encoder_type = Mock(), Mock()
        with patch.object(desktop_app, "prepare_desktop_model") as prepare:
            with patch.dict("sys.modules", {
                "maest_hf_features": SimpleNamespace(HfMaestEncoder=encoder_type),
                "application_predict": SimpleNamespace(classify=classify, load_application=validate),
            }):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    desktop_app.run_prediction(Path("example.wav"))
        prepare.assert_not_called()
        encoder_type.assert_not_called()
        classify.assert_not_called()

    def test_desktop_version_is_current_without_changing_historical_version(self):
        self.assertEqual(desktop_app.APP_VERSION, "1.2.0")
        self.assertEqual(Path("app_version.txt").read_text(encoding="utf-8").strip(), "1.1.0")

    def test_smoke_test_preserves_current_and_baseline_results(self):
        result = self.prediction_result()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result.json"
            with patch.object(desktop_app, "run_prediction", return_value=result) as prediction:
                self.assertEqual(desktop_app.smoke_test("example.wav", output), 0)
            prediction.assert_called_once_with(Path("example.wav"))
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload, {"ok": True, "result": result})

    def test_smoke_test_reports_release_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result.json"
            with patch.object(desktop_app, "run_prediction", side_effect=ValueError("Release mismatch")):
                self.assertEqual(desktop_app.smoke_test("example.wav", output), 1)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"ok": False, "error": "Release mismatch"},
            )

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
