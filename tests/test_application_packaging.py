"""Release artifact, relocation, and executable-output contract checks."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging/windows/MusicClassification.spec"
verifier_spec = importlib.util.spec_from_file_location(
    "windows_smoke_verifier", ROOT / "packaging/windows/verify_smoke.py")
smoke = importlib.util.module_from_spec(verifier_spec)
verifier_spec.loader.exec_module(smoke)


def spec_analysis():
    captured = {}

    def analysis(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(pure=[], scripts=[], binaries=[], datas=[])

    namespace = {
        "SPECPATH": str(SPEC.parent), "Analysis": analysis,
        "PYZ": lambda *a, **k: None, "EXE": lambda *a, **k: None,
        "COLLECT": lambda *a, **k: None,
    }
    exec(compile(SPEC.read_text(encoding="utf-8"), str(SPEC), "exec"), namespace)
    return captured


def successful_smoke_payload():
    def rows(thresholds, raw_thresholds, baseline=False):
        scores = (0.0699, 0.2483, 0.0078, 0.6237) if baseline else (0.0699, 0.3219, 0.0272, 0.6237)
        raw = (0.0699, 0.2483, 0.0301, 0.6237) if baseline else scores
        return [
            {"label": label, "score": score, "raw_score": raw_score,
             "threshold": cutoff, "raw_threshold": raw_cutoff,
             "selected": label in ({"rock"} if baseline else {"pop", "rock"})}
            for label, score, raw_score, cutoff, raw_cutoff in zip(
                smoke.LABELS, scores, raw, thresholds, raw_thresholds)
        ]

    return {"ok": True, "result": {
        "application_version": smoke.VERSION, "candidate_id": smoke.CANDIDATE_ID,
        "model_weights_sha256": smoke.WEIGHTS_SHA256, "status": "validated_candidate",
        "score_mode": "validated_candidate", "policy": "f1",
        "validation": {"verdict": "go", "tracks": 372, "artists": 250},
        "scores": rows(smoke.CANDIDATE_THRESHOLDS, smoke.CANDIDATE_THRESHOLDS),
        "predicted_tags": ["pop", "rock"],
        "baseline": {
            "application_version": "1.1.0", "score_mode": "weight_corrected",
            "policy": "f1", "shared_feature": True,
            "scores": rows(smoke.BASELINE_THRESHOLDS, smoke.BASELINE_RAW_THRESHOLDS, True),
            "predicted_tags": ["rock"],
        },
    }}


class ApplicationPackagingTests(unittest.TestCase):
    def test_spec_contains_all_candidate_receipts_and_legacy_comparison(self):
        captured = spec_analysis()
        destinations = {str(Path(dest) / Path(source).name) for source, dest in captured["datas"]}
        expected = {
            "artifacts/application_release.json", "artifacts/final_heads.npy",
            "artifacts/final_heads.json", "artifacts/score_correction.json",
            "app_version.txt", "experiments/final_holdout_plan.json",
        }
        expected.update(f"artifacts/candidates/20260927_v1/{name}" for name in (
            "candidate.json", "candidate.npz", "freeze.json", "future_exclusions.json",
            "threshold_selection.json", "training.json"))
        # Convert to POSIX for the same assertion on the Windows release runner.
        actual = {Path(name).as_posix() for name in destinations}
        self.assertTrue(expected <= actual, expected - actual)
        self.assertTrue({"application_predict", "application_release", "candidate_predict",
                         "predict_app", "score_correction"} <= set(captured["hiddenimports"]))
        for source, _ in captured["datas"]:
            self.assertTrue(Path(source).is_file(), source)

    def test_packaged_assets_and_modules_work_away_from_checkout(self):
        # Recreate the data layout specified to PyInstaller, then import only
        # copied application modules from an unrelated working directory.
        with tempfile.TemporaryDirectory(prefix="music release relocation ") as temporary:
            temp = Path(temporary)
            bundle = temp / "app _internal"
            elsewhere = temp / "unrelated working directory"
            elsewhere.mkdir()
            for source, destination in spec_analysis()["datas"]:
                target = bundle / destination / Path(source).name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            for name in (
                "application_release", "application_predict", "candidate_predict", "predict_app",
                "score_correction", "predict_maest", "maest_hf_features", "prepare_maest_hf",
                "prepare_mert",
            ):
                shutil.copyfile(ROOT / f"{name}.py", bundle / f"{name}.py")
            script = """
import json, pathlib, sys
sys._MEIPASS = sys.argv[1]
sys.path.insert(0, sys._MEIPASS)
import numpy as np
import application_predict, application_release, predict_app
class Encoder:
    calls = 0
    def extract(self, audio):
        self.calls += 1
        return np.zeros(2304, dtype=np.float32)
encoder = Encoder()
result = application_predict.classify('unused.wav', encoder=encoder, compare_baseline=True)
assert encoder.calls == 1
assert pathlib.Path(application_predict.__file__).parent == pathlib.Path(sys._MEIPASS)
assert application_release.ROOT == pathlib.Path(sys._MEIPASS)
assert predict_app.APP_VERSION == '1.1.0'
print(json.dumps(result))
"""
            process = subprocess.run(
                [sys.executable, "-I", "-c", script, str(bundle)], cwd=elsewhere,
                check=True, capture_output=True, text=True, timeout=60,
            )
            result = json.loads(process.stdout)
            self.assertEqual(result["application_version"], "1.2.0")
            self.assertEqual(result["model_weights_sha256"], smoke.WEIGHTS_SHA256)
            self.assertEqual(result["candidate_id"], smoke.CANDIDATE_ID)
            self.assertEqual(result["baseline"]["application_version"], "1.1.0")
            self.assertTrue(result["baseline"]["shared_feature"])
            self.assertEqual([row["threshold"] for row in result["scores"]],
                             list(smoke.CANDIDATE_THRESHOLDS))
            for index in (0, 3):
                self.assertEqual(result["scores"][index], result["baseline"]["scores"][index])

    def test_smoke_accepts_candidate_and_original_comparison(self):
        payload = successful_smoke_payload()
        original = deepcopy(payload)
        self.assertIs(smoke.verify(payload), payload["result"])
        self.assertEqual(payload, original)

    def test_smoke_rejects_stale_wrong_or_incomplete_executable_output(self):
        mutations = [
            (("ok",), False),
            (("result", "application_version"), "1.1.0"),
            (("result", "candidate_id"), "another-candidate"),
            (("result", "model_weights_sha256"), "0" * 64),
            (("result", "status"), "candidate_not_fresh_validated"),
            (("result", "score_mode"), "weight_corrected"),
            (("result", "policy"), "precision_target"),
            (("result", "validation", "verdict"), "not_passed"),
            (("result", "validation", "tracks"), 371),
            (("result", "validation", "artists"), 249),
            (("result", "scores", 2, "threshold"), 0.1321),
            (("result", "scores", 0, "score"), float("nan")),
            (("result", "scores", 3, "score"), 0.7),
            (("result", "predicted_tags"), ["pop", "ambient"]),
            (("result", "baseline"), None),
            (("result", "baseline", "application_version"), "1.2.0"),
            (("result", "baseline", "shared_feature"), False),
            (("result", "baseline", "scores", 2, "raw_threshold"), 0.2),
        ]
        for path, value in mutations:
            with self.subTest(path=path):
                payload = successful_smoke_payload()
                owner = payload
                for key in path[:-1]:
                    owner = owner[key]
                owner[path[-1]] = value
                with self.assertRaises(ValueError):
                    smoke.verify(payload)
        with self.assertRaises(ValueError):
            smoke.verify(successful_smoke_payload(), "1.1.0")

    def test_ci_checks_both_executables_without_rewriting_frozen_version(self):
        workflow = (ROOT / ".github/workflows/windows-installer.yml").read_text(encoding="utf-8")
        self.assertIn("from application_release import APP_VERSION, load_release", workflow)
        self.assertIn('"${{ github.ref_name }}" -ne "v$version"', workflow)
        self.assertNotIn('Set-Content "app_version.txt"', workflow)
        self.assertEqual(workflow.count("python packaging/windows/verify_smoke.py $result"), 2)
        self.assertIn("-r requirements-test.txt", workflow)
        self.assertIn("-m unittest discover -s tests -v", workflow)
        self.assertEqual((ROOT / "app_version.txt").read_text().strip(), "1.1.0")
        self.assertIn('#define AppVersion "1.2.0"',
                      (ROOT / "packaging/windows/installer.iss").read_text())


if __name__ == "__main__":
    unittest.main()
