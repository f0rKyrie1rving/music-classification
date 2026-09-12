"""Regression checks for preserving human work and auditing released results."""

import csv
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import diagnose_maest_hf as diagnosis
import reproduce_final as reproduction
from evaluate_release import (ROOT, assert_matches, bootstrap, bootstrap_indices,
                              evaluate, load_release, read_json, blind_summary)
from reproduce_final import stable_metadata, validate_features


class ListeningProtectionTests(unittest.TestCase):
    def test_completed_sheet_is_preserved_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.csv"
            original = (ROOT / "data/maest_error_listening_review.csv").read_bytes()
            path.write_bytes(original)
            with self.assertRaises(FileExistsError):
                diagnosis.write_listening_review([], path)
            self.assertEqual(path.read_bytes(), original)
            with patch.object(diagnosis, "REVIEW", path), patch.object(diagnosis, "load_plan") as load:
                with self.assertRaises(FileExistsError):
                    diagnosis.main()
                load.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def test_existing_blank_sheet_is_also_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.csv"
            original = b"label,auditor_hears_label,auditor_reason\nrock,,\n"
            path.write_bytes(original)
            with self.assertRaises(FileExistsError):
                diagnosis.write_listening_review([], path)
            self.assertEqual(path.read_bytes(), original)

    def test_new_sheet_has_blank_answers_and_unicode_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new.csv"
            diagnosis.write_listening_review([dict(label="rock", title="音乐", source_tags=["pop", "rock"],
                                                   neighbour_tags=[])], path)
            with path.open(encoding="utf-8", newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["title"], "音乐")
            self.assertEqual(row["source_tags"], "pop;rock")
            self.assertEqual(row["auditor_hears_label"], "")
            self.assertEqual(row["auditor_reason"], "")


class EvaluationAuditTests(unittest.TestCase):
    def test_released_scores_match_all_original_point_metrics_and_listening_counts(self):
        plan, _, rows, targets, scores = load_release()
        expected = read_json(ROOT / "data/evaluation/original_metrics.json")
        for policy, thresholds in plan["thresholds"].items():
            assert_matches(evaluate(targets, scores, thresholds), expected["holdout"][policy])
        actual = blind_summary(plan, rows)
        expected = read_json(ROOT / "data/evaluation/original_blind_summary.json")
        for key in ("overall", "by_label"):
            assert_matches(actual[key], expected[key])

    def test_changed_score_is_not_silently_accepted_as_archived_result(self):
        plan, _, _, targets, scores = load_release()
        scores[:] = 0
        expected = read_json(ROOT / "data/evaluation/original_metrics.json")
        with self.assertRaises(ValueError):
            assert_matches(evaluate(targets, scores, plan["thresholds"]["f1"]), expected["holdout"]["f1"])

    def test_cluster_draws_keep_every_artists_tracks_together(self):
        groups = np.array(["a", "a", "a", "b", "c", "c"])
        rng = np.random.default_rng(23)
        sizes = set()
        for _ in range(30):
            index = bootstrap_indices(rng, len(groups), groups)
            counts = np.bincount(index, minlength=len(groups))
            self.assertEqual(counts[0], counts[1])
            self.assertEqual(counts[1], counts[2])
            self.assertEqual(counts[4], counts[5])
            self.assertEqual(counts[0] + counts[3] + counts[4], 3)
            sizes.add(len(index))
        self.assertGreater(len(sizes), 1)

    def test_empty_label_replicates_are_retained_and_counted(self):
        targets, scores = np.zeros((3, 4), dtype=int), np.zeros((3, 4))
        result = bootstrap(targets, scores, [.5] * 4, 12, 11, np.array(["a", "a", "b"]))
        for label in result["empty_replicates"]:
            self.assertEqual(result["empty_replicates"][label], dict(no_positives=12, no_predictions=12))
            self.assertEqual(result["per_label"][label]["average_precision"], [0., 0.])


class ReproductionProvenanceTests(unittest.TestCase):
    def test_extraction_resumes_and_completed_cache_ignores_timing_but_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            rows = {"development": [{"track_id": str(i), "audio_file": str(i)} for i in range(12)],
                    "holdout": [{"track_id": "h", "audio_file": "h"}]}
            plan = {"model_receipt_sha256": "model"}
            audits = {"development": {}, "holdout": {}}
            original_sha = reproduction.sha256

            def digest(path):
                return "model" if Path(path).name == "SOURCES.json" else original_sha(path)

            def provenance(role, selected, audit, device):
                return dict(ids=[r["track_id"] for r in selected], role=role, device=device)

            class Encoder:
                fail = True

                def __init__(self, device):
                    pass

                def extract(self, path):
                    if self.fail and Path(path).name == "10":
                        raise RuntimeError("Simulated interruption after saved progress")
                    return np.full(2304, 0.25, dtype=np.float32)

            with patch.object(reproduction, "OUT", output), \
                    patch.object(reproduction, "inputs", return_value=(plan, rows, audits, [])), \
                    patch.object(reproduction, "sha256", side_effect=digest), \
                    patch.object(reproduction, "provenance", side_effect=provenance), \
                    patch.object(reproduction, "verify_audio"), \
                    patch("maest_hf_features.HfMaestEncoder", Encoder):
                with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
                    reproduction.extract("cpu")
                self.assertEqual(read_json(output / "development_progress.json")["completed"], 10)
                Encoder.fail = False
                reproduction.extract("cpu")
                self.assertEqual(read_json(output / "development_runtime.json")["resumed_tracks"], 10)
                self.assertNotIn("extraction_seconds", read_json(output / "development.json"))
                (output / "development_runtime.json").write_text('{"extraction_seconds": 99999}')
                reproduction.extract("cpu")
                np.save(output / "development.npy", np.ones((12, 2304), dtype=np.float32))
                with self.assertRaises(ValueError):
                    reproduction.extract("cpu")

    def test_elapsed_time_and_resume_counts_do_not_change_identity(self):
        expected = read_json(ROOT / "data/evaluation/development_feature_provenance.json")
        first = dict(expected, extraction_seconds=1.0, resumed_tracks=0, new_tracks=1206)
        second = dict(expected, extraction_seconds=9999.0, resumed_tracks=1200, new_tracks=6)
        self.assertEqual(stable_metadata(first), stable_metadata(second))
        second["audio_hashes"] = list(second["audio_hashes"])
        second["audio_hashes"][0] = "changed"
        self.assertNotEqual(stable_metadata(first), stable_metadata(second))

    def test_new_cache_rejects_stale_provenance_and_modified_features(self):
        expected = dict(ids=["a", "b"], audio_hashes=["hash-a", "hash-b"], device="cpu")
        meta = dict(provenance=deepcopy(expected), feature_sha256="correct")
        values = np.zeros((2, 2304), dtype=np.float32)
        validate_features(values, meta, expected, "correct")
        with self.assertRaises(ValueError):
            validate_features(values, meta, expected, "modified")
        meta["provenance"]["ids"].reverse()
        with self.assertRaises(ValueError):
            validate_features(values, meta, expected, "correct")


if __name__ == "__main__":
    unittest.main()
