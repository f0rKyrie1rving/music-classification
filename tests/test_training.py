"""Unit tests for split integrity, fitting, thresholds, and metrics."""

import unittest
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare_dataset import choose_split
from train import choose_thresholds, evaluate, validate_manifest
from improve_model import cross_validate, operating_metrics, precision_thresholds


class TrainingTests(unittest.TestCase):
    def test_thresholds_and_metrics(self):
        y = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
        scores = 0.1 + 0.8 * y
        thresholds = choose_thresholds(y, scores)
        report = evaluate(y, scores, thresholds, ["a", "b"])
        self.assertEqual(report["micro_f1"], 1)
        self.assertEqual(report["macro_ap"], 1)
        self.assertTrue(np.all((thresholds >= 0.1) & (thresholds <= 0.9)))

    def test_pipeline_scaler_fitted_only_on_training_rows(self):
        x = np.arange(24, dtype=float).reshape(12, 2)
        y = np.array([[i % 2, (i // 2) % 2] for i in range(12)])
        model = make_pipeline(StandardScaler(), OneVsRestClassifier(LogisticRegression()))
        model.fit(x, y)
        np.testing.assert_array_equal(model[0].mean_, x.mean(axis=0))
        scores = model.predict_proba(np.array([[1000, 1000]]))
        self.assertEqual(scores.shape, (1, 2))
        np.testing.assert_array_equal(model[0].mean_, x.mean(axis=0))

    def test_artist_leakage_is_rejected(self):
        manifest = dict(labels=["a"], tracks=[
            dict(track_id="one", artist_id="same", split="train"),
            dict(track_id="two", artist_id="same", split="test")])
        with self.assertRaisesRegex(ValueError, "Artist leakage"):
            validate_manifest(manifest)

    def test_duplicate_tracks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_manifest(dict(labels=["a"], tracks=[dict(track_id="one"), dict(track_id="one")]))

    def test_coverage_sampling_is_repeatable_and_respects_artist_cap(self):
        rows = [dict(track_id=str(i), artist_id=str(i // 2),
                     tags=[] if i % 5 == 0 else ["a" if i % 2 else "b"]) for i in range(40)]
        selected = choose_split(rows, ["a", "b"], 20, 5, 1)
        again = choose_split(rows, ["a", "b"], 20, 5, 1)
        self.assertEqual(selected, again)
        self.assertEqual(len({r["track_id"] for r in selected}), 20)
        self.assertEqual(max(Counter(r["artist_id"] for r in selected).values()), 1)
        self.assertEqual(sum(not r["tags"] for r in selected), 2)
        for label in ["a", "b"]:
            self.assertGreaterEqual(sum(label in r["tags"] for r in selected), 5)

    def test_precision_policy_rejects_tiny_perfect_subset(self):
        y = np.tile(np.array([[1], [1], [1], [1], [0], [0]]), (1, 4))
        scores = np.tile(np.array([[.99], [.6], [.6], [.6], [.65], [.65]]), (1, 4))
        plan = dict(precision_grid=[.5, .9], target_precision=.8,
                    min_recall=.3, min_oof_predictions=2)
        thresholds, supported = precision_thresholds(y, scores, plan)
        self.assertEqual(supported, [False] * 4)
        self.assertFalse((scores >= thresholds).any())
        report = operating_metrics(y, scores, thresholds, ["a", "b", "c", "d"])
        self.assertFalse(report["provisional_goal_met"])
        self.assertFalse(report["per_label"][0]["precision_defined"])

    def test_precision_policy_prefers_recall_among_feasible_choices(self):
        y = np.array([[1], [1], [1], [1], [0], [0]])
        scores = np.array([[.95], [.95], [.7], [.7], [.2], [.2]])
        plan = dict(precision_grid=[.5, .9], target_precision=.8,
                    min_recall=.3, min_oof_predictions=2)
        thresholds, supported = precision_thresholds(y, scores, plan)
        np.testing.assert_equal(thresholds, [.5])
        self.assertEqual(supported, [True])

    def test_grouped_oof_rejects_artist_leakage(self):
        with self.assertRaisesRegex(ValueError, "Artist leakage"):
            cross_validate(np.ones((4, 2)), np.array([[0], [1], [0], [1]]),
                           np.array(["a", "a", "b", "b"]), np.array([0, 1, 0, 1]),
                           dict(C=1, class_weight=None))


if __name__ == "__main__":
    unittest.main()
