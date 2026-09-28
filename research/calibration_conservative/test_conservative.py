"""Synthetic checks for artist isolation, conservative selection, and fallbacks.

These tests deliberately use no held-out project outcomes or fitted artifacts.
"""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from research.calibration.common import LABELS
from . import core
from .core import (make_folds, paired_cluster_se, select_strength,
                   fit_policy, predict_policy)


class ConservativeCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "folds": 5,
            "inner_seed": 2026092603,
            "lambda_grid": [0., .25, .5, .75, 1.],
            "tie_tolerance": 1e-12,
        }
        self.fit_config = {
            "slope_bounds": [.001, 100.],
            "intercept_bounds": [-20., 20.],
            "optimizer_options": {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        }

    def test_artist_folds_are_disjoint_exhaustive_and_deterministic(self):
        groups = np.repeat(np.arange(15), np.arange(1, 16) % 4 + 1)
        folds = make_folds(groups, 234, self.config)
        repeated = make_folds(groups.copy(), 234, dict(self.config))
        self.assertEqual(len(folds), 5)
        seen = []
        for fold, other in zip(folds, repeated):
            self.assertEqual(fold["fold"], other["fold"])
            train = np.asarray(fold["train_indices"], dtype=int)
            valid = np.asarray(fold["validation_indices"], dtype=int)
            np.testing.assert_array_equal(train, other["train_indices"])
            np.testing.assert_array_equal(valid, other["validation_indices"])
            self.assertGreater(len(train), 0)
            self.assertGreater(len(valid), 0)
            self.assertFalse(set(groups[train]) & set(groups[valid]))
            self.assertFalse(set(train) & set(valid))
            np.testing.assert_array_equal(np.sort(np.r_[train, valid]),
                                          np.arange(len(groups)))
            seen.extend(valid.tolist())
        np.testing.assert_array_equal(np.sort(seen), np.arange(len(groups)))

    def test_cluster_se_matches_unequal_size_hand_calculation(self):
        # Contrast mean is 1.5. Artist residual sums are -0.5, -2.0, 2.5.
        # The track-weighted cluster SE includes G/(G-1), with G=3, N=6.
        delta = np.array([1., 2., -1., 3., 0., 4.])
        groups = np.array(["a", "b", "b", "c", "c", "c"])
        expected = np.sqrt((3. / 2.) * (.5 ** 2 + 2. ** 2 + 2.5 ** 2)) / 6.
        self.assertAlmostEqual(paired_cluster_se(delta, groups), expected, places=14)
        self.assertAlmostEqual(paired_cluster_se(np.full(6, -.125), groups), 0.)
        # Duplicating every song changes neither the contrast nor the artist SE.
        self.assertAlmostEqual(paired_cluster_se(np.repeat(delta, 2),
                                                np.repeat(groups, 2)), expected)

    def test_no_signal_retains_identity_and_candidates_match_definition(self):
        y = np.tile([0., 1.], 5)
        raw = np.linspace(.1, .9, len(y))
        groups = np.repeat(np.arange(5), 2)
        result = select_strength(y, raw, raw.copy(), groups,
                                 self.config["lambda_grid"])
        self.assertEqual(result["selected_lambda"], 0.)
        self.assertEqual(result["best_lambda"], 0.)
        self.assertAlmostEqual(result["paired_se"], 0.)
        self.assertEqual(len(result["candidates"]), 5)
        for item in result["candidates"]:
            self.assertAlmostEqual(item["brier"], np.mean((raw - y) ** 2))
            self.assertAlmostEqual(item["delta_brier"], 0.)

    def test_strong_uniform_improvement_selects_full_calibration(self):
        y = np.tile([0., 1.], 10)
        raw = np.full(len(y), .5)
        calibrated = .1 + .8 * y
        groups = np.repeat(np.arange(10), 2)
        result = select_strength(y, raw, calibrated, groups,
                                 self.config["lambda_grid"])
        self.assertEqual(result["best_lambda"], 1.)
        self.assertEqual(result["selected_lambda"], 1.)
        self.assertAlmostEqual(result["paired_se"], 0.)
        self.assertAlmostEqual(result["threshold"], .01)
        for item in result["candidates"]:
            mixed = raw + item["lambda"] * (calibrated - raw)
            expected = np.mean((mixed - y) ** 2)
            self.assertAlmostEqual(item["brier"], expected)
            self.assertAlmostEqual(item["delta_brier"], expected - .25)

    def test_weak_variable_improvement_retains_identity(self):
        y = np.zeros(4)
        raw = np.full(4, .5)
        calibrated = np.array([0., .7, 0., .7])
        groups = np.arange(4)
        result = select_strength(y, raw, calibrated, groups,
                                 self.config["lambda_grid"])
        # Best grid candidate improves average Brier, but artist variability
        # is larger than that improvement, making identity eligible.
        self.assertEqual(result["best_lambda"], .5)
        self.assertEqual(result["selected_lambda"], 0.)
        best_brier = (.25 ** 2 + .6 ** 2) / 2.
        self.assertGreater(result["paired_se"], .25 - best_brier)
        self.assertGreaterEqual(result["threshold"], .25)

    def test_single_class_falls_back_to_logged_identity(self):
        groups = np.repeat(np.arange(15), 2)
        zcal = np.tile(np.linspace(-3., 3., len(groups))[:, None], (1, 4))
        ycal = np.zeros_like(zcal, dtype=int)
        folds = make_folds(groups, 1, self.config)
        policy, oof = fit_policy(zcal, ycal, groups, folds,
                                 self.fit_config, self.config)
        # Failed fits are represented as unavailable OOF values; the policy
        # records the failure and uses identity for subsequent predictions.
        self.assertTrue(np.isnan(oof).all())
        for label in LABELS:
            item = policy["labels"][label]
            self.assertEqual(item["selected_lambda"], 0.)
            self.assertTrue(item["fallback_reason"])
            self.assertIn("selection", item)
            self.assertIn("inner_fits", item)
            self.assertIn("full_fit", item)
        zeval = np.array([[-1000., 0., 1000., 2.], [1000., 1000., 1000., 1000.]])
        np.testing.assert_allclose(predict_policy(zeval, policy), expit(zeval),
                                   rtol=0, atol=0)

    def test_optimizer_failure_is_logged_and_cannot_apply_adjustment(self):
        groups = np.repeat(np.arange(20), 4)
        zcal = np.tile(np.array([-.2, -.1, .1, .2])[:, None], (20, 4))
        ycal = (zcal > 0).astype(int)
        folds = make_folds(groups, 19, self.config)
        real_calibrate = core.calibrate

        def fail_full_fit(z, y, method, config):
            if len(z) == len(zcal):
                return {"success": False, "message": "synthetic optimizer failure"}
            return real_calibrate(z, y, method, config)

        for failure_stage in ("inner", "full"):
            with self.subTest(failure_stage=failure_stage):
                options = ({"return_value": {"success": False,
                                               "message": "synthetic optimizer failure"}}
                           if failure_stage == "inner" else
                           {"side_effect": fail_full_fit})
                with patch.object(core, "calibrate", **options):
                    policy, _ = fit_policy(zcal, ycal, groups, folds,
                                           self.fit_config, self.config)
                for label in LABELS:
                    item = policy["labels"][label]
                    self.assertEqual(item["selected_lambda"], 0.)
                    self.assertIn("synthetic optimizer failure", item["fallback_reason"])
                    if failure_stage == "full":
                        self.assertGreater(item["selection"]["selected_lambda"], 0.)
                np.testing.assert_allclose(predict_policy(zcal, policy), expit(zcal),
                                           rtol=0, atol=0)

    def test_extreme_logits_are_bounded_and_labels_are_not_normalized(self):
        groups = np.repeat(np.arange(20), 2)
        zcal = np.tile(np.linspace(-2., 2., len(groups))[:, None], (1, 4))
        ycal = (zcal > 0).astype(int)
        folds = make_folds(groups, 17, self.config)
        policy, oof = fit_policy(zcal, ycal, groups, folds,
                                 self.fit_config, self.config)
        self.assertEqual(oof.shape, ycal.shape)
        self.assertTrue(np.isfinite(oof).all())
        self.assertTrue(((oof >= 0.) & (oof <= 1.)).all())
        zeval = np.tile(np.array([-1e6, 0., 1e6])[:, None], (1, 4))
        p = predict_policy(zeval, policy)
        self.assertEqual(p.shape, (3, 4))
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(((p >= 0.) & (p <= 1.)).all())
        self.assertTrue((np.diff(p, axis=0) >= 0.).all())
        self.assertGreater(p[-1].sum(), 3.99)
        self.assertLess(p[0].sum(), .01)

    def test_held_out_fold_answers_cannot_change_its_oof_predictions(self):
        groups = np.repeat(np.arange(20), 4)
        zcal = np.tile(np.array([-2., -.5, .5, 2.])[:, None], (20, 4))
        ycal = (zcal > 0).astype(int)
        folds = make_folds(groups, 19, self.config)
        _, original_oof = fit_policy(zcal, ycal, groups, folds,
                                     self.fit_config, self.config)
        valid = np.asarray(folds[0]["validation_indices"], dtype=int)
        altered = ycal.copy()
        altered[valid] = 1 - altered[valid]
        _, altered_oof = fit_policy(zcal, altered, groups, folds,
                                    self.fit_config, self.config)
        np.testing.assert_allclose(original_oof[valid], altered_oof[valid],
                                   rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
