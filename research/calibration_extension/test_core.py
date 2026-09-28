"""Synthetic checks for the controlled weighting and calibration comparisons."""
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import core


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path(__file__).with_name('config.json').read_text())
        rng = np.random.default_rng(329)
        self.x = rng.normal(size=(120, 6))
        self.x[:, -1] = 3.  # Constant feature must remain numerically usable.
        self.y = np.column_stack([rng.binomial(1, p, 120) for p in [.4, .3, .2, .25]])
        self.z = rng.normal(size=(120, 4)) + np.array([0., 1., 1.5, -.5])
        self.groups = np.repeat(np.arange(30), 4)

    def test_intercept_constant_logits_has_known_solution(self):
        z = np.full(100, 2.)
        y = np.r_[np.ones(20), np.zeros(80)]
        fit = core.fit_intercept(z, y, self.config['calibrator'])
        self.assertTrue(fit['success'])
        self.assertEqual(fit['slope'], 1.)
        self.assertAlmostEqual(fit['intercept'], np.log(.2/.8)-2., places=6)
        self.assertAlmostEqual(expit(2+fit['intercept']), .2, places=7)

    def test_intercept_rejects_single_class_and_nonfinite_input(self):
        for z, y in [(np.zeros(3), np.zeros(3)),
                     (np.array([0., np.nan, 1.]), np.array([0, 1, 1]))]:
            with self.assertRaises(ValueError):
                core.fit_intercept(z, y, self.config['calibrator'])

    def test_saved_heads_match_independently_fitted_sklearn_pipeline(self):
        heads = core.fit_heads(self.x, self.y, 'ambient_balanced', self.config['classifier'])
        scaler = StandardScaler().fit(self.x)
        expected = []
        for j in range(4):
            head = LogisticRegression(**self.config['classifier'],
                                      class_weight='balanced' if j == 2 else None)
            head.fit(scaler.transform(self.x), self.y[:, j])
            expected.append(head.decision_function(scaler.transform(self.x[:7])))
        np.testing.assert_allclose(core.logits(self.x[:7], heads), np.array(expected).T,
                                   atol=1e-12, rtol=1e-12)
        self.assertEqual(heads['scale'][-1], 1.)
        np.testing.assert_array_equal(heads['positive_fit'], self.y.sum(0))

    def test_only_ambient_head_changes_under_weight_intervention(self):
        balanced = core.fit_heads(self.x, self.y, 'ambient_balanced', self.config['classifier'])
        plain = core.fit_heads(self.x, self.y, 'unweighted', self.config['classifier'])
        for key in ['mean', 'scale']:
            np.testing.assert_array_equal(balanced[key], plain[key])
        for key in ['coef', 'intercept']:
            np.testing.assert_array_equal(balanced[key][[0, 1, 3]], plain[key][[0, 1, 3]])
        self.assertFalse(np.allclose(balanced['intercept'][2], plain['intercept'][2]))

    def test_prior_offset_uses_training_prevalence_with_correct_sign(self):
        balanced, _ = core.fit_treatments(self.z, self.y, self.groups, self.y,
                                        'ambient_balanced', 79, self.config)
        plain, _ = core.fit_treatments(self.z, self.y, self.groups, self.y,
                                     'unweighted', 79, self.config)
        npos = self.y[:, 2].sum()
        np.testing.assert_allclose(balanced['offsets'], [0, 0, np.log(npos/(len(self.y)-npos)), 0])
        np.testing.assert_array_equal(plain['offsets'], np.zeros(4))
        probabilities = core.predict_treatments(np.zeros((2, 4)), balanced)
        self.assertAlmostEqual(probabilities['prior_offset'][0, 2], npos/len(self.y))
        np.testing.assert_array_equal(probabilities['prior_offset'][:, [0, 1, 3]], .5)

    def test_blend_ablation_shares_all_fits_and_uses_minimum_candidate(self):
        fitted, _ = core.fit_treatments(self.z, self.y, self.groups, self.y,
                                       'ambient_balanced', 79, self.config)
        for label in core.LABELS:
            conservative = fitted['policy']['labels'][label]
            minimum = fitted['minimum_policy']['labels'][label]
            for key in ['inner_fits', 'full_fit', 'selection', 'fallback_reason']:
                self.assertEqual(conservative[key], minimum[key])
            if conservative['fallback_reason']:
                self.assertEqual(minimum['selected_lambda'], 0.)
            else:
                candidates = conservative['selection']['candidates']
                smallest = min(c['brier'] for c in candidates)
                winner = min(c['lambda'] for c in candidates if c['brier'] <= smallest+1e-12)
                self.assertEqual(minimum['selected_lambda'], winner)
                self.assertLessEqual(conservative['selected_lambda'], winner)

    def test_full_calibrator_failure_stops_instead_of_silent_substitution(self):
        with patch.object(core, 'fit_intercept', return_value={'success': False}):
            with self.assertRaises(RuntimeError):
                core.fit_treatments(self.z, self.y, self.groups, self.y,
                                    'ambient_balanced', 79, self.config)

    def test_extreme_predictions_remain_independent_binary_probabilities(self):
        fitted, _ = core.fit_treatments(self.z, self.y, self.groups, self.y,
                                       'ambient_balanced', 79, self.config)
        values = core.predict_treatments(np.array([[-1e6]*4, [0.]*4, [1e6]*4]), fitted)
        self.assertEqual(set(values), set(self.config['methods']))
        for p in values.values():
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue(((p >= 0) & (p <= 1)).all())
            self.assertTrue((np.diff(p, axis=0) >= 0).all())
            self.assertGreater(p[-1].sum(), 3.99)

    def test_fitting_interfaces_exclude_evaluation_inputs(self):
        self.assertEqual(list(inspect.signature(core.fit_heads).parameters),
                         ['xfit', 'yfit', 'weighting', 'config'])
        self.assertEqual(list(inspect.signature(core.fit_treatments).parameters),
                         ['zcal', 'ycal', 'artists', 'yfit', 'weighting', 'seed', 'config'])


if __name__ == '__main__':
    unittest.main()
