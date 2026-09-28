"""Synthetic tests for fit-only model development and hierarchical OOF choice."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import expit

from research.application_auto_improvement import core


class ApplicationAutoImprovementTests(unittest.TestCase):
    def test_artist_folds_keep_whole_artists_and_are_permutation_stable(self):
        artists = np.array(['a'] * 7 + ['b'] * 4 + ['c'] * 3 + ['d'] * 2 + ['e'] * 2 + ['f'])
        folds = core.artist_folds(artists, 3, seed=71)
        for artist in set(artists):
            self.assertEqual(len(set(folds[artists == artist])), 1)
        for fold in range(3):
            self.assertFalse(set(artists[folds == fold]) & set(artists[folds != fold]))
        permutation = np.random.default_rng(14).permutation(len(artists))
        np.testing.assert_array_equal(core.artist_folds(artists[permutation], 3, seed=71), folds[permutation])
        self.assertLessEqual(np.ptp(np.bincount(folds)), 7)
        with self.assertRaises(ValueError):
            core.artist_folds(['a', 'a'], 2)

    def fixture(self):
        return np.array([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1]], float), np.array([0, 0, 0, 0, 1, 1])

    def test_scaler_and_prior_offset_use_fit_rows_only(self):
        fit, y = self.fixture()
        model = core.fit_head(fit, y, C=.001, balanced=True)
        np.testing.assert_array_equal(model['mean'], fit.mean(0))
        np.testing.assert_allclose(model['scale'], [fit[:, 0].std(), 1.], rtol=0, atol=0)
        self.assertEqual(float(model['offset']), np.log(2/4))
        before = {k: v.copy() for k, v in model.items()}
        predictions = core.predict_head(np.array([[1e6, -1e6], [-1e6, 1e6]]), model)
        np.testing.assert_allclose(predictions['probability'], expit(predictions['logits'] + np.log(2/4)))
        for key in before:
            np.testing.assert_array_equal(before[key], model[key])
        np.testing.assert_array_equal(fit, self.fixture()[0])

    def test_baseline_weighting_only_changes_ambient_recipe(self):
        fit, y = self.fixture()
        ambient = core.fit_baseline_head(fit, y, 'ambient')
        pop = core.fit_baseline_head(fit, y, 'pop')
        self.assertTrue(bool(ambient['balanced'])); self.assertFalse(bool(pop['balanced']))
        self.assertEqual(float(pop['offset']), 0.)
        result = core.predict_head(fit, pop)
        np.testing.assert_array_equal(result['raw'], result['probability'])

    def test_fitted_head_roundtrips_without_pickle(self):
        fit, y = self.fixture(); model = core.fit_head(fit, y)
        self.assertTrue(all(isinstance(v, np.ndarray) and v.dtype != object for v in model.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'head.npz'; np.savez(path, **model)
            with np.load(path, allow_pickle=False) as loaded:
                reconstructed = core.predict_head(fit, dict(loaded))
            np.testing.assert_array_equal(reconstructed['raw'], core.predict_head(fit, model)['raw'])

    def test_fit_and_prediction_reject_invalid_arrays(self):
        fit, y = self.fixture()
        for bad_x, bad_y in ((fit, np.zeros(len(y))), (fit[:-1], y), (fit * np.nan, y), (fit, y + .5)):
            with self.assertRaises(ValueError):
                core.fit_head(bad_x, bad_y)
        model = core.fit_head(fit, y); model['scale'][0] = 0
        with self.assertRaises(ValueError):
            core.predict_head(fit, model)

    def test_metrics_match_independent_confusion_and_probability_losses(self):
        y = np.array([1, 0, 1, 0]); p = np.array([.8, .7, .2, .1]); d = p >= .5
        actual = core.binary_metrics(y, p, d)
        self.assertEqual([actual[k] for k in ('tp', 'fp', 'fn', 'tn')], [1, 1, 1, 1])
        self.assertEqual(actual['precision'], .5); self.assertEqual(actual['recall'], .5); self.assertEqual(actual['f1'], .5)
        self.assertAlmostEqual(actual['brier'], float(np.mean((p-y)**2)))
        self.assertAlmostEqual(actual['log_loss'], float(-np.log([.8, .3, .2, .9]).mean()))
        self.assertEqual(core.probability_metrics(y, y)['brier'], 0.)
        self.assertTrue(np.isfinite(core.probability_metrics(y, 1-y)['log_loss']))

    def select(self, y, bp, br, candidates, **kwargs):
        threshold = kwargs.pop('baseline_threshold', .5)
        return core.select_inner_oof(y, bp, br, np.array(br) >= threshold, candidates, threshold, **kwargs)

    def test_hierarchical_selection_uses_brier_before_threshold_and_regularization_ties(self):
        y = np.array([1, 1, 0, 0]); baseline = np.array([.6, .6, .4, .4])
        better = np.array([.9, .9, .1, .1])
        selected = self.select(y, baseline, baseline, [
            {'id': 'weak_regularization', 'C': .001, 'probability': better},
            {'id': 'strong_regularization', 'C': .0001, 'probability': better.copy()},
            {'id': 'slightly_worse', 'C': .00003, 'probability': np.array([.8, .8, .2, .2])},
        ])
        self.assertEqual(selected['selected_head'], 'strong_regularization')
        self.assertEqual(selected['decision_space'], 'probability')
        self.assertFalse(selected['fallback'])
        self.assertEqual(selected['selected_metrics']['fp'], 0)
        self.assertEqual(selected['selected_metrics']['tp'], 2)

    def test_head_tie_prefers_baseline_but_can_tune_its_raw_threshold(self):
        y = np.array([1, 1, 0, 0]); raw = np.array([.9, .8, .7, .2]); corrected = np.array([.7, .6, .4, .1])
        result = self.select(y, corrected, raw, [{'id': 'same', 'C': .0001, 'probability': corrected.copy()}])
        self.assertEqual(result['selected_head'], 'baseline')
        self.assertEqual(result['decision_space'], 'raw')
        self.assertEqual(result['selected_metrics']['fp'], 0)
        self.assertEqual(result['selected_metrics']['tp'], 2)
        self.assertTrue(result['threshold_changed'])
        self.assertEqual(result['selected_metrics']['brier'], core.probability_metrics(y, corrected)['brier'])
        self.assertGreater(result['threshold'], .7)

    def test_no_feasible_threshold_restores_exact_complete_baseline(self):
        y = np.array([1, 1, 0, 0]); raw = np.array([.7, .7, .3, .3]); probability = np.array([.51, .51, .49, .49])
        candidate = np.array([.9, .4, .45, .1])
        threshold = np.nextafter(.5, 0.)
        result = self.select(y, probability, raw, [{'id': 'improved_loss_bad_order', 'C': .0001, 'probability': candidate}],
                             baseline_threshold=threshold)
        self.assertTrue(result['fallback']); self.assertEqual(result['selected_head'], 'baseline')
        self.assertEqual(result['chosen_head_before_threshold'], 'improved_loss_bad_order')
        self.assertEqual(result['threshold'], threshold)
        self.assertEqual(result['decision_space'], 'raw')
        self.assertEqual(result['selected_metrics'], result['baseline_metrics'])
        self.assertFalse(result['head_changed']); self.assertFalse(result['threshold_changed'])

    def test_candidate_needs_both_probability_metrics_and_both_target_classes(self):
        y = np.array([1, 1, 0, 0]); p = np.array([.6, .6, .4, .4])
        result = self.select(y, p, p, [{'id': 'worse', 'C': .0001, 'probability': 1-p}])
        rejected = next(row for row in result['eligible_heads'] if row['id'] == 'worse')
        self.assertFalse(rejected['eligible'])
        self.assertEqual(set(rejected['reasons']), {'brier_worse', 'log_loss_worse'})
        with self.assertRaises(ValueError):
            self.select(np.zeros(4), p, p, [])
        with self.assertRaises(ValueError):
            core.select_inner_oof(y, p, p, np.zeros(4), [], .5)

    def test_improved_brier_is_rejected_if_log_loss_worsens(self):
        y = np.array([1] * 5 + [0] * 5)
        baseline = np.where(y == 1, .6, .4)
        candidate = np.where(y == 1, .99, .01); candidate[0] = 1e-6
        result = self.select(y, baseline, baseline, [{'id': 'overconfident_error', 'C': .0001,
                                                     'probability': candidate}])
        rejected = next(row for row in result['eligible_heads'] if row['id'] == 'overconfident_error')
        self.assertLess(rejected['brier'], result['baseline_metrics']['brier'])
        self.assertEqual(rejected['reasons'], ['log_loss_worse'])
        self.assertEqual(result['selected_head'], 'baseline')

    def test_original_threshold_is_included_exactly_and_raw_probability_spaces_are_distinct(self):
        y = np.array([1, 1, 0, 0]); raw = np.array([.39, .6, .2, .1]); corrected = np.array([.15, .3, .06, .03])
        threshold = .37499999999999994
        result = self.select(y, corrected, raw, [], baseline_threshold=threshold, threshold_grid=[.9])
        self.assertEqual(result['threshold'], threshold)
        self.assertEqual(result['selected_metrics']['tp'], 2)
        self.assertFalse(result['fallback'])


if __name__ == '__main__':
    unittest.main()
