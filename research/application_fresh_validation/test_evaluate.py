"""Synthetic-only checks; no new sampled songs or their predictions are read."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from . import evaluate as module


class FreshEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0],
                           [1, 0, 0, 1], [0, 1, 0, 0], [1, 1, 1, 1]])
        self.raw = np.array([[.2, .8, .5, .6], [.6, .4, .8, .1], [.4, .7, .7, .2],
                             [.7, .2, .2, .8], [.1, .6, .7, .3], [.8, .9, .6, .6]])
        self.fixed = self.raw.copy()
        self.fixed[:, 2] = [.3, .7, .6, .1, .4, .4]
        self.artists = np.array(['a', 'b', 'b', 'c', 'c', 'c'])
        self.decisions = {'f1': self.raw >= .5, 'precision_target': self.raw >= .8}

    def test_metrics_match_independent_scores_and_artist_counts(self):
        result = module.probability_metrics(self.y, self.raw, self.artists, self.decisions)
        brier = [brier_score_loss(self.y[:, j], self.raw[:, j]) for j in range(4)]
        logs = [log_loss(self.y[:, j], self.raw[:, j], labels=[0, 1]) for j in range(4)]
        self.assertAlmostEqual(result['macro_brier'], np.mean(brier))
        self.assertAlmostEqual(result['macro_log_loss'], np.mean(logs))
        for j, label in enumerate(module.LABELS):
            item = result['per_label'][label]
            self.assertAlmostEqual(item['brier'], brier[j])
            self.assertAlmostEqual(item['log_loss'], logs[j])
            self.assertEqual(item['positive_artists'], len(set(self.artists[self.y[:, j] == 1])))
        for policy, d in self.decisions.items():
            saved = result['policies'][policy]
            self.assertEqual(saved['false_positives'], ((d == 1) & (self.y == 0)).sum())
            self.assertEqual(saved['false_negatives'], ((d == 0) & (self.y == 1)).sum())
            self.assertEqual(saved['coverage'], d.any(1).mean())

    def test_single_outcome_label_retains_probability_metrics_without_ap(self):
        y = self.y.copy(); y[:, 2] = 0
        result = module.probability_metrics(y, self.raw, self.artists, self.decisions)
        self.assertIsNone(result['per_label']['ambient']['average_precision'])
        self.assertEqual(result['per_label']['ambient']['positive_artists'], 0)
        self.assertTrue(np.isfinite(result['macro_brier']))
        self.assertTrue(np.isfinite(result['macro_log_loss']))

    def test_paired_bootstrap_matches_independent_concatenated_artist_draws(self):
        result, draws = module.paired_bootstrap(self.y, self.raw, self.fixed, self.artists,
                                               seed=129, repetitions=79)
        unique = np.unique(self.artists)
        random = np.random.default_rng(129)
        manual = {'brier': [], 'log_loss': []}
        for _ in range(79):
            chosen = random.integers(0, len(unique), size=len(unique))
            ix = np.concatenate([np.flatnonzero(self.artists == unique[i]) for i in chosen])
            for metric in manual:
                per_label = []
                for j in range(4):
                    if metric == 'brier':
                        delta = (self.fixed[ix, j]-self.y[ix, j])**2-(self.raw[ix, j]-self.y[ix, j])**2
                        per_label.append(delta.mean())
                    else:
                        per_label.append(log_loss(self.y[ix, j], self.fixed[ix, j], labels=[0, 1])
                                         -log_loss(self.y[ix, j], self.raw[ix, j], labels=[0, 1]))
                manual[metric].append([np.mean(per_label)]+per_label)
        for metric, values in manual.items():
            np.testing.assert_allclose(draws['bootstrap_'+metric], values, atol=1e-15, rtol=0)
            np.testing.assert_allclose(result['metrics'][metric]['ci95'],
                                       np.quantile(np.array(values)[:, 0], [.025, .975]), atol=1e-15)
        self.assertEqual(draws['bootstrap_artist_indices'].shape, (79, 3))

    def test_duplicating_tracks_preserves_cluster_contrast(self):
        original, draws = module.paired_bootstrap(self.y, self.raw, self.fixed, self.artists,
                                                 seed=182, repetitions=17)
        repeated, again = module.paired_bootstrap(np.repeat(self.y, 2, axis=0),
                                                   np.repeat(self.raw, 2, axis=0),
                                                   np.repeat(self.fixed, 2, axis=0),
                                                   np.repeat(self.artists, 2), seed=182, repetitions=17)
        for metric in ['brier', 'log_loss']:
            np.testing.assert_allclose(draws['bootstrap_'+metric], again['bootstrap_'+metric], atol=1e-15)
            self.assertAlmostEqual(original['metrics'][metric]['delta'], repeated['metrics'][metric]['delta'])

    def test_identity_has_exact_zero_paired_changes(self):
        intervals, draws = module.paired_bootstrap(self.y, self.raw, self.raw, self.artists,
                                                  seed=271, repetitions=11)
        for metric in ['brier', 'log_loss']:
            np.testing.assert_array_equal(draws['bootstrap_'+metric], 0.)
            self.assertEqual(intervals['metrics'][metric]['ci95'], [0., 0.])

    def test_prespecified_decision_rule_and_coverage_gate(self):
        metrics = {'raw': {'macro_brier': .15, 'macro_log_loss': .5},
                   'weight_corrected': {'macro_brier': .14, 'macro_log_loss': .49}}
        intervals = {'metrics': {'brier': {'ci95': [-.02, -.001]}, 'log_loss': {'ci95': [-.03, .01]}}}
        coverage = {'track_fraction': 1., 'artist_fraction': 1.}
        self.assertEqual(module.decision_rule(metrics, intervals, coverage, 0)['verdict'], 'go')
        crossing = copy.deepcopy(intervals); crossing['metrics']['brier']['ci95'][1] = 0.
        self.assertEqual(module.decision_rule(metrics, crossing, coverage, 0)['verdict'], 'promising')
        for key in coverage:
            low = dict(coverage); low[key] = .799999
            self.assertEqual(module.decision_rule(metrics, intervals, low, 0)['verdict'], 'incomplete_evidence')
            low[key] = .8
            self.assertEqual(module.decision_rule(metrics, intervals, low, 0)['verdict'], 'go')
        mixed = copy.deepcopy(metrics); mixed['weight_corrected']['macro_log_loss'] = .51
        self.assertEqual(module.decision_rule(mixed, crossing, coverage, 0)['verdict'], 'inconclusive')
        for key in ['brier', 'log_loss']:
            harmful = copy.deepcopy(intervals); harmful['metrics'][key]['ci95'] = [.001, .02]
            self.assertEqual(module.decision_rule(mixed, harmful, coverage, 0)['verdict'], 'harmful')
            self.assertEqual(module.decision_rule(mixed, harmful, {'track_fraction': .5, 'artist_fraction': .5}, 0)['verdict'], 'harmful')
        with self.assertRaises(ValueError):
            module.decision_rule(metrics, intervals, coverage, 1)

    def test_coverage_counts_tracks_and_artists_and_rejects_replacements(self):
        selected = [{'track_id': str(i), 'artist_id': artist, 'tags': ['genre---rock']}
                    for i, artist in enumerate(['a', 'a', 'b', 'c'])]
        observed = selected[:2]
        coverage = module.coverage_counts(selected, observed)
        self.assertEqual(coverage['track_fraction'], .5)
        self.assertEqual(coverage['artist_fraction'], 1/3)
        self.assertEqual(coverage['missing_ids'], ['2', '3'])
        for key, replacement in [('track_id', 'new'), ('artist_id', 'new'), ('tags', [])]:
            changed = copy.deepcopy(observed); changed[0][key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.coverage_counts(selected, changed)
        with self.assertRaises(ValueError):
            module.coverage_counts(selected, observed+observed)

    def synthetic_bundle(self):
        width = module.FEATURE_WIDTH
        coefficients = np.zeros((4, width)); coefficients[:, :4] = np.eye(4)
        return {'means': np.zeros((4, width)), 'scales': np.ones((4, width)),
                'coefficients': coefficients, 'intercepts': np.zeros(4),
                'thresholds': {'f1': [.4, .275, .375, .275],
                               'precision_target': [.625, 1.01, 1.01, .525]}}

    def test_actual_old_and_new_interfaces_preserve_decisions_and_extremes(self):
        x = np.zeros((6, module.FEATURE_WIDTH))
        x[:, :4] = [[-1e100]*4, [1e100]*4, [0.]*4, [1., -1., 2., -2.],
                     [0., 0., np.log((.375-1e-8)/(1-.375+1e-8)), 0.],
                     [0., 0., np.log((.375+1e-8)/(1-.375-1e-8)), 0.]]
        z, raw, fixed, decisions, thresholds, checks = module.predict_and_check(
            x, self.synthetic_bundle(), {'logit_offsets': [0., 0., np.log(244/962), 0.]})
        np.testing.assert_array_equal(z, x[:, :4])
        np.testing.assert_array_equal(raw[:, [0, 1, 3]], fixed[:, [0, 1, 3]])
        self.assertEqual(checks['changed_tag_decisions'], 0)
        self.assertEqual(checks['label_policy_interface_checks'], len(x)*8)
        self.assertEqual(thresholds['precision_target'][2], 1.01)
        self.assertEqual(decisions['f1'][4:, 2].tolist(), [False, True])

    def test_interface_drift_is_an_integrity_error_not_statistical_failure(self):
        real = module.app_predict
        def changed(*args, **kwargs):
            result = real(*args, **kwargs)
            result[0]['selected'] = not result[0]['selected']
            return result
        with patch.object(module, 'app_predict', side_effect=changed), self.assertRaises(ValueError):
            module.predict_and_check(np.zeros((1, module.FEATURE_WIDTH)), self.synthetic_bundle(),
                                     {'logit_offsets': [0., 0., np.log(244/962), 0.]})

    def test_invalid_probability_inputs_and_too_few_artists_are_rejected(self):
        bad = self.raw.copy(); bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            module.probability_metrics(self.y, bad, self.artists, self.decisions)
        with self.assertRaises(ValueError):
            module.paired_bootstrap(self.y, self.raw, self.fixed, ['a']*len(self.y))


if __name__ == '__main__':
    unittest.main()
