import copy
import unittest

import numpy as np

from research.application_candidate_validation import evaluate as ev


class CandidateConfirmationEvaluationTests(unittest.TestCase):
    def fixture(self, n=300):
        y = np.tile([[0, 0, 1, 1], [1, 1, 0, 0]], (n//2, 1))
        artists = np.array(['artist-%03d' % (i % 250) for i in range(n)])
        baseline = .1+.8*y.astype(float)
        candidate = baseline.copy()
        for j in ev.FOCUS:
            rows = np.flatnonzero(y[:, j] == 0)[:20]
            baseline[rows, j] = .7
            candidate[rows[:10], j] = .7
        def policies(p):
            main = p >= .5
            selective = main.copy(); selective[:, ev.FOCUS] = False
            return {'f1': main, 'precision_target': selective}
        return y, artists, baseline, candidate, policies(baseline), policies(candidate)

    def assessment_fixture(self):
        y, artists, p, q, d, e = self.fixture()
        metrics = {'baseline': ev.probability_metrics(y, p, artists, d),
                   'candidate': ev.probability_metrics(y, q, artists, e)}
        intervals, _ = ev.paired_bootstrap(y, p, q, d, e, artists)
        coverage = {'observed_tracks': len(y), 'observed_artists': len(set(artists)),
                    'track_fraction': 1., 'artist_fraction': 1.}
        integrity = {'electronic_rock_exact': True, 'precision_target_exact': True}
        return y, artists, metrics, intervals, coverage, integrity

    def test_rates_count_label_cases_not_just_flagged_songs(self):
        y = np.zeros((2, 4), int)
        d = {'f1': np.ones_like(y), 'precision_target': np.zeros_like(y)}
        m = ev.probability_metrics(y, np.ones_like(y)*.7, ['a', 'b'], d)
        self.assertEqual(m['policies']['f1']['focus_false_positives'], 4)
        self.assertEqual(m['policies']['f1']['focus_false_positive_rate'], 2)
        self.assertIsNone(m['per_label']['pop']['average_precision'])

    def test_bootstrap_replays_clusters_and_all_labels_jointly(self):
        y, artists, p, q, d, e = self.fixture()
        intervals, draws = ev.paired_bootstrap(y, p, q, d, e, artists, repetitions=20)
        unique = np.unique(artists)
        for k, sample in enumerate(draws['bootstrap_artist_indices']):
            indices = np.concatenate([np.flatnonzero(artists == unique[j]) for j in sample])
            ma = ev.probability_metrics(y[indices], p[indices], artists[indices], {key: value[indices] for key, value in d.items()})
            mb = ev.probability_metrics(y[indices], q[indices], artists[indices], {key: value[indices] for key, value in e.items()})
            for key in ['macro_brier', 'macro_log_loss']:
                self.assertAlmostEqual(draws['bootstrap_delta_'+key][k], mb[key]-ma[key], places=14)
            for key in ['micro_f1', 'micro_recall']:
                self.assertAlmostEqual(draws['bootstrap_delta_'+key][k], mb['policies']['f1'][key]-ma['policies']['f1'][key], places=14)
            rate_delta = mb['policies']['f1']['focus_false_positive_rate']-ma['policies']['f1']['focus_false_positive_rate']
            self.assertAlmostEqual(draws['bootstrap_delta_focus_fp_rate'][k], rate_delta)
            self.assertAlmostEqual(draws['bootstrap_delta_focus_fp_count_equivalent'][k], rate_delta*len(y))
            self.assertEqual(draws['bootstrap_track_counts'][k], len(indices))
        self.assertEqual(intervals['metrics']['focus_fp_relative_change']['delta'], -.5)

    def test_probabilities_unchanged_when_only_thresholds_change(self):
        y, artists, p, _, d, e = self.fixture()
        a, b = ev.probability_metrics(y, p, artists, d), ev.probability_metrics(y, p, artists, e)
        self.assertEqual(a['macro_brier'], b['macro_brier'])
        self.assertEqual(a['macro_log_loss'], b['macro_log_loss'])
        self.assertEqual(a['per_label'], b['per_label'])

    def test_joint_benefit_go(self):
        y, artists, metrics, intervals, coverage, integrity = self.assessment_fixture()
        result = ev.assess(metrics, intervals, coverage, y, artists, integrity=integrity)
        self.assertEqual(result['verdict'], 'go')
        self.assertTrue(result['point_gate_passed'])

    def test_uncertain_either_benefit_only_promising(self):
        y, artists, metrics, intervals, coverage, integrity = self.assessment_fixture()
        for key in ['macro_brier', 'focus_fp_rate']:
            changed = copy.deepcopy(intervals)
            changed['metrics'][key]['ci95'][1] = 0.
            result = ev.assess(metrics, changed, coverage, y, artists, integrity=integrity)
            self.assertEqual(result['verdict'], 'promising')

    def test_failed_guard_not_saved_by_significant_benefits(self):
        y, artists, metrics, intervals, coverage, integrity = self.assessment_fixture()
        metrics['candidate']['policies']['f1']['micro_recall'] = .5
        result = ev.assess(metrics, intervals, coverage, y, artists, integrity=integrity)
        self.assertEqual(result['verdict'], 'not_passed')

    def test_coverage_and_class_support_take_priority(self):
        y, artists, metrics, intervals, coverage, integrity = self.assessment_fixture()
        for key in ['track_fraction', 'artist_fraction']:
            bad = {**coverage, key: .7999}
            result = ev.assess(metrics, intervals, bad, y, artists, integrity=integrity)
            self.assertEqual(result['verdict'], 'incomplete_evidence')
        y[:280, 1] = 0
        result = ev.assess(metrics, intervals, coverage, y, artists, integrity=integrity)
        self.assertEqual(result['verdict'], 'incomplete_evidence')
        self.assertFalse(result['evidence_checks']['pop_positive_tracks'])

    def test_no_baseline_false_positives_never_go_or_invent_relative_effect(self):
        y, artists, p, _, _, _ = self.fixture()
        p = .1+.8*y
        d = {'f1': y.astype(bool), 'precision_target': y.astype(bool)}
        intervals, _ = ev.paired_bootstrap(y, p, p, d, d, artists)
        self.assertIsNone(intervals['metrics']['focus_fp_relative_change']['delta'])
        self.assertIsNone(intervals['metrics']['focus_fp_relative_change']['ci95'])
        self.assertEqual(intervals['metrics']['focus_fp_relative_change']['undefined_replicates'], 2000)
        metrics = {arm: ev.probability_metrics(y, p, artists, d) for arm in ['baseline', 'candidate']}
        coverage = {'observed_tracks': 300, 'observed_artists': 250, 'track_fraction': 1., 'artist_fraction': 1.}
        result = ev.assess(metrics, intervals, coverage, y, artists,
                           integrity={'electronic_rock_exact': True, 'precision_target_exact': True})
        self.assertEqual(result['verdict'], 'not_passed')
        self.assertIsNone(result['focus_fp_relative_reduction'])

    def test_integrity_and_frozen_config_fail_closed(self):
        y, artists, metrics, intervals, coverage, integrity = self.assessment_fixture()
        with self.assertRaisesRegex(ValueError, 'Integrity'):
            ev.assess(metrics, intervals, coverage, y, artists, integrity={**integrity, 'electronic_rock_exact': False})
        config = copy.deepcopy(ev.DEFAULT_CONFIG); config['minimum_observed_tracks'] = 100
        with self.assertRaisesRegex(ValueError, 'frozen rule'):
            ev.assess(metrics, intervals, coverage, y, artists, config=config, integrity=integrity)

    def test_coverage_rejects_replacements_duplicates_and_changed_labels(self):
        rows = [{'track_id': 't1', 'artist_id': 'a1', 'tags': ['pop']},
                {'track_id': 't2', 'artist_id': 'a2', 'tags': []}]
        self.assertEqual(ev.coverage_counts(rows, rows[:1])['artist_fraction'], .5)
        self.assertEqual(ev.coverage_counts(rows, [])['track_fraction'], 0.)
        for bad in [[rows[0], rows[0]], [{**rows[0], 'track_id': 'replacement'}],
                    [{**rows[0], 'artist_id': 'changed'}], [{**rows[0], 'tags': []}]]:
            with self.assertRaises(ValueError):
                ev.coverage_counts(rows, bad)

    def test_predict_preserves_exact_controls_and_fails_changed_head(self):
        width = 2304
        baseline = {'means': np.zeros((4, width)), 'scales': np.ones((4, width)),
                    'coefficients': np.zeros((4, width)), 'intercepts': np.array([.3, .1, -.4, -.2]),
                    'thresholds': {'f1': [.39999999999999997, .27499999999999997, .37499999999999994, .27499999999999997],
                                   'precision_target': [.625, 1.01, 1.01, .525]}}
        baseline['coefficients'][:, 0] = [.2, .3, -.1, -.2]
        correction = {'logit_offsets': [0., 0., -1., 0.]}
        candidate = copy.deepcopy(baseline); candidate['offsets'] = np.zeros(4)
        candidate['coefficients'][1:3, 0] = [.1, -.3]
        candidate['thresholds']['f1'][1:3] = [.3, .4]
        x = np.zeros((3, width)); x[:, 0] = [-.7, 0., .4]
        p, checks = ev.predict_and_check(x, baseline, correction, candidate)
        self.assertTrue(checks['electronic_rock_exact'])
        self.assertEqual(checks['candidate_label_policy_interface_checks'], 24)
        self.assertTrue(np.array_equal(p['baseline']['decisions']['precision_target'], p['candidate']['decisions']['precision_target']))
        candidate['intercepts'][0] += .01
        with self.assertRaisesRegex(ValueError, 'electronic/rock'):
            ev.predict_and_check(x, baseline, correction, candidate)


if __name__ == '__main__':
    unittest.main()
