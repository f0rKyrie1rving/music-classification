"""Small synthetic construction checks; never train on the research pool."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from research.application_candidate import build


class CandidateBuildTest(unittest.TestCase):
    def setUp(self):
        self.config = build.read(build.HERE / 'config.json')
        self.artists = np.repeat([f'artist-{i}' for i in range(16)], 2)
        self.cohorts = np.repeat(['first', 'second'], 16)
        self.x = np.arange(96, dtype=float).reshape(32, 3) / 20
        self.y = np.zeros((32, 4), dtype=np.int64)
        self.y[:, 1] = np.arange(32) % 3 == 0
        self.y[[0, 1, 3, 5, 7, 8, 12, 17, 20], 2] = 1
        self.raw = [.4, .275, .375, .275]
        self.schedule = build.split_schedule(self.artists, self.cohorts, self.config)

    def provider(self, calls):
        def provide(name, label, recipe, indices):
            calls.append((name, label, recipe, indices.copy()))
            j = build.LABELS.index(label)
            positive = int(self.y[indices, j].sum())
            offset = (np.log(positive / (len(indices) - positive))
                      if recipe == 'baseline' and label == 'ambient' else 0.)
            return {'mean': self.x[indices].mean(0), 'scale': np.ones(3),
                    'coef': np.array([.3, -.2, .1]),
                    'intercept': np.asarray(-.4 if recipe == 'new' else .2),
                    'offset': np.asarray(offset)}
        return provide

    def test_split_has_whole_artists_cohort_balance_and_exact_coverage(self):
        held_rows = []
        for split in self.schedule:
            fit, held = split['train'], split['held']
            held_rows.extend(held)
            self.assertFalse(set(self.artists[fit]) & set(self.artists[held]))
            self.assertEqual(sorted(fit + held), list(range(len(self.x))))
            self.assertEqual(set(self.cohorts[held]), {'first', 'second'})
            for cohort in ['first', 'second']:
                self.assertEqual(np.sum(self.cohorts[held] == cohort), 4)
        self.assertEqual(sorted(held_rows), list(range(len(self.x))))
        permutation = np.arange(len(self.x))[::-1]
        reverse = build.split_schedule(self.artists[permutation], self.cohorts[permutation], self.config)
        for original, reordered in zip(self.schedule, reverse):
            self.assertEqual(set(self.artists[original['held']]),
                             set(self.artists[permutation][reordered['held']]))

    def test_selection_uses_fit_indices_fit_only_offsets_and_replays(self):
        calls = []
        arrays, selection = build.select_thresholds(
            self.x, self.y, self.schedule, self.config, self.raw, self.provider(calls))
        self.assertEqual(len(calls), 16)
        np.testing.assert_array_equal(arrays['coverage'], np.ones(len(self.x), dtype=int))
        ambient_cutoffs = []
        for split in self.schedule:
            fit, held = np.asarray(split['train']), np.asarray(split['held'])
            these = [call for call in calls if call[0].startswith(f"inner_{split['fold']}/")]
            self.assertEqual({(c[1], c[2]) for c in these},
                             {(label, recipe) for label in ['pop', 'ambient'] for recipe in ['baseline', 'new']})
            for call in these:
                np.testing.assert_array_equal(call[3], fit)
            offset = np.log(self.y[fit, 2].sum() / (len(fit) - self.y[fit, 2].sum()))
            expected = expit(np.log(self.raw[2]) - np.log1p(-self.raw[2]) + offset)
            ambient_cutoffs.append(expected)
            for recipe in ['baseline', 'new']:
                np.testing.assert_array_equal(arrays[f'{recipe}__original_cutoff'][held, 1],
                                              np.full(len(held), expected))
                np.testing.assert_array_equal(arrays[f'{recipe}__original_cutoff'][held, 0],
                                              np.full(len(held), self.raw[1]))
            z = (self.x[held] - self.x[fit].mean(0)) @ np.array([.3, -.2, .1]) + .2
            np.testing.assert_array_equal(arrays['baseline__probability'][held, 1], expit(z + offset))
        self.assertGreater(len(set(ambient_cutoffs)), 1)
        np.testing.assert_array_equal(arrays['baseline__original_decision'],
                                      arrays['baseline__raw'] >= np.asarray(self.raw)[[1, 2]])
        np.testing.assert_array_equal(arrays['new__original_decision'],
                                      arrays['new__probability'] >= arrays['new__original_cutoff'])
        second, second_selection = build.select_thresholds(
            self.x, self.y, self.schedule, self.config, self.raw, self.provider([]))
        self.assertEqual(selection, second_selection)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], second[key])

    def test_original_fallback_converts_using_full_training_prior(self):
        forced = {'kind': 'original', 'threshold': None, 'feasible': False, 'fallback': True}
        with patch.object(build, 'choose_policy', return_value=forced):
            arrays, selection = build.select_thresholds(
                self.x, self.y, self.schedule, self.config, self.raw, self.provider([]))
        offset = np.log(self.y[:, 2].sum() / (len(self.y) - self.y[:, 2].sum()))
        expected = expit(np.log(self.raw[2]) - np.log1p(-self.raw[2]) + offset)
        self.assertEqual(selection['final_f1_thresholds'], [.4, .275, expected, .275])
        self.assertEqual(selection['choices']['ambient']['full_development_reference_offset'], offset)
        self.assertTrue(selection['choices']['ambient']['fallback'])
        self.assertNotEqual(expected, float(arrays['new__original_cutoff'][0, 1]))

    def test_selection_rejects_duplicate_or_missing_oof_rows(self):
        duplicate = deepcopy(self.schedule)
        duplicate[0]['held'].append(duplicate[0]['held'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate OOF indices'):
            build.select_thresholds(self.x, self.y, duplicate, self.config, self.raw, self.provider([]))
        missing = deepcopy(self.schedule)
        missing[0]['held'].pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete OOF coverage'):
            build.select_thresholds(self.x, self.y, missing, self.config, self.raw, self.provider([]))

    def test_assembly_copies_real_application_controls_without_precision_loss(self):
        baseline = build.load_bundle()
        snapshot = {k: v.copy() for k, v in baseline.items() if isinstance(v, np.ndarray)}
        width = baseline['means'].shape[1]
        focus = {}
        for index, label in enumerate(['pop', 'ambient']):
            focus[label] = {'mean': np.full(width, .12345678912345678 + index),
                            'scale': np.full(width, 1.9876543219876543 + index),
                            'coef': np.full(width, .000000012345678912345 + index),
                            'intercept': np.asarray(.9876543219876543 + index)}
        candidate = build.assemble_candidate(baseline, focus)
        for key in ['means', 'scales', 'coefficients', 'intercepts']:
            self.assertEqual(candidate[key].dtype, np.dtype('float64'))
            np.testing.assert_array_equal(candidate[key][[0, 3]], baseline[key][[0, 3]])
            self.assertFalse(np.shares_memory(candidate[key], baseline[key]))
        for j, label in [(1, 'pop'), (2, 'ambient')]:
            for source, target in [('mean', 'means'), ('scale', 'scales'),
                                   ('coef', 'coefficients'), ('intercept', 'intercepts')]:
                np.testing.assert_array_equal(candidate[target][j], focus[label][source])
        np.testing.assert_array_equal(candidate['offsets'], np.zeros(4))
        for key, value in snapshot.items():
            np.testing.assert_array_equal(baseline[key], value)

    def test_failed_model_fit_retains_failure_and_cannot_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'freeze.json').write_text('{}')
            provider = build.model_provider(out, self.x, self.y, self.artists, self.config)
            indices = np.arange(20)
            with patch.object(build, 'fit_head', side_effect=RuntimeError('injected failure')) as fit:
                with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                    provider('inner_0/new_pop', 'pop', 'new', indices)
                status = out / 'models/inner_0/new_pop.status.json'
                self.assertEqual(build.read(status)['state'], 'failed')
                with self.assertRaisesRegex(ValueError, 'no automatic refit'):
                    provider('inner_0/new_pop', 'pop', 'new', indices)
                self.assertEqual(fit.call_count, 1)
            self.assertFalse((out / 'models/inner_0/new_pop.json').exists())

    def test_model_cache_replays_without_fit_and_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'freeze.json').write_text('{}')
            indices = np.arange(20)
            head = self.provider([])('test', 'ambient', 'baseline', indices)
            provider = build.model_provider(out, self.x, self.y, self.artists, self.config)
            with patch.object(build, 'fit_head', return_value=head) as fit:
                provider('inner_0/baseline_ambient', 'ambient', 'baseline', indices)
                np.testing.assert_array_equal(fit.call_args.args[0], self.x[indices])
                np.testing.assert_array_equal(fit.call_args.args[1], self.y[indices, 2])
                self.assertTrue(fit.call_args.kwargs['balanced'])
                self.assertEqual(fit.call_args.kwargs['C'], .001)
            replay = build.model_provider(out, self.x, self.y, self.artists, self.config, replay=True)
            with patch.object(build, 'fit_head', side_effect=AssertionError('replay fitted')):
                cached = replay('inner_0/baseline_ambient', 'ambient', 'baseline', indices)
                for key in head:
                    np.testing.assert_array_equal(cached[key], head[key])
                with self.assertRaisesRegex(ValueError, 'Model receipt changed'):
                    replay('inner_0/baseline_ambient', 'ambient', 'baseline', indices[::-1])
                with self.assertRaisesRegex(ValueError, 'Replay cannot refit'):
                    replay('inner_1/baseline_ambient', 'ambient', 'baseline', indices)


if __name__ == '__main__':
    unittest.main()
