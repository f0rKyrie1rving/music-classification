"""Synthetic checks of factorial data boundaries; no project cache is loaded."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from research.application_factorial import core, run


class ApplicationFactorialPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((run.HERE / 'config.json').read_text())
        self.config.update(outer_folds=3, inner_folds=2)
        n = 96
        self.x = np.random.default_rng(803).normal(size=(n, 6))
        self.x[:, 0] = np.arange(n)
        ix = np.arange(n)
        self.y = np.column_stack([
            ix % 2,
            (ix % 2 == 1) & (ix // 2 % 3 != 0),
            (ix % 2 == 0) & (ix // 2 % 4 != 0),
            ix % 2 == 0,
        ]).astype(int)
        self.artists = np.asarray([f'artist_{i//2:03d}' for i in ix])
        names = np.asarray(['dev1206', 'hist239', 'val266', 'fresh204'])
        self.cohorts = names[ix // 24]
        self.thresholds = np.asarray([.4, .27499999999999997,
                                      .37499999999999994, .27499999999999997])
        self.schedule = run.build_schedule(self.artists, self.cohorts, self.config)

    def same_payload(self, left, right):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.same_payload(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.same_payload(a, b)
        else:
            self.assertEqual(left, right)

    def provider(self, x, y, saved):
        def fit(name, indices):
            indices = np.asarray(indices)
            self.assertNotIn(name, saved, 'Each budget/model is fitted once per fold')
            heads = core.fit_recipes(x[indices], y[indices], self.config)
            saved[name] = {'indices': indices.copy(), 'heads': deepcopy(heads)}
            return heads
        return fit

    def evaluate(self, x=None, y=None):
        x = self.x if x is None else x
        y = self.y if y is None else y
        saved = {}
        result = run.evaluate_fold(x, y, self.artists, self.cohorts,
                                   self.schedule[0], self.config, self.thresholds,
                                   self.provider(x, y, saved))
        return result, saved

    def test_schedule_keeps_whole_artists_sources_and_nested_budget_prefixes(self):
        counts = np.zeros(len(self.y), dtype=int)
        for spec in self.schedule:
            train, test = np.asarray(spec['train']), np.asarray(spec['test'])
            np.add.at(counts, test, 1)
            self.assertFalse(set(train) & set(test))
            self.assertEqual(set(train) | set(test), set(range(len(self.y))))
            self.assertFalse(set(self.artists[train]) & set(self.artists[test]))
            inner_counts = {int(i): 0 for i in train}
            for inner in spec['inner']:
                fit, held = np.asarray(inner['train']), np.asarray(inner['test'])
                self.assertEqual(set(fit) | set(held), set(train))
                self.assertFalse(set(fit) & set(held))
                self.assertFalse(set(self.artists[fit]) & set(self.artists[held]))
                for index in held:
                    inner_counts[int(index)] += 1
            self.assertEqual(set(inner_counts.values()), {1})
            np.testing.assert_array_equal(spec['budgets']['full'], train)
            dev = np.asarray(spec['budgets']['dev_only'])
            np.testing.assert_array_equal(dev, train[self.cohorts[train] == 'dev1206'])
            for seed in self.config['sampling_seeds']:
                small = np.asarray(spec['budgets']['f40_' + seed])
                medium = np.asarray(spec['budgets']['f70_' + seed])
                self.assertTrue(set(small) <= set(medium) <= set(train))
                for fraction, subset in [(.4, small), (.7, medium)]:
                    for source in np.unique(self.cohorts):
                        available = set(self.artists[train[self.cohorts[train] == source]])
                        selected = set(self.artists[subset[self.cohorts[subset] == source]])
                        self.assertEqual(len(selected), int(np.ceil(fraction * len(available))))
                    for artist in set(self.artists[subset]):
                        self.assertEqual(set(subset[self.artists[subset] == artist]),
                                         set(train[self.artists[train] == artist]))
                    self.assertFalse(set(self.artists[subset]) & set(self.artists[test]))
        np.testing.assert_array_equal(counts, np.ones(len(self.y), dtype=int))

    def test_changing_all_outer_answers_cannot_change_fits_choices_or_scores(self):
        before, fitted_before = self.evaluate()
        changed_y = self.y.copy()
        changed_y[np.asarray(self.schedule[0]['test'])] = 1 - changed_y[self.schedule[0]['test']]
        after, fitted_after = self.evaluate(y=changed_y)
        self.same_payload(before, after)
        self.same_payload(fitted_before, fitted_after)
        test = np.asarray(self.schedule[0]['test'])
        for saved in fitted_before.values():
            fit = saved['indices']
            self.assertFalse(set(fit) & set(test))
            self.assertFalse(set(self.artists[fit]) & set(self.artists[test]))
            for recipe, heads in saved['heads'].items():
                for j, head in enumerate(heads):
                    np.testing.assert_allclose(head['mean'], self.x[fit].mean(axis=0),
                                               atol=1e-14, rtol=1e-14)
                    positives = int(self.y[fit, j].sum())
                    self.assertEqual(int(head['n_fit']), len(fit))
                    self.assertEqual(int(head['positive_fit']), positives)
                    expected = np.log(positives / (len(fit) - positives)) if recipe == 'baseline' and j == 2 else 0.
                    self.assertEqual(float(head['offset']), expected)

    def test_outer_features_cannot_change_training_models_or_inner_selection(self):
        before, fitted_before = self.evaluate()
        changed_x = self.x.copy()
        changed_x[np.asarray(self.schedule[0]['test'])] += 100.
        after, fitted_after = self.evaluate(x=changed_x)
        self.same_payload(fitted_before, fitted_after)
        self.same_payload(before['inner_arrays'], after['inner_arrays'])
        self.same_payload(before['choices'], after['choices'])
        self.assertFalse(np.array_equal(before['arrays']['full__baseline__probability'],
                                        after['arrays']['full__baseline__probability']))

    def test_each_budget_uses_its_own_prior_and_preserves_original_decisions(self):
        result, fits = self.evaluate()
        arrays = result['arrays']
        for budget in self.schedule[0]['budgets']:
            name = f"fold_{self.schedule[0]['fold']}/{budget}"
            head = fits[name]['heads']['baseline']
            cutoff = self.thresholds.copy()
            cutoff[2] = expit(np.log(cutoff[2] / (1 - cutoff[2])) + float(head[2]['offset']))
            actual = arrays[budget + '__baseline__original_cutoff']
            np.testing.assert_allclose(actual, np.broadcast_to(cutoff, actual.shape),
                                       atol=1e-15, rtol=1e-14)
            raw = arrays[budget + '__baseline__raw']
            pnew = arrays[budget + '__new__probability']
            np.testing.assert_array_equal(arrays[budget + '__baseline_original__decision'],
                                          raw >= self.thresholds)
            np.testing.assert_array_equal(arrays[budget + '__new_original__decision'], pnew >= cutoff)
            for key in ['logits', 'raw', 'probability', 'original_decision']:
                np.testing.assert_array_equal(arrays[budget + '__baseline__' + key][:, [0, 3]],
                                              arrays[budget + '__new__' + key][:, [0, 3]])
            if budget != 'full':
                self.assertNotIn(budget + '__baseline_tuned__decision', arrays)
                self.assertNotIn(budget + '__new_tuned__decision', arrays)
        for recipe in ['baseline', 'new']:
            original = arrays['full__' + recipe + '_original__decision']
            tuned = arrays['full__' + recipe + '_tuned__decision']
            np.testing.assert_array_equal(tuned[:, [0, 3]], original[:, [0, 3]])
            for j in self.config['focus']:
                choice = result['choices'][recipe][core.LABELS[j]]
                p = arrays['full__' + recipe + '__probability'][:, j]
                expected = original[:, j] if choice['kind'] == 'original' else p >= choice['threshold']
                np.testing.assert_array_equal(tuned[:, j], expected)

    def test_inner_cutoffs_use_each_fit_prior_and_both_heads_share_old_guards(self):
        result, fits = self.evaluate()
        inner = result['inner_arrays']
        indices = np.asarray(inner['indices'])
        np.testing.assert_array_equal(indices, self.schedule[0]['train'])
        for k, split in enumerate(self.schedule[0]['inner']):
            model = fits[f"fold_{self.schedule[0]['fold']}/inner_{k}"]
            np.testing.assert_array_equal(model['indices'], split['train'])
            rows = np.flatnonzero(np.isin(indices, split['test']))
            self.assertEqual(len(rows), len(split['test']))
            cutoff = self.thresholds.copy()
            offset = float(model['heads']['baseline'][2]['offset'])
            cutoff[2] = expit(np.log(cutoff[2] / (1 - cutoff[2])) + offset)
            for recipe in ['baseline', 'new']:
                actual = inner['inner__' + recipe + '__original_cutoff'][rows]
                np.testing.assert_allclose(actual, np.broadcast_to(cutoff, actual.shape),
                                           rtol=1e-14, atol=1e-15)
            np.testing.assert_array_equal(inner['inner__baseline__original_decision'][rows],
                                          inner['inner__baseline__raw'][rows] >= self.thresholds)
            np.testing.assert_array_equal(inner['inner__new__original_decision'][rows],
                                          inner['inner__new__probability'][rows] >= cutoff)
        for j in self.config['focus']:
            y = self.y[indices, j].astype(bool)
            d = inner['inner__baseline__original_decision'][:, j]
            expected = {'tp': int(np.sum(y & d)), 'fp': int(np.sum(~y & d)),
                        'fn': int(np.sum(y & ~d)), 'tn': int(np.sum(~y & ~d))}
            for recipe in ['baseline', 'new']:
                choice = result['choices'][recipe][core.LABELS[j]]
                for key, value in expected.items():
                    self.assertEqual(choice['reference_old_metrics'][key], value)
                self.assertFalse(choice['head_changed_by_policy'])

    def test_aggregate_rejects_duplicate_indices_inside_one_outer_fold(self):
        ix = np.arange(len(self.y))
        probability = np.full(self.y.shape, .4)
        arrays = {'indices': ix, 'full__baseline__probability': probability,
                  'full__baseline_original__decision': probability >= self.thresholds}
        summary, pooled = run.aggregate([{'arrays': arrays}], self.y, self.artists, self.cohorts)
        np.testing.assert_array_equal(pooled['targets'], self.y)
        duplicate = {k: np.concatenate([v, v[:1]]) for k, v in arrays.items()}
        with self.assertRaisesRegex(ValueError, 'duplicate evaluation indices'):
            run.aggregate([{'arrays': duplicate}], self.y, self.artists, self.cohorts)

    def test_failed_model_fit_is_recorded_and_cannot_be_automatically_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            provider = run.make_provider(out, self.x, self.y, self.artists,
                                         self.config, 'synthetic-freeze')
            indices = np.asarray(self.schedule[0]['train'])
            with patch.object(run, 'fit_recipes', side_effect=RuntimeError('Synthetic fit failed')) as fit:
                with self.assertRaisesRegex(RuntimeError, 'Synthetic fit failed'):
                    provider('fold_0/full', indices)
                fit.assert_called_once()
            status = json.loads((out / 'models/fold_0/full.status.json').read_text())
            self.assertEqual(status['state'], 'failed')
            self.assertEqual(status['indices'], indices.tolist())
            self.assertFalse((out / 'models/fold_0/full.npz').exists())
            with patch.object(run, 'fit_recipes', side_effect=AssertionError('Unexpected retry')) as fit:
                with self.assertRaisesRegex(ValueError, 'no automatic refit'):
                    provider('fold_0/full', indices)
                fit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
