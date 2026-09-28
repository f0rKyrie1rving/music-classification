"""Independent synthetic checks of the nested development runner's boundaries.

These tests never load project feature values or produce real candidate scores.
"""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from research.application_auto_improvement import run as pipeline


class ApplicationAutoPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((pipeline.HERE / 'config.json').read_text())
        self.config.update(outer_folds=3, inner_folds=2)
        rng = np.random.default_rng(812)
        n = 48
        x = rng.normal(size=(n, 5))
        x[:, 0] = np.arange(n)  # An exact row marker for fit-boundary assertions.
        y = np.column_stack([(np.arange(n) + j) % 2 for j in range(4)])
        groups = np.asarray([f'artist_{i//2:02d}' for i in range(n)])
        rows = [{'track_id': f'track_{i:04d}', 'artist_id': groups[i],
                 'phase_role': 'train'} for i in range(n)]
        thresholds = np.asarray([.4, .27499999999999997, .37499999999999994,
                                 .27499999999999997])
        self.data = rows, x, y, groups, thresholds, {}
        self.splits = pipeline.splits_for(groups, y, self.config)
        self.freeze_hash = 'synthetic-frozen-inputs'

    def quiet_outer(self, out, data=None, verify_only=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.process_outer(out, 0, data or self.data,
                                          self.splits['outer'][0], self.config,
                                          self.freeze_hash, verify_only)

    def test_outer_targets_cannot_change_fits_selection_or_predictions(self):
        """Changing test answers must change only scoring, never a fitted choice."""
        definition = self.splits['outer'][0]
        train = np.asarray(definition['training_indices'])
        test = np.asarray(definition['evaluation_indices'])
        fit_rows = []
        selected_rows = []
        original_fit = pipeline.fit_head
        original_select = pipeline.select

        def observed_fit(x, y, **kwargs):
            fit_rows.append(x[:, 0].astype(int))
            return original_fit(x, y, **kwargs)

        def observed_select(oof, config, thresholds):
            selected_rows.append(oof['indices'].copy())
            np.testing.assert_array_equal(oof['y'], self.data[2][train])
            return original_select(oof, config, thresholds)

        changed = list(self.data)
        changed[2] = self.data[2].copy()
        changed[2][test] = 1 - changed[2][test]
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            with patch.object(pipeline, 'fit_head', side_effect=observed_fit), \
                    patch.object(pipeline, 'select', side_effect=observed_select):
                before, _ = self.quiet_outer(a)
                after, _ = self.quiet_outer(b, tuple(changed))
            np.testing.assert_array_equal(before['y'], 1 - after['y'])
            for key in before.keys() - {'y'}:
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)
            folder = Path('outer/fold_0')
            self.assertEqual(pipeline.read(a / folder / 'selection.json'),
                             pipeline.read(b / folder / 'selection.json'))
            for model in (a / folder).rglob('*.npz'):
                if 'heads' not in model.parent.name:
                    continue
                left = pipeline.load_npz(model)
                right = pipeline.load_npz(b / model.relative_to(a))
                for key in left:
                    np.testing.assert_array_equal(left[key], right[key])
        self.assertGreater(len(fit_rows), 10)
        for indices in fit_rows:
            self.assertTrue(set(indices) <= set(train))
            self.assertFalse(set(self.data[3][indices]) & set(self.data[3][test]))
        self.assertEqual(len(selected_rows), 2)
        for indices in selected_rows:
            np.testing.assert_array_equal(indices, train)

    def test_saved_heads_are_fit_only_and_inner_rows_are_never_fitted(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.quiet_outer(out)
            folder = out / 'outer/fold_0'
            oof = pipeline.load_npz(folder / 'inner_oof.npz')
            for receipt in (folder / 'inner').rglob('*.json'):
                identity = pipeline.read(receipt)['identity']
                fit = np.asarray(identity['fit_indices'])
                fold = int(receipt.parents[1].name.removeprefix('fold_'))
                scored = oof['indices'][oof['fold_by_row'] == fold]
                self.assertFalse(set(fit) & set(scored))
                self.assertFalse(set(self.data[3][fit]) & set(self.data[3][scored]))
                head = pipeline.load_npz(receipt.with_suffix('.npz'))
                np.testing.assert_allclose(head['mean'], self.data[1][fit].mean(axis=0),
                                           atol=1e-14, rtol=1e-14)
                label = list(pipeline.LABELS).index(identity['label'])
                positives = int(self.data[2][fit, label].sum())
                self.assertEqual(int(head['positive_fit']), positives)
                expected = np.log(positives / (len(fit) - positives)) if identity['balanced'] else 0.
                self.assertEqual(float(head['offset']), expected)

    def test_readonly_replay_does_not_fit_and_rejects_model_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            arrays, record = self.quiet_outer(out)
            with patch.object(pipeline, 'fit_head', side_effect=AssertionError('Unexpected refit')):
                replay, replay_record = self.quiet_outer(out, verify_only=True)
                self.assertEqual(record, replay_record)
                for key in arrays:
                    np.testing.assert_array_equal(arrays[key], replay[key])
                path = out / 'outer/fold_0/refit_heads/baseline_pop.npz'
                altered = pipeline.load_npz(path)
                altered['intercept'] = altered['intercept'] + .01
                np.savez_compressed(path, **altered)
                with self.assertRaisesRegex(ValueError, 'head hash changed'):
                    self.quiet_outer(out, verify_only=True)

    def test_failed_fit_is_retained_and_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            args = (folder, 'bad_fit', self.data[1], self.data[2], np.arange(24),
                    1, .001, False, self.config, self.freeze_hash)
            with patch.object(pipeline, 'fit_head', side_effect=RuntimeError('Synthetic fit failure')) as fit:
                with self.assertRaisesRegex(RuntimeError, 'Synthetic fit failure'):
                    pipeline.fit_cached(*args)
                self.assertEqual(fit.call_count, 1)
            self.assertEqual(pipeline.read(folder / 'bad_fit.json')['state'], 'failed')
            self.assertFalse((folder / 'bad_fit.npz').exists())
            with patch.object(pipeline, 'fit_head', side_effect=AssertionError('Unexpected retry')):
                with self.assertRaisesRegex(ValueError, 'no automatic refit'):
                    pipeline.fit_cached(*args)

    def test_cached_model_identity_cannot_change_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            args = [folder, 'model', self.data[1], self.data[2], np.arange(24),
                    2, .001, True, self.config, self.freeze_hash]
            pipeline.fit_cached(*args)
            for index, value in [(4, np.arange(24)[::-1]), (6, .0003), (9, 'other-freeze')]:
                changed = list(args)
                changed[index] = value
                with self.subTest(changed_argument=index), \
                        patch.object(pipeline, 'fit_head', side_effect=AssertionError('Unexpected refit')):
                    with self.assertRaisesRegex(ValueError, 'fit identity changed'):
                        pipeline.fit_cached(*changed)

    def test_folds_ignore_labels_and_validate_all_artist_boundaries(self):
        alternative_y = 1 - self.data[2]
        self.assertEqual(self.splits, pipeline.splits_for(self.data[3], alternative_y, self.config))
        for definition in self.splits['outer']:
            train, test = map(np.asarray, (definition['training_indices'], definition['evaluation_indices']))
            self.assertEqual(set(train) | set(test), set(range(len(self.data[0]))))
            self.assertFalse(set(train) & set(test))
            self.assertFalse(set(self.data[3][train]) & set(self.data[3][test]))
            inner = np.asarray(definition['inner_fold_by_training_row'])
            for fold in range(self.config['inner_folds']):
                self.assertFalse(set(self.data[3][train[inner != fold]]) &
                                 set(self.data[3][train[inner == fold]]))
        with self.assertRaisesRegex(ValueError, 'Single-class'):
            pipeline.splits_for(self.data[3], np.zeros_like(self.data[2]), self.config)

    def synthetic_outer_arrays(self):
        rows, x, y, groups, thresholds, plan = self.data
        logits = np.column_stack([np.sin(x[:, 0] + j) for j in range(4)])
        probability = expit(logits)
        outputs = []
        for definition in self.splits['outer']:
            ix = np.asarray(definition['evaluation_indices'])
            values = {'indices': ix, 'ids': np.asarray([rows[i]['track_id'] for i in ix]),
                      'artists': groups[ix], 'y': y[ix]}
            for method in ['baseline', 'candidate']:
                values.update({method + '_logits': logits[ix], method + '_raw': probability[ix],
                               method + '_probability': probability[ix],
                               method + '_decision': probability[ix] >= thresholds})
            outputs.append(values)
        return outputs

    def test_aggregate_counts_rows_once_and_matches_independent_losses(self):
        outputs = self.synthetic_outer_arrays()
        arrays, summary = pipeline.aggregate(outputs, self.data, self.config)
        targets, p, decision = arrays['y'], arrays['baseline_probability'], arrays['baseline_decision']
        tp = int(np.sum(decision & (targets == 1)))
        fp = int(np.sum(decision & (targets == 0)))
        fn = int(np.sum(~decision & (targets == 1)))
        actual = summary['metrics']['baseline']
        self.assertEqual([actual[k] for k in ['tp', 'fp', 'fn']], [tp, fp, fn])
        self.assertAlmostEqual(actual['macro_brier'], np.square(p - targets).mean())
        self.assertAlmostEqual(actual['macro_log_loss'],
                               -np.where(targets == 1, np.log(p), np.log1p(-p)).mean())
        self.assertAlmostEqual(actual['micro_f1'], 2 * tp / (2 * tp + fp + fn))
        self.assertFalse(summary['gate']['passed'])  # Identical calls have no FP reduction.
        duplicate = deepcopy(outputs)
        duplicate[0] = {k: np.concatenate([v, v[:1]]) for k, v in duplicate[0].items()}
        with self.assertRaisesRegex(ValueError, 'repeated outer row index|missing/duplicated'):
            pipeline.aggregate(duplicate, self.data, self.config)
        with self.assertRaisesRegex(ValueError, 'missing/duplicated'):
            pipeline.aggregate(outputs[:-1], self.data, self.config)

    def test_each_gate_condition_is_binding_and_zero_fp_is_not_a_gain(self):
        base = {'micro_recall': .8, 'micro_f1': .7, 'macro_brier': .15, 'macro_log_loss': .45,
                'per_label': {name: {'fp': 10, 'recall': .8} for name in pipeline.LABELS}}
        candidate = deepcopy(base)
        candidate['per_label']['pop']['fp'] = 8
        candidate['macro_brier'] = .14
        candidate['macro_log_loss'] = .44
        self.assertTrue(pipeline.gate(base, candidate, True, self.config)['passed'])
        cases = [('focus_fp_relative_reduction', ['per_label', 'pop', 'fp'], 9),
                 ('each_focus_fp_nonincrease', ['per_label', 'ambient', 'fp'], 11),
                 ('micro_recall', ['micro_recall'], .769),
                 ('each_label_recall', ['per_label', 'pop', 'recall'], .749),
                 ('micro_f1', ['micro_f1'], .699),
                 ('macro_brier', ['macro_brier'], .151),
                 ('macro_log_loss', ['macro_log_loss'], .451)]
        for name, path, value in cases:
            changed = deepcopy(candidate)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(gate=name):
                assessment = pipeline.gate(base, changed, True, self.config)
                self.assertFalse(assessment['passed'])
                self.assertIn(name, assessment['failed_checks'])
        self.assertFalse(pipeline.gate(base, candidate, False, self.config)['passed'])
        zero = deepcopy(base)
        for label in pipeline.LABELS:
            zero['per_label'][label]['fp'] = 0
        self.assertFalse(pipeline.gate(zero, zero, True, self.config)['passed'])

    def test_failed_gate_prevents_final_fit_and_completed_run_resumes_without_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            pipeline.save(out / 'config.json', self.config)
            pipeline.save(out / 'splits.json', self.splits)
            pipeline.save(out / 'freeze.json', {'synthetic': True})
            outputs = self.synthetic_outer_arrays()
            with patch.object(pipeline, 'verify_frozen', return_value={}), \
                    patch.object(pipeline, 'inputs', return_value=self.data), \
                    patch.object(pipeline, 'process_outer', side_effect=[(a, {}) for a in outputs]) as process, \
                    patch.object(pipeline, 'final_candidate', side_effect=AssertionError('Gate failed')) as final, \
                    contextlib.redirect_stdout(io.StringIO()):
                pipeline.run(out)
                self.assertEqual(process.call_count, self.config['outer_folds'])
                final.assert_not_called()
            status = pipeline.read(out / 'status.json')
            self.assertEqual(status['state'], 'complete')
            self.assertFalse(status['gate_passed'])
            self.assertFalse(status['final_candidate_created'])
            self.assertFalse((out / 'final_candidate').exists())
            self.assertFalse((out / 'run.lock').exists())
            with patch.object(pipeline, 'verify_frozen', return_value={}), \
                    patch.object(pipeline, 'verify', return_value={}) as replay, \
                    patch.object(pipeline, 'process_outer', side_effect=AssertionError('Unexpected repeat')), \
                    contextlib.redirect_stdout(io.StringIO()):
                pipeline.run(out)
                replay.assert_called_once_with(out)
            with self.assertRaisesRegex(ValueError, 'Gate failed'):
                pipeline.final_candidate(out, self.data, self.splits, self.config,
                                         self.freeze_hash, {'gate': {'passed': False}})

    def test_freeze_rejects_changed_source_input_snapshot_and_local_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'out'
            (out / 'source_snapshot').mkdir(parents=True)
            paths = [root / 'source.py', root / 'input.json', out / 'splits.json',
                     out / 'source_snapshot/source.py']
            for path in paths:
                path.write_text('immutable synthetic fixture\n')
            frozen = {'runtime': {'synthetic': True}, 'numerical_thread_limit': 1, 'policy': 'f1',
                      'source_hashes': {'source.py': pipeline.digest(paths[0])},
                      'input_hashes': {'input.json': pipeline.digest(paths[1])},
                      'local_hashes': {'splits.json': pipeline.digest(paths[2])}}
            pipeline.save(out / 'freeze.json', frozen)
            with patch.object(pipeline, 'ROOT', root), \
                    patch.object(pipeline, 'runtime', return_value={'synthetic': True}), \
                    patch.object(pipeline, 'source_paths', return_value=[paths[0]]):
                self.assertEqual(pipeline.verify_frozen(out), frozen)
                for path in paths:
                    with self.subTest(changed_path=str(path.relative_to(root))):
                        original = path.read_bytes()
                        path.write_text('tampered fixture\n')
                        with self.assertRaisesRegex(ValueError, 'changed'):
                            pipeline.verify_frozen(out)
                        path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
