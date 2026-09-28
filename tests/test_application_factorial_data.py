"""Synthetic cache, retirement, and artist/cohort boundary tests; no real fitting."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research.application_factorial.data import (
    LABELS, broad_targets, cohort_artist_folds, combine_cohorts, digest,
    safe_path, save_retirement_ledger, validate_cache,
)

ONTOLOGY = {label: [label] for label in LABELS}
ONTOLOGY['pop'].append('electropop')
ONTOLOGY['electronic'].append('electropop')


def row(index, artist=None, tags=None, role='train'):
    return {'track_id': f'track_{index:07d}', 'artist_id': f'artist_{artist or index:06d}',
            'tags': ['pop'] if tags is None else tags, 'phase_role': role, 'targets': [0, 1, 0, 0]}


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'features.npy'
        self.rows = [row(1), row(2)]
        self.values = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.save(self.path, self.values, allow_pickle=False)
        self.meta = {'ids': [r['track_id'] for r in self.rows], 'shape': [2, 3], 'dtype': 'float32',
                     'feature_sha256': digest(self.path), 'audio_hashes': ['a' * 64, 'b' * 64]}

    def test_exact_order_bytes_and_lossless_float64_conversion(self):
        loaded = validate_cache(self.path, self.meta, self.rows, width=3)
        self.assertEqual(loaded.dtype, np.float64)
        np.testing.assert_array_equal(loaded, self.values)
        for rows, meta in [(self.rows[::-1], self.meta),
                           (self.rows, {**self.meta, 'feature_sha256': '0' * 64}),
                           (self.rows, {**self.meta, 'audio_hashes': ['a' * 64]}),
                           (self.rows, {**self.meta, 'shape': [3, 2]})]:
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                validate_cache(self.path, meta, rows, width=3)

    def test_corrupt_values_dtype_and_width_rejected_even_with_updated_hash(self):
        for value in [self.values.astype(np.float64), self.values[:, :2],
                      np.asarray([[np.nan, 1, 2], [3, 4, 5]], dtype=np.float32)]:
            np.save(self.path, value, allow_pickle=False)
            with self.subTest(dtype=value.dtype, shape=value.shape), self.assertRaises(ValueError):
                validate_cache(self.path, {**self.meta, 'feature_sha256': digest(self.path)}, self.rows, width=3)

    def test_duplicate_ids_and_unsafe_path_rejected(self):
        with self.assertRaises(ValueError):
            validate_cache(self.path, self.meta, [self.rows[0], self.rows[0]], width=3)
        for name in ['../outside', '/tmp/outside']:
            with self.assertRaises(ValueError):
                safe_path(self.root, name)
        (self.root / 'escape').symlink_to(self.root.parent)
        with self.assertRaises(ValueError):
            safe_path(self.root, 'escape/not-in-root')


class CohortTests(unittest.TestCase):
    def test_same_broad_ontology_and_original_fields_preserved_by_copy(self):
        source = [row(1, tags=['electropop']), row(2, role='new_holdout')]
        original = deepcopy(source)
        parts = [('a', source[:1], np.ones((1, 3))), ('b', source[1:], np.zeros((1, 3)))]
        rows, x, y, artists, cohorts = combine_cohorts(parts, ONTOLOGY, expected_counts={'a': (1, 1), 'b': (1, 1)})
        self.assertEqual(source, original)
        self.assertEqual(rows[0]['targets'], [0, 1, 0, 0])  # Retain original narrow target.
        self.assertEqual(rows[0]['proxy_labels'], [1, 1, 0, 0])
        self.assertEqual(rows[1]['phase_role'], 'new_holdout')
        self.assertEqual(rows[1]['original_role'], 'new_holdout')
        self.assertTrue(all(r['current_role'] == 'development_only' for r in rows))
        self.assertEqual(cohorts.tolist(), ['a', 'b'])
        self.assertEqual(artists.tolist(), ['artist_000001', 'artist_000002'])
        np.testing.assert_array_equal(x, [[1, 1, 1], [0, 0, 0]])
        np.testing.assert_array_equal(y, [[1, 1, 0, 0], [0, 1, 0, 0]])
        rows[0]['tags'].append('rock')
        self.assertEqual(source, original)

    def test_track_artist_overlap_wrong_order_and_counts_rejected(self):
        valid = [('a', [row(1)], np.ones((1, 2))), ('b', [row(2)], np.zeros((1, 2)))]
        counts = {'a': (1, 1), 'b': (1, 1)}
        cases = [valid[::-1],
                 [valid[0], ('b', [row(1)], np.zeros((1, 2)))],
                 [valid[0], ('b', [row(2, artist=1)], np.zeros((1, 2)))],
                 [valid[0], ('b', [row(2)], np.zeros((1, 3)))]]
        for parts in cases:
            with self.subTest(parts=parts), self.assertRaises(ValueError):
                combine_cohorts(parts, ONTOLOGY, expected_counts=counts)
        with self.assertRaises(ValueError):
            combine_cohorts(valid, ONTOLOGY, expected_counts={'a': (2, 1), 'b': (1, 1)})
        with self.assertRaises(ValueError):
            broad_targets([row(1, tags=[1])], ONTOLOGY)

    def test_cohort_folds_disjoint_complete_balanced_and_permutation_invariant(self):
        artists, cohorts = [], []
        for cohort in ['a', 'b', 'c', 'd']:
            for index, size in enumerate([5, 4, 4, 3, 3, 2, 2, 1, 1, 1]):
                artists.extend([f'{cohort}-{index}'] * size); cohorts.extend([cohort] * size)
        artists, cohorts = np.asarray(artists), np.asarray(cohorts)
        folds = cohort_artist_folds(artists, cohorts, 5, 'fixed-seed')
        self.assertEqual(folds.dtype, np.int64)
        self.assertEqual(set(folds), set(range(5)))
        for artist in set(artists):
            self.assertEqual(len(set(folds[artists == artist])), 1)
        for cohort in set(cohorts):
            counts = np.bincount(folds[cohorts == cohort], minlength=5)
            self.assertTrue((counts > 0).all())
            self.assertLessEqual(int(counts.max() - counts.min()), 1)
        permutation = np.random.default_rng(31).permutation(len(artists))
        permuted = cohort_artist_folds(artists[permutation], cohorts[permutation], 5, 'fixed-seed')
        np.testing.assert_array_equal(permuted, folds[permutation])
        np.testing.assert_array_equal(cohort_artist_folds(artists, cohorts, 5, 'fixed-seed'), folds)
        self.assertFalse(np.array_equal(cohort_artist_folds(artists, cohorts, 5, 'another-seed'), folds))

    def test_invalid_fold_boundaries_fail_without_redraw(self):
        for artists, cohorts, k in [(['x', 'x'], ['a', 'b'], 2),
                                    (['x', 'y', 'z'], ['a', 'b', 'b'], 2),
                                    (['x', 'y'], ['a'], 2),
                                    (['x', 'y'], ['a', 'a'], True),
                                    (['x', 'y'], ['a', 'a'], 1),
                                    ([1, 2], ['a', 'a'], 2),
                                    (['x', ''], ['a', 'a'], 2)]:
            with self.subTest(artists=artists, cohorts=cohorts, k=k), self.assertRaises(ValueError):
                cohort_artist_folds(artists, cohorts, k, 's')

    def test_nested_training_folds_keep_outer_evaluation_artists_absent(self):
        artists = np.asarray([f'a{i}' for i in range(20)] + [f'b{i}' for i in range(20)])
        cohorts = np.asarray(['a'] * 20 + ['b'] * 20)
        outer = cohort_artist_folds(artists, cohorts, 5, 'outer')
        for fold in range(5):
            train, test = outer != fold, outer == fold
            inner = cohort_artist_folds(artists[train], cohorts[train], 4, f'inner-{fold}')
            self.assertFalse(set(artists[train]) & set(artists[test]))
            for k in range(4):
                self.assertFalse(set(artists[train][inner != k]) & set(artists[train][inner == k]))
                self.assertEqual(set(cohorts[train][inner == k]), {'a', 'b'})


class LedgerTests(unittest.TestCase):
    def test_retirement_preserves_full_boundary_original_role_and_is_immutable(self):
        original = row(1, role='new_holdout')
        rows = [{**deepcopy(original), 'cohort': 'hist239', 'original_role': 'new_holdout',
                 'current_role': 'development_only', 'proxy_labels': [0, 1, 0, 0]}]
        audit = {'status': 'passed', 'tracks': 1, 'artists': 1, 'labels': list(LABELS),
                 'input_hashes': {'original.json': 'a' * 64}, 'limits': ['Development proxy labels only.'],
                 'future_exclusions': {'tracks': ['track_0000001', 'track_0000002'],
                                       'artists': ['artist_000001', 'artist_000002']}}
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            result = save_retirement_ledger(out, rows, audit)
            before = {name: (out / name).stat().st_mtime_ns for name in result}
            self.assertEqual(result, save_retirement_ledger(out, rows, audit))
            self.assertEqual(before, {name: (out / name).stat().st_mtime_ns for name in result})
            ledger = json.loads((out / 'role_ledger.json').read_text())
            self.assertEqual(ledger['records'][0]['original_phase_role'], 'new_holdout')
            self.assertFalse(ledger['records'][0]['future_fresh_eligible'])
            excluded = json.loads((out / 'future_exclusions.json').read_text())
            self.assertEqual(excluded['tracks'], audit['future_exclusions']['tracks'])
            self.assertEqual(excluded['artists'], audit['future_exclusions']['artists'])
            changed = deepcopy(rows); changed[0]['proxy_labels'][1] = 0
            with self.assertRaises(ValueError):
                save_retirement_ledger(out, changed, audit)
            self.assertEqual(result, {name: digest(out / name) for name in result})
        self.assertEqual(original['phase_role'], 'new_holdout')


if __name__ == '__main__':
    unittest.main()
