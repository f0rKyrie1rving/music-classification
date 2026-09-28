"""Application score correction keeps the released classifier's tag decisions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import predict_app
import predict_maest
import score_correction


ROOT = Path(__file__).resolve().parents[1]
WIDTH = predict_maest.FEATURE_WIDTH


class ScoreCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.bundle = predict_maest.load_bundle()
        self.record = json.loads(score_correction.CORRECTION.read_text(encoding='utf-8'))
        self.offsets = np.array([0., 0., np.log(244/962), 0.])

    def validate(self, record, bundle=None):
        return score_correction.validate_correction(
            record, self.bundle if bundle is None else bundle,
            predict_maest.METADATA, predict_maest.WEIGHTS, predict_maest.PLAN)

    def synthetic(self, logits):
        bundle = copy.deepcopy(self.bundle)
        bundle.update(means=np.zeros((4, WIDTH)), scales=np.ones((4, WIDTH)),
                      coefficients=np.zeros((4, WIDTH)), intercepts=np.asarray(logits, float))
        return bundle

    def compare_old(self, bundle, feature, policy, mode='raw'):
        old = predict_maest.predict_vector(bundle, feature, policy)
        new = predict_app.predict_vector(bundle, feature, policy, mode, self.record)
        for before, after in zip(old, new):
            self.assertEqual(before['selected'], after['selected'])
            self.assertEqual(before['score'], after['raw_score'])
            self.assertEqual(before['threshold'], after['raw_threshold'])
            if mode == 'raw':
                self.assertEqual(before, {key: after[key] for key in before})
        return old, new

    def test_packaged_recipe_is_bound_to_actual_training_counts(self):
        offsets = self.validate(self.record)
        np.testing.assert_array_equal(offsets, self.offsets)
        plan = json.loads(predict_maest.PLAN.read_text(encoding='utf-8'))
        tracks = json.loads((ROOT/'data/expanded_manifest.json').read_text(encoding='utf-8'))['tracks']
        by_id = {r['track_id']: r for r in tracks}
        positives = [sum(bool(set(by_id[i]['tags']) & set(plan['ontology'][label]))
                         for i in plan['fit_ids']) for label in predict_maest.LABELS]
        self.assertEqual(self.record['fit_count'], len(plan['fit_ids']))
        self.assertEqual(self.record['fit_positive_counts'], positives)
        self.assertEqual(positives, [435, 261, 244, 220])
        loaded = score_correction.load_correction(self.bundle, predict_maest.METADATA,
                                                  predict_maest.WEIGHTS, predict_maest.PLAN)
        self.assertEqual(loaded, self.record)

    def test_rejects_invalid_counts_formula_and_training_identity(self):
        mutations = [('fit_count', True), ('fit_count', 1205),
                     ('fit_positive_counts', [435, 261, 0, 220]),
                     ('fit_positive_counts', [435, 261, 1206, 220]),
                     ('fit_positive_counts', [435, 261, True, 220]),
                     ('fit_positive_counts', [435, 261, 245, 220]),
                     ('fit_ids_sha256', '0'*64), ('fit_manifest_sha256', '0'*64),
                     ('class_weight', [None]*4), ('logit_offsets', [0., 0., 0., 0.]),
                     ('logit_offsets', [0., 0., float('nan'), 0.]),
                     ('logit_offsets', [0., 0., float('inf'), 0.]),
                     ('labels', list(reversed(predict_maest.LABELS)))]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(self.record); changed[key] = value
                with self.assertRaises(ValueError):
                    self.validate(changed)

    def test_rejects_each_source_checksum_mismatch(self):
        for key in ['source_metadata_sha256', 'source_weights_sha256', 'source_plan_sha256']:
            changed = copy.deepcopy(self.record); changed[key] = '0'*64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(changed)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory)/'weights.npy'
            changed.write_bytes(predict_maest.WEIGHTS.read_bytes()+b'changed')
            with self.assertRaises(ValueError):
                score_correction.validate_correction(self.record, self.bundle,
                                                     predict_maest.METADATA, changed, predict_maest.PLAN)

    def test_artifact_pin_rejects_self_consistent_count_replacement(self):
        changed = copy.deepcopy(self.record)
        changed['fit_positive_counts'][2] = 245
        changed['logit_offsets'][2] = float(np.log(245/(1206-245)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'score_correction.json'
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                score_correction.load_correction(self.bundle, predict_maest.METADATA,
                                                 predict_maest.WEIGHTS, predict_maest.PLAN, path)

    def test_threshold_endpoints_disabled_sentinel_and_zero_offsets_exact(self):
        values = np.array([0., 1., 1.01, .5])
        result = score_correction.corrected_thresholds(values, [-3., 3., -100., -1.])
        np.testing.assert_array_equal(result[:3], values[:3])
        self.assertAlmostEqual(result[3], 1/(1+np.e))
        unchanged = np.array([.4, .275, .375, .275])
        np.testing.assert_array_equal(score_correction.corrected_thresholds(unchanged, np.zeros(4)), unchanged)
        corrected = score_correction.corrected_thresholds(unchanged, self.offsets)
        np.testing.assert_array_equal(corrected[[0, 1, 3]], unchanged[[0, 1, 3]])
        self.assertAlmostEqual(corrected[2], .13208228076506673)

    def test_invalid_threshold_or_offset_vectors_rejected(self):
        for thresholds, offsets in [([.1]*3, [0.]*4), ([-.1]*4, [0.]*4),
                                     ([np.nan]*4, [0.]*4), ([.2]*4, [np.inf]*4)]:
            with self.subTest(thresholds=thresholds, offsets=offsets), self.assertRaises(ValueError):
                score_correction.corrected_thresholds(thresholds, offsets)

    def test_raw_mode_matches_original_outputs_for_both_policies(self):
        features = np.random.default_rng(728).normal(size=(15, WIDTH))
        for feature in features:
            for policy in predict_maest.POLICIES:
                self.compare_old(self.bundle, feature, policy)

    def test_corrected_mode_retains_original_tags_for_both_policies(self):
        features = np.random.default_rng(728).normal(size=(15, WIDTH))
        for feature in features:
            for policy in predict_maest.POLICIES:
                old, new = self.compare_old(self.bundle, feature, policy, 'weight_corrected')
                for j in [0, 1, 3]:
                    self.assertEqual(old[j]['score'], new[j]['score'])
                    self.assertEqual(old[j]['threshold'], new[j]['threshold'])
                self.assertLessEqual(new[2]['score'], old[2]['score'])
                if policy == 'precision_target':
                    self.assertEqual(new[2]['threshold'], 1.01)
                    self.assertFalse(new[2]['selected'])

    def test_decisions_use_unrounded_raw_probability_at_threshold(self):
        threshold = self.bundle['thresholds']['f1'][2]
        selected = []
        for probability in [threshold-1e-8, threshold+1e-8]:
            z = np.log(probability)-np.log1p(-probability)
            bundle = self.synthetic([0., 0., z, 0.])
            before, after = self.compare_old(bundle, np.zeros(WIDTH), 'f1', 'weight_corrected')
            self.assertEqual(after[2]['raw_score'], after[2]['raw_threshold'])
            self.assertEqual(after[2]['score'], after[2]['threshold'])
            selected.append(after[2]['selected'])
        self.assertEqual(selected, [False, True])

    def test_extreme_finite_logits_remain_finite_and_multilabel(self):
        for value in [-1e300, 1e300]:
            bundle = self.synthetic([value]*4)
            for policy in predict_maest.POLICIES:
                old, result = self.compare_old(bundle, np.zeros(WIDTH), policy, 'weight_corrected')
                for row in result:
                    self.assertTrue(np.isfinite(row['score']))
                    self.assertGreaterEqual(row['score'], 0.)
                    self.assertLessEqual(row['score'], 1.)
                if value > 0:
                    self.assertEqual(sum(r['score'] for r in result), 4.)
                    if policy == 'precision_target':
                        self.assertEqual([r['selected'] for r in result], [True, False, False, True])

    def test_raw_classify_is_available_without_correction_file(self):
        class Encoder:
            def extract(self, audio):
                return np.zeros(WIDTH)
        result = predict_app.classify('synthetic.wav', encoder=Encoder(), score_mode='raw',
                                     correction_path=ROOT/'does-not-exist.json')
        self.assertEqual(result['score_mode'], 'raw')
        original = predict_maest.predict_vector(self.bundle, np.zeros(WIDTH))
        for before, after in zip(original, result['scores']):
            self.assertEqual(before, {key: after[key] for key in before})
        with self.assertRaises(FileNotFoundError):
            predict_app.classify('synthetic.wav', encoder=Encoder(),
                                 correction_path=ROOT/'does-not-exist.json')

    def test_corrected_mode_rejects_missing_or_nonfinite_offsets(self):
        for correction in [None, {'logit_offsets': [0., 0., np.nan, 0.]},
                           {'logit_offsets': [0., 0.]}]:
            with self.subTest(correction=correction), self.assertRaises(ValueError):
                predict_app.predict_vector(self.bundle, np.zeros(WIDTH), correction=correction)
        with self.assertRaises(ValueError):
            predict_app.predict_vector(self.bundle, np.zeros(WIDTH), score_mode='unknown')


if __name__ == '__main__':
    unittest.main()
