"""Application v1.2 promotion and inference regressions; no fitting or downloads."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import application_predict as application
import application_release
import candidate_predict
import predict_app


ROOT = Path(__file__).resolve().parents[1]
WIDTH = candidate_predict.FEATURE_WIDTH


class Encoder:
    def __init__(self, feature=None):
        self.feature = np.zeros(WIDTH) if feature is None else feature
        self.calls = 0

    def extract(self, audio):
        self.calls += 1
        return self.feature


class ApplicationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.release_bytes = (ROOT / 'artifacts/application_release.json').read_bytes()
        self.release = json.loads(self.release_bytes)
        self.directory = self.root / self.release['candidate_directory']
        shutil.copytree(ROOT / self.release['candidate_directory'], self.directory)
        self.release_path = self.root / 'artifacts/application_release.json'
        self.release_path.write_bytes(self.release_bytes)

    def save_release(self):
        self.release_path.write_text(json.dumps(self.release))

    def test_portable_application_receipt_loads_exact_frozen_candidate(self):
        release, bundle = application.load_application(root=self.root)
        self.assertEqual(release, self.release)
        self.assertEqual(application_release.APP_VERSION, '1.2.0')
        self.assertEqual(bundle['candidate_id'], self.release['candidate_id'])
        self.assertEqual(bundle['weights_sha256'], self.release['model_weights_sha256'])
        self.assertEqual(bundle['status'], 'candidate_not_fresh_validated')
        self.assertEqual(predict_app.APP_VERSION, '1.1.0')
        expected = candidate_predict.load_candidate()
        for name in candidate_predict.ARRAY_SHAPES:
            np.testing.assert_array_equal(bundle[name], expected[name])
        self.assertEqual(bundle['thresholds'], expected['thresholds'])

    def test_invalid_release_identity_or_evidence_fails_before_audio(self):
        changes = [
            ('format_version', True), ('application_version', '1.1.0'),
            ('candidate_id', 'some-other-candidate'),
            ('candidate_directory', '../outside'),
            ('candidate_directory', str(self.directory)),
            ('candidate_metadata_sha256', '0' * 64),
            ('model_weights_sha256', '0' * 64),
        ]
        original = copy.deepcopy(self.release)
        for key, value in changes:
            with self.subTest(key=key, value=value):
                self.release = copy.deepcopy(original)
                self.release[key] = value
                self.save_release()
                encoder = Encoder()
                with self.assertRaises((OSError, ValueError)):
                    application.classify('synthetic.wav', encoder=encoder, root=self.root)
                self.assertEqual(encoder.calls, 0)
        for key, value in [('verdict', 'promising'), ('tracks', 371),
                           ('artists', True), ('summary_sha256', '0' * 64),
                           ('freeze_sha256', 'bad')]:
            with self.subTest(validation_key=key):
                self.release = copy.deepcopy(original)
                self.release['validation'][key] = value
                self.save_release()
                with self.assertRaises((OSError, ValueError)):
                    application.load_application(root=self.root)

    def test_missing_release_or_candidate_fails_closed(self):
        self.release_path.unlink()
        with self.assertRaises((OSError, ValueError)):
            application.load_application(root=self.root)
        self.release_path.write_bytes(self.release_bytes)
        (self.directory / 'candidate.npz').unlink()
        encoder = Encoder()
        with self.assertRaises((OSError, ValueError)):
            application.classify('synthetic.wav', encoder=encoder, root=self.root)
        self.assertEqual(encoder.calls, 0)

    def test_changed_model_metadata_or_receipts_fail_closed(self):
        for name in ['candidate.json', *candidate_predict.REFERENCES.values()]:
            path = self.directory / name
            original = path.read_bytes()
            with self.subTest(file=name):
                path.write_bytes(original + b'changed')
                with self.assertRaises((OSError, ValueError)):
                    application.load_application(root=self.root)
                path.write_bytes(original)

    def test_comparison_requires_unchanged_root_local_baseline_before_audio(self):
        encoder = Encoder()
        with self.assertRaises((OSError, ValueError)):
            application.classify('synthetic.wav', encoder=encoder,
                                 compare_baseline=True, root=self.root)
        self.assertEqual(encoder.calls, 0)
        names = ['artifacts/final_heads.json', 'artifacts/final_heads.npy',
                 'artifacts/score_correction.json', 'experiments/final_holdout_plan.json',
                 'app_version.txt']
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
        result = application.classify('synthetic.wav', encoder=Encoder(),
                                      compare_baseline=True, root=self.root)
        self.assertEqual(result['baseline']['application_version'], '1.1.0')
        for name in names:
            path = self.root / name
            original = path.read_bytes()
            with self.subTest(file=name):
                path.write_bytes(original + b'changed')
                encoder = Encoder()
                with self.assertRaises((OSError, ValueError)):
                    application.classify('synthetic.wav', encoder=encoder,
                                         compare_baseline=True, root=self.root)
                self.assertEqual(encoder.calls, 0)
                path.write_bytes(original)

    def test_model_symlink_cannot_escape_application_root(self):
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside) / 'candidate.npz'
            path = self.directory / 'candidate.npz'
            external.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(external)
            with self.assertRaises(ValueError):
                application.load_application(root=self.root)

    def test_self_consistent_replacement_model_is_not_silently_promoted(self):
        path = self.directory / 'candidate.npz'
        path.write_bytes(path.read_bytes() + b'changed')
        metadata_path = self.directory / 'candidate.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['weights_sha256'] = candidate_predict.sha256(path)
        metadata_path.write_text(json.dumps(metadata))
        self.release['candidate_metadata_sha256'] = candidate_predict.sha256(metadata_path)
        self.release['model_weights_sha256'] = candidate_predict.sha256(path)
        self.save_release()
        with self.assertRaises(ValueError):
            application.load_application(root=self.root)

    def test_default_inference_reports_current_validation_without_old_correction(self):
        encoder = Encoder(np.random.default_rng(27).normal(size=WIDTH))
        with patch.object(predict_app, 'load_correction', side_effect=AssertionError('old correction applied')):
            result = application.classify('synthetic.wav', encoder=encoder, root=self.root)
        self.assertEqual(encoder.calls, 1)
        self.assertEqual(result['application_version'], '1.2.0')
        self.assertEqual(result['score_mode'], 'validated_candidate')
        self.assertEqual(result['status'], 'validated_candidate')
        self.assertEqual(result['candidate_id'], self.release['candidate_id'])
        self.assertEqual(result['model_weights_sha256'], self.release['model_weights_sha256'])
        self.assertEqual(result['validation'], self.release['validation'])
        self.assertNotIn('not yet confirmed', result['warning'])
        self.assertNotIn('baseline', result)
        expected = candidate_predict.predict_vector(candidate_predict.load_candidate(), encoder.feature)
        self.assertEqual(result['scores'], expected)
        self.assertEqual(result['predicted_tags'], [r['label'] for r in expected if r['selected']])
        self.assertTrue(all(r['score'] == r['raw_score'] for r in result['scores']))

    def test_baseline_comparison_extracts_once_and_preserves_v11_arithmetic(self):
        feature = np.random.default_rng(91).normal(size=WIDTH)
        encoder = Encoder(feature)
        with patch.object(predict_app, 'predict_vector', wraps=predict_app.predict_vector) as baseline:
            result = application.classify('synthetic.wav', encoder=encoder, compare_baseline=True)
        self.assertEqual(encoder.calls, 1)
        self.assertIs(baseline.call_args.args[1], feature)
        old_bundle = predict_app.load_bundle()
        correction = predict_app.load_correction(old_bundle, predict_app.METADATA,
            predict_app.WEIGHTS, predict_app.PLAN, predict_app.CORRECTION)
        self.assertEqual(result['baseline']['scores'], predict_app.predict_vector(
            old_bundle, feature, correction=correction))
        self.assertEqual(result['baseline']['application_version'], '1.1.0')
        self.assertEqual(result['baseline']['score_mode'], 'weight_corrected')
        self.assertIs(result['baseline']['shared_feature'], True)
        self.assertEqual(result['scores'], candidate_predict.predict_vector(
            candidate_predict.load_candidate(), feature))

    def test_near_threshold_selection_uses_unrounded_new_probability(self):
        release, original = application.load_application()
        bundle = copy.deepcopy(original)
        bundle.update(means=np.zeros((4, WIDTH)), scales=np.ones((4, WIDTH)),
                      coefficients=np.zeros((4, WIDTH)), intercepts=np.zeros(4))
        for index in [1, 2]:
            threshold = bundle['thresholds']['f1'][index]
            decisions = []
            for probability in [threshold - 1e-8, threshold + 1e-8]:
                bundle['intercepts'][index] = np.log(probability) - np.log1p(-probability)
                with patch.object(application, 'load_application', return_value=(release, bundle)):
                    row = application.classify('boundary.wav', encoder=Encoder())['scores'][index]
                self.assertEqual(row['score'], row['threshold'])
                self.assertEqual(row['raw_score'], row['score'])
                decisions.append(row['selected'])
            self.assertEqual(decisions, [False, True])

    def test_selective_policy_remains_suppressed_and_comparable(self):
        result = application.classify('synthetic.wav', policy='precision_target',
            encoder=Encoder(np.random.default_rng(12).normal(size=WIDTH)), compare_baseline=True)
        self.assertEqual(result['policy'], 'precision_target')
        self.assertEqual([r['selected'] for r in result['scores']],
                         [r['selected'] for r in result['baseline']['scores']])
        self.assertEqual([r['selected'] for r in result['scores'][1:3]], [False, False])
        self.assertEqual([r['threshold'] for r in result['scores'][1:3]], [1.01, 1.01])

    def test_unknown_policy_or_bad_feature_is_rejected(self):
        encoder = Encoder()
        with self.assertRaises(ValueError):
            application.classify('synthetic.wav', policy='unknown', encoder=encoder)
        self.assertEqual(encoder.calls, 0)
        for feature in [np.zeros(WIDTH - 1), np.zeros((1, WIDTH)), np.full(WIDTH, np.nan)]:
            with self.subTest(shape=feature.shape), self.assertRaises(ValueError):
                application.classify('synthetic.wav', encoder=Encoder(feature))


if __name__ == '__main__':
    unittest.main()
