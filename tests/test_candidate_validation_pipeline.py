"""Synthetic confirmation integrity tests; never acquire or score real songs."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research.application_candidate_validation import pipeline as p


class ConfirmationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root/'run'
        self.out.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_freeze_accepts_only_unobserved_stage(self):
        p.assert_unobserved(self.out)
        for name in ['archive_index_12.json', 'resolved_manifest.json', 'download_status.json',
                     'features.npy', 'features.json', 'manifest.json', 'predictions.npz',
                     'evaluation_status.json']:
            with self.subTest(name=name):
                path = self.out/name
                path.write_bytes(b'pre-freeze evidence')
                with self.assertRaises(ValueError):
                    p.assert_unobserved(self.out)
                path.unlink()

    def test_freeze_rejects_preexisting_acquisition_or_network_evidence(self):
        for name in ['audio', 'acquisition', 'feature_tracks', 'network_requests', 'index_headers']:
            with self.subTest(name=name):
                path = self.out/name
                path.mkdir()
                (path/'receipt').write_text('already observed')
                with self.assertRaises(ValueError):
                    p.assert_unobserved(self.out)
                (path/'receipt').unlink()
                path.rmdir()

    def test_untrusted_receipt_paths_cannot_escape_output(self):
        (self.out/'link').symlink_to(self.root, target_is_directory=True)
        for name in ['../outside', '/absolute', 'link/outside']:
            with self.assertRaises(ValueError):
                p.safe_file(self.out, name)

    def test_completed_json_and_vector_cannot_be_overwritten(self):
        p.immutable_json(self.out/'receipt.json', {'value': 'fixed'})
        p.immutable_json(self.out/'receipt.json', {'value': 'fixed'})
        with self.assertRaises(ValueError):
            p.immutable_json(self.out/'receipt.json', {'value': 'changed'})
        vector = np.arange(2304, dtype=np.float32)
        p.save_vector(self.out/'vector.npy', vector)
        with self.assertRaises(ValueError):
            p.save_vector(self.out/'vector.npy', vector+1)

    def test_acquisition_requires_exact_once_partition_and_no_replacements(self):
        rows = [{'track_id': 'track_0000001', 'artist_id': 'artist_1'},
                {'track_id': 'track_0000002', 'artist_id': 'artist_2'}]
        good = {'requested': 2, 'replacement_tracks': 0,
                'successful': [{'track_id': 'track_0000001'}],
                'failures': [{'track_id': 'track_0000002', 'error': 'fixed failure'}]}
        p.atomic_json(self.out/'download_status.json', good)
        _, success, failed = p.acquisition_rows(self.out, rows)
        self.assertEqual(set(success), {'track_0000001'})
        self.assertEqual(set(failed), {'track_0000002'})
        bad = []
        item = deepcopy(good); item['successful'] *= 2; bad.append(item)
        item = deepcopy(good); item['failures'] = []; bad.append(item)
        item = deepcopy(good); item['failures'][0]['track_id'] = 'track_0000001'; bad.append(item)
        item = deepcopy(good); item['replacement_tracks'] = 1; bad.append(item)
        for item in bad:
            p.atomic_json(self.out/'download_status.json', item)
            with self.assertRaises(ValueError):
                p.acquisition_rows(self.out, rows)

    def test_checkpoint_failure_is_preserved_and_changed_input_rejected(self):
        identity = {'track_id': 'track_0000001', 'audio_sha256': 'audio',
                    'freeze_sha256': 'freeze', 'download_status_sha256': 'downloads'}
        receipt = {**identity, 'status': 'failed', 'error': 'incomplete audio'}
        path = self.out/'track_0000001.json'
        p.atomic_json(path, receipt)
        actual, vector = p.verify_track_checkpoint(path, identity)
        self.assertEqual(actual, receipt)
        self.assertIsNone(vector)
        for key in identity:
            with self.assertRaises(ValueError):
                p.verify_track_checkpoint(path, {**identity, key: 'changed'})

    def test_checkpoint_detects_feature_file_mutation(self):
        identity = {'track_id': 'track_0000001', 'audio_sha256': 'audio',
                    'freeze_sha256': 'freeze', 'download_status_sha256': 'downloads'}
        path = self.out/'track_0000001.json'
        p.save_vector(path.with_suffix('.npy'), np.arange(2304, dtype=np.float32))
        p.atomic_json(path, {**identity, 'status': 'complete',
                            'feature_sha256': p.digest(path.with_suffix('.npy'))})
        _, vector = p.verify_track_checkpoint(path, identity)
        np.testing.assert_array_equal(vector, np.arange(2304, dtype=np.float32))
        with path.with_suffix('.npy').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaises(ValueError):
            p.verify_track_checkpoint(path, identity)

    def test_hash_verifier_binds_live_sources_inputs_local_and_snapshot(self):
        source = self.root/'predictor.py'; source.write_text('fixed source')
        weight = self.root/'weights.bin'; weight.write_bytes(b'fixed weights')
        local = self.out/'config.json'; local.write_text('fixed config')
        snapshot = self.out/'source_snapshot/predictor.py'
        snapshot.parent.mkdir(); snapshot.write_bytes(source.read_bytes())
        frozen = {'source_hashes': {'predictor.py': p.digest(source)},
                  'input_hashes': {'weights.bin': p.digest(weight)},
                  'local_hashes': {'config.json': p.digest(local)}}
        with patch.object(p, 'ROOT', self.root), patch.object(p, 'source_paths', return_value=[source]):
            p.verify_hashes(self.out, frozen)
            for path in [source, weight, local, snapshot]:
                previous = path.read_bytes(); path.write_bytes(previous+b'changed')
                with self.assertRaises(ValueError):
                    p.verify_hashes(self.out, frozen)
                path.write_bytes(previous)
        with patch.object(p, 'ROOT', self.root), patch.object(p, 'source_paths', return_value=[]):
            with self.assertRaises(ValueError):
                p.verify_hashes(self.out, frozen)

    def resolved_fixture(self):
        original = [{'track_id': 'track_0000001', 'artist_id': 'artist_1', 'tags': ['pop'],
                     'path': '12/1.mp3', 'archive_member': '12/1.low.mp3'},
                    {'track_id': 'track_0000002', 'artist_id': 'artist_2', 'tags': ['ambient'],
                     'path': '12/2.mp3', 'archive_member': '12/2.low.mp3'}]
        p.atomic_json(self.out/'selected_manifest.json', {'tracks': original})
        p.atomic_json(self.out/'freeze.json', {'fixed': True})
        p.atomic_json(self.out/'config.json', {'archives': ['12']})
        index = {'complete': True, 'archive_url': 'https://example.invalid/archive',
                 'archive_bytes': 8192, 'members': {'12/1.low.mp3': {'offset': 512, 'size': 200},
                                                  '12/2.low.mp3': {'offset': 1536, 'size': 300}}}
        p.atomic_json(self.out/'archive_index_12.json', index)
        resolved = {'tracks': [{**row, 'archive': {**index['members'][row['archive_member']],
                   'archive_url': index['archive_url'], 'archive_bytes': index['archive_bytes']}} for row in original],
                    'freeze_sha256': p.digest(self.out/'freeze.json'),
                    'selection_sha256': p.digest(self.out/'selected_manifest.json'),
                    'index_hashes': {'archive_index_12.json': p.digest(self.out/'archive_index_12.json')}}
        p.atomic_json(self.out/'resolved_manifest.json', resolved)
        return resolved

    def test_resolved_manifest_cannot_change_labels_order_or_audio_location(self):
        good = self.resolved_fixture()
        self.assertEqual(p.resolved_rows(self.out), good['tracks'])
        bad = []
        value = deepcopy(good); value['tracks'][0]['tags'] = ['rock']; bad.append(value)
        value = deepcopy(good); value['tracks'].reverse(); bad.append(value)
        value = deepcopy(good); value['tracks'][0]['artist_id'] = 'artist_9'; bad.append(value)
        value = deepcopy(good); value['tracks'][0]['archive']['offset'] += 1; bad.append(value)
        value = deepcopy(good); value['index_hashes'] = {}; bad.append(value)
        value = deepcopy(good); value['selection_sha256'] = 'changed'; bad.append(value)
        for value in bad:
            p.atomic_json(self.out/'resolved_manifest.json', value)
            with self.assertRaises(ValueError):
                p.resolved_rows(self.out)

    def test_resolved_manifest_rejects_changed_header_index(self):
        self.resolved_fixture()
        path = self.out/'archive_index_12.json'
        index = p.read(path); index['members']['12/1.low.mp3']['size'] += 1
        p.atomic_json(path, index)
        with self.assertRaises(ValueError):
            p.resolved_rows(self.out)

    def test_calculate_pure_pipeline_connects_actual_interfaces_and_bootstrap(self):
        config = p.read(p.HERE/'config.json')
        p.atomic_json(self.out/'config.json', config)
        rows = [{'track_id': f'track_{i:07d}', 'artist_id': f'artist_{i}', 'tags': ['pop']} for i in range(1, 4)]
        p.atomic_json(self.out/'selected_manifest.json', {'tracks': rows})
        baseline = {'means': np.zeros((4, 2304)), 'scales': np.ones((4, 2304)),
                    'coefficients': np.zeros((4, 2304)), 'intercepts': np.array([-.5, .5, 1., -.3]),
                    'thresholds': {'f1': [.39999999999999997, .27499999999999997, .37499999999999994, .27499999999999997],
                                   'precision_target': [.625, 1.01, 1.01, .525]}}
        candidate = deepcopy(baseline)
        candidate.update(offsets=np.zeros(4), candidate_id='synthetic')
        candidate['thresholds']['f1'][2] = .2
        candidate['intercepts'][2] = -.2
        correction = {'logit_offsets': [0., 0., -.7, 0.]}
        with patch.object(p, 'load_bundle', return_value=baseline), patch.object(p, 'load_candidate', return_value=candidate), \
                patch.object(p, 'load_correction', return_value=correction):
            arrays, records = p.calculate(self.out, rows, np.zeros((3, 2304), dtype=np.float32))
        self.assertEqual(arrays['targets'].shape, (3, 4))
        self.assertEqual(arrays['bootstrap_artist_indices'].shape, (2000, 3))
        self.assertEqual(records['summary.json']['assessment']['verdict'], 'incomplete_evidence')
        self.assertFalse(records['summary.json']['refitted'])
        np.testing.assert_array_equal(arrays['baseline__probability'][:, [0, 3]],
                                      arrays['candidate__probability'][:, [0, 3]])


if __name__ == '__main__':
    unittest.main()
