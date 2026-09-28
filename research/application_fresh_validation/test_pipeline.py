"""Synthetic integrity/resumption tests; no new-cohort audio or model inference."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from . import pipeline as p


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / 'run'
        self.out.mkdir()
        self.rows = [
            {'track_id': 'track_0000001', 'artist_id': 'artist_1'},
            {'track_id': 'track_0000002', 'artist_id': 'artist_2'},
        ]

    def tearDown(self):
        self.temp.cleanup()

    def test_paths_reject_traversal_absolute_and_symlink_escape(self):
        for name in ('../outside', '/outside'):
            with self.assertRaises(ValueError):
                p.safe_file(self.out, name)
        (self.out / 'link').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            p.safe_file(self.out, 'link/elsewhere')

    def test_selected_ids_reject_duplicates_or_path_injection(self):
        for rows in ([], self.rows + [self.rows[0]], [{'track_id': '../oops', 'artist_id': 'artist_1'}]):
            with self.assertRaises(ValueError):
                p.validate_ids(rows)

    def test_hash_selection_includes_available_second_song_and_fixed_order(self):
        candidates = self.rows + [{'track_id': 'track_0000003', 'artist_id': 'artist_1'}]
        config = {'sample_seed': 'fixed', 'artist_count': 2, 'tracks_per_artist': 2}
        expected = p.expected_selection_ids(candidates, config)
        self.assertEqual(set(expected), {r['track_id'] for r in candidates})
        self.assertEqual(expected, p.expected_selection_ids(list(reversed(candidates)), config))
        self.assertEqual(len(p.expected_selection_ids(candidates, {**config, 'tracks_per_artist': 1})), 2)

    def test_runtime_verification_rejects_changed_evaluation_dependency(self):
        frozen = {'runtime': {'numpy': p.importlib.metadata.version('numpy')},
                  'evaluation_runtime': {'scikit-learn': p.importlib.metadata.version('scikit-learn')}}
        p.verify_runtime(frozen)
        frozen['evaluation_runtime']['scikit-learn'] = 'changed'
        with self.assertRaises(ValueError):
            p.verify_runtime(frozen)

    def acquisition(self):
        return {'requested': 2, 'replacement_tracks': 0,
                'successful': [{'track_id': 'track_0000001'}],
                'failures': [{'track_id': 'track_0000002', 'error': 'decode failure'}]}

    def test_acquisition_preserves_failures_and_exact_partition(self):
        record = self.acquisition()
        p.atomic_json(self.out / 'download_status.json', record)
        received, success, failed = p.acquisition_rows(self.out, self.rows)
        self.assertEqual(received, record)
        self.assertEqual(set(success), {'track_0000001'})
        self.assertEqual(set(failed), {'track_0000002'})

    def test_duplicate_missing_overlap_and_replaced_downloads_rejected(self):
        malformed = []
        record = self.acquisition(); record['successful'] *= 2; malformed.append(record)
        record = self.acquisition(); record['failures'] = []; malformed.append(record)
        record = self.acquisition(); record['failures'][0]['track_id'] = 'track_0000001'; malformed.append(record)
        record = self.acquisition(); record['replacement_tracks'] = 1; malformed.append(record)
        record = self.acquisition(); record['failures'][0]['error'] = ''; malformed.append(record)
        for record in malformed:
            p.atomic_json(self.out / 'download_status.json', record)
            with self.assertRaises(ValueError):
                p.acquisition_rows(self.out, self.rows)

    def checkpoint(self, status='complete'):
        file = self.out / 'track_0000001.json'
        expected = {'track_id': 'track_0000001', 'audio_sha256': 'audio',
                    'freeze_sha256': 'freeze', 'download_status_sha256': 'download'}
        record = {**expected, 'status': status}
        if status == 'complete':
            p.save_vector(file.with_suffix('.npy'), np.arange(p.WIDTH, dtype=np.float32))
            record['feature_sha256'] = p.digest(file.with_suffix('.npy'))
        else:
            record['error'] = 'synthetic extraction failure'
        p.atomic_json(file, record)
        return file, expected

    def test_completed_checkpoint_resumes_and_rejects_changed_input(self):
        file, expected = self.checkpoint()
        _, vector = p.verify_track_checkpoint(file, expected)
        np.testing.assert_array_equal(vector, np.arange(p.WIDTH, dtype=np.float32))
        for field in ('audio_sha256', 'freeze_sha256', 'download_status_sha256'):
            with self.assertRaises(ValueError):
                p.verify_track_checkpoint(file, {**expected, field: 'different'})

    def test_checkpoint_detects_modified_feature_bytes(self):
        file, expected = self.checkpoint()
        with file.with_suffix('.npy').open('ab') as stream:
            stream.write(b'mutation')
        with self.assertRaises(ValueError):
            p.verify_track_checkpoint(file, expected)

    def test_failure_checkpoint_is_preserved_without_silent_retry(self):
        file, expected = self.checkpoint('failed')
        record, vector = p.verify_track_checkpoint(file, expected)
        self.assertIsNone(vector)
        self.assertEqual(record['error'], 'synthetic extraction failure')

    def test_orphan_array_adoption_requires_exact_recomputation(self):
        file = self.out / 'orphan.npy'
        vector = np.ones(p.WIDTH, dtype=np.float32)
        p.save_vector(file, vector)
        before = file.stat().st_mtime_ns
        p.save_vector(file, vector.copy())
        self.assertEqual(file.stat().st_mtime_ns, before)
        with self.assertRaises(ValueError):
            p.save_vector(file, vector + 1)

    def test_immutable_receipt_cannot_be_overwritten(self):
        file = self.out / 'receipt.json'
        p.immutable_json(file, {'source': 'fixed'})
        with self.assertRaises(ValueError):
            p.immutable_json(file, {'source': 'changed'})

    def synthetic_audio_receipt(self):
        (self.out / 'audio').mkdir()
        prefix_dir = self.out / 'acquisition/audio_prefixes'
        prefix_dir.mkdir(parents=True)
        receipt_dir = self.out / 'acquisition/download_receipts'
        receipt_dir.mkdir()
        track = 'track_0000001'
        audio = self.out / 'audio' / f'{track}.wav'
        sf.write(audio, np.full(661500, 0.1, dtype=np.float32), 22050, subtype='PCM_16')
        prefix = prefix_dir / f'{track}.mp3.part'
        prefix.write_bytes(b'abcdef')
        row = {'track_id': track, 'audio_file': str(audio.relative_to(self.root)),
               'archive': {'offset': 100, 'size': 10}, 'expected_full_sha256': 'unused-full-reference'}
        receipt = {'track_id': track, 'wav_sha256': p.digest(audio), 'downloaded_bytes': 6,
                   'byte_ranges': [{'byte_range': [100, 105], 'sha256': p.digest(prefix),
                                    'cache_file': str(prefix.relative_to(self.root))}],
                   'prefix_sha256': p.digest(prefix), 'full_mp3_sha256_verified': False,
                   'frames': 661500, 'sample_rate': 22050}
        file = receipt_dir / f'{track}.json'
        p.atomic_json(file, receipt)
        return row, receipt, file, prefix

    def test_partial_audio_receipt_checks_hashes_and_does_not_claim_full_hash(self):
        row, receipt, _, prefix = self.synthetic_audio_receipt()
        config = {'maximum_prefix_bytes_per_track': 1310720}
        with patch.object(p, 'ROOT', self.root):
            self.assertEqual(p.verify_audio_receipt(self.out, row, receipt, config), receipt)
            prefix.write_bytes(b'abcdeg')
            with self.assertRaises(ValueError):
                p.verify_audio_receipt(self.out, row, receipt, config)

    def test_audio_receipt_forged_full_file_claim_rejected(self):
        row, receipt, file, _ = self.synthetic_audio_receipt()
        receipt['full_mp3_sha256_verified'] = True
        p.atomic_json(file, receipt)
        with patch.object(p, 'ROOT', self.root):
            with self.assertRaises(ValueError):
                p.verify_audio_receipt(self.out, row, receipt, {'maximum_prefix_bytes_per_track': 1310720})

    def test_download_success_must_match_original_receipt(self):
        row, receipt, _, _ = self.synthetic_audio_receipt()
        changed = deepcopy(receipt); changed['downloaded_bytes'] += 1
        with patch.object(p, 'ROOT', self.root):
            with self.assertRaises(ValueError):
                p.verify_audio_receipt(self.out, row, changed, {'maximum_prefix_bytes_per_track': 1310720})

    def test_freeze_rejects_preexisting_audio_before_reading_other_inputs(self):
        (self.out / 'audio').mkdir()
        (self.out / 'audio/example.wav').write_bytes(b'already acquired')
        with self.assertRaises(ValueError):
            p.freeze(self.out)

    def synthetic_feature_run(self):
        (self.out / 'audio').mkdir()
        rows, receipts = [], []
        for item in self.rows:
            path = self.out / 'audio' / (item['track_id'] + '.wav')
            path.write_bytes(item['track_id'].encode())
            rows.append({**item, 'audio_file': str(path.relative_to(self.root)), 'tags': []})
            receipts.append({'track_id': item['track_id'], 'wav_sha256': p.digest(path)})
        (self.root / 'control.wav').write_bytes(b'old synthetic control')
        old = self.root / 'outputs/maest_hf'; old.mkdir(parents=True)
        np.save(old / 'features.npy', np.zeros((1, p.WIDTH), dtype=np.float32), allow_pickle=False)
        versions = {'numpy': p.importlib.metadata.version('numpy')}
        p.atomic_json(old / 'features.json', {'versions': versions})
        encoder_dir = self.root / 'models/mtg-upf-maest-519l'; encoder_dir.mkdir(parents=True)
        p.atomic_json(encoder_dir / 'SOURCES.json', {'synthetic': True})
        (self.root / 'maest_hf_features.py').write_text('# Synthetic source only\n')
        frozen = {'control_track_id': 'track_control', 'control_audio_file': 'control.wav', 'runtime': versions}
        p.atomic_json(self.out / 'freeze.json', frozen)
        p.atomic_json(self.out / 'config.json', {'maximum_prefix_bytes_per_track': 1310720})
        p.atomic_json(self.out / 'selected_manifest.json', {'tracks': rows})
        p.atomic_json(self.out / 'download_status.json', {
            'requested': 2, 'replacement_tracks': 0, 'successful': receipts, 'failures': []})
        return frozen

    def test_complete_feature_run_reuses_verified_cache_without_inference(self):
        frozen = self.synthetic_feature_run()
        encoder = Mock()
        encoder.extract.side_effect = lambda path: np.full(
            p.WIDTH, 0 if path.name == 'control.wav' else int(path.stem[-1]), dtype=np.float32)
        with patch.object(p, 'ROOT', self.root), patch.object(p, 'verify_frozen', return_value=frozen), \
                patch.object(p, 'verify_audio_receipt'), patch.object(p, 'targets', return_value=np.zeros((2, 4), int)), \
                patch('maest_hf_features.HfMaestEncoder', return_value=encoder) as factory:
            p.features(self.out)
            before = (self.out / 'features.npy').stat().st_mtime_ns
            p.features(self.out)
            self.assertEqual((self.out / 'features.npy').stat().st_mtime_ns, before)
            factory.assert_called_once_with('cpu')
            self.assertEqual(encoder.extract.call_count, 3)
        self.assertEqual(p.read(self.out / 'features.json')['ids'], [r['track_id'] for r in self.rows])

    def test_interrupted_extraction_resumes_completed_track_without_reextracting(self):
        frozen = self.synthetic_feature_run()
        calls, interrupted = [], [False]
        def extract(path):
            calls.append(path.name)
            if path.name == 'track_0000002.wav' and not interrupted[0]:
                interrupted[0] = True
                raise KeyboardInterrupt('synthetic process interruption')
            return np.full(p.WIDTH, 0 if path.name == 'control.wav' else int(path.stem[-1]), dtype=np.float32)
        encoder = Mock(); encoder.extract.side_effect = extract
        with patch.object(p, 'ROOT', self.root), patch.object(p, 'verify_frozen', return_value=frozen), \
                patch.object(p, 'verify_audio_receipt'), patch.object(p, 'targets', return_value=np.zeros((2, 4), int)), \
                patch('maest_hf_features.HfMaestEncoder', return_value=encoder):
            with self.assertRaises(KeyboardInterrupt):
                p.features(self.out)
            self.assertTrue((self.out / 'feature_tracks/track_0000001.json').exists())
            self.assertFalse((self.out / 'extraction_status.json').exists())
            p.features(self.out)
        self.assertEqual(calls.count('track_0000001.wav'), 1)
        self.assertEqual(calls.count('control.wav'), 2)
        self.assertEqual(p.read(self.out / 'extraction_status.json')['observed_tracks'], 2)


if __name__ == '__main__':
    unittest.main()
