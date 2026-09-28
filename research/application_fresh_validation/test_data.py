"""Isolated acquisition checks: no network calls or real research-data writes."""
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock

from research.application_fresh_validation import data


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_licenses_keep_verbatim_and_detect_ambiguous_entries(self):
        path = self.root / 'licenses.txt'
        block = ('04/123.mp3\nExample by Artist\nAvailable under a license: '
                 'https://creativecommons.org/licenses/by-nc-sa/3.0/')
        path.write_text(block + '\n\n04/124.mp3\nNo documented CC URL\n')
        result = data.read_licenses(path)
        self.assertEqual(result['04/123.mp3']['license_code'], 'by-nc-sa')
        self.assertEqual(result['04/123.mp3']['attribution_verbatim'], block)
        self.assertIsNone(result['04/124.mp3']['license_code'])
        path.write_text(block + '\n\n' + block)
        with self.assertRaises(ValueError):
            data.read_licenses(path)

    def test_licenses_reject_path_traversal(self):
        path = self.root / 'licenses.txt'
        path.write_text('../outside.mp3\nhttps://creativecommons.org/licenses/by/3.0/')
        with self.assertRaises(ValueError):
            data.read_licenses(path)

    def frame(self):
        records = []
        for artist in range(6):
            for track in range(1 + artist % 4):
                number = artist * 10 + track
                records.append({'track_id': f'track_{number:07d}',
                    'artist_id': f'artist_{artist:06d}', 'path': f'04/{number}.mp3',
                    'archive_member': f'04/{number}.low.mp3', 'tags': ['ambient']})
        self.write('config.json', {'sample_seed': 'fixed', 'artist_count': 4,
            'tracks_per_artist': 2})
        self.write('candidate_pool.json', {'tracks': records})
        self.write('archive_index_04.json', {'complete': True,
            'archive_url': 'https://example.invalid/test.tar', 'archive_bytes': 100000,
            'members': {r['archive_member']: {'offset': i * 1024 + 512, 'size': 128}
                        for i, r in enumerate(records)}})
        return records

    def test_sample_is_artist_grouped_input_order_and_tag_independent(self):
        records = self.frame()
        with patch.object(data, 'ROOT', self.root):
            data.sample(self.root)
            first = json.loads((self.root / 'selected_manifest.json').read_text())['tracks']
            self.assertEqual(len({r['artist_id'] for r in first}), 4)
            for artist in {r['artist_id'] for r in first}:
                self.assertLessEqual(sum(r['artist_id'] == artist for r in first), 2)
            self.assertTrue(all((self.root / r['audio_file']).is_relative_to(self.root)
                                for r in first))
            (self.root / 'selected_manifest.json').unlink()
            for row in records:
                row['tags'] = ['rock', 'pop']
            self.write('candidate_pool.json', {'tracks': list(reversed(records))})
            data.sample(self.root)
            second = json.loads((self.root / 'selected_manifest.json').read_text())['tracks']
        self.assertEqual([r['track_id'] for r in first], [r['track_id'] for r in second])

    def test_sample_refuses_incomplete_index_and_resampling(self):
        self.frame()
        index = json.loads((self.root / 'archive_index_04.json').read_text())
        index['complete'] = False
        self.write('archive_index_04.json', index)
        with patch.object(data, 'ROOT', self.root):
            with self.assertRaises(ValueError):
                data.sample(self.root)
            index['complete'] = True
            self.write('archive_index_04.json', index)
            data.sample(self.root)
            with self.assertRaises(FileExistsError):
                data.sample(self.root)

    def tar_image(self, names):
        result = b''
        for name in names:
            header = tarfile.TarInfo(name)
            header.size = 16
            result += header.tobuf() + b'x' * 16 + b'\0' * 496
        return result + b'\0' * 1024

    def run_index(self, archive):
        def fake_range(url, offset, count):
            self.assertGreaterEqual(offset, 0)
            self.assertLessEqual(offset + count, len(archive))
            return archive[offset:offset + count], len(archive)
        with patch('prepare_dataset.get_range', side_effect=fake_range):
            data.index_one(self.root, '04')

    def test_index_only_records_exact_allowlisted_members_without_extracting(self):
        self.write('candidate_pool.json', {'tracks': [
            {'path': '04/123.mp3', 'archive_member': '04/123.low.mp3'}]})
        self.run_index(self.tar_image(['../escape.mp3', '04/123.low.mp3']))
        index = json.loads((self.root / 'archive_index_04.json').read_text())
        self.assertTrue(index['complete'])
        self.assertEqual(index['members'], {'04/123.low.mp3': {'offset': 1536, 'size': 16}})
        self.assertFalse((self.root.parent / 'escape.mp3').exists())

    def test_index_rejects_duplicate_wanted_member_and_missing_members(self):
        self.write('candidate_pool.json', {'tracks': [
            {'path': '04/123.mp3', 'archive_member': '04/123.low.mp3'}]})
        with self.assertRaises(ValueError):
            self.run_index(self.tar_image(['04/123.low.mp3', '04/123.low.mp3']))
        with self.assertRaises(ValueError):
            self.run_index(self.tar_image(['04/999.low.mp3']))

    def test_index_rejects_bad_header_checksum(self):
        self.write('candidate_pool.json', {'tracks': [
            {'path': '04/123.mp3', 'archive_member': '04/123.low.mp3'}]})
        archive = bytearray(self.tar_image(['04/123.low.mp3']))
        archive[0] ^= 1
        with self.assertRaises(tarfile.HeaderError):
            self.run_index(bytes(archive))

    def test_download_failure_is_persisted_and_never_retried(self):
        row = {'track_id': 'track_0000001'}
        identity = {'freeze_sha256': 'fixed', 'selection_sha256': 'selected'}
        downloader = Mock(side_effect=RuntimeError('fixed retries exhausted'))
        first = data.download_one(self.root, row, identity, downloader)
        second = data.download_one(self.root, row, identity, downloader)
        self.assertEqual(first, second)
        self.assertEqual(first['state'], 'failed')
        self.assertEqual(downloader.call_count, 1)

    def test_interrupted_download_is_accounted_without_extra_attempt(self):
        row = {'track_id': 'track_0000001'}
        identity = {'freeze_sha256': 'fixed', 'selection_sha256': 'selected'}
        self.write('download_checkpoints/track_0000001.json', {
            **row, 'identity': identity, 'state': 'started', 'started_utc': 'before-crash'})
        downloader = Mock()
        result = data.download_one(self.root, row, identity, downloader)
        self.assertEqual(result['state'], 'failed')
        self.assertIn('Interrupted', result['error'])
        downloader.assert_not_called()
        with self.assertRaises(ValueError):
            data.download_one(self.root, row, {'freeze_sha256': 'different'}, downloader)

    def test_successful_download_resume_verifies_receipt_and_audio(self):
        row = {'track_id': 'track_0000001', 'audio_file': 'audio/track_0000001.wav'}
        path = self.root / row['audio_file']
        path.parent.mkdir(); path.write_bytes(b'checked-audio')
        receipt = {'track_id': row['track_id'], 'wav_sha256': data.digest(path)}
        self.write('acquisition/download_receipts/track_0000001.json', receipt)
        downloader = Mock(return_value=receipt)
        with patch.object(data, 'ROOT', self.root):
            data.download_one(self.root, row, {}, downloader)
            data.download_one(self.root, row, {}, downloader)
            self.assertEqual(downloader.call_count, 1)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                data.download_one(self.root, row, {}, downloader)


if __name__ == '__main__':
    unittest.main()
