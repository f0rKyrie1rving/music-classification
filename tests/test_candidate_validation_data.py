"""No-network checks of frozen confirmation selection and bounded acquisition."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research.application_candidate_validation import acquire, data


class ConfirmationDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, obj):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))
        return path

    def rows(self):
        return [{'track_id': f'track_{a * 10 + t:07d}', 'artist_id': f'artist_{a:06d}',
                 'path': f'12/{a * 10 + t}.mp3', 'archive_member': f'12/{a * 10 + t}.low.mp3',
                 'tags': ['pop']} for a in range(6) for t in range(1 + a % 4)]

    def index_setup(self, names):
        rows = [{'path': n.replace('.low.mp3', '.mp3'), 'archive_member': n} for n in names]
        self.write('candidate_pool.json', {'tracks': rows})
        self.write('freeze.json', {'frozen': True})
        self.write('selected_manifest.json', {'tracks': []})
        self.write('config.json', {'archives': ['12'], 'network': {
            'attempts_per_range': 3, 'connect_timeout_seconds': 8,
            'request_timeout_seconds': 25, 'index_workers': 4,
            'download_workers': 4, 'maximum_index_invocations': 3}})

    def tar(self, names):
        payload = b''
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = 16
            payload += info.tobuf() + b'x' * 16 + b'\0' * 496
        return payload + b'\0' * 1024

    def run_index(self, archive):
        def request(out, url, offset, count):
            self.assertLessEqual(offset + count, len(archive))
            return archive[offset:offset + count], len(archive)
        with patch.object(acquire, 'get_range', side_effect=request):
            return acquire.index_one(self.root, '12')

    def test_selection_is_artist_grouped_and_label_and_order_independent(self):
        rows = self.rows()
        config = {'sample_seed': 'fixed', 'artist_count': 4, 'tracks_per_artist': 2}
        with patch.object(data, 'ROOT', self.root):
            first = data.select_rows(rows, config, self.root)
            altered = [{**r, 'tags': ['ambient', 'rock']} for r in reversed(rows)]
            second = data.select_rows(altered, config, self.root)
        self.assertEqual([r['track_id'] for r in first], [r['track_id'] for r in second])
        self.assertEqual(len({r['artist_id'] for r in first}), 4)
        self.assertTrue(all('archive' not in r for r in first))
        self.assertTrue(all(sum(s['artist_id'] == r['artist_id'] for s in first) <= 2 for r in first))

    def test_selection_rejects_duplicates_and_insufficient_artists(self):
        config = {'sample_seed': 'fixed', 'artist_count': 7, 'tracks_per_artist': 2}
        with patch.object(data, 'ROOT', self.root):
            with self.assertRaises(ValueError):
                data.select_rows(self.rows(), config, self.root)
            with self.assertRaises(ValueError):
                data.select_rows(self.rows() + [self.rows()[0]], config, self.root)

    def test_frame_excludes_entire_historical_artist_and_checks_source_blob(self):
        contents = {'genres.tsv': 'id\tartist\talbum\tpath\tduration\ttags\n'
                    'track_0000001\tartist_000001\ta\t12/1.mp3\t31\tgenre---pop\n'
                    'track_0000002\tartist_000002\tb\t12/2.mp3\t31\tgenre---rock\n',
            'licenses.txt': '\n\n'.join(f'12/{i}.mp3\nhttps://creativecommons.org/licenses/by/3.0/' for i in (1, 2)),
            'checksums.txt': '\n'.join('a' * 64 + f' 12/{i}.low.mp3' for i in (1, 2))}
        blobs = {}
        for name, content in contents.items():
            # Hash the exact fixture bytes, without Windows newline translation.
            encoded = content.encode('utf-8')
            (self.root / name).write_bytes(encoded)
            blobs[name] = hashlib.sha1(f'blob {len(encoded)}\0'.encode() + encoded).hexdigest()
        config = {'source_blobs': blobs, 'archives': ['12'], 'allowed_license_codes': ['by'],
                  'expected_frame_tracks': 1, 'expected_frame_artists': 1}
        rows, counts, _ = data.build_frame(self.root, config, {'tracks': [], 'artists': ['artist_000001']})
        self.assertEqual([r['track_id'] for r in rows], ['track_0000002'])
        self.assertEqual(counts['excluded_historical_track_or_artist'], 1)
        (self.root / 'genres.tsv').write_text('changed')
        with self.assertRaises(ValueError):
            data.build_frame(self.root, config, {'tracks': [], 'artists': []})

    def test_index_accepts_only_allowlisted_members_without_extracting(self):
        self.index_setup(['12/123.low.mp3'])
        result = self.run_index(self.tar(['../escape.mp3', '12/123.low.mp3']))
        self.assertEqual(result['members'], {'12/123.low.mp3': {'offset': 1536, 'size': 16}})
        self.assertTrue(result['complete'])
        self.assertFalse((self.root.parent / 'escape.mp3').exists())

    def test_index_rejects_duplicate_or_corrupt_headers(self):
        self.index_setup(['12/123.low.mp3'])
        with self.assertRaises(ValueError):
            self.run_index(self.tar(['12/123.low.mp3', '12/123.low.mp3']))
        self.root = self.root / 'separate_corrupt_archive'
        self.index_setup(['12/123.low.mp3'])
        archive = bytearray(self.tar(['12/123.low.mp3']))
        archive[0] ^= 1
        with self.assertRaises(tarfile.HeaderError):
            self.run_index(bytes(archive))

    def test_index_resume_preserves_last_header_and_stops_at_three_invocations(self):
        self.index_setup(['12/123.low.mp3'])
        archive = self.tar(['12/123.low.mp3'])
        def fail_second(out, url, offset, count):
            if offset:
                raise RuntimeError('offline')
            return archive[:count], len(archive)
        with patch.object(acquire, 'get_range', side_effect=fail_second):
            with self.assertRaises(RuntimeError):
                acquire.index_one(self.root, '12')
        state = json.loads((self.root / 'archive_index_12.json').read_text())
        self.assertEqual(state['next_offset'], 1024)
        self.assertEqual(state['headers_scanned'], 1)
        with patch.object(acquire, 'get_range', side_effect=RuntimeError('offline')) as request:
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    acquire.index_one(self.root, '12')
            with self.assertRaises(ValueError):
                acquire.index_one(self.root, '12')
            self.assertEqual(request.call_count, 2)

    def test_resolve_preserves_selected_bytes_and_binds_index_hashes(self):
        self.index_setup(['12/123.low.mp3'])
        row = {'track_id': 'track_0000123', 'artist_id': 'artist_000001', 'path': '12/123.mp3',
               'archive_member': '12/123.low.mp3', 'audio_file': 'audio/track_0000123.wav'}
        self.write('selected_manifest.json', {'tracks': [row]})
        self.run_index(self.tar(['12/123.low.mp3']))
        before = (self.root / 'selected_manifest.json').read_bytes()
        with patch.object(acquire, 'verify_frozen'):
            result = acquire.resolve(self.root)
            self.assertEqual(acquire.resolve(self.root), result)
        self.assertEqual(before, (self.root / 'selected_manifest.json').read_bytes())
        self.assertEqual({k: v for k, v in result['tracks'][0].items() if k != 'archive'}, row)
        self.assertEqual(result['index_hashes']['archive_index_12.json'], data.digest(self.root / 'archive_index_12.json'))

    def test_range_logs_all_three_failed_attempts(self):
        self.index_setup([])
        with (patch.object(acquire.subprocess, 'run', return_value=SimpleNamespace(returncode=6, stderr='offline')) as run,
              patch.object(acquire.time, 'sleep')):
            with self.assertRaises(RuntimeError):
                acquire.get_range(self.root, 'https://example.invalid/a', 0, 512)
        self.assertEqual(run.call_count, 3)
        log = json.loads(next((self.root / 'network_requests').glob('*.json')).read_text())
        self.assertEqual([a['state'] for a in log['attempts']], ['failed'] * 3)

    def test_range_rejects_http_200_even_when_content_range_looks_valid(self):
        self.index_setup([])
        def response(cmd, **kwargs):
            Path(cmd[cmd.index('--output') + 1]).write_bytes(b'x' * 512)
            Path(cmd[cmd.index('--dump-header') + 1]).write_text('HTTP/2 200\nContent-Range: bytes 0-511/1024\n')
            return SimpleNamespace(returncode=0, stderr='')
        with patch.object(acquire.subprocess, 'run', side_effect=response), patch.object(acquire.time, 'sleep'):
            with self.assertRaises(RuntimeError):
                acquire.get_range(self.root, 'https://example.invalid/a', 0, 512)

    def test_range_success_retains_header_bytes_and_request_hash(self):
        self.index_setup([])
        acquire.CONTEXT.value = {'phase': 'index', 'shard': '12'}
        def response(cmd, **kwargs):
            Path(cmd[cmd.index('--output') + 1]).write_bytes(b'x' * 512)
            Path(cmd[cmd.index('--dump-header') + 1]).write_text('HTTP/2 206\nContent-Range: bytes 0-511/1024\n')
            return SimpleNamespace(returncode=0, stderr='')
        with patch.object(acquire.subprocess, 'run', side_effect=response):
            body, total = acquire.get_range(self.root, 'https://example.invalid/a', 0, 512)
        self.assertEqual((body, total), (b'x' * 512, 1024))
        log = json.loads(next((self.root / 'network_requests').glob('*.json')).read_text())
        attempt = log['attempts'][0]
        self.assertEqual(data.digest(self.root / attempt['index_header_file']), attempt['sha256'])

    def test_completed_track_failure_is_never_retried(self):
        downloader = Mock(side_effect=RuntimeError('three range attempts exhausted'))
        row = {'track_id': 'track_0000001'}
        first = acquire.download_one(self.root, row, {'freeze': 'same'}, downloader)
        second = acquire.download_one(self.root, row, {'freeze': 'same'}, downloader)
        self.assertEqual(first, second)
        self.assertEqual(downloader.call_count, 1)


if __name__ == '__main__':
    unittest.main()
