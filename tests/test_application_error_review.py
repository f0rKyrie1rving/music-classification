"""In-memory HTTP tests for the local reviewer; never bind a listening port."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import review_application_errors as review


class FakeSocket:
    def __init__(self, request):
        self.request = io.BytesIO(request)
        self.output = bytearray()

    def makefile(self, *args, **kwargs):
        return self.request

    def sendall(self, data):
        self.output.extend(data)


class ApplicationErrorReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio_path = self.root / 'private_track_title.wav'
        with wave.open(str(self.audio_path), 'wb') as audio:
            audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(22050)
            audio.writeframes(b'\x01\x00' * 32)
        self.audio = self.audio_path.read_bytes()
        self.manifest_path = self.root / 'manifest.json'
        self.answers_path = self.root / 'answers.json'
        self.manifest = {'format_version': 1, 'review_id': 'test-pilot', 'questions': [
            {'query_id': 'anon001', 'target': 'pop', 'audio_file': self.audio_path.name,
             'audio_sha256': review.sha256_bytes(self.audio)},
            {'query_id': 'anon002', 'target': 'ambient', 'audio_file': self.audio_path.name,
             'audio_sha256': review.sha256_bytes(self.audio)},
        ]}
        self.write_manifest()
        self.state = self.load()

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')

    def load(self):
        return review.ReviewState(self.manifest_path, self.answers_path, root=self.root, token='known-session-token')

    def request(self, method='GET', path='/', headers=None, body=b''):
        pairs = [('Host', '127.0.0.1:8123')]
        if headers is not None:
            pairs = [(k, v) for k, v in pairs if k not in headers]
            pairs += list(headers.items())
        raw = f'{method} {path} HTTP/1.1\r\n'.encode()
        raw += ''.join(f'{k}: {v}\r\n' for k, v in pairs).encode() + b'\r\n' + body
        connection = FakeSocket(raw)
        server = SimpleNamespace(state=self.state, server_address=('127.0.0.1', 8123))
        review.ReviewHandler(connection, ('127.0.0.1', 9999), server)
        head, response = bytes(connection.output).split(b'\r\n\r\n', 1)
        lines = head.decode().split('\r\n')
        status = int(lines[0].split()[1])
        response_headers = dict(line.split(': ', 1) for line in lines[1:])
        return status, response_headers, response

    def post(self, payload=None, changes=None, body=None):
        if payload is None:
            payload = {'query_id': 'anon001', 'answer': 'uncertain', 'reason': 'Need another listen'}
        encoded = json.dumps(payload).encode() if body is None else body
        headers = {'Origin': 'http://127.0.0.1:8123', 'Sec-Fetch-Site': 'same-origin',
                   'X-Review-Token': 'known-session-token', 'Content-Type': 'application/json',
                   'Content-Length': str(len(encoded))}
        for key, value in (changes or {}).items():
            if value is None:
                headers.pop(key, None)
            else:
                headers[key] = value
        return self.request('POST', '/api/answer', headers, encoded)

    def test_loading_or_getting_page_never_creates_answers(self):
        status, headers, body = self.request()
        self.assertEqual(status, 200)
        self.assertIn('试听核查试点', body.decode())
        self.assertIn('全部 84 个案例', body.decode())
        self.assertNotIn(self.audio_path.name, body.decode())
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        self.assertEqual(headers['X-Frame-Options'], 'DENY')
        self.assertFalse(self.answers_path.exists())

    def test_public_api_hides_private_audio_metadata(self):
        status, _, body = self.request(path='/api/questions', headers={'X-Review-Token': 'known-session-token'})
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(set(payload), {'questions', 'completed_count'})
        self.assertEqual(set(payload['questions'][0]), {'query_id', 'target', 'answer', 'history'})
        self.assertEqual(payload['completed_count'], 0)
        self.assertNotIn('audio_file', body.decode())
        self.assertNotIn(self.audio_path.name, body.decode())
        self.assertNotIn('score', body.decode())

    def test_question_api_and_audio_require_session_token(self):
        for path in ('/api/questions', '/audio/anon001', '/audio/anon001?token=wrong',
                     '/audio/anon001?token=known-session-token&token=known-session-token'):
            self.assertEqual(self.request(path=path)[0], 403)

    def test_audio_allowlist_rejects_paths_and_unknown_questions(self):
        for path in ('/audio/../manifest.json', '/audio/%2e%2e%2fmanifest.json', '/audio/unknown'):
            self.assertEqual(self.request(path=path + '?token=known-session-token')[0], 404)
        self.assertEqual(self.request(path='/manifest.json')[0], 404)

    def test_audio_ranges_are_bounded_and_report_correct_headers(self):
        path = '/audio/anon001?token=known-session-token'
        status, headers, body = self.request(path=path, headers={'Range': 'bytes=0-3'})
        self.assertEqual((status, body), (206, b'RIFF'))
        self.assertEqual(headers['Content-Range'], f'bytes 0-3/{len(self.audio)}')
        status, _, body = self.request(path=path, headers={'Range': 'bytes=-5'})
        self.assertEqual((status, body), (206, self.audio[-5:]))
        status, _, body = self.request(path=path, headers={'Range': 'bytes=5-999999'})
        self.assertEqual((status, body), (206, self.audio[5:]))
        self.assertEqual(self.request('HEAD', path=path)[2], b'')

    def test_malformed_and_unsatisfiable_ranges_return_416(self):
        invalid = ['bytes=99-2', 'bytes=-0', 'bytes=-', 'bytes=abc-def', 'bytes=0-1,3-5',
                   'items=0-5', 'bytes=999999-', 'bytes=' + '9' * 100 + '-']
        for value in invalid:
            status, headers, body = self.request(path='/audio/anon001?token=known-session-token',
                                                  headers={'Range': value})
            self.assertEqual(status, 416, value)
            self.assertEqual(headers['Content-Range'], f'bytes */{len(self.audio)}')
            self.assertEqual(body, b'')

    def test_same_origin_and_token_required_before_writing(self):
        for changes in ({'Origin': 'https://outside.example'}, {'Origin': None},
                        {'X-Review-Token': 'wrong'}, {'X-Review-Token': None},
                        {'Sec-Fetch-Site': 'cross-site'}, {'Host': 'outside.example:8123'}):
            self.assertEqual(self.post(changes=changes)[0], 403)
            self.assertFalse(self.answers_path.exists())
        self.assertEqual(self.request(headers={'Host': 'attacker.example:8123'})[0], 403)

    def test_only_explicit_valid_user_answers_are_persisted(self):
        for payload in ({'query_id': 'anon001', 'answer': 'auto', 'reason': ''},
                        {'query_id': 'unknown', 'answer': 'yes', 'reason': ''},
                        {'query_id': 'anon001', 'answer': 'yes', 'reason': 'x' * 2001},
                        {'query_id': 'anon001', 'answer': 'yes', 'reason': '', 'score': 0.9}):
            self.assertEqual(self.post(payload)[0], 400)
            self.assertFalse(self.answers_path.exists())
        self.assertEqual(self.post(body=b'{not JSON')[0], 400)
        self.assertEqual(self.post(changes={'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(self.post(changes={'Content-Length': '999999'})[0], 413)

    def test_answer_revision_history_survives_restart(self):
        self.assertEqual(self.post()[0], 200)
        self.assertEqual(self.post({'query_id': 'anon001', 'answer': 'yes', 'reason': 'Changed my judgment'})[0], 200)
        self.state = self.load()
        payload = self.state.public_state()
        self.assertEqual(payload['completed_count'], 1)
        question = payload['questions'][0]
        self.assertEqual(question['answer']['answer'], 'yes')
        self.assertEqual([h['answer'] for h in question['history']], ['uncertain', 'yes'])
        self.assertEqual(self.manifest, json.loads(self.manifest_path.read_text()))
        self.assertEqual(self.audio, self.audio_path.read_bytes())

    def test_atomic_save_failure_does_not_mark_answer_saved(self):
        with patch.object(review.os, 'replace', side_effect=OSError('synthetic write failure')):
            self.assertEqual(self.post()[0], 500)
        self.assertFalse(self.answers_path.exists())
        self.assertEqual(self.state.public_state()['completed_count'], 0)
        self.assertEqual(list(self.root.glob('answers.json.*.tmp')), [])

    def test_extended_manifest_preserves_prior_answers_and_rejects_reassigned_query(self):
        self.post()
        self.manifest['questions'].append({**self.manifest['questions'][0], 'query_id': 'anon003'})
        self.write_manifest()
        continued = self.load()
        self.assertEqual(len(continued.public_state()['questions']), 3)
        self.assertEqual(continued.public_state()['completed_count'], 1)
        self.manifest['questions'][0]['target'] = 'ambient'
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.load()

    def test_manifest_rejects_hidden_metadata_duplicates_and_escaping_audio(self):
        original = json.loads(json.dumps(self.manifest))
        cases = [lambda m: m['questions'][0].update(role='false-positive'),
                 lambda m: m['questions'].append(m['questions'][0].copy()),
                 lambda m: m['questions'][0].update(audio_file='../escape.wav'),
                 lambda m: m['questions'][0].update(audio_file=str(self.audio_path)),
                 lambda m: m['questions'][0].update(audio_sha256='0' * 64)]
        for modify in cases:
            self.manifest = json.loads(json.dumps(original)); modify(self.manifest); self.write_manifest()
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.load()

    def test_audio_modified_after_start_is_not_served(self):
        self.audio_path.write_bytes(self.audio[:-2] + b'xx')
        status, _, body = self.request(path='/audio/anon001?token=known-session-token')
        self.assertEqual(status, 409)
        self.assertNotIn(self.audio_path.name, body.decode())

    def test_server_binding_is_hardcoded_to_loopback(self):
        server = SimpleNamespace(server_address=('127.0.0.1', 54321), serve_forever=lambda: None,
                                 server_close=lambda: None)
        with patch.object(review, 'ROOT', self.root):
            with patch.object(review, 'ReviewState', return_value=self.state):
                with patch.object(review, 'ThreadingHTTPServer', return_value=server) as factory:
                    review.main(['--manifest', str(self.manifest_path), '--answers', str(self.answers_path)])
        factory.assert_called_once_with(('127.0.0.1', 0), review.ReviewHandler)


if __name__ == '__main__':
    unittest.main()
