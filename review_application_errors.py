"""Loopback-only listening pilot with model scores and source labels hidden.

Run only after creating an explicit review manifest. Nothing is answered
automatically. The private manifest selects audio; the browser sees anonymous
question IDs, target styles, and the reviewer's own answers only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parent
MAX_AUDIO_BYTES = 64 * 1024 * 1024
MAX_POST_BYTES = 16 * 1024
ANSWER_OPTIONS = {'yes', 'no', 'uncertain'}
ID_PATTERN = re.compile(r'[A-Za-z0-9_-]{1,80}')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def checked_path(root, name):
    """Resolve an explicit project-relative file, rejecting traversal/escape."""
    if not isinstance(name, str):
        raise ValueError('Expected a relative audio filename')
    path = Path(name)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError('Audio paths must be project-relative without traversal')
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError('Audio path escapes the project or is not a file')
    return resolved


def load_audio_bytes(path, expected_sha256):
    size = path.stat().st_size
    if not 12 <= size <= MAX_AUDIO_BYTES:
        raise ValueError('Audio file size is outside the review limit')
    data = path.read_bytes()
    if len(data) != size or sha256_bytes(data) != expected_sha256:
        raise ValueError('Review audio changed since the manifest was prepared')
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError('Review audio must be a WAV file')
    return data


def parse_byte_range(header, size):
    """Return inclusive single-range bounds; malformed/unsatisfiable means 416."""
    if type(size) is not int or size <= 0:
        raise ValueError('Empty audio cannot satisfy a byte range')
    if header is None:
        return 0, size - 1, False
    match = re.fullmatch(r'bytes=([0-9]*)-([0-9]*)', header.strip())
    if not match or not any(match.groups()) or any(len(v) > 20 for v in match.groups()):
        raise ValueError('Only one valid byte range is supported')
    first, last = match.groups()
    if not first:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError('Empty suffix range')
        return max(0, size - suffix), size - 1, True
    start = int(first)
    end = int(last) if last else size - 1
    if start >= size or end < start:
        raise ValueError('Unsatisfiable byte range')
    return start, min(end, size - 1), True


def atomic_save(path, record):
    """Durably publish one complete JSON document in the destination directory."""
    encoded = (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix=path.name + '.', suffix='.tmp',
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class ReviewState:
    def __init__(self, manifest_path, answers_path, root=ROOT, token=None):
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        self.answers_path = Path(answers_path).resolve()
        if (not self.answers_path.is_relative_to(self.root)
                or self.answers_path.suffix.lower() != '.json'
                or self.answers_path == self.manifest_path):
            raise ValueError('Choose a separate answers JSON inside this project')
        raw = self.manifest_path.read_bytes()
        self.manifest_sha256 = sha256_bytes(raw)
        manifest = json.loads(raw)
        if (not isinstance(manifest, dict)
                or set(manifest) != {'format_version', 'review_id', 'questions'}
                or type(manifest['format_version']) is not int or manifest['format_version'] != 1
                or not isinstance(manifest['review_id'], str) or not ID_PATTERN.fullmatch(manifest['review_id'])
                or not isinstance(manifest['questions'], list) or not manifest['questions']):
            raise ValueError('Invalid review manifest format')
        self.review_id = manifest['review_id']
        self.questions = []
        self.by_id = {}
        self.fingerprints = {}
        for question in manifest['questions']:
            if not isinstance(question, dict) or set(question) != {'query_id', 'target', 'audio_file', 'audio_sha256'}:
                raise ValueError('Each review question needs only its ID, target and explicit audio identity')
            query = question['query_id']
            if (not isinstance(query, str) or not ID_PATTERN.fullmatch(query) or query in self.by_id
                    or not isinstance(question['target'], str)
                    or question['target'] not in {'pop', 'ambient'}
                    or not isinstance(question['audio_sha256'], str)
                    or not re.fullmatch(r'[0-9a-f]{64}', question['audio_sha256'])):
                raise ValueError('Invalid or duplicate review question')
            path = checked_path(self.root, question['audio_file'])
            if path.suffix.lower() != '.wav' or path == self.answers_path:
                raise ValueError('Only the explicitly listed WAV audio can be reviewed')
            load_audio_bytes(path, question['audio_sha256'])
            private = {**question, 'resolved_path': path}
            self.questions.append(private)
            self.by_id[query] = private
            self.fingerprints[query] = sha256_bytes(json.dumps(
                {key: question[key] for key in ('query_id', 'target', 'audio_sha256')},
                sort_keys=True).encode())
        self.token = token or secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.record = self._load_answers()

    def _load_answers(self):
        if not self.answers_path.exists():
            return {'format_version': 1, 'review_id': self.review_id, 'created_at': utc_now(),
                    'question_fingerprints': dict(self.fingerprints),
                    'manifest_hashes': [self.manifest_sha256], 'answers': {}, 'history': []}
        record = json.loads(self.answers_path.read_text(encoding='utf-8'))
        required = {'format_version', 'review_id', 'created_at', 'question_fingerprints',
                    'manifest_hashes', 'answers', 'history'}
        if (not isinstance(record, dict) or set(record) != required or record['format_version'] != 1
                or record['review_id'] != self.review_id
                or not isinstance(record['question_fingerprints'], dict)
                or not isinstance(record['manifest_hashes'], list)
                or not isinstance(record['answers'], dict) or not isinstance(record['history'], list)):
            raise ValueError('Answers belong to another review or have an invalid format')
        for query, identity in self.fingerprints.items():
            if query in record['question_fingerprints'] and record['question_fingerprints'][query] != identity:
                raise ValueError('A previously reviewed question changed its target or audio')
        latest = {}
        for item in record['history']:
            if (not isinstance(item, dict) or set(item) != {'query_id', 'answer', 'reason', 'saved_at', 'manifest_sha256'}
                    or item['query_id'] not in record['question_fingerprints']
                    or item['answer'] not in ANSWER_OPTIONS or not isinstance(item['reason'], str)
                    or len(item['reason']) > 2000 or not isinstance(item['saved_at'], str)
                    or item['manifest_sha256'] not in record['manifest_hashes']):
                raise ValueError('Answer history is invalid')
            latest[item['query_id']] = {k: item[k] for k in ('answer', 'reason', 'saved_at')}
        if latest != record['answers']:
            raise ValueError('Latest answers do not match the saved review history')
        record['question_fingerprints'].update(self.fingerprints)
        if self.manifest_sha256 not in record['manifest_hashes']:
            record['manifest_hashes'].append(self.manifest_sha256)
        return record

    def public_state(self):
        with self.lock:
            questions = []
            for question in self.questions:
                query = question['query_id']
                history = [{k: h[k] for k in ('answer', 'reason', 'saved_at')}
                           for h in self.record['history'] if h['query_id'] == query]
                questions.append({'query_id': query, 'target': question['target'],
                                  'answer': self.record['answers'].get(query), 'history': history})
            return {'questions': questions, 'completed_count': sum(q['answer'] is not None for q in questions)}

    def audio(self, query):
        if query not in self.by_id:
            raise KeyError('Unknown audio question')
        question = self.by_id[query]
        current = checked_path(self.root, question['audio_file'])
        if current != question['resolved_path']:
            raise ValueError('Manifest audio path changed')
        return load_audio_bytes(current, question['audio_sha256'])

    def answer(self, payload):
        if (not isinstance(payload, dict) or set(payload) != {'query_id', 'answer', 'reason'}
                or not isinstance(payload['query_id'], str) or payload['query_id'] not in self.by_id
                or not isinstance(payload['answer'], str) or payload['answer'] not in ANSWER_OPTIONS
                or not isinstance(payload['reason'], str) or len(payload['reason']) > 2000):
            raise ValueError('Choose yes, no or uncertain and keep the reason within 2000 characters')
        with self.lock:
            if self.answers_path.resolve() != self.answers_path:
                raise ValueError('Answers destination changed since this review started')
            # Copy before saving so failed persistence never appears as a saved answer.
            next_record = json.loads(json.dumps(self.record))
            saved_at = utc_now()
            latest = {'answer': payload['answer'], 'reason': payload['reason'], 'saved_at': saved_at}
            next_record['answers'][payload['query_id']] = latest
            next_record['history'].append({**payload, 'saved_at': saved_at,
                                           'manifest_sha256': self.manifest_sha256})
            atomic_save(self.answers_path, next_record)
            self.record = next_record
            return self.public_state()


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>试听核查试点</title><style nonce="__NONCE__">
body{font:17px/1.55 system-ui,sans-serif;background:#f4f5f7;color:#19212c;margin:0}
main{max-width:760px;margin:32px auto;padding:26px;background:white;border-radius:14px}
h1{margin:0 0 10px;font-size:27px}h2{font-size:23px;margin:10px 0}.muted{color:#596473;font-size:15px}
.notice{background:#eef2f7;padding:12px 15px;border-radius:7px}audio{width:100%;margin:12px 0}
fieldset{border:0;padding:0;margin:18px 0;display:flex;flex-wrap:wrap;gap:12px}
label.option{border:1px solid #adb7c4;border-radius:8px;padding:12px 16px;cursor:pointer}
textarea{box-sizing:border-box;width:100%;min-height:90px;font:inherit;padding:9px;border:1px solid #adb7c4;border-radius:6px}
button,select{font:inherit;padding:9px 13px;border:1px solid #adb7c4;border-radius:6px;background:white;cursor:pointer}
button.primary{background:#2256a4;color:white;border-color:#2256a4}button:disabled{opacity:.45;cursor:default}
.nav{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:17px}.status{min-height:28px;margin:10px 0}
progress{width:100%;height:12px}.history-item{border-top:1px solid #ddd;padding:8px 0;white-space:pre-wrap}
@media(max-width:650px){main{margin:0;border-radius:0;padding:18px}fieldset{display:grid}}
</style></head><body><main>
<h1>试听核查试点</h1>
<p class="notice muted">页面隐藏模型分数、原始风格标签及歌曲资料，只显示匿名题号和本题要判断的风格。这轮试点不能代表全部 84 个案例。核查尚未完成时，不据此改动数据集。</p>
<div id="progressText" class="muted">正在载入…</div><progress id="progress" value="0" max="1"></progress>
<div class="nav"><label for="jump">跳转题目</label><select id="jump" aria-label="跳转题目"></select><button id="resume">继续未完成题目</button></div>
<p id="query" class="muted"></p><h2 id="target"></h2>
<p>只根据这段音频，判断本题的目标风格是否适用。拿不准时请选择“不确定”；不需要为了完成而猜测。</p>
<audio id="audio" controls preload="none" aria-label="本题音频"></audio>
<fieldset><legend>你的判断</legend>
<label class="option"><input type="radio" name="answer" value="yes"> 是，适用</label>
<label class="option"><input type="radio" name="answer" value="no"> 否，不适用</label>
<label class="option"><input type="radio" name="answer" value="uncertain"> 不确定</label></fieldset>
<label for="reason">理由（选填）</label><textarea id="reason" maxlength="2000" placeholder="可以简短记录听到的依据或不确定之处。"></textarea>
<div class="nav"><button id="previous">上一题</button><button id="save" class="primary">保存并继续</button><button id="next">下一题</button></div>
<p id="status" class="status" role="status" aria-live="polite"></p>
<details id="history"><summary id="historyTitle">本题保存历史</summary><div id="historyItems"></div></details>
<p class="muted">可以回到任何一题修改答案；每次保存都会保留历史。未保存的选择不会记入结果。回答单独保存，不会自动修改数据集。</p>
</main><script nonce="__NONCE__">
'use strict';const token=__TOKEN__;const $=id=>document.getElementById(id);
let state={questions:[],completed_count:0},index=0,dirty=false,busy=false;
const labels={yes:'是，适用',no:'否，不适用',uncertain:'不确定'};
function status(text){$('status').textContent=text;}
function setBusy(value){busy=value;$('save').disabled=value;$('reason').disabled=value;$('jump').disabled=value;$('resume').disabled=value;document.querySelectorAll('input[name="answer"]').forEach(input=>input.disabled=value);}
function paint(){const q=state.questions[index];if(!q)return;dirty=false;
 $('progressText').textContent=`已保存 ${state.completed_count} / ${state.questions.length} 题 · 当前第 ${index+1} 题`;
 $('progress').max=state.questions.length;$('progress').value=state.completed_count;
 $('jump').replaceChildren();state.questions.forEach((v,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${v.query_id}${v.answer?' · 已保存':''}`;$('jump').append(o);});$('jump').value=index;
 $('query').textContent=`匿名题号：${q.query_id}`;$('target').textContent=q.target==='pop'?'目标风格：Pop（流行音乐）':'目标风格：Ambient（氛围音乐）';
 $('audio').pause();$('audio').src=`/audio/${encodeURIComponent(q.query_id)}?token=${encodeURIComponent(token)}`;
 document.querySelectorAll('input[name="answer"]').forEach(input=>input.checked=!!q.answer&&input.value===q.answer.answer);
 $('reason').value=q.answer?q.answer.reason:'';$('previous').disabled=index===0;$('next').disabled=index===state.questions.length-1;
 $('historyTitle').textContent=`本题保存历史（${q.history.length} 次）`;$('historyItems').replaceChildren();
 q.history.forEach(h=>{const p=document.createElement('div');p.className='history-item';p.textContent=`${h.saved_at} · ${labels[h.answer]}${h.reason?'\n'+h.reason:''}`;$('historyItems').append(p);});
 status(q.answer?'已载入先前回答；修改后请再次保存。':'本题尚未保存回答。');}
function move(next){if(busy)return;if(dirty&&!window.confirm('这题有未保存的修改。离开并放弃这些修改吗？')){$('jump').value=index;return;}index=next;paint();}
async function getState(){const response=await fetch('/api/questions',{headers:{'X-Review-Token':token},cache:'no-store'});if(!response.ok)throw new Error('无法读取本地听辨会话。');return response.json();}
$('previous').onclick=()=>move(Math.max(0,index-1));$('next').onclick=()=>move(Math.min(state.questions.length-1,index+1));
$('jump').onchange=()=>move(Number($('jump').value));$('resume').onclick=()=>{const n=state.questions.findIndex(q=>!q.answer);if(n<0)status('本轮全部题目已经保存。可以回看或修改。');else move(n);};
document.querySelectorAll('input[name="answer"]').forEach(input=>input.onchange=()=>dirty=true);$('reason').oninput=()=>dirty=true;
$('save').onclick=async()=>{if(busy||!state.questions.length)return;const answer=document.querySelector('input[name="answer"]:checked');if(!answer){status('请先选择“是”“否”或“不确定”。');return;}
 setBusy(true);const old=index;try{const response=await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify({query_id:state.questions[index].query_id,answer:answer.value,reason:$('reason').value})});if(!response.ok)throw new Error('保存失败，答案尚未写入。请重试。');state=await response.json();dirty=false;index=Math.min(old+1,state.questions.length-1);paint();if(state.completed_count===state.questions.length)status('本轮全部题目已经保存。可以回看或修改。');}catch(error){status(error.message);}finally{setBusy(false);}};
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
getState().then(data=>{state=data;const first=state.questions.findIndex(q=>!q.answer);index=first<0?0:first;paint();}).catch(error=>status(error.message));
</script></body></html>'''


class ReviewHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, format, *args):
        # Default access logs would expose the session token carried by audio URLs.
        pass

    @property
    def origin(self):
        return f'http://127.0.0.1:{self.server.server_address[1]}'

    def _send(self, status, body=b'', content_type='application/json; charset=utf-8', extra=None):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Frame-Options', 'DENY')
        if extra:
            for key, value in extra.items():
                self.send_header(key, value)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _json(self, status, value):
        self._send(status, json.dumps(value, ensure_ascii=False).encode('utf-8'))

    def _error(self, status, message):
        self._json(status, {'error': message})

    def _same_origin(self, post=False):
        if self.headers.get_all('Host', []) != [self.origin.removeprefix('http://')]:
            return False
        origins = self.headers.get_all('Origin', [])
        if (post and origins != [self.origin]) or (origins and origins != [self.origin]):
            return False
        fetch_site = self.headers.get('Sec-Fetch-Site')
        return fetch_site is None or fetch_site in ({'same-origin'} if post else {'same-origin', 'none'})

    def _token(self, supplied):
        return isinstance(supplied, str) and secrets.compare_digest(supplied, self.server.state.token)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._same_origin():
            self._error(403, 'Only this local review origin is allowed')
            return
        url = urlsplit(self.path)
        if url.path == '/' and not url.query:
            nonce = secrets.token_urlsafe(20)
            page = PAGE.replace('__NONCE__', nonce).replace('__TOKEN__', json.dumps(self.server.state.token)).encode('utf-8')
            csp = (f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
                   "connect-src 'self'; media-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            self._send(200, page, 'text/html; charset=utf-8', {'Content-Security-Policy': csp})
        elif url.path == '/api/questions' and not url.query:
            if not self._token(self.headers.get('X-Review-Token')):
                self._error(403, 'Invalid review session')
                return
            self._json(200, self.server.state.public_state())
        elif url.path.startswith('/audio/'):
            query = url.path.removeprefix('/audio/')
            params = parse_qs(url.query, keep_blank_values=True)
            if set(params) != {'token'} or len(params['token']) != 1 or not self._token(params['token'][0]):
                self._error(403, 'Invalid review session')
                return
            if not ID_PATTERN.fullmatch(query) or query not in self.server.state.by_id:
                self._error(404, 'Unknown review audio')
                return
            try:
                audio = self.server.state.audio(query)
            except (OSError, ValueError):
                self._error(409, 'Review audio is unavailable or changed')
                return
            try:
                if len(self.headers.get_all('Range', [])) > 1:
                    raise ValueError('Duplicate Range')
                start, end, partial = parse_byte_range(self.headers.get('Range'), len(audio))
            except ValueError:
                self._send(416, b'', extra={'Content-Range': f'bytes */{len(audio)}', 'Accept-Ranges': 'bytes'})
                return
            headers = {'Accept-Ranges': 'bytes'}
            if partial:
                headers['Content-Range'] = f'bytes {start}-{end}/{len(audio)}'
            self._send(206 if partial else 200, audio[start:end + 1], 'audio/wav', headers)
        else:
            self._error(404, 'Unknown review route')

    def do_POST(self):
        if not self._same_origin(post=True) or not self._token(self.headers.get('X-Review-Token')):
            self._error(403, 'Invalid local review origin or session')
            return
        if self.path != '/api/answer':
            self._error(404, 'Unknown review route')
            return
        lengths = self.headers.get_all('Content-Length', [])
        if (self.headers.get('Transfer-Encoding') is not None or len(lengths) != 1
                or not re.fullmatch(r'[0-9]{1,8}', lengths[0])
                or not 0 < int(lengths[0]) <= MAX_POST_BYTES):
            self._error(413, 'Invalid or oversized request body')
            return
        if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
            self._error(415, 'Send a JSON answer')
            return
        try:
            length = int(lengths[0])
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('Incomplete request body')
            payload = json.loads(raw)
            updated = self.server.state.answer(payload)
        except (ValueError, UnicodeError, TypeError, KeyError):
            self._error(400, 'Invalid answer; nothing was saved')
            return
        except OSError:
            self._error(500, 'Could not save the answer; please retry')
            return
        self._json(200, updated)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--answers', type=Path, required=True)
    parser.add_argument('--port', type=int, default=0, help='Loopback port; 0 chooses a free port')
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error('--port must be between 0 and 65535')
    state = ReviewState(args.manifest, args.answers)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), ReviewHandler)
    server.daemon_threads = True
    server.state = state
    print(f'Listening pilot: http://127.0.0.1:{server.server_address[1]}/', flush=True)
    print(f'{len(state.questions)} questions; answers: {state.answers_path}', flush=True)
    print('No automatic answers. Press Ctrl-C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
