"""Frozen-cohort acquisition with bounded requests, logged retries and no replacement."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid

from evaluate_application_correction import ROOT, digest, read
from research.application_fresh_validation.data import download_one, now
from research.application_fresh_validation.pipeline import atomic_json, immutable_json, validate_ids

CONTEXT = threading.local()


def save_header(out, shard, offset, block):
    if not re.fullmatch(r'\d{2}', shard) or type(offset) is not int or offset < 0 or len(block) != 512:
        raise ValueError('Invalid raw TAR header')
    path = out / 'index_headers' / shard / f'{offset}.bin'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != block:
            raise ValueError('Previously acquired archive header changed')
    else:
        with path.open('xb') as stream:
            stream.write(block)
    return str(path.relative_to(out)), hashlib.sha256(block).hexdigest()


def verify_frozen(out):
    from .pipeline import verify_frozen as check
    return check(out)


@contextmanager
def process_lock(out, name):
    path = out / (name + '.lock')
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Process marker exists; inspect before resuming: ' + str(path)) from None
    os.close(descriptor)
    try:
        yield
    finally:
        path.unlink()


def get_range(out, url, start, size):
    """Same three-attempt bounded transport as prepare_dataset, with durable audits."""
    if type(start) is not int or type(size) is not int or start < 0 or not 0 < size <= 65536:
        raise ValueError('Unsafe byte range')
    config = read(out / 'config.json')['network']
    record = {'url': url, 'range': [start, start + size - 1],
              'context': getattr(CONTEXT, 'value', {'phase': 'unknown'}), 'created_utc': now(), 'attempts': []}
    path = out / 'network_requests' / (uuid.uuid4().hex + '.json')
    atomic_json(path, record)
    for attempt in range(config['attempts_per_range']):
        entry = {'attempt': attempt + 1, 'started_utc': now(), 'state': 'started'}
        record['attempts'].append(entry)
        atomic_json(path, record)
        with tempfile.TemporaryDirectory(prefix='music-confirm-range-') as tmp:
            body, headers = Path(tmp) / 'body', Path(tmp) / 'headers'
            result = subprocess.run(['/usr/bin/curl', '--fail', '--location', '--silent', '--show-error',
                '--connect-timeout', str(config['connect_timeout_seconds']),
                '--max-time', str(config['request_timeout_seconds']), '--range',
                f'{start}-{start + size - 1}', '--max-filesize', str(size),
                '--dump-header', str(headers), '--output', str(body), url], capture_output=True, text=True)
            error = result.stderr.strip()
            if result.returncode == 0 and body.exists() and headers.exists():
                header = headers.read_text()
                matches = re.findall(r'content-range:\s*bytes (\d+)-(\d+)/(\d+)', header, re.I)
                statuses = re.findall(r'^HTTP/\S+\s+(\d{3})', header, re.M)
                content = body.read_bytes()
                if (matches and statuses and statuses[-1] == '206'
                        and tuple(map(int, matches[-1][:2])) == (start, start + size - 1)
                        and len(content) == size and int(matches[-1][2]) >= start + size):
                    entry.update(state='success', completed_utc=now(), response_bytes=len(content),
                                 archive_bytes=int(matches[-1][2]), sha256=hashlib.sha256(content).hexdigest())
                    if record['context'].get('phase') == 'index':
                        name, _ = save_header(out, record['context']['shard'], start, content)
                        entry['index_header_file'] = name
                    atomic_json(path, record)
                    return content, int(matches[-1][2])
                error = 'Server did not return exact HTTP 206 bytes and Content-Range'
            entry.update(state='failed', completed_utc=now(), error=error or 'No valid response body')
            atomic_json(path, record)
        time.sleep(attempt + 1)
    raise RuntimeError('Range download failed: ' + record['attempts'][-1]['error'])


def index_one(out, shard):
    CONTEXT.value = {'phase': 'index', 'shard': shard}
    destination = out / f'archive_index_{shard}.json'
    wanted = {r['archive_member'] for r in read(out / 'candidate_pool.json')['tracks'] if r['path'][:2] == shard}
    url = f'https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low/raw_30s_audio-low-{shard}.tar'
    identity = {'freeze_sha256': digest(out / 'freeze.json'),
                'selection_sha256': digest(out / 'selected_manifest.json')}
    state = read(destination) if destination.exists() else {'archive_url': url, 'next_offset': 0,
        'headers_scanned': 0, 'members': {}, 'header_hashes': {}, 'complete': False,
        'identity': identity, 'invocations': []}
    if state['identity'] != identity or state['archive_url'] != url:
        raise ValueError('Stale index identity')
    if state['complete']:
        if set(state['members']) != wanted:
            raise ValueError('Complete index does not cover fixed frame')
        return state
    if len(state['invocations']) >= read(out / 'config.json')['network']['maximum_index_invocations']:
        raise ValueError('Fixed maximum index invocations exhausted')
    invocation = {'started_utc': now(), 'offset': state['next_offset'], 'state': 'started'}
    state['invocations'].append(invocation)
    atomic_json(destination, state)
    try:
        while not state['complete']:
            offset = state['next_offset']
            block, total = get_range(out, url, offset, 512)
            name, header_sha = save_header(out, shard, offset, block)
            state['header_hashes'][name] = header_sha
            if state.get('archive_bytes', total) != total:
                raise ValueError('Archive size changed')
            state['archive_bytes'] = total
            if not block.strip(b'\0'):
                state['complete'] = True
                break
            member = tarfile.TarInfo.frombuf(block, 'utf-8', 'strict')
            if member.size < 0:
                raise ValueError('Negative TAR member size')
            if member.isfile() and member.name in wanted:
                if member.name in state['members']:
                    raise ValueError('Duplicate indexed member')
                state['members'][member.name] = {'offset': offset + 512, 'size': member.size}
            state['next_offset'] = offset + 512 + ((member.size + 511) // 512) * 512
            state['headers_scanned'] += 1
            if state['headers_scanned'] > 2000 or state['next_offset'] > total or state['next_offset'] <= offset:
                raise ValueError('Unexpected TAR geometry')
            atomic_json(destination, state)
            if state['headers_scanned'] % 100 == 0:
                print('Index', shard, state['headers_scanned'], 'headers;', len(state['members']), 'frame tracks', flush=True)
        if set(state['members']) != wanted:
            raise ValueError('Official archive does not cover the complete fixed frame')
        invocation.update(state='success', completed_utc=now())
    except Exception as error:
        invocation.update(state='failed', completed_utc=now(), error=repr(error))
        raise
    finally:
        atomic_json(destination, state)
    print('Indexed', shard, len(wanted), 'frame tracks', flush=True)
    return state


def index(out):
    verify_frozen(out)
    with process_lock(out, 'index'):
        config = read(out / 'config.json')
        errors = []
        with ThreadPoolExecutor(max_workers=config['network']['index_workers']) as pool:
            jobs = {pool.submit(index_one, out, shard): shard for shard in config['archives']}
            for job in as_completed(jobs):
                try:
                    job.result()
                except Exception as error:
                    errors.append({'shard': jobs[job], 'error': repr(error)})
        if errors:
            raise RuntimeError('Index failures persisted; bounded resume is allowed: ' + str(errors))
    verify_frozen(out)


def resolve(out):
    verify_frozen(out)
    selected = read(out / 'selected_manifest.json')['tracks']
    validate_ids(selected)
    rows, indices = [], {}
    for shard in read(out / 'config.json')['archives']:
        path = out / f'archive_index_{shard}.json'
        idx = read(path)
        wanted = {r['archive_member'] for r in read(out / 'candidate_pool.json')['tracks'] if r['path'][:2] == shard}
        if (not idx['complete'] or set(idx['members']) != wanted
                or idx['identity'] != {'freeze_sha256': digest(out / 'freeze.json'),
                                      'selection_sha256': digest(out / 'selected_manifest.json')}):
            raise ValueError('Incomplete or stale archive index')
        indices[shard] = idx
    for row in selected:
        idx = indices[row['path'][:2]]
        rows.append({**row, 'archive': {**idx['members'][row['archive_member']],
            'archive_url': idx['archive_url'], 'archive_bytes': idx['archive_bytes']}})
    result = {'tracks': rows, 'freeze_sha256': digest(out / 'freeze.json'),
              'selection_sha256': digest(out / 'selected_manifest.json'),
              'index_hashes': {f'archive_index_{s}.json': digest(out / f'archive_index_{s}.json') for s in indices}}
    immutable_json(out / 'resolved_manifest.json', result)
    return result


def download(out):
    import prepare_dataset
    verify_frozen(out)
    resolved = resolve(out)
    if (out / 'download_status.json').exists():
        raise FileExistsError('Acquisition is complete; all failures remain fixed')
    with process_lock(out, 'download'):
        old_source, old_get_range = prepare_dataset.SOURCE, prepare_dataset.get_range
        prepare_dataset.SOURCE = out / 'acquisition'
        prepare_dataset.SOURCE.mkdir(exist_ok=True)
        prepare_dataset.get_range = lambda url, start, size: get_range(out, url, start, size)
        try:
            rows = resolved['tracks']
            identity = {'freeze_sha256': digest(out / 'freeze.json'),
                        'selection_sha256': digest(out / 'selected_manifest.json'),
                        'resolved_sha256': digest(out / 'resolved_manifest.json')}
            def one(row):
                CONTEXT.value = {'phase': 'audio', 'track_id': row['track_id']}
                return download_one(out, row, identity, prepare_dataset.download_track)
            results = {}
            workers = read(out / 'config.json')['network']['download_workers']
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for job in as_completed([pool.submit(one, row) for row in rows]):
                    result = job.result()
                    results[result['track_id']] = result
                    if len(results) % 20 == 0:
                        print('Acquired', len(results), '/', len(rows), '; failures',
                              sum(r['state'] == 'failed' for r in results.values()), flush=True)
            ordered = [results[r['track_id']] for r in rows]
            atomic_json(out / 'download_status.json', {'created_utc': now(), 'requested': len(rows),
                'successful': [r['receipt'] for r in ordered if r['state'] == 'success'],
                'failures': [{'track_id': r['track_id'], 'error': r['error']} for r in ordered if r['state'] == 'failed'],
                'replacement_tracks': 0, 'identity': identity,
                'checkpoint_policy': 'One downloader invocation per track; completed success/failure immutable; interrupted started requests fail; every range attempt logged.'})
        finally:
            prepare_dataset.SOURCE, prepare_dataset.get_range = old_source, old_get_range
    verify_frozen(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['index', 'resolve', 'download'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.out.resolve())
