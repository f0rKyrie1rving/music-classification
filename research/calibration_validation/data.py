"""Auditable new-data frame, artist sampling, bounded acquisition; no model scores."""
import argparse
import csv
import hashlib
import json
import re
import shutil
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from research.calibration.common import ROOT, read, save, digest, now, targets

HERE = Path(__file__).resolve().parent


def order_key(seed, kind, value):
    return hashlib.sha256(f'{seed}|{kind}|{value}'.encode()).hexdigest()


def read_licenses(path):
    result = {}
    for block in path.read_text().split('\n\n'):
        lines = block.strip().splitlines()
        if not lines:
            continue
        if not re.fullmatch(r'\d{2}/\d+\.mp3', lines[0]):
            raise ValueError('Unexpected license block')
        urls = re.findall(r'https?://creativecommons.org/licenses/([^/\s]+)/[^\s]+', block)
        if len(urls) > 1 or lines[0] in result:
            raise ValueError('Ambiguous license')
        result[lines[0]] = {'license_code': urls[0] if urls else None, 'attribution_verbatim': block.strip()}
    return result


def audit(out):
    if (out / 'candidate_pool.json').exists():
        raise FileExistsError('Candidate audit already exists')
    config = read(HERE / 'config.json')
    inputs = {}
    for name, expected in config['source_blobs'].items():
        path = out / 'source' / name
        data = path.read_bytes()
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if blob != expected:
            raise ValueError('Official metadata Git blob mismatch: ' + name)
        inputs[name] = {'sha256': digest(path), 'git_blob': blob, 'bytes': len(data)}
    historical = read(ROOT / 'data/expanded_manifest.json')['tracks']
    used_ids = {r['track_id'] for r in historical}
    used_artists = {r['artist_id'] for r in historical}
    if len(used_ids) != 1535 or len(used_artists) != 583:
        raise ValueError('Unexpected historical exclusion counts')
    licenses = read_licenses(out / 'source/licenses.txt')
    checksums = {}
    for line in (out / 'source/checksums.txt').read_text().splitlines():
        sha, name = line.split()
        if name in checksums:
            raise ValueError('Duplicate checksum member')
        checksums[name] = sha
    candidates = []
    counts = Counter()
    seen = set()
    with (out / 'source/genres.tsv').open(newline='') as stream:
        for values in list(csv.reader(stream, delimiter='\t'))[1:]:
            track_id, artist_id, album_id, path, duration, *tags = values
            counts['all_genre_rows'] += 1
            if track_id in seen:
                raise ValueError('Duplicate official track ID')
            seen.add(track_id)
            if path[:2] not in config['archives']:
                counts['outside_fixed_archives'] += 1
                continue
            counts['in_fixed_archives'] += 1
            if track_id in used_ids or artist_id in used_artists:
                counts['excluded_historical_track_or_artist'] += 1
                continue
            if path not in licenses or licenses[path]['license_code'] not in config['allowed_license_codes']:
                counts['excluded_license'] += 1
                continue
            member = path.replace('.mp3', '.low.mp3')
            if float(duration) < 30 or member not in checksums:
                counts['excluded_duration_or_missing_checksum'] += 1
                continue
            if not all(t.startswith('genre---') for t in tags):
                raise ValueError('Unexpected tag namespace')
            candidates.append({'track_id': track_id, 'artist_id': artist_id,
                'album_id': album_id, 'path': path, 'duration': float(duration),
                'tags': [t.removeprefix('genre---') for t in tags],
                'archive_member': member, 'expected_full_sha256': checksums[member],
                **licenses[path]})
    candidates.sort(key=lambda r: r['track_id'])
    counts['eligible_tracks'] = len(candidates)
    counts['eligible_artists'] = len({r['artist_id'] for r in candidates})
    if counts['eligible_artists'] < config['artist_count']:
        raise ValueError(f'Insufficient frame artists: {dict(counts)}; do not silently expand frame')
    cached = set()
    for directory in ['data/training_audio', 'data/expanded_audio', 'data/raw', 'data/previews', 'data/source/audio_prefixes']:
        for path in (ROOT / directory).glob('*'):
            match = re.search(r'track_\d{7}', path.name)
            if match:
                cached.add(match[0])
    if cached - used_ids:
        raise ValueError('Unaccounted local audio IDs require exclusion audit')
    exclusion = {'tracks': sorted(used_ids), 'artists': sorted(used_artists),
        'source': 'data/expanded_manifest.json',
        'source_sha256': digest(ROOT / 'data/expanded_manifest.json'),
        'limits': 'Project-record exclusion; unrecorded external exposure and encoder pretraining cannot be ruled out.'}
    save(out / 'exclusions.json', exclusion)
    save(out / 'candidate_pool.json', {'tracks': candidates})
    save(out / 'data_audit.json', {'created_utc': now(), 'counts': dict(counts),
        'source_commit': config['source_commit'], 'source_files': inputs,
        'historical_tracks': len(used_ids), 'historical_artists': len(used_artists),
        'local_audio_ids': len(cached), 'local_audio_ids_outside_history': len(cached-used_ids),
        'eligible_positive_tracks': targets(candidates).sum(0).tolist(),
        'no_predictions_computed': True})
    shutil.copy2(HERE / 'config.json', out / 'config.json')
    print('New-data frame:', dict(counts), flush=True)


def index_one(out, shard):
    from prepare_dataset import get_range
    destination = out / f'archive_index_{shard}.json'
    wanted = {r['archive_member'] for r in read(out / 'candidate_pool.json')['tracks'] if r['path'][:2] == shard}
    url = f'https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low/raw_30s_audio-low-{shard}.tar'
    state = read(destination) if destination.exists() else {'archive_url': url,
        'next_offset': 0, 'headers_scanned': 0, 'members': {}, 'complete': False}
    while not state['complete']:
        offset = state['next_offset']
        block, total = get_range(url, offset, 512)
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
        if state['headers_scanned'] % 25 == 0:
            save(destination, state)
        if state['headers_scanned'] % 100 == 0:
            print('Index', shard, state['headers_scanned'], 'headers;', len(state['members']), 'candidates', flush=True)
    save(destination, state)
    if set(state['members']) != wanted:
        raise ValueError('Official archive does not cover the fixed candidate frame')
    print('Indexed', shard, len(wanted), 'candidates', flush=True)


def index(out):
    config = read(out / 'config.json')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda shard: index_one(out, shard), config['archives']):
            pass


def sample(out):
    if (out / 'selected_manifest.json').exists():
        raise FileExistsError('Sample already selected')
    config = read(out / 'config.json')
    records = read(out / 'candidate_pool.json')['tracks']
    artists = sorted({r['artist_id'] for r in records}, key=lambda a: order_key(config['sample_seed'], 'artist', a))
    selected = []
    for artist in artists[:config['artist_count']]:
        songs = sorted([r for r in records if r['artist_id'] == artist],
                       key=lambda r: order_key(config['sample_seed'], 'track', r['track_id']))
        for row in songs[:config['tracks_per_artist']]:
            shard = row['path'][:2]
            idx = read(out / f'archive_index_{shard}.json')
            if not idx['complete']:
                raise ValueError('Index incomplete')
            entry = idx['members'][row['archive_member']]
            selected.append({**row, 'archive': {**entry, 'archive_url': idx['archive_url'],
                'archive_bytes': idx['archive_bytes']},
                'audio_file': str((out / 'audio' / f"{row['track_id']}.wav").relative_to(ROOT))})
    if len({r['artist_id'] for r in selected}) != config['artist_count']:
        raise ValueError('Incomplete artist selection')
    save(out / 'selected_manifest.json', {'created_utc': now(), 'tracks': selected,
        'sampling': '200 hash-ordered artists; up to two hash-ordered tracks each; no label quotas or replacement'})
    print('Fixed sample:', len(selected), 'tracks;', config['artist_count'], 'artists', flush=True)


def download(out):
    import prepare_dataset
    from .evaluate import verify_frozen
    verify_frozen(out)
    if (out / 'download_status.json').exists():
        raise FileExistsError('Acquisition status already exists; preserve all failures')
    # Reuse the reviewed bounded downloader; all new writes go in this run folder.
    prepare_dataset.SOURCE = out / 'acquisition'
    prepare_dataset.SOURCE.mkdir(exist_ok=True)
    rows = read(out / 'selected_manifest.json')['tracks']
    successful, failures = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(prepare_dataset.download_track, row): row for row in rows}
        for future in as_completed(jobs):
            row = jobs[future]
            try:
                receipt = future.result()
                successful.append({'track_id': row['track_id'], **receipt})
            except Exception as error:
                failures.append({'track_id': row['track_id'], 'error': repr(error)})
            if (len(successful)+len(failures)) % 20 == 0:
                print('Downloaded', len(successful), '/', len(rows), '; failures', len(failures), flush=True)
    save(out / 'download_status.json', {'created_utc': now(), 'requested': len(rows),
        'successful': successful, 'failures': failures, 'replacement_tracks': 0})
    verify_frozen(out)
    print('Acquisition completed:', len(successful), 'successes;', len(failures), 'failures', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'index', 'sample', 'download'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.out.resolve())
