"""Local-only, label-independent confirmation frame and pre-network selection."""
import argparse
from collections import Counter
import csv
import hashlib
from pathlib import Path
import re
import shutil

from evaluate_application_correction import ROOT, digest, read, targets
from research.application_fresh_validation.data import now, order_key, read_licenses
from research.application_fresh_validation.pipeline import atomic_json, validate_ids

HERE = Path(__file__).resolve().parent
METADATA = 'outputs/application_fresh_validation/20260927_v1/source'


def select_rows(records, config, out):
    """Selection uses IDs only: changing labels or input order cannot change it."""
    validate_ids(records)
    artists = sorted({r['artist_id'] for r in records},
                     key=lambda a: order_key(config['sample_seed'], 'artist', a))
    if len(artists) < config['artist_count']:
        raise ValueError('Insufficient artists; do not expand or replace the fixed frame')
    selected = []
    for artist in artists[:config['artist_count']]:
        songs = sorted((r for r in records if r['artist_id'] == artist),
                       key=lambda r: order_key(config['sample_seed'], 'track', r['track_id']))
        for row in songs[:config['tracks_per_artist']]:
            selected.append({**row, 'audio_file': str(
                (out / 'audio' / (row['track_id'] + '.wav')).relative_to(ROOT))})
    return selected


def build_frame(source, config, excluded):
    """Validate official source blobs and enumerate every eligible frame member."""
    inputs = {}
    for name, expected in config['source_blobs'].items():
        path = source / name
        content = path.read_bytes()
        blob = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        if blob != expected:
            raise ValueError('Official metadata Git blob mismatch: ' + name)
        inputs[name] = {'sha256': digest(path), 'git_blob': blob, 'bytes': len(content)}
    licenses = read_licenses(source / 'licenses.txt')
    checksums = {}
    for line in (source / 'checksums.txt').read_text().splitlines():
        sha, name = line.split()
        if name in checksums or not re.fullmatch('[a-f0-9]{64}', sha):
            raise ValueError('Invalid or duplicate official checksum')
        checksums[name] = sha
    old_ids, old_artists = set(excluded['tracks']), set(excluded['artists'])
    counts, seen, candidates = Counter(), set(), []
    with (source / 'genres.tsv').open(newline='') as stream:
        reader = csv.reader(stream, delimiter='\t')
        next(reader)
        for values in reader:
            track_id, artist_id, album_id, path, duration, *tags = values
            counts['all_genre_rows'] += 1
            if track_id in seen:
                raise ValueError('Duplicate official track ID')
            seen.add(track_id)
            if path[:2] not in config['archives']:
                counts['outside_fixed_archives'] += 1
                continue
            if not re.fullmatch(r'\d{2}/\d+\.mp3', path):
                raise ValueError('Unsafe official audio path')
            counts['in_fixed_archives'] += 1
            if track_id in old_ids or artist_id in old_artists:
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
    candidates.sort(key=lambda row: row['track_id'])
    validate_ids(candidates)
    counts['eligible_tracks'] = len(candidates)
    counts['eligible_artists'] = len({r['artist_id'] for r in candidates})
    for key, value in [('expected_frame_tracks', len(candidates)),
                       ('expected_frame_artists', counts['eligible_artists'])]:
        if value != config[key]:
            raise ValueError(f'Fixed frame count differs: {key}={value}')
    return candidates, dict(counts), inputs


def prepare(out):
    """No network, audio extraction, prediction, or archive-offset lookup occurs."""
    out = Path(out)
    if (out / 'selected_manifest.json').exists() or (out / 'freeze.json').exists():
        raise FileExistsError('Selection is already fixed; do not prepare it again')
    config = read(HERE / 'config.json')
    excluded_path = ROOT / config['historical_exclusions']
    excluded = read(excluded_path)
    for key, expected in [('tracks', config['expected_excluded_tracks']),
                          ('artists', config['expected_excluded_artists'])]:
        if len(excluded[key]) != expected or len(set(excluded[key])) != expected:
            raise ValueError('Historical exclusion count or uniqueness differs')
    source = ROOT / config.get('metadata_source', METADATA)
    candidates, counts, inputs = build_frame(source, config, excluded)
    selected = select_rows(candidates, config, out)
    cached = set()
    directories = ['data/training_audio', 'data/expanded_audio', 'data/raw', 'data/previews',
        'data/source/audio_prefixes', 'outputs/calibration_validation/20260927_v1/audio',
        'outputs/calibration_validation/20260927_v1/acquisition/audio_prefixes',
        'outputs/application_fresh_validation/20260927_v1/audio',
        'outputs/application_fresh_validation/20260927_v1/acquisition/audio_prefixes']
    for directory in directories:
        for path in (ROOT / directory).glob('*'):
            match = re.search(r'track_\d{7}', path.name)
            if match:
                cached.add(match[0])
    if cached - set(excluded['tracks']):
        raise ValueError('Unaccounted local audio IDs require exclusion audit')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'source').mkdir(exist_ok=True)
    for name in config['source_blobs']:
        destination = out / 'source' / name
        if destination.exists() and digest(destination) != inputs[name]['sha256']:
            raise ValueError('Existing copied source differs')
        shutil.copy2(source / name, destination)
    atomic_json(out / 'config.json', config)
    atomic_json(out / 'exclusions.json', {'tracks': excluded['tracks'], 'artists': excluded['artists'],
        'source_hashes': {config['historical_exclusions']: digest(excluded_path)},
        'limits': config['evaluation_scope']})
    atomic_json(out / 'candidate_pool.json', {'tracks': candidates})
    atomic_json(out / 'selected_manifest.json', {'created_utc': now(), 'tracks': selected,
        'sampling': f"First {config['artist_count']} hash-ordered artists; up to {config['tracks_per_artist']} hash-ordered tracks each; no labels, quotas, replacement or offsets"})
    atomic_json(out / 'future_exclusions.json', {
        'tracks': sorted(set(excluded['tracks']) | {r['track_id'] for r in candidates}),
        'artists': sorted(set(excluded['artists']) | {r['artist_id'] for r in candidates}),
        'scope': 'All prior frames plus this entire eligible frame, including unselected and failed tracks; no future fresh reuse.',
        'historical_exclusions_sha256': digest(excluded_path),
        'candidate_pool_sha256': digest(out / 'candidate_pool.json')})
    atomic_json(out / 'data_audit.json', {'created_utc': now(), 'counts': counts,
        'source_commit': config['source_commit'], 'source_files': inputs,
        'historical_tracks': len(excluded['tracks']), 'historical_artists': len(excluded['artists']),
        'selected_tracks': len(selected), 'selected_artists': len({r['artist_id'] for r in selected}),
        'local_audio_ids': len(cached), 'local_audio_ids_outside_history': 0,
        'eligible_positive_tracks': targets(candidates, read(ROOT / 'experiments/final_holdout_plan.json')).sum(0).tolist(),
        'no_predictions_computed': True, 'no_archive_requests_made': True})
    print('Fixed local-only sample:', len(selected), 'tracks;', config['artist_count'], 'artists', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    prepare(parser.parse_args().out.resolve())
