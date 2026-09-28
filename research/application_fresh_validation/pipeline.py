"""Freeze the released application and extract a new, fixed audio cohort safely."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from evaluate_application_correction import ROOT, digest, read, targets as target_builder

HERE = Path(__file__).resolve().parent
WIDTH = 2304
BASE_INPUTS = [
    'artifacts/score_correction.json', 'artifacts/final_heads.npy', 'artifacts/final_heads.json',
    'experiments/final_holdout_plan.json', 'data/expanded_manifest.json', 'app_version.txt',
    'outputs/maest_hf/features.npy', 'outputs/maest_hf/features.json',
    'models/mtg-upf-maest-519l/SOURCES.json',
]
BASE_SOURCES = [
    'prepare_dataset.py', 'maest_hf_features.py', 'prepare_maest_hf.py', 'prepare_mert.py',
    'predict_app.py', 'score_correction.py', 'predict_maest.py',
    'evaluate_application_correction.py',
]
LOCAL_FILES = [
    'config.json', 'protocol.md', 'exclusions.json', 'candidate_pool.json', 'data_audit.json',
    'selected_manifest.json', 'exposure_history.json',
    'metadata_amendment.json',
    'independent_audit.json',
]


def now():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def targets(rows):
    return target_builder(rows, read(ROOT / 'experiments/final_holdout_plan.json'))


def safe_file(base, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Expected a safe relative path: {name}')
    result = base / path
    if not result.resolve().is_relative_to(base.resolve()):
        raise ValueError(f'Frozen path escapes its root: {name}')
    return result


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def immutable_json(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Completed receipt differs: {path}')
    else:
        atomic_json(path, value)


def save_vector(path, vector):
    """An orphan array can be adopted only after an identical recomputation."""
    if path.exists():
        previous = np.load(path, allow_pickle=False)
        if previous.dtype != vector.dtype or not np.array_equal(previous, vector):
            raise ValueError(f'Existing feature array differs: {path}')
        return
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, vector, allow_pickle=False)
    temporary.replace(path)


def validate_ids(rows):
    ids = [r['track_id'] for r in rows]
    if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch(r'track_[0-9]+', i) for i in ids):
        raise ValueError('Empty, duplicate or unsafe selected IDs')
    if any(not re.fullmatch(r'artist_[0-9]+', r['artist_id']) for r in rows):
        raise ValueError('Invalid selected artist ID')
    return ids


def source_paths():
    sources = list(HERE.glob('*.py')) + [ROOT / name for name in BASE_SOURCES]
    sources += [HERE / 'config.json', HERE / 'protocol.md']
    return sorted(set(sources))


def expected_selection_ids(candidates, config):
    """Independently rebuild the declared artist-first selection and exact order."""
    def key(kind, value):
        return hashlib.sha256(f"{config['sample_seed']}|{kind}|{value}".encode()).hexdigest()
    artists = sorted({r['artist_id'] for r in candidates}, key=lambda a: key('artist', a))
    ids = []
    for artist in artists[:config['artist_count']]:
        tracks = sorted((r['track_id'] for r in candidates if r['artist_id'] == artist),
                        key=lambda track: key('track', track))
        ids.extend(tracks[:config['tracks_per_artist']])
    return ids


def verify_runtime(frozen):
    for versions in (frozen['runtime'], frozen['evaluation_runtime']):
        if any(importlib.metadata.version(package) != expected for package, expected in versions.items()):
            raise ValueError('Current runtime differs from frozen extraction/evaluation dependencies')


def freeze(out):
    out = Path(out)
    if (out / 'freeze.json').exists():
        raise FileExistsError('Application validation is already frozen')
    for name in ('download_status.json', 'features.npy', 'features.json', 'manifest.json',
                 'extraction_status.json', 'runtime_control.json', 'predictions.npz'):
        if (out / name).exists():
            raise ValueError(f'Acquisition or inference preceded freezing: {name}')
    for folder in ('audio', 'acquisition', 'feature_tracks'):
        if (out / folder).exists() and any((out / folder).iterdir()):
            raise ValueError(f'Acquisition or extraction preceded freezing: {folder}')
    config = read(out / 'config.json')
    if config != read(HERE / 'config.json') or config['maximum_prefix_bytes_per_track'] != 20 * 65536:
        raise ValueError('Config differs from source or bounded downloader settings')
    selected = read(out / 'selected_manifest.json')['tracks']
    ids = validate_ids(selected)
    excluded = read(out / 'exclusions.json')
    candidates = read(out / 'candidate_pool.json')['tracks']
    validate_ids(candidates)
    if ids != expected_selection_ids(candidates, config):
        raise ValueError('Selected IDs/order differ from the complete fixed hash selection')
    by_id = {r['track_id']: r for r in candidates}
    artists = [r['artist_id'] for r in selected]
    if (len(set(artists)) != config['artist_count']
            or max(Counter(artists).values()) > config['tracks_per_artist']
            or set(ids) & set(excluded['tracks']) or set(artists) & set(excluded['artists'])):
        raise ValueError('Selected sample violates count or historical exclusion rules')
    for row in selected:
        if row['track_id'] not in by_id or any(row[k] != v for k, v in by_id[row['track_id']].items()):
            raise ValueError('Selected metadata differs from candidate pool')
        idx = read(out / f"archive_index_{row['path'][:2]}.json")
        expected = {**idx['members'][row['archive_member']],
                    'archive_url': idx['archive_url'], 'archive_bytes': idx['archive_bytes']}
        if not idx['complete'] or row['archive'] != expected:
            raise ValueError('Selected archive location differs from complete index')
        if safe_file(ROOT, row['audio_file']).resolve() != (out / 'audio' / f"{row['track_id']}.wav").resolve():
            raise ValueError('Selected audio path is outside this run')
    # Loading these small arrays validates the real released model; no audio is scored.
    from predict_maest import load_bundle, METADATA, WEIGHTS, PLAN
    from score_correction import load_correction
    bundle = load_bundle(METADATA, WEIGHTS)
    load_correction(bundle, METADATA, WEIGHTS, PLAN)
    plan = read(ROOT / 'experiments/final_holdout_plan.json')
    previous = read(ROOT / 'outputs/maest_hf/features.json')
    runtime = {p: importlib.metadata.version(p) for p in previous['versions']}
    if runtime != previous['versions']:
        raise ValueError('Current encoder runtime differs from the historical control runtime')
    historical = read(ROOT / 'data/expanded_manifest.json')['tracks']
    old_by_id = {r['track_id']: r for r in historical}
    if (previous['ids'] != plan['fit_ids'] or len(previous['ids']) != 1206
            or previous['feature_sha256'] != digest(ROOT / 'outputs/maest_hf/features.npy')
            or plan['expanded_manifest_sha256'] != digest(ROOT / 'data/expanded_manifest.json')):
        raise ValueError('Historical control cache or final-model training data changed')
    control_path = old_by_id[previous['ids'][0]]['audio_file']
    if digest(safe_file(ROOT, control_path)) != previous['audio_hashes'][0]:
        raise ValueError('Historical control audio changed')
    encoder = read(ROOT / 'models/mtg-upf-maest-519l/SOURCES.json')
    inputs = BASE_INPUTS + [control_path]
    for name, expected_hash in excluded['source_hashes'].items():
        if digest(safe_file(ROOT, name)) != expected_hash:
            raise ValueError(f'Historical exposure source changed: {name}')
        inputs.append(name)
    for name, expected in encoder['files_sha256'].items():
        name = 'models/mtg-upf-maest-519l/' + name
        if digest(safe_file(ROOT, name)) != expected:
            raise ValueError(f'Encoder file differs from its pinned receipt: {name}')
        inputs.append(name)
    shutil.copy2(HERE / 'protocol.md', out / 'protocol.md')
    local = LOCAL_FILES + [f'archive_index_{s}.json' for s in config['archives']]
    local += [f'source/{name}' for name in config['source_blobs']]
    sources = source_paths()
    for required in ('data.py', 'pipeline.py', 'evaluate.py', 'test_pipeline.py'):
        if not (HERE / required).is_file():
            raise FileNotFoundError(f'Finish all implementation before freezing: {required}')
    for source in sources:
        dest = out / 'source_snapshot' / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    atomic_json(out / 'freeze.json', {
        'created_utc': now(), 'git_commit': git('rev-parse', 'HEAD'),
        'stage': 'before new audio acquisition, extraction and application predictions',
        'model': 'actual released final classifier fitted on 1206 development tracks',
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'input_hashes': {name: digest(safe_file(ROOT, name)) for name in inputs},
        'local_hashes': {name: digest(safe_file(out, name)) for name in local},
        'runtime': runtime,
        'evaluation_runtime': {p: importlib.metadata.version(p) for p in ('scipy', 'scikit-learn')},
        'control_track_id': previous['ids'][0], 'control_audio_file': control_path,
        'feature_width': WIDTH, 'device': 'cpu', 'refitted': False,
    })
    verify_frozen(out)
    print('Frozen released application, source and sample:', len(ids), 'tracks', flush=True)


def verify_frozen(out):
    out = Path(out)
    frozen = read(out / 'freeze.json')
    for key in ('source_hashes', 'input_hashes', 'local_hashes'):
        base = out if key == 'local_hashes' else ROOT
        for name, expected in frozen[key].items():
            if digest(safe_file(base, name)) != expected:
                raise ValueError(f'Changed frozen {key}: {name}')
    if not set(BASE_INPUTS).issubset(frozen['input_hashes']):
        raise ValueError('Frozen receipt omits actual application inputs')
    if not set(LOCAL_FILES).issubset(frozen['local_hashes']):
        raise ValueError('Frozen receipt omits sample provenance')
    if set(frozen['source_hashes']) != {str(p.relative_to(ROOT)) for p in source_paths()}:
        raise ValueError('Frozen source list differs from all current implementation files')
    if frozen['refitted'] is not False or frozen['device'] != 'cpu' or frozen['feature_width'] != WIDTH:
        raise ValueError('Unexpected frozen application/encoder policy')
    verify_runtime(frozen)
    return frozen


def acquisition_rows(out, selected):
    """Account for each requested ID exactly once; no replacement or silent retry."""
    acquisition = read(out / 'download_status.json')
    ids = validate_ids(selected)
    success_list, failures = acquisition['successful'], acquisition['failures']
    success = {r['track_id']: r for r in success_list}
    failed = {r['track_id']: r for r in failures}
    if (len(success) != len(success_list) or len(failed) != len(failures)
            or set(success) & set(failed) or set(success) | set(failed) != set(ids)
            or acquisition['requested'] != len(ids) or acquisition['replacement_tracks'] != 0
            or any(not isinstance(r.get('error'), str) or not r['error'] for r in failures)):
        raise ValueError('Acquisition IDs, counts or failures do not reconcile')
    return acquisition, success, failed


def verify_audio_receipt(out, row, expected, config):
    import soundfile as sf
    path = safe_file(ROOT, row['audio_file'])
    if path.resolve() != (out / 'audio' / f"{row['track_id']}.wav").resolve():
        raise ValueError('Acquired audio is outside the frozen run')
    receipt_path = out / 'acquisition/download_receipts' / f"{row['track_id']}.json"
    receipt = read(receipt_path)
    if receipt != expected or receipt['track_id'] != row['track_id'] or digest(path) != receipt['wav_sha256']:
        raise ValueError('Acquired audio or original download receipt changed')
    byte_ranges = receipt['byte_ranges']
    total, previous_end = 0, row['archive']['offset'] - 1
    for item in byte_ranges:
        start, end = item['byte_range']
        cached = safe_file(ROOT, item['cache_file'])
        if not cached.resolve().is_relative_to((out / 'acquisition/audio_prefixes').resolve()):
            raise ValueError('Encoded bytes are outside the acquisition cache')
        if (type(start) is not int or type(end) is not int or start > end or start <= previous_end
                or start < row['archive']['offset']
                or end >= row['archive']['offset'] + row['archive']['size']
                or cached.stat().st_size != end - start + 1 or digest(cached) != item['sha256']):
            raise ValueError('Encoded byte range or hash changed')
        total += end - start + 1
        previous_end = end
    if (not byte_ranges or total != receipt['downloaded_bytes']
            or not 0 < total <= config['maximum_prefix_bytes_per_track']
            or byte_ranges[0]['byte_range'][0] != row['archive']['offset']
            or receipt['prefix_sha256'] != byte_ranges[0]['sha256']):
        raise ValueError('Encoded prefix receipt is inconsistent')
    full = (len(byte_ranges) == 1 and total == row['archive']['size'])
    if (receipt['full_mp3_sha256_verified'] is not full
            or (full and receipt['prefix_sha256'] != row['expected_full_sha256'])):
        raise ValueError('Full-file checksum claim differs from acquired byte coverage')
    with sf.SoundFile(path) as source:
        if (source.frames != 661500 or source.samplerate != 22050 or source.channels != 1
                or source.subtype != 'PCM_16' or receipt['frames'] != source.frames
                or receipt['sample_rate'] != source.samplerate):
            raise ValueError('Decoded WAV geometry changed')
        samples = source.read(dtype='float32')
    if not np.isfinite(samples).all() or np.sqrt(np.mean(samples.astype(float) ** 2)) < 1e-5:
        raise ValueError('Decoded WAV is invalid or silent')
    return receipt


def verify_track_checkpoint(path, expected):
    record = read(path)
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('Feature checkpoint belongs to different input or frozen source')
    if record['status'] == 'failed':
        if not isinstance(record.get('error'), str) or not record['error']:
            raise ValueError('Failed feature checkpoint lacks a failure reason')
        return record, None
    if record['status'] != 'complete':
        raise ValueError('Invalid feature checkpoint status')
    array = path.with_suffix('.npy')
    if digest(array) != record['feature_sha256']:
        raise ValueError('Per-track feature cache changed')
    vector = np.load(array, allow_pickle=False)
    if vector.shape != (WIDTH,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
        raise ValueError('Invalid per-track feature cache')
    return record, vector


def verify_completed(out, selected, success, failed):
    status, metadata = read(out / 'extraction_status.json'), read(out / 'features.json')
    rows = read(out / 'manifest.json')['tracks']
    ids = validate_ids(rows)
    features = np.load(out / 'features.npy', allow_pickle=False)
    extraction_failed = {r['track_id'] for r in status['extraction_failures']}
    if len(extraction_failed) != len(status['extraction_failures']):
        raise ValueError('Duplicate extraction failure records')
    expected_rows = [r for r in selected if r['track_id'] in success and r['track_id'] not in extraction_failed]
    if (status['state'] != 'complete' or rows != expected_rows or extraction_failed - set(success)
            or metadata['ids'] != ids or features.shape != (len(rows), WIDTH)
            or metadata['shape'] != list(features.shape) or metadata['dtype'] != str(features.dtype)
            or features.dtype != np.float32 or not np.isfinite(features).all()
            or metadata['feature_sha256'] != digest(out / 'features.npy')
            or metadata['freeze_sha256'] != digest(out / 'freeze.json')
            or metadata['download_status_sha256'] != digest(out / 'download_status.json')
            or metadata['audio_hashes'] != [success[i]['wav_sha256'] for i in ids]):
        raise ValueError('Completed feature cache or cohort changed')
    for key, file in [('selected_manifest_sha256', 'selected_manifest.json'),
                      ('observed_manifest_sha256', 'manifest.json'),
                      ('download_status_sha256', 'download_status.json'),
                      ('runtime_control_sha256', 'runtime_control.json')]:
        if status[key] != digest(out / file):
            raise ValueError('Completed extraction provenance changed')
    if (status['download_failures'] != list(failed.values()) or status['observed_tracks'] != len(rows)
            or status['selected_tracks'] != len(selected)):
        raise ValueError('Completed extraction counts changed')
    by_id = {track: i for i, track in enumerate(ids)}
    failure_by_id = {r['track_id']: r['error'] for r in status['extraction_failures']}
    for row in selected:
        track = row['track_id']
        if track not in success:
            continue
        record, vector = verify_track_checkpoint(out / 'feature_tracks' / f'{track}.json', {
            'track_id': track, 'audio_sha256': success[track]['wav_sha256'],
            'freeze_sha256': digest(out / 'freeze.json'),
            'download_status_sha256': digest(out / 'download_status.json')})
        if vector is None:
            if failure_by_id.get(track) != record['error']:
                raise ValueError('Completed extraction failure receipt changed')
        elif track not in by_id or not np.array_equal(features[by_id[track]], vector):
            raise ValueError('Completed aggregate differs from a per-track feature receipt')
    return status


def features(out):
    from maest_hf_features import HfMaestEncoder
    out = Path(out)
    frozen = verify_frozen(out)
    config = read(out / 'config.json')
    previous = read(ROOT / 'outputs/maest_hf/features.json')
    versions = {p: importlib.metadata.version(p) for p in previous['versions']}
    if versions != previous['versions'] or versions != frozen['runtime']:
        raise ValueError('Encoder dependencies differ from the frozen historical runtime')
    selected = read(out / 'selected_manifest.json')['tracks']
    acquisition, success, failed = acquisition_rows(out, selected)
    for row in selected:
        if row['track_id'] in success:
            verify_audio_receipt(out, row, success[row['track_id']], config)
    if (out / 'extraction_status.json').exists():
        verify_completed(out, selected, success, failed)
        print('Completed feature cache verified; no overwrite', flush=True)
        return
    encoder = HfMaestEncoder('cpu')
    control_path = safe_file(ROOT, frozen['control_audio_file'])
    control = encoder.extract(control_path).astype(np.float32)
    expected = np.load(ROOT / 'outputs/maest_hf/features.npy', allow_pickle=False, mmap_mode='r')[0]
    if control.shape != (WIDTH,) or not np.isfinite(control).all() or not np.isfinite(expected).all():
        raise ValueError('Historical feature control is invalid')
    difference = float(np.max(np.abs(control - expected)))
    if not np.isfinite(difference) or difference > 1e-5:
        raise ValueError(f'Historical feature reproduction failed: {difference}')
    immutable_json(out / 'runtime_control.json', {
        'track_id': frozen['control_track_id'], 'max_absolute_feature_error': difference,
        'absolute_tolerance': 1e-5, 'versions': versions, 'device': 'cpu',
        'audio_sha256': digest(control_path), 'freeze_sha256': digest(out / 'freeze.json'),
    })
    vectors, rows, failures, audio_hashes = [], [], [], []
    cache = out / 'feature_tracks'
    cache.mkdir(exist_ok=True)
    begin = time.perf_counter()
    for row in selected:
        track = row['track_id']
        if track in failed:
            continue
        expected_receipt = {'track_id': track, 'audio_sha256': success[track]['wav_sha256'],
                            'freeze_sha256': digest(out / 'freeze.json'),
                            'download_status_sha256': digest(out / 'download_status.json')}
        checkpoint = cache / f'{track}.json'
        if checkpoint.exists():
            record, vector = verify_track_checkpoint(checkpoint, expected_receipt)
        else:
            path = safe_file(ROOT, row['audio_file'])
            if digest(path) != expected_receipt['audio_sha256']:
                raise ValueError('Audio changed during feature extraction')
            try:
                vector = encoder.extract(path).astype(np.float32)
                if vector.shape != (WIDTH,) or not np.isfinite(vector).all():
                    raise ValueError('Invalid feature vector')
            except Exception as error:
                record = {**expected_receipt, 'status': 'failed', 'error': repr(error)}
                vector = None
            else:
                save_vector(checkpoint.with_suffix('.npy'), vector)
                record = {**expected_receipt, 'status': 'complete',
                          'feature_sha256': digest(checkpoint.with_suffix('.npy'))}
            immutable_json(checkpoint, record)
        if vector is None:
            failures.append({'track_id': track, 'error': record['error']})
        else:
            vectors.append(vector); rows.append(row); audio_hashes.append(success[track]['wav_sha256'])
        if (len(rows) + len(failures)) % 10 == 0:
            print('Features', len(rows), '/', len(success), 'failures', len(failures),
                  'elapsed', round(time.perf_counter()-begin, 1), 's', flush=True)
    if not rows:
        raise ValueError('No usable new validation audio; per-track failures retained')
    verify_frozen(out)
    for row in selected:
        if row['track_id'] in success:
            verify_audio_receipt(out, row, success[row['track_id']], config)
    values = np.stack(vectors)
    save_vector(out / 'features.npy', values)
    immutable_json(out / 'manifest.json', {'tracks': rows, 'scope': 'observed subset of fixed new sample; no replacement'})
    metadata = {'ids': [r['track_id'] for r in rows], 'shape': list(values.shape),
                'dtype': str(values.dtype), 'feature_sha256': digest(out / 'features.npy'),
                'audio_hashes': audio_hashes, 'versions': versions, 'device': 'cpu',
                'extractor_sha256': digest(ROOT / 'maest_hf_features.py'),
                'encoder_receipt_sha256': digest(ROOT / 'models/mtg-upf-maest-519l/SOURCES.json'),
                'freeze_sha256': digest(out / 'freeze.json'),
                'download_status_sha256': digest(out / 'download_status.json')}
    metadata['extraction_seconds'] = (read(out / 'features.json')['extraction_seconds']
                                      if (out / 'features.json').exists() else time.perf_counter()-begin)
    immutable_json(out / 'features.json', metadata)
    y = targets(rows); artists = np.asarray([r['artist_id'] for r in rows])
    atomic_json(out / 'extraction_status.json', {
        'state': 'complete', 'completed_utc': now(), 'selected_tracks': len(selected),
        'selected_artists': len({r['artist_id'] for r in selected}),
        'observed_tracks': len(rows), 'observed_artists': len(set(artists)),
        'positive_tracks': y.sum(0).tolist(),
        'positive_artists': [len(set(artists[y[:, j] == 1])) for j in range(4)],
        'download_failures': acquisition['failures'], 'extraction_failures': failures,
        'selected_manifest_sha256': digest(out / 'selected_manifest.json'),
        'observed_manifest_sha256': digest(out / 'manifest.json'),
        'download_status_sha256': digest(out / 'download_status.json'),
        'runtime_control_sha256': digest(out / 'runtime_control.json'),
    })
    verify_completed(out, selected, success, failed)
    print('New application feature cache complete:', values.shape, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'verify_frozen', 'features'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.out.resolve())
