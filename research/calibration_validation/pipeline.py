"""Freeze the new sample, then extract features using verified old encoder inputs."""
import argparse
import importlib.metadata
import shutil
import time
from pathlib import Path

import numpy as np
from research.calibration.common import ROOT, read, save, digest, now, git, targets
from .evaluate import verify_frozen

HERE = Path(__file__).resolve().parent


def freeze(out):
    if (out / 'freeze.json').exists():
        raise FileExistsError('Validation plan already frozen')
    config = read(out / 'config.json')
    if config != read(HERE / 'config.json') or config['maximum_prefix_bytes_per_track'] != 20 * 65536:
        raise ValueError('Config differs from bounded downloader settings')
    selected = read(out / 'selected_manifest.json')['tracks']
    exclusions = read(out / 'exclusions.json')
    candidate = {r['track_id']: r for r in read(out / 'candidate_pool.json')['tracks']}
    ids = [r['track_id'] for r in selected]
    artists = [r['artist_id'] for r in selected]
    if len(set(ids)) != len(ids) or len(set(artists)) != config['artist_count']:
        raise ValueError('Selected track/artist count changed')
    if set(ids) & set(exclusions['tracks']) or set(artists) & set(exclusions['artists']):
        raise ValueError('History exclusion failed')
    for row in selected:
        if any(row[k] != v for k, v in candidate[row['track_id']].items()):
            raise ValueError('Selected metadata differs from source candidate')
    shutil.copy2(HERE / 'protocol.md', out / 'protocol.md')
    local = ['config.json', 'protocol.md', 'model_reference.json', 'model_hashes.json',
        'numerical_checks.json', 'exclusions.json', 'candidate_pool.json', 'data_audit.json',
        'selected_manifest.json', 'exposure_history.json']
    local += [f'archive_index_{s}.json' for s in config['archives']]
    local += [f'source/{name}' for name in config['source_blobs']]
    sources = list(HERE.glob('*.py')) + [ROOT / name for name in
        ['prepare_dataset.py', 'maest_hf_features.py', 'prepare_maest_hf.py', 'prepare_mert.py']]
    sources += list((ROOT / 'research/calibration').glob('*.py'))
    inputs = list(read(out / 'model_hashes.json')['files'])
    receipt = read(ROOT / 'models/mtg-upf-maest-519l/SOURCES.json')
    inputs += ['models/mtg-upf-maest-519l/SOURCES.json']
    inputs += ['models/mtg-upf-maest-519l/' + name for name in receipt['files_sha256']]
    for source in sources:
        dest = out / 'source_snapshot' / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    save(out / 'freeze.json', {'created_utc': now(), 'git_commit': git('rev-parse', 'HEAD'),
        'stage': 'before acquisition/feature extraction/new model predictions; candidate metadata audited',
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'input_hashes': {name: digest(ROOT / name) for name in inputs},
        'local_hashes': {name: digest(out / name) for name in local}})
    verify_frozen(out)
    print('Frozen fixed model, sample and protocol:', len(ids), 'tracks', flush=True)


def features(out):
    from maest_hf_features import HfMaestEncoder
    verify_frozen(out)
    for name in ['features.npy', 'features.json', 'manifest.json', 'extraction_status.json']:
        if (out / name).exists():
            raise FileExistsError(name)
    previous = read(ROOT / 'outputs/maest_hf/features.json')
    versions = {p: importlib.metadata.version(p) for p in previous['versions']}
    if versions != previous['versions']:
        raise ValueError('Encoder dependencies differ from the original feature runtime')
    encoder = HfMaestEncoder('cpu')
    old_id = previous['ids'][0]
    old_rows = read(ROOT / 'data/expanded_manifest.json')['tracks']
    control_path = ROOT / next(r['audio_file'] for r in old_rows if r['track_id'] == old_id)
    control = encoder.extract(control_path).astype(np.float32)
    expected = np.load(ROOT / 'outputs/maest_hf/features.npy', allow_pickle=False, mmap_mode='r')[0]
    if control.shape != (2304,) or not np.isfinite(control).all() or not np.isfinite(expected).all():
        raise ValueError('Historical feature control is invalid')
    difference = float(np.max(np.abs(control-expected)))
    if not np.isfinite(difference) or difference > 1e-5:
        raise ValueError(f'Historical feature reproduction failed: {difference}')
    save(out / 'runtime_control.json', {'created_utc': now(), 'track_id': old_id,
        'max_absolute_feature_error': difference, 'absolute_tolerance': 1e-5,
        'versions': versions, 'device': 'cpu', 'audio_sha256': digest(control_path)})
    selected = read(out / 'selected_manifest.json')['tracks']
    acquisition = read(out / 'download_status.json')
    success = {r['track_id']: r for r in acquisition['successful']}
    failed = {r['track_id']: r for r in acquisition['failures']}
    selected_ids = {r['track_id'] for r in selected}
    if set(success) & set(failed) or set(success) | set(failed) != selected_ids:
        raise ValueError('Acquisition did not account for every selected track')
    if len(success)+len(failed) != acquisition['requested']:
        raise ValueError('Acquisition counts do not reconcile')
    vectors, rows, extraction_failures, audio_hashes = [], [], [], []
    begin = time.perf_counter()
    for index, row in enumerate(selected):
        if row['track_id'] in failed:
            continue
        path = ROOT / row['audio_file']
        receipt = read(out / 'acquisition/download_receipts' / f"{row['track_id']}.json")
        if receipt != success[row['track_id']] or digest(path) != receipt['wav_sha256']:
            raise ValueError('Acquired audio/receipt changed')
        try:
            vector = encoder.extract(path).astype(np.float32)
            if vector.shape != (2304,) or not np.isfinite(vector).all():
                raise ValueError('Invalid feature vector')
        except Exception as error:
            extraction_failures.append({'track_id': row['track_id'], 'error': repr(error)})
            continue
        vectors.append(vector); rows.append(row); audio_hashes.append(receipt['wav_sha256'])
        if len(rows) % 20 == 0:
            print('Features', len(rows), '/', len(success), 'elapsed', round(time.perf_counter()-begin, 1), 's', flush=True)
    if not rows:
        raise ValueError('No usable validation audio')
    values = np.stack(vectors)
    np.save(out / 'features.npy', values, allow_pickle=False)
    save(out / 'manifest.json', {'tracks': rows, 'scope': 'observed subset of fixed selected sample; no replacement'})
    save(out / 'features.json', {'ids': [r['track_id'] for r in rows], 'shape': list(values.shape),
        'dtype': str(values.dtype), 'feature_sha256': digest(out / 'features.npy'),
        'audio_hashes': audio_hashes, 'versions': versions, 'device': 'cpu',
        'extractor_sha256': digest(ROOT / 'maest_hf_features.py'),
        'encoder_receipt_sha256': digest(ROOT / 'models/mtg-upf-maest-519l/SOURCES.json'),
        'extraction_seconds': time.perf_counter()-begin})
    y = targets(rows); artists = np.asarray([r['artist_id'] for r in rows])
    save(out / 'extraction_status.json', {'state': 'complete', 'completed_utc': now(),
        'selected_tracks': len(selected), 'selected_artists': len({r['artist_id'] for r in selected}),
        'observed_tracks': len(rows), 'observed_artists': len(set(artists)),
        'positive_tracks': y.sum(0).tolist(),
        'positive_artists': [len(set(artists[y[:,j]==1])) for j in range(4)],
        'download_failures': acquisition['failures'], 'extraction_failures': extraction_failures,
        'selected_manifest_sha256': digest(out / 'selected_manifest.json'),
        'observed_manifest_sha256': digest(out / 'manifest.json'),
        'download_status_sha256': digest(out / 'download_status.json'),
        'runtime_control_sha256': digest(out / 'runtime_control.json')})
    verify_frozen(out)
    print('New feature cache complete:', values.shape, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'features'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); globals()[args.action](args.out.resolve())
