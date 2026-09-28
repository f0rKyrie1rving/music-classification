"""Read-only verification of retained caches and public historical predictions."""
import argparse
import importlib.metadata
import platform
import sys
import numpy as np
from .common import ROOT, LABELS, development, digest, git, now, read, save, targets


def audit():
    plan = read(ROOT / 'experiments/final_holdout_plan.json')
    manifest_path = ROOT / 'data/expanded_manifest.json'
    manifest = read(manifest_path)['tracks']
    assert digest(manifest_path) == plan['expanded_manifest_sha256'] == '01d93dbe0474e517861e0d0cbe95ca4a4e8dc769f180ec74d2e3d34490689077'
    rows, y, groups = development()
    ids = [r['track_id'] for r in rows]
    paths = ['outputs/maest_hf/' + f for f in (
        'features.npy', 'features.json', 'development_scores.npz',
        'development_metrics.json', 'broad_genre_candidate.joblib')]
    paths += ['outputs/final_maest/' + f for f in (
        'holdout_features.npy', 'holdout_features.json', 'final_broad_genre_model.joblib')]
    paths += ['outputs/reproduction/development.npy', 'outputs/reproduction/development.json',
              'models/mtg-upf-maest-519l/SOURCES.json']
    assets = {}
    for name in paths:
        path = ROOT / name
        assets[name] = {'exists': path.exists(), 'bytes': path.stat().st_size if path.exists() else None,
                        'sha256': digest(path) if path.exists() else None,
                        'validation': 'presence/hash inventory only; not loaded as executable model'}
    feature_path = ROOT / 'outputs/maest_hf/features.npy'
    meta = read(feature_path.with_suffix('.json'))
    public = read(ROOT / 'data/evaluation/development_feature_provenance.json')
    assert all(meta[k] == v for k, v in public.items())
    assert digest(feature_path) == meta['feature_sha256'] == plan['development_features_sha256']
    assert digest(feature_path.with_suffix('.json')) == plan['development_feature_meta_sha256']
    assert meta['ids'] == ids and meta['test_tracks'] == 0 and len(set(ids)) == 1206
    provenance = {'expanded_manifest_sha256': 'data/expanded_manifest.json',
                  'model_plan_sha256': 'experiments/maest_hf_plan.json',
                  'encoder_manifest_sha256': 'models/mtg-upf-maest-519l/SOURCES.json',
                  'extractor_source_sha256': 'maest_hf_features.py',
                  'driver_source_sha256': 'extract_expanded_maest_hf.py'}
    for key, name in provenance.items():
        assert digest(ROOT / name) == meta[key], name
    x = np.load(feature_path, mmap_mode='r', allow_pickle=False)
    assert x.shape == (1206, 2304) and x.dtype == np.float32 and np.isfinite(x).all()
    assets[str(feature_path.relative_to(ROOT))]['validation'] = 'full recorded hash, provenance, shape, dtype, finite values and ID order verified; ready for CPU training'
    held = read(ROOT / 'data/evaluation/holdout.json')
    assert held['labels'] == LABELS and held['plan_sha256'] == digest(ROOT / 'experiments/final_holdout_plan.json')
    assert [r['track_id'] for r in held['rows']] == plan['holdout_ids']
    by_id = {r['track_id']: r for r in manifest}
    hy = np.array([r['targets'] for r in held['rows']])
    hp = np.array([r['scores'] for r in held['rows']])
    assert np.array_equal(hy, targets([by_id[i] for i in plan['holdout_ids']]))
    assert len(set(plan['holdout_ids'])) == 239 and hp.shape == (239, 4)
    assert np.isfinite(hp).all() and ((hp >= 0) & (hp <= 1)).all()
    assert all(r['artist_id'] == by_id[r['track_id']]['artist_id'] for r in held['rows'])
    counts = {}
    expected = {'train': (903, 361, [316,212,174,170]),
                'development_validation': (303,108,[119,49,70,50]),
                'new_holdout': (239,89,[70,50,48,36]),
                'historical_test_excluded': (90,65,[39,30,23,23])}
    for role, (n, a, positive) in expected.items():
        subset = [r for r in manifest if r['phase_role'] == role]
        sy = targets(subset)
        artists = np.array([r['artist_id'] for r in subset])
        assert (len(subset), len(set(artists)), sy.sum(0).tolist()) == (n,a,positive)
        counts[role] = {'tracks': n, 'artists': a, 'positive': positive,
                        'positive_artists': [len(set(artists[sy[:,j] == 1])) for j in range(4)]}
    role_artists = {role: {r['artist_id'] for r in manifest if r['phase_role'] == role} for role in expected}
    assert not role_artists['train'] & role_artists['development_validation']
    assert not set(groups) & role_artists['new_holdout']
    hold_features = ROOT / 'outputs/final_maest/holdout_features.npy'
    if hold_features.exists():
        hm = read(hold_features.with_suffix('.json'))
        hx = np.load(hold_features, mmap_mode='r', allow_pickle=False)
        assert hm['ids'] == plan['holdout_ids'] and digest(hold_features) == hm['feature_sha256']
        assert hx.shape == (239,2304) and hx.dtype == np.float32 and np.isfinite(hx).all()
        assets[str(hold_features.relative_to(ROOT))]['validation'] = 'hash, IDs, shape, dtype and finite values verified; unused in new experiment'
    score_path = ROOT / 'outputs/maest_hf/development_scores.npz'
    if score_path.exists():
        with np.load(score_path, allow_pickle=False) as scores:
            assert scores['ids'].tolist() == ids and np.array_equal(scores['y'], y)
            for key, shape in [('oof_scores',(903,4)), ('validation_scores',(303,4))]:
                assert scores[key].shape == shape and np.isfinite(scores[key]).all()
            assets[str(score_path.relative_to(ROOT))]['validation'] = 'IDs, broad targets and shapes verified; NOT reused for new predictions'
    receipt = read(ROOT / 'models/mtg-upf-maest-519l/SOURCES.json')
    model_files = {name: (ROOT / 'models/mtg-upf-maest-519l' / name).exists() and
                   digest(ROOT / 'models/mtg-upf-maest-519l' / name) == sha
                   for name, sha in receipt['files_sha256'].items()}
    audio = {name: {'exists': (ROOT / name).exists(), 'wav_files': len(list((ROOT / name).glob('*.wav'))),
                     'validation': 'presence only; decoding/audio hashes not rerun because verified features suffice'}
             for name in ('data/training_audio', 'data/expanded_audio')}
    return {'created_utc': now(), 'git_commit': git('rev-parse','HEAD'), 'git_branch': git('branch','--show-current'),
            'origin': git('remote','get-url','origin'), 'git_status': git('status','--short'),
            'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'versions': {p: importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','matplotlib')},
            'assets': assets, 'audio': audio, 'encoder_file_hash_checks': model_files,
            'counts': counts, 'development': {'tracks': len(rows), 'artists': len(set(groups)), 'positive': y.sum(0).tolist()},
            'historical_test_holdout_artist_overlap': len(role_artists['new_holdout'] & role_artists['historical_test_excluded']),
            'holdout_zero_label_rows': int((hy.sum(1)==0).sum()), 'development_ready': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True)
    args = parser.parse_args(); result = audit(); save(args.out, result)
    print('Audit passed: 1206 x 2304 float32; original SHA-256 and ID order verified.')
