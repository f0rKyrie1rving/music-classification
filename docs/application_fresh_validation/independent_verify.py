"""Independent read-only reconstruction of the frozen fresh application study.

Run only after evaluation completes. This imports application APIs for comparison,
but never imports the fresh-study evaluation/pipeline helpers and never extracts
features, fits a model, downloads audio, or rewrites a study result. The sole
output is OUT/independent_verification.json, including failures if a check aborts.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ['electronic', 'pop', 'ambient', 'rock']
POLICIES = ['f1', 'precision_target']
SEED, REPLICATES, EPS = 2026092802, 2000, 1e-12
EXPECTED_IDS_HASH = 'f7083ca5a190bddc2a2bd20e35d65b418c068358041ebc7234657be1aa773268'
# Independently captured while acquisition was ongoing, before any fresh scores:
# preserve the 28 completed DNS failures across later acquisition and evaluation.
PREPREDICTION_FAILED_IDS = '''track_0138811 track_0341608 track_0345611 track_0364009
track_0364010 track_0385608 track_0387011 track_0388810 track_0403708 track_0416010
track_0448210 track_0452711 track_0468109 track_0807111 track_0966110 track_1035010
track_1035011 track_1078511 track_1177008 track_1177009 track_1196911 track_1246808
track_1274711 track_1287910 track_1333311 track_1373108 track_1411909 track_1411910'''.split()
PREPREDICTION_FAILURES_HASH = '76e88800674fe5357c607049210264804120a3464b76aad6aec944d01de6c2d4'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def contained(base, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe referenced path: ' + str(name))
    result = (base / p).resolve()
    if not result.is_relative_to(base.resolve()):
        raise ValueError('Referenced path escapes root: ' + str(name))
    return result


class Checks:
    def __init__(self):
        self.counts = Counter()
        self.stage = 'initial'
        self.hashes = {}

    def require(self, condition, name):
        self.counts[self.stage] += 1
        if not bool(condition):
            raise AssertionError(name)

    def equal(self, actual, expected, name):
        self.require(actual == expected, name)

    def close(self, actual, expected, name, atol=1e-12):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == b.shape and np.allclose(a, b, rtol=0, atol=atol), name)

    def structure(self, actual, expected, name):
        if isinstance(expected, dict):
            self.equal(set(actual), set(expected), name + ': keys')
            for key, value in expected.items():
                self.structure(actual[key], value, name + '/' + key)
        elif isinstance(expected, list):
            self.equal(len(actual), len(expected), name + ': length')
            for i, value in enumerate(expected):
                self.structure(actual[i], value, name + '/' + str(i))
        elif isinstance(expected, (float, np.floating)):
            self.close(actual, expected, name)
        else:
            self.equal(actual, expected, name)

    def file(self, path, expected=None):
        value = digest(path)
        self.hashes[str(path.relative_to(ROOT))] = value
        if expected is not None:
            self.equal(value, expected, 'Hash mismatch: ' + str(path))
        return value


def all_history(config, checks):
    tracks, artists = set(), set()
    for name in config['history_sources']:
        obj = read(ROOT / name)
        rows = next(obj[key] for key in ('tracks', 'records', 'candidates', 'samples')
                    if isinstance(obj.get(key), list))
        tracks.update(row['track_id'] for row in rows)
        artists.update(row['artist_id'] for row in rows)
    checks.equal((len(tracks), len(artists)), (1929, 837), 'Audited exposure union')
    return tracks, artists


def selection(out, config, checks):
    checks.stage = 'independent_sampling_and_exposure'
    history_tracks, history_artists = all_history(config, checks)
    excluded = read(out / 'exclusions.json')
    checks.equal(set(excluded['tracks']), history_tracks, 'Excluded track union')
    checks.equal(set(excluded['artists']), history_artists, 'Excluded artist union')
    for name, expected in excluded['source_hashes'].items():
        checks.file(ROOT / name, expected)
    for name, expected in config['source_blobs'].items():
        data = (out / 'source' / name).read_bytes()
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        checks.equal(blob, expected, 'Official Git blob: ' + name)
    licenses = {}
    for block in (out / 'source/licenses.txt').read_text().split('\n\n'):
        lines = block.strip().splitlines()
        if not lines:
            continue
        urls = re.findall(r'https?://creativecommons.org/licenses/([^/\s]+)/', block)
        checks.require(len(urls) <= 1 and lines[0] not in licenses, 'Unique license record')
        licenses[lines[0]] = (urls[0] if urls else None, block.strip())
    sums = {}
    for line in (out / 'source/checksums.txt').read_text().splitlines():
        value, name = line.split()
        checks.require(name not in sums, 'Unique official checksum')
        sums[name] = value
    expected = []
    with (out / 'source/genres.tsv').open(newline='') as stream:
        for row in list(csv.reader(stream, delimiter='\t'))[1:]:
            track, artist, album, path, duration, *tags = row
            member = path.replace('.mp3', '.low.mp3')
            if (path[:2] not in config['archives'] or track in history_tracks
                    or artist in history_artists or path not in licenses
                    or licenses[path][0] not in config['allowed_license_codes']
                    or float(duration) < 30 or member not in sums):
                continue
            expected.append({'track_id': track, 'artist_id': artist, 'album_id': album,
                             'path': path, 'duration': float(duration),
                             'tags': [tag.removeprefix('genre---') for tag in tags],
                             'archive_member': member, 'expected_full_sha256': sums[member],
                             'license_code': licenses[path][0],
                             'attribution_verbatim': licenses[path][1]})
    expected.sort(key=lambda row: row['track_id'])
    checks.equal(read(out / 'candidate_pool.json')['tracks'], expected, 'Independent candidate reconstruction')
    checks.equal((len(expected), len({row['artist_id'] for row in expected})), (266, 171), 'Fixed frame size')
    key = lambda kind, value: hashlib.sha256(f"{config['sample_seed']}|{kind}|{value}".encode()).hexdigest()
    artists = sorted({row['artist_id'] for row in expected}, key=lambda x: key('artist', x))
    wanted = []
    for artist in artists:
        songs = sorted((r for r in expected if r['artist_id'] == artist), key=lambda r: key('track', r['track_id']))
        wanted.extend(songs[:2])
    selected = read(out / 'selected_manifest.json')['tracks']
    expected_ids = [row['track_id'] for row in wanted]
    checks.equal([row['track_id'] for row in selected], expected_ids, 'Exact hash-selected order')
    checks.equal(hashlib.sha256('\n'.join(expected_ids).encode()).hexdigest(), EXPECTED_IDS_HASH,
                 'Independent pre-acquisition ID hash')
    checks.equal((len(selected), len(artists)), (232, 171), 'Fixed selected cohort')
    for actual, original in zip(selected, wanted, strict=True):
        for name, value in original.items():
            checks.equal(actual[name], value, 'Selected source field: ' + name)
        index = read(out / f"archive_index_{actual['path'][:2]}.json")
        checks.require(index['complete'], 'Complete selected archive index')
        checks.equal(actual['archive'], {**index['members'][actual['archive_member']],
                     'archive_url': index['archive_url'], 'archive_bytes': index['archive_bytes']}, 'TAR location')
        checks.equal(contained(ROOT, actual['audio_file']), out / 'audio' / (actual['track_id'] + '.wav'), 'Audio location')
    return selected, history_tracks, history_artists


def verify_acquisition_and_features(out, selected, checks):
    checks.stage = 'acquisition_and_feature_integrity'
    download, extraction, meta = [read(out / name) for name in
                                 ['download_status.json', 'extraction_status.json', 'features.json']]
    successes = {r['track_id']: r for r in download['successful']}
    failures = {r['track_id']: r for r in download['failures']}
    selected_ids = {r['track_id'] for r in selected}
    checks.equal(len(successes), len(download['successful']), 'Unique download successes')
    checks.equal(len(failures), len(download['failures']), 'Unique download failures')
    checks.require(not set(successes) & set(failures), 'No success/failure overlap')
    checks.equal(set(successes) | set(failures), selected_ids, 'Every selected download accounted for')
    checks.require(set(PREPREDICTION_FAILED_IDS) <= set(failures), 'All pre-prediction DNS failures retained')
    prior_hashes = {track: checks.file(out / 'download_checkpoints' / (track + '.json'))
                    for track in PREPREDICTION_FAILED_IDS}
    checks.equal(hashlib.sha256(json.dumps(prior_hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                 PREPREDICTION_FAILURES_HASH, 'Pre-prediction failure checkpoints byte-identical')
    checks.equal(download['requested'], len(selected), 'Requested count')
    checks.equal(download['replacement_tracks'], 0, 'No replacement')
    checks.equal(extraction['state'], 'complete', 'Extraction complete')
    extraction_failed = {r['track_id']: r['error'] for r in extraction['extraction_failures']}
    checks.equal(len(extraction_failed), len(extraction['extraction_failures']), 'Unique extraction failures')
    checks.require(set(extraction_failed) <= set(successes), 'Extraction failures had downloaded audio')
    expected_rows = [r for r in selected if r['track_id'] in successes and r['track_id'] not in extraction_failed]
    rows = read(out / 'manifest.json')['tracks']
    checks.equal(rows, expected_rows, 'Observed cohort exactly follows persisted failures')
    checks.require(bool(rows), 'Nonempty observed cohort')
    freeze_sha, selected_sha, download_sha = [checks.file(out / name) for name in
                                            ['freeze.json', 'selected_manifest.json', 'download_status.json']]
    checks.equal(meta['freeze_sha256'], freeze_sha, 'Feature frozen identity')
    checks.equal(meta['download_status_sha256'], download_sha, 'Feature acquisition identity')
    checks.equal(meta['ids'], [r['track_id'] for r in rows], 'Feature order')
    checks.equal(meta['audio_hashes'], [successes[r['track_id']]['wav_sha256'] for r in rows], 'Feature audio order')
    checks.file(out / 'features.npy', meta['feature_sha256'])
    for key, name in [('selected_manifest_sha256', 'selected_manifest.json'),
                      ('observed_manifest_sha256', 'manifest.json'),
                      ('download_status_sha256', 'download_status.json'),
                      ('runtime_control_sha256', 'runtime_control.json')]:
        checks.file(out / name, extraction[key])
    checks.equal(extraction['download_failures'], download['failures'], 'All download failures retained')
    values = np.load(out / 'features.npy', allow_pickle=False)
    checks.require(values.shape == (len(rows), 2304) and values.dtype == np.float32 and np.isfinite(values).all(), 'Feature matrix')
    checks.equal(meta['shape'], list(values.shape), 'Recorded feature shape')
    checks.equal(meta['dtype'], str(values.dtype), 'Recorded feature dtype')
    positions = {row['track_id']: i for i, row in enumerate(rows)}
    for row in selected:
        track = row['track_id']
        checkpoint = read(out / 'download_checkpoints' / (track + '.json'))
        checks.equal(checkpoint['track_id'], track, 'Download checkpoint ID')
        checks.equal(checkpoint['identity'], {'freeze_sha256': freeze_sha, 'selection_sha256': selected_sha}, 'Download checkpoint source')
        if track in failures:
            checks.equal(checkpoint['state'], 'failed', 'Download failure preserved')
            checks.equal(checkpoint['error'], failures[track]['error'], 'Download error preserved')
            continue
        checks.equal(checkpoint['state'], 'success', 'Download success preserved')
        receipt = successes[track]
        checks.equal(checkpoint['receipt'], receipt, 'Download success receipt unchanged')
        checks.equal(read(out / 'acquisition/download_receipts' / (track + '.json')), receipt, 'Original acquisition receipt')
        checks.file(contained(ROOT, row['audio_file']), receipt['wav_sha256'])
        total, previous_end = 0, row['archive']['offset'] - 1
        for segment in receipt['byte_ranges']:
            start, end = segment['byte_range']
            cache = contained(ROOT, segment['cache_file'])
            checks.require(cache.is_relative_to(out / 'acquisition/audio_prefixes'), 'Prefix cache location')
            checks.require(row['archive']['offset'] <= start <= end < row['archive']['offset'] + row['archive']['size']
                           and start > previous_end, 'Encoded byte coverage')
            checks.equal(cache.stat().st_size, end - start + 1, 'Prefix length')
            checks.file(cache, segment['sha256'])
            total += end - start + 1
            previous_end = end
        checks.equal(total, receipt['downloaded_bytes'], 'Encoded bytes reconciled')
        checks.require(0 < total <= 1310720, 'Encoded budget')
        checks.equal(receipt['byte_ranges'][0]['byte_range'][0], row['archive']['offset'], 'First encoded byte')
        checks.equal(receipt['prefix_sha256'], receipt['byte_ranges'][0]['sha256'], 'Prefix hash')
        full = len(receipt['byte_ranges']) == 1 and total == row['archive']['size']
        checks.equal(receipt['full_mp3_sha256_verified'], full, 'Full-file claim')
        if full:
            checks.equal(receipt['prefix_sha256'], row['expected_full_sha256'], 'Official full-file checksum')
        feature_path = out / 'feature_tracks' / (track + '.json')
        record = read(feature_path)
        for name, expected in [('track_id', track), ('audio_sha256', receipt['wav_sha256']),
                               ('freeze_sha256', freeze_sha), ('download_status_sha256', download_sha)]:
            checks.equal(record[name], expected, 'Per-track feature source')
        if track in extraction_failed:
            checks.equal(record['status'], 'failed', 'Extraction failure preserved')
            checks.equal(record['error'], extraction_failed[track], 'Extraction error preserved')
        else:
            checks.equal(record['status'], 'complete', 'Feature completion')
            checks.file(feature_path.with_suffix('.npy'), record['feature_sha256'])
            checks.require(np.array_equal(values[positions[track]], np.load(feature_path.with_suffix('.npy'), allow_pickle=False)),
                           'Aggregate row matches per-track cache')
    control = read(out / 'runtime_control.json')
    checks.equal(control['freeze_sha256'], freeze_sha, 'Historical control frozen source')
    checks.require(control['device'] == 'cpu' and 0 <= control['max_absolute_feature_error'] <= 1e-5,
                   'Historical feature control passed')
    return rows, values, extraction


def average_precision(y, scores):
    if not 0 < y.sum() < len(y):
        return None
    order = np.argsort(-scores, kind='stable')
    truth, ordered = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered)-1]
    true_positives = np.cumsum(truth)[ends]
    precision = true_positives / (ends + 1)
    positive_increments = np.diff(np.r_[0, true_positives])
    return float(np.dot(precision, positive_increments) / y.sum())


def losses(y, scores):
    clipped = np.maximum(np.minimum(scores, 1-EPS), EPS)
    return {'brier': np.square(scores-y), 'log_loss': -(y*np.log(clipped)+(1-y)*np.log(1-clipped))}


def metrics(y, scores, artists, decisions):
    loss = losses(y, scores)
    result = {'tracks': len(y), 'artists': len(set(artists)),
              **{'macro_'+name: float(array.mean()) for name, array in loss.items()},
              'per_label': {}, 'policies': {}}
    for j, label in enumerate(LABELS):
        result['per_label'][label] = {'positive_tracks': int(y[:, j].sum()),
            'positive_artists': len(set(artists[y[:, j] == 1])), 'positive_fraction': float(y[:, j].mean()),
            'mean_score': float(scores[:, j].mean()),
            **{name: float(array[:, j].mean()) for name, array in loss.items()},
            'average_precision': average_precision(y[:, j], scores[:, j])}
    for policy, predicted in decisions.items():
        tp = ((predicted == 1) & (y == 1)).sum(0)
        fp = ((predicted == 1) & (y == 0)).sum(0)
        fn = ((predicted == 0) & (y == 1)).sum(0)
        divide = lambda a, b: np.divide(a, b, out=np.zeros(np.shape(a), dtype=float), where=np.asarray(b) != 0)
        precision, recall = divide(tp, tp+fp), divide(tp, tp+fn)
        f1 = divide(2*tp, 2*tp+fp+fn)
        result['policies'][policy] = {'micro_precision': float(divide(tp.sum(), tp.sum()+fp.sum())),
            'micro_recall': float(divide(tp.sum(), tp.sum()+fn.sum())),
            'micro_f1': float(divide(2*tp.sum(), 2*tp.sum()+fp.sum()+fn.sum())), 'macro_f1': float(f1.mean()),
            'false_positives': int(fp.sum()), 'false_negatives': int(fn.sum()),
            'coverage': float(predicted.any(1).mean()),
            'per_label': {label: {'precision': float(precision[j]), 'recall': float(recall[j]),
                                  'f1': float(f1[j]), 'false_positives': int(fp[j]),
                                  'false_negatives': int(fn[j]), 'positive_decisions': int(predicted[:, j].sum())}
                          for j, label in enumerate(LABELS)}}
    return result


def run(out, checks):
    checks.stage = 'frozen_sources_and_completed_results'
    frozen = read(out / 'freeze.json')
    config = read(out / 'config.json')
    checks.equal(config['bootstrap_seed'], SEED, 'Prespecified bootstrap seed')
    checks.equal(config['bootstrap_replicates'], REPLICATES, 'Prespecified replicates')
    checks.equal(config['minimum_observed_fraction'], .8, 'Prespecified coverage gate')
    checks.equal(config['archives'], ['08', '09', '10', '11'], 'Fixed archive frame')
    checks.equal(config['artist_count'], 171, 'Census size')
    checks.equal(config['tracks_per_artist'], 2, 'Song cap')
    for key in ['source_hashes', 'input_hashes', 'local_hashes']:
        base = out if key == 'local_hashes' else ROOT
        for name, expected in frozen[key].items():
            checks.file(contained(base, name), expected)
    for versions in [frozen['runtime'], frozen['evaluation_runtime']]:
        for name, expected in versions.items():
            checks.equal(importlib.metadata.version(name), expected, 'Frozen runtime: ' + name)
    status = read(out / 'evaluation_status.json')
    checks.equal(status['state'], 'complete', 'Evaluation is complete')
    checks.equal(set(status['hashes']), {'predictions.npz', 'metrics.json', 'intervals.json', 'summary.json', 'public_predictions.json'}, 'Complete result hash list')
    for name, expected in status['hashes'].items():
        checks.file(out / name, expected)
    selected, history_tracks, history_artists = selection(out, config, checks)
    rows, x, extraction = verify_acquisition_and_features(out, selected, checks)
    ids = [r['track_id'] for r in rows]
    artists = np.asarray([r['artist_id'] for r in rows])
    checks.require(not set(ids) & history_tracks and not set(artists) & history_artists, 'Observed histories disjoint')
    checks.stage = 'actual_release_reconstruction'
    plan = read(ROOT / 'experiments/final_holdout_plan.json')
    meta = read(ROOT / 'artifacts/final_heads.json')
    correction = read(ROOT / 'artifacts/score_correction.json')
    weights = np.load(ROOT / 'artifacts/final_heads.npy', allow_pickle=False)
    checks.require(weights.shape == (3, 4, 2304) and np.isfinite(weights).all(), 'Released parameters')
    checks.equal(meta['parameter_order'], ['mean', 'scale', 'coefficient'], 'Parameter order')
    checks.equal(meta['labels'], LABELS, 'Released label order')
    original = {r['track_id']: r for r in read(ROOT / 'data/expanded_manifest.json')['tracks']}
    target = lambda records: np.asarray([[int(bool(set(row['tags']) & set(plan['ontology'][label])))
                                        for label in LABELS] for row in records], dtype=int)
    fit_y = target([original[i] for i in plan['fit_ids']])
    checks.equal(correction['fit_count'], len(fit_y), 'Actual fit count')
    checks.equal(correction['fit_positive_counts'], fit_y.sum(0).tolist(), 'Actual fit positives')
    checks.equal(correction['class_weight'], [None, None, 'balanced', None], 'Only ambient weighted')
    offsets = np.array([0., 0., np.log(fit_y[:, 2].sum()/(len(fit_y)-fit_y[:, 2].sum())), 0.])
    checks.close(correction['logit_offsets'], offsets, 'Training-derived offsets', atol=1e-15)
    y = target(rows)
    checks.equal(extraction['positive_tracks'], y.sum(0).tolist(), 'Extraction target counts')
    checks.equal(extraction['positive_artists'], [len(set(artists[y[:, j] == 1])) for j in range(4)],
                 'Extraction positive artist counts')
    z = np.stack([np.sum(((np.asarray(row, dtype=np.float64)-weights[0])/weights[1])*weights[2], axis=1)
                  + np.asarray(meta['intercepts'], dtype=np.float64) for row in x])
    raw = np.exp(-np.logaddexp(0., -z))
    fixed = raw.copy()
    fixed[:, 2] = np.exp(-np.logaddexp(0., -(z[:, 2]+offsets[2])))
    thresholds = {}
    decisions = {}
    for policy in POLICIES:
        old_thresholds = np.asarray(meta['thresholds'][policy], float)
        threshold = old_thresholds.copy()
        if 0 < threshold[2] < 1:
            odds = threshold[2]/(1-threshold[2]) * np.exp(offsets[2])
            threshold[2] = odds/(1+odds)
        thresholds[policy] = threshold
        decisions[policy] = raw >= old_thresholds
        checks.require(np.array_equal(decisions[policy], fixed >= threshold), 'Transformed decision equivalence: ' + policy)
    with np.load(out / 'predictions.npz', allow_pickle=False) as archive:
        stored = {key: archive[key] for key in archive.files}
    for name, expected in [('ids', ids), ('artists', artists.tolist()), ('y', y.tolist())]:
        checks.equal(stored[name].tolist(), expected, 'Prediction identities/labels: ' + name)
    for name, expected in [('logits', z), ('raw', raw), ('corrected', fixed)]:
        checks.close(stored[name], expected, 'Reconstructed ' + name)
    checks.require(np.array_equal(raw[:, [0, 1, 3]], fixed[:, [0, 1, 3]]), 'Unchanged-label exact negative controls')
    for policy in POLICIES:
        checks.require(np.array_equal(stored['decisions_'+policy], decisions[policy]), 'Stored decisions: ' + policy)
    sys.path.insert(0, str(ROOT))
    from predict_maest import load_bundle, predict_vector as old_api
    from predict_app import predict_vector as new_api
    bundle = load_bundle()
    for i, vector in enumerate(x):
        for policy in POLICIES:
            old = old_api(bundle, vector, policy)
            new_raw = new_api(bundle, vector, policy, 'raw')
            new_fixed = new_api(bundle, vector, policy, 'weight_corrected', correction)
            for j, label in enumerate(LABELS):
                checks.equal(old[j], {key: new_raw[j][key] for key in old[j]}, 'Old API equals new raw API')
                checks.equal(new_fixed[j], {'label': label, 'score': round(float(fixed[i, j]), 4),
                    'threshold': round(float(thresholds[policy][j]), 4), 'selected': bool(decisions[policy][i, j]),
                    'raw_score': round(float(raw[i, j]), 4), 'raw_threshold': round(float(meta['thresholds'][policy][j]), 4)},
                    'Corrected API equals independent calculation')
    checks.stage = 'independent_metrics'
    computed = {'raw': metrics(y, raw, artists, decisions), 'weight_corrected': metrics(y, fixed, artists, decisions)}
    checks.structure(read(out / 'metrics.json'), computed, 'All metrics')
    checks.stage = 'independent_paired_artist_bootstrap'
    unique = np.unique(artists)
    sampled = np.random.default_rng(SEED).integers(0, len(unique), size=(REPLICATES, len(unique)))
    checks.equal(stored['bootstrap_artists'].tolist(), unique.tolist(), 'Bootstrap artist order')
    checks.require(np.array_equal(stored['bootstrap_artist_indices'], sampled), 'Every seeded bootstrap draw')
    clusters = [np.flatnonzero(artists == artist) for artist in unique]
    intervals = read(out / 'intervals.json')
    checks.equal((intervals['seed'], intervals['replicates'], intervals['artists']), (SEED, REPLICATES, len(unique)), 'Bootstrap metadata')
    differences = {name: array-losses(y, raw)[name] for name, array in losses(y, fixed).items()}
    rebuilt = {name: np.empty((REPLICATES, 5)) for name in differences}
    # Intentionally concatenate every sampled artist's actual rows; do not reuse
    # the evaluator's compressed artist_sums / sizes implementation.
    for b, draw in enumerate(sampled):
        row_indices = np.concatenate([clusters[int(index)] for index in draw])
        for name, delta in differences.items():
            label_change = delta[row_indices].mean(0)
            rebuilt[name][b] = np.r_[label_change.mean(), label_change]
            checks.close(stored['bootstrap_'+name][b], rebuilt[name][b], f'{name} draw {b}')
    for name, delta in differences.items():
        quantiles = np.quantile(rebuilt[name], [.025, .975], axis=0)
        expected = {'delta': float(delta.mean()), 'ci95': quantiles[:, 0].tolist(),
                    'per_label': {label: {'delta': float(delta[:, j].mean()), 'ci95': quantiles[:, j+1].tolist()}
                                  for j, label in enumerate(LABELS)}}
        checks.structure(intervals['metrics'][name], expected, 'Paired intervals: ' + name)
    checks.stage = 'coverage_verdict_and_public_rows'
    selected_artists = {r['artist_id'] for r in selected}
    coverage = {'selected_tracks': len(selected), 'observed_tracks': len(rows),
                'selected_artists': len(selected_artists), 'observed_artists': len(unique),
                'track_fraction': len(rows)/len(selected), 'artist_fraction': len(unique)/len(selected_artists),
                'minimum_required_fraction': .8, 'missing_ids': [r['track_id'] for r in selected if r['track_id'] not in set(ids)]}
    summary = read(out / 'summary.json')
    checks.structure(summary['coverage'], coverage, 'Coverage')
    for name in ['selected_tracks', 'observed_tracks', 'selected_artists', 'observed_artists']:
        checks.equal(extraction[name], coverage[name], 'Extraction count: ' + name)
    enough = coverage['track_fraction'] >= .8 and coverage['artist_fraction'] >= .8
    both = all(computed['weight_corrected']['macro_'+name] < computed['raw']['macro_'+name] for name in ['brier', 'log_loss'])
    bounds = {name: np.quantile(array[:, 0], [.025, .975]) for name, array in rebuilt.items()}
    if any(bound[0] > 0 for bound in bounds.values()):
        verdict = 'harmful'
    elif not enough:
        verdict = 'incomplete_evidence'
    elif both and bounds['brier'][1] < 0:
        verdict = 'go'
    elif both and bounds['brier'][0] <= 0 <= bounds['brier'][1]:
        verdict = 'promising'
    else:
        verdict = 'inconclusive'
    for name, expected in {'verdict': verdict, 'go': verdict == 'go', 'both_point_estimates_improve': both,
                           'minimum_coverage_met': enough, 'changed_tag_decisions': 0}.items():
        checks.equal(summary['assessment'][name], expected, 'Frozen gate: ' + name)
    checks.equal(status['verdict'], verdict, 'Completion verdict')
    for name in ['macro_brier', 'macro_log_loss']:
        checks.structure(summary[name], {method: values[name] for method, values in computed.items()}, 'Summary ' + name)
    relative = (1-computed['weight_corrected']['macro_brier']/computed['raw']['macro_brier']
                if computed['raw']['macro_brier'] > 0 else None)
    checks.structure(summary['relative_brier_reduction'], relative, 'Relative reduction')
    checks.equal(summary['interface_checks'], {'label_policy_interface_checks': len(rows)*8,
                 'changed_tag_decisions': 0, 'transformed_threshold_mismatches': 0}, 'Interface check counts')
    public = read(out / 'public_predictions.json')
    checks.structure(public['coverage'], coverage, 'Public coverage')
    for key, name in [('model_weights_sha256', ROOT/'artifacts/final_heads.npy'),
                      ('model_metadata_sha256', ROOT/'artifacts/final_heads.json'),
                      ('correction_sha256', ROOT/'artifacts/score_correction.json'),
                      ('freeze_sha256', out/'freeze.json'), ('feature_sha256', out/'features.npy')]:
        checks.file(name, public[key])
    checks.equal(public['labels'], LABELS, 'Public label order')
    checks.close(public['logit_offsets'], offsets, 'Public correction')
    checks.structure(public['raw_thresholds'], meta['thresholds'], 'Public raw thresholds')
    checks.structure(public['corrected_thresholds'], {p: t.tolist() for p, t in thresholds.items()}, 'Public corrected thresholds')
    checks.equal(len(public['rows']), len(rows), 'Public row count')
    for i, row in enumerate(rows):
        checks.structure(public['rows'][i], {'track_id': row['track_id'], 'artist_id': row['artist_id'],
            'source_tags': row['tags'], 'targets': y[i].tolist(), 'logits': z[i].tolist(),
            'raw_scores': raw[i].tolist(), 'corrected_scores': fixed[i].tolist(),
            'selected': {p: d[i].tolist() for p, d in decisions.items()}}, 'Public row ' + str(i))
    # Check hash identities again after reading every result.
    for name, expected in status['hashes'].items():
        checks.file(out / name, expected)
    return {'selected_tracks': len(selected), 'observed_tracks': len(rows), 'observed_artists': len(unique),
            'bootstrap_draws_independently_reconstructed': REPLICATES, 'verdict': verdict,
            'macro_brier': {method: values['macro_brier'] for method, values in computed.items()},
            'macro_log_loss': {method: values['macro_log_loss'] for method, values in computed.items()},
            'coverage': coverage, 'download_failures': len(extraction['download_failures']),
            'extraction_failures': len(extraction['extraction_failures']),
            'preserved_preprediction_dns_failures': len(PREPREDICTION_FAILED_IDS),
            'unchanged_decisions': True, 'selection_ids_sha256': EXPECTED_IDS_HASH}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'outputs/application_fresh_validation'):
        raise ValueError('Expected a fresh-validation output directory')
    checks = Checks()
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'verification_source_sha256': digest(Path(__file__)),
              'scope': 'Independent artifact-to-API numerical reconstruction; no refit, no encoder inference, no result mutation',
              'absolute_tolerance': 1e-12, 'errors': []}
    try:
        report.update(run(out, checks))
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['errors'] = [{'stage': checks.stage, 'error': repr(error), 'traceback': traceback.format_exc()}]
    report['checks_per_category'] = dict(checks.counts)
    report['checks_total'] = sum(checks.counts.values())
    report['verified_file_hashes'] = checks.hashes
    destination = out / 'independent_verification.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['status', 'checks_total', 'errors']}, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
