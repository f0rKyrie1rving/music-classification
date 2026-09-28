"""Independent artifact and numerical checks after fixed-model validation.

Use ``python -m research.calibration_validation.verify_results --out PATH``.
No model or policy is fitted and no acquisition or inference pipeline is invoked.
The only written artifact is PATH/verification.json; unit tests are separate.
"""
import argparse
import hashlib
import wave
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from research.calibration.common import ROOT, LABELS, digest, now, read, save
from .evaluate import verify_frozen

METHODS = ('identity', 'sigmoid', 'conservative_sigmoid')


class Checks:
    def __init__(self):
        self.counts, self.errors, self.hashes = {}, [], {}

    def check(self, condition, category, description):
        self.counts[category] = self.counts.get(category, 0) + 1
        if not bool(condition):
            self.errors.append(description)

    def equal(self, actual, expected, category, description):
        self.check(np.array_equal(actual, expected), category, description)

    def close(self, actual, expected, category, description, atol=1e-12):
        self.check(np.allclose(actual, expected, rtol=0, atol=atol), category, description)

    def file_hash(self, path, expected, description):
        actual = digest(path)
        self.hashes[str(path.relative_to(ROOT))] = actual
        self.check(actual == expected, 'hashes', description)


def unique_rows(rows, checks, description):
    result = {r['track_id']: r for r in rows}
    checks.check(len(result) == len(rows), 'sample_accounting', description + ': duplicate IDs')
    return result


def check_sample(out, config, checks):
    historical = read(ROOT / 'data/expanded_manifest.json')['tracks']
    old_ids = {r['track_id'] for r in historical}
    old_artists = {r['artist_id'] for r in historical}
    exclusion = read(out / 'exclusions.json')
    checks.equal(exclusion['tracks'], sorted(old_ids), 'sample_accounting', 'Historical track exclusions')
    checks.equal(exclusion['artists'], sorted(old_artists), 'sample_accounting', 'Historical artist exclusions')
    checks.file_hash(ROOT / 'data/expanded_manifest.json', exclusion['source_sha256'], 'Historical manifest hash')
    candidates = read(out / 'candidate_pool.json')['tracks']
    candidate_map = unique_rows(candidates, checks, 'Candidates')
    for name, expected in config['source_blobs'].items():
        path = out / 'source' / name
        content = path.read_bytes()
        git_blob = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        checks.check(git_blob == expected, 'hashes', 'Official source Git blob: ' + name)
    audit = read(out / 'data_audit.json')
    checks.check(audit['counts']['eligible_tracks'] == len(candidates), 'sample_accounting', 'Candidate track count')
    checks.check(audit['counts']['eligible_artists'] == len({r['artist_id'] for r in candidates}),
                 'sample_accounting', 'Candidate artist count')
    for name, record in audit['source_files'].items():
        checks.file_hash(out / 'source' / name, record['sha256'], 'Audited source hash: ' + name)
    for row in candidates:
        checks.check(row['track_id'] not in old_ids and row['artist_id'] not in old_artists,
                     'sample_accounting', 'Candidate overlaps history: ' + row['track_id'])
        checks.check(row['path'][:2] in config['archives'] and row['duration'] >= 30
                     and row['license_code'] in config['allowed_license_codes'],
                     'sample_accounting', 'Candidate eligibility: ' + row['track_id'])
    seed = config['sample_seed']

    def key(kind, value):
        return hashlib.sha256(f'{seed}|{kind}|{value}'.encode()).hexdigest()

    artist_order = sorted({r['artist_id'] for r in candidates}, key=lambda a: key('artist', a))
    expected = []
    for artist in artist_order[:config['artist_count']]:
        tracks = sorted((r for r in candidates if r['artist_id'] == artist),
                        key=lambda r: key('track', r['track_id']))
        expected.extend(r['track_id'] for r in tracks[:config['tracks_per_artist']])
    selected = read(out / 'selected_manifest.json')['tracks']
    selected_map = unique_rows(selected, checks, 'Selected')
    checks.equal([r['track_id'] for r in selected], expected, 'sample_accounting',
                 'Selected order independently reconstructed from hash seed')
    checks.check(len({r['artist_id'] for r in selected}) == config['artist_count'],
                 'sample_accounting', 'Selected artist count')
    indexes = {shard: read(out / f'archive_index_{shard}.json') for shard in config['archives']}
    for row in selected:
        track = row['track_id']
        checks.check(all(row[k] == v for k, v in candidate_map[track].items()),
                     'sample_accounting', 'Selected source metadata: ' + track)
        index = indexes[row['path'][:2]]
        expected_archive = {**index['members'][row['archive_member']],
                            'archive_url': index['archive_url'], 'archive_bytes': index['archive_bytes']}
        checks.check(index['complete'] and row['archive'] == expected_archive,
                     'sample_accounting', 'Selected archive position: ' + track)
        checks.check(row['audio_file'] == str((out / 'audio' / f'{track}.wav').relative_to(ROOT)),
                     'sample_accounting', 'Selected output path: ' + track)
    return selected, selected_map, old_ids, old_artists


def check_acquisition(out, config, selected, selected_map, checks):
    acquisition = read(out / 'download_status.json')
    success = unique_rows(acquisition['successful'], checks, 'Download successes')
    failed = unique_rows(acquisition['failures'], checks, 'Download failures')
    checks.check(not (set(success) & set(failed)) and set(success) | set(failed) == set(selected_map),
                 'sample_accounting', 'All selected IDs accounted for in acquisition')
    checks.check(acquisition['requested'] == len(selected) and acquisition['replacement_tracks'] == 0,
                 'sample_accounting', 'Acquisition requested count and no replacement')
    for track, record in failed.items():
        checks.check(bool(record.get('error')), 'sample_accounting', 'Download failure reason: ' + track)
    for track, record in success.items():
        receipt_path = out / 'acquisition/download_receipts' / f'{track}.json'
        checks.check(read(receipt_path) == record, 'receipts', 'Receipt equals download status: ' + track)
        checks.hashes[str(receipt_path.relative_to(ROOT))] = digest(receipt_path)
        row = selected_map[track]
        wav = ROOT / row['audio_file']
        checks.file_hash(wav, record['wav_sha256'], 'Acquired WAV hash: ' + track)
        with wave.open(str(wav), 'rb') as audio:
            checks.check(audio.getnframes() == record['frames'] == 661500 and
                         audio.getframerate() == record['sample_rate'] == 22050 and
                         audio.getnchannels() == 1 and audio.getsampwidth() == 2,
                         'receipts', 'WAV duration and format: ' + track)
        total = 0
        previous_end = None
        archive = row['archive']
        for index, byte_range in enumerate(record['byte_ranges']):
            start, end = byte_range['byte_range']
            cache = ROOT / byte_range['cache_file']
            checks.file_hash(cache, byte_range['sha256'], 'Encoded cache hash: ' + track)
            count = end - start + 1
            total += count
            checks.check(count > 0 and cache.stat().st_size == count and
                         archive['offset'] <= start <= end < archive['offset'] + archive['size'],
                         'receipts', 'Encoded byte range geometry: ' + track)
            checks.check(previous_end is None or start > previous_end,
                         'receipts', 'Acquired ranges are disjoint: ' + track)
            previous_end = end
            if index == 0:
                checks.check(start == archive['offset'] and record['prefix_sha256'] == byte_range['sha256'],
                             'receipts', 'Initial prefix identity: ' + track)
        checks.check(total == record['downloaded_bytes'] and 0 < total <= config['maximum_prefix_bytes_per_track'],
                     'receipts', 'Acquisition byte budget: ' + track)
        full = len(record['byte_ranges']) == 1 and total == archive['size']
        checks.check(record['full_mp3_sha256_verified'] == full, 'receipts', 'Full-file verification claim: ' + track)
        if full:
            checks.check(record['prefix_sha256'] == row['expected_full_sha256'],
                         'receipts', 'Official complete-member checksum: ' + track)
    return acquisition, success, failed


def check_features(out, selected, selected_map, old_ids, old_artists, acquisition, success, failed, checks):
    status = read(out / 'extraction_status.json')
    checks.check(status['state'] == 'complete', 'features', 'Extraction completed')
    for name, key in [('selected_manifest.json', 'selected_manifest_sha256'),
                      ('manifest.json', 'observed_manifest_sha256'),
                      ('download_status.json', 'download_status_sha256'),
                      ('runtime_control.json', 'runtime_control_sha256')]:
        checks.file_hash(out / name, status[key], 'Extraction input hash: ' + name)
    extraction_failed = unique_rows(status['extraction_failures'], checks, 'Extraction failures')
    checks.check(set(extraction_failed).issubset(success), 'sample_accounting', 'Only acquired tracks can fail extraction')
    checks.check(status['download_failures'] == acquisition['failures'], 'sample_accounting', 'Download failure records preserved')
    for track, record in extraction_failed.items():
        checks.check(bool(record.get('error')), 'sample_accounting', 'Extraction failure reason: ' + track)
    rows = read(out / 'manifest.json')['tracks']
    observed = unique_rows(rows, checks, 'Observed')
    expected_rows = [r for r in selected if r['track_id'] in success and r['track_id'] not in extraction_failed]
    checks.check(rows == expected_rows, 'sample_accounting', 'Observed subset preserves selection order and metadata')
    checks.check(not (set(observed) & old_ids) and not ({r['artist_id'] for r in rows} & old_artists),
                 'sample_accounting', 'Observed IDs and artists disjoint from all project history')
    checks.check(set(observed) | set(failed) | set(extraction_failed) == set(selected_map),
                 'sample_accounting', 'Every selected track appears in one observed/failure category')
    metadata = read(out / 'features.json')
    checks.file_hash(out / 'features.npy', metadata['feature_sha256'], 'Feature array hash')
    features = np.load(out / 'features.npy', allow_pickle=False)
    checks.equal(metadata['ids'], [r['track_id'] for r in rows], 'features', 'Feature order')
    checks.check(features.shape == (len(rows), 2304) and list(features.shape) == metadata['shape']
                 and features.dtype == np.float32 and metadata['dtype'] == 'float32'
                 and np.isfinite(features).all(), 'features', 'Feature shape, dtype and finite values')
    checks.equal(metadata['audio_hashes'], [success[r['track_id']]['wav_sha256'] for r in rows],
                 'features', 'Feature audio hash order')
    checks.file_hash(ROOT / 'maest_hf_features.py', metadata['extractor_sha256'], 'Frozen feature extractor')
    checks.file_hash(ROOT / 'models/mtg-upf-maest-519l/SOURCES.json', metadata['encoder_receipt_sha256'], 'Encoder receipt')
    control = read(out / 'runtime_control.json')
    old_metadata = read(ROOT / 'outputs/maest_hf/features.json')
    checks.check(metadata['device'] == control['device'] == 'cpu' and
                 metadata['versions'] == control['versions'] == old_metadata['versions'],
                 'features', 'Historical encoder runtime and device reproduced')
    checks.check(control['track_id'] == old_metadata['ids'][0] and
                 0 <= control['max_absolute_feature_error'] <= control['absolute_tolerance'] <= 1e-5,
                 'features', 'Historical feature control tolerance')
    historical = read(ROOT / 'data/expanded_manifest.json')['tracks']
    control_row = next(r for r in historical if r['track_id'] == control['track_id'])
    checks.file_hash(ROOT / control_row['audio_file'], control['audio_sha256'], 'Historical control audio hash')
    ontology = read(ROOT / 'experiments/final_holdout_plan.json')['ontology']
    y = np.array([[int(any(tag in ontology[label] for tag in row['tags']))
                   for label in LABELS] for row in rows], dtype=int)
    artists = np.array([r['artist_id'] for r in rows])
    checks.equal(status['positive_tracks'], y.sum(0), 'sample_accounting', 'Observed positive track counts')
    checks.equal(status['positive_artists'], [len(set(artists[y[:, j] == 1])) for j in range(4)],
                 'sample_accounting', 'Observed positive artist counts')
    checks.check(status['selected_tracks'] == len(selected) and
                 status['selected_artists'] == len({r['artist_id'] for r in selected}) and
                 status['observed_tracks'] == len(rows) and status['observed_artists'] == len(set(artists)),
                 'sample_accounting', 'Extraction track and artist counts')
    return rows, features, y, artists


def check_predictions(out, rows, features, y, artists, checks):
    reference = read(out / 'model_reference.json')
    seed = reference['primary_seed']
    parameters = read(ROOT / reference['base_run'] / f'seed_{seed}' / 'parameters.json')
    policy = read(ROOT / reference['conservative_run'] / 'selection' / f'{seed}_policy.json')
    checks.check(seed == read(out / 'config.json')['primary_seed'] == 20260926,
                 'predictions', 'Originally designated primary model retained')
    x = np.asarray(features, dtype=np.float64)
    z = np.empty((len(rows), 4), dtype=float)
    for j, label in enumerate(LABELS):
        head = parameters['heads'][label]
        checks.equal(head['classes'], [0, 1], 'predictions', label + ': classifier class order')
        centered = (x - np.asarray(head['scaler_mean'])) / np.asarray(head['scaler_scale'])
        # Feature-wise sum is independent of production matrix multiplication.
        z[:, j] = np.sum(centered * np.asarray(head['coef'])[0], axis=1) + head['intercept'][0]
    raw, full, conservative = expit(z), np.empty_like(z), np.empty_like(z)
    for j, label in enumerate(LABELS):
        fitted = parameters['calibrators']['sigmoid'][label]
        item = policy['labels'][label]
        strength = item['selected_lambda']
        checks.check(fitted['success'] and strength in (0., .25, .5, .75, 1.),
                     'predictions', label + ': fixed parameters valid')
        policy_fit = item['full_fit']
        checks.check((policy_fit is None and strength == 0) or (policy_fit is not None
                     and policy_fit['success'] and policy_fit['slope'] == fitted['slope']
                     and policy_fit['intercept'] == fitted['intercept']),
                     'predictions', label + ': policy full fit matches original sigmoid')
        full[:, j] = expit(fitted['slope'] * z[:, j] + fitted['intercept'])
        conservative[:, j] = raw[:, j] * (1 - strength) + full[:, j] * strength
    with np.load(out / 'predictions.npz', allow_pickle=False) as stored:
        result = {key: stored[key] for key in stored.files}
    checks.equal(result['ids'], [r['track_id'] for r in rows], 'predictions', 'Prediction IDs')
    checks.equal(result['artists'], artists, 'predictions', 'Prediction artists')
    checks.equal(result['y'], y, 'predictions', 'Binary labels independently reconstructed from frozen ontology')
    checks.close(result['logits'], z, 'predictions', 'Saved-head logits reconstructed independently')
    checks.equal(result['identity'], expit(result['logits']), 'predictions', 'Raw probabilities exactly preserve saved logits')
    for method, rebuilt in [('identity', raw), ('sigmoid', full), ('conservative_sigmoid', conservative)]:
        checks.close(result[method], rebuilt, 'predictions', 'Probability formula: ' + method)
        checks.check(result[method].shape == y.shape and np.isfinite(result[method]).all()
                     and ((result[method] >= 0) & (result[method] <= 1)).all(),
                     'predictions', 'Finite bounded aligned probabilities: ' + method)
    unchanged = []
    for j, label in enumerate(LABELS):
        if policy['labels'][label]['selected_lambda'] == 0:
            checks.equal(result['conservative_sigmoid'][:, j], result['identity'][:, j],
                         'predictions', 'Zero-strength label exactly unchanged: ' + label)
            unchanged.append(label)
    return result, policy, unchanged


def check_metrics(out, predictions, policy, checks):
    metrics = read(out / 'metrics.json')
    y = predictions['y']
    summary = read(out / 'summary.json')
    checks.check(summary['primary_seed'] == read(out / 'model_reference.json')['primary_seed'],
                 'metrics', 'Summary primary model seed')
    expected_methods = []
    for method in METHODS:
        p = predictions[method]
        brier, logarithmic = [], []
        checks.check(metrics[method]['n'] == len(y), 'metrics', method + ': track count')
        for j, label in enumerate(LABELS):
            item = metrics[method]['per_label'][label]
            brier.append(brier_score_loss(y[:, j], p[:, j]))
            logarithmic.append(log_loss(y[:, j], np.clip(p[:, j], 1e-12, 1 - 1e-12), labels=[0, 1]))
            checks.close([item['brier'], item['log_loss'], item['mean_score'], item['positive_fraction']],
                         [brier[-1], logarithmic[-1], p[:, j].mean(), y[:, j].mean()],
                         'metrics', method + '/' + label + ': sklearn metrics and averages')
            if 0 < y[:, j].sum() < len(y):
                checks.close(item['ap'], average_precision_score(y[:, j], p[:, j]),
                             'metrics', method + '/' + label + ': sklearn AP')
            else:
                checks.check(item['ap'] is None, 'metrics', method + '/' + label + ': undefined AP')
        checks.close([metrics[method]['macro_brier'], metrics[method]['macro_log_loss']],
                     [np.mean(brier), np.mean(logarithmic)], 'metrics', method + ': macro metrics')
        expected_methods.append({'method': method, 'macro_brier': metrics[method]['macro_brier'],
            'macro_log_loss': metrics[method]['macro_log_loss'],
            'delta_brier_vs_raw': metrics[method]['macro_brier'] - metrics['identity']['macro_brier'],
            'delta_brier_vs_full_sigmoid': metrics[method]['macro_brier'] - metrics['sigmoid']['macro_brier']})
    checks.check(summary['methods'] == expected_methods, 'metrics', 'Summary matches per-method metrics')
    checks.check(summary['tracks'] == len(y) and summary['artists'] == len(set(predictions['artists'])),
                 'metrics', 'Summary population counts')
    checks.check(summary['selected_strengths'] == {label: policy['labels'][label]['selected_lambda'] for label in LABELS},
                 'metrics', 'Summary uses saved policy strengths')


def check_bootstrap(out, config, data, checks):
    y, artists_by_row = data['y'], data['artists']
    artists = np.unique(artists_by_row)
    clusters = [np.flatnonzero(artists_by_row == artist) for artist in artists]
    sizes = np.asarray([len(indices) for indices in clusters])
    losses = {}
    for method in METHODS:
        p = data[method]
        q = np.clip(p, 1e-12, 1 - 1e-12)
        losses[method] = {'brier': (p - y) ** 2,
                          'log_loss': -(y * np.log(q) + (1 - y) * np.log1p(-q))}
    repetitions = config['bootstrap_replicates']
    rng = np.random.default_rng(config['bootstrap_seed'])
    weights = np.zeros((repetitions, len(artists)), dtype=int)
    for i in range(repetitions):
        for artist in rng.integers(0, len(artists), size=len(artists)):
            weights[i, artist] += 1
    counts = weights @ sizes
    intervals = read(out / 'intervals.json')
    for name, baseline in [('raw', 'identity'), ('full_sigmoid', 'sigmoid')]:
        interval = intervals[name]
        checks.check(interval['replicates'] == repetitions and interval['seed'] == config['bootstrap_seed']
                     and interval['artists'] == len(artists) and interval['baseline'] == baseline,
                     'bootstrap', name + ': bootstrap metadata')
        with np.load(out / f'{name}_draws.npz', allow_pickle=False) as saved:
            for metric in ['brier', 'log_loss']:
                delta = losses['conservative_sigmoid'][metric] - losses[baseline][metric]
                totals = np.array([delta[ix].sum(0) for ix in clusters])
                by_label = weights @ totals / counts[:, None]
                rebuilt = np.column_stack([by_label.mean(1), by_label])
                checks.check(saved[metric].shape == (repetitions, 5), 'bootstrap', name + '/' + metric + ': draw shape')
                checks.close(saved[metric][0], rebuilt[0], 'bootstrap_first_draw', name + '/' + metric + ': first shared sample')
                checks.close(saved[metric], rebuilt, 'bootstrap', name + '/' + metric + ': all shared artist samples')
                lower, upper = np.quantile(rebuilt, [.025, .975], axis=0)
                point = np.r_[delta.mean(), delta.mean(0)]
                for j, label in enumerate(['macro'] + LABELS):
                    result = interval['results']['conservative_sigmoid'][metric][label]
                    checks.close([result['delta'], *result['ci95']], [point[j], lower[j], upper[j]],
                                 'bootstrap', name + '/' + metric + '/' + label + ': paired interval')
    summary = read(out / 'summary.json')
    checks.check(summary['primary_interval'] == intervals['raw']['results']['conservative_sigmoid']['brier']['macro']
                 and summary['secondary_interval'] == intervals['full_sigmoid']['results']['conservative_sigmoid']['brier']['macro'],
                 'bootstrap', 'Summary intervals match paired comparisons')
    return {'replicates_rebuilt_per_comparison': repetitions, 'comparisons': 2,
            'first_sample_tracks': int(counts[0]), 'sorted_artist_ids': artists.tolist(),
            'first_sample_artist_multiplicities': weights[0].tolist(),
            'shared_sampling_check': 'Both comparison arrays independently match the same regenerated artist samples.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_dir():
        parser.error('An existing completed validation output directory is required')
    checks, details = Checks(), {}
    try:
        frozen = verify_frozen(out)
        checks.check(True, 'frozen_inputs', 'Production frozen-input verification')
        for name, expected in frozen['source_hashes'].items():
            checks.file_hash(out / 'source_snapshot' / name, expected, 'Frozen source snapshot: ' + name)
        status = read(out / 'score_status.json')
        checks.check(status['state'] == 'complete' and status['classifier_or_calibrator_refitted'] is False,
                     'frozen_inputs', 'Completed score receipt with no refitting')
        expected_inputs = {'freeze.json', 'manifest.json', 'features.json', 'features.npy',
                           'model_reference.json', 'model_hashes.json', 'numerical_checks.json'}
        expected_outputs = {'predictions.npz', 'metrics.json', 'summary.json', 'intervals.json',
                            'raw_draws.npz', 'full_sigmoid_draws.npz'}
        checks.check(set(status['input_hashes']) == expected_inputs and set(status['output_hashes']) == expected_outputs,
                     'hashes', 'Score status contains every input/output hash')
        for group in ['input_hashes', 'output_hashes']:
            for name, expected in status[group].items():
                checks.file_hash(out / name, expected, 'Score artifact hash: ' + name)
        for name, expected in status['acquisition_hashes'].items():
            checks.file_hash(ROOT / name, expected, 'Score acquisition artifact hash: ' + name)
        config = read(out / 'config.json')
        selected, selected_map, old_ids, old_artists = check_sample(out, config, checks)
        acquisition, success, failed = check_acquisition(out, config, selected, selected_map, checks)
        rows, features, y, artists = check_features(out, selected, selected_map, old_ids, old_artists,
                                                   acquisition, success, failed, checks)
        acquisition_paths = [out / name for name in ['selected_manifest.json', 'download_status.json',
                                                     'extraction_status.json', 'runtime_control.json']]
        for row in rows:
            acquisition_paths += [out / 'acquisition/download_receipts' / f"{row['track_id']}.json",
                                  ROOT / row['audio_file']]
        checks.check(set(status['acquisition_hashes']) == {str(p.relative_to(ROOT)) for p in acquisition_paths},
                     'hashes', 'Score status includes every observed acquisition artifact')
        predictions, policy, unchanged = check_predictions(out, rows, features, y, artists, checks)
        check_metrics(out, predictions, policy, checks)
        details['bootstrap'] = check_bootstrap(out, config, predictions, checks)
        details.update(selected_tracks=len(selected), observed_tracks=len(rows),
                       observed_artists=len(set(artists)), unchanged_labels=unchanged)
        # Detect mutation during the verification itself as well.
        verify_frozen(out)
        for name, expected in list(checks.hashes.items()):
            checks.check(digest(ROOT / name) == expected, 'hashes', 'Artifact unchanged during checks: ' + name)
    except Exception as error:
        checks.errors.append(f'{type(error).__name__}: {error}')
    report = {'created_utc': now(), 'status': 'passed' if not checks.errors else 'failed',
              'scope': 'Independent numerical/artifact verification; synthetic unit-test results are separate.',
              'checks_per_category': checks.counts, 'checks_total': sum(checks.counts.values()),
              'errors': checks.errors, 'verified_artifact_hashes': checks.hashes, **details}
    save(out / 'verification.json', report)
    print(f"Numerical verification: {report['status']}; {report['checks_total']} checks; {len(checks.errors)} errors")
    for error in checks.errors:
        print(error)
    if checks.errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
