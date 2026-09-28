"""Read-only cache audit and explicit retirement of previously exposed cohorts.

This module never extracts features, fits models, or changes historical records.
Call save_retirement_ledger before any fit in the new development experiment.
"""
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('electronic', 'pop', 'ambient', 'rock')
COHORTS = ('dev1206', 'hist239', 'val266', 'fresh204')
COUNTS = {'dev1206': (1206, 469), 'hist239': (239, 89),
          'val266': (266, 200), 'fresh204': (204, 150)}
POSITIVES = {'dev1206': [435, 261, 244, 220], 'hist239': [70, 50, 48, 36],
             'val266': [101, 58, 55, 45], 'fresh204': [84, 26, 52, 36]}
VALIDATION = 'outputs/calibration_validation/20260927_v1'
FRESH = 'outputs/application_fresh_validation/20260927_v1'
ENCODER = 'models/mtg-upf-maest-519l/SOURCES.json'
EXTRACTOR = 'maest_hf_features.py'
PACKAGES = ('torch', 'transformers', 'safetensors', 'numpy', 'soundfile', 'soxr')


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def safe_path(root, name):
    name = Path(name)
    require(not name.is_absolute() and '..' not in name.parts, 'Unsafe input path')
    path = (Path(root) / name).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Input escapes project')
    return path


def broad_targets(rows, ontology):
    require(set(ontology) == set(LABELS), 'Ontology label identity')
    require(all(isinstance(tags, list) and tags and all(isinstance(tag, str) and tag for tag in tags)
                for tags in ontology.values()), 'Invalid ontology entries')
    for row in rows:
        require(isinstance(row.get('tags'), list)
                and all(isinstance(tag, str) and tag for tag in row['tags']), 'Invalid source tags')
    return np.asarray([[int(bool(set(row['tags']) & set(ontology[label])))
                        for label in LABELS] for row in rows], dtype=np.int64).reshape(-1, 4)


def validate_rows(rows):
    require(isinstance(rows, list) and bool(rows), 'Empty/invalid cohort rows')
    for row in rows:
        require(isinstance(row, dict), 'Invalid track record')
        require(isinstance(row.get('track_id'), str)
                and re.fullmatch(r'track_\d{7}', row['track_id']), 'Invalid track ID')
        require(isinstance(row.get('artist_id'), str)
                and re.fullmatch(r'artist_\d{6}', row['artist_id']), 'Invalid artist ID')
    require(len({row['track_id'] for row in rows}) == len(rows), 'Duplicate track ID')


def validate_cache(path, metadata, rows, *, width=2304):
    """Validate bytes, exact manifest order, and finite float32 representation."""
    validate_rows(rows)
    ids = [row['track_id'] for row in rows]
    require(metadata['ids'] == ids, 'Feature ID/order mismatch')
    require(metadata['feature_sha256'] == digest(path), 'Feature hash mismatch')
    x = np.load(path, allow_pickle=False, mmap_mode='r')
    require(x.shape == (len(rows), width) and x.dtype == np.float32
            and np.isfinite(x).all(), 'Invalid feature geometry/dtype/values')
    require(metadata['shape'] == list(x.shape) and metadata['dtype'] == 'float32',
            'Feature metadata geometry')
    require(len(metadata['audio_hashes']) == len(rows)
            and all(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha)
                    for sha in metadata['audio_hashes']), 'Audio hash provenance')
    return np.asarray(x, dtype=np.float64)


def combine_cohorts(parts, ontology, *, expected_counts=COUNTS):
    """Copy source records, retaining their original roles and narrow targets."""
    require([name for name, _, _ in parts] == list(expected_counts), 'Cohort ordering')
    all_rows, arrays, cohorts, tracks, artists = [], [], [], set(), set()
    for cohort, rows, x in parts:
        validate_rows(rows)
        cohort_ids = {row['track_id'] for row in rows}
        cohort_artists = {row['artist_id'] for row in rows}
        require(not tracks & cohort_ids, 'Track overlap between cohorts')
        require(not artists & cohort_artists, 'Artist overlap between cohorts')
        require((len(rows), len(cohort_artists)) == expected_counts[cohort], 'Cohort counts')
        require(x.ndim == 2 and len(x) == len(rows) and x.dtype == np.float64
                and np.isfinite(x).all(), 'Invalid converted features')
        if arrays:
            require(x.shape[1] == arrays[0].shape[1], 'Different representation widths')
        y = broad_targets(rows, ontology)
        for row, target in zip(rows, y):
            require(not set(row) & {'cohort', 'original_role', 'current_role', 'proxy_labels'},
                    'Source record already has new role fields')
            original_role = row.get('phase_role', {
                'val266': 'observed_calibration_validation',
                'fresh204': 'observed_fixed_application_validation'}.get(cohort))
            require(isinstance(original_role, str) and original_role, 'Missing original role')
            all_rows.append({**deepcopy(row), 'cohort': cohort, 'original_role': original_role,
                             'current_role': 'development_only', 'proxy_labels': target.tolist()})
        tracks.update(cohort_ids); artists.update(cohort_artists)
        arrays.append(x); cohorts.extend([cohort] * len(rows))
    return (all_rows, np.vstack(arrays), broad_targets(all_rows, ontology),
            np.asarray([row['artist_id'] for row in all_rows], dtype=str), np.asarray(cohorts, dtype=str))


def cohort_artist_folds(artists, cohorts, n_splits, seed):
    """Label-free greedy balance within each cohort, never split an artist.

    Sort cohort names; within each, descending track count then SHA256 of
    ``seed|cohort|artist``. Assign to least cohort tracks, cohort artists, total
    tracks, then fold index. Every cohort must have at least n_splits artists.
    """
    artists, cohorts = np.asarray(artists), np.asarray(cohorts)
    require(artists.ndim == cohorts.ndim == 1 and artists.shape == cohorts.shape and artists.size,
            'Invalid artist/cohort vectors')
    require(artists.dtype.kind in 'US' and cohorts.dtype.kind in 'US'
            and all(str(a) for a in artists) and all(str(c) for c in cohorts), 'Invalid grouping strings')
    require(isinstance(n_splits, (int, np.integer)) and not isinstance(n_splits, bool)
            and n_splits >= 2, 'Invalid fold count')
    require(isinstance(seed, (str, int)) and not isinstance(seed, bool), 'Invalid fold seed')
    ownership = {}
    for artist, cohort in zip(artists.tolist(), cohorts.tolist()):
        require(artist not in ownership or ownership[artist] == cohort, 'Artist spans cohorts')
        ownership[artist] = cohort
    assignments, total_tracks = {}, np.zeros(n_splits, dtype=np.int64)
    for cohort in sorted(set(cohorts.tolist())):
        sizes = Counter(artists[cohorts == cohort].tolist())
        require(len(sizes) >= n_splits, 'Too few artists in a cohort for every fold')
        order = sorted(sizes, key=lambda a: (-sizes[a],
            hashlib.sha256(f'{seed}|{cohort}|{a}'.encode()).hexdigest(), a))
        track_counts, artist_counts = np.zeros(n_splits, dtype=np.int64), np.zeros(n_splits, dtype=np.int64)
        for artist in order:
            fold = min(range(n_splits), key=lambda k: (track_counts[k], artist_counts[k], total_tracks[k], k))
            assignments[artist] = fold
            track_counts[fold] += sizes[artist]; artist_counts[fold] += 1; total_tracks[fold] += sizes[artist]
    return np.asarray([assignments[a] for a in artists], dtype=np.int64)


def load_pool(root=ROOT):
    """Return rows, x64, y64, artist/cohort strings, audit, and frozen input Paths."""
    root = Path(root).resolve(); hashes = {}

    def checked(name, expected=None):
        path = safe_path(root, name)
        sha = digest(path)
        require(expected is None or sha == expected, 'Provenance hash mismatch: ' + str(name))
        relative = str(path.relative_to(root))
        require(relative not in hashes or hashes[relative] == sha, 'Input changed during audit: ' + relative)
        hashes[relative] = sha
        return path

    def read(name, expected=None):
        return json.loads(checked(name, expected).read_text(encoding='utf-8'))

    plan = read('experiments/final_holdout_plan.json')
    require(plan['labels'] == list(LABELS), 'Original ontology label order')
    original = read('data/expanded_manifest.json', plan['expanded_manifest_sha256'])['tracks']
    validate_rows(original)
    by_id = {row['track_id']: row for row in original}
    require(len(original) == 1535 and len({r['artist_id'] for r in original}) == 583, 'Original frame changed')
    require(len(set(plan['fit_ids'])) == 1206 and len(set(plan['holdout_ids'])) == 239, 'Plan IDs')
    development = [by_id[i] for i in plan['fit_ids']]
    historical = [by_id[i] for i in plan['holdout_ids']]
    require(all(r['phase_role'] in ('train', 'development_validation') for r in development), 'Development roles')
    require(all(r['phase_role'] == 'new_holdout' for r in historical), 'Historical roles')
    encoder = read(ENCODER, plan['model_receipt_sha256'])
    checked(EXTRACTOR, plan['feature_source_sha256'])
    for name, sha in encoder['files_sha256'].items():
        checked(str(Path(ENCODER).parent / name), sha)
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    parts, cohort_audits = [], {}

    def cache(cohort, rows, name, meta, feature_key, encoder_key):
        require(meta['versions'] == versions and meta['device'] == 'cpu', 'Different encoder runtime/device')
        require(meta[feature_key] == hashes[EXTRACTOR] and meta[encoder_key] == hashes[ENCODER],
                'Different encoder representation')
        x = validate_cache(checked(name, meta['feature_sha256']), meta, rows)
        parts.append((cohort, rows, x))
        cohort_audits[cohort] = {'tracks': len(rows), 'artists': len({r['artist_id'] for r in rows}),
            'positive_counts': broad_targets(rows, plan['ontology']).sum(0).tolist(),
            'feature_file': name, 'feature_sha256': meta['feature_sha256'],
            'ids_sha256': hashlib.sha256('\n'.join(meta['ids']).encode()).hexdigest(),
            'audio_hashes_verified_against_original_receipts': True}
        require(cohort_audits[cohort]['positive_counts'] == POSITIVES[cohort], 'Unexpected target counts')

    devmeta = read('outputs/maest_hf/features.json', plan['development_feature_meta_sha256'])
    require(devmeta['feature_sha256'] == plan['development_features_sha256'] and devmeta['test_tracks'] == 0,
            'Original development cache identity')
    for key, name in {'expanded_manifest_sha256': 'data/expanded_manifest.json',
                      'model_plan_sha256': 'experiments/maest_hf_plan.json',
                      'driver_source_sha256': 'extract_expanded_maest_hf.py'}.items():
        checked(name, devmeta[key])
    public = read('data/evaluation/development_feature_provenance.json')
    require(all(devmeta[k] == v for k, v in public.items()), 'Public provenance changed')
    dev_audit = read('data/expanded_development_audit.json')
    require(dev_audit['manifest_sha256'] == plan['expanded_manifest_sha256']
            and set(dev_audit['tracks']) == set(plan['fit_ids']) and dev_audit['verified_tracks'] == 1206,
            'Original development audio audit')
    require(devmeta['audio_hashes'] == [dev_audit['tracks'][r['track_id']]['wav_sha256'] for r in development],
            'Development audio receipt mismatch')
    cache('dev1206', development, 'outputs/maest_hf/features.npy', devmeta,
          'extractor_source_sha256', 'encoder_manifest_sha256')

    histmeta = read('outputs/final_maest/holdout_features.json')
    for key, name in {'plan_sha256': 'experiments/final_holdout_plan.json',
                      'expanded_manifest_sha256': 'data/expanded_manifest.json',
                      'extractor_source_sha256': 'extract_holdout_maest_hf.py'}.items():
        checked(name, histmeta[key])
    hist_audit = read('data/final_holdout_audit.json', histmeta['holdout_audit_sha256'])
    require(hist_audit['plan_sha256'] == histmeta['plan_sha256']
            and hist_audit['manifest_sha256'] == plan['expanded_manifest_sha256']
            and set(hist_audit['tracks']) == set(plan['holdout_ids'])
            and hist_audit['verified_tracks'] == 239 and hist_audit['historical_test_audio_used'] == 0
            and histmeta['historical_test_tracks'] == 0, 'Historical audio audit identity')
    require(histmeta['audio_hashes'] == [hist_audit['tracks'][r['track_id']]['wav_sha256'] for r in historical],
            'Historical audio receipt mismatch')
    cache('hist239', historical, 'outputs/final_maest/holdout_features.npy', histmeta,
          'feature_source_sha256', 'encoder_manifest_sha256')

    candidate_frames = []
    for cohort, base in [('val266', VALIDATION), ('fresh204', FRESH)]:
        frozen = read(base + '/freeze.json')
        # Only extraction and label provenance are relevant here; no historical scores are loaded.
        for name in (EXTRACTOR, 'prepare_dataset.py', 'prepare_maest_hf.py', 'prepare_mert.py',
                     'research/' + ('calibration_validation' if cohort == 'val266' else 'application_fresh_validation') + '/pipeline.py'):
            checked(name, frozen['source_hashes'][name])
        for name in (ENCODER, 'experiments/final_holdout_plan.json', 'data/expanded_manifest.json'):
            checked(name, frozen['input_hashes'][name])
        local = {}
        for name in ('selected_manifest.json', 'candidate_pool.json', 'config.json', 'exclusions.json',
                     'source/genres.tsv', 'source/licenses.txt', 'source/checksums.txt'):
            path = checked(base + '/' + name, frozen['local_hashes'][name])
            if name.endswith('.json'):
                local[name] = json.loads(path.read_text())
        config = local['config.json']
        for name, expected in config['source_blobs'].items():
            content = safe_path(root, base + '/source/' + name).read_bytes()
            require(hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest() == expected,
                    'Official metadata Git blob changed')
        candidate_frames.append(local['candidate_pool.json']['tracks'])
        selected = local['selected_manifest.json']['tracks']; validate_rows(selected)
        status = read(base + '/extraction_status.json')
        require(status['state'] == 'complete' and not status['extraction_failures'], 'Incomplete extraction')
        for key, name in [('selected_manifest_sha256', 'selected_manifest.json'),
                          ('observed_manifest_sha256', 'manifest.json'),
                          ('download_status_sha256', 'download_status.json'),
                          ('runtime_control_sha256', 'runtime_control.json')]:
            checked(base + '/' + name, status[key])
        rows = read(base + '/manifest.json')['tracks']; validate_rows(rows)
        downloads = read(base + '/download_status.json')
        success = {r['track_id']: r for r in downloads['successful']}
        failures = {r['track_id']: r for r in downloads['failures']}
        require(len(success) == len(downloads['successful']) and len(failures) == len(downloads['failures'])
                and not set(success) & set(failures)
                and set(success) | set(failures) == {r['track_id'] for r in selected}
                and downloads['requested'] == len(selected) and downloads['replacement_tracks'] == 0,
                'Acquisition accounting')
        require(rows == [r for r in selected if r['track_id'] in success], 'Observed manifest/order changed')
        require(status['download_failures'] == downloads['failures']
                and status['observed_tracks'] == len(rows)
                and status['observed_artists'] == len({r['artist_id'] for r in rows})
                and status['selected_tracks'] == len(selected)
                and status['selected_artists'] == len({r['artist_id'] for r in selected}), 'Extraction counts')
        meta = read(base + '/features.json')
        require(meta['audio_hashes'] == [success[r['track_id']]['wav_sha256'] for r in rows], 'Acquired audio hashes')
        for row in rows:
            require(read(base + '/acquisition/download_receipts/' + row['track_id'] + '.json')
                    == success[row['track_id']], 'Original audio receipt changed')
        control = read(base + '/runtime_control.json')
        require(control['track_id'] == devmeta['ids'][0] and control['audio_sha256'] == devmeta['audio_hashes'][0]
                and control['device'] == 'cpu' and control['versions'] == versions
                and np.isfinite(control['max_absolute_feature_error'])
                and 0 <= control['max_absolute_feature_error'] <= control['absolute_tolerance'] <= 1e-5,
                'Historical encoder control failed')
        if cohort == 'fresh204':
            require(meta['freeze_sha256'] == hashes[base + '/freeze.json'] == control['freeze_sha256']
                    and meta['download_status_sha256'] == status['download_status_sha256'], 'Fresh extraction chain')
        cache(cohort, rows, base + '/features.npy', meta, 'extractor_sha256', 'encoder_receipt_sha256')
        require(status['positive_tracks'] == cohort_audits[cohort]['positive_counts'], 'Extracted target counts')

    rows, x, y, artists, cohorts = combine_cohorts(parts, plan['ontology'])
    require(len(rows) == 1915 and len(set(artists)) == 908, 'Expanded pool dimensions')
    require(not {r['track_id'] for r in rows} & set(plan['historical_test_ids']), 'Old 90-track test entered pool')
    # Use the complete candidate frames, including unselected and failed acquisitions.
    prior = read(FRESH + '/exclusions.json')
    for name, sha in prior['source_hashes'].items():
        checked(name, sha)
    union_tracks = {r['track_id'] for frame in [original, *candidate_frames] for r in frame}
    union_artists = {r['artist_id'] for frame in [original, *candidate_frames] for r in frame}
    require(union_tracks == set(prior['tracks']) | {r['track_id'] for r in candidate_frames[-1]}
            and union_artists == set(prior['artists']) | {r['artist_id'] for r in candidate_frames[-1]}
            and (len(union_tracks), len(union_artists)) == (2195, 1008), 'Future exclusion boundary changed')
    for name in ('genres.tsv', 'licenses.txt', 'checksums.txt'):
        require(hashes[VALIDATION + '/source/' + name] == hashes[FRESH + '/source/' + name],
                'Different official metadata versions between cohorts')

    # Old manifests sometimes retain a subset of irrelevant fine genres. Preserve those records;
    # require identical four-label targets and never replace historical tags with current metadata.
    source_rows = {}
    with safe_path(root, VALIDATION + '/source/genres.tsv').open(newline='') as stream:
        reader = csv.reader(stream, delimiter='\t'); next(reader)
        for values in reader:
            require(len(values) >= 5 and values[0] not in source_rows, 'Invalid official metadata row')
            require(all(tag.startswith('genre---') for tag in values[5:]), 'Wrong source namespace')
            source_rows[values[0]] = {'artist_id': values[1], 'path': values[3],
                                     'tags': [tag.removeprefix('genre---') for tag in values[5:]]}
    differences = []
    for row in rows:
        source = source_rows[row['track_id']]
        require(row['artist_id'] == source['artist_id'] and row['path'] == source['path']
                and set(row['tags']) <= set(source['tags'])
                and np.array_equal(broad_targets([row], plan['ontology']), broad_targets([source], plan['ontology'])),
                'Source identity or broad target mismatch')
        if set(row['tags']) != set(source['tags']):
            differences.append(row['track_id'])
    audit = {'status': 'passed', 'tracks': len(rows), 'artists': len(set(artists)),
        'labels': list(LABELS), 'ontology': deepcopy(plan['ontology']), 'cohort_order': list(COHORTS),
        'cohorts': cohort_audits, 'positive_counts': y.sum(0).tolist(),
        'feature_shape': list(x.shape), 'feature_dtype': str(x.dtype),
        'encoder_receipt_sha256': hashes[ENCODER], 'extractor_sha256': hashes[EXTRACTOR],
        'encoder_runtime': versions, 'device': 'cpu', 'input_hashes': dict(sorted(hashes.items())),
        'prior_excluded_tracks': 1929, 'prior_excluded_artists': 837,
        'future_exclusions': {'tracks': sorted(union_tracks), 'artists': sorted(union_artists)},
        'fine_tag_subset_differences': differences, 'broad_target_source_disagreements': 0,
        'old_90_test_tracks_in_pool': 0,
        'old_90_test_artists_overlapping_pool': len(set(artists) &
            {by_id[i]['artist_id'] for i in plan['historical_test_ids']}),
        'limits': ['All rows are exposed development data; no cohort is fresh for future models.',
                   'Labels are unchanged source-tag proxies, not human-confirmed musical truth.',
                   'Original records and old results remain unchanged; earlier fixed-model claims retain their original scope.',
                   'Feature bytes and recorded audio receipts checked; audio decoding/extraction not repeated.',
                   'Artist aliases, external exposure and upstream pretraining overlap remain unknown.']}
    return rows, x, y, artists, cohorts, audit, [safe_path(root, name) for name in sorted(hashes)]


def save_retirement_ledger(out, rows, audit):
    """Publish immutable, deterministic new ledgers; call before freezing/fitting."""
    require(audit.get('status') == 'passed' and len(rows) == audit['tracks'], 'Pool audit not passed')
    validate_rows(rows)
    require(len({r['artist_id'] for r in rows}) == audit['artists'], 'Ledger artist count')
    require(all(r['current_role'] == 'development_only' for r in rows), 'Pool role mismatch')
    boundary = audit['future_exclusions']
    require({r['track_id'] for r in rows} <= set(boundary['tracks'])
            and {r['artist_id'] for r in rows} <= set(boundary['artists']), 'Retired pool missing from future exclusions')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        original = {k: v for k, v in row.items()
                    if k not in ('cohort', 'original_role', 'current_role', 'proxy_labels')}
        records.append({'track_id': row['track_id'], 'artist_id': row['artist_id'], 'cohort': row['cohort'],
            'original_role': row['original_role'], 'original_phase_role': row.get('phase_role'),
            'original_record_sha256': hashlib.sha256(json.dumps(original, sort_keys=True, separators=(',', ':'),
                                                     allow_nan=False).encode()).hexdigest(),
            'current_role': 'development_only', 'future_fresh_eligible': False,
            'source_tags': deepcopy(row['tags']), 'proxy_targets': list(row['proxy_labels'])})
    ledger = {'format_version': 1, 'stage': 'retirement before any new model fit',
        'tracks': audit['tracks'], 'artists': audit['artists'], 'labels': audit['labels'],
        'source_input_hashes': audit['input_hashes'], 'records': records, 'limits': audit['limits']}
    exclusions = {**deepcopy(audit['future_exclusions']), 'format_version': 1,
        'scope': 'Entire historical and candidate frames, including unselected and failed tracks; exclude all artist IDs.',
        'future_fresh_eligible': False, 'source_input_hashes': audit['input_hashes']}
    values = {'role_ledger.json': ledger, 'future_exclusions.json': exclusions}
    contents = {name: (json.dumps(value, indent=2, allow_nan=False) + '\n').encode() for name, value in values.items()}
    for name, content in contents.items():
        require(not (out / name).exists() or (out / name).read_bytes() == content,
                'Refusing to overwrite a different retirement ledger')
    for name, content in contents.items():
        path = out / name
        if path.exists():
            continue
        fd, temporary = tempfile.mkstemp(prefix='.' + name, suffix='.tmp', dir=out)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(content); stream.flush(); os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            os.unlink(temporary)
    return {name: digest(out / name) for name in values}
