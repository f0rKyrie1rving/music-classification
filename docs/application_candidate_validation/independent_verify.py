"""Independent fixed-candidate confirmation audit; never fits or selects a model.

Reconstructs sampling, application arithmetic, confusion/loss statistics and
paired artist draws separately from the confirmation implementation.
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
import tarfile
import sys

import soundfile as sf

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('electronic', 'pop', 'ambient', 'rock')
POLICIES = ('f1', 'precision_target')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            result.update(chunk)
    return result.hexdigest()


def archive(path):
    with np.load(path, allow_pickle=False) as values:
        return dict(values)


class Audit:
    def __init__(self):
        self.checks = 0
        self.maximum_absolute_difference = 0.

    def check(self, condition, context):
        self.checks += 1
        if not bool(condition):
            raise AssertionError(context)

    def same(self, actual, expected, context, tolerance=1e-12):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict) and set(actual) == set(expected), context+'/keys')
            for key in expected:
                self.same(actual[key], expected[key], context+'/'+str(key), tolerance)
        elif isinstance(expected, (list, tuple)):
            self.check(len(actual) == len(expected), context+'/length')
            for index, value in enumerate(expected):
                self.same(actual[index], value, context+'/'+str(index), tolerance)
        elif isinstance(expected, np.ndarray):
            actual = np.asarray(actual)
            self.check(actual.shape == expected.shape, context+'/shape')
            if expected.dtype.kind == 'f':
                self.check(np.array_equal(np.isfinite(actual), np.isfinite(expected)), context+'/finite')
                finite = np.isfinite(expected)
                difference = float(np.max(np.abs(actual[finite]-expected[finite]))) if finite.any() else 0.
                self.maximum_absolute_difference = max(self.maximum_absolute_difference, difference)
                self.check(np.allclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True), context+'/values')
            else:
                self.check(np.array_equal(actual, expected), context+'/values')
        elif isinstance(expected, float):
            self.check(actual is not None and np.isfinite(actual), context+'/finite')
            self.maximum_absolute_difference = max(self.maximum_absolute_difference, abs(actual-expected))
            self.check(abs(actual-expected) <= tolerance*(1+abs(expected)), context)
        else:
            self.check(actual == expected, context)


def sigmoid(z):
    return np.exp(-np.logaddexp(0., -np.asarray(z)))


def confusion(y, decisions):
    y, decisions = np.asarray(y, bool), np.asarray(decisions, bool)
    tp, fp = int((y & decisions).sum()), int((~y & decisions).sum())
    fn, tn = int((y & ~decisions).sum()), int((~y & ~decisions).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
        precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def losses(y, probability):
    clipped = np.clip(probability, 1e-12, 1-1e-12)
    return {'brier': np.square(probability-y),
            'log_loss': -(y*np.log(clipped)+(1-y)*np.log1p(-clipped))}


def average_precision(y, probability):
    """Noninterpolated AP, grouping equal scores before each precision step."""
    positive = int(np.sum(y))
    if positive == 0 or positive == len(y):
        return None
    order = np.argsort(-probability, kind='stable')
    outcome, score = y[order], probability[order]
    ends = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score)-1]
    cumulative = np.cumsum(outcome)[ends]
    return float(np.sum(np.diff(np.r_[0, cumulative]) * cumulative/(ends+1))/positive)


def predictions(x, arrays, intercepts, offsets, thresholds):
    """Installed row-sum arithmetic; never a matrix multiplication substitute."""
    means, scales, coefficients = arrays
    z = np.asarray([np.sum(((row-means)/scales)*coefficients, axis=1)+intercepts for row in x])
    raw = sigmoid(z)
    probability = raw.copy()
    changed = offsets != 0
    probability[:, changed] = sigmoid(z[:, changed]+offsets[changed])
    return {'logits': z, 'raw': raw, 'probability': probability,
            'decisions': {policy: raw >= np.asarray(thresholds[policy]) for policy in POLICIES}}


def frame_and_selection(out, config, exclusions):
    licenses = {}
    for block in (out/'source/licenses.txt').read_text().split('\n\n'):
        lines = block.strip().splitlines()
        if lines:
            code = re.findall(r'https?://creativecommons.org/licenses/([^/\s]+)/[^\s]+', block)
            licenses[lines[0]] = (code[0] if len(code) == 1 else None, block.strip())
    checksums = dict(line.split()[::-1] for line in (out/'source/checksums.txt').read_text().splitlines())
    frame = []
    with (out/'source/genres.tsv').open(newline='') as stream:
        for values in list(csv.reader(stream, delimiter='\t'))[1:]:
            track, artist, album, path, duration, *tags = values
            member = path.replace('.mp3', '.low.mp3')
            if (path[:2] in config['archives'] and track not in exclusions['tracks']
                    and artist not in exclusions['artists'] and float(duration) >= 30
                    and licenses.get(path, (None,))[0] in config['allowed_license_codes'] and member in checksums):
                frame.append({'track_id': track, 'artist_id': artist, 'album_id': album,
                    'path': path, 'duration': float(duration),
                    'tags': [tag.removeprefix('genre---') for tag in tags],
                    'archive_member': member, 'expected_full_sha256': checksums[member],
                    'license_code': licenses[path][0], 'attribution_verbatim': licenses[path][1]})
    frame.sort(key=lambda row: row['track_id'])
    def key(kind, value):
        return hashlib.sha256(f"{config['sample_seed']}|{kind}|{value}".encode()).hexdigest()
    artists = sorted({row['artist_id'] for row in frame}, key=lambda artist: key('artist', artist))
    chosen = []
    for artist in artists[:config['artist_count']]:
        songs = sorted((row for row in frame if row['artist_id'] == artist), key=lambda row: key('track', row['track_id']))
        chosen.extend(songs[:config['tracks_per_artist']])
    return frame, chosen


def metrics(y, p, artists, decisions):
    loss = losses(y, p)
    result = {'tracks': len(y), 'artists': len(set(artists)),
              **{'macro_'+key: float(value.mean()) for key, value in loss.items()},
              'per_label': {}, 'policies': {}}
    for j, label in enumerate(LABELS):
        positive = y[:, j] == 1
        result['per_label'][label] = {
            'positive_tracks': int(positive.sum()), 'negative_tracks': int((~positive).sum()),
            'positive_artists': len(set(artists[positive])),
            'negative_artists': len(set(artists[~positive])),
            'positive_fraction': float(positive.mean()), 'mean_score': float(p[:, j].mean()),
            **{key: float(value[:, j].mean()) for key, value in loss.items()},
            'average_precision': average_precision(y[:, j], p[:, j])}
    for policy, d in decisions.items():
        counts = [confusion(y[:, j], d[:, j]) for j in range(4)]
        micro = confusion(y, d)
        focus_fp = counts[1]['fp']+counts[2]['fp']
        result['policies'][policy] = {
            'true_positives': micro['tp'], 'false_positives': micro['fp'],
            'false_negatives': micro['fn'], 'focus_false_positives': focus_fp,
            'focus_false_positive_rate': focus_fp/len(y),
            **{'micro_'+key: micro[key] for key in ['precision', 'recall', 'f1']},
            'macro_f1': float(np.mean([row['f1'] for row in counts])),
            'coverage': float(np.any(d, axis=1).mean()),
            'per_label': {label: {**{key: counts[j][key] for key in ['precision', 'recall', 'f1']},
                'true_positives': counts[j]['tp'], 'false_positives': counts[j]['fp'],
                'false_negatives': counts[j]['fn'], 'positive_decisions': int(d[:, j].sum())}
                for j, label in enumerate(LABELS)}}
    return result


def bootstrap(y, arms, artists, config):
    """Directly concatenate sampled artists' rows; no cluster-sum implementation."""
    unique = np.unique(artists)
    indices = [np.flatnonzero(artists == artist) for artist in unique]
    sample = np.random.default_rng(config['bootstrap_seed']).integers(
        0, len(unique), (config['bootstrap_replicates'], len(unique)))
    names = ['macro_brier', 'macro_log_loss', 'micro_f1', 'micro_recall',
             'focus_fp_rate', 'focus_fp_count_equivalent']
    def values(rows, arm):
        target = y[rows]
        d, p = arm['decisions']['f1'][rows], arm['probability'][rows]
        loss = losses(target, p)
        c = confusion(target, d)
        rate = float(np.sum(d[:, [1, 2]] & (target[:, [1, 2]] == 0))/len(rows))
        return ({'macro_brier': float(loss['brier'].mean()),
                 'macro_log_loss': float(loss['log_loss'].mean()),
                 'micro_f1': c['f1'], 'micro_recall': c['recall'],
                 'focus_fp_rate': rate, 'focus_fp_count_equivalent': rate*len(y)},
                {key: value.mean(0) for key, value in loss.items()})
    rows = np.arange(len(y))
    points = {name: values(rows, arm) for name, arm in arms.items()}
    saved = {'bootstrap_artists': unique, 'bootstrap_artist_indices': sample,
             'bootstrap_track_counts': np.zeros(len(sample), dtype=np.int64)}
    for name in names+['focus_fp_relative_change']:
        saved['bootstrap_delta_'+name] = np.empty(len(sample))
    for name in ['brier', 'log_loss']:
        saved['bootstrap_delta_per_label_'+name] = np.empty((len(sample), 4))
    for k, draw in enumerate(sample):
        rows = np.concatenate([indices[j] for j in draw])
        saved['bootstrap_track_counts'][k] = len(rows)
        before, before_label = values(rows, arms['baseline'])
        after, after_label = values(rows, arms['candidate'])
        for name in names:
            saved['bootstrap_delta_'+name][k] = after[name]-before[name]
        saved['bootstrap_delta_focus_fp_relative_change'][k] = (
            (after['focus_fp_rate']-before['focus_fp_rate'])/before['focus_fp_rate']
            if before['focus_fp_rate'] else np.nan)
        for name in ['brier', 'log_loss']:
            saved['bootstrap_delta_per_label_'+name][k] = after_label[name]-before_label[name]
    def interval(before, after, delta):
        finite = delta[np.isfinite(delta)]
        return {'baseline': before, 'candidate': after,
                'delta': after-before if after is not None and before is not None else None,
                'ci95': np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                'valid_replicates': len(finite)}
    report = {'metrics': {}, 'per_label_losses': {}}
    for name in names:
        report['metrics'][name] = interval(points['baseline'][0][name], points['candidate'][0][name],
                                           saved['bootstrap_delta_'+name])
    before, after = [points[name][0]['focus_fp_rate'] for name in ['baseline', 'candidate']]
    relative = (after-before)/before if before else None
    draw = saved['bootstrap_delta_focus_fp_relative_change']
    report['metrics']['focus_fp_relative_change'] = {
        **interval(0., relative, draw), 'undefined_replicates': int(np.isnan(draw).sum())}
    for name in ['brier', 'log_loss']:
        report['per_label_losses'][name] = {
            label: interval(float(points['baseline'][1][name][j]), float(points['candidate'][1][name][j]),
                saved['bootstrap_delta_per_label_'+name][:, j]) for j, label in enumerate(LABELS)}
    return report, saved


def preflight(out, audit):
    config, frozen = read(out/'config.json'), read(out/'freeze.json')
    for group, base in [('source_hashes', ROOT), ('input_hashes', ROOT), ('local_hashes', out)]:
        for name, expected in frozen[group].items():
            audit.check(sha(base/name) == expected, group+'/'+name)
            if group == 'source_hashes':
                audit.check(sha(out/'source_snapshot'/name) == expected, 'source_snapshot/'+name)
    audit.check(frozen['refitted'] is False, 'no_model_refit')
    for versions in [frozen['runtime'], frozen['evaluation_runtime']]:
        for package, version in versions.items():
            audit.same(importlib.metadata.version(package), version, 'runtime/'+package)
    model_names = [str(path.relative_to(ROOT)) for path in (ROOT/config['candidate_directory']).iterdir()]
    model_names += ['artifacts/final_heads.npy', 'artifacts/final_heads.json', 'artifacts/score_correction.json']
    for name in model_names:
        audit.check(sha(out/'model_snapshot'/name) == frozen['input_hashes'][name], 'model_snapshot/'+name)
    for name, expected in config['source_blobs'].items():
        data = (out/'source'/name).read_bytes()
        audit.check(hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest() == expected,
                    'official_metadata/'+name)
    excluded = read(out/'exclusions.json')
    historical = read(ROOT/config['historical_exclusions'])
    for key, count in [('tracks', 2195), ('artists', 1008)]:
        audit.same(excluded[key], historical[key], 'historical_exclusions/'+key)
        audit.check(len(set(excluded[key])) == len(excluded[key]) == count, 'exclusion_count/'+key)
    frame, selected = frame_and_selection(out, config, excluded)
    audit.same(read(out/'candidate_pool.json')['tracks'], frame, 'independent_official_frame')
    audit.check(len(frame) == 640 and len({r['artist_id'] for r in frame}) == 288, 'frame_size')
    recorded = read(out/'selected_manifest.json')['tracks']
    for row in selected:
        row['audio_file'] = str((out/'audio'/(row['track_id']+'.wav')).relative_to(ROOT))
    audit.same(recorded, selected, 'independent_hash_selection')
    audit.check(len(selected) == 372 and len({r['artist_id'] for r in selected}) == 250, 'selection_size')
    ids = [r['track_id'] for r in selected]
    audit.check(hashlib.sha256(json.dumps(ids).encode()).hexdigest() ==
                '192714883ffba98f15d7d14b522af405b7ada81999db92af5eb88c8a4c56606a',
                'preprediction_independent_selection_commitment')
    future = read(out/'future_exclusions.json')
    for key, field, count in [('tracks', 'track_id', 2835), ('artists', 'artist_id', 1296)]:
        expected = sorted(set(excluded[key]) | {r[field] for r in frame})
        audit.same(future[key], expected, 'future_exclusions/'+key)
        audit.check(len(expected) == count, 'future_count/'+key)
    return config, frozen, frame, selected


def archive_indices(out, audit, config, frame, selected):
    indices = {}
    for shard in config['archives']:
        state = read(out/f'archive_index_{shard}.json')
        wanted = {r['archive_member'] for r in frame if r['path'][:2] == shard}
        offset, count, rebuilt, visited = 0, 0, {}, {}
        while True:
            name = f'index_headers/{shard}/{offset}.bin'
            block = (out/name).read_bytes()
            audit.check(len(block) == 512 and sha(out/name) == state['header_hashes'][name], name+'/hash')
            visited[name] = sha(out/name)
            if not block.strip(b'\0'):
                break
            header = tarfile.TarInfo.frombuf(block, 'utf-8', 'strict')
            audit.check(header.size >= 0, name+'/size')
            if header.isfile() and header.name in wanted:
                audit.check(header.name not in rebuilt, name+'/unique_member')
                rebuilt[header.name] = {'offset': offset+512, 'size': header.size}
            next_offset = offset+512+512*((header.size+511)//512)
            audit.check(offset < next_offset <= state['archive_bytes'], name+'/geometry')
            offset, count = next_offset, count+1
            audit.check(count <= 2000, name+'/bounded_headers')
        audit.same(rebuilt, state['members'], shard+'/independent_tar_index')
        audit.same(visited, state['header_hashes'], shard+'/all_headers_accounted')
        audit.check(set(rebuilt) == wanted and state['complete'] and state['headers_scanned'] == count,
                    shard+'/complete_frame_index')
        audit.check(1 <= len(state['invocations']) <= 3, shard+'/bounded_invocations')
        audit.same(state['identity'], {'freeze_sha256': sha(out/'freeze.json'),
                   'selection_sha256': sha(out/'selected_manifest.json')}, shard+'/index_identity')
        indices[shard] = state
    resolved = read(out/'resolved_manifest.json')
    expected = []
    for row in selected:
        state = indices[row['path'][:2]]
        expected.append({**row, 'archive': {**state['members'][row['archive_member']],
            'archive_url': state['archive_url'], 'archive_bytes': state['archive_bytes']}})
    audit.same(resolved, {'tracks': expected, 'freeze_sha256': sha(out/'freeze.json'),
        'selection_sha256': sha(out/'selected_manifest.json'),
        'index_hashes': {f'archive_index_{s}.json': sha(out/f'archive_index_{s}.json') for s in config['archives']}},
        'resolved_selection_identity')
    return expected


def audio_and_features(out, audit, config, selected):
    acquisition = read(out/'download_status.json')
    success = {row['track_id']: row for row in acquisition['successful']}
    failed = {row['track_id']: row for row in acquisition['failures']}
    ids = [row['track_id'] for row in selected]
    identity = {'freeze_sha256': sha(out/'freeze.json'), 'selection_sha256': sha(out/'selected_manifest.json'),
                'resolved_sha256': sha(out/'resolved_manifest.json')}
    audit.same(acquisition['identity'], identity, 'acquisition_identity')
    audit.check(acquisition['requested'] == len(ids) and acquisition['replacement_tracks'] == 0, 'acquisition_counts')
    audit.check(len(success) == len(acquisition['successful']) and len(failed) == len(acquisition['failures'])
                and not set(success)&set(failed) and set(success)|set(failed) == set(ids), 'exact_selected_partition')
    audit.check({p.stem for p in (out/'download_checkpoints').glob('*.json')} == set(ids), 'all_download_checkpoints')
    feature_rows, vectors, audio_hashes, extraction_failures = [], [], [], []
    full_files, prefix_files = 0, 0
    for row in selected:
        track = row['track_id']
        checkpoint = read(out/'download_checkpoints'/f'{track}.json')
        audit.same(checkpoint['identity'], identity, track+'/checkpoint_identity')
        audit.same(checkpoint['track_id'], track, track+'/checkpoint_track')
        if track in failed:
            audit.check(checkpoint['state'] == 'failed' and checkpoint['error'] == failed[track]['error'], track+'/fixed_failure')
            continue
        receipt = success[track]
        audit.check(checkpoint['state'] == 'success', track+'/completed_acquisition')
        audit.same(checkpoint['receipt'], receipt, track+'/checkpoint_receipt')
        audit.same(read(out/'acquisition/download_receipts'/f'{track}.json'), receipt, track+'/download_receipt')
        audio = ROOT/row['audio_file']
        audit.check(audio.resolve() == (out/'audio'/f'{track}.wav').resolve(), track+'/audio_path')
        audit.check(sha(audio) == receipt['wav_sha256'], track+'/audio_hash')
        with sf.SoundFile(audio) as source:
            audit.check((source.frames, source.samplerate, source.channels, source.subtype) == (661500, 22050, 1, 'PCM_16'), track+'/audio_geometry')
            samples = source.read(dtype='float32')
        audit.check(np.isfinite(samples).all() and np.sqrt(np.mean(samples.astype(float)**2)) >= 1e-5, track+'/valid_signal')
        total, end = 0, row['archive']['offset']-1
        for record in receipt['byte_ranges']:
            start, final = record['byte_range']
            path = ROOT/record['cache_file']
            audit.check(path.resolve().is_relative_to((out/'acquisition/audio_prefixes').resolve()), track+'/prefix_path')
            audit.check(start > end and start >= row['archive']['offset'] and final >= start
                        and final < row['archive']['offset']+row['archive']['size'], track+'/prefix_range')
            audit.check(path.stat().st_size == final-start+1 and sha(path) == record['sha256'], track+'/prefix_bytes')
            total += final-start+1; end = final
        audit.check(0 < total <= config['maximum_prefix_bytes_per_track'] and total == receipt['downloaded_bytes'], track+'/bounded_bytes')
        audit.check(receipt['byte_ranges'][0]['byte_range'][0] == row['archive']['offset']
                    and receipt['prefix_sha256'] == receipt['byte_ranges'][0]['sha256'], track+'/initial_prefix')
        full = len(receipt['byte_ranges']) == 1 and total == row['archive']['size']
        audit.check(receipt['full_mp3_sha256_verified'] is full, track+'/checksum_claim')
        if full:
            audit.check(receipt['prefix_sha256'] == row['expected_full_sha256'], track+'/full_official_checksum')
            full_files += 1
        else:
            prefix_files += 1
        path = out/'feature_tracks'/f'{track}.json'
        feature = read(path)
        for name, value in {'track_id': track, 'audio_sha256': receipt['wav_sha256'],
                            'freeze_sha256': sha(out/'freeze.json'),
                            'download_status_sha256': sha(out/'download_status.json')}.items():
            audit.same(feature[name], value, track+'/feature_identity/'+name)
        if feature['status'] == 'failed':
            audit.check(isinstance(feature['error'], str) and bool(feature['error']), track+'/feature_failure_reason')
            extraction_failures.append({'track_id': track, 'error': feature['error']})
            continue
        audit.check(feature['status'] == 'complete' and sha(path.with_suffix('.npy')) == feature['feature_sha256'], track+'/feature_hash')
        vector = np.load(path.with_suffix('.npy'), allow_pickle=False)
        audit.check(vector.dtype == np.float32 and vector.shape == (2304,) and np.isfinite(vector).all(), track+'/feature_geometry')
        feature_rows.append(row); vectors.append(vector); audio_hashes.append(receipt['wav_sha256'])
    recorded = read(out/'manifest.json')['tracks']
    audit.same(recorded, feature_rows, 'observed_exact_ordered_subset')
    x = np.load(out/'features.npy', allow_pickle=False)
    audit.check(np.array_equal(x, np.stack(vectors)) and x.dtype == np.float32, 'aggregate_features')
    meta = read(out/'features.json')
    for key, value in {'ids': [r['track_id'] for r in recorded], 'shape': list(x.shape), 'dtype': 'float32',
            'feature_sha256': sha(out/'features.npy'), 'audio_hashes': audio_hashes,
            'freeze_sha256': sha(out/'freeze.json'), 'download_status_sha256': sha(out/'download_status.json'),
            'extractor_sha256': sha(ROOT/'maest_hf_features.py'),
            'encoder_receipt_sha256': sha(ROOT/'models/mtg-upf-maest-519l/SOURCES.json'), 'device': 'cpu'}.items():
        audit.same(meta[key], value, 'feature_metadata/'+key)
    status = read(out/'extraction_status.json')
    audit.same(status['download_failures'], acquisition['failures'], 'reported_download_failures')
    audit.same(status['extraction_failures'], extraction_failures, 'reported_extraction_failures')
    audit.check(status['state'] == 'complete', 'extraction_complete')
    for key, name in [('selected_manifest_sha256', 'selected_manifest.json'),
                      ('observed_manifest_sha256', 'manifest.json'), ('download_status_sha256', 'download_status.json'),
                      ('runtime_control_sha256', 'runtime_control.json')]:
        audit.same(status[key], sha(out/name), 'extraction_lineage/'+key)
    control = read(out/'runtime_control.json')
    audit.check(control['device'] == 'cpu' and control['max_absolute_feature_error'] <= 1e-5,
                'historical_encoder_control')
    return recorded, x.astype(np.float64), {'full_mp3_checksums': full_files, 'partial_prefixes': prefix_files,
                    'download_failures': len(failed), 'extraction_failures': len(extraction_failures)}


def assess_independently(metrics, intervals, coverage, y, artists, config):
    baseline, candidate = metrics['baseline'], metrics['candidate']
    before, after = baseline['policies']['f1'], candidate['policies']['f1']
    old_fp, new_fp = before['focus_false_positives'], after['focus_false_positives']
    reduction = 1-new_fp/old_fp if old_fp else None
    point = {'focus_fp_reduction': reduction is not None and reduction+1e-12 >= .1,
             'each_focus_fp_nonincrease': all(after['per_label'][label]['false_positives'] <= before['per_label'][label]['false_positives'] for label in ['pop', 'ambient']),
             'micro_recall_guard': after['micro_recall']+.03+1e-12 >= before['micro_recall'],
             'each_label_recall_guard': all(after['per_label'][label]['recall']+.05+1e-12 >= before['per_label'][label]['recall'] for label in LABELS),
             'micro_f1_nondecrease': after['micro_f1']+1e-12 >= before['micro_f1'],
             'macro_brier_nonincrease': candidate['macro_brier'] <= baseline['macro_brier']+1e-12,
             'macro_log_loss_nonincrease': candidate['macro_log_loss'] <= baseline['macro_log_loss']+1e-12,
             'electronic_rock_exact': True, 'precision_target_exact': True}
    evidence = {'track_coverage': coverage['track_fraction'] >= .8,
                'artist_coverage': coverage['artist_fraction'] >= .8,
                'observed_tracks': len(y) >= 250, 'observed_artists': len(set(artists)) >= 200}
    support = {}
    for j in [1, 2]:
        label = LABELS[j]; support[label] = {}
        for kind, target in [('positive', 1), ('negative', 0)]:
            mask = y[:, j] == target
            for unit, number in [('tracks', int(mask.sum())), ('artists', len(set(artists[mask])))]:
                key = kind+'_'+unit
                support[label][key] = number
                evidence[label+'_'+key] = number >= (30 if unit == 'tracks' else 20)
    confidence = {key+'_ci_upper_below_zero': intervals['metrics'][key]['ci95'] is not None
                  and intervals['metrics'][key]['ci95'][1] < 0 for key in ['macro_brier', 'focus_fp_rate']}
    verdict = ('incomplete_evidence' if not all(evidence.values()) else
               'not_passed' if not all(point.values()) else 'go' if all(confidence.values()) else 'promising')
    return {'verdict': verdict, 'go': verdict == 'go', 'point_gate_passed': all(point.values()),
            'point_checks': point, 'focus_fp_relative_reduction': reduction,
            'evidence_sufficient': all(evidence.values()), 'evidence_checks': evidence,
            'focus_support': support, 'confidence_checks': confidence}


def network_audit(out, audit, selected):
    by_id = {row['track_id']: row for row in selected}
    counts = Counter()
    for path in (out/'network_requests').glob('*.json'):
        record = read(path)
        start, end = record['range']
        context = record['context']
        counts[context['phase']] += 1
        audit.check(0 <= start <= end and end-start+1 <= 65536, path.name+'/bounded_range')
        audit.check(1 <= len(record['attempts']) <= 3, path.name+'/bounded_attempts')
        for k, entry in enumerate(record['attempts']):
            audit.check(entry['attempt'] == k+1 and entry['state'] in ['success', 'failed', 'started'], path.name+'/attempt_sequence')
            if entry['state'] != 'success':
                continue
            audit.check(k == len(record['attempts'])-1 and entry['response_bytes'] == end-start+1, path.name+'/terminal_success')
            if context['phase'] == 'index':
                audit.check(entry['index_header_file'] == f"index_headers/{context['shard']}/{start}.bin"
                            and sha(out/entry['index_header_file']) == entry['sha256'], path.name+'/header_bytes')
            elif context['phase'] == 'audio':
                row = by_id[context['track_id']]
                audit.check(record['url'] == row['archive']['archive_url'] and start >= row['archive']['offset']
                            and end < row['archive']['offset']+row['archive']['size'], path.name+'/selected_audio_range')
                receipt_path = out/'acquisition/download_receipts'/f"{row['track_id']}.json"
                if receipt_path.exists():
                    receipt = read(receipt_path)
                    prefix = ROOT/receipt['byte_ranges'][0]['cache_file']
                    begin = start-row['archive']['offset']
                    if begin+end-start+1 <= prefix.stat().st_size:
                        with prefix.open('rb') as stream:
                            stream.seek(begin); data = stream.read(end-start+1)
                        audit.check(hashlib.sha256(data).hexdigest() == entry['sha256'], path.name+'/audio_request_bytes')
            else:
                audit.check(False, path.name+'/known_network_phase')
    return dict(counts)


def verify(out):
    audit = Audit()
    config, frozen, frame, chosen = preflight(out, audit)
    selected = archive_indices(out, audit, config, frame, chosen)
    requests = network_audit(out, audit, selected)
    rows, x, acquisition = audio_and_features(out, audit, config, selected)
    metadata = read(ROOT/'artifacts/final_heads.json')
    parameters = np.load(ROOT/'artifacts/final_heads.npy', allow_pickle=False)
    correction = read(ROOT/'artifacts/score_correction.json')
    candidate_dir = ROOT/config['candidate_directory']
    candidate = read(candidate_dir/'candidate.json')
    cparams = archive(candidate_dir/'candidate.npz')
    audit.check(sha(candidate_dir/'candidate.json') == config['candidate_metadata_sha256']
                and sha(candidate_dir/'candidate.npz') == config['candidate_weights_sha256'], 'fixed_candidate_identity')
    arms = {'baseline': predictions(x, parameters, np.asarray(metadata['intercepts']),
            np.asarray(correction['logit_offsets']), metadata['thresholds']),
            'candidate': predictions(x, [cparams['means'], cparams['scales'], cparams['coefficients']],
            cparams['intercepts'], cparams['offsets'], candidate['thresholds'])}
    for name in ['logits', 'raw', 'probability']:
        audit.check(np.array_equal(arms['baseline'][name][:, [0, 3]], arms['candidate'][name][:, [0, 3]]), 'exact_copied_controls/'+name)
    for policy in POLICIES:
        audit.check(np.array_equal(arms['baseline']['decisions'][policy][:, [0, 3]],
                    arms['candidate']['decisions'][policy][:, [0, 3]]), 'exact_copied_controls/'+policy)
    audit.check(np.array_equal(arms['baseline']['decisions']['precision_target'],
                               arms['candidate']['decisions']['precision_target']), 'exact_selective_policy')
    ontology = read(ROOT/'experiments/final_holdout_plan.json')['ontology']
    y = np.asarray([[int(bool(set(row['tags'])&set(ontology[label]))) for label in LABELS] for row in rows], dtype=np.int64)
    ids = np.asarray([row['track_id'] for row in rows]); artists = np.asarray([row['artist_id'] for row in rows])
    expected = {'ids': ids, 'artists': artists, 'targets': y}
    for name, arm in arms.items():
        for key in ['logits', 'raw', 'probability']:
            expected[name+'__'+key] = arm[key]
        for policy, d in arm['decisions'].items():
            expected[name+'__'+policy+'__decision'] = d
    saved = archive(out/'predictions.npz')
    measured = {name: metrics(y, arm['probability'], artists, arm['decisions']) for name, arm in arms.items()}
    audit.same(read(out/'metrics.json'), measured, 'independent_metrics_and_ap')
    intervals, draws = bootstrap(y, arms, artists, config)
    expected.update(draws)
    audit.same(saved, expected, 'independent_predictions_and_direct_artist_draws')
    recorded_intervals = read(out/'intervals.json')
    for key, value in intervals.items():
        audit.same(recorded_intervals[key], value, 'independent_intervals/'+key)
    for key, value in {'seed': 2026092805, 'replicates': 2000, 'artists': len(set(artists))}.items():
        audit.same(recorded_intervals[key], value, 'bootstrap_fixed_settings/'+key)
    coverage = {'selected_tracks': len(selected), 'observed_tracks': len(rows),
                'selected_artists': len({r['artist_id'] for r in selected}), 'observed_artists': len(set(artists)),
                'track_fraction': len(rows)/len(selected), 'artist_fraction': len(set(artists))/250,
                'missing_ids': [r['track_id'] for r in selected if r['track_id'] not in set(ids)]}
    audit.same(read(out/'coverage.json'), coverage, 'independent_coverage')
    assessment = assess_independently(measured, intervals, coverage, y, artists, config)
    summary = read(out/'summary.json')
    for key, value in assessment.items():
        audit.same(summary['assessment'][key], value, 'independent_assessment/'+key)
    audit.same(summary['metrics'], read(out/'metrics.json'), 'summary_metrics')
    audit.same(summary['intervals'], recorded_intervals, 'summary_intervals')
    audit.same(summary['coverage'], coverage, 'summary_coverage')
    audit.check(summary['refitted'] is False and summary['candidate_changed'] is False
                and summary['application_replaced'] is False, 'no_refit_tuning_deployment')
    status = read(out/'evaluation_status.json')
    audit.check(status['state'] == 'complete' and status['freeze_sha256'] == sha(out/'freeze.json'), 'complete_frozen_evaluation')
    for name, expected_hash in status['output_hashes'].items():
        audit.check(sha(out/name) == expected_hash, 'evaluation_output_hash/'+name)
    # Validate installed public interfaces separately, retaining manual full-precision calculations.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import candidate_predict
    import predict_app
    base_bundle = {**metadata, 'means': parameters[0], 'scales': parameters[1], 'coefficients': parameters[2],
                   'intercepts': np.asarray(metadata['intercepts'])}
    candidate_bundle = {**candidate, **cparams}
    for i, row in enumerate(x):
        for policy in POLICIES:
            interfaces = {'baseline': predict_app.predict_vector(base_bundle, row, policy, 'weight_corrected', correction),
                          'candidate': candidate_predict.predict_vector(candidate_bundle, row, policy)}
            for arm, details in interfaces.items():
                for j, item in enumerate(details):
                    audit.check(item['selected'] == bool(arms[arm]['decisions'][policy][i, j])
                                and item['score'] == round(float(arms[arm]['probability'][i, j]), 4)
                                and item['raw_score'] == round(float(arms[arm]['raw'][i, j]), 4),
                                f'actual_interface/{i}/{policy}/{arm}/{j}')
            for j in [0, 3]:
                audit.same(interfaces['baseline'][j], interfaces['candidate'][j], f'copied_display/{i}/{policy}/{j}', 0.)
    result = {'state': 'passed', 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'checks': audit.checks, 'maximum_absolute_difference': audit.maximum_absolute_difference,
              'tracks': len(rows), 'artists': len(set(artists)), 'verdict': assessment['verdict'],
              'no_fitting_or_data_acquisition': True, 'bootstrap_reconstruction': 'direct artist row concatenation',
              'proxy_labels_unchanged': True, 'acquisition': acquisition, 'network_requests': requests,
              'freeze_sha256': sha(out/'freeze.json'), 'predictions_sha256': sha(out/'predictions.npz'),
              'summary_sha256': sha(out/'summary.json'), 'verifier_sha256': sha(Path(__file__))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/application_candidate_validation/20260928_v1')
    args = parser.parse_args()
    result = verify(args.out.resolve())
    (args.out/'independent_verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
