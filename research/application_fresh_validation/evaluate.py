"""One frozen comparison of the actual v1.1 application on new project artists.

No fitting, parameter selection, threshold selection or outcome-based sampling.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from predict_maest import (FEATURE_WIDTH, LABELS, POLICIES, METADATA, WEIGHTS,
                           PLAN, load_bundle, predict_vector as old_predict)
from predict_app import predict_vector as app_predict
from score_correction import CORRECTION, load_correction, corrected_thresholds

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 2026092802
MINIMUM_COVERAGE = .8
EPS = 1e-12
RESULT_FILES = ('predictions.npz', 'metrics.json', 'intervals.json', 'summary.json',
                'public_predictions.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


def targets(rows, ontology):
    return np.array([[int(bool(set(r['tags']) & set(ontology[label])))
                      for label in LABELS] for r in rows], dtype=int)


def validate(y, p, artists):
    y, p, artists = np.asarray(y), np.asarray(p, float), np.asarray(artists)
    if (not len(y) or y.shape != p.shape or y.ndim != 2 or y.shape[1] != 4
            or artists.shape != (len(y),) or not np.isin(y, [0, 1]).all()
            or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()):
        raise ValueError('Expected aligned binary outcomes, finite probabilities and artists')
    return y, p, artists


def loss_arrays(y, p):
    q = np.clip(p, EPS, 1-EPS)
    return {'brier': (p-y)**2, 'log_loss': -y*np.log(q)-(1-y)*np.log1p(-q)}


def probability_metrics(y, p, artists, decisions):
    y, p, artists = validate(y, p, artists)
    losses = loss_arrays(y, p)
    result = {'tracks': len(y), 'artists': len(set(artists)),
              **{'macro_'+k: float(v.mean()) for k, v in losses.items()},
              'per_label': {}, 'policies': {}}
    for j, label in enumerate(LABELS):
        pos = int(y[:, j].sum())
        result['per_label'][label] = {
            'positive_tracks': pos, 'positive_artists': len(set(artists[y[:, j] == 1])),
            'positive_fraction': float(y[:, j].mean()), 'mean_score': float(p[:, j].mean()),
            **{k: float(v[:, j].mean()) for k, v in losses.items()},
            'average_precision': (float(average_precision_score(y[:, j], p[:, j]))
                                  if 0 < pos < len(y) else None)}
    if set(decisions) != set(POLICIES):
        raise ValueError('Both fixed application policies must be evaluated')
    for policy, predicted in decisions.items():
        predicted = np.asarray(predicted)
        if predicted.shape != y.shape or not np.isin(predicted, [0, 1]).all():
            raise ValueError('Invalid policy decisions')
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, predicted, average='micro', zero_division=0)
        label_precision, label_recall, label_f1, _ = precision_recall_fscore_support(
            y, predicted, average=None, zero_division=0)
        fp, fn = ((predicted == 1) & (y == 0)).sum(0), ((predicted == 0) & (y == 1)).sum(0)
        result['policies'][policy] = {
            'micro_precision': float(precision), 'micro_recall': float(recall),
            'micro_f1': float(f1), 'macro_f1': float(label_f1.mean()),
            'false_positives': int(fp.sum()), 'false_negatives': int(fn.sum()),
            'coverage': float(predicted.any(1).mean()),
            'per_label': {label: {'precision': float(label_precision[j]),
                                   'recall': float(label_recall[j]), 'f1': float(label_f1[j]),
                                   'false_positives': int(fp[j]), 'false_negatives': int(fn[j]),
                                   'positive_decisions': int(predicted[:, j].sum())}
                          for j, label in enumerate(LABELS)}}
    return result


def paired_bootstrap(y, raw, corrected, artists, seed=BOOTSTRAP_SEED,
                     repetitions=BOOTSTRAP_REPLICATES):
    y, raw, artists = validate(y, raw, artists)
    _, corrected, _ = validate(y, corrected, artists)
    unique, inverse, sizes = np.unique(artists, return_inverse=True, return_counts=True)
    if len(unique) < 2 or repetitions < 2:
        raise ValueError('Bootstrap needs at least two observed artists and two replicates')
    sampled = np.random.default_rng(seed).integers(0, len(unique), size=(repetitions, len(unique)))
    raw_loss, fixed_loss = loss_arrays(y, raw), loss_arrays(y, corrected)
    result = {'replicates': repetitions, 'seed': seed, 'artists': len(unique),
              'direction': 'weight_corrected minus raw; negative favors correction',
              'scope': ('Pointwise paired artist-cluster percentile intervals conditional on the fixed '
                        'release and correction, with no refitting or multiplicity adjustment. '
                        'They do not account for missing tracks, finite-frame sampling fractions, '
                        'within-artist song selection, annotation error or pretraining exposure.'),
              'metrics': {}}
    draws = {'bootstrap_artists': unique, 'bootstrap_artist_indices': sampled}
    for metric in raw_loss:
        differences = fixed_loss[metric]-raw_loss[metric]
        artist_sums = np.array([np.bincount(inverse, weights=differences[:, j], minlength=len(unique))
                                for j in range(4)]).T
        label_draws = artist_sums[sampled].sum(1)/sizes[sampled].sum(1)[:, None]
        all_draws = np.column_stack([label_draws.mean(1), label_draws])
        draws['bootstrap_'+metric] = all_draws
        interval = np.quantile(all_draws, [.025, .975], axis=0)
        result['metrics'][metric] = {
            'delta': float(differences.mean()), 'ci95': interval[:, 0].tolist(),
            'per_label': {label: {'delta': float(differences[:, j].mean()),
                                  'ci95': interval[:, j+1].tolist()}
                          for j, label in enumerate(LABELS)}}
    return result, draws


def coverage_counts(selected, observed):
    expected = {r['track_id']: r for r in selected}
    ids = [r['track_id'] for r in observed]
    if not selected or len(expected) != len(selected) or not observed or len(ids) != len(set(ids)):
        raise ValueError('Selected/observed IDs must be nonempty and unique')
    if not set(ids).issubset(expected):
        raise ValueError('Observed tracks must come from the frozen selection without replacements')
    for row in observed:
        source = expected[row['track_id']]
        if row['artist_id'] != source['artist_id'] or row['tags'] != source['tags']:
            raise ValueError('Observed artist or tags changed after selection')
    selected_artists = {r['artist_id'] for r in selected}
    observed_artists = {r['artist_id'] for r in observed}
    return {'selected_tracks': len(selected), 'observed_tracks': len(observed),
            'selected_artists': len(selected_artists), 'observed_artists': len(observed_artists),
            'track_fraction': len(observed)/len(selected),
            'artist_fraction': len(observed_artists)/len(selected_artists),
            'minimum_required_fraction': MINIMUM_COVERAGE,
            'missing_ids': [r['track_id'] for r in selected if r['track_id'] not in set(ids)]}


def decision_rule(metrics, intervals, coverage, tag_changes):
    if tag_changes:
        raise ValueError('Integrity failure: application tag decisions changed')
    brier, ll = (intervals['metrics'][key] for key in ['brier', 'log_loss'])
    both_improve = all(metrics['weight_corrected']['macro_'+k] < metrics['raw']['macro_'+k]
                       for k in ['brier', 'log_loss'])
    complete_enough = min(coverage['track_fraction'], coverage['artist_fraction']) >= MINIMUM_COVERAGE
    if brier['ci95'][0] > 0 or ll['ci95'][0] > 0:
        verdict = 'harmful'
    elif not complete_enough:
        verdict = 'incomplete_evidence'
    elif both_improve and brier['ci95'][1] < 0:
        verdict = 'go'
    elif both_improve and brier['ci95'][0] <= 0 <= brier['ci95'][1]:
        verdict = 'promising'
    else:
        verdict = 'inconclusive'
    return {'verdict': verdict, 'go': verdict == 'go', 'both_point_estimates_improve': both_improve,
            'minimum_coverage_met': complete_enough, 'changed_tag_decisions': tag_changes,
            'interpretation': ('Frozen operational evidence rule, not a universal calibration guarantee, '
                               'proof of recognition improvement, or multiplicity-adjusted hypothesis test.')}


def predict_and_check(x, bundle, correction):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != FEATURE_WIDTH or not np.isfinite(x).all():
        raise ValueError('Invalid finite MAEST feature matrix')
    z = np.array([np.sum(((row-bundle['means'])/bundle['scales'])*bundle['coefficients'], axis=1)
                  +bundle['intercepts'] for row in x])
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite logits')
    raw = np.exp(-np.logaddexp(0., -z))
    offsets = np.asarray(correction['logit_offsets'], float)
    fixed = raw.copy(); changed = offsets != 0
    fixed[:, changed] = np.exp(-np.logaddexp(0., -(z[:, changed]+offsets[changed])))
    thresholds = {p: corrected_thresholds(bundle['thresholds'][p], offsets) for p in POLICIES}
    decisions = {p: raw >= np.asarray(bundle['thresholds'][p]) for p in POLICIES}
    transformed_mismatches = sum(int(np.sum(decisions[p] != (fixed >= thresholds[p]))) for p in POLICIES)
    if transformed_mismatches:
        raise ValueError('Integrity failure: transformed thresholds changed a decision')
    checks = 0
    for i, row in enumerate(x):
        for policy in POLICIES:
            old = old_predict(bundle, row, policy)
            new_raw = app_predict(bundle, row, policy, 'raw')
            new_fixed = app_predict(bundle, row, policy, 'weight_corrected', correction)
            for j, (a, b, c) in enumerate(zip(old, new_raw, new_fixed, strict=True)):
                if (any(a[k] != b[k] for k in a) or a['selected'] != c['selected']
                        or a['label'] != c['label'] or a['threshold'] != c['raw_threshold']
                        or c['selected'] != bool(decisions[policy][i, j])
                        or c['raw_score'] != round(float(raw[i, j]), 4)
                        or c['score'] != round(float(fixed[i, j]), 4)
                        or c['threshold'] != round(float(thresholds[policy][j]), 4)):
                    raise ValueError('Integrity failure: application interface differs from full-precision calculation')
                checks += 1
    return z, raw, fixed, decisions, thresholds, {'label_policy_interface_checks': checks,
                                                'changed_tag_decisions': 0,
                                                'transformed_threshold_mismatches': 0}


def evaluate(out):
    from .pipeline import (verify_frozen, acquisition_rows, verify_completed,
                           verify_audio_receipt)
    out = Path(out).resolve()
    if any((out/name).exists() for name in RESULT_FILES) or (out/'evaluation_status.json').exists():
        raise FileExistsError('Evaluation already started; retain the recorded run rather than overwrite it')
    frozen = verify_frozen(out)
    config = read(out/'config.json')
    if (config['bootstrap_seed'] != BOOTSTRAP_SEED or config['bootstrap_replicates'] != BOOTSTRAP_REPLICATES
            or config['minimum_observed_fraction'] != MINIMUM_COVERAGE):
        raise ValueError('Frozen evaluation settings differ from the implementation')
    needed_sources = ['predict_maest.py', 'predict_app.py', 'score_correction.py',
                      str(Path(__file__).relative_to(ROOT))]
    needed_inputs = ['artifacts/final_heads.npy', 'artifacts/final_heads.json',
                     'artifacts/score_correction.json', 'experiments/final_holdout_plan.json',
                     'data/expanded_manifest.json', 'app_version.txt']
    for names, section in [(needed_sources, 'source_hashes'), (needed_inputs, 'input_hashes')]:
        for name in names:
            if frozen[section].get(name) != digest(ROOT/name):
                raise ValueError('Required application model/source was not frozen: '+name)
    rows = read(out/'manifest.json')['tracks']
    selected = read(out/'selected_manifest.json')['tracks']
    _, successful, failed = acquisition_rows(out, selected)
    extraction = verify_completed(out, selected, successful, failed)
    for row in selected:
        if row['track_id'] in successful:
            verify_audio_receipt(out, row, successful[row['track_id']], config)
    coverage = coverage_counts(selected, rows)
    for name in ['selected_tracks', 'observed_tracks', 'selected_artists', 'observed_artists']:
        if extraction[name] != coverage[name]:
            raise ValueError('Extraction and evaluation coverage counts differ')
    ids = [r['track_id'] for r in rows]
    artists = np.array([r['artist_id'] for r in rows])
    excluded = read(out/'exclusions.json')
    if set(ids) & set(excluded['tracks']) or set(artists) & set(excluded['artists']):
        raise ValueError('Previously exposed project track/artist in evaluation')
    meta = read(out/'features.json')
    if (meta['ids'] != ids or meta['feature_sha256'] != digest(out/'features.npy')
            or meta['freeze_sha256'] != digest(out/'freeze.json')
            or meta['download_status_sha256'] != digest(out/'download_status.json')):
        raise ValueError('Feature identities or receipt hashes changed')
    x = np.load(out/'features.npy', allow_pickle=False)
    if x.shape != (len(rows), FEATURE_WIDTH) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Expected a complete float32 MAEST feature cache')
    plan = read(PLAN)
    original = {r['track_id']: r for r in read(ROOT/'data/expanded_manifest.json')['tracks']}
    fitrows = [original[i] for i in plan['fit_ids']]
    if set(ids) & set(original) or set(artists) & {r['artist_id'] for r in original.values()}:
        raise ValueError('Original project track/artist leakage')
    y = targets(rows, plan['ontology'])
    bundle = load_bundle()
    correction = load_correction(bundle, METADATA, WEIGHTS, PLAN)
    if (correction['fit_count'] != len(fitrows)
            or correction['fit_positive_counts'] != targets(fitrows, plan['ontology']).sum(0).tolist()):
        raise ValueError('Correction training counts no longer match the release')
    save(out/'evaluation_status.json', {'state': 'running', 'started_utc': now()})
    try:
        z, raw, fixed, decisions, thresholds, checks = predict_and_check(x, bundle, correction)
        metrics = {method: probability_metrics(y, p, artists, decisions)
                   for method, p in [('raw', raw), ('weight_corrected', fixed)]}
        intervals, draws = paired_bootstrap(y, raw, fixed, artists)
        assessment = decision_rule(metrics, intervals, coverage, checks['changed_tag_decisions'])
        summary = {'scope': 'Frozen actual-application comparison on new documented project track and artist IDs',
                   'labels': list(LABELS), 'coverage': coverage, 'assessment': assessment,
                   'interface_checks': checks,
                   'macro_brier': {m: v['macro_brier'] for m, v in metrics.items()},
                   'macro_log_loss': {m: v['macro_log_loss'] for m, v in metrics.items()},
                   'relative_brier_reduction': (1-metrics['weight_corrected']['macro_brier']/metrics['raw']['macro_brier']
                                                if metrics['raw']['macro_brier'] > 0 else None),
                   'limitations': ['Observed uploader tags are incomplete and apply to whole tracks, while audio uses the first 30 seconds.',
                       'Novelty is relative to audited project records; artist aliases, undocumented exposure and encoder pretraining overlap remain unknown.',
                       'The restricted four-shard licensed sample is not an external music catalogue or a streaming population.',
                       'Missing audio/features can bias the observed sample even if the prespecified coverage threshold is met.',
                       'All outcomes are retained. These tracks are consumed for future method-development decisions.']}
        np.savez_compressed(out/'predictions.npz', ids=np.array(ids), artists=artists, y=y,
                            logits=z, raw=raw, corrected=fixed,
                            **{'decisions_'+p: d for p, d in decisions.items()}, **draws)
        save(out/'metrics.json', metrics)
        save(out/'intervals.json', intervals)
        save(out/'summary.json', summary)
        public = {'scope': summary['scope'], 'labels': list(LABELS), 'coverage': coverage,
                  'model_weights_sha256': digest(WEIGHTS), 'model_metadata_sha256': digest(METADATA),
                  'correction_sha256': digest(CORRECTION), 'freeze_sha256': digest(out/'freeze.json'),
                  'feature_sha256': digest(out/'features.npy'), 'logit_offsets': correction['logit_offsets'],
                  'raw_thresholds': bundle['thresholds'],
                  'corrected_thresholds': {p: t.tolist() for p, t in thresholds.items()},
                  'rows': [{'track_id': r['track_id'], 'artist_id': r['artist_id'], 'source_tags': r['tags'],
                            'targets': y[i].tolist(), 'logits': z[i].tolist(),
                            'raw_scores': raw[i].tolist(), 'corrected_scores': fixed[i].tolist(),
                            'selected': {p: d[i].tolist() for p, d in decisions.items()}}
                           for i, r in enumerate(rows)]}
        save(out/'public_predictions.json', public)
        verify_frozen(out)
        save(out/'evaluation_status.json', {'state': 'complete', 'completed_utc': now(),
                                           'hashes': {name: digest(out/name) for name in RESULT_FILES},
                                           'verdict': assessment['verdict']})
        print(json.dumps({'assessment': assessment, 'coverage': coverage,
                          'macro_brier': summary['macro_brier'], 'macro_log_loss': summary['macro_log_loss']}, indent=2))
        return summary
    except Exception as error:
        save(out/'evaluation_status.json', {'state': 'failed', 'failed_utc': now(), 'error': repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    evaluate(parser.parse_args().out)


if __name__ == '__main__':
    main()
