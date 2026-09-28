"""Frozen paired evaluation of a fixed candidate against the actual v1.1 app.

Pure evaluation functions only: no fitting, sampling, downloading or file writes.
"""
import numpy as np
from sklearn.metrics import average_precision_score

import candidate_predict
import predict_app
from predict_maest import LABELS, POLICIES
from research.application_fresh_validation.evaluate import predict_and_check as baseline_predict

EPS = 1e-12
FOCUS = (1, 2)
BOOTSTRAP_SEED = 2026092805
BOOTSTRAP_REPLICATES = 2000
DEFAULT_CONFIG = {
    'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_replicates': BOOTSTRAP_REPLICATES,
    'focus': list(FOCUS), 'labels': list(LABELS), 'primary_policy': 'f1',
    'minimum_observed_fraction': .8,
    'minimum_observed_tracks': 250, 'minimum_observed_artists': 200,
    'minimum_focus_positive_tracks': 30, 'minimum_focus_negative_tracks': 30,
    'minimum_focus_positive_artists': 20, 'minimum_focus_negative_artists': 20,
    'gate': {'minimum_focus_fp_reduction': .1,
             'maximum_micro_recall_drop': .03, 'maximum_label_recall_drop': .05,
             'metric_tolerance': EPS, 'require_focus_fp_rate_ci_upper_below_zero': True,
             'require_macro_brier_ci_upper_below_zero': True},
}


def validate(y, p, artists):
    y, p, artists = np.asarray(y), np.asarray(p, float), np.asarray(artists)
    if (y.ndim != 2 or y.shape[1] != 4 or not len(y) or y.shape != p.shape
            or artists.shape != (len(y),) or not np.isin(y, [0, 1]).all()
            or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
            or any(not str(a) for a in artists)):
        raise ValueError('Expected aligned binary targets, probabilities and nonempty artist IDs')
    return y.astype(np.int64), p, artists


def validate_decisions(y, decisions):
    if set(decisions) != set(POLICIES):
        raise ValueError('Both fixed application policies must be evaluated')
    result = {}
    for policy, value in decisions.items():
        value = np.asarray(value)
        if value.shape != y.shape or not np.isin(value, [0, 1]).all():
            raise ValueError('Expected binary decisions aligned with target rows')
        result[policy] = value.astype(bool)
    return result


def loss_arrays(y, p):
    q = np.clip(p, EPS, 1-EPS)
    return {'brier': (p-y)**2, 'log_loss': -y*np.log(q)-(1-y)*np.log1p(-q)}


def _divide(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.zeros(a.shape, dtype=float), where=b != 0)


def probability_metrics(y, p, artists, decisions):
    y, p, artists = validate(y, p, artists)
    decisions = validate_decisions(y, decisions)
    losses = loss_arrays(y, p)
    result = {'tracks': len(y), 'artists': len(set(artists)),
              **{'macro_'+k: float(v.mean()) for k, v in losses.items()},
              'per_label': {}, 'policies': {}}
    for j, label in enumerate(LABELS):
        pos = y[:, j] == 1
        result['per_label'][label] = {
            'positive_tracks': int(pos.sum()), 'negative_tracks': int((~pos).sum()),
            'positive_artists': len(set(artists[pos])),
            'negative_artists': len(set(artists[~pos])),
            'positive_fraction': float(pos.mean()), 'mean_score': float(p[:, j].mean()),
            **{k: float(v[:, j].mean()) for k, v in losses.items()},
            'average_precision': float(average_precision_score(y[:, j], p[:, j]))
            if 0 < pos.sum() < len(y) else None}
    for policy, d in decisions.items():
        tp, fp, fn = (np.sum((d == 1) & (y == 1), 0),
                      np.sum((d == 1) & (y == 0), 0),
                      np.sum((d == 0) & (y == 1), 0))
        precision, recall, f1 = _divide(tp, tp+fp), _divide(tp, tp+fn), _divide(2*tp, 2*tp+fp+fn)
        result['policies'][policy] = {
            'true_positives': int(tp.sum()), 'false_positives': int(fp.sum()),
            'false_negatives': int(fn.sum()), 'focus_false_positives': int(fp[list(FOCUS)].sum()),
            'focus_false_positive_rate': float(fp[list(FOCUS)].sum()/len(y)),
            'micro_precision': float(_divide(tp.sum(), (tp+fp).sum())),
            'micro_recall': float(_divide(tp.sum(), (tp+fn).sum())),
            'micro_f1': float(_divide(2*tp.sum(), (2*tp+fp+fn).sum())),
            'macro_f1': float(f1.mean()), 'coverage': float(d.any(1).mean()),
            'per_label': {label: {'precision': float(precision[j]), 'recall': float(recall[j]),
                                  'f1': float(f1[j]), 'true_positives': int(tp[j]),
                                  'false_positives': int(fp[j]), 'false_negatives': int(fn[j]),
                                  'positive_decisions': int(d[:, j].sum())}
                          for j, label in enumerate(LABELS)}}
    return result


def coverage_counts(selected, observed):
    expected = {r['track_id']: r for r in selected}
    ids = [r['track_id'] for r in observed]
    if not selected or len(expected) != len(selected) or len(ids) != len(set(ids)):
        raise ValueError('Selected IDs must be nonempty; selected and observed IDs must be unique')
    if not set(ids).issubset(expected):
        raise ValueError('Observed tracks must come from frozen selection without replacements')
    for row in observed:
        source = expected[row['track_id']]
        if row['artist_id'] != source['artist_id'] or row['tags'] != source['tags']:
            raise ValueError('Observed artist or source tags changed after selection')
    selected_artists = {r['artist_id'] for r in selected}
    observed_artists = {r['artist_id'] for r in observed}
    return {'selected_tracks': len(selected), 'observed_tracks': len(observed),
            'selected_artists': len(selected_artists), 'observed_artists': len(observed_artists),
            'track_fraction': len(observed)/len(selected),
            'artist_fraction': len(observed_artists)/len(selected_artists),
            'missing_ids': [r['track_id'] for r in selected if r['track_id'] not in set(ids)]}


def paired_bootstrap(y, baseline_p, candidate_p, baseline_decisions, candidate_decisions,
                     artists, seed=BOOTSTRAP_SEED, repetitions=BOOTSTRAP_REPLICATES):
    """Resample artist clusters jointly for both arms and all labels, without fits."""
    y, baseline_p, artists = validate(y, baseline_p, artists)
    _, candidate_p, _ = validate(y, candidate_p, artists)
    decisions = [validate_decisions(y, d)['f1'] for d in [baseline_decisions, candidate_decisions]]
    unique, inverse, sizes = np.unique(artists, return_inverse=True, return_counts=True)
    if len(unique) < 2 or repetitions < 2:
        raise ValueError('Bootstrap requires at least two artists and replicates')
    sampled = np.random.default_rng(seed).integers(0, len(unique), (repetitions, len(unique)))
    denominators = sizes[sampled].sum(1)

    def cluster_sum(v):
        v = np.asarray(v)
        if v.ndim == 1:
            return np.bincount(inverse, weights=v, minlength=len(unique))[sampled].sum(1)
        return np.column_stack([cluster_sum(v[:, j]) for j in range(v.shape[1])])

    arm_draws, arm_points = [], []
    per_label_draws = {k: [] for k in ['brier', 'log_loss']}
    per_label_points = {k: [] for k in ['brier', 'log_loss']}
    for p, d in zip([baseline_p, candidate_p], decisions, strict=True):
        losses = loss_arrays(y, p)
        draws, points = {}, {}
        for key, value in losses.items():
            label_draws = cluster_sum(value)/denominators[:, None]
            draws['macro_'+key], points['macro_'+key] = label_draws.mean(1), float(value.mean())
            per_label_draws[key].append(label_draws)
            per_label_points[key].append(value.mean(0))
        tp, fp, fn = ((d & (y == 1)).sum(1), (d & (y == 0)).sum(1), (~d & (y == 1)).sum(1))
        dtp, dfp, dfn = cluster_sum(tp), cluster_sum(fp), cluster_sum(fn)
        draws['micro_f1'] = _divide(2*dtp, 2*dtp+dfp+dfn)
        points['micro_f1'] = float(_divide(2*tp.sum(), 2*tp.sum()+fp.sum()+fn.sum()))
        draws['micro_recall'] = _divide(dtp, dtp+dfn)
        points['micro_recall'] = float(_divide(tp.sum(), tp.sum()+fn.sum()))
        focus_fp = (d[:, FOCUS] & (y[:, FOCUS] == 0)).sum(1)
        draws['focus_fp_rate'] = cluster_sum(focus_fp)/denominators
        points['focus_fp_rate'] = float(focus_fp.mean())
        draws['focus_fp_count_equivalent'] = draws['focus_fp_rate']*len(y)
        points['focus_fp_count_equivalent'] = float(focus_fp.sum())
        arm_draws.append(draws); arm_points.append(points)
    result = {'replicates': int(repetitions), 'seed': int(seed), 'artists': len(unique),
              'direction': 'candidate minus baseline; negative favors candidate for losses and false positives',
              'scope': 'Paired artist-cluster 95% percentile intervals, conditional on this fixed candidate, '
                       'observed licensed frame and successful acquisition. No finite-population correction, '
                       'refitting or adjustment for annotation omissions, missing audio, within-artist song '
                       'selection, undocumented exposure or encoder pretraining. Only the prespecified '
                       'intersection of macro-Brier and focus-FP-rate benefits supports the go rule; '
                       'other intervals are descriptive, without simultaneous coverage guarantees.',
              'count_interval_definition': 'focus_fp_count_equivalent = observed track count times focus FP '
                                           'rate; rate counts both focus-label FP cases per track and can exceed 1.',
              'metrics': {}, 'per_label_losses': {}}
    saved = {'bootstrap_artists': unique, 'bootstrap_artist_indices': sampled,
             'bootstrap_track_counts': denominators}

    def interval(a, b, delta):
        finite = np.asarray(delta)[np.isfinite(delta)]
        return {'baseline': a, 'candidate': b, 'delta': None if a is None or b is None else b-a,
                'ci95': np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                'valid_replicates': len(finite)}

    for key in arm_draws[0]:
        delta = arm_draws[1][key]-arm_draws[0][key]
        saved['bootstrap_delta_'+key] = delta
        result['metrics'][key] = interval(arm_points[0][key], arm_points[1][key], delta)
    before, after = arm_draws[0]['focus_fp_rate'], arm_draws[1]['focus_fp_rate']
    relative = np.divide(after-before, before, out=np.full(len(before), np.nan), where=before != 0)
    point_before, point_after = arm_points[0]['focus_fp_rate'], arm_points[1]['focus_fp_rate']
    relative_point = (point_after-point_before)/point_before if point_before > 0 else None
    result['metrics']['focus_fp_relative_change'] = interval(0.0, relative_point, relative)
    result['metrics']['focus_fp_relative_change']['undefined_replicates'] = int((before == 0).sum())
    saved['bootstrap_delta_focus_fp_relative_change'] = relative
    for key in per_label_draws:
        result['per_label_losses'][key] = {}
        delta = per_label_draws[key][1]-per_label_draws[key][0]
        saved['bootstrap_delta_per_label_'+key] = delta
        for j, label in enumerate(LABELS):
            result['per_label_losses'][key][label] = interval(
                float(per_label_points[key][0][j]), float(per_label_points[key][1][j]), delta[:, j])
    return result, saved


def assess(metrics, intervals, coverage, y, artists, config=None, integrity=None):
    """Apply fixed evidence requirements, point safeguards, then joint benefit rule."""
    config = DEFAULT_CONFIG if config is None else config
    for key, value in DEFAULT_CONFIG.items():
        if config.get(key) != value:
            raise ValueError('Evaluation setting differs from frozen rule: '+key)
    if (intervals['seed'] != BOOTSTRAP_SEED or intervals['replicates'] != BOOTSTRAP_REPLICATES):
        raise ValueError('Bootstrap intervals differ from frozen seed or replicate count')
    if not integrity or integrity.get('electronic_rock_exact') is not True or integrity.get('precision_target_exact') is not True:
        raise ValueError('Integrity failure: copied labels or selective decisions changed')
    baseline, candidate = metrics['baseline'], metrics['candidate']
    y, _, artists = validate(y, np.zeros_like(y, float), artists)
    if (coverage['observed_tracks'] != len(y) or coverage['observed_artists'] != len(set(artists))
            or any(m['tracks'] != len(y) or m['artists'] != len(set(artists)) for m in [baseline, candidate])):
        raise ValueError('Coverage and evaluated row identities disagree')
    old, new = baseline['policies']['f1'], candidate['policies']['f1']
    old_fp, new_fp = old['focus_false_positives'], new['focus_false_positives']
    reduction = 1-new_fp/old_fp if old_fp > 0 else None
    gate = config['gate']
    point = {
        'focus_fp_reduction': reduction is not None and reduction+EPS >= gate['minimum_focus_fp_reduction'],
        'each_focus_fp_nonincrease': all(new['per_label'][LABELS[j]]['false_positives'] <= old['per_label'][LABELS[j]]['false_positives'] for j in FOCUS),
        'micro_recall_guard': new['micro_recall']+gate['maximum_micro_recall_drop']+EPS >= old['micro_recall'],
        'each_label_recall_guard': all(new['per_label'][label]['recall']+gate['maximum_label_recall_drop']+EPS >= old['per_label'][label]['recall'] for label in LABELS),
        'micro_f1_nondecrease': new['micro_f1']+EPS >= old['micro_f1'],
        'macro_brier_nonincrease': candidate['macro_brier'] <= baseline['macro_brier']+EPS,
        'macro_log_loss_nonincrease': candidate['macro_log_loss'] <= baseline['macro_log_loss']+EPS,
        'electronic_rock_exact': True, 'precision_target_exact': True,
    }
    evidence = {'track_coverage': coverage['track_fraction'] >= config['minimum_observed_fraction'],
                'artist_coverage': coverage['artist_fraction'] >= config['minimum_observed_fraction'],
                'observed_tracks': len(y) >= config['minimum_observed_tracks'],
                'observed_artists': len(set(artists)) >= config['minimum_observed_artists']}
    support = {}
    for j in FOCUS:
        label = LABELS[j]; support[label] = {}
        for kind, target in [('positive', 1), ('negative', 0)]:
            mask = y[:, j] == target
            for unit, value in [('tracks', int(mask.sum())), ('artists', len(set(artists[mask])))]:
                key = kind+'_'+unit
                support[label][key] = value
                evidence[label+'_'+key] = value >= config['minimum_focus_'+key]
    confidence = {key+'_ci_upper_below_zero': intervals['metrics'][key]['ci95'] is not None
                  and intervals['metrics'][key]['ci95'][1] < 0 for key in ['macro_brier', 'focus_fp_rate']}
    if not all(evidence.values()):
        verdict = 'incomplete_evidence'
    elif not all(point.values()):
        verdict = 'not_passed'
    elif all(confidence.values()):
        verdict = 'go'
    else:
        verdict = 'promising'
    return {'verdict': verdict, 'go': verdict == 'go', 'point_gate_passed': all(point.values()),
            'point_checks': point, 'focus_fp_relative_reduction': reduction,
            'evidence_sufficient': all(evidence.values()), 'evidence_checks': evidence,
            'focus_support': support, 'confidence_checks': confidence,
            'interpretation': 'A frozen operational confirmation rule for this sampled catalogue and proxy '
                              'labels. No tuning, automatic deployment, universal calibration guarantee '
                              'or claim of human-adjudicated genre correctness follows from this verdict.'}


def predict_and_check(x, baseline_bundle, correction, candidate_bundle):
    """Use application arithmetic, then check public interfaces and copied controls."""
    z, raw, probability, decisions, _, old_checks = baseline_predict(x, baseline_bundle, correction)
    prediction = {'baseline': {'logits': z, 'raw': raw, 'probability': probability, 'decisions': decisions}}
    values = {policy: candidate_predict.predict_arrays(candidate_bundle, x, policy) for policy in POLICIES}
    first = values[POLICIES[0]]
    for policy in POLICIES:
        if any(not np.array_equal(values[policy][key], first[key]) for key in ['logits', 'raw', 'probability']):
            raise ValueError('Candidate probabilities changed across threshold policies')
    prediction['candidate'] = {key: first[key] for key in ['logits', 'raw', 'probability']}
    prediction['candidate']['decisions'] = {policy: values[policy]['decision'] for policy in POLICIES}
    for key in ['logits', 'raw', 'probability']:
        if not np.array_equal(prediction['baseline'][key][:, [0, 3]], prediction['candidate'][key][:, [0, 3]]):
            raise ValueError('Integrity failure: electronic/rock probabilities or logits changed')
    for policy in POLICIES:
        if not np.array_equal(decisions[policy][:, [0, 3]], values[policy]['decision'][:, [0, 3]]):
            raise ValueError('Integrity failure: electronic/rock decisions changed')
    if not np.array_equal(decisions['precision_target'], values['precision_target']['decision']):
        raise ValueError('Integrity failure: selective policy decisions changed')
    checks = 0
    for i, row in enumerate(x):
        for policy in POLICIES:
            actual = candidate_predict.predict_vector(candidate_bundle, row, policy)
            old = predict_app.predict_vector(baseline_bundle, row, policy, 'weight_corrected', correction)
            for j, item in enumerate(actual):
                if (item['selected'] != bool(values[policy]['decision'][i, j])
                        or item['score'] != round(float(first['probability'][i, j]), 4)
                        or item['raw_score'] != round(float(first['raw'][i, j]), 4)
                        or item['threshold'] != round(float(candidate_bundle['thresholds'][policy][j]), 4)
                        or (j in (0, 3) and item != old[j])):
                    raise ValueError('Candidate public interface differs from full-precision results')
                checks += 1
    return prediction, {**old_checks, 'candidate_label_policy_interface_checks': checks,
                        'electronic_rock_exact': True, 'precision_target_exact': True}
