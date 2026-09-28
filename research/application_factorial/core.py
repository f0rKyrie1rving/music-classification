"""Fixed head/threshold interventions; no files, downloads or test-set selection."""
import hashlib

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score

from research.application_auto_improvement.core import (
    LABELS, fit_head, predict_head, binary_metrics, probability_metrics)

TOL = 1e-12
FOCUS = (1, 2)


def _binary(y):
    y = np.asarray(y)
    if y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected nonempty binary outcomes')
    return y.astype(np.int64, copy=False)


def _probability(p, shape):
    p = np.asarray(p, float)
    if p.shape != shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected aligned finite probabilities')
    return p


def _decision(d, shape):
    d = np.asarray(d)
    if d.shape != shape or not np.isin(d, [0, 1]).all():
        raise ValueError('Expected aligned binary decisions')
    return d.astype(bool, copy=False)


def fit_recipes(xfit, yfit, config):
    """Fit both prescribed recipes on exactly the supplied training rows."""
    x, y = np.asarray(xfit, float), np.asarray(yfit)
    if (x.ndim != 2 or not all(x.shape) or y.shape != (len(x), 4)
            or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all()):
        raise ValueError('Expected aligned finite features and four binary targets')
    original_C = float(config.get('baseline_C', .001))
    new_C = float(config.get('new_C', .0003))
    if original_C != .001 or new_C != .0003:
        raise ValueError('This factorial fixes baseline C=.001 and new C=.0003')
    settings = config['classifier']
    baseline = [fit_head(x, y[:, j], C=original_C, balanced=j == 2, **settings) for j in range(4)]
    new = baseline.copy()
    for j in FOCUS:
        new[j] = fit_head(x, y[:, j], C=new_C, balanced=False, **settings)
    return {'baseline': baseline, 'new': new}


def mapped_cutoff(rawthresholds, baseline_heads):
    """Equivalent baseline probability cutoff, using its own fit-only offsets."""
    raw = np.asarray(rawthresholds, float)
    if raw.shape != (4,) or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError('Expected four finite nonnegative raw thresholds')
    offsets = np.asarray([float(head['offset']) for head in baseline_heads])
    if offsets.shape != (4,) or not np.isfinite(offsets).all():
        raise ValueError('Invalid baseline offsets')
    result = raw.copy()
    changed = (offsets != 0) & (raw > 0) & (raw < 1)
    result[changed] = expit(np.log(raw[changed])-np.log1p(-raw[changed])+offsets[changed])
    return result


def predict_recipes(x, heads, rawthresholds):
    if set(heads) != {'baseline', 'new'} or any(len(v) != 4 for v in heads.values()):
        raise ValueError('Both fixed four-head recipes are required')
    cutoff = mapped_cutoff(rawthresholds, heads['baseline'])
    result = {}
    for recipe in ['baseline', 'new']:
        columns = [predict_head(x, head) for head in heads[recipe]]
        values = {name: np.column_stack([v[name] for v in columns]) for name in ['logits', 'raw', 'probability']}
        values['original_cutoff'] = cutoff.copy()
        # Preserve the original baseline raw comparison exactly, including ties.
        values['original_decision'] = (values['raw'] >= np.asarray(rawthresholds)
                                       if recipe == 'baseline' else values['probability'] >= cutoff)
        result[recipe] = values
    for key in ['logits', 'raw', 'probability', 'original_decision']:
        if not np.array_equal(result['baseline'][key][:, [0, 3]], result['new'][key][:, [0, 3]]):
            raise ValueError('Electronic/rock controls changed between recipes')
    return result


def _decision_metrics(y, decision):
    truth = y.astype(bool)
    tp = int(np.sum(truth & decision)); fp = int(np.sum(~truth & decision))
    fn = int(np.sum(truth & ~decision)); tn = int(np.sum(~truth & ~decision))
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp/(tp+fp) if tp+fp else 0., 'recall': tp/(tp+fn) if tp+fn else 0.,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def choose_policy(y, p, originaldecision, referenceolddecision, grid, recalltol=.03):
    """Threshold only: a failed search retains THIS head's original policy.

    The original-policy candidate may represent a different mapped probability
    cutoff on each inner fold. No head is screened, replaced or selected here.
    Both recipes use the same old-baseline decision metrics as their constraints.
    """
    y = _binary(y); p = _probability(p, y.shape)
    original = _decision(originaldecision, y.shape)
    reference = _decision(referenceolddecision, y.shape)
    thresholds = np.asarray(grid, float)
    if (thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all()
            or np.any((thresholds < 0) | (thresholds > 1)) or not np.isfinite(recalltol)
            or not 0 <= recalltol <= 1):
        raise ValueError('Invalid threshold grid or recall margin')
    ref = _decision_metrics(y, reference)
    candidates = [('original', None, original)]
    candidates += [('constant', float(t), p >= t) for t in sorted(set(thresholds.tolist()))]
    audit = []
    for kind, threshold, decision in candidates:
        values = _decision_metrics(y, decision)
        failures = []
        if values['fp'] > ref['fp']: failures.append('false_positives')
        if values['recall']+TOL < ref['recall']-recalltol: failures.append('recall')
        if values['f1']+TOL < ref['f1']: failures.append('f1')
        audit.append({'kind': kind, 'threshold': threshold, **values,
                      'feasible': not failures, 'failed_guards': failures})
    feasible = [r for r in audit if r['feasible']]
    if feasible:
        winner = min(feasible, key=lambda r: (r['fp'], -r['tp'], r['kind'] != 'original',
                                              -r['threshold'] if r['threshold'] is not None else 0.))
    else:
        winner = audit[0]
    return {'kind': winner['kind'], 'threshold': winner['threshold'],
            'feasible': bool(feasible), 'fallback': not bool(feasible),
            'fallback_reason': None if feasible else 'no_feasible_threshold; retain_same_head_original_policy',
            'reference_old_metrics': ref, 'selected_metrics': {k: winner[k] for k in ref},
            'recall_tolerance': float(recalltol), 'metric_tolerance': TOL,
            'candidate_audit': audit, 'head_changed_by_policy': False}


def apply_policy(p, originaldecision, record):
    probability = np.asarray(p, float)
    if probability.ndim != 1 or not len(probability):
        raise ValueError('Expected one label probability vector')
    probability = _probability(probability, probability.shape)
    original = _decision(originaldecision, probability.shape)
    if record['kind'] == 'original':
        if record['threshold'] is not None:
            raise ValueError('Original policy must retain the supplied row-specific decisions')
        return original.copy()
    if record['kind'] != 'constant' or record['threshold'] is None:
        raise ValueError('Unknown policy')
    threshold = float(record['threshold'])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid selected threshold')
    return probability >= threshold


def metrics(y, p, decision, artists=None):
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != 4 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected nonempty four-label targets')
    p = _probability(p, y.shape); decision = _decision(decision, y.shape)
    if artists is not None:
        artists = np.asarray(artists)
        if artists.shape != (len(y),): raise ValueError('Artist IDs are not aligned')
    per = {}
    for j, label in enumerate(LABELS):
        row = binary_metrics(y[:, j], p[:, j], decision[:, j])
        row['average_precision'] = float(average_precision_score(y[:, j], p[:, j])) if y[:, j].sum() else None
        row['mean_probability'] = float(p[:, j].mean())
        row['positive_fraction'] = float(y[:, j].mean())
        per[label] = row
    proper = probability_metrics(y, p)
    classification = _decision_metrics(y.reshape(-1), decision.reshape(-1))
    aps = [r['average_precision'] for r in per.values() if r['average_precision'] is not None]
    return {'tracks': len(y), 'artists': len(set(artists.tolist())) if artists is not None else None,
            'tp': classification['tp'], 'fp': classification['fp'], 'fn': classification['fn'],
            'micro_precision': classification['precision'], 'micro_recall': classification['recall'],
            'micro_f1': classification['f1'], 'macro_f1': float(np.mean([r['f1'] for r in per.values()])),
            'macro_brier': proper['brier'], 'macro_log_loss': proper['log_loss'],
            'macro_ap': float(np.mean(aps)) if aps else None, 'macro_ap_valid_labels': len(aps),
            'coverage': float(np.any(decision, axis=1).mean()), 'per_label': per}


def factorial_checks(predictions, decisions):
    """decisions[recipe]['original'/'tuned'] are N x 4 boolean arrays."""
    checks = {}
    for key in ['logits', 'raw', 'probability', 'original_decision']:
        checks['control_'+key] = bool(np.array_equal(predictions['baseline'][key][:, [0, 3]],
                                                    predictions['new'][key][:, [0, 3]]))
    for recipe in ['baseline', 'new']:
        original = predictions[recipe]['original_decision']
        supplied = _decision(decisions[recipe]['original'], original.shape)
        tuned = _decision(decisions[recipe]['tuned'], original.shape)
        checks[recipe+'_original_policy_preserved'] = bool(np.array_equal(supplied, original))
        checks[recipe+'_threshold_controls_unchanged'] = bool(np.array_equal(tuned[:, [0, 3]], original[:, [0, 3]]))
    if not all(checks.values()): raise ValueError('Factorial control failed: '+str(checks))
    return checks


def nested_subsets(artists, cohorts, fractions, seed):
    """Return float-fraction -> original-row indices; nest whole artists by source."""
    groups, sources = np.asarray(artists), np.asarray(cohorts)
    fractions = np.asarray(fractions, float)
    if (groups.ndim != 1 or sources.shape != groups.shape or not len(groups)
            or fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all()
            or np.any((fractions <= 0) | (fractions > 1)) or len(set(fractions)) != len(fractions)):
        raise ValueError('Invalid source-stratified subset request')
    assignment = {}
    for artist, source in zip(groups.tolist(), sources.tolist(), strict=True):
        if not isinstance(artist, str) or not artist or not isinstance(source, str) or not source:
            raise ValueError('Artist/source must be nonempty strings')
        if artist in assignment and assignment[artist] != source: raise ValueError('Artist appears in multiple source cohorts')
        assignment[artist] = source
    order = {source: sorted([a for a,s in assignment.items() if s == source],
                            key=lambda a: (hashlib.sha256(f'{seed}|{source}|artist|{a}'.encode()).hexdigest(), a))
             for source in sorted(set(sources.tolist()))}
    result = {}
    for fraction in sorted(fractions.tolist()):
        chosen = {a for sequence in order.values() for a in sequence[:int(np.ceil(fraction*len(sequence)))]}
        result[fraction] = np.flatnonzero(np.isin(groups, list(chosen)))
    return result
